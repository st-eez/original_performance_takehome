"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Pre-load all hash stage constants before loop entry so that no
        # load("const") instructions appear inside the loop body.
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            self.scratch_const(val1)
            self.scratch_const(val3)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        # Scalar scratch registers
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")

        # Hardware round loop: allocate counter and rounds constant
        round_counter = self.alloc_scratch("round_counter")
        rounds_addr = self.alloc_scratch("rounds_const")
        self.add("load", ("const", rounds_addr, rounds))

        # Hardware inner batch loop: allocate counter and batch_size constant
        batch_counter = self.alloc_scratch("batch_counter")
        batch_size_addr = self.alloc_scratch("batch_size_const")
        self.add("load", ("const", batch_size_addr, batch_size))

        # Initialize round counter before outer loop
        self.add("load", ("const", round_counter, 0))

        # Record the PC of the outer loop start
        loop_start = len(self.instrs)

        # Initialize batch counter at the start of each outer iteration
        self.add("load", ("const", batch_counter, 0))

        # Record the PC of the inner loop start
        inner_loop_start = len(self.instrs)

        # Inner loop body (emitted once, iterated via hardware loop)
        # idx = mem[inp_indices_p + batch_counter]
        self.add("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], batch_counter))
        self.add("load", ("load", tmp_idx, tmp_addr))
        # val = mem[inp_values_p + batch_counter]
        self.add("alu", ("+", tmp_addr, self.scratch["inp_values_p"], batch_counter))
        self.add("load", ("load", tmp_val, tmp_addr))
        # node_val = mem[forest_values_p + idx]
        self.add("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx))
        self.add("load", ("load", tmp_node_val, tmp_addr))
        # val = myhash(val ^ node_val)
        self.add("alu", ("^", tmp_val, tmp_val, tmp_node_val))
        hash_slots = self.build_hash(tmp_val, tmp1, tmp2)
        for engine, slot in hash_slots:
            self.add(engine, slot)
        # idx = 2*idx + (1 if val % 2 == 0 else 2)
        self.add("alu", ("%", tmp1, tmp_val, two_const))
        self.add("alu", ("==", tmp1, tmp1, zero_const))
        self.add("flow", ("select", tmp3, tmp1, one_const, two_const))
        self.add("alu", ("*", tmp_idx, tmp_idx, two_const))
        self.add("alu", ("+", tmp_idx, tmp_idx, tmp3))
        # idx = 0 if idx >= n_nodes else idx
        self.add("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"]))
        self.add("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const))
        # mem[inp_indices_p + batch_counter] = idx
        self.add("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], batch_counter))
        self.add("store", ("store", tmp_addr, tmp_idx))
        # mem[inp_values_p + batch_counter] = val
        self.add("alu", ("+", tmp_addr, self.scratch["inp_values_p"], batch_counter))
        self.add("store", ("store", tmp_addr, tmp_val))

        # Inner loop control: increment batch_counter, compare, jump back
        self.add("flow", ("add_imm", batch_counter, batch_counter, 1))
        self.add("alu", ("<", tmp1, batch_counter, batch_size_addr))
        self.add("flow", ("cond_jump", tmp1, inner_loop_start))

        # Outer loop control: increment round counter, compare, jump back
        self.add("flow", ("add_imm", round_counter, round_counter, 1))
        self.add("alu", ("<", tmp1, round_counter, rounds_addr))
        self.add("flow", ("cond_jump", tmp1, loop_start))

        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
