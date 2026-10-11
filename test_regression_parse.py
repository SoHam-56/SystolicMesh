#!/usr/bin/env python3
"""regression._parse: a log holding an assertion firing fails as ASSERT, whatever the testbench's counts say; a clean log passes. truth_gate: model words pass, planted errors fail."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import regression as reg  # noqa: E402

CLEAN = """ Passed Sets:    23
 Failed Sets:    0
 Total Elements: 5888
  Set 0 : 59 cycles  (590 ns)
  Set 1 : 61 cycles  (610 ns)
SUCCESS
"""
FIRED = ("[2975000] %Error: SystolicMesh.sv:545: Assertion failed in TB_SystolicMesh.dut.a_pack_one_pass: "
         "SystolicMesh: a packed set is part of an accumulated sum\n")


def test_clean_log_passes():
    r = reg._parse(CLEAN)
    assert r["status"] == "PASS" and r["passed"] == 23 and r["avg_cyc"] == 60, r


def test_assertion_firing_fails():
    r = reg._parse(CLEAN + FIRED)  # the TB's own counts still read clean
    assert r["status"] == "ASSERT", r


def test_assertion_without_error_prefix_fails():
    r = reg._parse(CLEAN + "Assertion failed in TB_SystolicMesh.dut.a_read_outstanding\n")
    assert r["status"] == "ASSERT", r


def test_error_line_is_not_a_pass():
    r = reg._parse(CLEAN + "%Error: TB_SystolicMesh.sv:12: some runtime error\n")
    assert r["status"] == "NO-RUN", r


def _dump(stim, N, rows):
    """A truth dump like TB_SystolicMesh's: per result its expected file, then each element's sum and word."""
    path = os.path.join(stim, "truth_dump.txt")
    with open(path, "w") as fh:
        for i, (name, pre, post) in enumerate(rows):
            fh.write(f"R {name} {i}\n" + "".join(f"{int(a):08x} {int(b):08x}\n" for a, b in zip(pre, post)))
    return path


def _model_rows(stim, fmt):
    """Every result the test's files hold, words from mesh_model (bf16: the sum is the word widened, which narrows back exactly)."""
    rows = []
    for name in sorted(n for n in os.listdir(stim) if n.startswith(("matrixC", "accC"))):
        post = reg._words(os.path.join(stim, name))
        rows.append((name, (post << 16) if fmt == "bf16" else post, post))
    return rows


def _gen(fmt, gens, N=8):
    import atexit
    import shutil
    import tempfile
    stim = tempfile.mkdtemp(prefix="truth_mesh_")
    atexit.register(shutil.rmtree, stim, True)
    reg.stim_format.configure(fmt, 2, 1)
    for g in gens:
        g(stim, N)
    return stim


def test_truth_passes_the_model_and_fails_a_dropped_product():
    from matmul_tests import gen_mm_accum, gen_mm_bias_special, gen_mm_packed_garbage
    for fmt in ("fp32", "bf16"):
        for gens in ((gen_mm_bias_special,), (gen_mm_accum,), (gen_mm_packed_garbage,)):
            stim, N = _gen(fmt, gens), 8
            rows = _model_rows(stim, fmt)
            res = reg.truth_gate(_dump(stim, N, rows), stim, N, fmt)
            assert res["matmul"].ok and res["matmul"].n == N * N * len(rows), (fmt, gens, truth_text(res))
            assert (fmt == "bf16") == ("narrow" in res) and all(r.ok for r in res.values()), (fmt, gens, truth_text(res))
            # set 0 without its first product, A[:, 0] B[0, :]
            name, pre, post = rows[0]
            passes, _ = reg._operands(stim, name, N)
            A, B = (reg.truth.values(fmt, m) for m in passes[0])
            drop = reg.truth.values("fp32", pre).reshape(N, N) - np.outer(A[:, 0], B[0, :])
            bad = reg.truth.encode("fp32", drop).ravel()
            res = reg.truth_gate(_dump(stim, N, [(name, bad, reg.truth.narrow_ref(fmt, bad) if fmt == "bf16" else bad)] + rows[1:]), stim, N, fmt)
            assert not res["matmul"].ok and res["matmul"].over > 0, (fmt, gens, truth_text(res))
            lines, misses = reg.truth_lines(res)
            assert any(m.startswith("[FAIL] truth: matmul ") for m in misses), misses


def test_truth_packed_sums_each_block_alone():
    from matmul_tests import gen_mm_packed_garbage
    stim, N = _gen("fp32", (gen_mm_packed_garbage,)), 8
    rows = _model_rows(stim, "fp32")
    name, pre, post = rows[0]  # set 0 is packed: off-block weights are random and must not count
    assert os.path.exists(os.path.join(stim, "packShift_0.mem"))
    A = reg.truth.values("fp32", reg._words(os.path.join(stim, "matrixA_0.mem"))).reshape(N, N)
    B = reg.truth.values("fp32", reg._words(os.path.join(stim, "matrixB_0.mem"))).reshape(N, N)
    full = reg.truth.encode("fp32", A @ B).ravel()
    res = reg.truth_gate(_dump(stim, N, [(name, full, full)]), stim, N, "fp32")
    assert not res["matmul"].ok, truth_text(res)


def test_truth_narrow_fails_truncation():
    from matmul_tests import gen_mm_random
    stim, N = _gen("bf16", (gen_mm_random,)), 8
    rows = _model_rows(stim, "bf16")
    name, pre, post = rows[0]
    above = pre | 0xC000  # above half an ULP: round to nearest goes up, the word kept is truncated
    res = reg.truth_gate(_dump(stim, N, [(name, above, post)] + rows[1:]), stim, N, "bf16")
    assert res["matmul"].ok and not res["narrow"].ok and res["narrow"].over == N * N, truth_text(res)


def test_truth_int8_exact_and_wrap():
    from matmul_tests import gen_mm_random
    stim, N = _gen("int8", (gen_mm_random,)), 8
    rows = _model_rows(stim, "int8")
    res = reg.truth_gate(_dump(stim, N, [r for r in rows if r[0] != "matrixC_1.mem"]), stim, N, "int8")
    assert res["matmul"].ok and res["matmul"].worst == 0.0, truth_text(res)
    # set 1's bias sits within 4096 of INT32_MAX: the mesh's int32 sums wrap, the true sums do not
    res = reg.truth_gate(_dump(stim, N, [r for r in rows if r[0] == "matrixC_1.mem"]), stim, N, "int8")
    passes, bias = reg._operands(stim, "matrixC_1.mem", N)
    t, _ = reg.truth.matmul_ref([tuple(reg.truth.values("int8", m) for m in passes[0])], reg.truth.values("int32", bias))
    assert not res["matmul"].ok and res["matmul"].over == int((t > 2**31 - 1).sum()) > 0, truth_text(res)


def test_truth_verdict():
    log = ("  Simulations : 2\n  Passed      : 2\n  TRUTH: 2 simulations gated, 0 with misses\nAll tests passed — Sanity Clean\n")
    assert reg.SPEC.verdict("fp32/N8_T2/mesh", log, 0) == "PASS"
    assert reg.SPEC.verdict("fp32/N8_T2/mesh", log.replace("TRUTH: 2", "TRUTH: 1"), 0).startswith("FAIL (1 of 2 simulations gated")
    assert reg.SPEC.verdict("fp32/N8_T2/mesh", log.replace("  TRUTH: 2 simulations gated, 0 with misses\n", ""), 0).startswith("FAIL (0 of 2")
    bad = log + "[FAIL] truth: matmul 3.1% of sum|ab|\n"
    assert reg.SPEC.verdict("fp32/N8_T2/mesh", bad, 0).startswith("FAIL (true function: matmul"), reg.SPEC.verdict("fp32/N8_T2/mesh", bad, 0)


def truth_text(res):
    return "; ".join(reg.truth.stage_line(r) for r in res.values())


MIN_TESTS = 9  # tests this file holds; fewer run means one was lost, which is a failure

if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    if len(tests) < MIN_TESTS:
        print(f"FAIL: {len(tests)} tests ran, fewer than MIN_TESTS {MIN_TESTS}")
        sys.exit(1)
    print(f"ALL {len(tests)} PASSED")
