#!/usr/bin/env python3
"""regression._parse: a log holding an assertion firing fails as ASSERT, whatever the testbench's counts say; a clean log passes. truth_gate: model words pass, planted errors fail."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # regression.py, at the repository root
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
            assert (fmt == "bf16") == ("narrow" in res) != ("transfer" in res) and all(r.ok for r in res.values()), (fmt, gens, truth_text(res))
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
    res = reg.truth_gate(_dump(stim, N, rows), stim, N, "int8")
    assert res["matmul"].ok and res["matmul"].worst == 0.0, truth_text(res)
    # set 1's bias sits within 4096 of INT32_MAX: its sums past int32 wrap, as the int32 accumulator's contract says, and pass
    name, pre, post = next(r for r in rows if r[0] == "matrixC_1.mem")
    passes, bias = reg._operands(stim, name, N)
    t, _ = reg.truth.matmul_ref([tuple(reg.truth.values("int8", m) for m in passes[0])], reg.truth.values("int32", bias))
    past = (t > 2**31 - 1).ravel()
    assert past.sum() > 0 and res["matmul"].n_ovf == int(past.sum()), (int(past.sum()), truth_text(res))
    # the same sums saturated at INT32_MAX instead of wrapped fail, element for element
    sat = np.where(past, 2**31 - 1, reg.truth.values("int32", pre)).astype(np.int64) & 0xFFFFFFFF
    res = reg.truth_gate(_dump(stim, N, [(name, sat, sat)]), stim, N, "int8")
    assert not res["matmul"].ok and res["matmul"].over == int(past.sum()), truth_text(res)


def test_truth_transfer_fails_a_changed_word():
    from matmul_tests import gen_mm_random
    for fmt in ("fp32", "int8"):
        stim, N = _gen(fmt, (gen_mm_random,)), 8
        rows = _model_rows(stim, fmt)
        name, pre, post = rows[0]
        post = post.copy()
        post[[3, 5]] = post[[5, 3]]  # two delivered words swapped: the sums are right, the words are not
        res = reg.truth_gate(_dump(stim, N, [(name, pre, post)] + rows[1:]), stim, N, fmt)
        assert res["matmul"].ok and not res["transfer"].ok and res["transfer"].over == (2 if pre[3] != pre[5] else 0), truth_text(res)


def test_truth_int8_drop_kills_every_affected_element():
    from matmul_tests import gen_mm_random
    stim, N = _gen("int8", (gen_mm_random,)), 8
    rows = _model_rows(stim, "int8")
    for name, pre, post in rows:
        passes, _ = reg._operands(stim, name, N)
        A, B = (reg.truth.values("int8", m) for m in passes[0])
        term = np.outer(A[:, 0], B[0, :]).ravel().astype(np.int64)  # the first product of every PE
        bad = (reg.truth.values("int32", pre).astype(np.int64) - term) & 0xFFFFFFFF
        res = reg.truth_gate(_dump(stim, N, [(name, bad, bad)]), stim, N, "int8")
        assert res["matmul"].over == int((term != 0).sum()) > 0, (name, int((term != 0).sum()), truth_text(res))


def test_truth_verdict():
    log = ("  Simulations : 2\n  Passed      : 2\n  TRUTH: 2 simulations gated, 0 with misses\nAll tests passed — Sanity Clean\n")
    assert reg.SPEC.verdict("fp32/N8_T2/mesh", log, 0) == "PASS"
    assert reg.SPEC.verdict("fp32/N8_T2/mesh", log.replace("TRUTH: 2", "TRUTH: 1"), 0).startswith("FAIL (1 of 2 simulations gated")
    assert reg.SPEC.verdict("fp32/N8_T2/mesh", log.replace("  TRUTH: 2 simulations gated, 0 with misses\n", ""), 0).startswith("FAIL (0 of 2")
    bad = log + "[FAIL] truth: matmul 3.1% of sum|ab|\n"
    assert reg.SPEC.verdict("fp32/N8_T2/mesh", bad, 0).startswith("FAIL (true function: matmul"), reg.SPEC.verdict("fp32/N8_T2/mesh", bad, 0)


def truth_text(res):
    return "; ".join(reg.truth.stage_line(r) for r in res.values())


def test_assert_stop_is_not_a_build_failure():
    """A simulation stopped by a firing assertion reports ASSERT and is not taken for a build failure that skips the configuration's other tests."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "regression.py")).read()
    assert "'SIMFAIL' if SIM_STARTED in out else 'BUILDFAIL'" in src and reg.SIM_STARTED == "SYSTOLIC MESH VERIFICATION"
    tb = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "testbenches", "top", "TB_SystolicMesh.sv")).read()
    assert reg.SIM_STARTED in tb, "the banner regression.py waits for is the bench's"
    raw = " SYSTOLIC MESH VERIFICATION\n%Error: x.sv:1: Assertion failed in TOP.a: boom\nSIMFAIL: make exited 2\n"
    assert reg._parse(raw)["status"] == "ASSERT" and "BUILDFAIL" not in raw


def test_x_counts_and_words_fail_by_name():
    """A 4-state simulator's x in a count, a cycle count or a dumped sum fails the test by name; a VCS assertion is ASSERT, as Verilator's."""
    r = reg._parse(CLEAN.replace("Total Elements: 5888", "Total Elements: x"), "mm_relu N8 T2")
    assert r["status"] == "FAIL" and r["xline"] == "FAIL: X in element count of test mm_relu N8 T2 ('x')", r
    r = reg._parse(CLEAN.replace("Set 1 : 61 cycles", "Set 1 : X cycles"), "mm_relu N8 T2")
    assert r["status"] == "FAIL" and r["xline"].startswith("FAIL: X in cycle count 1 of test mm_relu N8 T2"), r
    assert reg._parse(CLEAN)["status"] == "PASS" and "xline" not in reg._parse(CLEAN)
    vcs = 'Error: "/t/SystolicMesh.sv", 545: TB_SystolicMesh.dut.a_pack_one_pass: at time 2975 ns\n'
    assert reg._parse(CLEAN + vcs)["status"] == "ASSERT"
    from matmul_tests import gen_mm_bias_special
    stim, N = _gen("fp32", (gen_mm_bias_special,)), 8
    rows = _model_rows(stim, "fp32")
    name, pre, post = rows[0]
    path = _dump(stim, N, rows)
    t = open(path).read().splitlines()
    k = next(i for i, x in enumerate(t) if x.startswith("R ")) + 1 + 37
    t[k] = t[k].split()[0] + " xxxxxxxx"
    open(path, "w").write("\n".join(t) + "\n")
    try:
        reg.truth_gate(path, stim, N, "fp32")
        raise AssertionError("an x word passed the truth dump's parser")
    except reg.fourstate.Unknown as e:
        assert str(e).startswith(f"X in output word 37 of test {name}"), e


def test_build_failure_names_its_error():
    """A failed build's error lines reach the console with the log's path: Verilator's V3TSP internal error, a linker error, or the tail when no line matches."""
    tsp = ("verilator --binary ...\nmake[1]: Leaving directory '/t/SystolicMesh'\n%Error: Internal Error: ../V3TSP.cpp:353: No unmarked edges found in tour\n"
           "                        ... See the manual at https://verilator.org/verilator_doc.html for more assistance.\nmake[1]: *** [Makefile:125: verilator] Error 1\n\nBUILDFAIL: make exited 2\n")
    e = reg.build_errors(tsp)
    assert any("V3TSP.cpp:353" in x for x in e) and any("Error 1" in x for x in e) and not any("See the manual" in x for x in e), e
    ld = "g++ ...\n" * 5000 + "x.o: relocation truncated to fit: R_X86_64_PC32 against `.bss'\ncollect2: error: ld returned 1 exit status\nBUILDFAIL: make exited 2\n"
    assert any("relocation truncated" in x for x in reg.build_errors(ld)) and any("collect2: error" in x for x in reg.build_errors(ld))
    assert reg.build_errors("a\nb\nc\nBUILDFAIL: make exited 2\n")[-1] == "BUILDFAIL: make exited 2"  # no error line: the tail
    assert len(reg.build_errors("%Error: x\n" * 100)) <= reg.BUILD_ERROR_LINES
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "regression.py")).read()
    assert "build_errors(raw)" in src and "build log:" in src, "the group runner prints the failed build's error lines and its log"


MIN_TESTS = 14  # tests this file holds; fewer run means one was lost, which is a failure

if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    if len(tests) < MIN_TESTS:
        print(f"FAIL: {len(tests)} tests ran, fewer than MIN_TESTS {MIN_TESTS}")
        sys.exit(1)
    print(f"ALL {len(tests)} PASSED")
