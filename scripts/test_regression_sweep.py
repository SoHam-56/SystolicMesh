#!/usr/bin/env python3
"""make regression's list, commands and verdicts: the model checks once, the mesh at every N, tile size and format (collapse-k 1)."""
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # regression.py, at the repository root
import regression as reg  # noqa: E402

ROOT = reg.ROOT
CHECKS = ["model-tests", "aril-fpu", "aril-narrow"]


def test_vcs_commands():
    """SIM=vcs: each mesh point asks regression.py for VCS (make vcs), aril-narrow runs fp32ToBf16's bench on VCS, and make -n names vcs, never verilator."""
    import copy
    spec = copy.copy(reg.SPEC)
    spec.sim = "vcs"
    a = spec.argv("bf16/N8_T2/mesh-held", None)
    assert a[-1] == "SIM=vcs" and a[:3] == ["make", "mesh-regression", "MATRIX_SIZE=8"] and reg.SPEC.argv("bf16/N8_T2/mesh-held", None) == a[:-1], a
    assert spec.argv("aril-narrow", None) == ["make", "-C", "ArithmeticLibrary", "fp32ToBf16-check", "SIM=vcs"]
    env = {k: x for k, x in os.environ.items() if not k.startswith("MAKE") and k != "SIM"}
    run = lambda *x: subprocess.run(["make", "-s", "-n", *x], cwd=ROOT, capture_output=True, text=True, env=env)
    r = run(*a[1:])
    assert r.returncode == 0 and "--sim vcs" in r.stdout, r.stdout[-300:]
    r = run("vcs", "FMT=int8", "MATRIX_SIZE=8", "TILE_SIZE=2", "COLLAPSE_K=0", "HELD_ACCUM=1", "EXTRA_FLAGS=-DASSERT_SELFTEST")
    for want in ("-top TB_SystolicMesh", "+define+FMT_INT8", "-pvalue+TB_SystolicMesh.MATRIX_SIZE=8", "-pvalue+TB_SystolicMesh.TILE_SIZE=2",
                 "-pvalue+TB_SystolicMesh.COLLAPSE_K=0", "-pvalue+TB_SystolicMesh.HELD_ACCUM=1", "+define+ASSERT_SELFTEST", "./sim  +vcs+dumpvarsoff"):
        assert want in r.stdout, (want, r.stdout[-500:])
    assert "verilator" not in r.stdout.lower().replace("make verilator", "") and "-debug" not in r.stdout, r.stdout[-300:]
    r = run(*spec.argv("aril-narrow", None)[1:])
    assert r.returncode == 0 and "vcs -full64" in r.stdout and "verilator " not in r.stdout, r.stdout[-300:]
    r = run("regression", "SIM=vcs", "SMALL=1", f"PYTHON={sys.executable}")
    pts = [x for x in r.stdout.splitlines() if "--action point " in x]
    assert pts and all(" --sim vcs" in x for x in pts) and " --sim vcs" in r.stdout.splitlines()[-1], r.stdout[-300:]
    v = "RESULT: PASSED\nfp32ToBf16: 52 corners against their table, 7 values, 0 errors\n"
    assert spec.verdict("aril-narrow", v, 0) == "PASS" and reg.SPEC.verdict("aril-narrow", v, 0).startswith("FAIL (no 'lint fp32ToBf16")
    assert spec.verdict("aril-narrow", v + "FAIL: X in output word 3 of test narrow bf16\n", 0).startswith("FAIL (X in output word 3")


def test_points():
    small, full = reg.SPEC.points(True, list(reg.FORMATS)), reg.SPEC.points(False, list(reg.FORMATS))
    assert small[:3] == full[:3] == CHECKS, small[:4]  # format-independent: once
    held = lambda pts: [p for p in pts if p.endswith("/mesh-held")]
    assert held(small) == held(full) == [f"{f}/N{n}_T{t}/mesh-held" for f in reg.FORMATS for n in (8, 16) for t in (2, 4)], held(small)
    assert all(reg.SPEC.parse(p)["ck"] == 0 and reg.SPEC.parse(p)["held"] == 1 for p in held(small))
    small, full = [p for p in small if p not in held(small)], [p for p in full if p not in held(full)]
    sweep = lambda pts: sorted({(reg.SPEC.parse(p)["n"], reg.SPEC.parse(p)["tile"]) for p in pts[3:]})
    assert sweep(small) == [(8, 2), (8, 4), (8, 8), (16, 2), (16, 4), (16, 8), (16, 16)], sweep(small)
    assert sweep(full) == sweep(small) + [(32, t) for t in (2, 4, 8, 16, 32)] + [(64, t) for t in (2, 4, 8, 16, 32, 64)], sweep(full)
    assert len(small) == len(set(small)) == 3 + 7 * 3 * 2 and len(full) == len(set(full)) == 3 + 18 * 3 * 2, (len(small), len(full))
    for f in reg.FORMATS:  # collapse-k 1 and 0 at every N and tile size
        assert [p.rsplit("/", 1)[1] for p in full if p.startswith(f"{f}/N64_T2/")] == ["mesh", "mesh-ck0"], f
    assert [reg.SPEC.parse(p)["ck"] for p in ("fp32/N8_T2/mesh", "fp32/N8_T2/mesh-ck0")] == [1, 0]
    assert reg.SPEC.points(True, ["int8"]) == CHECKS + [p for p in small + held(reg.SPEC.points(True, ["int8"])) if p.startswith("int8/")]
    assert reg.SPEC.points(False, []) == []
    assert not any(reg.SPEC.not_applicable(p) for p in full)  # every point runs under VCS too
    for bad in ("fp32/N8_T8/mesh-held", "fp32/N32_T2/mesh-held", "fp32/N8_T16/mesh", "fp16/N8_T2/mesh", "fp32/N12_T4/mesh", "fp32/model-tests", "fp32/N8_T2/pipeline", "fp32/N8_T2/mesh-ck2", "nosuch"):
        try:
            reg.SPEC.parse(bad)
        except ValueError:
            continue
        raise AssertionError(f"parse accepted {bad}")


def test_commands():
    py = sys.executable
    assert reg.SPEC.argv("bf16/N32_T8/mesh", None) == ["make", "mesh-regression", "MATRIX_SIZE=32",
                                                       "REGRESSION_OPTS=--format bf16 --collapse-k 1 --tiles 8 --no-checks", f"PYTHON={py}"]
    assert reg.SPEC.argv("int8/N64_T4/mesh-ck0", None) == ["make", "mesh-regression", "MATRIX_SIZE=64",
                                                           "REGRESSION_OPTS=--format int8 --collapse-k 0 --tiles 4 --no-checks", f"PYTHON={py}"]
    assert reg.SPEC.argv("bf16/N8_T2/mesh-held", None) == ["make", "mesh-regression", "MATRIX_SIZE=8",
                                                           "REGRESSION_OPTS=--format bf16 --collapse-k 0 --tiles 2 --no-checks --held-accum --group matmul",
                                                           f"PYTHON={py}"]
    assert reg.SPEC.argv("model-tests", None) == ["make", "model-tests", f"PYTHON={py}"] and reg.SPEC.argv("aril-narrow", None)[:2] == ["make", "aril-narrow"]
    assert [reg.make_args(n, ck) for n, ck in ((64, 0), (64, 1), (32, 0))] == [["LARGE_CODE=1"], [], []]  # a collapse-k 0 N = 64 binary passes 2 GB
    env = {k: x for k, x in os.environ.items() if not k.startswith("MAKE")}
    big = subprocess.run(["make", "-s", "-n", "verilator", "LARGE_CODE=1"], cwd=ROOT, capture_output=True, text=True, env=env).stdout
    assert "-CFLAGS -mcmodel=large -CFLAGS -fno-pie -LDFLAGS -no-pie -LDFLAGS -Wl,--no-relax" in big, big[-300:]  # ld cannot relax GOTPCREL past 2 GB
    assert "mcmodel" not in subprocess.run(["make", "-s", "-n", "verilator"], cwd=ROOT, capture_output=True, text=True, env=env).stdout
    pe = lambda n: reg.regress.point_env(reg.SPEC, n, "s", 8)
    assert "OPT_FAST" not in pe("fp32/N8_T2/mesh") and "OPT_FAST" not in pe("int8/N16_T4/mesh-ck0") and pe("bf16/N32_T8/mesh")["OPT_FAST"] == "-O0"  # -O0 only from N = 32: below it the long mesh simulations lose more than the build gains (rs1_spd_mesh)
    make_n = lambda *a: subprocess.run(["make", "-s", "-n", *a], cwd=ROOT, capture_output=True, text=True, env=env).stdout
    cores = reg.regress._cores()
    assert f"-j {cores} --build-jobs {cores} " in make_n("verilator"), "Verilator takes every core"
    assert "--trace" not in make_n("verilator") and "--trace -DENABLE_TRACE" in make_n("verilator", "TRACE=1")  # no waveform by default
    assert "verilator TRACE=1" in make_n("wave"), "make wave builds with TRACE=1"
    tb = open(os.path.join(ROOT, "testbenches", "top", "TB_SystolicMesh.sv")).read()
    assert re.search(r"`ifdef ENABLE_TRACE\s+\$dumpfile\([^)]*\);\s+\$dumpvars\([^)]*\);\s+`endif", tb), "the TB dumps only in a TRACE=1 build"


def test_build_config():
    """A build's format is make's FMT (the define) and its sizes are -G overrides of TB_SystolicMesh's parameters; no script edits the bench."""
    assert reg.build_args("bf16", 16, 4, 5, 1) == ["FMT=bf16", "MATRIX_SIZE=16", "TILE_SIZE=4", "NUM_TEST_SETS=5", "COLLAPSE_K=1"]
    assert reg.build_args("int8", 64, 8, 3, 0) == ["FMT=int8", "MATRIX_SIZE=64", "TILE_SIZE=8", "NUM_TEST_SETS=3", "COLLAPSE_K=0", "LARGE_CODE=1"]
    assert reg.build_args("fp32", 8, 2, 5, 0, 1)[-1] == "HELD_ACCUM=1"
    env = {k: x for k, x in os.environ.items() if not k.startswith("MAKE") and k not in ("FMT", "MATRIX_SIZE", "TILE_SIZE", "NUM_TEST_SETS", "COLLAPSE_K")}
    make_n = lambda *a: subprocess.run(["make", "-s", "-n", *a], cwd=ROOT, capture_output=True, text=True, env=env)
    for f in reg.FORMATS:
        out = make_n("verilator", *reg.build_args(f, 8, 2, 7, 0)).stdout
        assert f"+define+FMT_{f.upper()} " in out and "-GMATRIX_SIZE=8 -GTILE_SIZE=2 -GNUM_TEST_SETS=7 -GCOLLAPSE_K=0" in out, (f, out[-300:])
        assert all(f"+define+FMT_{g.upper()}" not in out for g in reg.FORMATS if g != f), f  # one define per build
    dflt = make_n("verilator").stdout
    assert "HELD_ACCUM" not in dflt  # off by default: every other point's build is unchanged
    assert "-GHELD_ACCUM=1" in make_n("verilator", *reg.build_args("fp32", 8, 2, 5, 0, 1)).stdout
    assert "+define+FMT_FP32 " in dflt and "-GMATRIX_SIZE=16 -GTILE_SIZE=4 -GNUM_TEST_SETS=5 -GCOLLAPSE_K=1" in dflt, dflt[-300:]  # the bench's defaults
    assert "-GMATRIX_SIZE" not in make_n("verilator", "TOP_MODULE=TB_PE").stdout  # the sizes are TB_SystolicMesh's only
    assert make_n("verilator", "FMT=fp16").returncode != 0 and make_n("lint", "FMT=bf16 int8").returncode != 0
    tb = open(os.path.join(ROOT, "testbenches", "top", "TB_SystolicMesh.sv")).read()
    for k in ("MATRIX_SIZE", "TILE_SIZE", "NUM_TEST_SETS", "COLLAPSE_K"):
        assert re.search(rf"^\s*parameter int {k}\s*=", tb, re.M) and not re.search(rf"localparam\s+(int\s+)?{k}\s*=", tb), k
    src = open(os.path.join(ROOT, "regression.py")).read()
    assert not re.search(r"TB_SystolicMesh\.sv|localparam|re\.subn?\(", src), "regression.py edits no source file"


def test_verdicts():
    mesh = ("  Simulations : 12\n  Passed      : \x1b[92m12\x1b[0m\n  TRUTH: 12 simulations gated, 0 with misses\n"
            "  Verdict : ✅  All tests passed — Sanity Clean\n")
    v = lambda p, t, rc=0: reg.SPEC.verdict(p, t, rc)
    assert v("fp32/N8_T2/mesh", mesh) == "PASS" and v("fp32/N8_T2/mesh", mesh, 1) == "FAIL (exit 1)"
    assert v("fp32/N8_T2/mesh", mesh.replace("  TRUTH: 12 simulations gated, 0 with misses\n", "")) == "FAIL (0 of 12 simulations gated on the true function)"
    assert v("fp32/N8_T2/mesh", mesh.replace("\x1b[92m12", "\x1b[92m11")) == "FAIL (11 of 12 simulations passed)"
    assert v("fp32/N8_T2/mesh", mesh.replace("12", "0")) == "FAIL (0 of 0 simulations passed)"
    assert v("fp32/N8_T2/mesh", "Simulations : 3\n") == "FAIL (no 'Simulations : X', 'Passed : X' with Sanity Clean)"
    assert v("fp32/N8_T2/mesh", mesh + "[x] Assertion failed in TB\n") == "FAIL (log: [x] Assertion failed in TB)"
    tests = "ALL PASS\nALL 13 PASSED\nALL 4 PASSED\nALL 3 PASSED\n"
    assert v("model-tests", tests) == "PASS" and v("model-tests", tests.replace("ALL 4 PASSED\n", "")) == "FAIL (3 of 4 model test files passed)"
    assert v("aril-fpu", "RESULT: PASSED\n") == "PASS" and v("aril-fpu", "done\n") == "FAIL (no 'RESULT: PASSED')"
    narrow = ("fp32ToBf16: 40 corners against their table, 1196608 values against round to nearest even (tie sweep 196608, random 1000000), 0 errors\n"
              "RESULT: PASSED\nlint fp32ToBf16: 0 warnings, 0 errors\n")
    assert v("aril-narrow", narrow) == "PASS" and v("aril-narrow", narrow + "RESULT: FAILED\n").startswith("FAIL")


def test_regression_target():
    env = {k: x for k, x in os.environ.items() if not k.startswith("MAKE")}
    run = lambda *a: subprocess.run(["make", "-s", "-n", *a, f"PYTHON={sys.executable}"], cwd=ROOT, capture_output=True, text=True, env=env)
    for args, pts in ((("SMALL=1",), reg.SPEC.points(True, list(reg.FORMATS))), ((), reg.SPEC.points(False, list(reg.FORMATS))),
                      (("SMALL=1", "FMT=int8"), reg.SPEC.points(True, ["int8"]))):
        r = run("regression", *args)
        lines = r.stdout.splitlines()
        names = [x.split("--name ")[1].split()[0] for x in lines if "--action point " in x]
        assert r.returncode == 0 and names == pts, (args, r.stderr[-200:], names[:4])
        assert all(x.rstrip().endswith("--keep-going") and "--run " in x for x in lines if "--action point " in x), lines[:1]
        assert "--action summary" in lines[-1] and ("--small" in lines[-1]) == ("SMALL=1" in args), lines[-1]
    one = run("regression/fp32/N8_T2/mesh").stdout.splitlines()
    assert len(one) == 1 and "--keep-going" not in one[0] and "--run" not in one[0], one
    assert run("regression", "SMALL=2").returncode != 0 and run("regression", "FMT=fp16").returncode != 0
    for extra in ({"FMT": "int8"}, {"SMALL": "1"}, {"SMALL": "2"}):  # exported, they do not narrow it: only make's command line does
        r = subprocess.run(["make", "-s", "-n", "regression", f"PYTHON={sys.executable}"], cwd=ROOT, capture_output=True, text=True, env=dict(env, **extra))
        names = [x.split("--name ")[1].split()[0] for x in r.stdout.splitlines() if "--action point " in x]
        assert r.returncode == 0 and names == reg.SPEC.points(False, list(reg.FORMATS)), (extra, r.stderr[-200:], len(names))
    r = subprocess.run(["make", "-s", "-n", "regression", f"PYTHON={sys.executable}"], cwd=ROOT, capture_output=True, text=True, env=dict(env, FMT="fp16"))
    assert r.returncode != 0 and "FMT=fp16: not one of fp32 bf16 int8" in r.stderr, r.stderr[-200:]  # an exported FMT is still the build's define: fp16 stops make
    r = run("mesh-regression", "MATRIX_SIZE=32", "REGRESSION_OPTS=--tiles 8")  # one N, as before: every tile size, or --tiles
    assert r.returncode == 0 and all(x in r.stdout for x in ("regression.py", "--matrix-size 32", "--group all", "--tiles 8")), r.stdout


MIN_TESTS = 5  # tests this file holds; fewer run means one was lost, which is a failure

if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    if len(tests) < MIN_TESTS:
        print(f"FAIL: {len(tests)} tests ran, fewer than MIN_TESTS {MIN_TESTS}")
        sys.exit(1)
    print(f"ALL {len(tests)} PASSED")
