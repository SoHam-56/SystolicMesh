#!/usr/bin/env python3
"""make regression's list, commands and verdicts: the model checks once, the mesh at every N, tile size and format (collapse-k 1)."""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import regression as reg  # noqa: E402

ROOT = reg.ROOT
CHECKS = ["model-tests", "aril-fpu", "aril-narrow"]


def test_points():
    small, full = reg.SPEC.points(True, list(reg.FORMATS)), reg.SPEC.points(False, list(reg.FORMATS))
    assert small[:3] == full[:3] == CHECKS, small[:4]  # format-independent: once
    sweep = lambda pts: sorted({(reg.SPEC.parse(p)["n"], reg.SPEC.parse(p)["tile"]) for p in pts[3:]})
    assert sweep(small) == [(8, 2), (8, 4), (8, 8), (16, 2), (16, 4), (16, 8), (16, 16)], sweep(small)
    assert sweep(full) == sweep(small) + [(32, t) for t in (2, 4, 8, 16, 32)] + [(64, t) for t in (2, 4, 8, 16, 32, 64)], sweep(full)
    assert len(small) == len(set(small)) == 3 + 7 * 3 and len(full) == len(set(full)) == 3 + 18 * 3, (len(small), len(full))
    assert reg.SPEC.points(True, ["int8"]) == CHECKS + [p for p in small if p.startswith("int8/")] and reg.SPEC.points(False, []) == []
    for bad in ("fp32/N8_T16/mesh", "fp16/N8_T2/mesh", "fp32/N12_T4/mesh", "fp32/model-tests", "fp32/N8_T2/pipeline", "nosuch"):
        try:
            reg.SPEC.parse(bad)
        except ValueError:
            continue
        raise AssertionError(f"parse accepted {bad}")


def test_commands():
    py = sys.executable
    assert reg.SPEC.argv("bf16/N32_T8/mesh", None) == ["make", "mesh-regression", "MATRIX_SIZE=32",
                                                       "REGRESSION_OPTS=--format bf16 --collapse-k 1 --tiles 8 --no-checks", f"PYTHON={py}"]
    assert reg.SPEC.argv("model-tests", None) == ["make", "model-tests", f"PYTHON={py}"] and reg.SPEC.argv("aril-narrow", None)[:2] == ["make", "aril-narrow"]


def test_verdicts():
    mesh = "  Simulations : 12\n  Passed      : \x1b[92m12\x1b[0m\n  Verdict : ✅  All tests passed — Sanity Clean\n"
    v = lambda p, t, rc=0: reg.SPEC.verdict(p, t, rc)
    assert v("fp32/N8_T2/mesh", mesh) == "PASS" and v("fp32/N8_T2/mesh", mesh, 1) == "FAIL (exit 1)"
    assert v("fp32/N8_T2/mesh", mesh.replace("\x1b[92m12", "\x1b[92m11")) == "FAIL (11 of 12 simulations passed)"
    assert v("fp32/N8_T2/mesh", mesh.replace("12", "0")) == "FAIL (0 of 0 simulations passed)"
    assert v("fp32/N8_T2/mesh", "Simulations : 3\n") == "FAIL (no 'Simulations : X', 'Passed : X' with Sanity Clean)"
    assert v("fp32/N8_T2/mesh", mesh + "[x] Assertion failed in TB\n") == "FAIL (log: [x] Assertion failed in TB)"
    tests = "ALL PASS\nALL 13 PASSED\nALL 4 PASSED\nALL 3 PASSED\n"
    assert v("model-tests", tests) == "PASS" and v("model-tests", tests.replace("ALL 4 PASSED\n", "")) == "FAIL (3 of 4 model test files passed)"
    assert v("aril-fpu", "RESULT: PASSED\n") == "PASS" and v("aril-fpu", "done\n") == "FAIL (no 'RESULT: PASSED')"
    narrow = "RESULT: PASSED\nfpNarrow lint (MAN_W 7, 23): 0 warnings, 0 errors\nbad_format OK: fired\n"
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
    for extra in ({"FMT": "int8"}, {"SMALL": "1"}, {"FMT": "fp16", "SMALL": "2"}):  # exported, they do not narrow it: only make's command line does
        r = subprocess.run(["make", "-s", "-n", "regression", f"PYTHON={sys.executable}"], cwd=ROOT, capture_output=True, text=True, env=dict(env, **extra))
        names = [x.split("--name ")[1].split()[0] for x in r.stdout.splitlines() if "--action point " in x]
        assert r.returncode == 0 and names == reg.SPEC.points(False, list(reg.FORMATS)), (extra, r.stderr[-200:], len(names))
    r = run("mesh-regression", "MATRIX_SIZE=32", "REGRESSION_OPTS=--tiles 8")  # one N, as before: every tile size, or --tiles
    assert r.returncode == 0 and all(x in r.stdout for x in ("regression.py", "--matrix-size 32", "--group all", "--tiles 8")), r.stdout


MIN_TESTS = 4  # tests this file holds; fewer run means one was lost, which is a failure

if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    if len(tests) < MIN_TESTS:
        print(f"FAIL: {len(tests)} tests ran, fewer than MIN_TESTS {MIN_TESTS}")
        sys.exit(1)
    print(f"ALL {len(tests)} PASSED")
