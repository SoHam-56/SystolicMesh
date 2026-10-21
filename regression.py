#!/usr/bin/env python3
"""
regression.py — SystolicMesh IP Regression Suite
=================================================
Exploits make's incremental compilation to avoid redundant rebuilds.

How it works
------------
A build is one configuration: make's FMT (the define) and the TB parameters MATRIX_SIZE, TILE_SIZE, NUM_TEST_SETS and COLLAPSE_K,
which make passes as Verilator -G overrides; no script edits a source file. The stimulus (.mem files) is read at run time, so tests
of one configuration share one build: Verilator rebuilds only when the configuration changes.

Execution order
---------------
  first the model checks (make aril-fpu, aril-narrow, model-tests), each against its own reference
  for each tile T:
      for each matmul test: write .mem files, make FMT=f MATRIX_SIZE=N TILE_SIZE=T NUM_TEST_SETS=MATMUL_NUM_SETS COLLAPSE_K=ck
      for each conv test:   the same with CONV_NUM_SETS (the first test of each builds, the rest only simulate)

Usage
-----
  python regression.py --matrix-size 16
  python regression.py --matrix-size 16 --group matmul
  python regression.py --matrix-size 16 --group conv
  python regression.py --matrix-size 16 --fast
  python regression.py --matrix-size 64
  python regression.py --matrix-size 16 --format int8 --collapse-k 0
  python regression.py --matrix-size 16 --no-checks   # simulations only, without the model checks

Exit code: 0 = all pass, 1 = any failure or interrupted.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

from conv_tests import CONV_NUM_SETS, CONV_TESTS
from matmul_tests import MATMUL_NUM_SETS, MATMUL_TESTS, pow2_tile_sizes
import stim_format

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ArithmeticLibrary", "Common", "models"))
import truth  # noqa: E402  AriL's true-function gate: the mesh's sums against the float64 sum of its operands

FORMATS = ("fp32", "bf16", "int8")  # make's FMT: the build's define
FMT, COLLAPSE = "fp32", 1  # set from --format and --collapse-k

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
MATRIX_N = 0  # this run's N, set in main
STIM_DIR = os.path.join(ROOT, "testbenches", "stimulus")
TRUTH_DUMP = os.path.join(ROOT, "Verilator", "truth_dump.txt")  # TB_SystolicMesh's sums and words of every result it checked
RESULTS_DIR = os.path.join(ROOT, "testbenches", "results", "readiness")

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------
_G = "\033[92m"
_R = "\033[91m"
_Y = "\033[93m"
_B = "\033[94m"
_X = "\033[0m"
_O = "\033[1m"

ok = lambda s: f"{_G}{_O}{s}{_X}"
err = lambda s: f"{_R}{_O}{s}{_X}"
wrn = lambda s: f"{_Y}{s}{_X}"
hdr = lambda s: f"{_O}{_B}{s}{_X}"

# ---------------------------------------------------------------------------
# Make runner
# ---------------------------------------------------------------------------


def _make(tile: int, num_sets: int) -> tuple:
    """Runs `make` in the project root for one configuration; returns (stdout and stderr, wall seconds). Verilator rebuilds only on a new configuration."""
    t0 = time.time()
    r = subprocess.run(["make", *build_args(FMT, MATRIX_N, tile, num_sets, COLLAPSE)], cwd=ROOT, capture_output=True, text=True)
    out = r.stdout + r.stderr
    if r.returncode != 0:
        out += f"\nBUILDFAIL: make exited {r.returncode}\n"
    return out, time.time() - t0


def make_args(n: int, ck: int) -> list:
    """The build's make variables: the large code model where the binary passes 2 GB (collapse-k 0 at N = 64)."""
    return ["LARGE_CODE=1"] if ck == 0 and n >= 64 else []


def build_args(fmt: str, n: int, tile: int, num_sets: int, ck: int) -> list:
    """make's variables for one configuration: the define (FMT), TB_SystolicMesh's sizes as -G overrides, and make_args."""
    return [f"FMT={fmt}", f"MATRIX_SIZE={n}", f"TILE_SIZE={tile}", f"NUM_TEST_SETS={num_sets}", f"COLLAPSE_K={ck}", *make_args(n, ck)]


# Make targets checked against their own references, not mesh_model's agreement with the RTL (a bug shared by both passes that).
MODEL_CHECKS = ("aril-fpu", "aril-narrow", "model-tests")


def _run_checks() -> list:
    """Runs each MODEL_CHECKS target, its output in RESULTS_DIR/check_<target>.log: [(target, "PASS" or "FAIL (exit n)")]."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = []
    for t in MODEL_CHECKS:
        print(f"  [check] make {t}", end="  ", flush=True)
        r = subprocess.run(["make", t], cwd=ROOT, capture_output=True, text=True)
        with open(os.path.join(RESULTS_DIR, f"check_{t}.log"), "w") as fh:
            fh.write(r.stdout + r.stderr)
        v = "PASS" if r.returncode == 0 else f"FAIL (exit {r.returncode})"
        print(ok(v) if v == "PASS" else err(v))
        out.append((t, v))
    return out


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


def _parse(raw: str) -> dict:
    if "Assertion failed" in raw:  # a firing assertion leaves the exit code and the TB's counts untouched
        return dict(status="ASSERT", passed=0, failed=0, elements=0, tol=0, fail_els=0, avg_cyc=0)
    if "BUILDFAIL" in raw or "version `GLIBC" in raw or "%Error" in raw:
        return dict(status="NO-RUN", passed=0, failed=0, elements=0, tol=0,
                    fail_els=0, avg_cyc=0)
    if "FATAL" in raw or "Timeout" in raw:
        return dict(
            status="TIMEOUT",
            passed=0,
            failed=0,
            elements=0,
            tol=0,
            fail_els=0,
            avg_cyc=0,
        )

    def _i(pat):
        m = re.search(pat, raw)
        return int(m.group(1)) if m else 0

    passed = _i(r"Passed Sets:\s*(\d+)")
    failed = _i(r"Failed Sets:\s*(\d+)")
    elements = _i(r"Total Elements:\s*(\d+)")
    tol = _i(r"Tol Passed Els:\s*(\d+)")
    fail_els = raw.count("[FAIL]")
    cycles = re.findall(r"Set\s+\d+\s*:\s*(\d+)\s*cycles", raw)
    avg_cyc = sum(int(c) for c in cycles) // len(cycles) if cycles else 0
    # No sets and no elements means the testbench never checked anything. Reporting that
    # as FAIL hides the real cause, which is usually a stale or unrunnable simulator binary.
    if passed + failed == 0 or elements == 0:
        return dict(status="NO-RUN", passed=0, failed=0, elements=0, tol=0,
                    fail_els=0, avg_cyc=avg_cyc)
    status = "PASS" if fail_els == 0 and "SUCCESS" in raw else "FAIL"

    return dict(
        status=status,
        passed=passed,
        failed=failed,
        elements=elements,
        tol=tol,
        fail_els=fail_els,
        avg_cyc=avg_cyc,
    )


# ---------------------------------------------------------------------------
# True-function gate: each result's sums against the float64 sum of the operands the mesh received
# ---------------------------------------------------------------------------


def _words(path: str) -> np.ndarray:
    with open(path) as fh:
        return np.array([int(t, 16) for t in fh.read().split()], dtype=np.int64)


def _operands(stim: str, name: str, N: int) -> tuple:
    """(passes, bias words or None) of the result checked against expected file name: matrixC<sfx>.mem or accC.mem."""
    f = lambda n: os.path.join(stim, n)
    if name == "accC.mem":
        passes, p = [], 0
        while os.path.exists(f(f"accA_{p}.mem")):
            passes.append((_words(f(f"accA_{p}.mem")).reshape(N, N), _words(f(f"accB_{p}.mem")).reshape(N, N)))
            p += 1
        bias = _words(f("accBias.mem")) if os.path.exists(f("accBias.mem")) else None
        return passes, bias
    m = re.fullmatch(r"matrixC(_\d+)?\.mem", name)
    if not m:
        raise ValueError(f"truth dump: unknown expected file {name}")
    sfx = m.group(1) or ""
    A, B = _words(f(f"matrixA{sfx}.mem")).reshape(N, N), _words(f(f"matrixB{sfx}.mem")).reshape(N, N)
    if os.path.exists(f(f"packShift{sfx}.mem")):  # a packed set sums each block alone: the PEs skip off-block weights
        b = N >> int(_words(f(f"packShift{sfx}.mem"))[0])
        B = np.where(np.kron(np.eye(N // b), np.ones((b, b))).astype(bool), B, 0)
    bias = _words(f(f"matrixBias{sfx}.mem")) if os.path.exists(f(f"matrixBias{sfx}.mem")) else None
    return [(A, B)], bias


def transfer(fmt: str, written, delivered) -> "truth.Result":
    """fp32 and int8 write each sum as its word: the word the result link delivers must be that sum, exactly."""
    written, delivered = np.asarray(written, dtype=np.int64).ravel(), np.asarray(delivered, dtype=np.int64).ravel()
    bad = np.flatnonzero(written != delivered)
    return truth.Result(bad.size == 0, fmt, "transfer", int(written.size), float(bad.size), 0.0, int(bad[0]) if bad.size else 0,
                        int(bad.size), int(bad.size), 0, 0, 0, [(int(k), int(written[k]), int(delivered[k])) for k in bad[:truth.FIRST]])


def truth_gate(dump: str, stim: str, N: int, fmt: str) -> dict:
    """{stage: Result} over every result in the dump: matmul on the sums before narrowing, then bf16 narrow on (sum, word) pairs,
    fp32 and int8 transfer (the delivered word is the sum written)."""
    if not os.path.exists(dump):
        raise RuntimeError(f"no truth dump at {dump}")
    tok = open(dump).read().split("\n")
    out, cache, i = {"matmul": [], "narrow": [], "transfer": []}, {}, 0
    while i < len(tok):
        if not tok[i].startswith("R "):
            i += 1
            continue
        name = tok[i].split()[1]
        rows = [ln.split() for ln in tok[i + 1:i + 1 + N * N]]
        if len(rows) != N * N or any(len(r) != 2 for r in rows):
            raise RuntimeError(f"truth dump: result '{tok[i]}' holds {len(rows)} of {N * N} element lines")
        pre = np.array([int(r[0], 16) for r in rows], dtype=np.int64).reshape(N, N)
        post = np.array([int(r[1], 16) for r in rows], dtype=np.int64)
        if name not in cache:
            cache[name] = _operands(stim, name, N)
        passes, bias = cache[name]
        out["matmul"].append(truth.gate_matmul(fmt, passes, pre, bias))
        if fmt == "bf16":
            out["narrow"].append(truth.gate_narrow("bf16", pre.ravel(), post))
        else:
            out["transfer"].append(transfer(fmt, pre, post))
        i += 1 + N * N
    if not out["matmul"]:
        raise RuntimeError(f"truth dump {dump} holds no result")
    return {k: truth.merge(v) for k, v in out.items() if v}


def truth_lines(res: dict) -> tuple:
    """(lines, misses): a stage line per gate, and a [FAIL] line per stage that misses."""
    lines, bad = [], []
    for k, r in res.items():
        lines.append("        " + truth.stage_line(r))
        if not r.ok:
            w = {"narrow": f"{r.worst_lsb} ULP", "transfer": f"{r.over} words differ"}.get(k, truth.worst_text(r.worst) + " of sum|ab|")
            how = {"narrow": "index, fp32 sum, word, round to nearest even", "transfer": "index, sum written, word delivered"}.get(
                k, "index, sum, true sum, sum|ab|")
            bad.append(f"[FAIL] truth: {k} {w}, {r.over} of {r.n} over the bound; first ({how}): {r.first[:3]}")
    return lines, bad


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------


def _print_result(r: dict) -> None:
    sym = ok("PASS") if r["status"] == "PASS" else err("FAIL")
    tot = max(r["elements"], 1)
    ep = 100 * (r["elements"] - r["tol"] - r["fail_els"]) / tot
    tp = 100 * r["tol"] / tot
    sets = f"{r['passed']}/{r['passed'] + r['failed']}"
    print(
        f"      {sym}  sets {sets:<5}  "
        f"exact {ep:5.1f}%  tol {tp:5.1f}%  "
        f"fail {r['fail_els']:3d}  "
        f"{r['avg_cyc']:6d} cyc  "
        f"{r['wall']:5.1f}s"
    )


# ---------------------------------------------------------------------------
# Group runner — patch TB once, simulate all tests in the group
# ---------------------------------------------------------------------------


def _run_group(tests: list, tile: int, N: int, num_sets: int, group_label: str) -> list:
    """Every test of one configuration (FMT, N, tile, num_sets, collapse-k): the first make builds it, the rest only simulate."""
    results = []

    print(hdr(f"\n  ╔══  {group_label}  TILE={tile}  N={N}  SETS={num_sets}"))
    stim_format.configure(FMT, tile, COLLAPSE)

    for idx, test in enumerate(tests):
        print(f"  ║  [{idx+1}/{len(tests)}] {test['name']}", end="  ", flush=True)

        # Write stimulus: only .mem files change
        os.makedirs(STIM_DIR, exist_ok=True)
        test["gen_fn"](STIM_DIR, N)
        stim_format.check_widths(STIM_DIR)
        if os.path.exists(TRUTH_DUMP):
            os.remove(TRUTH_DUMP)  # a stale dump would be gated as this test's

        # make: the configuration's first test builds it, the rest reuse the build
        raw, wall = _make(tile, num_sets)

        parsed = _parse(raw)
        if parsed["status"] == "PASS" and test.get("expect", "") not in raw:  # a stage the test needs never ran
            parsed["status"] = "FAIL"
            print(err(f"[FAIL] '{test['expect']}' is not in the log"), end="  ")
        tres = {}
        if parsed["status"] in ("PASS", "FAIL"):  # the TB ran to its summary: gate what it dumped
            try:
                tres = truth_gate(TRUTH_DUMP, STIM_DIR, N, FMT)
                tl, tbad = truth_lines(tres)
            except (RuntimeError, ValueError, OSError) as e:
                tl, tbad = [], [f"[FAIL] truth: none ({e})"]
            if os.path.exists(TRUTH_DUMP):  # kept beside the log, so a run's sums can be judged again offline
                os.makedirs(RESULTS_DIR, exist_ok=True)
                shutil.copy(TRUTH_DUMP, os.path.join(RESULTS_DIR, f"{test['name']}_N{N}_T{tile}_truth_dump.txt"))
            raw += "\n".join(["", "TRUE-FUNCTION GATE (truth.py; the bit-exact compare above is a diagnostic, still required):"] + tl + tbad) + "\n"
            if tbad:
                parsed["status"] = "FAIL"
        log_name = f"{test['name']}_N{N}_T{tile}.log"
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(os.path.join(RESULTS_DIR, log_name), "w") as fh:
            fh.write(raw)

        r = dict(
            name=test["name"],
            group=test.get("group", ""),
            tile=tile,
            N=N,
            wall=wall,
            log=log_name,
            truth=tres,
            **parsed,
        )
        results.append(r)
        _print_result(r)
        for ln in [x for x in raw.splitlines() if x.startswith(("        TRUTH ", "[FAIL] truth:"))]:
            print(f"      {ln.strip()}")
        if "BUILDFAIL" in raw:  # the next test would rebuild the same config and fail the same way
            for rest in tests[idx + 1:]:
                results.append(dict(name=rest["name"], group=rest.get("group", ""), tile=tile, N=N, wall=0.0,
                                    log=log_name, status="NO-RUN", passed=0, failed=0, elements=0, tol=0,
                                    fail_els=0, avg_cyc=0))
            print(wrn(f"  ║  build failed at T={tile}: {len(tests) - idx - 1} remaining tests not run"))
            break

    print("  ╚══")

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _report(results: list, N: int, fast: bool, group: str, checks: list) -> str:
    """Plain-text readiness report alongside the per-run logs."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    passed = sum(1 for r in results if r["status"] == "PASS")
    total = len(results)
    bad = total - passed + sum(v != "PASS" for _, v in checks) + (total == 0)  # no simulation is a failure
    verdict = ("READY" if bad == 0
               else f"NOT READY ({bad} failure{'s' if bad > 1 else ''})")

    W = 96
    L = []
    L.append("=" * W)
    L.append("SYSTOLICMESH IP — READINESS REPORT")
    L.append("=" * W)
    L.append("")
    L.append(f"Date          : {ts}")
    L.append(f"Matrix size   : {N}x{N}")
    L.append(f"Group         : {group}")
    L.append(f"Mode          : {'fast' if fast else 'full'}")
    L.append(f"Runs          : {total}    Passed: {passed}    Failed: {total - passed}")
    L.append("")
    L.append(f"RESULT        : {verdict}")
    L.append("")

    def _rows(title, rows):
        if not rows:
            return
        L.append("-" * W)
        L.append(title.upper())
        L.append("-" * W)
        L.append(f"{'test':<22}{'tile':>6}{'result':>9}{'sets':>8}{'exact':>9}"
                 f"{'tol':>9}{'fails':>7}{'avg cyc':>10}{'wall':>8}")
        L.append("-" * W)
        for r in rows:
            tot = max(r["elements"], 1)
            ep = 100 * (r["elements"] - r["tol"] - r["fail_els"]) / tot
            tp = 100 * r["tol"] / tot
            sets = f"{r['passed']}/{r['passed'] + r['failed']}"
            L.append(f"{r['name']:<22}{'T=' + str(r['tile']):>6}{r['status']:>9}"
                     f"{sets:>8}{ep:>8.1f}%{tp:>8.1f}%{r['fail_els']:>7}"
                     f"{r['avg_cyc']:>10}{r['wall']:>7.0f}s")
        L.append("")

    _rows("Matrix multiplication", [r for r in results if r["group"] == "matmul"])
    _rows("Convolution", [r for r in results if r["group"] == "conv"])

    L.append("-" * W)
    L.append("TRUE-FUNCTION GATE PER TEST (matmul: worst |err| / sum|ab| of the sums; narrow: worst ULP; misses of all elements)")
    L.append("-" * W)
    for r in results:
        t = r.get("truth") or {}
        mm, nw = t.get("matmul"), t.get("narrow")
        L.append(f"{r['name']:<22}{'T=' + str(r['tile']):>6}  "
                 + (f"matmul {truth.worst_text(mm.worst):>9} {mm.over:>5} of {mm.n:<7}" if mm else "matmul not gated")
                 + (f"  narrow {nw.worst_lsb} ULP {nw.over} of {nw.n}" if nw else ""))
    L.append("")

    if checks:
        L.append("-" * W)
        L.append("MODEL CHECKS (own references: hand corners, an integer RNE, float64)")
        L.append("-" * W)
        L += [f"{'make ' + t:<22}{v:>9}  check_{t}.log" for t, v in checks]
        L.append("")

    L.append("-" * W)
    L.append("NOTES")
    L.append("-" * W)
    L.append("* One build per (tile x group): make passes FMT and the sizes; no source file is edited")
    L.append("* Gate: truth.py, the sums before narrowing within max(1% of sum|ab|, 2^-126) of the float64 sum of the operands (int8: exact, int32-wrapped);"
             " bf16 narrowing exact; fp32 and int8 delivered words equal to the sums written")
    L.append("* Compare: bit-exact against mesh_model (matmul in the float formats, matmul_int in int8), a diagnostic that is still required")
    L.append(f"* Format: {FMT}  |  collapse-k {COLLAPSE}")
    L.append("* Tool: Verilator  |  Clock: 10 ns")
    L.append("=" * W)

    path = os.path.join(RESULTS_DIR, f"readiness_report_{FMT}_ck{COLLAPSE}.log")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SystolicMesh regression suite",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--matrix-size",
        type=int,
        required=True,
        help="Matrix dimension N (power of 2, e.g. 8, 16, 32, 64)",
    )
    parser.add_argument(
        "--group",
        choices=["all", "matmul", "conv"],
        default="all",
        help=(
            "all    — every matmul and conv test  (default)\n"
            "matmul — matrix-multiplication tests only\n"
            "conv   — convolution tests only"
        ),
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="One tile size only (middle power-of-2 divisor of N)",
    )
    parser.add_argument("--tiles", type=int, nargs="+", help="run only these tile sizes (default: every power-of-2 divisor)")
    parser.add_argument("--format", choices=sorted(FORMATS), default="fp32",
                        help="number format of the build; every format compares bit for bit against mesh_model")
    parser.add_argument("--collapse-k", type=int, choices=[0, 1], default=1,
                        help="1: full-depth arrays (the RTL default); 0: depth slices and the reduce tree")
    parser.add_argument("--no-checks", action="store_true",
                        help="skip the model checks (make aril-fpu, aril-narrow, model-tests); SIENNA's gate runs them as steps")
    args = parser.parse_args()
    global FMT, COLLAPSE, MATRIX_N
    FMT, COLLAPSE, MATRIX_N = args.format, args.collapse_k, args.matrix_size
    stim_format.configure(FMT, 2, COLLAPSE)

    N = args.matrix_size
    if N < 2 or (N & (N - 1)) != 0:
        print(err(f"[ERROR] --matrix-size must be a power of 2  (got {N})"))
        sys.exit(1)

    # ── Build test lists ──────────────────────────────────────────────────
    mm_tests = [dict(t, group="matmul") for t in MATMUL_TESTS]
    conv_tests = [dict(t, group="conv") for t in CONV_TESTS]

    if args.group == "matmul":
        run_mm, run_conv = mm_tests, []
    elif args.group == "conv":
        run_mm, run_conv = [], conv_tests
    else:
        run_mm, run_conv = mm_tests, conv_tests
    run_mm = [t for t in run_mm if COLLAPSE or not t.get("packed")]  # packing is collapse-k only (the mesh asserts it)
    run_mm = [t for t in run_mm if not (t.get("float_only") and stim_format.is_int())]  # float values beyond int8's range

    # Conv tests use a general 3x3 layout (2x2 below N = 9) when N is not a perfect square; see conv_tests._general_pair.

    # ── Tile list ─────────────────────────────────────────────────────────
    all_tiles = pow2_tile_sizes(N)
    tiles = [all_tiles[len(all_tiles) // 2]] if args.fast else all_tiles
    if args.tiles:
        bad = [x for x in args.tiles if x not in all_tiles]
        if bad:
            print(err(f"[ERROR] --tiles {bad}: valid tile sizes for N={N} are {all_tiles}"))
            sys.exit(1)
        tiles = [x for x in all_tiles if x in args.tiles]

    # For conv, find the maximum NUM_TEST_SETS needed across all conv tests.
    # We compute this once here so it's stable for the whole run.
    conv_num_sets = CONV_NUM_SETS
    if run_conv:
        os.makedirs(STIM_DIR, exist_ok=True)
        for t in run_conv:
            n = t["gen_fn"](STIM_DIR, N)
            if n > conv_num_sets:
                conv_num_sets = n

    # ── Compile count — for display only: one build per (tile × group) ──
    n_configs = len(tiles) * (bool(run_mm) + bool(run_conv))
    n_sims = (len(run_mm) + len(run_conv)) * len(tiles)

    # ── Header ────────────────────────────────────────────────────────────
    print(hdr(f"\n{'═'*64}"))
    print(hdr(f"  SystolicMesh IP — Regression Suite"))
    print(hdr(f"{'═'*64}"))
    print(f"  Matrix size  : {N}×{N}")
    print(f"  Group        : {args.group}")
    print(f"  Tile sweep   : {tiles}{'  (fast)' if args.fast else ''}")
    print(
        f"  Tests        : {len(run_mm) + len(run_conv)}  "
        f"({len(run_mm)} matmul + {len(run_conv)} conv)"
    )
    print(f"  TB compiles  : {n_configs}  (one per tile × group)")
    print(f"  Simulations  : {n_sims}")
    print(hdr(f"{'═'*64}"))

    all_results: list = []
    checks = [] if args.no_checks else _run_checks()

    # ── Main loop ─────────────────────────────────────────────────────────
    try:
        for tile in tiles:
            if run_mm:
                r = _run_group(run_mm, tile, N, MATMUL_NUM_SETS, "MATMUL")
                all_results.extend(r)

            if run_conv:
                r = _run_group(run_conv, tile, N, conv_num_sets, "CONV")
                all_results.extend(r)

    except KeyboardInterrupt:
        print(wrn("\n  [Interrupted]"))

    # ── Summary ───────────────────────────────────────────────────────────
    passed = sum(1 for r in all_results if r["status"] == "PASS")
    total = len(all_results)

    print(hdr(f"\n{'═'*64}"))
    print(hdr(f"  SUMMARY  —  {N}×{N}  [{args.group}]"))
    print(hdr(f"{'═'*64}"))
    print(f"  Simulations : {total}")
    print(f"  Passed      : {ok(passed)}")
    print(f"  Failed      : {err(total - passed) if total - passed else ok(0)}")
    print()

    by_name: dict = {}
    for r in all_results:
        by_name.setdefault(r["name"], []).append(r)

    for name, rows in by_name.items():
        all_pass = all(r["status"] == "PASS" for r in rows)
        sym = ok("PASS") if all_pass else err("FAIL")
        tlist = ", ".join(f"T{r['tile']}" for r in rows)
        print(f"  {sym}  {name:<28}  [{tlist}]")
    for t, v in checks:
        print(f"  {ok('PASS') if v == 'PASS' else err('FAIL')}  {'make ' + t:<28}  [model check{'' if v == 'PASS' else ', ' + v}]")

    gated = [r for r in all_results if r.get("truth")]
    for k in ("matmul", "narrow", "transfer"):
        m = truth.merge([r["truth"].get(k) for r in gated])
        if m:
            print(f"  {truth.stage_line(m)}")
    print(f"  TRUTH: {len(gated)} simulations gated, {sum(not all(x.ok for x in r['truth'].values()) for r in gated)} with misses")
    if total == 0:  # finishing cleanly with nothing simulated is not a pass
        print(err(f"  FAIL: no simulation ran (group {args.group}, tiles {tiles}, format {FMT}, collapse-k {COLLAPSE})"))
    clean = total > 0 and passed == total and all(v == "PASS" for _, v in checks)
    verdict = (
        ok("✅  All tests passed — Sanity Clean")
        if clean
        else err("❌  Failures detected — see logs")
    )
    print(hdr(f"\n  Verdict : {verdict}"))

    rpt = _report(all_results, N, args.fast, args.group, checks)
    print(f"  Report  : {rpt}")
    print(hdr(f"{'═'*64}\n"))

    sys.exit(0 if clean else 1)


# ── make regression: the model checks once, then the mesh at every N, tile size and format ──


def _engine():
    """The library's make regression engine, under its own name (SIENNA and GPNAE each have a regression.py too)."""
    import importlib.util
    s = importlib.util.spec_from_file_location("aril_regress", os.path.join(ROOT, "ArithmeticLibrary", "Common", "regress.py"))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


regress = _engine()
MODEL_CHECKS = ("model-tests", "aril-fpu", "aril-narrow")  # against their own references, format-independent
MODEL_TEST_FILES = 4  # model-tests' scripts, each ends 'ALL PASS' or 'ALL <n> PASSED'


class MeshSpec(regress.Spec):
    title = "SYSTOLICMESH REGRESSION"
    formats = tuple(FORMATS)
    sized = True
    drop = ("MATRIX_SIZE", "TILE_SIZE", "NUM_TEST_SETS", "COLLAPSE_K", "REGRESSION_GROUP", "FAST", "REGRESSION_OPTS")

    def points(self, small, formats):
        if not formats:
            return []
        return list(MODEL_CHECKS) + [f"{f}/N{n}_T{t}/{k}" for f in formats for n in (regress.SIZES_SMALL if small else regress.SIZES)
                                     for t in pow2_tile_sizes(n) for k in ("mesh", "mesh-ck0")]

    def parse(self, name):
        if name in MODEL_CHECKS:
            return {"n": 0, "kind": name}
        m = re.fullmatch(r"(\w+)/N(\d+)_T(\d+)/mesh(-ck0)?", name)
        if not m or m.group(1) not in FORMATS or int(m.group(2)) not in regress.SIZES or int(m.group(3)) not in pow2_tile_sizes(int(m.group(2))):
            raise ValueError(f"{name}: points are {', '.join(MODEL_CHECKS)} and <fmt>/N<n>_T<t>/mesh (N 8, 16, 32 or 64, T a power-of-2 divisor)")
        return {"n": int(m.group(2)), "tile": int(m.group(3)), "fmt": m.group(1), "kind": "mesh", "ck": 0 if m.group(4) else 1}

    def argv(self, name, opts):
        p = self.parse(name)
        if p["kind"] != "mesh":
            return ["make", p["kind"]] + ([f"PYTHON={sys.executable}"] if p["kind"] == "model-tests" else [])
        return ["make", "mesh-regression", f"MATRIX_SIZE={p['n']}",
                f"REGRESSION_OPTS=--format {p['fmt']} --collapse-k {p['ck']} --tiles {p['tile']} --no-checks", f"PYTHON={sys.executable}"]

    def verdict(self, name, text, rc):
        v = regress.common_verdict(text, rc)
        if v:
            return v
        text = regress.ANSI.sub("", text)
        if name == "model-tests":
            n = len(re.findall(r"^ALL (?:PASS|\d+ PASSED)$", text, re.M))
            return "PASS" if n == MODEL_TEST_FILES else f"FAIL ({n} of {MODEL_TEST_FILES} model test files passed)"
        if name == "aril-fpu":
            return regress.marker_verdict(text, rc, (r"^RESULT: PASSED",), (r"RESULT: FAILED",))
        if name == "aril-narrow":
            return regress.marker_verdict(text, rc, (r"^RESULT: PASSED", r"^bad_format OK:", r"^fpNarrow lint \(MAN_W 7, 23\): 0 warnings, 0 errors$"),
                                          (r"RESULT: FAILED",))
        s, m = re.findall(r"^\s*Simulations\s*:\s*(\d+)\s*$", text, re.M), re.findall(r"^\s*Passed\s*:\s*(\d+)\s*$", text, re.M)
        g = re.findall(r"^\s*TRUTH: (\d+) simulations gated, (\d+) with misses$", text, re.M)
        miss = sorted(set(re.findall(r"^\s*\[FAIL\] truth: (\w+)", text, re.M)))
        if miss:
            return f"FAIL (true function: {', '.join(miss)} misses in {g[-1][1] if g else '?'} simulations)"
        if not s or not m or "Sanity Clean" not in text:
            return "FAIL (no 'Simulations : X', 'Passed : X' with Sanity Clean)"
        if not g or int(g[-1][0]) != int(s[-1]):
            return f"FAIL ({g[-1][0] if g else 0} of {s[-1]} simulations gated on the true function)"
        x, y = int(m[-1]), int(s[-1])
        return "PASS" if x == y > 0 else f"FAIL ({x} of {y} simulations passed)"


SPEC = MeshSpec(ROOT, os.path.join(ROOT, "testbenches", "results", "regression"))


if __name__ == "__main__":
    if "--action" in sys.argv[1:]:  # make regression's points, one point, the table
        regress.main(SPEC)
    else:
        main()
