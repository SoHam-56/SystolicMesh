#!/usr/bin/env python3
"""
regression.py — SystolicMesh IP Regression Suite
=================================================
Exploits make's incremental compilation to avoid redundant rebuilds.

How it works
------------
The TB has three compile-time localparams: MATRIX_SIZE, TILE_SIZE,
NUM_TEST_SETS.  These define one RTL configuration.  The stimulus (.mem
files) are read at runtime and are not RTL — changing them never triggers
a recompile.

The script therefore:
  1. Patches the TB once for a given (N, tile, num_sets) config.
  2. Runs `make` for every test in that config back-to-back.
     - First test  : make sees the TB timestamp changed → compiles TB, then sims.
     - Every subsequent test in the same config : only .mem files changed →
       make skips compilation and runs the simulator directly.
  3. Patches the TB to the next config (next tile size) → make recompiles TB.
  4. Repeats.

Result: one TB compilation per (tile × group), zero redundant compiles.
No changes to the Makefile are needed.

Execution order
---------------
  for each tile T:
      patch TB(N, T, MATMUL_NUM_SETS)     ← TB file touched, triggers compile
      for each matmul test:
          write .mem files
          make                             ← 1st: compile+sim  |  rest: sim only
      patch TB(N, T, CONV_NUM_SETS)       ← TB file touched, triggers compile
      for each conv test:
          write .mem files
          make                             ← 1st: compile+sim  |  rest: sim only
  restore TB to original

Usage
-----
  python regression.py --matrix-size 16
  python regression.py --matrix-size 16 --group matmul
  python regression.py --matrix-size 16 --group conv
  python regression.py --matrix-size 16 --fast
  python regression.py --matrix-size 64

Exit code: 0 = all pass, 1 = any failure or interrupted.
"""

import argparse
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime

from conv_tests import CONV_NUM_SETS, CONV_TESTS
from matmul_tests import MATMUL_NUM_SETS, MATMUL_TESTS, pow2_tile_sizes

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
STIM_DIR = os.path.join(ROOT, "testbenches", "stimulus")
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
# Testbench patch / restore
# ---------------------------------------------------------------------------


def _find_tb() -> str:
    for root, _, files in os.walk(ROOT):
        for f in files:
            if f == "TB_SystolicMesh.sv":
                return os.path.join(root, f)
    return None


def _patch_tb(
    tb_path: str, matrix_size: int, tile_size: int, num_test_sets: int
) -> str:
    """
    Overwrite the three compile-time localparams in the TB.
    Returns the original file text so the caller can restore it later.
    Touching this file is what tells make to recompile the TB.
    """
    with open(tb_path) as fh:
        original = fh.read()
    patched = original
    patched = re.sub(
        r"(localparam\s+MATRIX_SIZE\s*=\s*)\d+", rf"\g<1>{matrix_size}", patched
    )
    patched = re.sub(
        r"(localparam\s+TILE_SIZE\s*=\s*)\d+", rf"\g<1>{tile_size}", patched
    )
    patched = re.sub(
        r"(localparam\s+int\s+NUM_TEST_SETS\s*=\s*)\d+",
        rf"\g<1>{num_test_sets}",
        patched,
    )
    with open(tb_path, "w") as fh:
        fh.write(patched)
    return original


def _restore_tb(tb_path: str, original: str) -> None:
    with open(tb_path, "w") as fh:
        fh.write(original)


# ---------------------------------------------------------------------------
# Make runner
# ---------------------------------------------------------------------------


def _make() -> tuple:
    """
    Run `make` in the project root.
    Returns (combined stdout+stderr, wall_seconds).
    make decides internally whether to recompile based on file timestamps:
      - TB .sv changed  → compile TB then simulate
      - only .mem changed → simulate directly, no recompile
    """
    t0 = time.time()
    r = subprocess.run(["make"], cwd=ROOT, capture_output=True, text=True)
    out = r.stdout + r.stderr
    if r.returncode != 0:
        out += f"\nBUILDFAIL: make exited {r.returncode}\n"
    return out, time.time() - t0


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


def _parse(raw: str) -> dict:
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


def _run_group(
    tests: list, tile: int, N: int, num_sets: int, tb_path: str, group_label: str
) -> list:
    """
    Patch the TB for (N, tile, num_sets) — this is the only moment the TB
    file is written, so make will compile exactly once for this config.
    Then call make for each test; subsequent calls only touch .mem files
    so make skips compilation and runs the simulator directly.
    """
    results = []

    print(hdr(f"\n  ╔══  {group_label}  TILE={tile}  N={N}  SETS={num_sets}"))

    # ── Single TB patch for the whole group ──────────────────────────────
    original = _patch_tb(tb_path, N, tile, num_sets)

    try:
        for idx, test in enumerate(tests):
            print(f"  ║  [{idx+1}/{len(tests)}] {test['name']}", end="  ", flush=True)

            # Write stimulus — only .mem files change, TB is untouched
            os.makedirs(STIM_DIR, exist_ok=True)
            test["gen_fn"](STIM_DIR, N)

            # make: compiles TB on first call (TB timestamp changed),
            #       skips compile on all subsequent calls (no RTL change)
            raw, wall = _make()

            parsed = _parse(raw)
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
                **parsed,
            )
            results.append(r)
            _print_result(r)

    finally:
        # Restore after the whole group so we don't touch the TB mid-group
        _restore_tb(tb_path, original)
        print(f"  ╚══  TB restored")

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _report(results: list, N: int, fast: bool, group: str) -> str:
    """Plain-text readiness report alongside the per-run logs."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    passed = sum(1 for r in results if r["status"] == "PASS")
    total = len(results)
    verdict = ("READY" if passed == total
               else f"NOT READY ({total - passed} failure{'s' if total-passed>1 else ''})")

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
    L.append("NOTES")
    L.append("-" * W)
    L.append("* TB patched once per (tile x group); make handles incremental compilation")
    L.append("* Tolerance: RELATIVE <= 1%")
    L.append("* Reference: float64 matmul cast to float32")
    L.append("* Data type: IEEE-754 Float32")
    L.append("* Tool: Verilator  |  Clock: 10 ns")
    L.append("=" * W)

    path = os.path.join(RESULTS_DIR, "readiness_report.log")
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
    args = parser.parse_args()

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

    if run_conv:
        K = int(math.isqrt(N))
        if K * K != N:
            print(
                err(
                    f"[ERROR] Conv tests require N = K²  (perfect square). "
                    f"N={N} is not.  Try N=16, 64, 256 — or use --group matmul."
                )
            )
            sys.exit(1)

    # ── Tile list ─────────────────────────────────────────────────────────
    all_tiles = pow2_tile_sizes(N)
    tiles = [all_tiles[len(all_tiles) // 2]] if args.fast else all_tiles

    # For conv, find the maximum NUM_TEST_SETS needed across all conv tests.
    # We compute this once here so it's stable for the whole run.
    conv_num_sets = CONV_NUM_SETS
    if run_conv:
        os.makedirs(STIM_DIR, exist_ok=True)
        for t in run_conv:
            n = t["gen_fn"](STIM_DIR, N)
            if n > conv_num_sets:
                conv_num_sets = n

    # ── Compile count — for display only ─────────────────────────────────
    # One TB patch (and therefore one compile) per (tile × group).
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

    tb_path = _find_tb()
    if not tb_path:
        print(err("[ERROR] TB_SystolicMesh.sv not found under project root"))
        sys.exit(1)

    with open(tb_path) as fh:
        _original_tb = fh.read()

    all_results: list = []

    # ── Main loop ─────────────────────────────────────────────────────────
    try:
        for tile in tiles:
            if run_mm:
                r = _run_group(run_mm, tile, N, MATMUL_NUM_SETS, tb_path, "MATMUL")
                all_results.extend(r)

            if run_conv:
                r = _run_group(run_conv, tile, N, conv_num_sets, tb_path, "CONV")
                all_results.extend(r)

    except KeyboardInterrupt:
        print(wrn("\n  [Interrupted]"))
    finally:
        _restore_tb(tb_path, _original_tb)
        print(f"\n  {ok('✓')} TB_SystolicMesh.sv restored to original")

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

    verdict = (
        ok("✅  All tests passed — Sanity Clean")
        if passed == total
        else err("❌  Failures detected — see logs")
    )
    print(hdr(f"\n  Verdict : {verdict}"))

    rpt = _report(all_results, N, args.fast, args.group)
    print(f"  Report  : {rpt}")
    print(hdr(f"{'═'*64}\n"))

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
