#!/usr/bin/env python3
"""Hierarchy check: at each define and mesh mode the AriL units elaborated under SystolicMesh are exactly that precision's; args: --files DESIGN_FILES."""
import argparse
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
UNITS = {  # every arithmetic unit of ArithmeticLibrary: per-format units, generic units and the int8-only units
    "fp32Multiplier", "fp32Adder", "fp32Divider", "bf16Multiplier", "bf16Adder", "bf16WideMultiplier", "fp32ToBf16",
    "int8Multiplier", "int16Multiplier", "int24Multiplier", "int32Adder", "fpMultiplier", "fpAdder", "fpMulWiden", "fpNarrow",
    "intMultiplier", "intAdder", "fxMac", "int8Requantizer", "int8QuantizedAdder",
}
EXPECTED = {  # the mesh's units per precision: product into the accumulator, the accumulator add, and the narrowing where the result is narrower
    "fp32": {"fp32Multiplier", "fp32Adder"},
    "bf16": {"bf16WideMultiplier", "fp32Adder", "fp32ToBf16"},
    "int8": {"int8Multiplier", "int32Adder"},
}
EXPECTED["default"] = EXPECTED["fp32"]  # no define elaborates fp32


def units_by_place(xml: str) -> tuple:
    """(origNames of the cells under the mesh instance dut, origNames of every other cell) from the XML's cell hierarchy."""
    orig = {m.group(1): m.group(2) for m in re.finditer(r'<module\b[^>]*\bname="([^"]+)"[^>]*\borigName="([^"]+)"', xml)}
    under, other = set(), set()
    for m in re.finditer(r'<cell\b[^>]*\bsubmodname="([^"]+)"[^>]*\bhier="([^"]+)"', xml):
        (under if m.group(2).startswith("TB_SystolicMesh.dut.") else other).add(orig.get(m.group(1), m.group(1)))
    if not under:
        raise RuntimeError("no cell under TB_SystolicMesh.dut in the XML")
    return under, other


def elaborate(fmt: str, ck: int, files: list, verilator: str, out: str) -> str:
    d = [] if fmt == "default" else [f"+define+FMT_{fmt.upper()}"]
    x = os.path.join(out, f"hier_{fmt}_ck{ck}.xml")
    # the bench is the top: Verilator cannot elaborate a top with interface ports, and the bench instantiates only the mesh's units
    cmd = [verilator, "--xml-only", "--xml-output", x, "--timing", "-Wno-fatal", "-Wno-lint", "-Wno-style", *d, "--top-module", "TB_SystolicMesh",
           "-GMATRIX_SIZE=8", "-GTILE_SIZE=2", f"-GCOLLAPSE_K={ck}", "--Mdir", os.path.join(out, f"obj_{fmt}_ck{ck}"), *files,
           os.path.join(ROOT, "testbenches", "TB_SystolicMesh.sv")]
    r = subprocess.run(cmd, cwd=os.path.join(ROOT, "src"), capture_output=True, text=True)
    if r.returncode != 0:
        errs = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.startswith("%Error")]
        raise RuntimeError(f"{fmt} ck{ck}: verilator exit {r.returncode}: {' | '.join(errs[:3]) or (r.stdout + r.stderr)[-400:]}")
    return open(x).read()


def judge(fmt: str, units: set) -> tuple:
    got = units & UNITS
    return got == EXPECTED[fmt], got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--verilator", default="verilator")
    ap.add_argument("--out", default=os.path.join(ROOT, "Verilator_hier"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = [(f, ck) for f in ("fp32", "bf16", "int8", "default") for ck in (1, 0)]
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as ex:  # one elaboration per core
        xmls = dict(zip(jobs, ex.map(lambda j: elaborate(*j, a.files, a.verilator, a.out), jobs)))
    places = {j: units_by_place(x) for j, x in xmls.items()}
    found = {j: p[0] for j, p in places.items()}
    outside = {j: p[1] & UNITS for j, p in places.items()}
    bad = 0
    for (f, ck), mods in found.items():
        ok, got = judge(f, mods)
        ok = ok and not outside[(f, ck)]  # a unit beside the mesh would be the bench's, and the bench has none
        bad += not ok
        print(f"check-hier {f} ck{ck}: {'OK' if ok else 'FAIL'}: units {', '.join(sorted(got)) or 'none'}"
              + ("" if ok else f" (want {', '.join(sorted(EXPECTED[f]))})"))
    ok, _ = judge("fp32", found[("bf16", 1)])  # control: bf16's hierarchy is not fp32's
    print(f"check-hier control (bf16 hierarchy judged as fp32): {'FAIL, as it must' if not ok else 'PASSED: the check cannot fail'}")
    bad += ok
    print(f"check-hier: {len(found)} elaborations, {bad} failing")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
