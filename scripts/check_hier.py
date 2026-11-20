#!/usr/bin/env python3
"""Hierarchy check: at each define and mesh mode the AriL units elaborated under SystolicMesh are exactly that precision's; args: --files DESIGN_FILES."""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # the SystolicMesh repository
UNITS = {  # every arithmetic unit of ArithmeticLibrary: per-format units, the generic integer units and the int8-only units
    "fp32Multiplier", "fp32Adder", "fp32Divider", "bf16Multiplier", "bf16Adder", "bf16WideMultiplier", "fp32ToBf16",
    "int8Multiplier", "int16Multiplier", "int24Multiplier", "int32Adder", "intMultiplier", "intAdder", "int8Requantizer", "int8QuantizedAdder",
}
EXPECTED = {  # the mesh's units per precision: product into the accumulator, the accumulator add, and the narrowing where the result is narrower
    "fp32": {"fp32Multiplier", "fp32Adder"},
    "bf16": {"bf16WideMultiplier", "fp32Adder", "fp32ToBf16"},
    "int8": {"int8Multiplier", "int32Adder"},
}
EXPECTED["default"] = EXPECTED["fp32"]  # no define elaborates fp32


def hier_cells(tree: dict, top: str) -> list:
    """Every instance below the top module as (hierarchical name, origName of the module it instantiates), from Verilator's --json-only
    tree: a MODULE holds its CELLs, inside any named generate block (GENBLOCK), and a CELL's modp names the MODULE it instantiates."""
    modules = {m["addr"]: m for m in tree["modulesp"] if m.get("type") == "MODULE"}

    def cells(node, path):  # the CELLs in a module's body with their generate-block path, not descending into another module
        if isinstance(node, dict):
            t = node.get("type")
            if t == "CELL":
                yield node, path
                return
            if t == "GENBLOCK" and node.get("name"):
                path = f"{path}.{node['name']}"
            for v in node.values():
                if isinstance(v, (list, dict)):
                    yield from cells(v, path)
        elif isinstance(node, list):
            for v in node:
                yield from cells(v, path)

    root = next((m for m in modules.values() if m.get("origName") == top), None)
    if root is None:
        raise RuntimeError(f"no {top} module in the JSON")
    out, stack = [], [(root, top)]
    while stack:
        mod, path = stack.pop()
        for body in (v for k, v in mod.items() if isinstance(v, (list, dict))):
            for c, p in cells(body, path):
                child = modules.get(c.get("modp"))
                if child is None:
                    continue
                h = f"{p}.{c['origName']}"
                out.append((h, child.get("origName", child["name"])))
                stack.append((child, h))
    return out


def units_by_place(tree: dict) -> tuple:
    """(origNames of the modules instantiated under the mesh instance dut, origNames of every other one)."""
    under, other = set(), set()
    for h, u in hier_cells(tree, "TB_SystolicMesh"):
        (under if h.startswith("TB_SystolicMesh.dut.") else other).add(u)
    if not under:
        raise RuntimeError("no cell under TB_SystolicMesh.dut in the JSON")
    return under, other


def elaborate(fmt: str, ck: int, files: list, verilator: str, out: str) -> dict:
    d = [] if fmt == "default" else [f"+define+FMT_{fmt.upper()}"]
    x = os.path.join(out, f"hier_{fmt}_ck{ck}.json")
    # the bench is the top: Verilator cannot elaborate a top with interface ports, and the bench instantiates only the mesh's units
    cmd = [verilator, "--json-only", "--json-only-output", x, "--json-only-meta-output", x[:-5] + ".meta.json", "--timing", "-Wno-fatal", "-Wno-lint", "-Wno-style", *d, "--top-module", "TB_SystolicMesh",
           "-GMATRIX_SIZE=8", "-GTILE_SIZE=2", f"-GCOLLAPSE_K={ck}", "--Mdir", os.path.join(out, f"obj_{fmt}_ck{ck}"), *files,
           os.path.join(ROOT, "testbenches", "top", "TB_SystolicMesh.sv")]
    r = subprocess.run(cmd, cwd=os.path.join(ROOT, "src"), capture_output=True, text=True)
    if r.returncode != 0:
        errs = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.startswith("%Error")]
        raise RuntimeError(f"{fmt} ck{ck}: verilator exit {r.returncode}: {' | '.join(errs[:3]) or (r.stdout + r.stderr)[-400:]}")
    with open(x) as fh:
        return json.load(fh)


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
        trees = dict(zip(jobs, ex.map(lambda j: elaborate(*j, a.files, a.verilator, a.out), jobs)))
    places = {j: units_by_place(x) for j, x in trees.items()}
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
