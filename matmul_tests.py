#!/usr/bin/env python3
"""
matmul_tests.py — Matrix Multiplication Test Generators for SystolicMesh
=========================================================================
All generators accept (stim_dir, N) and always write exactly NUM_SETS=5
stimulus sets, padding with repeats of the last real set when needed.
This fixed set count lets regression.py compile once per tile size and
run every matmul test through the same binary without recompiling.

Standalone usage
----------------
  python matmul_tests.py --list
  python matmul_tests.py --list --matrix-size 64
  python matmul_tests.py --gen mm_random   --matrix-size 16
  python matmul_tests.py --gen mm_diagonal --matrix-size 64 --dir /tmp/stim

Supported matrix sizes: any power of 2  (8, 16, 32, 64, ...)

Test catalogue
--------------
  mm_random        Random float32 matrices (baseline)
  mm_identity      A @ I = A  (no data corruption)
  mm_zero_b        A @ 0 = 0  (zero propagation)
  mm_ones          [1s] @ [1s] = N×[1s]  (known integer result)
  mm_diagonal      Diagonal × diagonal  (sparse data flow)
  mm_large_values  Values ±100  (accumulator range stress)
  mm_small_values  Values ±1e-6  (underflow / denormal stress)
  mm_alternating   ±1 checkerboard  (sign alternation in accumulation)
"""

import argparse
import os
import struct

import numpy as np

# SIENNA_SEED shifts every generator seed; unset reproduces the fixed stimulus.
_SEED_OFFSET = int(os.environ.get("SIENNA_SEED", "0"))


def _seed(s: int) -> None:
    np.random.seed(s + _SEED_OFFSET)

# Fixed number of test sets written by every matmul generator.
# regression.py compiles the TB with NUM_TEST_SETS=MATMUL_NUM_SETS once per
# tile and re-uses the binary for all matmul tests.
MATMUL_NUM_SETS = 5


# ---------------------------------------------------------------------------
# Shared helpers  (also imported by regression.py and conv_tests.py)
# ---------------------------------------------------------------------------

def _f2h(f: float) -> str:
    """Pack a float32 value as an 8-character big-endian hex string."""
    return "".join(f"{b:02x}" for b in struct.pack(">f", float(f)))


def write_mem(path: str, data) -> None:
    """Write a flat sequence of float32 values to a .mem file (one hex word per line)."""
    with open(path, "w") as fh:
        for v in np.asarray(data).flatten():
            fh.write(_f2h(v) + "\n")


def pow2_tile_sizes(N: int) -> list:
    """Return all power-of-2 divisors of N in ascending order, e.g. N=16 → [2,4,8,16]."""
    return [2**i for i in range(1, N.bit_length()) if N % (2**i) == 0]


def _ref_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Float64 reference matmul cast back to float32."""
    return (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32)


def _write_set(A: np.ndarray, B: np.ndarray,
               stim_dir: str, suffix: str = "") -> None:
    """Compute C = A @ B and write all three .mem files for one set."""
    C = _ref_matmul(A, B)
    write_mem(os.path.join(stim_dir, f"matrixA{suffix}.mem"), A)
    write_mem(os.path.join(stim_dir, f"matrixB{suffix}.mem"), B)
    write_mem(os.path.join(stim_dir, f"matrixC{suffix}.mem"), C)


def _pad_to(sets: list, target: int) -> list:
    """
    Pad a list of (A, B) pairs to exactly `target` entries with fresh random pairs.

    Padding by repeating the last pair (the previous behaviour) cannot detect a stale
    result: a mesh that replays set i-1 produces byte-identical output and passes.
    Every set must differ from its neighbour for back-to-back runs to mean anything.
    """
    if not sets:
        raise ValueError("_pad_to needs at least one real set")
    dim = sets[0][0].shape[0]
    while len(sets) < target:
        _seed(9000 + len(sets))
        A = np.random.uniform(-1, 1, (dim, dim)).astype(np.float32)
        B = np.random.uniform(-1, 1, (dim, dim)).astype(np.float32)
        sets.append((A, B))
    return sets[:target]


def _write_all(sets: list, stim_dir: str) -> int:
    """Write a list of (A, B) pairs as indexed .mem files. Returns count written."""
    for i, (A, B) in enumerate(sets):
        _write_set(A, B, stim_dir, f"_{i}")
    return len(sets)


# ---------------------------------------------------------------------------
# Generators
#
# Uniform signature:  gen_*(stim_dir: str, N: int) -> int
#   Always returns MATMUL_NUM_SETS (the fixed compiled-in set count).
# ---------------------------------------------------------------------------

def gen_mm_random(stim_dir: str, N: int) -> int:
    """5 random float32 matrix pairs, values in [-1, 1]."""
    sets = []
    for i in range(5):
        _seed(100 + i)
        A = np.random.uniform(-1, 1, (N, N)).astype(np.float32)
        B = np.random.uniform(-1, 1, (N, N)).astype(np.float32)
        sets.append((A, B))
    return _write_all(_pad_to(sets, MATMUL_NUM_SETS), stim_dir)


def gen_mm_identity(stim_dir: str, N: int) -> int:
    """A @ I = A  — verifies no data corruption through the mesh."""
    sets = []
    for i in range(3):
        _seed(200 + i)
        A = np.random.uniform(-1, 1, (N, N)).astype(np.float32)
        sets.append((A, np.eye(N, dtype=np.float32)))
    return _write_all(_pad_to(sets, MATMUL_NUM_SETS), stim_dir)


def gen_mm_zero_b(stim_dir: str, N: int) -> int:
    """A @ 0 = 0  — verifies zero propagation, no spurious accumulation."""
    sets = []
    for i in range(3):
        _seed(300 + i)
        A = np.random.uniform(-1, 1, (N, N)).astype(np.float32)
        sets.append((A, np.zeros((N, N), dtype=np.float32)))
    return _write_all(_pad_to(sets, MATMUL_NUM_SETS), stim_dir)


def gen_mm_ones(stim_dir: str, N: int) -> int:
    """[1s] @ [1s] = N*[1s]  — integer-like result, easy to hand-verify."""
    A = np.ones((N, N), dtype=np.float32)
    B = np.ones((N, N), dtype=np.float32)
    sets = [(A, B)]
    return _write_all(_pad_to(sets, MATMUL_NUM_SETS), stim_dir)


def gen_mm_diagonal(stim_dir: str, N: int) -> int:
    """Diagonal A @ diagonal B = diag(d_a * d_b)  — tests sparse data flow."""
    _seed(400)
    d_a = np.random.uniform(-2, 2, N).astype(np.float32)
    d_b = np.random.uniform(-2, 2, N).astype(np.float32)
    sets = [(np.diag(d_a), np.diag(d_b))]
    return _write_all(_pad_to(sets, MATMUL_NUM_SETS), stim_dir)


def gen_mm_large_values(stim_dir: str, N: int) -> int:
    """Values near ±100  — stresses accumulator range without overflow."""
    sets = []
    for i in range(3):
        _seed(500 + i)
        A = np.random.uniform(-100, 100, (N, N)).astype(np.float32)
        B = np.random.uniform(-100, 100, (N, N)).astype(np.float32)
        sets.append((A, B))
    return _write_all(_pad_to(sets, MATMUL_NUM_SETS), stim_dir)


def gen_mm_small_values(stim_dir: str, N: int) -> int:
    """Values near ±1e-6  — stresses underflow / denormal handling."""
    sets = []
    for i in range(3):
        _seed(600 + i)
        A = np.random.uniform(-1e-6, 1e-6, (N, N)).astype(np.float32)
        B = np.random.uniform(-1e-6, 1e-6, (N, N)).astype(np.float32)
        sets.append((A, B))
    return _write_all(_pad_to(sets, MATMUL_NUM_SETS), stim_dir)


def gen_mm_alternating(stim_dir: str, N: int) -> int:
    """±1 checkerboard  — tests sign alternation in accumulation."""
    pattern = np.array(
        [1 if (i + j) % 2 == 0 else -1 for i in range(N) for j in range(N)],
        dtype=np.float32,
    ).reshape(N, N)
    sets = [(pattern, pattern)]
    return _write_all(_pad_to(sets, MATMUL_NUM_SETS), stim_dir)


# ---------------------------------------------------------------------------
# Test catalogue
#
# No matrix_size or tile_sizes — derived at runtime from N.
# ---------------------------------------------------------------------------

MATMUL_TESTS = [
    dict(name="mm_random",       description="Random float32 matrices (baseline)",             gen_fn=gen_mm_random),
    dict(name="mm_identity",     description="A @ I = A  (no data corruption)",                gen_fn=gen_mm_identity),
    dict(name="mm_zero_b",       description="A @ 0 = 0  (zero propagation)",                  gen_fn=gen_mm_zero_b),
    dict(name="mm_ones",         description="[1s] @ [1s] = N×[1s]  (known integer result)",   gen_fn=gen_mm_ones),
    dict(name="mm_diagonal",     description="Diagonal × diagonal  (sparse data flow)",         gen_fn=gen_mm_diagonal),
    dict(name="mm_large_values", description="Values ±100  (accumulator range stress)",         gen_fn=gen_mm_large_values),
    dict(name="mm_small_values", description="Values ±1e-6  (underflow / denormal stress)",     gen_fn=gen_mm_small_values),
    dict(name="mm_alternating",  description="±1 checkerboard  (sign alternation in accum.)",  gen_fn=gen_mm_alternating),
]


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def _list_tests(N: int) -> None:
    tiles = ", ".join(str(t) for t in pow2_tile_sizes(N))
    print(f"\n  Matrix size    : {N}×{N}")
    print(f"  Tile sweep     : {tiles}")
    print(f"  Sets per test  : {MATMUL_NUM_SETS} (fixed — short tests zero-padded)")
    print(f"\n  {'Name':<22}  Description")
    print("  " + "─" * 68)
    for t in MATMUL_TESTS:
        print(f"  {t['name']:<22}  {t['description']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="MatMul stimulus generator")
    parser.add_argument("--gen",         type=str, default=None,
                        help="Test name to generate (e.g. mm_random)")
    parser.add_argument("--matrix-size", type=int, default=8,
                        help="Matrix dimension N — must be a power of 2 (default: 8)")
    parser.add_argument("--dir",         type=str, default="testbenches/stimulus",
                        help="Output directory for .mem files")
    parser.add_argument("--list",        action="store_true",
                        help="Print all available test names and exit")
    args = parser.parse_args()

    N = args.matrix_size
    if N < 2 or (N & (N - 1)) != 0:
        print(f"[ERROR] --matrix-size must be a power of 2  (got {N})")
        raise SystemExit(1)

    if args.list or args.gen is None:
        _list_tests(N)
        return

    match = [t for t in MATMUL_TESTS if t["name"] == args.gen]
    if not match:
        print(f"[ERROR] Unknown test '{args.gen}'.  Use --list to see valid names.")
        raise SystemExit(1)

    test = match[0]
    os.makedirs(args.dir, exist_ok=True)
    test["gen_fn"](args.dir, N)

    print(f"\nGenerated {MATMUL_NUM_SETS} set(s) for '{test['name']}' → {args.dir}/")
    print(f"Testbench parameters:")
    print(f"  MATRIX_SIZE   = {N}")
    print(f"  NUM_TEST_SETS = {MATMUL_NUM_SETS}")
    print(f"  TILE_SIZE     = one of {pow2_tile_sizes(N)}")


if __name__ == "__main__":
    main()
