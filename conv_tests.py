#!/usr/bin/env python3
"""
conv_tests.py — Convolution Test Generators for SystolicMesh
=============================================================
All generators accept (stim_dir, N) and always write exactly CONV_NUM_SETS=3
stimulus sets, padding with repeats of the last real set when needed.
This fixed set count lets regression.py compile once per tile size and
run every conv test through the same binary without recompiling.

The hardware sees only A @ B.  Convolution is purely a data-layout concern
handled here.  This file also serves as the RTL im2col pre-processor spec.

Standalone usage
----------------
  python conv_tests.py --list
  python conv_tests.py --list --matrix-size 64
  python conv_tests.py --gen conv_random          --matrix-size 16
  python conv_tests.py --gen conv_adv_stride      --matrix-size 16 --debug
  python conv_tests.py --gen conv_adv_multi_in    --matrix-size 64

Supported matrix sizes: perfect-square powers of 2
  N=16  → K=4   N=64  → K=8   N=256 → K=16

────────────────────────────────────────────────────────────────────────────
DATA LAYOUT REFERENCE  (for RTL im2col pre-processor)
────────────────────────────────────────────────────────────────────────────

Basic — single-channel, non-overlapping stride (S = K)
───────────────────────────────────────────────────────
  P = N = K²    matrix dimension = kernel elements = num_patches
  Image: K*K × K*K pixels  (exactly P non-overlapping patches)

  A[P×P]  row i  = patch_i.flatten()
  B[P×P]  col *  = kernel.flatten() repeated across all P columns
  C[P×P]  C[i,*] = dot(patch_i, kernel)  — all columns identical

  SRAM readback:  conv_out[r,c] = SRAM[ (r*out_W + c) * P ]

Advanced A — Overlapping stride  (S < K)
─────────────────────────────────────────
  num_patches > P  →  ceil(num_patches/P) hardware passes (NUM_TEST_SETS).
  Each pass: A with real patch rows + zero padding, same B.
  Readback: patch i → batch = i//P, row = i%P, SRAM addr = row*P.

Advanced B — Multi-output-channel  (P distinct filters, single pass)
─────────────────────────────────────────────────────────────────────
  A[P×P]  row i  = patch_i.flatten()
  B[P×P]  col j  = filter_j.flatten()   (P different filters)
  C[P×P]  C[i,j] = dot(patch_i, filter_j)

  SRAM readback:
    All P responses at position i : SRAM[ i*P .. i*P+P-1 ]
    Spatial map for filter j      : SRAM[ 0*P+j, 1*P+j, ..., (P-1)*P+j ]

Advanced C — Multi-input-channel  (C_in=4, N_out=P//2 filters)
────────────────────────────────────────────────────────────────
  P = C_in × K²   (patch vector spans all input channels)
  A[P×P]  row i  = patch_i flattened across all C_in channels
  B[P×P]  col j  = filter_j flattened across all C_in channels
  C[P×P]  C[i,j] = true multi-channel dot product
────────────────────────────────────────────────────────────────────────────
"""

import argparse
import math
import os
import struct

import numpy as np

# SIENNA_SEED shifts every generator seed; unset reproduces the fixed stimulus.
_SEED_OFFSET = int(os.environ.get("SIENNA_SEED", "0"))


def _seed(s: int) -> None:
    np.random.seed(s + _SEED_OFFSET)

from matmul_tests import write_mem, pow2_tile_sizes, _ref_matmul

# Fixed number of test sets written by every conv generator.
# Advanced-stride may write more (it needs multiple batches per test);
# in that case CONV_NUM_SETS is applied as a floor and the TB is compiled
# with the actual batch count for that test — see regression.py.
CONV_NUM_SETS = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f2h(f: float) -> str:
    return "".join(f"{b:02x}" for b in struct.pack(">f", float(f)))


def _write_set(A: np.ndarray, B: np.ndarray,
               stim_dir: str, suffix: str = "") -> np.ndarray:
    C = _ref_matmul(A, B)
    write_mem(os.path.join(stim_dir, f"matrixA{suffix}.mem"), A)
    write_mem(os.path.join(stim_dir, f"matrixB{suffix}.mem"), B)
    write_mem(os.path.join(stim_dir, f"matrixC{suffix}.mem"), C)
    return C


def _pad_to(sets: list, target: int) -> list:
    """Pad list of (A,B) pairs to `target` by repeating the last entry."""
    while len(sets) < target:
        sets.append(sets[-1])
    return sets[:target]


def _write_all(sets: list, stim_dir: str) -> int:
    for i, (A, B) in enumerate(sets):
        _write_set(A, B, stim_dir, f"_{i}")
    return len(sets)


def _kernel_size(N: int) -> int:
    """K = sqrt(N).  Raises ValueError if N is not a perfect square."""
    K = int(math.isqrt(N))
    if K * K != N:
        raise ValueError(
            f"N={N} is not a perfect square. "
            f"Conv tests require N=K²  (e.g. 16→K=4, 64→K=8, 256→K=16)."
        )
    return K


def _im2col_patches(image: np.ndarray, K: int, stride: int):
    """Extract K×K patches from a 2-D or 3-D (H×W×C) image."""
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    H, W, _ = image.shape
    patches, positions = [], []
    for r in range(0, H - K + 1, stride):
        for c in range(0, W - K + 1, stride):
            patches.append(image[r:r+K, c:c+K, :].flatten())
            positions.append((r, c))
    return patches, positions


def _debug_table(tag: str, C: np.ndarray, patches: list, P: int, out_side: int) -> None:
    print(f"\n  [DEBUG] {tag}  —  {len(patches)} patches, P={P}, map {out_side}×{out_side}")
    print(f"  {'patch':>6}  {'SRAM addr':>10}  {'conv_out':>14}  hex")
    print(f"  {'─'*52}")
    for i in range(min(len(patches), P)):
        val = float(C[i, 0])
        print(f"  {i:6d}  {i*P:10d}  {val:+14.6f}  {_f2h(val)}")


# ---------------------------------------------------------------------------
# Basic generators
# ---------------------------------------------------------------------------

def _basic_pair(img_size: int, K: int, seed: int) -> tuple:
    """Build one (A, B) im2col pair for a non-overlapping-stride test."""
    P           = K * K
    _seed(seed)
    image       = np.random.uniform(-1, 1, (img_size, img_size)).astype(np.float32)
    kernel      = np.random.uniform(-1, 1, (K, K)).astype(np.float32)
    patches, _  = _im2col_patches(image, K, K)
    num_patches = len(patches)
    A           = np.zeros((P, P), dtype=np.float32)
    A[:num_patches] = np.stack(patches)
    B           = np.tile(kernel.flatten()[:, np.newaxis], (1, P)).astype(np.float32)
    return A, B


def gen_conv_random(stim_dir: str, N: int, debug: bool = False) -> int:
    """K*K × K*K image, non-overlapping stride → P patches, 100% utilisation."""
    K   = _kernel_size(N)
    img = K * K
    sets = [_basic_pair(img, K, seed=700 + i) for i in range(3)]
    n = _write_all(_pad_to(sets, CONV_NUM_SETS), stim_dir)
    if debug:
        C = _ref_matmul(*sets[0])
        _debug_table("conv_random set_0", C, list(range(N)), N, K)
    return n


def gen_conv_zero_kernel(stim_dir: str, N: int, debug: bool = False) -> int:
    """Zero kernel → all outputs must be exactly zero."""
    K, P = _kernel_size(N), N
    img  = K * K
    sets = []
    for i in range(3):
        _seed(800 + i)
        image   = np.random.uniform(-1, 1, (img, img)).astype(np.float32)
        patches, _ = _im2col_patches(image, K, K)
        A = np.stack(patches).astype(np.float32)
        B = np.zeros((P, P), dtype=np.float32)
        sets.append((A, B))
    return _write_all(_pad_to(sets, CONV_NUM_SETS), stim_dir)


def gen_conv_ones_kernel(stim_dir: str, N: int, debug: bool = False) -> int:
    """All-ones kernel → output = sum of each patch element."""
    K, P = _kernel_size(N), N
    img  = K * K
    sets = []
    for i in range(3):
        _seed(900 + i)
        image   = np.random.uniform(-1, 1, (img, img)).astype(np.float32)
        patches, _ = _im2col_patches(image, K, K)
        A = np.stack(patches).astype(np.float32)
        B = np.tile(np.ones(P, dtype=np.float32)[:, np.newaxis], (1, P))
        sets.append((A, B))
    return _write_all(_pad_to(sets, CONV_NUM_SETS), stim_dir)


def gen_conv_impulse_kernel(stim_dir: str, N: int, debug: bool = False) -> int:
    """Center-impulse kernel → output = center pixel of each patch."""
    K, P   = _kernel_size(N), N
    img    = K * K
    center = (K // 2) * K + (K // 2)
    k_vec  = np.zeros(P, dtype=np.float32)
    k_vec[center] = 1.0
    sets = []
    for i in range(3):
        _seed(1000 + i)
        image   = np.random.uniform(-1, 1, (img, img)).astype(np.float32)
        patches, _ = _im2col_patches(image, K, K)
        A = np.stack(patches).astype(np.float32)
        B = np.tile(k_vec[:, np.newaxis], (1, P))
        sets.append((A, B))
    return _write_all(_pad_to(sets, CONV_NUM_SETS), stim_dir)


def gen_conv_padded(stim_dir: str, N: int, debug: bool = False) -> int:
    """Half-size image → ~25% utilisation (zero-padding in A)."""
    K   = _kernel_size(N)
    img = K * (K // 2)          # half the tiles per side
    sets = [_basic_pair(img, K, seed=1100 + i) for i in range(3)]
    return _write_all(_pad_to(sets, CONV_NUM_SETS), stim_dir)


def gen_conv_large_kernel(stim_dir: str, N: int, debug: bool = False) -> int:
    """Kernel values ±10 — stresses output range."""
    K, P = _kernel_size(N), N
    img  = K * K
    sets = []
    for i in range(3):
        _seed(1200 + i)
        image   = np.random.uniform(-1, 1, (img, img)).astype(np.float32)
        patches, _ = _im2col_patches(image, K, K)
        k_vec   = np.random.uniform(-10, 10, P).astype(np.float32)
        A = np.stack(patches).astype(np.float32)
        B = np.tile(k_vec[:, np.newaxis], (1, P))
        sets.append((A, B))
    return _write_all(_pad_to(sets, CONV_NUM_SETS), stim_dir)


# ---------------------------------------------------------------------------
# Advanced generator A — Overlapping stride
# ---------------------------------------------------------------------------

def gen_conv_adv_stride(stim_dir: str, N: int,
                        seed: int = 42, debug: bool = False) -> int:
    """
    Overlapping stride (S = K//2).  num_patches > P → multiple batches.

    Each batch is one NUM_TEST_SET.  The TB is compiled with the actual
    batch count for this test (regression.py handles this — see note in
    ADVANCED_CONV_SETS in regression.py).

    B is identical across all batches (same kernel).

    Readback:  patch i → batch = i//P,  row = i%P,  SRAM addr = row*P
    """
    K        = _kernel_size(N)
    P        = N
    S        = max(1, K // 2)
    img_out  = K // 2 + 1                      # output map side
    img_size = (img_out - 1) * S + K           # smallest image giving img_out patches/side

    _seed(seed)
    image   = np.random.uniform(-1, 1, (img_size, img_size)).astype(np.float32)
    kernel  = np.random.uniform(-1, 1, (K, K)).astype(np.float32)
    k_vec   = kernel.flatten()
    B       = np.tile(k_vec[:, np.newaxis], (1, P)).astype(np.float32)

    patches, _ = _im2col_patches(image, K, S)
    num_patches = len(patches)
    n_batches   = math.ceil(num_patches / P)

    # Write each batch as a separate set
    sets = []
    for b in range(n_batches):
        start = b * P
        end   = min(start + P, num_patches)
        batch = patches[start:end]
        A     = np.zeros((P, P), dtype=np.float32)
        A[:len(batch)] = np.stack(batch)
        sets.append((A, B))

    # Pad to at least CONV_NUM_SETS so the TB always gets enough sets
    n = _write_all(_pad_to(sets, CONV_NUM_SETS), stim_dir)

    if debug:
        all_out = np.array([np.dot(p, k_vec) for p in patches], dtype=np.float32)
        out_map = all_out.reshape(img_out, img_out)
        print(f"\n  [DEBUG] adv_stride  K={K}, S={S}, "
              f"img={img_size}×{img_size}, "
              f"patches={num_patches}, batches={n_batches}")
        print(f"  Expected {img_out}×{img_out} output map:\n{np.round(out_map, 4)}")
        for i, val in enumerate(all_out):
            b_i, r_i = divmod(i, P)
            print(f"    patch {i:3d}  batch {b_i}  row {r_i:3d}  "
                  f"SRAM[{r_i*P}]  {val:+.4f}")
    return n


# ---------------------------------------------------------------------------
# Advanced generator B — Multi-output-channel
# ---------------------------------------------------------------------------

def gen_conv_adv_multi_out(stim_dir: str, N: int,
                           seed: int = 42, debug: bool = False) -> int:
    """
    P distinct filters in a single pass.

    A[P×P] col  i  = patch_i.flatten()
    B[P×P] col  j  = filter_j.flatten()
    C[P×P] C[i,j] = dot(patch_i, filter_j)
    """
    K   = _kernel_size(N)
    P   = N
    img = K * K

    sets = []
    for i in range(3):
        _seed(42 + i)
        image   = np.random.uniform(-1, 1, (img, img)).astype(np.float32)
        filters = np.random.uniform(-1, 1, (P, P)).astype(np.float32)
        patches, _ = _im2col_patches(image, K, K)
        A = np.stack(patches).astype(np.float32)
        B = filters.T.astype(np.float32)
        sets.append((A, B))

    n = _write_all(_pad_to(sets, CONV_NUM_SETS), stim_dir)

    if debug:
        C = _ref_matmul(*sets[0])
        print(f"\n  [DEBUG] adv_multi_out  K={K}, P filters={P}")
        for f in range(min(2, P)):
            print(f"  Filter {f}:\n{np.round(C[:, f].reshape(K, K), 4)}")
    return n


# ---------------------------------------------------------------------------
# Advanced generator C — Multi-input-channel
# ---------------------------------------------------------------------------

def gen_conv_adv_multi_in(stim_dir: str, N: int,
                          seed: int = 42, debug: bool = False) -> int:
    """
    C_in=4 input channels folded into the patch vector.
    K derived from N:  C_in × K² = N  →  K = sqrt(N // C_in).
    N_out = P // 2 active output filters (remaining B columns are zero).
    """
    C_IN  = 4
    P     = N
    K_sq  = P // C_IN
    K     = int(math.isqrt(K_sq))
    if K * K != K_sq:
        raise ValueError(
            f"N={N} with C_in={C_IN} gives K²={K_sq}, not a perfect square. "
            f"Use N=16 (K=2), N=64 (K=4), N=256 (K=8)."
        )
    STRIDE  = K
    IMG     = K * K
    N_OUT   = P // 2

    sets = []
    for i in range(3):
        _seed(seed + i)
        image   = np.random.uniform(-1, 1, (IMG, IMG, C_IN)).astype(np.float32)
        filters = np.random.uniform(-1, 1, (N_OUT, C_IN, K, K)).astype(np.float32)
        patches, _ = _im2col_patches(image, K, STRIDE)
        A = np.stack(patches).astype(np.float32)
        B = np.zeros((P, P), dtype=np.float32)
        for j in range(N_OUT):
            B[:, j] = filters[j].flatten()
        sets.append((A, B))

    n = _write_all(_pad_to(sets, CONV_NUM_SETS), stim_dir)

    if debug:
        C = _ref_matmul(*sets[0])
        print(f"\n  [DEBUG] adv_multi_in  C_in={C_IN}, K={K}, N_out={N_OUT}, P={P}")
        for f in range(min(2, N_OUT)):
            print(f"  Filter {f}:\n{np.round(C[:, f].reshape(K, K), 4)}")
    return n


# ---------------------------------------------------------------------------
# Test catalogue
# ---------------------------------------------------------------------------

CONV_TESTS = [
    # ── Basic ───────────────────────────────────────────────────────────────
    dict(name="conv_random",       description="Random kernel, 100% utilisation (baseline)",        gen_fn=gen_conv_random,        advanced=False),
    dict(name="conv_zero_kernel",  description="Zero kernel → output must be all-zero",             gen_fn=gen_conv_zero_kernel,   advanced=False),
    dict(name="conv_ones_kernel",  description="All-ones kernel → output = patch element sum",      gen_fn=gen_conv_ones_kernel,   advanced=False),
    dict(name="conv_impulse",      description="Impulse kernel → output = center pixel",            gen_fn=gen_conv_impulse_kernel,advanced=False),
    dict(name="conv_padded",       description="Half-size image → ~25% utilisation",                gen_fn=gen_conv_padded,        advanced=False),
    dict(name="conv_large_kern",   description="Kernel values ±10 (output range stress)",           gen_fn=gen_conv_large_kernel,  advanced=False),
    # ── Advanced ────────────────────────────────────────────────────────────
    dict(name="conv_adv_stride",   description="Overlapping stride → multi-batch assembly",         gen_fn=gen_conv_adv_stride,    advanced=True),
    dict(name="conv_adv_multi_out",description="Multi-output-channel: P filters in one pass",      gen_fn=gen_conv_adv_multi_out, advanced=True),
    dict(name="conv_adv_multi_in", description="Multi-input-channel: C_in=4 channels folded in P", gen_fn=gen_conv_adv_multi_in,  advanced=True),
]


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def _list_tests(N: int) -> None:
    K     = int(math.isqrt(N))
    valid = K * K == N
    tiles = ", ".join(str(t) for t in pow2_tile_sizes(N))
    print(f"\n  Matrix size    : {N}×{N}   K={K}{'  ✓' if valid else '  ✗ (must be perfect square)'}")
    print(f"  Tile sweep     : {tiles}")
    print(f"  Sets per test  : {CONV_NUM_SETS} (fixed — short tests padded)")
    print(f"\n  {'Name':<26}  {'Adv':<5}  Description")
    print("  " + "─" * 75)
    for t in CONV_TESTS:
        adv = "⚡" if t["advanced"] else " "
        print(f"  {t['name']:<26}  {adv:<5}  {t['description']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Convolution stimulus generator")
    parser.add_argument("--gen",         type=str, default=None)
    parser.add_argument("--matrix-size", type=int, default=16,
                        help="N = K²  (perfect-square power of 2, default: 16)")
    parser.add_argument("--dir",         type=str, default="testbenches/stimulus")
    parser.add_argument("--debug",       action="store_true")
    parser.add_argument("--list",        action="store_true")
    args = parser.parse_args()

    N = args.matrix_size
    if N < 4 or (N & (N - 1)) != 0:
        print(f"[ERROR] --matrix-size must be a power of 2 ≥ 4  (got {N})")
        raise SystemExit(1)

    if args.list or args.gen is None:
        _list_tests(N)
        return

    match = [t for t in CONV_TESTS if t["name"] == args.gen]
    if not match:
        print(f"[ERROR] Unknown test '{args.gen}'.  Use --list to see valid names.")
        raise SystemExit(1)

    test = match[0]
    os.makedirs(args.dir, exist_ok=True)

    import inspect
    sig    = inspect.signature(test["gen_fn"])
    kwargs = dict(stim_dir=args.dir, N=N)
    if "debug" in sig.parameters:
        kwargs["debug"] = args.debug

    n_sets = test["gen_fn"](**kwargs)

    print(f"\nGenerated {n_sets} set(s) for '{test['name']}' → {args.dir}/")
    print(f"Testbench parameters:")
    print(f"  MATRIX_SIZE   = {N}")
    print(f"  NUM_TEST_SETS = {n_sets}")
    print(f"  TILE_SIZE     = one of {pow2_tile_sizes(N)}")


if __name__ == "__main__":
    main()
