#!/usr/bin/env python3
"""mesh_model.matmul_packed: every block of a packed set equals its job run alone as an unpacked set, bit for bit."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mesh_model as mm  # noqa: E402
from mesh_model import fpu  # noqa: E402


def _bits(x, fmt):
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.int64)
    return u if fmt == "fp32" else u >> (23 - fpu.FORMATS[fmt].m)


def _case(fmt, N, shift, seed, garbage=False):
    rng = np.random.RandomState(seed)
    b = N >> shift
    A = _bits(rng.uniform(-1, 1, (N, N)), fmt)
    B = _bits(rng.uniform(-1, 1, (N, N)) if garbage else np.zeros((N, N)), fmt)
    for c in range(N // b):
        B[c * b:(c + 1) * b, c * b:(c + 1) * b] = _bits(rng.uniform(-1, 1, (b, b)), fmt)
    return A, B, _bits(rng.uniform(-0.5, 0.5, N), fmt), b


def _alone(f, A, B, bias, N, c, b):
    """Block c's job as an unpacked set: its inputs and weights at the top left, zeros elsewhere."""
    Ac = np.zeros((N, N), np.int64)
    Ac[:, :b] = A[:, c * b:(c + 1) * b]
    Wc = np.zeros((N, N), np.int64)
    Wc[:b, :b] = B[c * b:(c + 1) * b, c * b:(c + 1) * b]
    bc = np.zeros(N, np.int64)
    bc[:b] = bias[c * b:(c + 1) * b]
    return mm.matmul(f, [(Ac, Wc)], N, 4, 1, bc)[:, :b]


def test_shift0_is_matmul():
    for fmt in ("fp32", "bf16"):
        f = fpu.FORMATS[fmt]
        for N in (8, 16):
            A, B, bias, _ = _case(fmt, N, 1, 11 * N, garbage=True)
            assert np.array_equal(mm.matmul_packed(f, A, B, N, 0, bias), mm.matmul(f, [(A, B)], N, 4, 1, bias)), (fmt, N)


def test_packed_block_equals_alone():
    for fmt in ("fp32", "bf16"):
        f = fpu.FORMATS[fmt]
        for N in (8, 16, 32):
            for shift in range(1, N.bit_length() - 1):
                for garbage in (False, True):
                    A, B, bias, b = _case(fmt, N, shift, 100 * N + 10 * shift + garbage, garbage)
                    P = mm.matmul_packed(f, A, B, N, shift, bias)
                    for c in range(N // b):
                        got = P[:, c * b:(c + 1) * b]
                        assert np.array_equal(got, _alone(f, A, B, bias, N, c, b)), (fmt, N, shift, garbage, c)


def _float(bits, fmt):
    return (np.asarray(bits, np.int64) << (23 - fpu.FORMATS[fmt].m)).astype(np.uint32).view(np.float32).astype(np.float64)


def test_bf16_k640_within_one_ulp():
    # K = 640 (ad01's first layer): fp32 sums narrowed once are within 1 bf16 ulp of the largest output on 99% of words
    f, N, P = fpu.BF16, 32, 20
    rng = np.random.RandomState(640)
    passes = [(_bits(rng.uniform(-1, 1, (N, N)), "bf16"), _bits(rng.uniform(-1, 1, (N, N)), "bf16")) for _ in range(P)]
    got = _float(mm.matmul(f, passes, N, 4, 1), "bf16")
    exact = np.hstack([_float(A, "bf16") for A, _ in passes]) @ np.vstack([_float(B, "bf16") for _, B in passes])
    ref = _float(fpu.narrow(f, exact.astype(np.float32).view(np.uint32).astype(np.int64)), "bf16")
    ulp = 2.0 ** (np.floor(np.log2(np.abs(ref).max())) - f.m)
    ok = float(np.mean(np.abs(got - ref) <= ulp))
    assert ok >= 0.99, f"{100 * ok:.1f}% of words within 1 ulp ({ulp}) of the largest output, max error {np.abs(got - ref).max()}"


def test_unskipped_mesh_differs():
    # The probe's finding: without the skip, a block starting mid-rotation rounds differently from its job alone.
    f = fpu.FORMATS["fp32"]
    A, B, bias, b = _case("fp32", 16, 2, 7)
    plain = mm.matmul(f, [(A, B)], 16, 4, 1, bias)
    alone = np.hstack([_alone(f, A, B, bias, 16, c, b) for c in range(16 // b)])
    assert int(np.sum(plain != alone)) > 0


MIN_TESTS = 4  # tests this file holds; fewer run means one was lost, which is a failure

if __name__ == "__main__":
    bad = ran = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            ran += 1
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                bad += 1
                print(f"FAIL {name}: {e!r}")
    if ran < MIN_TESTS:
        bad += 1
        print(f"FAIL: {ran} tests ran, fewer than MIN_TESTS {MIN_TESTS}")
    print(f"{'ALL PASS' if bad == 0 else f'{bad} FAILED'}")
    sys.exit(1 if bad else 0)
