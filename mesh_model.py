#!/usr/bin/env python3
"""Bit-exact model of SystolicMesh's arithmetic in the build's format: each PE sums product n of a set into slot n mod U
(products counted across a set's passes, the first U adding to +0), then the reducer's pairwise tree adds the U partials of every
depth slice with the bias as its last input. Operand order as the RTL: mul(A=a, B=b), PE add(A=slot, B=product), tree add(A=left, B=right)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ArithmeticLibrary", "Common", "models"))
import fpu  # noqa: E402

ADD_LAT = 5  # sienna_fmt_pkg::add_lat, every supported format


def matmul(f, passes, N, T, collapse_k=1, bias=None):
    AK = N if collapse_k else T  # depth of each array's product
    RP = 1 if collapse_k else N // T  # depth slices, one array each
    U = min(AK, ADD_LAT + 1)
    acc = np.zeros((RP, U, N, N), dtype=np.int64)
    g = 0
    for A, B in passes:
        A = np.asarray(A, dtype=np.int64)
        B = np.asarray(B, dtype=np.int64)
        for kk in range(AK):
            u = g % U
            for rp in range(RP):
                k = rp * AK + kk
                a = np.broadcast_to(A[:, k][:, None], (N, N))
                b = np.broadcast_to(B[k, :][None, :], (N, N))
                acc[rp, u] = fpu.add(f, acc[rp, u], fpu.mul(f, a, b)[0])[0]
            g += 1
    return _reduce(f, [acc[rp, u] for rp in range(RP) for u in range(U)], N, bias)


def _reduce(f, parts, N, bias):
    """The reducer: a pairwise tree over the partials with the bias as its last input; an odd entry waits a level."""
    bias_row = np.zeros(N, dtype=np.int64) if bias is None else np.asarray(bias, dtype=np.int64)
    level = list(parts) + [np.broadcast_to(bias_row[None, :], (N, N))]
    while len(level) > 1:
        nxt = [fpu.add(f, level[2 * m], level[2 * m + 1])[0] for m in range(len(level) // 2)]
        if len(level) % 2:
            nxt.append(level[-1])  # an odd entry out waits a level, as the RTL's PASS delay
        level = nxt
    return np.asarray(level[0], dtype=np.int64)


def matmul_packed(f, A, B, N, shift, bias=None):
    """A packed collapse-k set: PE (i, j) adds only its block's products k (b = N >> shift), the r-th into slot r mod U, unwritten slots +0."""
    A = np.asarray(A, dtype=np.int64)
    B = np.asarray(B, dtype=np.int64)
    b = N >> shift
    U = min(N, ADD_LAT + 1)
    acc = np.zeros((U, N, N), dtype=np.int64)
    for c in range(N // b):
        cols = slice(c * b, (c + 1) * b)
        for r in range(b):
            k = c * b + r
            a = np.broadcast_to(A[:, k][:, None], (N, b))
            w = np.broadcast_to(B[k, cols][None, :], (N, b))
            acc[r % U][:, cols] = fpu.add(f, acc[r % U][:, cols], fpu.mul(f, a, w)[0])[0]
    return _reduce(f, [acc[u] for u in range(U)], N, bias)


def matmul_int(passes, N, bias=None):
    """The int8 mesh: int8 x int8 products over a set's passes plus the bias, summed in wrapping int32; int8 values in, int64 out."""
    acc = np.zeros((N, N), dtype=np.int64)
    for A, B in passes:
        A = np.asarray(A, dtype=np.int64)
        B = np.asarray(B, dtype=np.int64)
        if A.shape != (N, N) or B.shape != (N, N):
            raise ValueError(f"mesh sets are N x N; got {A.shape} and {B.shape}")
        if min(A.min(), B.min()) < -128 or max(A.max(), B.max()) > 127:
            raise ValueError("int8 operands out of range: pass values in [-128, 127], not bit patterns")
        acc += A @ B
    if bias is not None:
        b = np.asarray(bias, dtype=np.int64)
        if b.shape != (N,) or b.min() < -2**31 or b.max() >= 2**31:
            raise ValueError(f"bias must be {N} int32 values")
        acc += b[None, :]
    return ((acc + 2**31) % 2**32) - 2**31
