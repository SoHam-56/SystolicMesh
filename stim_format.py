#!/usr/bin/env python3
"""The mesh stimulus's number format: operand words in the format and the bit-exact expected result from mesh_model in every
format (fp32 included), for the TB's exact compare. int8 operands are 2 hex digits, its int32 results 8; bias words are 8 in every
format (int32, or the float bias widened exactly to fp32)."""
import os
import struct

import numpy as np

import mesh_model
from mesh_model import fpu

FORMAT, TILE, COLLAPSE_K = "fp32", 4, 1
INT_FORMATS = {"int8": (8, 32)}  # operand bits, result bits (sienna_fmt_pkg::out_w)
BIAS_DIGITS = 8  # sienna_fmt_pkg::acc_w = 32 in every format


def configure(fmt: str, tile: int, collapse_k: int) -> None:
    global FORMAT, TILE, COLLAPSE_K
    if fmt not in fpu.FORMATS and fmt not in INT_FORMATS:
        raise ValueError(f"unknown mesh format {fmt}")
    FORMAT, TILE, COLLAPSE_K = fmt, tile, collapse_k


def is_int() -> bool:
    return FORMAT in INT_FORMATS


def digits() -> int:
    """Hex digits of an operand word."""
    return INT_FORMATS[FORMAT][0] // 4 if is_int() else (fpu.FORMATS[FORMAT].w + 3) // 4


def result_digits() -> int:
    """Hex digits of a result word (out_w): int32 in int8, the format's own in the float formats (narrowed from fp32 sums)."""
    return INT_FORMATS[FORMAT][1] // 4 if is_int() else digits()


def rand(lo, hi, shape) -> np.ndarray:
    """Random operands: float formats draw uniform [lo, hi) float32 exactly as before; int8 draws the whole int8 range."""
    if is_int():
        return np.random.randint(-128, 128, shape).astype(np.float32)
    return np.random.uniform(lo, hi, shape).astype(np.float32)


def int8_bias(N, suffix) -> np.ndarray:
    """The int8 mesh TB's int32 bias for set <suffix>: set 1 within 4096 below INT32_MAX, so its positive sums wrap; others uniform."""
    s = int(suffix.lstrip("_") or 0)
    rng = np.random.default_rng(7000 + s)  # its own generator: the operand draws stay as they were
    if s == 1:
        return (2**31 - 1) - rng.integers(0, 4096, N)
    return rng.integers(-2**31, 2**31, N)


def float_bias(N, seed) -> np.ndarray:
    """A float bias row as the format's bits: +0, -0 and the smallest and largest subnormals of each sign in 3 of every 4 columns."""
    f = fpu.FORMATS[FORMAT]
    sg = 1 << (f.w - 1)
    special = [0, sg, 1, f.mmask, sg | 1, sg | f.mmask]
    b = to_bits(np.random.default_rng(seed).uniform(-1, 1, N).astype(np.float32))
    cols = [c for c in range(N) if c % 4 != 3]
    for i, c in enumerate(cols):
        b[c] = special[i % len(special)]
    return b


def to_int(x) -> np.ndarray:
    """int8 operand values as int64; anything that is not an integer in [-128, 127] is an error, never rounded or clipped."""
    v = np.asarray(x, dtype=np.float64)
    if not (np.all(v == np.round(v)) and v.min() >= -128 and v.max() <= 127):
        raise ValueError(f"int8 operands must be integers in [-128, 127]; got values in [{v.min()}, {v.max()}]")
    return v.astype(np.int64)


def to_bits(x) -> np.ndarray:
    """float32 values, rounded to the format (fp32 nearest, then nearest at the format's width); subnormals flush to zero."""
    f = fpu.FORMATS[FORMAT]
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.int64)
    b = np.vectorize(lambda v: fpu.from_fp32(int(v), f.m))(u) if f.m != 23 else u
    return np.where(((b >> f.m) & f.emax) == 0, b & (1 << (f.w - 1)), b).astype(np.int64)


def to_float(bits) -> np.ndarray:
    f = fpu.FORMATS[FORMAT]
    return (np.asarray(bits, dtype=np.int64) << (23 - f.m)).astype(np.uint32).view(np.float32)


def _write_words(path, bits, d=None) -> None:
    d = d or digits()
    with open(path, "w") as fh:
        fh.write("".join(f"{int(v):0{d}x}\n" for v in np.asarray(bits).flatten()))


def _f2h(v) -> str:
    return "".join(f"{b:02x}" for b in struct.pack(">f", float(v)))


def _bias_words(bias) -> np.ndarray:
    """Bias file words: int8's int32 values, a float format's bits widened exactly to fp32."""
    if is_int():
        return np.asarray(bias, dtype=np.int64) & 0xFFFFFFFF
    return fpu.widen(fpu.FORMATS[FORMAT], bias) & 0xFFFFFFFF


def write_set(A, B, stim_dir, suffix="", bias=None, pack=0):
    """Writes one set's matrixA/B/C<suffix>.mem (bias: matrixBias, int8 values or the float format's bits; pack != 0 packShift,
    C per block alone); returns C (int8: int64)."""
    N = B.shape[1]
    if suffix in ("", "_0"):  # a test's first set: another test's accumulate files would run in this one
        for fn in os.listdir(stim_dir):
            if fn.startswith("acc") and fn.endswith(".mem"):
                os.remove(os.path.join(stim_dir, fn))
    bf = os.path.join(stim_dir, f"matrixBias{suffix}.mem")
    if bias is not None:
        _write_words(bf, _bias_words(bias), BIAS_DIGITS)
    elif os.path.exists(bf):
        os.remove(bf)  # a stale bias would bias this set
    # A set shorter than N x N is written with its zero rows: the staging bank is not reset, so the test must not rely on it.
    assert A.shape[1] == N and A.shape[0] <= N and B.shape[0] <= N, f"unsupported set shapes {A.shape} and {B.shape}"
    A = np.vstack([A, np.zeros((N - A.shape[0], N), dtype=np.float32)]).astype(np.float32)
    B = np.vstack([B, np.zeros((N - B.shape[0], N), dtype=np.float32)]).astype(np.float32)
    pf = os.path.join(stim_dir, f"packShift{suffix}.mem")
    if pack:
        with open(pf, "w") as fh:
            fh.write(f"{int(pack):x}\n")
    elif os.path.exists(pf):
        os.remove(pf)  # a stale shift would pack this set
    if is_int():
        Ai, Bi = to_int(A), to_int(B)
        b = N >> pack
        Bm = Bi * np.kron(np.eye(N // b, dtype=np.int64), np.ones((b, b), np.int64)) if pack else Bi  # the skip ignores off-block weights
        Ci = mesh_model.matmul_int([(Ai, Bm)], N, bias)
        _write_words(os.path.join(stim_dir, f"matrixA{suffix}.mem"), Ai & 0xFF)
        _write_words(os.path.join(stim_dir, f"matrixB{suffix}.mem"), Bi & 0xFF)
        _write_words(os.path.join(stim_dir, f"matrixC{suffix}.mem"), Ci & 0xFFFFFFFF, result_digits())
        return Ci
    Ab, Bb = to_bits(A), to_bits(B)
    f = fpu.FORMATS[FORMAT]
    Cb = (mesh_model.matmul_packed(f, Ab, Bb, N, pack, bias) if pack
          else mesh_model.matmul(f, [(Ab, Bb)], N, TILE, COLLAPSE_K, bias))
    if FORMAT == "fp32":  # operands as their float32 words, exactly as before; C is the mesh's own bit-exact result
        for name, M in (("A", A), ("B", B)):
            with open(os.path.join(stim_dir, f"matrix{name}{suffix}.mem"), "w") as fh:
                for v in np.asarray(M).flatten():
                    fh.write(_f2h(v) + "\n")
        _write_words(os.path.join(stim_dir, f"matrixC{suffix}.mem"), Cb)
        return to_float(Cb)
    for name, M in (("A", Ab), ("B", Bb), ("C", Cb)):
        _write_words(os.path.join(stim_dir, f"matrix{name}{suffix}.mem"), M)
    return to_float(Cb)


def write_accum(passes, stim_dir, bias=None):
    """Writes one sum over len(passes) N x N passes as accA_<p>/accB_<p>.mem, its bias accBias.mem and result accC.mem; returns C."""
    N = passes[0][1].shape[1]
    if is_int():
        ops = [(to_int(A), to_int(B)) for A, B in passes]
        C = mesh_model.matmul_int(ops, N, bias)
        words = [(A & 0xFF, B & 0xFF) for A, B in ops]
        Cw = C & 0xFFFFFFFF
    else:
        ops = [(to_bits(A), to_bits(B)) for A, B in passes]
        C = Cw = mesh_model.matmul(fpu.FORMATS[FORMAT], ops, N, TILE, COLLAPSE_K, bias)
        words = ops
    for p, (A, B) in enumerate(words):
        _write_words(os.path.join(stim_dir, f"accA_{p}.mem"), A)
        _write_words(os.path.join(stim_dir, f"accB_{p}.mem"), B)
    _write_words(os.path.join(stim_dir, "accC.mem"), Cw, result_digits())
    if bias is not None:
        _write_words(os.path.join(stim_dir, "accBias.mem"), _bias_words(bias), BIAS_DIGITS)
    return C


def check_widths(stim_dir) -> None:
    """Every stimulus word must be its kind's width in this format: a stale file from another format would be truncated silently."""
    for fn in sorted(os.listdir(stim_dir)):
        if fn.startswith(("matrix", "acc")) and fn.endswith(".mem"):
            d = (BIAS_DIGITS if fn.startswith(("matrixBias", "accBias")) else
                 result_digits() if fn.startswith(("matrixC", "accC")) else digits())
            for i, ln in enumerate(open(os.path.join(stim_dir, fn))):
                if ln.strip() and len(ln.strip()) != d:
                    raise ValueError(f"{fn}:{i + 1}: word '{ln.strip()}' is not {d} hex digits ({FORMAT})")
