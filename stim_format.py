#!/usr/bin/env python3
"""The mesh stimulus's number format. fp32 writes float32 words and a float64 reference exactly as before; bf16 writes bf16
words and the bit-exact expected result from mesh_model, for the TB's exact compare."""
import os
import struct

import numpy as np

import mesh_model
from mesh_model import fpu

FORMAT, TILE, COLLAPSE_K = "fp32", 4, 1


def configure(fmt: str, tile: int, collapse_k: int) -> None:
    global FORMAT, TILE, COLLAPSE_K
    FORMAT, TILE, COLLAPSE_K = fmt, tile, collapse_k


def digits() -> int:
    return (fpu.FORMATS[FORMAT].w + 3) // 4


def to_bits(x) -> np.ndarray:
    """float32 values, rounded to the format (fp32 nearest, then nearest at the format's width); subnormals flush to zero."""
    f = fpu.FORMATS[FORMAT]
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.int64)
    b = np.vectorize(lambda v: fpu.from_fp32(int(v), f.m))(u) if f.m != 23 else u
    return np.where(((b >> f.m) & f.emax) == 0, b & (1 << (f.w - 1)), b).astype(np.int64)


def to_float(bits) -> np.ndarray:
    f = fpu.FORMATS[FORMAT]
    return (np.asarray(bits, dtype=np.int64) << (23 - f.m)).astype(np.uint32).view(np.float32)


def _write_words(path, bits) -> None:
    d = digits()
    with open(path, "w") as fh:
        fh.write("".join(f"{int(v):0{d}x}\n" for v in np.asarray(bits).flatten()))


def _f2h(v) -> str:
    return "".join(f"{b:02x}" for b in struct.pack(">f", float(v)))


def write_set(A, B, stim_dir, suffix=""):
    """Write matrixA/B/C<suffix>.mem for one set; returns C as floats."""
    if FORMAT == "fp32":
        C = (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32)
        for name, M in (("A", A), ("B", B), ("C", C)):
            with open(os.path.join(stim_dir, f"matrix{name}{suffix}.mem"), "w") as fh:
                for v in np.asarray(M).flatten():
                    fh.write(_f2h(v) + "\n")
        return C
    N = B.shape[1]
    # A set shorter than N x N fills the staging bank row-major and leaves the rest zero; the model sees the same bank.
    assert A.shape[1] == N and B.shape[0] <= N and A.shape[0] <= N, f"unsupported set shapes {A.shape} and {B.shape}"
    Ab, Bb = to_bits(A), to_bits(B)
    pad = lambda M: np.concatenate([M.flatten(), np.zeros(N * N - M.size, dtype=np.int64)]).reshape(N, N)
    Cb = mesh_model.matmul(fpu.FORMATS[FORMAT], [(pad(Ab), pad(Bb))], N, TILE, COLLAPSE_K)[:A.shape[0], :]
    for name, M in (("A", Ab), ("B", Bb), ("C", Cb)):
        _write_words(os.path.join(stim_dir, f"matrix{name}{suffix}.mem"), M)
    return to_float(Cb)


def check_widths(stim_dir) -> None:
    """Every matrix word must be this format's width: a stale fp32 file in a bf16 run would be truncated silently."""
    d = digits()
    for fn in sorted(os.listdir(stim_dir)):
        if fn.startswith("matrix") and fn.endswith(".mem"):
            for i, ln in enumerate(open(os.path.join(stim_dir, fn))):
                if ln.strip() and len(ln.strip()) != d:
                    raise ValueError(f"{fn}:{i + 1}: word '{ln.strip()}' is not {d} hex digits ({FORMAT})")
