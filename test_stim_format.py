#!/usr/bin/env python3
"""stim_format's expected C is the bit-exact mesh result in every format: fp32 from mesh_model.matmul, int8 from matmul_int."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matmul_tests as mt  # noqa: E402
import mesh_model as mm  # noqa: E402
import stim_format as sf  # noqa: E402


def _words(d, name):
    return open(os.path.join(d, name)).read().split()


def _wrap32(v):
    return ((v + 2**31) % 2**32) - 2**31


def test_fp32_expected_is_the_mesh_result():
    # N=64 padding set 9001: output 1085 is a near-zero cancellation; the RTL returns 0x38050000 (3.1710e-5), the
    # correctly rounded sum is 0x3803ab66 (3.1392e-5), 1.01% away, so a golden from float64 fails correct hardware
    mt._seed(9001)
    A = np.random.uniform(-1, 1, (64, 64)).astype(np.float32)
    B = np.random.uniform(-1, 1, (64, 64)).astype(np.float32)
    sf.configure("fp32", 2, 1)
    with tempfile.TemporaryDirectory() as d:
        sf.write_set(A, B, d)
        C = [int(x, 16) for x in open(os.path.join(d, "matrixC.mem")).read().split()]
        A_words = open(os.path.join(d, "matrixA.mem")).read().split()
    assert C[1085] == 0x38050000, hex(C[1085])
    assert A_words[0] == f"{int(A.view(np.uint32).flat[0]):08x}"  # operands still written as their float32 bits


def test_int8_words_and_expected():
    # -1 x 127 over N=8 is -1016 (fffffc08); -128 x 127 is -130048 (fffe0400), beyond int16, so the sums must be int32
    N = 8
    A = np.full((N, N), -128, dtype=np.float32)
    A[0, :] = -1
    B = np.full((N, N), 127, dtype=np.float32)
    sf.configure("int8", 4, 1)
    with tempfile.TemporaryDirectory() as d:
        C = sf.write_set(A, B, d)
        a, b, c = _words(d, "matrixA.mem"), _words(d, "matrixB.mem"), _words(d, "matrixC.mem")
    assert len(a) == len(b) == len(c) == N * N
    assert a[0] == "ff" and a[N] == "80" and b[0] == "7f", (a[0], a[N], b[0])
    assert {len(w) for w in a + b} == {2} and {len(w) for w in c} == {8}
    assert c[0] == "fffffc08" and c[N] == "fffe0400", (c[0], c[N])
    assert int(C[0, 0]) == -1016 and int(C[1, 0]) == -130048


def test_int8_largest_sum():
    # -128 x -128 over K=64 is 64 x 16384 = 2^20, the largest sum one mesh set can produce
    N = 64
    A = np.full((N, N), -128, dtype=np.float32)
    sf.configure("int8", 2, 0)
    with tempfile.TemporaryDirectory() as d:
        sf.write_set(A, A, d)
        c = _words(d, "matrixC.mem")
    assert set(c) == {"00100000"}, sorted(set(c))[:3]


def test_matmul_int_wraps_like_the_adders():
    # the RTL wraps every add mod 2^32; wrapping only the final sum is the same because integer addition is associative
    N = 8
    rng = np.random.default_rng(5)
    passes = [(rng.integers(-128, 128, (N, N)), rng.integers(-128, 128, (N, N))) for _ in range(3)]
    bias = rng.integers(-2**31, 2**31, N)
    C = mm.matmul_int(passes, N, bias)
    for i in range(N):
        for j in range(N):
            s = 0
            for A, B in passes:
                for k in range(N):
                    s = _wrap32(s + int(A[i, k]) * int(B[k, j]))
            assert C[i, j] == _wrap32(s + int(bias[j])), (i, j, C[i, j])
    E = mm.matmul_int([(np.eye(N, dtype=np.int64), np.ones((N, N), dtype=np.int64))], N, np.full(N, 2**31 - 1))
    assert (E == -2**31).all()  # 2^31 - 1 + 1 wraps to INT32_MIN


def test_int8_bias_file_and_wrap():
    # set _1's bias sits within 4096 below INT32_MAX, so every positive sum of the set wraps negative, as the int32 adders do
    N = 8
    b = sf.int8_bias(N, "_1")
    assert (b <= 2**31 - 1).all() and (b > 2**31 - 1 - 4096).all()
    A = np.full((N, N), 127, dtype=np.float32)
    sf.configure("int8", 4, 1)
    with tempfile.TemporaryDirectory() as d:
        C = sf.write_set(A, A, d, "_1", bias=b)
        w = _words(d, "matrixBias_1.mem")
        sf.check_widths(d)
    assert len(w) == N and {len(x) for x in w} == {8}
    assert (C == _wrap32(N * 127 * 127 + b[None, :])).all() and (C < 0).all()
    sf.configure("bf16", 4, 1)
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "matrixBias_0.mem"), "w").write("0001\n")
        try:
            sf.check_widths(d)
        except ValueError:
            return
    raise AssertionError("check_widths took a 4-digit bias file in bf16: the mesh would read it zero-extended, not widened")


def test_float_bias_widened_and_special():
    # bf16 bias bits widen exactly (x << 16) into 8-digit words; the model adds them in fp32; ±0 and subnormals in 3 of 4 columns
    N = 8
    sf.configure("bf16", 4, 1)
    b = sf.float_bias(N, 1)
    assert [int(x) for x in b[[0, 1, 2, 4, 5, 6]]] == [0x0000, 0x8000, 0x0001, 0x007F, 0x8001, 0x807F], [hex(x) for x in b]
    A = sf.rand(-1, 1, (N, N)).astype(np.float32)
    B = sf.rand(-1, 1, (N, N)).astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        sf.write_set(A, B, d, "_0", b)
        w = _words(d, "matrixBias_0.mem")
        c = [int(x, 16) for x in _words(d, "matrixC_0.mem")]
        sf.check_widths(d)
        sf.write_set(A, B, d, "_0")
        assert not os.path.exists(os.path.join(d, "matrixBias_0.mem")), "a stale bias would bias the next set"
    assert w[:3] == ["00000000", "80000000", "00010000"] and w[4] == "007f0000", w
    f = mm.fpu.BF16
    assert c == [int(x) for x in mm.matmul(f, [(sf.to_bits(A), sf.to_bits(B))], N, 4, 1, b).flatten()]


def test_accum_files():
    # one sum over 3 passes: accA/accB per pass, accC the model's result over every pass, accBias 8 digits; a test's first set clears them
    N = 8
    for fmt in ("fp32", "bf16", "int8"):
        sf.configure(fmt, 4, 1)
        passes = [(sf.rand(-1, 1, (N, N)).astype(np.float32), sf.rand(-1, 1, (N, N)).astype(np.float32)) for _ in range(3)]
        bias = sf.int8_bias(N, "_0") if fmt == "int8" else sf.float_bias(N, 3)
        with tempfile.TemporaryDirectory() as d:
            C = sf.write_accum(passes, d, bias)
            sf.check_widths(d)
            names = sorted(os.listdir(d))
            assert names == ["accA_0.mem", "accA_1.mem", "accA_2.mem", "accB_0.mem", "accB_1.mem", "accB_2.mem", "accBias.mem",
                             "accC.mem"], names
            assert {len(x) for x in _words(d, "accBias.mem")} == {8}
            assert {len(x) for x in _words(d, "accC.mem")} == {sf.result_digits()}
            if fmt == "int8":
                ref = mm.matmul_int([(sf.to_int(A), sf.to_int(B)) for A, B in passes], N, bias)
            else:
                ref = mm.matmul(mm.fpu.FORMATS[fmt], [(sf.to_bits(A), sf.to_bits(B)) for A, B in passes], N, 4, 1, bias)
            assert np.array_equal(C, ref), fmt
            sf.write_set(passes[0][0], passes[0][1], d, "_0")
            assert not [x for x in os.listdir(d) if x.startswith("acc")], "another test's accumulate files would run"


def test_int8_rejects_what_is_not_int8():
    sf.configure("int8", 4, 1)
    for bad in (0.5, 128.0, -129.0):
        A = np.zeros((4, 4), dtype=np.float32)
        A[1, 2] = bad
        with tempfile.TemporaryDirectory() as d:
            try:
                sf.write_set(A, np.zeros((4, 4), dtype=np.float32), d)
            except ValueError:
                continue
        raise AssertionError(f"write_set took the int8 operand {bad}")
    try:
        mm.matmul_int([(np.full((4, 4), 200), np.zeros((4, 4)))], 4)  # a bit pattern, not a value
    except ValueError:
        return
    raise AssertionError("matmul_int took an operand of 200")


def test_int8_check_widths():
    sf.configure("int8", 4, 1)
    ones = np.ones((4, 4), dtype=np.float32)
    with tempfile.TemporaryDirectory() as d:
        sf.write_set(ones, ones, d, "_0")
        sf.check_widths(d)  # 2-digit operands, 8-digit results
        for name, word in (("matrixA_0.mem", "3f800000"), ("matrixC_0.mem", "04")):
            with open(os.path.join(d, name), "w") as fh:
                fh.write(word + "\n")
            try:
                sf.check_widths(d)
            except ValueError:
                sf.write_set(ones, ones, d, "_0")
                continue
            raise AssertionError(f"check_widths took '{word}' in {name}")


def test_pack_shift_file_and_off_block_weights():
    # a packed write makes packShift<suffix>.mem and ignores off-block weights; an unpacked write removes the file
    N = 8
    for fmt in ("fp32", "int8"):
        sf.configure(fmt, 4, 1)
        A = sf.rand(-1, 1, (N, N)).astype(np.float32)
        B = sf.rand(-1, 1, (N, N)).astype(np.float32)
        Bz = np.where(np.kron(np.eye(2), np.ones((4, 4))).astype(bool), B, np.float32(0.0))  # shift 1: b = 4
        with tempfile.TemporaryDirectory() as d:
            pf = os.path.join(d, "packShift_1.mem")
            C = sf.write_set(A, B, d, "_1", None, 1)
            assert open(pf).read().split() == ["1"]
            assert (C == sf.write_set(A, Bz, d, "_1", None, 1)).all(), f"{fmt}: off-block weights reached C"
            sf.write_set(A, B, d, "_1")
            assert not os.path.exists(pf), "a stale packShift would pack the next set"


def test_rand():
    # int8 draws the whole range (4096 draws: missing -128 or 127 has probability about 2e-7); floats draw exactly as before
    sf.configure("int8", 4, 1)
    np.random.seed(1)
    x = sf.rand(-1, 1, (64, 64))
    assert (x == np.round(x)).all() and x.min() == -128 and x.max() == 127
    sf.configure("fp32", 4, 1)
    np.random.seed(1)
    y = sf.rand(-1, 1, (8, 8))
    np.random.seed(1)
    assert (y == np.random.uniform(-1, 1, (8, 8)).astype(np.float32)).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
