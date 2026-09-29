#!/usr/bin/env python3
"""stim_format's expected C is the bit-exact mesh result in every format, fp32 included."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matmul_tests as mt  # noqa: E402
import stim_format as sf  # noqa: E402


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


if __name__ == "__main__":
    test_fp32_expected_is_the_mesh_result()
    print("PASS test_fp32_expected_is_the_mesh_result")
