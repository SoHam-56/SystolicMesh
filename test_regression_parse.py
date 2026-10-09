#!/usr/bin/env python3
"""regression._parse: a log holding an assertion firing fails as ASSERT, whatever the testbench's counts say; a clean log passes."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import regression as reg  # noqa: E402

CLEAN = """ Passed Sets:    23
 Failed Sets:    0
 Total Elements: 5888
  Set 0 : 59 cycles  (590 ns)
  Set 1 : 61 cycles  (610 ns)
SUCCESS
"""
FIRED = ("[2975000] %Error: SystolicMesh.sv:545: Assertion failed in TB_SystolicMesh.dut.a_pack_one_pass: "
         "SystolicMesh: a packed set is part of an accumulated sum\n")


def test_clean_log_passes():
    r = reg._parse(CLEAN)
    assert r["status"] == "PASS" and r["passed"] == 23 and r["avg_cyc"] == 60, r


def test_assertion_firing_fails():
    r = reg._parse(CLEAN + FIRED)  # the TB's own counts still read clean
    assert r["status"] == "ASSERT", r


def test_assertion_without_error_prefix_fails():
    r = reg._parse(CLEAN + "Assertion failed in TB_SystolicMesh.dut.a_read_outstanding\n")
    assert r["status"] == "ASSERT", r


def test_error_line_is_not_a_pass():
    r = reg._parse(CLEAN + "%Error: TB_SystolicMesh.sv:12: some runtime error\n")
    assert r["status"] == "NO-RUN", r


MIN_TESTS = 4  # tests this file holds; fewer run means one was lost, which is a failure

if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    if len(tests) < MIN_TESTS:
        print(f"FAIL: {len(tests)} tests ran, fewer than MIN_TESTS {MIN_TESTS}")
        sys.exit(1)
    print(f"ALL {len(tests)} PASSED")
