"""Verify that rotary encoding yields position-relative attention scores.

Tests three properties of 2D rotation as position encoding:
  1. Shifting both positions by the same offset leaves the dot product unchanged.
  2. All position pairs with equal difference produce identical dot products.
  3. Rotation preserves vector norm at any position, including negative ones.

Exit code 0 if all tests pass, 1 otherwise.
"""

import sys

import numpy as np

from scripts.rope import rotate_2d

ATOL = 1e-9


def test_shift_invariance() -> bool:
    rng = np.random.default_rng(42)
    a, b = rng.normal(size=2), rng.normal(size=2)
    theta = 0.1
    m, n, shift = 3, 7, 1000

    before = np.dot(rotate_2d(a, m, theta), rotate_2d(b, n, theta))
    after = np.dot(rotate_2d(a, m + shift, theta), rotate_2d(b, n + shift, theta))
    return abs(before - after) < ATOL


def test_equal_difference_equal_score() -> bool:
    rng = np.random.default_rng(42)
    a, b = rng.normal(size=2), rng.normal(size=2)
    theta = 0.1
    pairs = [(0, 4), (10, 14), (100, 104), (5000, 5004), (-50, -46)]

    scores = [
        np.dot(rotate_2d(a, m, theta), rotate_2d(b, n, theta))
        for m, n in pairs
    ]
    return max(scores) - min(scores) < ATOL


def test_norm_preservation() -> bool:
    rng = np.random.default_rng(42)
    v = rng.normal(size=2)
    theta = 0.1
    positions = [0, 1, 50, 500, -1, -50, -500]

    norm = np.linalg.norm(v)
    return all(
        abs(np.linalg.norm(rotate_2d(v, p, theta)) - norm) < ATOL
        for p in positions
    )


def main() -> int:
    tests = [
        ("shift_invariance", test_shift_invariance),
        ("equal_difference_equal_score", test_equal_difference_equal_score),
        ("norm_preservation", test_norm_preservation),
    ]
    failed = 0
    for name, fn in tests:
        ok = fn()
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        failed += not ok
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())