"""Verify equivalence between the two RoPE dimension-pairing conventions.

The RoFormer paper pairs adjacent dimensions (2i, 2i+1). HuggingFace
Transformers pairs split halves (i, i + dim/2). Both are valid rotary
encodings and produce identical attention scores, but only when tensor
layouts are converted with the correct permutation. Applying one
convention to weights trained under the other, without that permutation,
silently corrupts the result.

Exit code 0 if all tests pass, 1 otherwise.
"""

import sys

import numpy as np

from scripts.rope import (
    adjacent_to_half_permutation,
    apply_rope_adjacent,
    apply_rope_half,
    inverse_frequencies,
)

ATOL = 1e-9
DIM = 8


def test_both_conventions_preserve_norm() -> bool:
    rng = np.random.default_rng(0)
    freqs = inverse_frequencies(DIM)
    for _ in range(5):
        x = rng.normal(size=DIM)
        pos = int(rng.integers(0, 10000))
        norm = np.linalg.norm(x)
        if abs(np.linalg.norm(apply_rope_adjacent(x, pos, freqs)) - norm) > ATOL:
            return False
        if abs(np.linalg.norm(apply_rope_half(x, pos, freqs)) - norm) > ATOL:
            return False
    return True


def test_conventions_equivalent_under_permutation() -> bool:
    rng = np.random.default_rng(1)
    freqs = inverse_frequencies(DIM)
    q, k = rng.normal(size=DIM), rng.normal(size=DIM)
    m, n = 3, 11

    perm = adjacent_to_half_permutation(DIM)
    inv = np.argsort(perm)

    score_adjacent = np.dot(
        apply_rope_adjacent(q, m, freqs),
        apply_rope_adjacent(k, n, freqs),
    )
    score_half = np.dot(
        apply_rope_half(q[perm], m, freqs)[inv],
        apply_rope_half(k[perm], n, freqs)[inv],
    )
    return abs(score_adjacent - score_half) < ATOL


def test_layout_mismatch_changes_score() -> bool:
    """A missing permutation must produce a different score, not an error.

    This confirms the failure mode is silent: valid shapes, valid values,
    wrong semantics. The test passes when the mismatch is detectable only
    by comparing against the correct result.
    """
    rng = np.random.default_rng(2)
    freqs = inverse_frequencies(DIM)
    q, k = rng.normal(size=DIM), rng.normal(size=DIM)
    m, n = 3, 11

    perm = adjacent_to_half_permutation(DIM)

    score_consistent = np.dot(
        apply_rope_half(q[perm], m, freqs),
        apply_rope_half(k[perm], n, freqs),
    )
    score_mismatched = np.dot(
        apply_rope_half(q[perm], m, freqs),
        apply_rope_half(k, n, freqs),  # permutation omitted
    )
    return abs(score_consistent - score_mismatched) > ATOL


def main() -> int:
    tests = [
        ("both_conventions_preserve_norm", test_both_conventions_preserve_norm),
        ("conventions_equivalent_under_permutation", test_conventions_equivalent_under_permutation),
        ("layout_mismatch_changes_score", test_layout_mismatch_changes_score),
    ]
    failed = 0
    for name, fn in tests:
        ok = fn()
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        failed += not ok
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())