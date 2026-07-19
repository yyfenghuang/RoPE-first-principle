"""Measure the effect of inconsistent position ids after KV-cache truncation.

Scenario: a prefix of length P is dropped from a cache of length P + S.
Cached keys were rotated at their original absolute positions [P, P+S)
before truncation, and that rotation cannot be undone by slicing tensors.
If the next query's position id is then derived from the truncated cache
length (S) instead of the full history (P + S), query and keys disagree
about the positional frame. Some relative distances become negative,
which a causally trained attention never encounters.

This script quantifies the divergence between the correct and the
mismatched query position using a synthetic single-head attention with
identity projections, so the only source of divergence is RoPE itself.

Output: one line per suffix length with cosine similarity and L2 distance
between the correct and mismatched attention outputs. A similarity below
1.0 with unchanged output norm demonstrates the silent failure mode:
magnitude is preserved by orthogonality while direction drifts.

Exit code 0 if divergence is detected at every tested length, 1 otherwise.
"""

import sys

import numpy as np

from scripts.rope import attention_single_query, cosine_similarity, inverse_frequencies

DIM = 16
PREFIX_LEN = 20
SUFFIX_LENGTHS = [2, 4, 8, 16, 32, 64, 128]


def run_scenario(prefix_len: int, suffix_len: int, seed: int) -> tuple[float, float]:
    """Return (cosine_similarity, l2_distance) between correct and mismatched outputs."""
    rng = np.random.default_rng(seed)
    freqs = inverse_frequencies(DIM)
    total = prefix_len + suffix_len

    kept_tokens = rng.normal(size=(suffix_len, DIM)) * 0.5
    new_query = rng.normal(size=(1, DIM)) * 0.5
    tokens = np.concatenate([kept_tokens, new_query])

    # Keys keep their original absolute positions; only the query position differs.
    key_positions = np.concatenate([np.arange(prefix_len, total), [total]])

    out_correct, _, _ = attention_single_query(
        tokens, query_position=total, key_positions=key_positions, freqs=freqs
    )
    out_mismatched, _, _ = attention_single_query(
        tokens, query_position=suffix_len, key_positions=key_positions, freqs=freqs
    )
    return (
        cosine_similarity(out_correct, out_mismatched),
        float(np.linalg.norm(out_correct - out_mismatched)),
    )


def main() -> int:
    print(f"{'suffix_len':>10}  {'cosine_sim':>10}  {'l2_dist':>8}")
    all_diverged = True
    for suffix_len in SUFFIX_LENGTHS:
        sim, l2 = run_scenario(PREFIX_LEN, suffix_len, seed=1)
        print(f"{suffix_len:>10}  {sim:>10.6f}  {l2:>8.6f}")
        all_diverged &= sim < 1.0 - 1e-9

    print(f"{'PASS' if all_diverged else 'FAIL'}  divergence_detected_at_all_lengths")
    return 0 if all_diverged else 1


if __name__ == "__main__":
    sys.exit(main())