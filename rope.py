"""Minimal RoPE (Rotary Position Embedding) primitives.

Pure functions only. No I/O, no state, no model dependencies.
See rope_implementation_docs.md for derivations and design notes.
"""

import numpy as np


def rotation_matrix_2d(angle: float) -> np.ndarray:
    """Return the 2x2 counter-clockwise rotation matrix for the given angle in radians."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def rotate_2d(v: np.ndarray, position: int, theta: float) -> np.ndarray:
    """Rotate a 2D vector by (position * theta) radians."""
    return rotation_matrix_2d(position * theta) @ v


def inverse_frequencies(dim: int, base: float = 10000.0) -> np.ndarray:
    """Return the RoPE frequency spectrum theta_i = base^(-2i/dim) for i in [0, dim/2).

    dim must be even. The result has shape (dim // 2,).
    """
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}")
    i = np.arange(dim // 2)
    return base ** (-2.0 * i / dim)


def apply_rope_adjacent(x: np.ndarray, position: int, freqs: np.ndarray) -> np.ndarray:
    """Apply RoPE using adjacent-pair convention: dims (2i, 2i+1) rotate by position * freqs[i].

    This is the layout written in the RoFormer paper (Su et al. 2023, Eq. 15).
    x has shape (dim,); freqs has shape (dim // 2,).
    """
    dim = x.shape[-1]
    out = np.empty_like(x)
    angles = position * freqs
    cos, sin = np.cos(angles), np.sin(angles)
    x_even, x_odd = x[0::2], x[1::2]
    out[0::2] = x_even * cos - x_odd * sin
    out[1::2] = x_even * sin + x_odd * cos
    return out


def apply_rope_half(x: np.ndarray, position: int, freqs: np.ndarray) -> np.ndarray:
    """Apply RoPE using split-half convention: dims (i, i + dim/2) rotate by position * freqs[i].

    This is the layout used by HuggingFace Transformers (rotate_half).
    x has shape (dim,); freqs has shape (dim // 2,).
    """
    dim = x.shape[-1]
    half = dim // 2
    angles = position * freqs
    cos = np.concatenate([np.cos(angles), np.cos(angles)])
    sin = np.concatenate([np.sin(angles), np.sin(angles)])
    rotated_half = np.concatenate([-x[half:], x[:half]])
    return x * cos + rotated_half * sin


def adjacent_to_half_permutation(dim: int) -> np.ndarray:
    """Return the index permutation mapping adjacent-pair layout to split-half layout.

    perm[j] is the index in the adjacent layout whose value belongs at slot j
    in the split-half layout. Given x in adjacent layout, x[perm] is the
    equivalent input for apply_rope_half. Use np.argsort(perm) to map
    outputs back.
    """
    half = dim // 2
    perm = np.empty(dim, dtype=int)
    perm[:half] = np.arange(half) * 2        # first members of each pair
    perm[half:] = np.arange(half) * 2 + 1    # second members of each pair
    return perm


def attention_single_query(
    tokens: np.ndarray,
    query_position: int,
    key_positions: np.ndarray,
    freqs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute single-head attention for the last token as query.

    Uses identity Q/K/V projections so the only transformation applied
    is RoPE. This isolates positional effects from learned weights.

    tokens: (seq_len, dim). The last row is the query; all rows are keys and values.
    query_position: position id used to rotate the query.
    key_positions: (seq_len,) position ids used to rotate each key.

    Returns (output, scores, weights) where output is the attention-weighted
    sum of values, scores are pre-softmax logits, weights are post-softmax.
    """
    dim = tokens.shape[-1]
    q = apply_rope_adjacent(tokens[-1], query_position, freqs)
    scores = np.array([
        np.dot(q, apply_rope_adjacent(tokens[j], key_positions[j], freqs))
        for j in range(tokens.shape[0])
    ]) / np.sqrt(dim)
    exp = np.exp(scores - scores.max())
    weights = exp / exp.sum()
    return weights @ tokens, scores, weights


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine of the angle between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))