# rope-first-principle

Verification of the mathematical claims behind Rotary Position Embedding
(RoPE). NumPy for the library and tests, Matplotlib for the animations.

## Layout

```
RoPE_principle.md           Derivation from first principles.
rope_implementation_docs.md Per-function notes.

rope.py                     Core primitives. Pure functions, no I/O.
test_invariance.py          Relative invariance and norm preservation.
test_conventions.py         Adjacent-pair vs split-half layout equivalence.
test_position_mismatch.py   Position ids inconsistent with cached rotations.

animations/                 One script per GIF. Each writes to assets/.
assets/                     Rendered GIFs.
notebook/                   Walkthrough: reads rope.py and assets/.
```

## Usage

```
python3 test_invariance.py
python3 test_conventions.py
python3 test_position_mismatch.py
```

One line per test. Exit 0 if all pass, 1 otherwise.

```
python3 animations/anim_score_matrix.py
```

Writes the corresponding GIF to `assets/`.

## What is verified

| Claim | Test |
|---|---|
| Rotating q and k independently by absolute positions yields scores that depend only on relative distance | test_invariance.py |
| Rotation preserves vector norm at any position, including invalid ones | test_invariance.py |
| The adjacent-pair and split-half layouts are equivalent under an explicit permutation | test_conventions.py |
| Omitting that permutation corrupts scores without raising any error | test_conventions.py |
| A query position derived from truncated cache length, against keys holding their original rotations, diverts attention output | test_position_mismatch.py |

Properties of the encoding, not of any trained model. See
`RoPE_principle.md`, section 1.

## Reference

Su et al., "RoFormer: Enhanced Transformer with Rotary Position
Embedding", arXiv:2104.09864.