# rope-first-principle

Minimal, dependency-light verification of the mathematical claims behind
Rotary Position Embedding (RoPE). Pure NumPy for the core library and
tests. Matplotlib for the animations. No models, no frameworks, no GPU.

## Layout

```
rope.py                     Core primitives. Pure functions, no I/O.
test_invariance.py          2D case: relative invariance and norm preservation.
test_conventions.py         Adjacent-pair vs split-half layout equivalence.
test_position_mismatch.py   Effect of inconsistent position ids after cache truncation.
rope_implementation_docs.md Per-function derivations and design notes.

animations/                 Animation generator scripts (matplotlib, self-contained).
assets/                      Pre-rendered GIFs, one per animation script.
notebook/                   Walkthrough notebook: math, code, and GIFs together.
```

`animations/*.py` are standalone: each can be run directly
(`python3 animations/anim_score_matrix.py`) or pasted whole into a
Colab cell. Each regenerates its GIF into `assets/` on run.

`notebook/rope_first_principle_walkthrough.ipynb` narrates the reasoning
and displays the GIFs already in `assets/`. It does not regenerate them;
if you change an animation script, rerun it manually and the notebook
will pick up the new GIF next time it's opened. The notebook imports
`rope.py` directly and runs the test scripts via subprocess to show
live PASS/FAIL output, so its explanations stay tied to code that
actually executes rather than to hardcoded claims.

## Usage

```
python3 test_invariance.py
python3 test_conventions.py
python3 test_position_mismatch.py
```

Each script prints one line per test and exits 0 on success, 1 on failure.

To regenerate an animation:

```
python3 animations/anim_score_matrix.py
```

writes (or overwrites) the corresponding file in `assets/`.

## What is verified

| Claim | Test |
|---|---|
| Rotating q and k independently by absolute positions yields scores that depend only on relative distance | test_invariance.py |
| Rotation preserves vector norm at any position, including invalid ones | test_invariance.py |
| The paper's adjacent-pair layout and the HuggingFace split-half layout are equivalent under an explicit permutation | test_conventions.py |
| Omitting that permutation corrupts scores without raising any error | test_conventions.py |
| Deriving a query position from truncated cache length, while cached keys retain original absolute rotations, measurably diverts attention output | test_position_mismatch.py |

## Scope

This repository verifies properties of the encoding itself. Claims about
how positional errors affect a trained model depend on that model's
weights and training distribution and are out of scope here. See the
final section of rope_implementation_docs.md.

## Reference

Su et al., "RoFormer: Enhanced Transformer with Rotary Position
Embedding", arXiv:2104.09864.