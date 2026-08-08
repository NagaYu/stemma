"""The sketch must be blind to relabelling and rescaling, and only to those.

Claim: low-false-positive -- neuron permutation and per-row/per-column rescaling
are the two gauge symmetries a transformer's weights are defined up to. If the
fingerprint moved under them, two copies of the *same* model could look
unrelated (false negative); if it were invariant to everything, two unrelated
models would collide (false positive). This file pins both ends.
"""

from __future__ import annotations

import numpy as np
import pytest

from stemma.sketch import sketch_distance, sketch_model, sketch_similarity, tensor_invariants
from stemma.types import SKETCH_DIM

# --------------------------------------------------------------------------- #
# Thresholds (judgement calls, stated once here)
# --------------------------------------------------------------------------- #

#: CONTRACT.md fixes this tolerance: L-inf <= 5e-2 under P/Q/D1/D2.
INVARIANCE_LINF_TOL: float = 5e-2

#: Range of the positive diagonal rescalings. A 16x dynamic range is far beyond
#: anything LayerNorm folding produces in practice, so passing here is a strong
#: statement rather than a token one.
DIAG_LO: float = 0.25
DIAG_HI: float = 4.0

#: sketch_distance of a model against itself. Not exactly 0.0 only because the
#: sketch is float32; anything above this would mean the sketch is not a pure
#: function of the weights.
SELF_DISTANCE_TOL: float = 1e-5

#: sketch_distance must be symmetric to within float32 round-off.
SYMMETRY_TOL: float = 1e-6

#: Rows sampled per tensor. Small: this file tests invariance, not scale.
MAX_ROWS: int = 64


def _trained_looking(rng: np.random.Generator, m: int, n: int) -> np.ndarray:
    """A matrix with a decaying spectrum and heavy-tailed row norms."""
    r = min(m, n)
    U = rng.standard_normal((m, r))
    V = rng.standard_normal((r, n))
    decay = np.exp(-np.arange(r, dtype=np.float64) / max(1.0, r / 3.0))
    W = (U * decay) @ V
    W *= np.exp(0.4 * rng.standard_normal((m, 1)))
    return W.astype(np.float32)


@pytest.mark.parametrize("shape", [(96, 64), (64, 96), (128, 128), (256, 64)])
def test_tensor_invariants_are_permutation_and_scale_invariant(shape) -> None:
    """D1 P W Q D2 must sketch identically to W, for permutations P,Q and D>0."""
    m, n = shape
    rng = np.random.default_rng(7)
    W = _trained_looking(rng, m, n)

    base = tensor_invariants(W, k=48, seed=0)
    assert base.shape == (32,)
    assert np.all(np.isfinite(base)), "invariants must never contain nan/inf"

    for trial in range(3):
        r = np.random.default_rng(100 + trial)
        P = r.permutation(m)
        Q = r.permutation(n)
        d1 = r.uniform(DIAG_LO, DIAG_HI, size=m).astype(np.float32)
        d2 = r.uniform(DIAG_LO, DIAG_HI, size=n).astype(np.float32)
        W2 = (d1[:, None] * W[P][:, Q] * d2[None, :]).astype(np.float32)

        got = tensor_invariants(W2, k=48, seed=0)
        linf = float(np.max(np.abs(got - base)))
        assert linf <= INVARIANCE_LINF_TOL, (
            f"shape={shape} trial={trial}: L-inf {linf:.4g} exceeds "
            f"{INVARIANCE_LINF_TOL}; worst feature index "
            f"{int(np.argmax(np.abs(got - base)))}"
        )


def test_tensor_invariants_are_deterministic() -> None:
    """Same matrix + same seed -> bit-identical invariants (no hidden RNG state)."""
    rng = np.random.default_rng(3)
    W = _trained_looking(rng, 96, 64)
    a = tensor_invariants(W, k=48, seed=0)
    b = tensor_invariants(W, k=48, seed=0)
    np.testing.assert_array_equal(a, b)


def test_tensor_invariants_separate_unrelated_matrices() -> None:
    """Invariance must not be degeneracy: two independent draws must differ.

    Claim: low-false-positive.
    """
    W1 = _trained_looking(np.random.default_rng(1), 128, 128)
    W2 = _trained_looking(np.random.default_rng(2), 128, 128)
    f1 = tensor_invariants(W1, k=48, seed=0)
    f2 = tensor_invariants(W2, k=48, seed=0)
    assert float(np.max(np.abs(f1 - f2))) > INVARIANCE_LINF_TOL / 5.0


def test_sketch_distance_is_zero_against_itself(tiny_parent: str) -> None:
    """A model compared to itself has distance ~0 and similarity ~1."""
    s = sketch_model(tiny_parent, max_rows=MAX_ROWS, k=32, seed=0)
    assert s.vector.shape == (SKETCH_DIM,)
    assert s.present.any(), "no (role, depth) slot was filled for a Llama-shaped model"

    d = sketch_distance(s, s)
    assert d == pytest.approx(0.0, abs=SELF_DISTANCE_TOL)
    assert sketch_similarity(s, s) == pytest.approx(1.0, abs=SELF_DISTANCE_TOL)


def test_sketch_model_is_reproducible(tiny_parent: str) -> None:
    """Two sketches of the same directory are identical and mutually distance-0."""
    a = sketch_model(tiny_parent, max_rows=MAX_ROWS, k=32, seed=0)
    b = sketch_model(tiny_parent, max_rows=MAX_ROWS, k=32, seed=0)
    np.testing.assert_allclose(a.vector, b.vector, rtol=0, atol=0)
    assert sketch_distance(a, b) == pytest.approx(0.0, abs=SELF_DISTANCE_TOL)


def test_sketch_distance_is_symmetric(tiny_parent: str, tiny_unrelated: str) -> None:
    """d(a,b) == d(b,a); a directional statistic must not leak into the sketch."""
    a = sketch_model(tiny_parent, max_rows=MAX_ROWS, k=32, seed=0)
    b = sketch_model(tiny_unrelated, max_rows=MAX_ROWS, k=32, seed=0)
    assert sketch_distance(a, b) == pytest.approx(sketch_distance(b, a), abs=SYMMETRY_TOL)
    assert 0.0 <= sketch_distance(a, b) <= 2.0


def test_sketch_distance_orders_child_before_unrelated(
    tiny_parent: str, tiny_child_sft: str, tiny_unrelated: str
) -> None:
    """A derived model must sketch closer to its parent than a stranger does.

    Claim: low-false-positive -- this is the ordering the phylogeny index relies
    on when it shortlists candidate parents.
    """
    p = sketch_model(tiny_parent, max_rows=MAX_ROWS, k=32, seed=0)
    c = sketch_model(tiny_child_sft, max_rows=MAX_ROWS, k=32, seed=0)
    u = sketch_model(tiny_unrelated, max_rows=MAX_ROWS, k=32, seed=0)
    assert sketch_distance(p, c) < sketch_distance(p, u)


def test_sketch_meta_records_what_it_read(tiny_parent: str) -> None:
    """Sketch.meta must carry the fields CONTRACT.md lists, from headers alone."""
    s = sketch_model(tiny_parent, max_rows=MAX_ROWS, k=32, seed=0)
    for key in (
        "n_layers",
        "hidden_size",
        "vocab_size",
        "dtypes",
        "architectures",
        "tensor_count",
        "param_count",
        "fused_qkv",
    ):
        assert key in s.meta, f"Sketch.meta is missing {key!r}"
    assert int(s.meta["hidden_size"]) == 128
    assert int(s.meta["vocab_size"]) == 2048
    assert int(s.meta["n_layers"]) == 4
    assert s.meta["fused_qkv"] is False
