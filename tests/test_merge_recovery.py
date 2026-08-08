"""Recovering *which* models were merged, and in what proportion.

Claim: merge-recovery -- a symmetric similarity score can say "these three
models are related". It cannot say "this one is 0.7 of that plus 0.3 of the
other". This file asserts the thing the project claims and the baselines
structurally cannot do: the named parent set and the mixing ratios.

Ground truth comes from ``conftest.linear_merge``, which builds the merged
checkpoint as ``base + 0.7*(A - base) + 0.3*(B - base)`` -- exactly the
task-vector arithmetic the decomposer inverts.
"""

from __future__ import annotations

import numpy as np
import pytest

from stemma.merge_decompose import (
    decompose_merge,
    mixing_mae,
    nnls_l1,
    parent_set_prf,
    task_vectors,
)
from stemma.types import MergeDecomposition

# --------------------------------------------------------------------------- #
# Thresholds (judgement calls, stated once here)
# --------------------------------------------------------------------------- #

#: Required accuracy on the mixing ratios, over the union of predicted and true
#: keys. 0.1 means "0.7/0.3 must not be confused with 0.5/0.5".
MAE_TOL: float = 0.10

#: What "~0" means for a decoy candidate. Below decompose_merge's default
#: support_threshold of 0.05, so a decoy at this level is also *excluded* from
#: the reported parent set, not merely small.
DECOY_TOL: float = 0.05

#: Coordinate budget. The fixtures have ~0.5 M parameters in tensors big enough
#: to qualify; 50 k coordinates is plenty to identify 4 candidates and keeps the
#: suite fast.
COORDS: int = 50_000

#: Fraction of the target the residual must fall below. The merge is exactly
#: linear, so anything but a near-zero residual means the solver missed.
RESIDUAL_TOL: float = 0.05


def _decompose(case, **kw) -> MergeDecomposition:
    return decompose_merge(
        case.merged,
        case.candidates,
        base=case.base,
        coords=COORDS,
        seed=0,
        **kw,
    )


def test_mixing_ratios_are_recovered_within_tolerance(tiny_merged) -> None:
    """0.7 / 0.3 must come back as 0.7 / 0.3, with the decoys at zero."""
    d = _decompose(tiny_merged)
    pred = d.as_dict()

    mae = mixing_mae(pred, tiny_merged.truth)
    assert mae <= MAE_TOL, (
        f"mixing MAE {mae:.4f} exceeds {MAE_TOL}; predicted={pred} truth={tiny_merged.truth}"
    )
    for name, want in tiny_merged.truth.items():
        assert pred.get(name, 0.0) == pytest.approx(want, abs=MAE_TOL * 2)


def test_decoy_candidates_get_approximately_zero_weight(tiny_merged) -> None:
    """Candidates that contributed nothing must be assigned ~0 and dropped."""
    d = _decompose(tiny_merged)
    pred = d.as_dict()
    for decoy in tiny_merged.decoys:
        assert pred.get(decoy, 0.0) <= DECOY_TOL, (
            f"decoy {decoy} received weight {pred.get(decoy)}"
        )
        assert decoy not in d.selected


def test_parent_set_f1_is_perfect(tiny_merged) -> None:
    """The reported parent set must be exactly the two true sources."""
    d = _decompose(tiny_merged)
    precision, recall, f1 = parent_set_prf(d.selected, list(tiny_merged.truth))
    assert f1 == pytest.approx(1.0), (
        f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}; selected={d.selected} "
        f"truth={list(tiny_merged.truth)}"
    )


def test_decomposition_explains_the_child(tiny_merged) -> None:
    """An exactly-linear merge must leave an almost-zero residual and r2 ~ 1."""
    d = _decompose(tiny_merged)
    assert d.residual <= RESIDUAL_TOL, f"residual {d.residual:.4f}"
    assert d.r2 >= 1.0 - RESIDUAL_TOL, f"r2 {d.r2:.4f}"
    assert d.base == tiny_merged.base
    assert d.stats is not None and d.stats.bytes_read > 0


def test_decomposition_is_deterministic(tiny_merged) -> None:
    """Same seed, same answer -- the benchmark numbers have to be reproducible."""
    a = _decompose(tiny_merged)
    b = _decompose(tiny_merged)
    np.testing.assert_allclose(a.coefficients, b.coefficients, rtol=0, atol=0)
    assert a.selected == b.selected


def test_base_can_be_inferred_when_not_supplied(tiny_merged) -> None:
    """With the base present among the candidates, it must be found, not required.

    Claim: merge-recovery -- in the real setting nobody hands you the base
    checkpoint; the decomposer has to identify it from the candidate set.
    """
    candidates = [tiny_merged.base, *tiny_merged.candidates]
    d = decompose_merge(tiny_merged.merged, candidates, coords=COORDS, seed=0)
    assert d.base == tiny_merged.base
    pred = d.as_dict()
    mae = mixing_mae({k: v for k, v in pred.items() if k != tiny_merged.base}, tiny_merged.truth)
    assert mae <= MAE_TOL, f"inferred-base mixing MAE {mae:.4f}; predicted={pred}"


def test_a_plain_child_decomposes_onto_its_single_parent(
    tiny_parent: str, tiny_child_sft: str, tiny_sibling_sft: str
) -> None:
    """A non-merged model must not be reported as a mixture.

    Claim: low-false-positive -- over-reporting merges would fabricate lineage
    edges, so the single-parent case is a control for the merge machinery too.
    """
    d = decompose_merge(
        tiny_child_sft,
        [tiny_child_sft, tiny_sibling_sft],
        base=tiny_parent,
        coords=COORDS,
        seed=0,
    )
    pred = d.as_dict()
    assert pred[tiny_child_sft] > 0.8
    assert pred[tiny_sibling_sft] <= DECOY_TOL * 4
    assert d.selected == [tiny_child_sft]


# --------------------------------------------------------------------------- #
# The pieces the decomposition is built from
# --------------------------------------------------------------------------- #


def test_task_vectors_share_one_coordinate_system(tiny_merged) -> None:
    """T and t_child must be sampled at identical coordinates in every model."""
    T, y, used, stats = task_vectors(
        tiny_merged.base,
        list(tiny_merged.truth),
        tiny_merged.merged,
        coords=COORDS,
        seed=0,
    )
    assert T.shape == (2, y.size)
    assert used, "no shared tensor was selected"
    assert stats.bytes_read > 0

    # The merge is exact in weight space, so it is exact in task-vector space.
    reconstruction = 0.7 * T[0] + 0.3 * T[1]
    rel = float(np.linalg.norm(reconstruction - y) / max(np.linalg.norm(y), 1e-12))
    assert rel < 1e-4, f"task vectors do not reproduce the merge (rel err {rel:.3g})"


def test_nnls_l1_respects_nonnegativity_and_the_simplex() -> None:
    """The solver's constraints must actually hold, not just approximately."""
    rng = np.random.default_rng(0)
    T = rng.standard_normal((3, 500))
    w_true = np.array([0.7, 0.3, 0.0])
    y = w_true @ T

    w = nnls_l1(T, y, l1=0.0, sum_to_one=False, nonneg=True)
    assert np.all(w >= -1e-9)
    np.testing.assert_allclose(w, w_true, atol=1e-6)

    w = nnls_l1(T, y, l1=1e-3, sum_to_one=True, nonneg=True)
    assert np.all(w >= -1e-9)
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-6)


def test_mixing_mae_uses_the_union_of_keys() -> None:
    """A parent that was missed entirely must be penalised, not ignored."""
    assert mixing_mae({"a": 0.7, "b": 0.3}, {"a": 0.7, "b": 0.3}) == pytest.approx(0.0)
    # Missing "b" is treated as 0.0 -> mean(|0|, |0.3|) over the union {a, b}.
    assert mixing_mae({"a": 0.7}, {"a": 0.7, "b": 0.3}) == pytest.approx(0.15)


def test_parent_set_prf_is_a_set_metric() -> None:
    """Sanity-pin the scoring function the headline F1 number comes from."""
    assert parent_set_prf(["a", "b"], ["b", "a"]) == (1.0, 1.0, 1.0)
    p, r, f1 = parent_set_prf(["a", "b", "c"], ["a", "b"])
    assert p == pytest.approx(2 / 3) and r == pytest.approx(1.0)
    assert f1 == pytest.approx(2 * (2 / 3) / (2 / 3 + 1.0))
    assert parent_set_prf([], ["a"]) == (0.0, 0.0, 0.0)
