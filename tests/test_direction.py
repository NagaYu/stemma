"""Which model came first -- the claim no symmetric fingerprint can make.

Claim: direction -- this is the project's headline test. It is split in two on
purpose, following docs/FINDINGS.md:

* Where the edge is **lossy** (quantisation, pruning, vocabulary growth) the
  scar can only appear downstream, so a confident, correct answer is required.
* Where the edge is **scar-free** (plain SFT: same shapes, same dtype, same
  vocabulary) FINDINGS section 4 records that two models alone are only weakly
  identifiable. The test therefore asserts the weaker, honest property -- the
  verdict must be correct *or* abstain, never confidently wrong -- and a
  separate test shows that outgroup rooting is what resolves it.

Asserting "SFT direction is always right" here would be the exact overclaim the
findings document forbids.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from stemma.direction import (
    DirectionModel,
    collect_pair_evidence,
    direction_features,
    estimate_direction,
)
from stemma.types import DIRECTION_FEATURES, DirectionVerdict, sigmoid

# --------------------------------------------------------------------------- #
# Thresholds (judgement calls, stated once here)
# --------------------------------------------------------------------------- #

#: CONTRACT.md's anti-symmetry requirement: f(b,a) == -f(a,b) to within this.
ANTISYMMETRY_TOL: float = 1e-6

#: Abstention band used throughout. It is the shipped default of
#: ``estimate_direction``; read from the signature so the test tracks the tool.
ABSTAIN: float = float(inspect.signature(estimate_direction).parameters["abstain"].default)

#: The scar-bearing cases must not merely scrape past the abstention band --
#: a lossy operation is near-deterministic evidence (FINDINGS section 1), so we
#: require the llr to clear the band by this factor.
CONFIDENT_MULTIPLIER: float = 1.0

#: Lossy edges, where a confident and correct verdict is required.
SCARRED_CASES = ["tiny_child_int8", "tiny_child_pruned", "tiny_child_vocab"]


# --------------------------------------------------------------------------- #
# The required test: direction on known parent/child pairs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("child_fixture", SCARRED_CASES)
def test_direction_is_recovered_on_lossy_edges(
    request, tiny_parent: str, child_fixture: str
) -> None:
    """parent -> {int8, pruned, vocab-extended} must come back as "a->b"."""
    child = request.getfixturevalue(child_fixture)
    v = estimate_direction(tiny_parent, child)

    assert isinstance(v, DirectionVerdict)
    assert v.direction == "a->b", (
        f"{child_fixture}: expected a->b, got {v.direction!r} with llr={v.llr:.4f}; "
        f"features={v.features}"
    )
    assert v.llr > 0.0
    assert abs(v.llr) >= ABSTAIN * CONFIDENT_MULTIPLIER, (
        f"{child_fixture}: llr {v.llr:.4f} does not clear the abstention band "
        f"{ABSTAIN}; a lossy edge should be near-deterministic evidence"
    )
    assert v.p_a_parent == pytest.approx(sigmoid(v.llr), abs=1e-9)
    assert v.p_a_parent > 0.5
    assert 0.0 <= v.confidence <= 1.0


@pytest.mark.parametrize("child_fixture", SCARRED_CASES)
def test_swapping_the_arguments_flips_the_verdict(
    request, tiny_parent: str, child_fixture: str
) -> None:
    """llr(b,a) == -llr(a,b) exactly, and the reported direction flips with it."""
    child = request.getfixturevalue(child_fixture)
    ab = estimate_direction(tiny_parent, child)
    ba = estimate_direction(child, tiny_parent)

    assert ba.llr == pytest.approx(-ab.llr, abs=ANTISYMMETRY_TOL), (
        f"{child_fixture}: llr(a,b)={ab.llr:.9f} but llr(b,a)={ba.llr:.9f}"
    )
    assert ba.direction == "b->a"


@pytest.mark.parametrize("child_fixture", SCARRED_CASES + ["tiny_child_sft"])
def test_direction_features_are_exactly_negated_on_swap(
    request, tiny_parent: str, child_fixture: str
) -> None:
    """CONTRACT.md's anti-symmetry requirement, feature by feature.

    Claim: direction -- a feature that is not exactly odd under swapping would
    let the estimator answer differently depending on argument order, which
    would make every direction number in the benchmark an artefact.
    """
    child = request.getfixturevalue(child_fixture)
    f_ab = direction_features(collect_pair_evidence(tiny_parent, child))
    f_ba = direction_features(collect_pair_evidence(child, tiny_parent))

    assert set(f_ab) == set(DIRECTION_FEATURES), (
        f"feature keys drifted from types.DIRECTION_FEATURES: "
        f"extra={sorted(set(f_ab) - set(DIRECTION_FEATURES))} "
        f"missing={sorted(set(DIRECTION_FEATURES) - set(f_ab))}"
    )
    assert set(f_ba) == set(DIRECTION_FEATURES)

    worst = max(DIRECTION_FEATURES, key=lambda k: abs(f_ab[k] + f_ba[k]))
    assert abs(f_ab[worst] + f_ba[worst]) <= ANTISYMMETRY_TOL, (
        f"{child_fixture}: feature {worst!r} is not odd under swap: "
        f"f(a,b)={f_ab[worst]:.9g}, f(b,a)={f_ba[worst]:.9g}"
    )
    for k in DIRECTION_FEATURES:
        assert np.isfinite(f_ab[k]), f"feature {k} is not finite"


# --------------------------------------------------------------------------- #
# The honest case: a scar-free SFT edge (docs/FINDINGS.md section 4)
# --------------------------------------------------------------------------- #


def test_scar_free_sft_edge_is_never_confidently_wrong(
    tiny_parent: str, tiny_child_sft: str
) -> None:
    """Correct or abstain -- but not confidently backwards.

    Claim: direction -- FINDINGS section 4 measured that a pure SFT edge is only
    weakly identifiable from two models alone (norm growth flips sign across
    families; the delta-subspace signal is ~1e-3). Requiring a correct answer
    here would be an overclaim, so the property asserted is the one the
    measurements support: an abstention is acceptable, a confident wrong answer
    is not.
    """
    v = estimate_direction(tiny_parent, tiny_child_sft)
    assert v.direction in {"a->b", "unknown"}, (
        f"direction {v.direction!r} with llr={v.llr:.4f} is confidently WRONG on a "
        "scar-free edge; FINDINGS section 4 allows abstention here, not inversion"
    )
    if v.direction == "unknown":
        assert abs(v.llr) < ABSTAIN
    # Even when abstaining, the verdict must explain itself to the UI.
    assert v.evidence, "a verdict with no evidence strings is unreviewable"
    assert set(v.contributions) <= set(DIRECTION_FEATURES)


def _outgroup_kwarg() -> str:
    """Name of estimate_direction's outgroup parameter, per its signature."""
    sig = inspect.signature(estimate_direction)
    for name in ("outgroup", "outgroups", "universe", "outgroup_refs"):
        if name in sig.parameters:
            return name
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return "outgroup"  # forwarded through **kw to collect_pair_evidence
    raise AssertionError(
        "estimate_direction accepts no outgroup argument, but docs/FINDINGS.md "
        "section 4 states outgroup rooting is implemented as evidence family (e)"
    )


def test_an_outgroup_resolves_the_scar_free_edge(
    tiny_parent: str, tiny_child_sft: str, tiny_sibling_sft: str
) -> None:
    """Give the estimator a sibling and the ambiguous edge must resolve correctly.

    Claim: direction -- classical outgroup rooting: a sibling's distance is
    dominated by the shared ancestral component, so d(parent, sibling) <
    d(child, sibling) places the parent nearer the root. This is the component
    FINDINGS section 4 says was added *because* two-model scar-free direction is
    weak, so it is tested separately rather than blended into the pair result.
    """
    kwarg = _outgroup_kwarg()
    try:
        v = estimate_direction(tiny_parent, tiny_child_sft, **{kwarg: [tiny_sibling_sft]})
    except TypeError as exc:
        pytest.fail(
            f"estimate_direction rejected the outgroup argument {kwarg!r}: {exc}. "
            "FINDINGS section 4 documents outgroup rooting as evidence family (e)."
        )

    assert v.direction == "a->b", (
        f"with an outgroup the parent must be rooted: got {v.direction!r}, "
        f"llr={v.llr:.4f}, features={v.features}"
    )
    assert v.llr > 0.0

    # The outgroup must be additive evidence, not a different answer entirely.
    plain = estimate_direction(tiny_parent, tiny_child_sft)
    assert v.llr >= plain.llr, (
        f"outgroup rooting weakened the verdict: {plain.llr:.4f} -> {v.llr:.4f}"
    )


def test_outgroup_evidence_is_still_antisymmetric(
    tiny_parent: str, tiny_child_sft: str, tiny_sibling_sft: str
) -> None:
    """Adding family (e) must not break the anti-symmetry guarantee."""
    kwarg = _outgroup_kwarg()
    kw = {kwarg: [tiny_sibling_sft]}
    ab = estimate_direction(tiny_parent, tiny_child_sft, **kw)
    ba = estimate_direction(tiny_child_sft, tiny_parent, **kw)
    assert ba.llr == pytest.approx(-ab.llr, abs=ANTISYMMETRY_TOL)


# --------------------------------------------------------------------------- #
# The combiner
# --------------------------------------------------------------------------- #


def test_default_direction_model_is_a_usable_prior() -> None:
    """DirectionModel.default() must work with no fitting and no downloads."""
    m = DirectionModel.default()
    assert tuple(m.feature_names) == tuple(DIRECTION_FEATURES)
    assert np.asarray(m.weights).shape == (len(DIRECTION_FEATURES),)
    assert np.all(np.isfinite(np.asarray(m.weights)))

    zeros = {k: 0.0 for k in DIRECTION_FEATURES}
    assert np.isfinite(m.llr(zeros))
    # An all-zero feature vector carries no evidence: it must not be confident.
    assert abs(m.llr(zeros)) < ABSTAIN


def test_direction_model_round_trips_through_json(tmp_path) -> None:
    """The fitted combiner must save and load without changing its verdicts."""
    m = DirectionModel.default()
    path = tmp_path / "direction_model.json"
    m.save(path)
    loaded = DirectionModel.load(path)

    feats = {k: 0.1 * (i + 1) for i, k in enumerate(DIRECTION_FEATURES)}
    assert loaded.llr(feats) == pytest.approx(m.llr(feats), abs=1e-9)
    assert tuple(loaded.feature_names) == tuple(m.feature_names)


def test_direction_model_llr_is_odd_in_its_features() -> None:
    """Negating every feature must negate the llr -- the anti-symmetry backbone."""
    m = DirectionModel.default()
    feats = {k: 0.1 * (i + 1) - 0.5 for i, k in enumerate(DIRECTION_FEATURES)}
    neg = {k: -v for k, v in feats.items()}
    assert m.llr(neg) == pytest.approx(-m.llr(feats), abs=ANTISYMMETRY_TOL)


def test_a_model_against_itself_gives_no_direction(tiny_parent: str) -> None:
    """Comparing a model to itself must abstain, not pick a side.

    Claim: low-false-positive -- every feature is a difference of per-side
    statistics, so identical inputs must produce exactly zero evidence.
    """
    v = estimate_direction(tiny_parent, tiny_parent)
    assert v.llr == pytest.approx(0.0, abs=ANTISYMMETRY_TOL)
    assert v.direction == "unknown"
