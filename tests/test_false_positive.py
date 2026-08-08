"""Unrelated models must not look related, even when they share an architecture.

Claim: low-false-positive -- this is the required test the project spec names
first, and the hard case is deliberate: ``tiny_unrelated`` has the *same shapes*,
the *same generator* and the *same statistics* as ``tiny_parent`` and differs
only in its random seed. Any fingerprint that keys on architecture rather than
on the weights themselves fails here.
"""

from __future__ import annotations

import inspect

import pytest

from stemma.direction import relatedness_score
from stemma.phylogeny import build_phylogeny, find_candidate_parents

# --------------------------------------------------------------------------- #
# Thresholds (judgement calls, stated once here)
# --------------------------------------------------------------------------- #

#: The margin a derived model must beat an unrelated one by. 0.25 on a [0, 1]
#: score is a quarter of the whole range: not a hair's breadth, and not so wide
#: that only a degenerate scorer could pass.
FP_MARGIN: float = 0.25

#: Stemma's *documented operating point*: the relatedness below which
#: ``build_phylogeny`` refuses to draw any edge at all. Read from the shipped
#: default so this test tracks the threshold the tool actually ships with,
#: rather than a number copied into the test and left to drift.
OPERATING_THRESHOLD: float = float(
    inspect.signature(build_phylogeny).parameters["relatedness_threshold"].default
)

#: Tolerance for the symmetry check. relatedness_score is symmetric by contract.
SYMMETRY_TOL: float = 1e-6


@pytest.fixture(scope="module")
def scores(request):
    """Cache relatedness scores; each one reads real tensors off disk."""
    cache = {}

    def get(a: str, b: str) -> float:
        key = (a, b)
        if key not in cache:
            cache[key] = float(relatedness_score(a, b))
        return cache[key]

    return get


def test_unrelated_same_shape_model_scores_far_below_a_real_child(
    scores, tiny_parent: str, tiny_child_sft: str, tiny_unrelated: str
) -> None:
    """The required test: a stranger with identical shapes must not pass as kin."""
    related = scores(tiny_parent, tiny_child_sft)
    unrelated = scores(tiny_parent, tiny_unrelated)

    assert unrelated < related, (
        f"an unrelated same-shape model scored {unrelated:.4f}, at or above the "
        f"real child's {related:.4f}"
    )
    assert related - unrelated >= FP_MARGIN, (
        f"margin {related - unrelated:.4f} is under {FP_MARGIN}: "
        f"child={related:.4f} unrelated={unrelated:.4f}"
    )


def test_unrelated_same_shape_model_falls_below_the_operating_threshold(
    scores, tiny_parent: str, tiny_unrelated: str
) -> None:
    """The control must land below the threshold at which Stemma draws edges."""
    unrelated = scores(tiny_parent, tiny_unrelated)
    assert unrelated < OPERATING_THRESHOLD, (
        f"same-shape/different-seed control scored {unrelated:.4f}, at or above "
        f"the shipped operating threshold {OPERATING_THRESHOLD}; at this setting "
        "Stemma would invent a lineage edge between two independent models"
    )


def test_unrelated_architecture_scores_low_too(
    scores, tiny_parent: str, tiny_unrelated_arch: str
) -> None:
    """The easy control, kept because a regression here would be silent."""
    unrelated = scores(tiny_parent, tiny_unrelated_arch)
    assert unrelated < OPERATING_THRESHOLD


@pytest.mark.parametrize(
    "child_fixture",
    ["tiny_child_sft", "tiny_child_int8", "tiny_child_pruned", "tiny_child_vocab"],
)
def test_every_true_child_clears_the_operating_threshold(
    scores, request, tiny_parent: str, child_fixture: str
) -> None:
    """The other side of the coin: real descendants must survive the filter.

    A threshold that rejects unrelated models by rejecting *everything* would
    pass the false-positive test and be useless, so it is pinned from both ends.
    """
    child = request.getfixturevalue(child_fixture)
    score = scores(tiny_parent, child)
    assert score >= OPERATING_THRESHOLD, (
        f"{child_fixture} scored {score:.4f}, below the operating threshold "
        f"{OPERATING_THRESHOLD}; Stemma would miss a real derivation"
    )


def test_relatedness_score_is_symmetric_and_bounded(
    scores, tiny_parent: str, tiny_child_int8: str, tiny_unrelated: str
) -> None:
    """relatedness_score is a *symmetric* [0, 1] statistic; direction lives elsewhere."""
    for a, b in ((tiny_parent, tiny_child_int8), (tiny_parent, tiny_unrelated)):
        ab = scores(a, b)
        ba = scores(b, a)
        assert ab == pytest.approx(ba, abs=SYMMETRY_TOL), (
            f"relatedness_score is not symmetric: {ab} vs {ba}"
        )
        assert 0.0 <= ab <= 1.0


def test_a_model_is_maximally_related_to_itself(scores, tiny_parent: str) -> None:
    """Self-relatedness must top the scale, or the score is not calibrated."""
    assert scores(tiny_parent, tiny_parent) >= 0.99


def test_candidate_retrieval_does_not_shortlist_a_stranger(
    tiny_parent: str, tiny_child_sft: str, tiny_unrelated: str, tiny_unrelated_arch: str
) -> None:
    """End-to-end: the phylogeny builder must not draw an edge to an unrelated model.

    Claim: low-false-positive -- the pairwise score is only useful if the graph
    layer inherits its discipline, so this exercises the shipped default path.
    """
    p = build_phylogeny([tiny_parent, tiny_child_sft, tiny_unrelated, tiny_unrelated_arch])
    touching_stranger = [
        (e.parent, e.child)
        for e in p.edges
        if tiny_unrelated in (e.parent, e.child) or tiny_unrelated_arch in (e.parent, e.child)
    ]
    assert not touching_stranger, (
        f"phylogeny linked an unrelated model: {touching_stranger}"
    )


def test_find_candidate_parents_signature_is_available() -> None:
    """The retrieval entry point the FPR benchmark axis uses must exist."""
    params = inspect.signature(find_candidate_parents).parameters
    assert "max_distance" in params
