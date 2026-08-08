"""Licence propagation must produce evidence, never an accusation.

Claim: direction -- there is no "ancestor" and no "descendant" without a
*directed* edge, so every finding in this file is downstream of the direction
claim. The second half of the file is the guardrail: Stemma may report a
licensing inconsistency with a path and a confidence, and it may not assert that
anyone violated or infringed anything.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from stemma.rights import (
    DISCLAIMER,
    bom_to_spdx,
    build_bom,
    detect_conflicts,
    license_facts_for,
    normalize_license_id,
    propagate,
)
from stemma.types import BOM, Edge, LicenseFacts, Phylogeny, RightsConflict, TransferStats

# --------------------------------------------------------------------------- #
# Thresholds and vocabulary (judgement calls, stated once here)
# --------------------------------------------------------------------------- #

#: Words that would turn evidence into a legal claim. They may appear *only*
#: inside the disclaimer, whose whole job is to say Stemma is not making one.
FORBIDDEN_SUBSTRINGS = ("violat", "infring", "illegal")

ANCESTOR = "someorg/nc-base-model"
MIDDLE = "otherorg/mid-model"
DESCENDANT = "thirdorg/apache-descendant"

#: Edge confidences of the hand-built DAG; their product is the path confidence.
CONF_TOP = 0.90
CONF_BOTTOM = 0.85


@pytest.fixture()
def nc_lineage() -> Phylogeny:
    """cc-by-nc-4.0 ancestor -> intermediate -> apache-2.0 descendant."""
    return Phylogeny(
        nodes=[ANCESTOR, MIDDLE, DESCENDANT],
        edges=[
            Edge(
                parent=ANCESTOR,
                child=MIDDLE,
                confidence=CONF_TOP,
                relation="finetuned",
                evidence=["orphan embedding rows only in the child"],
            ),
            Edge(
                parent=MIDDLE,
                child=DESCENDANT,
                confidence=CONF_BOTTOM,
                relation="quantized",
                evidence=["int8 value lattice present only in the child"],
            ),
        ],
        root_candidates=[ANCESTOR],
        meta={"index_backend": "numpy"},
    )


@pytest.fixture()
def nc_facts() -> Dict[str, LicenseFacts]:
    return {
        ANCESTOR: license_facts_for(ANCESTOR, "cc-by-nc-4.0", source="model_card"),
        MIDDLE: license_facts_for(MIDDLE, "apache-2.0", source="model_card"),
        DESCENDANT: license_facts_for(DESCENDANT, "apache-2.0", source="model_card"),
    }


def _strings(obj: Any) -> List[str]:
    """Every string anywhere in a nested JSON-like structure."""
    out: List[str] = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.append(str(k))
            out.extend(_strings(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_strings(v))
    return out


# --------------------------------------------------------------------------- #
# The licence table itself
# --------------------------------------------------------------------------- #


def test_known_licences_are_recognised() -> None:
    """The permissions table must classify the licences CONTRACT.md lists."""
    assert license_facts_for("x", "cc-by-nc-4.0").commercial_use is False
    assert license_facts_for("x", "apache-2.0").commercial_use is True
    assert license_facts_for("x", "mit").commercial_use is True
    assert license_facts_for("x", "cc-by-sa-4.0").share_alike is True
    assert license_facts_for("x", "gpl-3.0").share_alike is True
    # An unrecognised string must become "unknown permissions", not a guess.
    assert license_facts_for("x", "totally-made-up-1.0").commercial_use is None
    assert normalize_license_id(None) == "unknown"
    assert normalize_license_id("CC-BY-NC 4.0") == "cc-by-nc-4.0"


# --------------------------------------------------------------------------- #
# Conflict detection
# --------------------------------------------------------------------------- #


def test_noncommercial_ancestor_is_reported_with_a_path_and_confidence(
    nc_lineage: Phylogeny, nc_facts: Dict[str, LicenseFacts]
) -> None:
    """The headline rights finding: NC ancestor under a permissive descendant."""
    conflicts = detect_conflicts(nc_lineage, nc_facts)
    hits = [
        c
        for c in conflicts
        if c.kind == "noncommercial_ancestor"
        and c.descendant == DESCENDANT
        and c.ancestor == ANCESTOR
    ]
    assert hits, (
        "no noncommercial_ancestor conflict was raised; got "
        f"{[(c.kind, c.ancestor, c.descendant) for c in conflicts]}"
    )
    c = hits[0]
    assert isinstance(c, RightsConflict)
    assert c.path == [ANCESTOR, MIDDLE, DESCENDANT], f"path was {c.path}"
    assert c.confidence == pytest.approx(CONF_TOP * CONF_BOTTOM, abs=1e-9)
    assert c.severity in {"info", "warning", "high"}
    assert ANCESTOR in c.message and DESCENDANT in c.message
    assert "review" in c.message.lower(), "every finding must ask for human review"


def test_no_conflict_when_the_ancestor_is_permissive(nc_lineage: Phylogeny) -> None:
    """A permissive ancestor must not produce a non-commercial finding.

    Claim: low-false-positive -- the rights layer inherits the project's
    false-positive discipline; inventing licence conflicts would be worse than
    reporting none.
    """
    facts = {
        ANCESTOR: license_facts_for(ANCESTOR, "apache-2.0", source="model_card"),
        MIDDLE: license_facts_for(MIDDLE, "apache-2.0", source="model_card"),
        DESCENDANT: license_facts_for(DESCENDANT, "apache-2.0", source="model_card"),
    }
    conflicts = detect_conflicts(nc_lineage, facts)
    assert not [c for c in conflicts if c.kind == "noncommercial_ancestor"]


def test_propagate_carries_the_strictest_ancestral_term_downstream(
    nc_lineage: Phylogeny, nc_facts: Dict[str, LicenseFacts]
) -> None:
    """Propagation must tighten, never loosen, what a descendant may permit."""
    out = propagate(nc_lineage, nc_facts)
    assert set(out) >= set(nc_lineage.nodes)
    assert out[DESCENDANT].commercial_use is not True, (
        "an apache-2.0 model below a cc-by-nc ancestor must not still read as "
        "unrestricted for commercial use after propagation"
    )
    # The declared facts must not be mutated in place.
    assert nc_facts[DESCENDANT].commercial_use is True


# --------------------------------------------------------------------------- #
# BOM + SPDX serialisation
# --------------------------------------------------------------------------- #


@pytest.fixture()
def nc_bom(nc_lineage: Phylogeny, nc_facts: Dict[str, LicenseFacts]) -> BOM:
    conflicts = detect_conflicts(nc_lineage, nc_facts)
    return build_bom(
        nc_lineage,
        nc_facts,
        conflicts,
        root=DESCENDANT,
        transfer=TransferStats(
            bytes_read=77_100_000, requests=48, seconds=12.5, full_size_bytes=1_976_200_000
        ),
    )


def test_bom_serialises_to_json(nc_bom: BOM) -> None:
    """The BOM must round-trip through JSON with its evidence intact."""
    payload = json.loads(nc_bom.to_json())
    assert payload["root"] == DESCENDANT
    assert payload["version"] == nc_bom.version
    assert len(payload["components"]) == 3
    assert len(payload["relationships"]) == 2
    assert payload["conflicts"], "the BOM dropped the rights evidence"
    assert payload["disclaimer"].strip()

    rel = {(r["parent"], r["child"]): r for r in payload["relationships"]}
    assert rel[(ANCESTOR, MIDDLE)]["confidence"] == pytest.approx(CONF_TOP)
    assert rel[(MIDDLE, DESCENDANT)]["relation"] == "quantized"

    tr = payload["transfer"]
    assert tr["bytes_read"] == 77_100_000
    assert tr["full_size_bytes"] == 1_976_200_000
    assert tr["reduction"] == pytest.approx(1_976_200_000 / 77_100_000)


def test_bom_to_spdx_is_well_formed(nc_bom: BOM) -> None:
    """The SPDX-flavoured export must be complete, linked and JSON-serialisable."""
    doc = bom_to_spdx(nc_bom, created="2026-01-01T00:00:00Z")
    json.dumps(doc)  # must not raise

    assert doc["spdxVersion"] == "SPDX-2.3"
    assert doc["SPDXID"] == "SPDXRef-DOCUMENT"
    assert doc["dataLicense"]
    assert doc["documentNamespace"].startswith("https://")
    assert doc["creationInfo"]["created"] == "2026-01-01T00:00:00Z"
    assert doc["creationInfo"]["creators"]

    ids = {p["SPDXID"] for p in doc["packages"]}
    assert len(doc["packages"]) == 3
    assert len(ids) == 3, "SPDXIDs must be unique"
    for pkg in doc["packages"]:
        assert pkg["name"]
        assert pkg["downloadLocation"]
        assert pkg["licenseDeclared"]
        assert pkg["licenseConcluded"]

    described = doc["documentDescribes"]
    assert described and described[0] in ids

    for rel in doc["relationships"]:
        assert rel["spdxElementId"] in ids | {"SPDXRef-DOCUMENT"}
        assert rel["relatedSpdxElement"] in ids
        assert rel["relationshipType"]

    # A model touched by a conflict must not get a concluded licence.
    by_name = {p["name"]: p for p in doc["packages"]}
    assert by_name[DESCENDANT]["licenseConcluded"] == "NOASSERTION"


# --------------------------------------------------------------------------- #
# The guardrail
# --------------------------------------------------------------------------- #


def test_the_disclaimer_is_present_in_every_output(nc_bom: BOM) -> None:
    """Neither the BOM nor the SPDX export may ship without the disclaimer."""
    assert nc_bom.disclaimer.strip()
    assert "does NOT constitute a legal determination" in nc_bom.disclaimer
    assert "human must review" in nc_bom.disclaimer.lower()
    assert nc_bom.disclaimer in nc_bom.to_json()
    assert DISCLAIMER == nc_bom.disclaimer

    spdx_text = json.dumps(bom_to_spdx(nc_bom, created="2026-01-01T00:00:00Z"))
    assert nc_bom.disclaimer in spdx_text, "the SPDX export dropped the disclaimer"


@pytest.mark.parametrize("form", ["bom", "spdx"])
def test_no_output_asserts_infringement(nc_bom: BOM, form: str) -> None:
    """Outside the disclaimer, the words of legal accusation must not appear.

    Claim: low-false-positive -- Stemma reports statistical evidence about
    weights. Saying "violation" would convert a probabilistic finding into a
    legal claim the method cannot support, which is the failure mode this whole
    project has to avoid.
    """
    if form == "bom":
        blob = nc_bom.to_json()
    else:
        blob = json.dumps(bom_to_spdx(nc_bom, created="2026-01-01T00:00:00Z"))

    # The disclaimer legitimately contains "infringement"; strip every copy of
    # it (and its JSON-escaped form) before searching the remainder.
    stripped = blob.replace(nc_bom.disclaimer, "")
    stripped = stripped.replace(json.dumps(nc_bom.disclaimer)[1:-1], "")

    lowered = stripped.lower()
    for word in FORBIDDEN_SUBSTRINGS:
        assert word not in lowered, (
            f"{form} output contains {word!r} outside the disclaimer; Stemma must "
            "report evidence, not assert a legal conclusion"
        )


def test_conflict_messages_stay_evidential(
    nc_lineage: Phylogeny, nc_facts: Dict[str, LicenseFacts]
) -> None:
    """Each conflict message must hedge and must ask for human review."""
    conflicts = detect_conflicts(nc_lineage, nc_facts)
    assert conflicts
    for c in conflicts:
        low = c.message.lower()
        for word in FORBIDDEN_SUBSTRINGS:
            assert word not in low, f"{c.kind} message contains {word!r}: {c.message}"
        assert "review" in low, f"{c.kind} message does not ask for review: {c.message}"
        assert any(
            hedge in low for hedge in ("evidence", "suggests", "possible", "may ")
        ), f"{c.kind} message is not hedged: {c.message}"
        assert 0.0 <= c.confidence <= 1.0
