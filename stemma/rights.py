"""License facts, DAG propagation, conflict evidence and AI-BOM export.

This module is the "so what" layer: once :mod:`stemma.phylogeny` has produced an
oriented, confidence-weighted lineage, the licence terms of every ancestor can
be walked down the graph and any inconsistency surfaced with the *product of the
edge confidences* attached to it.

Claim: direction -- inherited licence obligations only make sense on a DAG whose
edges have a direction; a symmetric similarity score can never tell you which
model inherited from which, and therefore can never produce this report.

Wording policy: everything emitted here is phrased as *evidence + confidence +
human review required*. Stemma never asserts infringement or that a licence has
been violated.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .types import (
    BOM,
    BOM_VERSION,
    Edge,
    LicenseFacts,
    ModelRef,
    Phylogeny,
    RightsConflict,
    Sketch,
    TransferStats,
)
from .utils import get_logger, human_bytes, is_local_path, local_path_of, stable_hash

log = get_logger(__name__)

#: Standing disclaimer reproduced on every artifact this module emits.
DISCLAIMER = BOM.__dataclass_fields__["disclaimer"].default

#: Edge confidence above which we are willing to call a path "confident".
CONFIDENT_EDGE = 0.8


# --------------------------------------------------------------------------- #
# Known-licence table
# --------------------------------------------------------------------------- #

#: ``license id -> facts``. ``commercial_use=None`` means *unknown*, which is
#: deliberately distinct from ``False``: we never guess.
KNOWN_LICENSES: Dict[str, Dict[str, Any]] = {
    "apache-2.0": {
        "name": "Apache License 2.0",
        "spdx": "Apache-2.0",
        "commercial_use": True,
        "share_alike": False,
        "attribution_required": True,
        "restrictions": ["patent-retaliation clause", "state changes to modified files"],
    },
    "mit": {
        "name": "MIT License",
        "spdx": "MIT",
        "commercial_use": True,
        "share_alike": False,
        "attribution_required": True,
        "restrictions": [],
    },
    "bsd-3-clause": {
        "name": "BSD 3-Clause License",
        "spdx": "BSD-3-Clause",
        "commercial_use": True,
        "share_alike": False,
        "attribution_required": True,
        "restrictions": ["no endorsement using contributor names"],
    },
    "llama2": {
        "name": "Llama 2 Community License",
        "spdx": "LicenseRef-Llama-2-Community",
        "commercial_use": True,
        "share_alike": False,
        "attribution_required": True,
        "restrictions": [
            "acceptable use policy applies to derivatives",
            "separate licence required above the stated monthly-active-user threshold",
            "licence terms must be passed to downstream recipients",
        ],
    },
    "llama3": {
        "name": "Llama 3 Community License",
        "spdx": "LicenseRef-Llama-3-Community",
        "commercial_use": True,
        "share_alike": False,
        "attribution_required": True,
        "restrictions": [
            "acceptable use policy applies to derivatives",
            "separate licence required above the stated monthly-active-user threshold",
            "derivative model names are expected to carry the Llama marker",
        ],
    },
    "llama3.1": {
        "name": "Llama 3.1 Community License",
        "spdx": "LicenseRef-Llama-3.1-Community",
        "commercial_use": True,
        "share_alike": False,
        "attribution_required": True,
        "restrictions": [
            "acceptable use policy applies to derivatives",
            "separate licence required above the stated monthly-active-user threshold",
            "derivative model names are expected to carry the Llama marker",
        ],
    },
    "llama3.2": {
        "name": "Llama 3.2 Community License",
        "spdx": "LicenseRef-Llama-3.2-Community",
        "commercial_use": True,
        "share_alike": False,
        "attribution_required": True,
        "restrictions": [
            "acceptable use policy applies to derivatives",
            "separate licence required above the stated monthly-active-user threshold",
            "derivative model names are expected to carry the Llama marker",
        ],
    },
    "gemma": {
        "name": "Gemma Terms of Use",
        "spdx": "LicenseRef-Gemma-Terms",
        "commercial_use": True,
        "share_alike": False,
        "attribution_required": True,
        "restrictions": [
            "prohibited use policy applies to derivatives",
            "use restrictions must be passed to downstream recipients",
        ],
    },
    "tongyi-qianwen": {
        "name": "Tongyi Qianwen (Qwen) License",
        "spdx": "LicenseRef-Tongyi-Qianwen",
        "commercial_use": True,
        "share_alike": False,
        "attribution_required": True,
        "restrictions": [
            "separate licence required above the stated monthly-active-user threshold",
            "licence terms must be passed to downstream recipients",
        ],
    },
    "cc-by-nc-4.0": {
        "name": "Creative Commons Attribution-NonCommercial 4.0",
        "spdx": "CC-BY-NC-4.0",
        "commercial_use": False,
        "share_alike": False,
        "attribution_required": True,
        "restrictions": ["non-commercial use only"],
    },
    "cc-by-nc-sa-4.0": {
        "name": "Creative Commons Attribution-NonCommercial-ShareAlike 4.0",
        "spdx": "CC-BY-NC-SA-4.0",
        "commercial_use": False,
        "share_alike": True,
        "attribution_required": True,
        "restrictions": ["non-commercial use only", "derivatives must use the same licence"],
    },
    "cc-by-sa-4.0": {
        "name": "Creative Commons Attribution-ShareAlike 4.0",
        "spdx": "CC-BY-SA-4.0",
        "commercial_use": True,
        "share_alike": True,
        "attribution_required": True,
        "restrictions": ["derivatives must use the same licence"],
    },
    "cc-by-4.0": {
        "name": "Creative Commons Attribution 4.0",
        "spdx": "CC-BY-4.0",
        "commercial_use": True,
        "share_alike": False,
        "attribution_required": True,
        "restrictions": [],
    },
    "cc0-1.0": {
        "name": "Creative Commons Zero v1.0 Universal",
        "spdx": "CC0-1.0",
        "commercial_use": True,
        "share_alike": False,
        "attribution_required": False,
        "restrictions": [],
    },
    "openrail": {
        "name": "OpenRAIL",
        "spdx": "LicenseRef-OpenRAIL",
        "commercial_use": True,
        "share_alike": True,
        "attribution_required": True,
        "restrictions": [
            "behavioural use restrictions apply",
            "use restrictions must be passed to downstream recipients",
        ],
    },
    "bigscience-openrail-m": {
        "name": "BigScience OpenRAIL-M",
        "spdx": "LicenseRef-BigScience-OpenRAIL-M",
        "commercial_use": True,
        "share_alike": True,
        "attribution_required": True,
        "restrictions": [
            "behavioural use restrictions apply",
            "use restrictions must be passed to downstream recipients",
        ],
    },
    "creativeml-openrail-m": {
        "name": "CreativeML OpenRAIL-M",
        "spdx": "LicenseRef-CreativeML-OpenRAIL-M",
        "commercial_use": True,
        "share_alike": True,
        "attribution_required": True,
        "restrictions": [
            "behavioural use restrictions apply",
            "use restrictions must be passed to downstream recipients",
        ],
    },
    "gpl-3.0": {
        "name": "GNU General Public License v3.0",
        "spdx": "GPL-3.0-only",
        "commercial_use": True,
        "share_alike": True,
        "attribution_required": True,
        "restrictions": ["derivative works must be released under the same licence"],
    },
    "agpl-3.0": {
        "name": "GNU Affero General Public License v3.0",
        "spdx": "AGPL-3.0-only",
        "commercial_use": True,
        "share_alike": True,
        "attribution_required": True,
        "restrictions": [
            "derivative works must be released under the same licence",
            "network use counts as distribution",
        ],
    },
    "unknown": {
        "name": "Unknown / undeclared",
        "spdx": "NOASSERTION",
        "commercial_use": None,
        "share_alike": False,
        "attribution_required": False,
        "restrictions": ["licence not declared or not recognised"],
    },
}

#: Spellings seen in the wild -> canonical key in :data:`KNOWN_LICENSES`.
LICENSE_ALIASES: Dict[str, str] = {
    "apache2": "apache-2.0",
    "apache-2": "apache-2.0",
    "apache": "apache-2.0",
    "apache-license-2.0": "apache-2.0",
    "mit-license": "mit",
    "bsd-3": "bsd-3-clause",
    "bsd3": "bsd-3-clause",
    "bsd": "bsd-3-clause",
    "llama-2": "llama2",
    "llama2-community": "llama2",
    "llama-2-community": "llama2",
    "llama-3": "llama3",
    "llama-3.1": "llama3.1",
    "llama31": "llama3.1",
    "llama-3.2": "llama3.2",
    "llama32": "llama3.2",
    "gemma-terms-of-use": "gemma",
    "google-gemma": "gemma",
    "qwen": "tongyi-qianwen",
    "qwen-license": "tongyi-qianwen",
    "tongyi_qianwen": "tongyi-qianwen",
    "cc-by-nc": "cc-by-nc-4.0",
    "cc-by-nc-sa": "cc-by-nc-sa-4.0",
    "cc-by-sa": "cc-by-sa-4.0",
    "cc-by": "cc-by-4.0",
    "cc0": "cc0-1.0",
    "openrail-m": "openrail",
    "bigscience-bloom-rail-1.0": "bigscience-openrail-m",
    "gpl3": "gpl-3.0",
    "gplv3": "gpl-3.0",
    "gpl-3.0-only": "gpl-3.0",
    "agpl3": "agpl-3.0",
    "agpl-3.0-only": "agpl-3.0",
    "other": "unknown",
    "unlicensed": "unknown",
    "none": "unknown",
    "": "unknown",
}


def normalize_license_id(raw: Optional[str]) -> str:
    """Canonicalise a licence string to a key of :data:`KNOWN_LICENSES`.

    Claim: infra -- model cards spell the same licence a dozen ways; collapsing
    them is what lets the propagation step compare ancestor and descendant terms
    at all.

    Unrecognised (but non-empty) strings are returned lower-cased and unchanged
    so the caller can still show them, while :func:`license_facts_for` will mark
    ``commercial_use=None``.
    """
    if raw is None:
        return "unknown"
    s = str(raw).strip().lower()
    s = s.replace("_", "-").replace(" ", "-")
    s = re.sub(r"^license[:-]", "", s)
    s = re.sub(r"-+", "-", s).strip("-")
    if not s:
        return "unknown"
    if s in KNOWN_LICENSES:
        return s
    if s in LICENSE_ALIASES:
        return LICENSE_ALIASES[s]
    return s


def license_facts_for(model_id: ModelRef, license_id: Optional[str], **overrides: Any) -> LicenseFacts:
    """Build :class:`LicenseFacts` for ``license_id`` from the known table.

    Claim: infra -- one place decides what a licence *permits*, so the conflict
    detector never re-derives permissions and can never disagree with the BOM.
    """
    key = normalize_license_id(license_id)
    entry = KNOWN_LICENSES.get(key)
    if entry is None:
        log.debug("unrecognised licence %r for %s; recording as unknown permissions", license_id, model_id)
        entry = KNOWN_LICENSES["unknown"]
        facts = LicenseFacts(
            model_id=model_id,
            license=key,
            license_name=str(license_id) if license_id else None,
            commercial_use=None,
            share_alike=False,
            attribution_required=False,
            restrictions=[f"licence {key!r} is not in Stemma's known-licence table"],
            source="unknown",
        )
    else:
        facts = LicenseFacts(
            model_id=model_id,
            license=key,
            license_name=entry["name"],
            commercial_use=entry["commercial_use"],
            share_alike=bool(entry["share_alike"]),
            attribution_required=bool(entry["attribution_required"]),
            restrictions=list(entry["restrictions"]),
            source="unknown",
        )
    for k, v in overrides.items():
        if hasattr(facts, k) and v is not None:
            setattr(facts, k, v)
    return facts


# --------------------------------------------------------------------------- #
# Model-card reading
# --------------------------------------------------------------------------- #


def _parse_yaml(text: str) -> Dict[str, Any]:
    """Parse a small YAML mapping, with a hand-rolled fallback."""
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # pragma: no cover - pyyaml ships with hub
        log.debug("pyyaml unavailable/failed (%s); using minimal parser", exc)

    out: Dict[str, Any] = {}
    key: Optional[str] = None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.lstrip().startswith("- ") and key:
            item = line.lstrip()[2:].strip().strip("'\"")
            out.setdefault(key, [])
            if isinstance(out[key], list):
                out[key].append(item)
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*:\s*(.*)$", line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if val in ("", "|", ">"):
            out[key] = []
        elif val.startswith("[") and val.endswith("]"):
            out[key] = [x.strip().strip("'\"") for x in val[1:-1].split(",") if x.strip()]
        else:
            out[key] = val.strip("'\"")
    return out


def read_frontmatter(text: str) -> Dict[str, Any]:
    """Extract the leading ``---`` YAML frontmatter block of a model card.

    Claim: infra -- the licence field of a Hugging Face model card lives here,
    and reading it is a few kilobytes rather than a download.
    """
    if not text:
        return {}
    stripped = text.lstrip("﻿ \t\r\n")
    if not stripped.startswith("---"):
        return {}
    parts = stripped.split("\n")
    if not parts:
        return {}
    body: List[str] = []
    for line in parts[1:]:
        if line.strip() in ("---", "..."):
            break
        body.append(line)
    return _parse_yaml("\n".join(body))


def _as_str_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, (list, tuple, set)):
        return [str(x) for x in v]
    return [str(v)]


def _local_license(ref: ModelRef) -> Tuple[Optional[str], Optional[str], List[str], str, bool]:
    """Return ``(license, license_name, tags, source, gated)`` for a local dir."""
    path = local_path_of(str(ref))
    if path.is_file():
        path = path.parent
    lic: Optional[str] = None
    name: Optional[str] = None
    tags: List[str] = []
    source = "unknown"
    gated = False

    cfg = path / "config.json"
    if cfg.exists():
        try:
            with open(cfg, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                declared = data.get("stemma_license")
                if declared:
                    lic = str(declared)
                    name = str(data.get("stemma_license_name") or "") or None
                    tags = _as_str_list(data.get("tags"))
                    gated = bool(data.get("stemma_gated", False))
                    source = "declared"
        except Exception as exc:
            log.warning("could not read %s (%s)", cfg, exc)

    if lic is None:
        for cand in ("README.md", "readme.md", "MODEL_CARD.md"):
            rd = path / cand
            if not rd.exists():
                continue
            try:
                fm = read_frontmatter(rd.read_text(encoding="utf-8", errors="replace"))
            except Exception as exc:
                log.warning("could not read %s (%s)", rd, exc)
                continue
            if fm:
                lic = fm.get("license")
                lic = str(lic) if lic is not None else None
                name = fm.get("license_name")
                name = str(name) if name is not None else None
                tags = _as_str_list(fm.get("tags"))
                gated = bool(fm.get("extra_gated_prompt") or fm.get("extra_gated_fields"))
                source = "model_card"
            break
    return lic, name, tags, source, gated


def fetch_license_facts(
    ref: ModelRef,
    *,
    token: Optional[str] = None,
    offline: bool = False,
) -> LicenseFacts:
    """Read the licence declaration for one model (card metadata / local files).

    Claim: low-transfer -- licence provenance is recovered from a few kilobytes
    of card metadata, never from the checkpoint, so it adds nothing measurable
    to the byte budget the benchmark reports.

    ``offline=True`` guarantees no network call is made. Anything absent or not
    in :data:`KNOWN_LICENSES` yields ``commercial_use=None`` (unknown) rather
    than a guess.
    """
    ref = str(ref)
    lic: Optional[str] = None
    lic_name: Optional[str] = None
    lic_link: Optional[str] = None
    tags: List[str] = []
    source = "unknown"
    gated = False

    if is_local_path(ref):
        lic, lic_name, tags, source, gated = _local_license(ref)
    elif offline:
        log.debug("offline=True: not contacting the Hub for %s", ref)
    else:
        try:
            from huggingface_hub import HfApi  # lazy: no network at import time

            info = HfApi().model_info(ref, files_metadata=False, token=token)
            card = getattr(info, "cardData", None) or getattr(info, "card_data", None) or {}
            if not isinstance(card, dict):
                try:
                    card = dict(card)
                except Exception:  # pragma: no cover
                    card = {}
            lic = card.get("license") or lic
            lic_name = card.get("license_name") or lic_name
            lic_link = card.get("license_link") or lic_link
            tags = _as_str_list(getattr(info, "tags", None)) or _as_str_list(card.get("tags"))
            g = getattr(info, "gated", False)
            gated = bool(g) and str(g).lower() != "false"
            if lic or lic_name:
                source = "model_card"
        except Exception as exc:
            log.debug("HfApi().model_info(%s) failed: %s", ref, exc)

        try:
            from huggingface_hub import ModelCard  # lazy

            card = ModelCard.load(ref, token=token)
            data = getattr(card, "data", None)
            d = data.to_dict() if hasattr(data, "to_dict") else dict(data or {})
            lic = d.get("license") or lic
            lic_name = d.get("license_name") or lic_name
            lic_link = d.get("license_link") or lic_link
            card_tags = _as_str_list(d.get("tags"))
            tags = sorted(set(tags) | set(card_tags))
            if lic or lic_name:
                source = "model_card"
        except Exception as exc:
            log.debug("ModelCard.load(%s) failed: %s", ref, exc)

    facts = license_facts_for(ref, lic)
    facts.model_id = ref
    if lic_name:
        facts.license_name = str(lic_name)
    facts.raw_tags = sorted({str(t) for t in tags})
    facts.gated = bool(gated) or any(t.lower() in ("gated", "extra-gated") for t in facts.raw_tags)
    facts.source = source if lic else ("declared" if source == "declared" else "unknown")
    if lic is None:
        facts.license = "unknown"
        facts.commercial_use = None
        if "licence not declared or not recognised" not in facts.restrictions:
            facts.restrictions = list(facts.restrictions) + ["no licence declared in the model card"]
    if lic_link:
        facts.restrictions = list(facts.restrictions) + [f"licence text: {lic_link}"]
    if facts.gated:
        facts.restrictions = list(facts.restrictions) + [
            "access is gated; downstream redistribution terms need checking"
        ]
    return facts


# --------------------------------------------------------------------------- #
# Propagation
# --------------------------------------------------------------------------- #


def _copy_facts(f: LicenseFacts) -> LicenseFacts:
    return LicenseFacts(
        model_id=f.model_id,
        license=f.license,
        license_name=f.license_name,
        commercial_use=f.commercial_use,
        share_alike=bool(f.share_alike),
        attribution_required=bool(f.attribution_required),
        gated=bool(f.gated),
        restrictions=list(f.restrictions),
        source=f.source,
        raw_tags=list(f.raw_tags),
    )


def _unknown_facts(node: ModelRef) -> LicenseFacts:
    f = license_facts_for(node, "unknown")
    f.model_id = node
    f.source = "unknown"
    return f


def topological_order(p: Phylogeny) -> List[ModelRef]:
    """Parents-before-children ordering of the DAG (stable, cycle-tolerant).

    Claim: direction -- only an oriented graph has a topological order; this is
    the concrete place where the direction claim buys something a symmetric
    method cannot offer.
    """
    indeg: Dict[ModelRef, int] = {n: 0 for n in p.nodes}
    children: Dict[ModelRef, List[ModelRef]] = {n: [] for n in p.nodes}
    for e in p.edges:
        if e.parent not in indeg or e.child not in indeg:
            continue
        indeg[e.child] += 1
        children[e.parent].append(e.child)
    ready = [n for n in p.nodes if indeg[n] == 0]
    out: List[ModelRef] = []
    while ready:
        n = ready.pop(0)
        out.append(n)
        for c in children.get(n, []):
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
    if len(out) < len(p.nodes):  # residual cycle: append the rest deterministically
        leftover = [n for n in p.nodes if n not in set(out)]
        log.warning("phylogeny still contains a cycle; %d node(s) ordered arbitrarily", len(leftover))
        out.extend(leftover)
    return out


def _strictest_commercial(values: Sequence[Optional[bool]]) -> Optional[bool]:
    """False (forbidden) < None (unknown) < True (permitted)."""
    vals = list(values)
    if any(v is False for v in vals):
        return False
    if any(v is None for v in vals):
        return None
    return True


def propagate(p: Phylogeny, facts: Dict[ModelRef, LicenseFacts]) -> Dict[ModelRef, LicenseFacts]:
    """Push ancestor restrictions down the DAG, returning a NEW facts dict.

    Claim: direction -- licence obligations flow parent -> child and only that
    way; computing the effective terms of a model therefore *requires* the
    oriented lineage Stemma recovers, which is the whole point of the project.

    The input mapping is never mutated. For each node the result reports the
    strictest constraint over the node itself and all of its ancestors.
    """
    effective: Dict[ModelRef, LicenseFacts] = {}
    for n in p.nodes:
        src = facts.get(n)
        effective[n] = _copy_facts(src) if src is not None else _unknown_facts(n)

    for node in topological_order(p):
        cur = effective[node]
        parents = [e for e in p.parents_of(node) if e.parent in effective]
        if not parents:
            continue
        comm = [cur.commercial_use] + [effective[e.parent].commercial_use for e in parents]
        cur.commercial_use = _strictest_commercial(comm)
        cur.share_alike = bool(cur.share_alike) or any(effective[e.parent].share_alike for e in parents)
        cur.attribution_required = bool(cur.attribution_required) or any(
            effective[e.parent].attribution_required for e in parents
        )
        cur.gated = bool(cur.gated) or any(effective[e.parent].gated for e in parents)
        inherited: List[str] = []
        for e in parents:
            pf = effective[e.parent]
            for r in pf.restrictions:
                text = r if r.startswith("inherited from ") else f"inherited from {e.parent}: {r}"
                inherited.append(text)
        merged = list(cur.restrictions)
        for r in inherited:
            if r not in merged:
                merged.append(r)
        cur.restrictions = merged
    return effective


# --------------------------------------------------------------------------- #
# Conflict evidence
# --------------------------------------------------------------------------- #


def _best_ancestor_paths(p: Phylogeny) -> Dict[ModelRef, Dict[ModelRef, Tuple[List[ModelRef], float, float]]]:
    """For each node, the highest-confidence path up to each of its ancestors.

    Returns ``{descendant: {ancestor: (path_ancestor_to_descendant, product, min_edge)}}``.
    """
    order = topological_order(p)
    best: Dict[ModelRef, Dict[ModelRef, Tuple[List[ModelRef], float, float]]] = {n: {} for n in p.nodes}
    for node in order:
        for e in p.parents_of(node):
            par = e.parent
            if par not in best:
                continue
            conf = float(np.clip(e.confidence, 0.0, 1.0))
            direct = ([par, node], conf, conf)
            cur = best[node].get(par)
            if cur is None or direct[1] > cur[1]:
                best[node][par] = direct
            for anc, (path, prod, lo) in best.get(par, {}).items():
                if anc == node:
                    continue  # residual cycle guard
                cand = (list(path) + [node], prod * conf, min(lo, conf))
                prev = best[node].get(anc)
                if prev is None or cand[1] > prev[1]:
                    best[node][anc] = cand
    return best


def _fmt_path(path: Sequence[ModelRef]) -> str:
    return " -> ".join(str(x) for x in path)


def _license_label(f: LicenseFacts) -> str:
    if f.license and f.license != "unknown":
        return f"{f.license_name or f.license} ({f.license})"
    return "no declared licence"


def detect_conflicts(p: Phylogeny, facts: Dict[ModelRef, LicenseFacts]) -> List[RightsConflict]:
    """Report licence inconsistencies along the lineage, with path confidence.

    Claim: direction -- every finding here is of the form "X appears to descend
    from Y"; without a *directed* edge there is no descendant, no ancestor and
    no inconsistency to report. Confidence is the product of the edge
    confidences along the inferred path, which is the confidence-weighting the
    project pitches.

    Wording is evidential by construction: nothing here asserts infringement or
    a licence violation, and every message asks for human review.
    """
    conflicts: List[RightsConflict] = []
    paths = _best_ancestor_paths(p)
    seen: set = set()

    for desc, ancs in paths.items():
        df = facts.get(desc) or _unknown_facts(desc)
        for anc, (path, prod, lo) in sorted(ancs.items(), key=lambda kv: str(kv[0])):
            af = facts.get(anc) or _unknown_facts(anc)
            confident = lo > CONFIDENT_EDGE
            conf = float(np.clip(prod, 0.0, 1.0))
            pretty = _fmt_path(path)

            def emit(kind: str, severity: str, message: str) -> None:
                key = (desc, anc, kind)
                if key in seen:
                    return
                seen.add(key)
                conflicts.append(
                    RightsConflict(
                        descendant=desc,
                        ancestor=anc,
                        kind=kind,
                        severity=severity,
                        message=message,
                        path=list(path),
                        confidence=conf,
                    )
                )

            # (1) non-commercial ancestor under a commercially-licensed child
            if af.commercial_use is False and df.commercial_use is True:
                emit(
                    "noncommercial_ancestor",
                    "high" if confident else "warning",
                    (
                        f"Weight-level evidence suggests {desc} is derived from {anc} "
                        f"(path {pretty}; combined edge confidence {conf:.2f}). "
                        f"{anc} declares {_license_label(af)}, which is recorded as "
                        f"non-commercial, while {desc} declares {_license_label(df)}, "
                        f"which is recorded as permitting commercial use. "
                        "This is evidence of a possible licensing inconsistency, not a "
                        "determination that any licence has been breached. Human review required."
                    ),
                )

            # (2) share-alike ancestor with a non-share-alike descendant
            if af.share_alike and not df.share_alike and df.license and df.license != "unknown":
                emit(
                    "share_alike_broken",
                    "high" if confident else "warning",
                    (
                        f"Weight-level evidence suggests {desc} is derived from {anc} "
                        f"(path {pretty}; combined edge confidence {conf:.2f}). "
                        f"{anc} declares {_license_label(af)}, which is recorded as "
                        f"share-alike, while {desc} declares {_license_label(df)}, which "
                        "is not. Whether the share-alike term reaches model weights is an "
                        "unsettled question; this is evidence for review, not a finding of "
                        "non-compliance. Human review required."
                    ),
                )

            # (3) ancestor whose licence we could not establish
            if af.license in (None, "unknown") or af.commercial_use is None:
                emit(
                    "unknown_ancestor",
                    "warning" if confident else "info",
                    (
                        f"Weight-level evidence suggests {desc} is derived from {anc} "
                        f"(path {pretty}; combined edge confidence {conf:.2f}), but no "
                        f"recognised licence could be read for {anc} "
                        f"({_license_label(af)}). Downstream terms for {desc} therefore "
                        "cannot be established from the model card alone. Human review required."
                    ),
                )

            # (4) gated ancestor
            if af.gated:
                emit(
                    "gated_ancestor",
                    "warning" if confident else "info",
                    (
                        f"Weight-level evidence suggests {desc} is derived from {anc} "
                        f"(path {pretty}; combined edge confidence {conf:.2f}). Access to "
                        f"{anc} is gated, so its redistribution terms may not carry over "
                        f"automatically to {desc}. This is evidence for review, not an "
                        "assertion that any term has been breached. Human review required."
                    ),
                )

    rank = {"high": 0, "warning": 1, "info": 2}
    conflicts.sort(key=lambda c: (rank.get(c.severity, 3), -c.confidence, str(c.descendant), str(c.ancestor)))
    return conflicts


# --------------------------------------------------------------------------- #
# BOM
# --------------------------------------------------------------------------- #


def _sketch_hash(s: Optional[Sketch]) -> Optional[str]:
    if s is None:
        return None
    v = np.nan_to_num(np.asarray(s.vector, dtype=np.float64).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    return stable_hash({"version": s.version, "vector": [round(float(x), 6) for x in v]})


def _source_of(ref: ModelRef) -> str:
    return "local" if is_local_path(str(ref)) else "huggingface"


def _facts_dict(f: LicenseFacts) -> Dict[str, Any]:
    d = asdict(f)
    d["commercial_use"] = None if f.commercial_use is None else bool(f.commercial_use)
    d["commercial_use_label"] = (
        "unknown" if f.commercial_use is None else ("permitted" if f.commercial_use else "not permitted")
    )
    return d


def build_bom(
    p: Phylogeny,
    facts: Dict[ModelRef, LicenseFacts],
    conflicts: Sequence[RightsConflict],
    *,
    root: ModelRef,
    transfer: Optional[TransferStats | Mapping[str, Any]] = None,
    sketches: Optional[Mapping[ModelRef, Sketch]] = None,
) -> BOM:
    """Assemble the AI Bill of Materials for ``root``'s reconstructed lineage.

    Claim: low-transfer -- the BOM carries the byte accounting next to every
    inferred relationship, so a reader can see that the whole lineage was
    recovered from Range reads rather than full checkpoint downloads.
    """
    sketches = sketches or {}
    components: List[Dict[str, Any]] = []
    for n in p.nodes:
        f = facts.get(n) or _unknown_facts(n)
        s = sketches.get(n)
        meta = dict(getattr(s, "meta", {}) or {})
        components.append(
            {
                "id": str(n),
                "name": Path(str(n)).name if _source_of(n) == "local" else str(n),
                "source": _source_of(n),
                "sketch_hash": _sketch_hash(s),
                "sketch_version": getattr(s, "version", None),
                "license": _facts_dict(f),
                "params": meta.get("param_count"),
                "architectures": meta.get("architectures"),
                "n_layers": meta.get("n_layers"),
                "hidden_size": meta.get("hidden_size"),
                "vocab_size": meta.get("vocab_size"),
                "dtypes": meta.get("dtypes"),
                "is_root_candidate": n in set(p.root_candidates),
                "inferred": n != root,
            }
        )

    relationships: List[Dict[str, Any]] = []
    for e in p.edges:
        relationships.append(
            {
                "parent": str(e.parent),
                "child": str(e.child),
                "relation": str(e.relation),
                "confidence": float(e.confidence),
                "weight": None if e.weight is None else float(e.weight),
                "evidence": list(e.evidence),
                "basis": "weight-level statistical evidence (Stemma direction model)",
            }
        )

    conflict_rows: List[Dict[str, Any]] = []
    for c in conflicts or ():
        conflict_rows.append(asdict(c) if is_dataclass(c) else dict(c))

    if isinstance(transfer, TransferStats):
        red = transfer.reduction
        tr: Dict[str, Any] = {
            "bytes_read": int(transfer.bytes_read),
            "bytes_read_human": human_bytes(transfer.bytes_read),
            "requests": int(transfer.requests),
            "seconds": float(transfer.seconds),
            "full_size_bytes": int(transfer.full_size_bytes),
            "full_size_human": human_bytes(transfer.full_size_bytes),
            "cache_hits": int(transfer.cache_hits),
            "reduction": None if not np.isfinite(red) else float(red),
        }
    elif transfer:
        tr = dict(transfer)
    else:
        tr = dict(p.meta.get("transfer", {}) or {})
    tr.setdefault("index_backend", p.meta.get("index_backend"))
    tr.setdefault("seconds_total", p.meta.get("seconds"))

    return BOM(
        root=str(root),
        version=BOM_VERSION,
        generated_by="stemma",
        components=components,
        relationships=relationships,
        conflicts=conflict_rows,
        transfer=tr,
    )


# --------------------------------------------------------------------------- #
# SPDX-flavoured export
# --------------------------------------------------------------------------- #

_SPDX_SAFE = re.compile(r"[^A-Za-z0-9.\-]")


def _spdx_id(prefix: str, name: str, i: int) -> str:
    return f"SPDXRef-{prefix}-{i}-{_SPDX_SAFE.sub('-', str(name))[:60].strip('-') or 'unnamed'}"


def spdx_license_id(license_id: Optional[str]) -> str:
    """Map a Stemma licence key to an SPDX id (or a ``LicenseRef-``/NOASSERTION).

    Claim: infra -- keeps the exported document readable by SPDX tooling while
    being honest that model licences such as Llama or Gemma have no SPDX id.
    """
    key = normalize_license_id(license_id)
    entry = KNOWN_LICENSES.get(key)
    if entry is None:
        return "LicenseRef-" + (_SPDX_SAFE.sub("-", key).strip("-") or "unknown")
    return str(entry.get("spdx") or "NOASSERTION")


def bom_to_spdx(bom: BOM, *, created: Optional[str] = None) -> Dict[str, Any]:
    """Convert a :class:`BOM` into an SPDX-2.3-*flavoured* JSON-ready dict.

    Claim: infra -- provenance is only useful if it leaves Stemma in a format
    other tooling can ingest; this is the interchange form of the direction and
    merge findings, with each relationship's confidence carried in an annotation.

    The output is clearly labelled as SPDX-flavoured: it follows the SPDX 2.3
    shape but is *not* validated against the official schema, and it never
    concludes a licence that Stemma could not read.
    """
    stamp = created or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ns_key = stable_hash({"root": bom.root, "components": [c.get("id") for c in bom.components]})

    conflicted = {str(c.get("descendant")) for c in bom.conflicts} | {
        str(c.get("ancestor")) for c in bom.conflicts
    }

    id_by_model: Dict[str, str] = {}
    packages: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []

    for i, comp in enumerate(bom.components):
        mid = str(comp.get("id"))
        spdx_ref = _spdx_id("Package", comp.get("name") or mid, i)
        id_by_model[mid] = spdx_ref
        lic = (comp.get("license") or {})
        declared = spdx_license_id(lic.get("license"))
        # We only "conclude" a licence we actually read and that no conflict touches.
        if declared == "NOASSERTION" or lic.get("source") == "unknown" or mid in conflicted:
            concluded = "NOASSERTION"
        else:
            concluded = declared
        download = (
            f"https://huggingface.co/{mid}" if comp.get("source") == "huggingface" else "NOASSERTION"
        )
        packages.append(
            {
                "SPDXID": spdx_ref,
                "name": str(comp.get("name") or mid),
                "downloadLocation": download,
                "filesAnalyzed": False,
                "licenseDeclared": declared,
                "licenseConcluded": concluded,
                "copyrightText": "NOASSERTION",
                "supplier": "NOASSERTION",
                "versionInfo": comp.get("sketch_hash") or "NOASSERTION",
                "primaryPackagePurpose": "MODEL",
                "comment": (
                    f"stemma component; source={comp.get('source')}; "
                    f"license_source={lic.get('source')}; "
                    f"commercial_use={lic.get('commercial_use_label')}; "
                    f"params={comp.get('params')}. "
                    "licenseConcluded is NOASSERTION wherever Stemma could not read a "
                    "licence or found evidence needing review."
                ),
                "externalRefs": [],
            }
        )
        annotations.append(
            {
                "SPDXID": spdx_ref,
                "annotationDate": stamp,
                "annotationType": "OTHER",
                "annotator": "Tool: stemma",
                "comment": json.dumps(
                    {
                        "stemma": "component",
                        "model_id": mid,
                        "sketch_hash": comp.get("sketch_hash"),
                        "license": lic.get("license"),
                        "share_alike": lic.get("share_alike"),
                        "attribution_required": lic.get("attribution_required"),
                        "gated": lic.get("gated"),
                        "restrictions": lic.get("restrictions", []),
                    },
                    ensure_ascii=False,
                ),
            }
        )

    relationships: List[Dict[str, Any]] = []
    root_ref = id_by_model.get(str(bom.root))
    if root_ref:
        relationships.append(
            {
                "spdxElementId": "SPDXRef-DOCUMENT",
                "relationshipType": "DESCRIBES",
                "relatedSpdxElement": root_ref,
            }
        )
    for rel in bom.relationships:
        child = id_by_model.get(str(rel.get("child")))
        parent = id_by_model.get(str(rel.get("parent")))
        if not child or not parent:
            continue
        rtype = "GENERATED_FROM" if str(rel.get("relation")) == "merge" else "DESCENDANT_OF"
        conf = float(rel.get("confidence", 0.0) or 0.0)
        weight = rel.get("weight")
        relationships.append(
            {
                "spdxElementId": child,
                "relationshipType": rtype,
                "relatedSpdxElement": parent,
                "comment": (
                    f"stemma inferred relation={rel.get('relation')} "
                    f"confidence={conf:.3f}"
                    + (f" mixing_weight={float(weight):.3f}" if weight is not None else "")
                    + ". Statistical evidence from weights; human review required."
                ),
            }
        )
        annotations.append(
            {
                "SPDXID": child,
                "annotationDate": stamp,
                "annotationType": "OTHER",
                "annotator": "Tool: stemma",
                "comment": json.dumps(
                    {
                        "stemma": "relationship",
                        "parent": rel.get("parent"),
                        "child": rel.get("child"),
                        "relation": rel.get("relation"),
                        "confidence": conf,
                        "mixing_weight": None if weight is None else float(weight),
                        "evidence": list(rel.get("evidence", []))[:8],
                    },
                    ensure_ascii=False,
                ),
            }
        )

    for j, c in enumerate(bom.conflicts):
        target = id_by_model.get(str(c.get("descendant")), "SPDXRef-DOCUMENT")
        annotations.append(
            {
                "SPDXID": target,
                "annotationDate": stamp,
                "annotationType": "REVIEW",
                "annotator": "Tool: stemma",
                "comment": json.dumps(
                    {
                        "stemma": "rights_evidence",
                        "index": j,
                        "kind": c.get("kind"),
                        "severity": c.get("severity"),
                        "confidence": c.get("confidence"),
                        "path": c.get("path", []),
                        "message": c.get("message"),
                        "note": "evidence only; not a determination of licence compliance",
                    },
                    ensure_ascii=False,
                ),
            }
        )

    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"stemma-ai-bom-{_SPDX_SAFE.sub('-', str(bom.root))}",
        "documentNamespace": f"https://stemma.invalid/spdx/{ns_key}",
        "documentDescribes": [root_ref] if root_ref else [],
        "creationInfo": {
            "created": stamp,
            "creators": [f"Tool: stemma-{bom.version}"],
            "licenseListVersion": "3.21",
            "comment": (
                "SPDX-flavoured document produced by Stemma. It follows the SPDX 2.3 "
                "shape for interchange but is NOT validated against the official SPDX "
                "schema, and non-SPDX model licences are emitted as LicenseRef- ids."
            ),
        },
        "packages": packages,
        "relationships": relationships,
        "annotations": annotations,
        "comment": (
            "SPDX-flavoured AI Bill of Materials. Relationships are statistical "
            "inferences from model weights with per-edge confidences, not verified "
            "provenance records. " + str(bom.disclaimer)
        ),
        "stemma": {
            "version": bom.version,
            "generated_by": bom.generated_by,
            "root": bom.root,
            "transfer": bom.transfer,
            "n_conflicts": len(bom.conflicts),
            "disclaimer": bom.disclaimer,
            "spdx_flavoured": True,
        },
    }
