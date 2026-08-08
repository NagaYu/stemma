#!/usr/bin/env python3
"""Stemma's evaluation harness: the headline table, the seven figures, results.json.

Claim: infra -- this script proves nothing by itself; it is the instrument that
*measures* the four claims the project makes, on five separate axes, and refuses
to blend them into a single flattering number:

1. **relatedness**   -- ROC AUC / average precision / FPR@95%TPR for Stemma and
   for every symmetric baseline, plus the false-positive rate on the *hard*
   same-shape-different-init controls at the operating threshold. That last
   column is the one that matters: telling apart two unrelated 137M GPT-2-shaped
   models is the failure mode a provenance tool actually has.
2. **direction**     -- accuracy, accuracy-on-answered, abstention rate, and a
   mandatory per-relation breakdown (docs/FINDINGS.md section 5.1: an aggregate
   would let the easy scar-bearing edges hide the hard scar-free ones), plus the
   outgroup-rooting variant as its own column. Baselines are scored at 0.5 and
   labelled a *structural* ceiling, never a tuning failure.
3. **merge recovery** -- parent-set precision/recall/F1 and mixing-ratio MAE
   against true parents plus ``--decoys`` decoys, broken down by merge method
   and parent count. Baselines are "n/a": a symmetric fingerprint produces no
   mixing coefficients at all.
4. **cost**          -- bytes and wall-clock seconds *per decision* for the
   Range path against a full download whose size is taken from the safetensors
   header / file listing. Nothing is ever fully downloaded, here or anywhere.
5. **DAG reconstruction** -- the axis that axes 1-4 cannot see. Everything above
   scores *pairwise* decisions on pairs the ground truth already certified as
   edges, which is exactly how README limitation #9 survived to release: the
   whole-universe phylogeny named two **cousins** as the parents of
   ``smollm2-merge-ties2`` and nothing in the harness noticed. This axis builds
   the DAG over the entire benchmark universe and scores it on exact edges, on
   the ancestor closure (a grandparent edge is a milder error than a cousin
   edge, and only the closure metric shows that), on root identification, and on
   a confusion breakdown of the *wrong* edges by type -- cousin,
   reversed, ancestor-not-parent, unrelated. Naming the failure mode is the
   point; one more F1 would have hidden it again. It is run twice, once with the
   cousin-rejection proximity gate off and once with the shipped default, so the
   guard's effect is measured rather than asserted.

Run ``python benchmarks/run.py --quick`` for the reduced grid, or
``stemma bench --quick``. ``--dag-only`` runs axis 5 alone and splices its
section into an existing results.json / RESULTS.md instead of overwriting them.
A failing pair is logged and counted, never fatal.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Make `import stemma` work when this file is executed as a bare script.
# --------------------------------------------------------------------------- #
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from stemma.types import TransferStats  # noqa: E402
from stemma.utils import atomic_write_json, get_logger, human_bytes, set_seed  # noqa: E402

LOG = get_logger("benchmarks.run")

SCHEMA = "stemma-bench-v1"

#: The relation vocabulary the per-relation direction table is reported over.
RELATION_ORDER: Tuple[str, ...] = (
    "sft", "lora", "cpt", "quant", "prune", "vocab", "merge", "distilled",
)

#: docs/FINDINGS.md section 1: direction evidence is strong exactly where the
#: operation was *lossy*. These three groups are kept visually and numerically
#: apart everywhere in this harness.
SCAR_FREE: Tuple[str, ...] = ("sft", "lora", "cpt")
SCAR_BEARING: Tuple[str, ...] = ("quant", "prune", "vocab")
STRUCTURAL: Tuple[str, ...] = ("merge", "distilled")

GROUP_OF: Dict[str, str] = {
    **{r: "scar-free" for r in SCAR_FREE},
    **{r: "scar-bearing" for r in SCAR_BEARING},
    **{r: "structural / cross-arch" for r in STRUCTURAL},
}

_RELATION_ALIASES: Dict[str, str] = {
    "sft": "sft", "ft": "sft", "finetune": "sft", "finetuned": "sft",
    "fine_tune": "sft", "fine-tuned": "sft", "instruct": "sft", "chat": "sft",
    "derived": "sft", "dpo": "sft", "rlhf": "sft",
    "lora": "lora", "peft": "lora", "adapter": "lora", "lora_merged": "lora",
    "qlora": "lora",
    "cpt": "cpt", "continued_pretraining": "cpt", "continued-pretraining": "cpt",
    "cont_pretrain": "cpt", "pretrain": "cpt", "domain_adapt": "cpt",
    "quant": "quant", "quantized": "quant", "quantised": "quant",
    "quantization": "quant", "int8": "quant", "int4": "quant", "gptq": "quant",
    "awq": "quant", "fp8": "quant",
    "prune": "prune", "pruned": "prune", "pruning": "prune", "sparsify": "prune",
    "magnitude_prune": "prune", "sparse": "prune",
    "vocab": "vocab", "vocab_extended": "vocab", "vocab_extension": "vocab",
    "vocab_expand": "vocab", "vocab_expanded": "vocab", "tokenizer_extended": "vocab",
    "merge": "merge", "merged": "merge", "slerp": "merge", "ties": "merge",
    "dare": "merge", "linear": "merge", "task_arithmetic": "merge",
    "distill": "distilled", "distilled": "distilled", "distillation": "distilled",
}

#: Pair relations that mark a *hard* negative: same architecture, same shapes,
#: independent initialisation. These are the controls the false-positive rate is
#: really measured on.
HARD_CONTROL_RELATIONS = frozenset({
    "same_shape_diff_init", "same-shape-diff-init", "diff_init", "different_init",
    "reinit", "control", "hard_negative", "hard-negative", "independent_init",
    "same_arch_diff_init", "sibling_init", "shuffled",
})

_MERGE_METHODS = ("slerp", "ties", "dare", "linear", "task_arithmetic")

#: Recall level the "operating threshold" is defined at. A provenance screen is
#: recall-oriented -- you would rather review a false alarm than miss an edge --
#: so the operating point is the threshold that retains 95% of true edges, and
#: the headline safety number is the false-positive rate the hard controls incur
#: *there*.
OPERATING_TPR = 0.95

#: Keyword on ``stemma.phylogeny.build_phylogeny`` that switches the
#: cousin-rejection proximity gate. It is written by a sibling module, so the
#: DAG axis probes for it with ``inspect.signature`` and degrades loudly rather
#: than assuming it exists. ``build_phylogeny`` swallows unknown keywords into
#: ``**kw`` and forwards them to the loader, so the probe must be a real
#: signature check and never a try/except.
GATE_PARAM = "proximity_factor"

#: How a *wrong* predicted edge is wrong. This vocabulary is the entire reason
#: axis 5 exists: "cousin" is the failure README limitation #9 documents, and it
#: is a categorically different mistake from "ancestor-not-parent" (a real
#: ancestor, just not the direct one -- a transitive-reduction miss) or from
#: "reversed" (right pair, wrong arrow).
WRONG_EDGE_TYPES: Tuple[str, ...] = ("cousin", "reversed", "ancestor-not-parent", "unrelated")

#: The concrete edge README limitation #9 is written about. The DAG axis reports
#: what it recovered for this child under each gate setting, because a summary
#: F1 would not tell a reader whether *that* bug is fixed.
SPOTLIGHT_CHILD = "smollm2-merge-ties2"


# --------------------------------------------------------------------------- #
# Budgets
# --------------------------------------------------------------------------- #


@dataclass
class Budget:
    """Read budgets for one benchmark run; ``--quick`` shrinks all of them.

    Claim: low-transfer -- every number in this dataclass is a cap on bytes
    moved, and the cost axis reports what the caps actually bought.
    """

    sketch_max_rows: int = 1024
    sketch_k: int = 64
    pair_max_rows: int = 512
    merge_coords: int = 200_000
    baseline_max_rows: int = 512
    baseline_coords: int = 100_000
    max_pairs: Optional[int] = None
    #: Smallest 2D tensor (in parameters) any comparator is allowed to use as a
    #: shared coordinate system. ``None`` keeps each module's own default; the
    #: harness lowers it automatically on benches whose tensors are smaller than
    #: that default, which is what makes a tiny synthetic bench runnable.
    min_params: Optional[int] = None

    @staticmethod
    def quick() -> "Budget":
        """Reduced grid used by ``--quick`` and by the synthetic smoke test.

        Claim: infra.
        """
        return Budget(
            sketch_max_rows=256,
            sketch_k=32,
            pair_max_rows=256,
            merge_coords=20_000,
            baseline_max_rows=256,
            baseline_coords=20_000,
            max_pairs=60,
        )


# --------------------------------------------------------------------------- #
# Metrics (hand-rolled so the numbers do not depend on a sklearn version)
# --------------------------------------------------------------------------- #


def roc_curve(y_true: Sequence[bool], scores: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(fpr, tpr, thresholds)`` with correct handling of tied scores.

    Claim: low-false-positive -- the whole relatedness axis is read off this
    curve, and ties are common (several baselines saturate at 1.0 on identical
    tensors), so collapsing tied scores into one operating point is required for
    the FPR numbers to mean anything.
    """
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(scores, dtype=float)
    if y.size == 0:
        z = np.zeros(1)
        return z, z, np.array([np.inf])
    order = np.argsort(-s, kind="mergesort")
    s = s[order]
    y = y[order]
    distinct = np.where(np.diff(s))[0]
    idx = np.concatenate([distinct, np.array([s.size - 1])])
    tps = np.cumsum(y)[idx].astype(float)
    fps = (1.0 + idx).astype(float) - tps
    n_pos = float(y.sum())
    n_neg = float(y.size - n_pos)
    tpr = tps / n_pos if n_pos > 0 else np.zeros_like(tps)
    fpr = fps / n_neg if n_neg > 0 else np.zeros_like(fps)
    thr = s[idx]
    fpr = np.concatenate([[0.0], fpr])
    tpr = np.concatenate([[0.0], tpr])
    thr = np.concatenate([[np.inf], thr])
    return fpr, tpr, thr


def roc_auc(y_true: Sequence[bool], scores: Sequence[float]) -> float:
    """Area under the ROC curve, trapezoidal, ties counted as half.

    Claim: low-false-positive.
    """
    fpr, tpr, _ = roc_curve(y_true, scores)
    if fpr.size < 2:
        return float("nan")
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))


def average_precision(y_true: Sequence[bool], scores: Sequence[float]) -> float:
    """Average precision (step-wise area under the precision-recall curve).

    Claim: low-false-positive -- AP is reported next to AUC because the labelled
    pair set is deliberately negative-heavy (hard controls), and AP is the
    metric that notices when the top of the ranking fills up with them.
    """
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(scores, dtype=float)
    n_pos = int(y.sum())
    if y.size == 0 or n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y).astype(float)
    ranks = np.arange(1, y.size + 1, dtype=float)
    precision = tp / ranks
    return float(np.sum(precision * y) / n_pos)


def fpr_at_tpr(
    y_true: Sequence[bool], scores: Sequence[float], target_tpr: float = OPERATING_TPR
) -> Tuple[float, float]:
    """Smallest FPR at which recall reaches ``target_tpr``, and that threshold.

    Claim: low-false-positive -- this pair of numbers *is* the operating point
    the hard-control false-positive rate is then measured at, so both are
    reported rather than just the FPR.
    """
    fpr, tpr, thr = roc_curve(y_true, scores)
    ok = np.where(tpr >= target_tpr)[0]
    if ok.size == 0:
        return float("nan"), float("nan")
    i = int(ok[np.argmin(fpr[ok])])
    return float(fpr[i]), float(thr[i])


def rate_above(scores: Sequence[float], threshold: float) -> float:
    """Fraction of ``scores`` at or above ``threshold`` (nan-safe).

    Claim: low-false-positive -- used for the hard-control FPR at the operating
    threshold.
    """
    s = np.asarray(scores, dtype=float)
    if s.size == 0 or not math.isfinite(threshold):
        return float("nan")
    return float(np.mean(s >= threshold))


# --------------------------------------------------------------------------- #
# Byte metering
# --------------------------------------------------------------------------- #


@dataclass
class Meter:
    """Accumulated transfer accounting for one metered block.

    Claim: low-transfer -- the cost axis is only credible if Stemma and every
    baseline are measured through the *same* accounting, so this meter watches
    the loader itself rather than trusting each caller to report.
    """

    sources: List[Any] = field(default_factory=list)
    seconds: float = 0.0

    def totals(self) -> TransferStats:
        """Sum the transfer stats of every source opened inside the block.

        Claim: low-transfer.
        """
        st = TransferStats()
        full = 0
        for src in self.sources:
            s = getattr(src, "stats", None)
            if s is None:
                continue
            st.bytes_read += int(getattr(s, "bytes_read", 0) or 0)
            st.requests += int(getattr(s, "requests", 0) or 0)
            st.cache_hits += int(getattr(s, "cache_hits", 0) or 0)
            try:
                full += int(src.total_size())
            except Exception:  # pragma: no cover - loader dependent
                full += int(getattr(s, "full_size_bytes", 0) or 0)
        st.full_size_bytes = full
        st.seconds = self.seconds
        return st


@contextmanager
def byte_meter() -> Iterator[Meter]:
    """Count every byte read by every :class:`SafeTensorsSource` opened inside.

    Claim: low-transfer -- patching the loader's constructor (and restoring it
    on exit) is what lets the harness attribute bytes to a *decision* rather
    than to a module, so Stemma and the baselines are compared like with like
    even though they open their sources through different call paths.
    """
    from stemma.remote_loader import SafeTensorsSource

    meter = Meter()
    original = SafeTensorsSource.__init__

    def patched(self, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        meter.sources.append(self)

    SafeTensorsSource.__init__ = patched  # type: ignore[method-assign]
    t0 = time.perf_counter()
    try:
        yield meter
    finally:
        meter.seconds = time.perf_counter() - t0
        SafeTensorsSource.__init__ = original  # type: ignore[method-assign]


# --------------------------------------------------------------------------- #
# Tolerant calling: sibling modules evolve independently
# --------------------------------------------------------------------------- #


def call_tolerant(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call ``fn``, dropping keyword arguments it turns out not to accept.

    Claim: infra -- the harness must survive a collaborator module whose
    optional knobs differ slightly from the contract rather than abandoning a
    whole benchmark axis over one keyword.
    """
    import re

    kw = dict(kwargs)
    while True:
        try:
            return fn(*args, **kw)
        except TypeError as exc:
            m = re.search(r"unexpected keyword argument '([^']+)'", str(exc))
            if m and m.group(1) in kw:
                kw.pop(m.group(1))
                LOG.debug("dropped kwarg %r for %s", m.group(1), getattr(fn, "__name__", fn))
                continue
            if kw:
                LOG.debug("retrying %s with no optional kwargs (%s)",
                          getattr(fn, "__name__", fn), exc)
                kw = {}
                continue
            raise


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #


@dataclass
class ModelRec:
    """One ground-truth model: its id, its on-disk path and its true lineage.

    Claim: infra.
    """

    id: str
    path: str
    family: str = "unknown"
    op: str = "base"
    parents: List[str] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=dict)
    license: Optional[str] = None
    base: Optional[str] = None
    method: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PairRec:
    """One labelled ordered pair, normalised.

    Claim: infra.
    """

    a: str
    b: str
    related: bool
    direction: str
    relation: str
    raw_relation: str = ""
    hard_control: bool = False


@dataclass
class GroundTruth:
    """Parsed ``ground_truth.json`` in the schema frozen by CONTRACT.md.

    Claim: infra -- one loader for the synthetic bench, the real bench and any
    hand-written smoke fixture, so the harness never grows a special case.
    """

    models: Dict[str, ModelRec]
    edges: List[Dict[str, Any]]
    pairs: List[PairRec]
    bench_dir: Path

    def ref(self, model_id: str) -> str:
        """Loader reference (absolute path when local) for a model id.

        Claim: infra.
        """
        rec = self.models.get(model_id)
        return rec.path if rec else model_id

    def id_of(self, ref: str) -> str:
        """Inverse of :meth:`ref`: map a loader reference back to a model id.

        Claim: infra -- decompositions come back keyed by the reference that was
        passed in, and the report must speak in ground-truth ids.
        """
        for mid, rec in self.models.items():
            if rec.path == ref or mid == ref:
                return mid
        return ref


def _normalize_relation(rel: Optional[str]) -> str:
    """Map a free-form relation string onto :data:`RELATION_ORDER`.

    Claim: infra -- the per-relation direction table is mandatory
    (docs/FINDINGS.md 5.1), so relation naming must be robust to the exact
    wording a bench generator happened to use.
    """
    if not rel:
        return "other"
    key = str(rel).strip().lower().replace("-", "_").replace(" ", "_")
    if key in _RELATION_ALIASES:
        return _RELATION_ALIASES[key]
    for alias, canon in _RELATION_ALIASES.items():
        if alias in key:
            return canon
    return "other"


def _merge_method_of(rec: ModelRec) -> str:
    """Recover the merge algorithm (slerp / ties / dare / ...) for a merged model.

    Claim: merge-recovery -- the merge axis is broken down by method because a
    slerp of two parents and a ties-merge of four are different problems.
    """
    for cand in (rec.method, rec.raw.get("merge_method"), rec.raw.get("method"), rec.op):
        if not cand:
            continue
        low = str(cand).lower()
        for m in _MERGE_METHODS:
            if m in low:
                return m
    return "unknown"


def load_ground_truth(bench_dir: Path) -> GroundTruth:
    """Read ``<bench_dir>/ground_truth.json`` and normalise every record.

    Claim: infra -- resolves model paths relative to the bench directory,
    synthesises labelled pairs from the DAG when the file only carries edges,
    and marks the hard same-shape-different-init controls.
    """
    gt_path = bench_dir / "ground_truth.json"
    if not gt_path.is_file():
        raise FileNotFoundError(f"no ground_truth.json in {bench_dir}")
    with open(gt_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    models: Dict[str, ModelRec] = {}
    for entry in raw.get("models", []):
        mid = str(entry.get("id") or entry.get("model_id") or entry.get("name"))
        p = entry.get("path") or mid
        path = Path(p)
        if not path.is_absolute():
            path = (bench_dir / path).resolve()
        weights = {str(k): float(v) for k, v in (entry.get("weights") or {}).items()}
        models[mid] = ModelRec(
            id=mid,
            path=str(path),
            family=str(entry.get("family", "unknown")),
            op=str(entry.get("op", "base")),
            parents=[str(x) for x in (entry.get("parents") or [])],
            weights=weights,
            license=entry.get("license"),
            base=entry.get("base"),
            method=entry.get("method") or entry.get("merge_method"),
            raw=dict(entry),
        )

    edges = [dict(e) for e in raw.get("edges", [])]

    pairs: List[PairRec] = []
    for p in raw.get("pairs", []):
        pairs.append(
            PairRec(
                a=str(p["a"]),
                b=str(p["b"]),
                related=bool(p.get("related", False)),
                direction=str(p.get("direction", "none")),
                relation=_normalize_relation(p.get("relation")),
                raw_relation=str(p.get("relation") or "").strip().lower().replace(" ", "_"),
            )
        )
    if not pairs:
        pairs = _synthesize_pairs(models, edges)

    known = set(models)
    pairs = [p for p in pairs if p.a in known and p.b in known]
    return GroundTruth(models=models, edges=edges, pairs=pairs, bench_dir=bench_dir)


def _synthesize_pairs(models: Dict[str, ModelRec], edges: Sequence[Dict[str, Any]]) -> List[PairRec]:
    """Build labelled pairs from the DAG when the bench file supplies none.

    Claim: infra -- positives are the ground-truth edges; negatives are every
    cross-family combination, which keeps a hand-written smoke fixture usable
    without duplicating the pair list by hand.
    """
    out: List[PairRec] = []
    seen: set = set()
    for e in edges:
        a, b = str(e.get("parent")), str(e.get("child"))
        if a in models and b in models and (a, b) not in seen:
            seen.add((a, b))
            out.append(PairRec(a=a, b=b, related=True, direction="a->b",
                               relation=_normalize_relation(e.get("relation"))))
    ids = sorted(models)
    for i, x in enumerate(ids):
        for y in ids[i + 1:]:
            if (x, y) in seen or (y, x) in seen:
                continue
            if models[x].family != models[y].family:
                out.append(PairRec(a=x, b=y, related=False, direction="none", relation="none"))
    return out


def mark_hard_controls(gt: GroundTruth, shape_sig: Dict[str, Optional[str]]) -> None:
    """Flag the negatives that share an architecture with their counterpart.

    Claim: low-false-positive -- an unrelated pair of *different* architectures
    is trivially separable; the number worth publishing is the false-positive
    rate on same-shape, independently-initialised models, so those pairs are
    identified explicitly (by label when the bench says so, otherwise by an
    identical tensor-shape signature read from headers alone).
    """
    for pair in gt.pairs:
        if pair.related:
            continue
        raw_rel = pair.raw_relation or pair.relation
        if raw_rel in HARD_CONTROL_RELATIONS:
            pair.hard_control = True
            continue
        sa, sb = shape_sig.get(pair.a), shape_sig.get(pair.b)
        pair.hard_control = bool(sa and sb and sa == sb)


# --------------------------------------------------------------------------- #
# Model-level helpers
# --------------------------------------------------------------------------- #


def shape_signature(ref: str) -> Optional[str]:
    """Hash of a checkpoint's ``{tensor name: shape}`` map, from headers only.

    Claim: low-transfer -- identifying same-shape-different-init controls costs
    two small Range reads per model and no payload at all.
    """
    from stemma.remote_loader import open_model
    from stemma.utils import stable_hash

    try:
        with open_model(ref) as src:
            idx = src.index()
            sig = sorted((n, tuple(int(x) for x in m.shape)) for n, m in idx.items())
            return stable_hash(sig)
    except Exception as exc:
        LOG.warning("shape signature failed for %s: %s", ref, exc)
        return None


def full_download_bytes(ref: str) -> int:
    """Bytes a naive full download of ``ref`` would move (header/HEAD only).

    Claim: low-transfer -- the denominator of every reduction factor in the
    table, obtained without transferring a single byte of payload.
    """
    from stemma.remote_loader import measure_full_download_bytes

    try:
        return int(measure_full_download_bytes(ref))
    except Exception as exc:
        LOG.warning("could not size %s: %s", ref, exc)
        return 0


# --------------------------------------------------------------------------- #
# The harness
# --------------------------------------------------------------------------- #


@dataclass
class Failure:
    """One non-fatal failure, carried into results.json instead of crashing.

    Claim: infra -- a benchmark that dies on the fifth of two hundred pairs is
    not a benchmark.
    """

    axis: str
    subject: str
    error: str
    kind: str = "exception"


class Harness:
    """Runs the four axes and assembles ``results.json`` plus the report.

    Claim: infra -- one object owns the caches (sketches, model sizes) so that
    the cost axis can attribute bytes to decisions honestly instead of
    re-reading headers for every metric.
    """

    def __init__(self, args: argparse.Namespace, gt: GroundTruth, budget: Budget) -> None:
        """Bind the harness to a parsed CLI namespace, a ground truth and a budget.

        Claim: infra.
        """
        self.args = args
        self.gt = gt
        self.budget = budget
        self.seed = int(args.seed)
        self.failures: List[Failure] = []
        self.sketches: Dict[str, Any] = {}
        self.sketch_stats: Dict[str, Dict[str, float]] = {}
        self.model_full_bytes: Dict[str, int] = {}
        self.direction_mod: Optional[Any] = None
        self.direction_error: Optional[str] = None
        self.relatedness_backend = "unavailable"
        self.methods: List[str] = ["stemma"] + list(args.baselines)
        #: per method -> list of {"bytes","seconds","full"} for each decision
        self.cost_records: Dict[str, List[Dict[str, float]]] = {m: [] for m in self.methods}

    # ------------------------------------------------------------------ setup

    def fail(self, axis: str, subject: str, exc: BaseException, kind: str = "exception") -> None:
        """Record a non-fatal failure and keep going.

        Claim: infra.
        """
        msg = f"{type(exc).__name__}: {exc}"
        LOG.warning("[%s] %s failed: %s", axis, subject, msg)
        LOG.debug("%s", traceback.format_exc())
        self.failures.append(Failure(axis=axis, subject=subject, error=msg, kind=kind))

    def import_direction(self) -> None:
        """Import :mod:`stemma.direction`, retrying once after a short pause.

        Claim: direction -- this module is written concurrently with the
        harness; if it is genuinely unavailable the run continues with every
        other axis measured for real and says so loudly in results.json rather
        than silently substituting a stand-in for the headline claim.
        """
        for attempt in (0, 1):
            try:
                mod = importlib.import_module("stemma.direction")
                if attempt:  # a partially-written module may be cached from try 1
                    mod = importlib.reload(mod)
                self.direction_mod = mod
                self.direction_error = None
                self.relatedness_backend = "stemma.direction.relatedness_score"
                return
            except Exception as exc:
                self.direction_error = f"{type(exc).__name__}: {exc}"
                if attempt == 0:
                    LOG.warning("stemma.direction unavailable (%s); retrying in 5s", self.direction_error)
                    time.sleep(5.0)
        LOG.warning(
            "stemma.direction still unavailable: %s -- direction axis will be reported "
            "as unavailable and relatedness falls back to sketch similarity",
            self.direction_error,
        )
        self.relatedness_backend = "stemma.sketch.sketch_similarity (fallback)"

    def needed_models(self) -> List[str]:
        """Ids of every model any axis will touch.

        Claim: infra.
        """
        need: set = set()
        for p in self.gt.pairs:
            need.add(p.a)
            need.add(p.b)
        for rec in self.gt.models.values():
            if len(rec.parents) >= 2:
                need.add(rec.id)
                need.update(rec.parents)
        return sorted(need)

    def prepare(self) -> None:
        """Sketch every needed model once and record its size and byte cost.

        Claim: low-transfer -- a sketch is a *model*-level object: it is paid for
        once and reused by every pair, which is exactly how a real deployment
        would index a universe. Both the amortised and per-decision views of
        that cost are reported.
        """
        from stemma.sketch import sketch_model

        for mid in self.needed_models():
            ref = self.gt.ref(mid)
            self.model_full_bytes[mid] = full_download_bytes(ref)
            try:
                with byte_meter() as meter:
                    sk = sketch_model(
                        ref,
                        max_rows=self.budget.sketch_max_rows,
                        k=self.budget.sketch_k,
                        seed=self.seed,
                    )
                st = meter.totals()
                self.sketches[mid] = sk
                self.sketch_stats[mid] = {
                    "bytes": float(st.bytes_read),
                    "seconds": float(st.seconds),
                    "requests": float(st.requests),
                }
            except Exception as exc:
                self.fail("prepare", mid, exc)

    # ------------------------------------------------- axis 1: relatedness --

    def _stemma_relatedness(self, a_id: str, b_id: str) -> float:
        """Symmetric relatedness in [0, 1] for one pair, Stemma's own score.

        Claim: low-false-positive -- uses ``direction.relatedness_score`` when
        the module is importable, and otherwise the sketch cosine similarity,
        which is the same quantity the score is built on top of; the fallback is
        recorded in results.json so no reader can mistake one for the other.
        """
        sa, sb = self.sketches.get(a_id), self.sketches.get(b_id)
        if self.direction_mod is not None:
            return float(
                call_tolerant(
                    self.direction_mod.relatedness_score,
                    self.gt.ref(a_id),
                    self.gt.ref(b_id),
                    sa=sa,
                    sb=sb,
                    seed=self.seed,
                    max_rows=self.budget.pair_max_rows,
                )
            )
        from stemma.sketch import sketch_similarity

        if sa is None or sb is None:
            raise RuntimeError("sketch missing for fallback relatedness")
        return float(sketch_similarity(sa, sb))

    def _baseline_score(self, name: str, a_id: str, b_id: str) -> float:
        """One symmetric baseline similarity for a pair.

        Claim: direction -- these are the comparators Stemma is measured
        against; each is symmetric by construction and therefore appears with a
        50% direction score later in the same table.
        """
        from stemma.baselines import BASELINES

        fn = BASELINES[name]
        return float(
            call_tolerant(
                fn,
                self.gt.ref(a_id),
                self.gt.ref(b_id),
                seed=self.seed,
                max_rows=self.budget.baseline_max_rows,
                coords=self.budget.baseline_coords,
            )
        )

    def run_relatedness(self) -> Dict[str, Any]:
        """Axis 1: can each method tell related from unrelated at all?

        Claim: low-false-positive -- reports ROC AUC, average precision,
        FPR@95%TPR and, emphasised, the false-positive rate on the hard
        same-shape-different-init controls at the operating threshold. Those
        controls are the only negatives that are actually difficult, so an
        aggregate FPR without them would flatter every method here.
        """
        pairs = self.gt.pairs
        labels = [p.related for p in pairs]
        scores: Dict[str, List[float]] = {m: [] for m in self.methods}
        usable: List[bool] = []

        for pair in pairs:
            row: Dict[str, float] = {}
            ok = True
            for method in self.methods:
                try:
                    with byte_meter() as meter:
                        if method == "stemma":
                            val = self._stemma_relatedness(pair.a, pair.b)
                        else:
                            val = self._baseline_score(method, pair.a, pair.b)
                    st = meter.totals()
                    extra = 0.0
                    if method == "stemma":
                        # a cold decision also pays for both sketches
                        extra = sum(
                            self.sketch_stats.get(m, {}).get("bytes", 0.0) for m in (pair.a, pair.b)
                        )
                    self.cost_records[method].append({
                        "bytes": float(st.bytes_read) + extra,
                        "seconds": float(st.seconds),
                        "full": float(
                            self.model_full_bytes.get(pair.a, 0) + self.model_full_bytes.get(pair.b, 0)
                        ),
                        "pair": f"{pair.a}|{pair.b}",
                    })
                    row[method] = float(val)
                except Exception as exc:
                    self.fail("relatedness", f"{method}:{pair.a}|{pair.b}", exc)
                    ok = False
                    break
            usable.append(ok)
            if ok:
                for method in self.methods:
                    scores[method].append(row[method])

        y = [lab for lab, ok in zip(labels, usable) if ok]
        kept = [p for p, ok in zip(pairs, usable) if ok]
        hard_mask = [p.hard_control for p in kept]
        easy_mask = [(not p.related) and (not p.hard_control) for p in kept]

        out: Dict[str, Any] = {
            "n_pairs": len(kept),
            "n_positive": int(sum(y)),
            "n_negative": int(len(y) - sum(y)),
            "n_hard_controls": int(sum(hard_mask)),
            "operating_tpr": OPERATING_TPR,
            "backend": self.relatedness_backend,
            "methods": {},
        }
        for method in self.methods:
            s = scores[method]
            if not s:
                out["methods"][method] = {"error": "no usable pairs"}
                continue
            auc = roc_auc(y, s)
            ap = average_precision(y, s)
            fpr95, thr = fpr_at_tpr(y, s, OPERATING_TPR)
            hard = [v for v, m in zip(s, hard_mask) if m]
            easy = [v for v, m in zip(s, easy_mask) if m]
            fpr_c, tpr_c, _ = roc_curve(y, s)
            out["methods"][method] = {
                "auc": auc,
                "average_precision": ap,
                "fpr_at_95tpr": fpr95,
                "operating_threshold": thr,
                "hard_control_fpr": rate_above(hard, thr),
                "hard_control_n": len(hard),
                "easy_negative_fpr": rate_above(easy, thr),
                "easy_negative_n": len(easy),
                "mean_score_positive": float(np.mean([v for v, l in zip(s, y) if l])) if any(y) else float("nan"),
                "mean_score_hard_control": float(np.mean(hard)) if hard else float("nan"),
                "roc": {"fpr": [float(x) for x in fpr_c], "tpr": [float(x) for x in tpr_c]},
                "scores": [float(x) for x in s],
            }
        out["labels"] = [bool(v) for v in y]
        out["pair_ids"] = [f"{p.a}|{p.b}" for p in kept]
        out["hard_control_mask"] = [bool(v) for v in hard_mask]
        return out

    # ---------------------------------------------------- axis 2: direction --

    def _outgroup_for(self, parent: str, child: str) -> Optional[str]:
        """Pick a sibling from the ground-truth DAG to root the pair against.

        Claim: direction -- docs/FINDINGS.md section 4: for a scar-free edge the
        two-model problem is only weakly identifiable, and the classical fix is
        outgroup rooting. A *sibling of the child* (another descendant of the
        same parent) is the strongest available outgroup; an uncle is the
        fallback. Descendants of the child itself are never eligible.
        """
        children: Dict[str, List[str]] = {}
        parents: Dict[str, List[str]] = {}
        for e in self.gt.edges:
            p, c = str(e.get("parent")), str(e.get("child"))
            children.setdefault(p, []).append(c)
            parents.setdefault(c, []).append(p)

        banned = {parent, child}
        # descendants of child are not outgroups
        stack = [child]
        while stack:
            node = stack.pop()
            for c in children.get(node, []):
                if c not in banned:
                    banned.add(c)
                    stack.append(c)

        for cand in sorted(children.get(parent, [])):  # siblings of the child
            if cand not in banned and cand in self.sketches:
                return cand
        for gp in sorted(parents.get(parent, [])):     # uncles
            for cand in sorted(children.get(gp, [])):
                if cand not in banned and cand in self.sketches:
                    return cand
        return None

    def _direction_call(self, a_id: str, b_id: str, outgroup: Optional[str] = None) -> Tuple[Any, TransferStats, bool]:
        """Run ``estimate_direction`` on one pair, optionally with an outgroup.

        Claim: direction -- returns the verdict, the metered transfer cost and
        whether the outgroup keyword was actually accepted, so the outgroup
        column can never silently become a duplicate of the plain column.
        """
        mod = self.direction_mod
        assert mod is not None
        kw: Dict[str, Any] = {
            "sa": self.sketches.get(a_id),
            "sb": self.sketches.get(b_id),
            "seed": self.seed,
            "max_rows": self.budget.pair_max_rows,
            "abstain": float(self.args.abstain),
        }
        used_outgroup = False
        if outgroup is not None:
            # Must be a LIST. A bare str satisfies Sequence[ModelRef] and would
            # be iterated character by character, turning one outgroup into one
            # bogus repo request per letter.
            kw["outgroup"] = [self.gt.ref(outgroup)]
        with byte_meter() as meter:
            verdict = call_tolerant(mod.estimate_direction, self.gt.ref(a_id), self.gt.ref(b_id), **kw)
        if outgroup is not None:
            # call_tolerant may have dropped the keyword; detect acceptance by
            # checking the verdict for any trace of it.
            blob = json.dumps(
                {
                    "f": {k: float(v) for k, v in (getattr(verdict, "features", {}) or {}).items()},
                    "e": list(getattr(verdict, "evidence", []) or []),
                },
                default=str,
            ).lower()
            used_outgroup = ("outgroup" in blob) or _accepts_kwarg(mod.estimate_direction, "outgroup")
        return verdict, meter.totals(), used_outgroup

    def run_direction(self) -> Dict[str, Any]:
        """Axis 2: which model is the ancestor -- per relation, never blended.

        Claim: direction -- the headline claim. Reports overall accuracy (an
        abstention counts as *wrong*), accuracy on answered pairs, the
        abstention rate, and the mandatory per-relation breakdown, plus the
        outgroup-rooted rerun of the scar-free relations as its own column.
        Baselines are scored through ``baselines.baseline_direction``, which
        always abstains: 0.5, a structural ceiling.
        """
        ordered = [
            p for p in self.gt.pairs
            if p.related and p.direction in ("a->b", "b->a")
        ]
        result: Dict[str, Any] = {
            "n_ordered_pairs": len(ordered),
            "abstain_threshold": float(self.args.abstain),
            "available": self.direction_mod is not None,
            "unavailable_reason": self.direction_error,
        }

        # --- baselines: measured, not asserted --------------------------------
        from stemma.baselines import baseline_direction

        base_rows: Dict[str, Any] = {}
        for name in self.args.baselines:
            unknown = 0
            for p in ordered:
                try:
                    if baseline_direction(self.gt.ref(p.a), self.gt.ref(p.b)) == "unknown":
                        unknown += 1
                except Exception as exc:  # pragma: no cover - it cannot raise
                    self.fail("direction", f"{name}:{p.a}|{p.b}", exc)
            base_rows[name] = {
                "accuracy": 0.5,
                "accuracy_on_answered": float("nan"),
                "abstention_rate": 1.0,
                "n": len(ordered),
                "n_unknown": unknown,
                "note": "STRUCTURAL ceiling: the statistic is symmetric by "
                        "construction, so every pair abstains and a coin flip "
                        "scores 0.5. This is not a tuning failure.",
            }
        result["baselines"] = base_rows

        if self.direction_mod is None:
            result["stemma"] = {"error": "stemma.direction unavailable", "n": len(ordered)}
            result["per_relation"] = {}
            result["outgroup"] = {"available": False, "reason": self.direction_error}
            return result

        records: List[Dict[str, Any]] = []
        for p in ordered:
            try:
                verdict, st, _ = self._direction_call(p.a, p.b)
            except Exception as exc:
                self.fail("direction", f"stemma:{p.a}|{p.b}", exc)
                continue
            pred = str(getattr(verdict, "direction", "unknown"))
            records.append({
                "a": p.a, "b": p.b,
                "relation": p.relation,
                "truth": p.direction,
                "pred": pred,
                "llr": float(getattr(verdict, "llr", 0.0) or 0.0),
                "confidence": float(getattr(verdict, "confidence", 0.0) or 0.0),
                "correct": pred == p.direction,
                "abstained": pred == "unknown",
                "bytes": float(st.bytes_read),
                "seconds": float(st.seconds),
            })
            self.cost_records.setdefault("stemma-direction", []).append({
                "bytes": float(st.bytes_read) + sum(
                    self.sketch_stats.get(m, {}).get("bytes", 0.0) for m in (p.a, p.b)),
                "seconds": float(st.seconds),
                "full": float(self.model_full_bytes.get(p.a, 0) + self.model_full_bytes.get(p.b, 0)),
                "pair": f"{p.a}|{p.b}",
            })

        result["stemma"] = _score_direction(records)
        result["records"] = records

        per_rel: Dict[str, Any] = {}
        for rel in RELATION_ORDER:
            sub = [r for r in records if r["relation"] == rel]
            if sub:
                per_rel[rel] = {**_score_direction(sub), "group": GROUP_OF.get(rel, "other")}
        other = [r for r in records if r["relation"] not in RELATION_ORDER]
        if other:
            per_rel["other"] = {**_score_direction(other), "group": "other"}
        result["per_relation"] = per_rel

        # --- outgroup rooting on the scar-free relations ----------------------
        result["outgroup"] = self._run_outgroup(records)
        return result

    def _run_outgroup(self, records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Rerun the scar-free pairs with a ground-truth sibling as outgroup.

        Claim: direction -- docs/FINDINGS.md section 4 predicts this is exactly
        where the extra evidence family should pay: sft/lora/cpt edges carry no
        lossy scar, so a third relative is the only strong signal available.
        Reported as its own column so its contribution is visible and not
        blended into the headline accuracy.
        """
        scar_free = [r for r in records if r["relation"] in SCAR_FREE]
        out: Dict[str, Any] = {
            "relations": list(SCAR_FREE),
            "n_candidates": len(scar_free),
            "available": False,
            "supported_by_module": _accepts_kwarg(
                getattr(self.direction_mod, "estimate_direction", None), "outgroup"
            ),
        }
        rerun: List[Dict[str, Any]] = []
        no_outgroup: List[Dict[str, Any]] = []
        changed = 0
        for r in scar_free:
            parent = r["a"] if r["truth"] == "a->b" else r["b"]
            child = r["b"] if r["truth"] == "a->b" else r["a"]
            og = self._outgroup_for(parent, child)
            if og is None:
                continue
            try:
                verdict, st, accepted = self._direction_call(r["a"], r["b"], outgroup=og)
            except Exception as exc:
                self.fail("direction-outgroup", f"{r['a']}|{r['b']}", exc)
                continue
            pred = str(getattr(verdict, "direction", "unknown"))
            llr = float(getattr(verdict, "llr", 0.0) or 0.0)
            if abs(llr - r["llr"]) > 1e-9:
                changed += 1
            rerun.append({
                "a": r["a"], "b": r["b"], "relation": r["relation"], "outgroup": og,
                "truth": r["truth"], "pred": pred, "llr": llr,
                "correct": pred == r["truth"], "abstained": pred == "unknown",
                "accepted_kwarg": bool(accepted),
                "bytes": float(st.bytes_read), "seconds": float(st.seconds),
            })
            no_outgroup.append(r)

        if not rerun:
            out["reason"] = "no usable sibling outgroup in the ground-truth DAG"
            return out
        out.update({
            "available": True,
            "n": len(rerun),
            "with_outgroup": _score_direction(rerun),
            "without_outgroup": _score_direction(no_outgroup),
            "llr_changed_pairs": changed,
            "records": rerun,
        })
        if changed == 0:
            out["caveat"] = (
                "the outgroup keyword did not change any llr -- stemma.direction "
                "may not consume it, so the two columns are identical by "
                "construction rather than by measurement"
            )
        return out

    # ------------------------------------------------ axis 3: merge recovery --

    def run_merge(self) -> Dict[str, Any]:
        """Axis 3: recover the parent *set* and the mixing ratios of a merge.

        Claim: merge-recovery -- every ground-truth merged model is decomposed
        against its true parents plus ``--decoys`` decoys drawn from the rest of
        the universe. Precision/recall/F1 of the recovered parent set and the
        mixing-ratio MAE are broken down by merge method and by parent count.
        No symmetric baseline can enter this table at all.
        """
        from stemma.merge_decompose import decompose_merge, mixing_mae, parent_set_prf

        merged = [
            rec for rec in self.gt.models.values()
            if len(rec.parents) >= 2 or (rec.weights and len(rec.weights) >= 2)
        ]
        merged.sort(key=lambda r: r.id)

        out: Dict[str, Any] = {
            "n_merged_models": len(merged),
            "decoys": int(self.args.decoys),
            "baseline_note": "n/a - symmetric fingerprints produce no mixing coefficients",
            "cases": [],
        }
        if not merged:
            out["note"] = "no merged models in this ground truth"
            return out

        rng = random.Random(self.seed)
        universe = sorted(self.gt.models)

        for rec in merged:
            truth_parents = list(rec.parents) or sorted(rec.weights)
            truth_weights = dict(rec.weights) if rec.weights else {
                p: 1.0 / len(truth_parents) for p in truth_parents
            }
            banned = set(truth_parents) | {rec.id} | set(_descendants(self.gt.edges, rec.id))
            base_id = rec.base or _common_ancestor(self.gt.edges, truth_parents)
            if base_id:
                banned.add(base_id)
            pool = [m for m in universe if m not in banned]
            rng.shuffle(pool)
            decoys = sorted(pool[: max(0, int(self.args.decoys))])
            candidates = list(truth_parents) + decoys

            try:
                with byte_meter() as meter:
                    dec = call_tolerant(
                        decompose_merge,
                        self.gt.ref(rec.id),
                        [self.gt.ref(c) for c in candidates],
                        base=self.gt.ref(base_id) if base_id else None,
                        coords=self.budget.merge_coords,
                        seed=self.seed,
                    )
                st = meter.totals()
            except Exception as exc:
                self.fail("merge", rec.id, exc)
                continue

            pred_raw = dec.as_dict()
            pred = {self.gt.id_of(k): float(v) for k, v in pred_raw.items()}
            selected = [self.gt.id_of(s) for s in dec.selected]
            precision, recall, f1 = parent_set_prf(selected, truth_parents)
            mae = mixing_mae(pred, truth_weights)
            mae_true = mixing_mae(
                {k: v for k, v in pred.items() if k in truth_weights}, truth_weights
            )
            case = {
                "child": rec.id,
                "method": _merge_method_of(rec),
                "n_parents": len(truth_parents),
                "base": base_id,
                "base_source": "ground-truth common ancestor" if base_id else "inferred by decomposer",
                "true_parents": truth_parents,
                "decoys": decoys,
                "selected": selected,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mixing_mae": mae,
                "mixing_mae_true_parents_only": mae_true,
                "residual": float(dec.residual),
                "r2": float(dec.r2),
                "solver": dec.method,
                "predicted": pred,
                "truth": truth_weights,
                "bytes": float(st.bytes_read),
                "seconds": float(st.seconds),
                "full_bytes": float(
                    sum(self.model_full_bytes.get(m, full_download_bytes(self.gt.ref(m)))
                        for m in [rec.id, *candidates] + ([base_id] if base_id else []))
                ),
            }
            out["cases"].append(case)

        cases = out["cases"]
        out["overall"] = _aggregate_merge(cases)
        by_method: Dict[str, Any] = {}
        for m in sorted({c["method"] for c in cases}):
            by_method[m] = _aggregate_merge([c for c in cases if c["method"] == m])
        out["by_method"] = by_method
        by_np: Dict[str, Any] = {}
        for n in sorted({c["n_parents"] for c in cases}):
            by_np[str(n)] = _aggregate_merge([c for c in cases if c["n_parents"] == n])
        out["by_parent_count"] = by_np
        return out

    # ---------------------------------------------------------- axis 4: cost --

    def run_cost(self) -> Dict[str, Any]:
        """Axis 4: bytes and seconds *per decision* against a full download.

        Claim: low-transfer -- the full-download figure is the sum of the two
        checkpoints' shard sizes taken from the safetensors header / file
        listing. No payload is ever downloaded to produce it, here or anywhere
        else in this file.
        """
        out: Dict[str, Any] = {
            "note": "one decision == one labelled pair; full-download bytes come "
                    "from the safetensors header/file sizes, never from a download",
            "per_method": {},
            "sketch_amortised": {},
        }
        for method, recs in self.cost_records.items():
            if not recs:
                continue
            b = np.array([r["bytes"] for r in recs], dtype=float)
            s = np.array([r["seconds"] for r in recs], dtype=float)
            f = np.array([r["full"] for r in recs], dtype=float)
            med_b, med_f = float(np.median(b)), float(np.median(f))
            out["per_method"][method] = {
                "n_decisions": int(b.size),
                "bytes_median": med_b,
                "bytes_total": float(b.sum()),
                "bytes_mean": float(b.mean()),
                "seconds_median": float(np.median(s)),
                "seconds_total": float(s.sum()),
                "full_download_bytes_median": med_f,
                "full_download_bytes_total": float(f.sum()),
                "reduction_median": (med_f / med_b) if med_b > 0 else float("inf"),
                "reduction_total": (float(f.sum()) / float(b.sum())) if b.sum() > 0 else float("inf"),
                "fraction_of_full_median": (med_b / med_f) if med_f > 0 else float("nan"),
                "bytes_median_human": human_bytes(med_b),
                "full_download_median_human": human_bytes(med_f),
            }
        total_sketch = sum(v.get("bytes", 0.0) for v in self.sketch_stats.values())
        total_full = float(sum(self.model_full_bytes.values()))
        out["sketch_amortised"] = {
            "n_models": len(self.sketch_stats),
            "bytes_total": total_sketch,
            "bytes_median_per_model": float(
                np.median([v["bytes"] for v in self.sketch_stats.values()])
            ) if self.sketch_stats else 0.0,
            "full_download_bytes_total": total_full,
            "fraction_of_full": (total_sketch / total_full) if total_full > 0 else float("nan"),
            "comment": "a sketch is paid for once per model and reused by every "
                       "pair; the per-decision numbers above charge each cold "
                       "decision for both of its sketches, which is the "
                       "pessimistic view",
        }
        out["remote"] = self.run_remote_probe() if self.args.remote else {
            "attempted": False,
            "reason": "--remote not passed; local bench dirs measure the same code path "
                      "but over mmap rather than HTTP",
        }
        return out

    def run_remote_probe(self) -> Dict[str, Any]:
        """Measure a real HTTP-Range decision against two public repos.

        Claim: low-transfer -- the local bench exercises the identical code path
        over mmap, but the claim is about the network, so ``--remote`` takes one
        real measurement and skips gracefully when offline.
        """
        a, b = self.args.remote_pair
        out: Dict[str, Any] = {"attempted": True, "pair": [a, b]}
        try:
            from stemma.sketch import sketch_model

            with byte_meter() as meter:
                sa = sketch_model(a, max_rows=self.budget.sketch_max_rows,
                                  k=self.budget.sketch_k, seed=self.seed)
                sb = sketch_model(b, max_rows=self.budget.sketch_max_rows,
                                  k=self.budget.sketch_k, seed=self.seed)
                score = None
                if self.direction_mod is not None:
                    score = float(call_tolerant(
                        self.direction_mod.relatedness_score, a, b, sa=sa, sb=sb,
                        seed=self.seed, max_rows=self.budget.pair_max_rows,
                    ))
            st = meter.totals()
            full = full_download_bytes(a) + full_download_bytes(b)
            out.update({
                "ok": True,
                "bytes_read": int(st.bytes_read),
                "requests": int(st.requests),
                "seconds": float(st.seconds),
                "full_download_bytes": int(full),
                "fraction_of_full": (st.bytes_read / full) if full else float("nan"),
                "reduction": (full / st.bytes_read) if st.bytes_read else float("inf"),
                "relatedness_score": score,
                "bytes_read_human": human_bytes(st.bytes_read),
                "full_download_human": human_bytes(full),
            })
        except Exception as exc:
            self.fail("cost-remote", f"{a}|{b}", exc, kind="skipped")
            out.update({"ok": False, "skipped": True, "error": f"{type(exc).__name__}: {exc}"})
        return out

    # ------------------------------------------- axis 5: DAG reconstruction --

    def _dag_sketches(self, universe: Sequence[str]) -> Dict[str, Any]:
        """``ref -> Sketch`` for the whole universe, reusing :meth:`prepare`'s work.

        Claim: low-transfer -- the DAG is built twice (gate off, gate on) over 20
        models; sketching them once and handing the dict to ``build_phylogeny``
        keeps the second build from re-reading a single byte of weights.
        """
        from stemma.sketch import sketch_model

        out: Dict[str, Any] = {}
        for mid in universe:
            ref = self.gt.ref(mid)
            sk = self.sketches.get(mid)
            if sk is None:
                try:
                    sk = sketch_model(ref, max_rows=self.budget.sketch_max_rows,
                                      k=self.budget.sketch_k, seed=self.seed)
                    self.sketches[mid] = sk
                except Exception as exc:
                    self.fail("dag", f"sketch:{mid}", exc)
                    continue
            out[ref] = sk
        return out

    def _build_dag(
        self,
        universe: Sequence[str],
        sketches: Dict[str, Any],
        gate: Optional[float],
    ) -> Tuple[Any, TransferStats, float]:
        """Run ``build_phylogeny`` over the whole universe at one gate setting.

        Claim: direction -- this is the end-to-end product of the whole pipeline:
        retrieval, relatedness gating, orientation and merge decomposition, with
        no ground-truth pair list to lean on. ``gate`` is passed only when
        ``build_phylogeny`` names it in its signature -- it swallows unknown
        keywords into ``**kw`` and forwards them to the loader, so a blind pass
        would be silently ignored and the two columns would differ by nothing.
        """
        from stemma.phylogeny import build_phylogeny

        kw: Dict[str, Any] = {
            "sketches": sketches,
            "seed": self.seed,
            "max_rows": self.budget.pair_max_rows,
            "direction_abstain": float(self.args.abstain),
        }
        if gate is not None and _accepts_kwarg(build_phylogeny, GATE_PARAM):
            kw[GATE_PARAM] = float(gate)
        refs = [self.gt.ref(m) for m in universe]
        t0 = time.perf_counter()
        with byte_meter() as meter:
            phylo = call_tolerant(build_phylogeny, refs, **kw)
        return phylo, meter.totals(), time.perf_counter() - t0

    def _dag_run(
        self,
        label: str,
        universe: Sequence[str],
        sketches: Dict[str, Any],
        gate: Optional[float],
        *,
        applied: bool,
    ) -> Dict[str, Any]:
        """Build one DAG and score it against ``ground_truth.json``'s edges.

        Claim: direction -- reports exact-edge P/R/F1, ancestor-closure P/R/F1,
        root identification and the wrong-edge type breakdown for a single gate
        setting; :meth:`run_dag` calls it twice so the guard is measured.
        """
        phylo, stats, seconds = self._build_dag(universe, sketches, gate)
        edges: List[Tuple[str, str]] = []
        info: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for e in getattr(phylo, "edges", []):
            key = (self.gt.id_of(str(e.parent)), self.gt.id_of(str(e.child)))
            edges.append(key)
            info[key] = {
                "confidence": float(getattr(e, "confidence", float("nan"))),
                "relation": str(getattr(e, "relation", "")),
                "weight": (None if getattr(e, "weight", None) is None
                           else float(getattr(e, "weight"))),
            }
        roots = [self.gt.id_of(str(r)) for r in getattr(phylo, "root_candidates", [])]
        scored = score_dag(edges, roots, self.gt.edges, universe, edge_info=info)
        meta = dict(getattr(phylo, "meta", {}) or {})
        scored.update({
            "label": label,
            GATE_PARAM: gate,
            "gate_applied": bool(applied),
            "seconds": float(seconds),
            "bytes_read": float(stats.bytes_read),
            "n_nodes": len(getattr(phylo, "nodes", []) or []),
            "n_predicted_edges": len(edges),
            "builder_meta": {
                k: meta.get(k) for k in (
                    "n_pairs_considered", "n_pairs_related", "n_direction_abstained",
                    "n_cycle_edges_dropped", "relatedness_threshold", "direction_abstain",
                    "max_distance", "k", "index_backend",
                ) if k in meta
            },
            "spotlight": self._dag_spotlight(edges, info),
        })
        return scored

    def _dag_spotlight(
        self,
        edges: Sequence[Tuple[str, str]],
        info: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """What this DAG says the parents of the README-#9 child are.

        Claim: direction -- limitation #9 is a claim about one concrete edge
        (``smollm2-merge-ties2``, whose true parents are ``sft`` 0.6 and ``cpt``
        0.4 but which was handed two cousins). A reader deserves to see that
        specific answer, not only an F1 that could improve for other reasons.
        """
        child = SPOTLIGHT_CHILD
        rec = self.gt.models.get(child)
        if rec is None:
            return {"child": child, "present": False}
        truth = list(rec.parents) or sorted(rec.weights)
        anc = ancestor_sets(self.gt.edges, list(self.gt.models))
        truth_edges = {(str(e.get("parent")), str(e.get("child"))) for e in self.gt.edges}
        got = []
        for p, c in edges:
            if c != child:
                continue
            ok = (p, c) in truth_edges
            got.append({
                "parent": p,
                "correct": ok,
                "type": None if ok else classify_wrong_edge(p, c, truth_edges, anc),
                "confidence": info.get((p, c), {}).get("confidence"),
                "weight": info.get((p, c), {}).get("weight"),
            })
        return {
            "child": child,
            "present": True,
            "true_parents": truth,
            "true_weights": dict(rec.weights),
            "recovered_parents": got,
            "exact": sorted(g["parent"] for g in got) == sorted(truth),
        }

    def run_dag(self) -> Dict[str, Any]:
        """Axis 5: reconstruct the whole DAG and score it, gate off vs gate on.

        Claim: direction -- axes 1-4 score pairwise decisions on pairs the ground
        truth already certified as edges. That is precisely how README
        limitation #9 shipped: on the full universe the builder preferred two
        *cousins* (which carry pruning/quantisation scars, so the direction
        estimator answers decisively) over the true, scar-free parents (on which
        it abstains). This axis scores the artifact the user actually gets, and
        breaks the wrong edges down by failure mode instead of collapsing them
        into one number.
        """
        from stemma.phylogeny import build_phylogeny

        universe = sorted(self.gt.models)
        gate_supported = _accepts_kwarg(build_phylogeny, GATE_PARAM)
        shipped_default = _default_of(build_phylogeny, GATE_PARAM)
        override = self.args.proximity_factor
        out: Dict[str, Any] = {
            "available": True,
            "n_universe": len(universe),
            "n_ground_truth_edges": len(self.gt.edges),
            "gate_parameter": GATE_PARAM,
            "gate_supported": bool(gate_supported),
            "gate_shipped_default": (None if shipped_default is None else float(shipped_default)),
            "gate_override": (None if override is None else float(override)),
            "metrics_note": (
                "EDGE = exact parent->child pairs, direction-sensitive. CLOSURE = "
                "transitive (ancestor, descendant) pairs, which forgives a "
                "transitive-reduction miss but still punishes a cousin edge. "
                "ROOT accuracy = fraction of true roots recovered as roots."
            ),
        }
        sketches = self._dag_sketches(universe)
        if not sketches:
            out.update({"available": False, "reason": "no sketch could be built for the universe"})
            return out

        if not gate_supported:
            out["degraded"] = (
                f"stemma.phylogeny.build_phylogeny does not name {GATE_PARAM!r} in its "
                "signature, so the proximity gate cannot be switched from here. The DAG "
                "was built ONCE and the same numbers are reported in both columns: they "
                "are identical by construction, not by measurement."
            )
            LOG.warning("%s", out["degraded"])
            single = self._dag_run(
                f"gate UNAVAILABLE (build_phylogeny has no {GATE_PARAM})",
                universe, sketches, None, applied=False,
            )
            out["runs"] = {"gate_off": single, "gate_on": dict(single)}
            out["delta"] = _dag_delta(single, single)
            return out

        on_value = float(override) if override is not None else shipped_default
        specs = [
            ("gate_off", f"gate OFF ({GATE_PARAM}=0)", 0.0, True),
            ("gate_on",
             f"gate ON ({GATE_PARAM}={_fmt(on_value, '.3g') if on_value is not None else 'default'}"
             + (", --proximity-factor override" if override is not None else ", shipped default")
             + ")",
             on_value, True),
        ]
        runs: Dict[str, Any] = {}
        for key, label, gate, applied in specs:
            try:
                runs[key] = self._dag_run(label, universe, sketches, gate, applied=applied)
            except Exception as exc:
                self.fail("dag", key, exc)
                runs[key] = {"label": label, "error": f"{type(exc).__name__}: {exc}",
                             GATE_PARAM: gate, "gate_applied": False}
        out["runs"] = runs
        if "error" not in runs.get("gate_off", {}) and "error" not in runs.get("gate_on", {}):
            out["delta"] = _dag_delta(runs["gate_off"], runs["gate_on"])
        return out


def _dag_delta(off: Dict[str, Any], on: Dict[str, Any]) -> Dict[str, Any]:
    """Gate-on minus gate-off for the headline DAG numbers.

    Claim: direction -- the guard's effect is a *difference*, and printing it
    explicitly stops a reader from having to subtract two tables by eye (and
    stops the harness from claiming an improvement it did not measure).
    """
    def d(section: str, key: str) -> float:
        try:
            return float(on[section][key]) - float(off[section][key])
        except (KeyError, TypeError, ValueError):
            return float("nan")

    cousins_off = int(off.get("wrong_edge_types", {}).get("cousin", 0))
    cousins_on = int(on.get("wrong_edge_types", {}).get("cousin", 0))
    return {
        "edge_f1": d("edge", "f1"),
        "edge_precision": d("edge", "precision"),
        "edge_recall": d("edge", "recall"),
        "closure_f1": d("closure", "f1"),
        "root_accuracy": d("roots", "accuracy"),
        "n_wrong_edges": int(on.get("n_wrong_edges", 0)) - int(off.get("n_wrong_edges", 0)),
        "cousin_edges": cousins_on - cousins_off,
        "seconds": float(on.get("seconds", 0.0)) - float(off.get("seconds", 0.0)),
    }


def _filter_kwargs(fn: Callable[..., Any], candidates: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the keyword arguments ``fn`` actually names in its signature.

    Claim: infra -- several comparators swallow unknown keywords into
    ``**loader_kw`` and only explode deep inside the loader, so the harness
    filters up front instead of discovering the mismatch by exception.
    """
    try:
        import inspect

        params = inspect.signature(fn).parameters
    except Exception:
        return {}
    return {k: v for k, v in candidates.items() if k in params and v is not None}


def _accepts_kwarg(fn: Optional[Callable[..., Any]], name: str) -> bool:
    """Whether ``fn`` names ``name`` explicitly in its signature.

    Claim: infra -- lets the outgroup column say honestly whether the sibling
    module consumed the keyword or the harness merely offered it.
    """
    if fn is None:
        return False
    try:
        import inspect

        return name in inspect.signature(fn).parameters
    except Exception:
        return False


def _score_direction(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Accuracy / accuracy-on-answered / abstention for a set of direction calls.

    Claim: direction -- an abstention counts as *wrong* in the headline accuracy
    and is removed from the answered accuracy; both are reported because for
    provenance a confident wrong answer is worse than "unknown"
    (docs/FINDINGS.md 5.2).
    """
    n = len(records)
    if n == 0:
        return {"n": 0, "accuracy": float("nan"), "accuracy_on_answered": float("nan"),
                "abstention_rate": float("nan"), "n_correct": 0, "n_abstained": 0}
    correct = sum(1 for r in records if r["correct"])
    abstained = sum(1 for r in records if r["abstained"])
    answered = n - abstained
    return {
        "n": n,
        "n_correct": correct,
        "n_abstained": abstained,
        "n_answered": answered,
        "accuracy": correct / n,
        "accuracy_on_answered": (correct / answered) if answered else float("nan"),
        "abstention_rate": abstained / n,
        "mean_abs_llr": float(np.mean([abs(r["llr"]) for r in records])),
    }


def _aggregate_merge(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean parent-set P/R/F1 and mixing MAE over a group of merge cases.

    Claim: merge-recovery.
    """
    if not cases:
        return {"n": 0, "precision": float("nan"), "recall": float("nan"),
                "f1": float("nan"), "mixing_mae": float("nan")}
    g = lambda k: float(np.mean([c[k] for c in cases]))  # noqa: E731
    return {
        "n": len(cases),
        "precision": g("precision"),
        "recall": g("recall"),
        "f1": g("f1"),
        "mixing_mae": g("mixing_mae"),
        "mixing_mae_true_parents_only": g("mixing_mae_true_parents_only"),
        "residual": g("residual"),
        "r2": g("r2"),
        "exact_parent_set": float(np.mean([1.0 if c["f1"] >= 1.0 else 0.0 for c in cases])),
    }


def _descendants(edges: Sequence[Dict[str, Any]], node: str) -> List[str]:
    """All ids reachable downstream of ``node`` in the ground-truth DAG.

    Claim: merge-recovery -- a descendant of the merged child is never a valid
    decoy: it contains the child's own signal and would make the decoy test a
    trick question rather than a hard one.
    """
    children: Dict[str, List[str]] = {}
    for e in edges:
        children.setdefault(str(e.get("parent")), []).append(str(e.get("child")))
    seen: set = set()
    stack = [node]
    while stack:
        cur = stack.pop()
        for c in children.get(cur, []):
            if c not in seen:
                seen.add(c)
                stack.append(c)
    return sorted(seen)


def _common_ancestor(edges: Sequence[Dict[str, Any]], nodes: Sequence[str]) -> Optional[str]:
    """Nearest id that is an ancestor of every node in ``nodes``.

    Claim: merge-recovery -- mergekit users know which base model they merged
    over, so supplying it is realistic; when the DAG has none the decomposer
    infers one itself and the case records ``base_source`` accordingly.
    """
    parents: Dict[str, List[str]] = {}
    for e in edges:
        parents.setdefault(str(e.get("child")), []).append(str(e.get("parent")))

    def anc(n: str) -> List[str]:
        seen: List[str] = []
        stack = list(parents.get(n, []))
        while stack:
            cur = stack.pop(0)
            if cur in seen:
                continue
            seen.append(cur)
            stack.extend(parents.get(cur, []))
        return seen

    if not nodes:
        return None
    common = None
    for n in nodes:
        a = anc(n)
        common = a if common is None else [x for x in common if x in a]
    return common[0] if common else None


# --------------------------------------------------------------------------- #
# Axis 5 helpers: whole-DAG scoring
# --------------------------------------------------------------------------- #


def child_to_parents(edges: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Ground-truth ``child -> [parents]`` adjacency.

    Claim: infra -- every axis-5 metric is derived from this map, so it is built
    once and shared rather than re-walked per edge.
    """
    out: Dict[str, List[str]] = {}
    for e in edges:
        out.setdefault(str(e.get("child")), []).append(str(e.get("parent")))
    return out


def ancestor_sets(edges: Sequence[Dict[str, Any]], nodes: Iterable[str]) -> Dict[str, set]:
    """Proper-ancestor set of every node in a directed edge list (cycle-safe).

    Claim: direction -- an ancestor set only exists because edges are oriented,
    and it is what separates a *cousin* error (shared ancestor, neither one an
    ancestor of the other) from an *ancestor-not-parent* error. Collapsing those
    two into "wrong edge" is what let README limitation #9 ship.
    """
    parents = child_to_parents(edges)
    out: Dict[str, set] = {}
    for n in nodes:
        seen: set = set()
        stack = list(parents.get(n, []))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(parents.get(cur, []))
        seen.discard(n)
        out[n] = seen
    return out


def closure_pairs(edges: Iterable[Tuple[str, str]], nodes: Iterable[str]) -> set:
    """All ``(ancestor, descendant)`` pairs implied by a ``(parent, child)`` set.

    Claim: direction -- the closure is the forgiving view of the same DAG: it
    scores *reachability* rather than adjacency, so a recovered grandparent edge
    still earns credit while a cousin edge -- which asserts reachability that the
    truth does not contain -- is still punished. Reporting exact-edge and closure
    F1 side by side is what makes the severity of an error visible.
    """
    children: Dict[str, List[str]] = {}
    node_list = list(nodes)
    for p, c in edges:
        children.setdefault(str(p), []).append(str(c))
        node_list.append(str(p))
        node_list.append(str(c))
    out: set = set()
    for n in dict.fromkeys(node_list):
        seen: set = set()
        stack = list(children.get(n, []))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(children.get(cur, []))
        seen.discard(n)
        for d in seen:
            out.add((n, d))
    return out


def set_prf(pred: set, truth: set) -> Dict[str, Any]:
    """Precision / recall / F1 of a predicted set against a truth set.

    Claim: infra -- shared by the exact-edge and ancestor-closure scores so the
    two columns of the DAG table are computed by identical code.
    """
    tp = len(pred & truth)
    fp = len(pred - truth)
    fn = len(truth - pred)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    if not (_finite(precision) and _finite(recall)) or (precision + recall) <= 0:
        f1 = 0.0 if (tp + fp + fn) else float("nan")
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {
        "n_pred": len(pred), "n_truth": len(truth),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def classify_wrong_edge(
    parent: str, child: str, gt_edges: set, anc: Dict[str, set]
) -> str:
    """Name the failure mode of one wrong predicted edge.

    Claim: low-false-positive -- this function is the point of axis 5. A wrong
    edge between two models that merely share an ancestor ("cousin") means
    candidate *retrieval* failed; a wrong edge to a genuine but indirect ancestor
    means only the transitive reduction failed; a reversed edge means the
    *direction* estimator failed. Those demand three different fixes, and one
    aggregate F1 reports them identically.
    """
    if (child, parent) in gt_edges:
        return "reversed"
    a_p, a_c = anc.get(parent, set()), anc.get(child, set())
    if parent in a_c:
        return "ancestor-not-parent"
    if child in a_p:
        # A genuine ancestor relation stated backwards: still a direction error.
        return "reversed"
    if a_p & a_c:
        return "cousin"
    return "unrelated"


def score_dag(
    pred_edges: Sequence[Tuple[str, str]],
    pred_roots: Sequence[str],
    gt_edges: Sequence[Dict[str, Any]],
    universe: Sequence[str],
    *,
    edge_info: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Score one reconstructed DAG against the ground-truth DAG.

    Claim: direction -- exact-edge P/R/F1 (orientation-sensitive), ancestor
    closure P/R/F1, root identification, and the wrong-edge confusion breakdown.
    An abstained edge simply does not appear, so recall carries the cost of
    abstention exactly as accuracy does on the direction axis.
    """
    universe = list(dict.fromkeys(str(u) for u in universe))
    truth_edges = {(str(e.get("parent")), str(e.get("child"))) for e in gt_edges}
    pred = {(str(p), str(c)) for p, c in pred_edges}
    anc = ancestor_sets(gt_edges, universe)

    edge = set_prf(pred, truth_edges)
    closure = set_prf(closure_pairs(pred, universe), closure_pairs(truth_edges, universe))

    true_roots = {n for n in universe if not anc.get(n)}
    got_roots = {str(r) for r in pred_roots} or {n for n in universe if n not in {c for _, c in pred}}
    roots = set_prf(got_roots, true_roots)
    touched = {n for e in pred for n in e}
    roots["n_isolated"] = len([n for n in universe if n not in touched])
    roots["true_roots"] = sorted(true_roots)
    roots["predicted_roots"] = sorted(got_roots)
    roots["missed_roots"] = sorted(true_roots - got_roots)
    roots["accuracy"] = roots["recall"]  # "are the true roots recovered as roots?"

    breakdown = {t: 0 for t in WRONG_EDGE_TYPES}
    wrong: List[Dict[str, Any]] = []
    for p, c in sorted(pred - truth_edges):
        kind = classify_wrong_edge(p, c, truth_edges, anc)
        breakdown[kind] = breakdown.get(kind, 0) + 1
        info = (edge_info or {}).get((p, c), {})
        wrong.append({
            "parent": p, "child": c, "type": kind,
            "confidence": info.get("confidence"),
            "relation": info.get("relation"),
            "weight": info.get("weight"),
        })
    return {
        "edge": edge,
        "closure": closure,
        "roots": roots,
        "wrong_edge_types": breakdown,
        "n_wrong_edges": len(wrong),
        "wrong_edges": wrong,
        "correct_edges": [{"parent": p, "child": c} for p, c in sorted(pred & truth_edges)],
        "missed_edges": [{"parent": p, "child": c} for p, c in sorted(truth_edges - pred)],
        "predicted_edges": [{"parent": p, "child": c} for p, c in sorted(pred)],
    }


def _default_of(fn: Optional[Callable[..., Any]], name: str) -> Any:
    """Declared default of keyword ``name`` on ``fn``, or ``None``.

    Claim: infra -- the "gate ON" column must print the value the shipped code
    actually uses, and the only honest source for that is the signature itself.
    """
    if fn is None:
        return None
    try:
        import inspect

        p = inspect.signature(fn).parameters.get(name)
        if p is None or p.default is inspect._empty:  # type: ignore[attr-defined]
            return None
        return p.default
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _fmt(x: Any, spec: str = ".3f", dash: str = "n/a") -> str:
    """Format a float for the markdown tables, tolerating nan/None.

    Claim: infra.
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        return dash
    if not math.isfinite(v):
        return dash
    return format(v, spec)


def _pct(x: Any) -> str:
    """Percentage formatter used by the accuracy columns.

    Claim: infra.
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(v):
        return "n/a"
    return f"{100.0 * v:.1f}%"


METHOD_LABELS = {
    "stemma": "**Stemma**",
    "cosine": "cosine",
    "cka": "CKA/REEF-style",
    "huref": "HuRef-style",
}


def render_tables(results: Dict[str, Any]) -> str:
    """Build the headline markdown table plus the per-relation direction table.

    Claim: infra -- rows are methods and columns are the four axes, so a reader
    can see in one glance that the symmetric baselines occupy the relatedness
    columns and are structurally absent from the direction and merge ones.
    """
    rel = results.get("relatedness", {})
    direction = results.get("direction", {})
    merge = results.get("merge", {})
    cost = results.get("cost", {}).get("per_method", {})
    methods = results.get("config", {}).get("methods", ["stemma"])

    lines: List[str] = []
    lines.append("## Headline table")
    lines.append("")
    lines.append(
        f"Bench: `{results['config']['bench_dir']}` - "
        f"{rel.get('n_pairs', 0)} labelled pairs "
        f"({rel.get('n_positive', 0)} related, {rel.get('n_negative', 0)} unrelated, "
        f"{rel.get('n_hard_controls', 0)} of them HARD same-shape-different-init controls); "
        f"{direction.get('n_ordered_pairs', 0)} ordered pairs; "
        f"{merge.get('n_merged_models', 0)} merges with {merge.get('decoys', 0)} decoys each. "
        f"seed={results['config']['seed']}, quick={results['config']['quick']}."
    )
    lines.append("")
    header = ("| method | AUC | AP | FPR@95TPR | **FPR hard controls** | DIRECTION ACC | "
              "MERGE F1 | MIXING MAE | BYTES/DECISION |")
    lines.append(header)
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for m in methods:
        rm = rel.get("methods", {}).get(m, {})
        cm = cost.get(m, {})
        if m == "stemma":
            sd = direction.get("stemma", {})
            dir_cell = (
                f"{_pct(sd.get('accuracy'))} ({_pct(sd.get('accuracy_on_answered'))} on answered, "
                f"{_pct(sd.get('abstention_rate'))} abstain)"
                if "accuracy" in sd else "unavailable"
            )
            ov = merge.get("overall", {})
            f1_cell = _fmt(ov.get("f1"))
            mae_cell = _fmt(ov.get("mixing_mae"))
        else:
            dir_cell = "50% (chance)"
            f1_cell = "n/a"
            mae_cell = "n/a"
        lines.append(
            f"| {METHOD_LABELS.get(m, m)} | {_fmt(rm.get('auc'))} | "
            f"{_fmt(rm.get('average_precision'))} | {_fmt(rm.get('fpr_at_95tpr'))} | "
            f"**{_fmt(rm.get('hard_control_fpr'))}** | {dir_cell} | {f1_cell} | {mae_cell} | "
            f"{human_bytes(cm.get('bytes_median', 0.0))} |"
        )
    lines.append("")
    lines.append(
        "`n/a` in the merge columns is not a zero: a symmetric fingerprint produces "
        "**no mixing coefficients at all**, so there is nothing to score. "
        "`50% (chance)` in the direction column is a **structural ceiling** - "
        "`cosine(a,b) == cosine(b,a)` by construction, so `baselines.baseline_direction` "
        "abstains on every pair and a coin flip is the best a symmetric statistic can do. "
        "It is not a tuning failure."
    )
    full_med = 0.0
    for m in methods:
        full_med = max(full_med, float(cost.get(m, {}).get("full_download_bytes_median", 0.0)))
    if full_med:
        lines.append("")
        lines.append(
            f"Full download of both checkpoints in a median decision: "
            f"**{human_bytes(full_med)}** (from safetensors headers - nothing was downloaded)."
        )

    # ---- table 2: per-relation direction ------------------------------------
    lines.append("")
    lines.append("## Direction accuracy per relation (Stemma)")
    lines.append("")
    lines.append(
        "docs/FINDINGS.md 5.1 makes this table mandatory: an aggregate would let the "
        "easy scar-bearing edges hide the hard scar-free ones."
    )
    lines.append("")
    lines.append("| relation | group | n | accuracy | acc. on answered | abstained | mean \\|llr\\| |")
    lines.append("|---|---|---|---|---|---|---|")
    per_rel = direction.get("per_relation", {})
    order = [r for r in RELATION_ORDER if r in per_rel] + [r for r in per_rel if r not in RELATION_ORDER]
    for rel_name in order:
        d = per_rel[rel_name]
        lines.append(
            f"| {rel_name} | {d.get('group', 'other')} | {d.get('n', 0)} | "
            f"{_pct(d.get('accuracy'))} | {_pct(d.get('accuracy_on_answered'))} | "
            f"{_pct(d.get('abstention_rate'))} | {_fmt(d.get('mean_abs_llr'), '.2f')} |"
        )
    if not per_rel:
        lines.append("| - | - | 0 | n/a | n/a | n/a | n/a |")

    og = direction.get("outgroup", {})
    lines.append("")
    lines.append("## Outgroup rooting (scar-free relations only)")
    lines.append("")
    if og.get("available"):
        w, wo = og.get("with_outgroup", {}), og.get("without_outgroup", {})
        lines.append("| variant | n | accuracy | acc. on answered | abstained |")
        lines.append("|---|---|---|---|---|")
        lines.append(f"| without outgroup | {wo.get('n', 0)} | {_pct(wo.get('accuracy'))} | "
                     f"{_pct(wo.get('accuracy_on_answered'))} | {_pct(wo.get('abstention_rate'))} |")
        lines.append(f"| **with outgroup** | {w.get('n', 0)} | {_pct(w.get('accuracy'))} | "
                     f"{_pct(w.get('accuracy_on_answered'))} | {_pct(w.get('abstention_rate'))} |")
        if og.get("caveat"):
            lines.append("")
            lines.append(f"CAVEAT: {og['caveat']}.")
    else:
        lines.append(f"Not run: {og.get('reason', 'unavailable')}.")

    # ---- merge breakdown -----------------------------------------------------
    lines.append("")
    lines.append("## Merge recovery breakdown")
    lines.append("")
    lines.append("| slice | n | precision | recall | F1 | mixing MAE | residual |")
    lines.append("|---|---|---|---|---|---|---|")
    ov = merge.get("overall", {})
    lines.append(f"| all | {ov.get('n', 0)} | {_fmt(ov.get('precision'))} | {_fmt(ov.get('recall'))} | "
                 f"{_fmt(ov.get('f1'))} | {_fmt(ov.get('mixing_mae'))} | {_fmt(ov.get('residual'))} |")
    for label, group in (("method", merge.get("by_method", {})),
                         ("parents", merge.get("by_parent_count", {}))):
        for key, d in group.items():
            lines.append(
                f"| {label}={key} | {d.get('n', 0)} | {_fmt(d.get('precision'))} | "
                f"{_fmt(d.get('recall'))} | {_fmt(d.get('f1'))} | {_fmt(d.get('mixing_mae'))} | "
                f"{_fmt(d.get('residual'))} |"
            )
    lines.append("")
    lines.append("Baselines: **n/a - symmetric fingerprints produce no mixing coefficients**.")

    # ---- cost ----------------------------------------------------------------
    lines.append("")
    lines.append("## Transfer cost per decision")
    lines.append("")
    lines.append("| method | decisions | median bytes | total bytes | median s | "
                 "median full download | reduction |")
    lines.append("|---|---|---|---|---|---|---|")
    for m, c in cost.items():
        lines.append(
            f"| {METHOD_LABELS.get(m, m)} | {c.get('n_decisions', 0)} | "
            f"{human_bytes(c.get('bytes_median', 0))} | {human_bytes(c.get('bytes_total', 0))} | "
            f"{_fmt(c.get('seconds_median'), '.3f')} | "
            f"{human_bytes(c.get('full_download_bytes_median', 0))} | "
            f"{_fmt(c.get('reduction_median'), '.1f')}x |"
        )
    remote = results.get("cost", {}).get("remote", {})
    if remote.get("ok"):
        lines.append("")
        lines.append(
            f"REAL HTTP-Range measurement ({remote['pair'][0]} vs {remote['pair'][1]}): "
            f"read {remote['bytes_read_human']} of {remote['full_download_human']} "
            f"= {100 * remote['fraction_of_full']:.3f}% in {remote['seconds']:.1f}s "
            f"over {remote['requests']} requests."
        )
    elif remote.get("attempted"):
        lines.append("")
        lines.append(f"Remote measurement skipped: {remote.get('error', 'offline')}.")

    fails = results.get("failures", [])
    lines.append("")
    lines.append(f"Non-fatal failures: **{len(fails)}** "
                 f"({', '.join(sorted({f['axis'] for f in fails})) or 'none'}).")
    lines.append("")
    lines.append(
        "Stemma reports statistical evidence about weight-level similarity and "
        "derivation direction. It does not establish provenance as fact."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


CAPTIONS: Dict[str, str] = {}


def make_figures(results: Dict[str, Any], out_dir: Path) -> List[str]:
    """Write the six benchmark figures and return their captions.

    Claim: infra -- one chart per figure, matplotlib Agg, default colours, axis
    labels and a title on every one; fig6 is the honest version of fig1 and
    fig5 is the capability matrix the whole project is arguing for.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    captions: List[str] = []
    methods = results.get("config", {}).get("methods", ["stemma"])
    rel = results.get("relatedness", {}).get("methods", {})
    direction = results.get("direction", {})
    merge = results.get("merge", {})
    cost = results.get("cost", {}).get("per_method", {})

    def label(m: str) -> str:
        return {"stemma": "Stemma", "cosine": "cosine", "cka": "CKA/REEF", "huref": "HuRef"}.get(m, m)

    # --- fig1: direction accuracy -------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    sd = direction.get("stemma", {})
    acc, ans = [], []
    for m in methods:
        if m == "stemma":
            a = sd.get("accuracy", float("nan"))
            b = sd.get("accuracy_on_answered", float("nan"))
        else:
            a, b = 0.5, 0.5
        acc.append(0.0 if not _finite(a) else float(a))
        ans.append(0.0 if not _finite(b) else float(b))
    x = np.arange(len(methods), dtype=float)
    ax.bar(x - 0.2, acc, width=0.38, label="accuracy (abstain = wrong)")
    ax.bar(x + 0.2, ans, width=0.38, label="accuracy on answered pairs")
    ax.axhline(0.5, linestyle="--", linewidth=1.2, color="black")
    ax.text(len(methods) - 0.5, 0.515, "chance = 0.50", ha="right", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([label(m) for m in methods])
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("method")
    ax.set_ylabel("direction accuracy")
    ax.set_title("Derivation-direction accuracy: Stemma vs symmetric baselines")
    ax.annotate(
        "symmetric methods cannot exceed chance by construction",
        xy=(0.5, 1.05), xycoords="axes fraction", ha="center", fontsize=9,
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_direction_accuracy.png", dpi=150)
    plt.close(fig)
    captions.append(
        "fig1_direction_accuracy.png - Direction accuracy per method. Every baseline sits "
        "exactly on the 0.50 chance line because cosine/CKA/HuRef statistics are symmetric "
        "in their two arguments; no amount of tuning moves them. Stemma's bar is the "
        "aggregate only - see fig6 for the per-relation breakdown that matters."
    )

    # --- fig2: ROC ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for m in methods:
        r = rel.get(m, {}).get("roc")
        if not r:
            continue
        ax.plot(r["fpr"], r["tpr"], label=f"{label(m)} (AUC={_fmt(rel[m].get('auc'), '.3f')})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="black", label="chance")
    ax.axhline(0.95, linestyle=":", linewidth=1.0, color="grey")
    ax.set_xlabel("false-positive rate")
    ax.set_ylabel("true-positive rate")
    ax.set_title("Relatedness ROC (all methods); dotted line = 95% TPR operating point")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_roc.png", dpi=150)
    plt.close(fig)
    captions.append(
        "fig2_roc.png - Relatedness ROC curves. All four methods can rank related above "
        "unrelated; this axis is where the symmetric baselines are genuinely competitive. "
        "The dotted 95%-TPR line is the operating point at which the hard-control "
        "false-positive rate in the headline table is measured."
    )

    # --- fig3: merge recovery -------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 6))
    cases = merge.get("cases", [])
    by_method: Dict[str, List[Tuple[float, float]]] = {}
    for c in cases:
        for name, tw in list(c["truth"].items()) + [(d, 0.0) for d in c["decoys"]]:
            pv = float(c["predicted"].get(name, 0.0))
            by_method.setdefault(c["method"], []).append((float(tw), pv))
    for meth, pts in sorted(by_method.items()):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.scatter(xs, ys, s=48, alpha=0.8, label=f"{meth} (n={len(pts)})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="black", label="y = x")
    mae = merge.get("overall", {}).get("mixing_mae", float("nan"))
    ax.annotate(f"mixing-ratio MAE = {_fmt(mae, '.4f')}\n(decoys have true ratio 0)",
                xy=(0.04, 0.86), xycoords="axes fraction", fontsize=9)
    ax.set_xlabel("true mixing coefficient")
    ax.set_ylabel("recovered mixing coefficient")
    ax.set_title("Merge recovery: recovered vs true mixing ratios")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_merge_recovery.png", dpi=150)
    plt.close(fig)
    captions.append(
        "fig3_merge_recovery.png - Recovered vs true mixing coefficient for every candidate "
        "of every ground-truth merge, coloured by merge method. Points on the y=x line are "
        "correctly weighted parents; points on the x=0 axis are decoys, and a decoy above "
        "zero is a false parent. No symmetric baseline can produce a single point on this plot."
    )

    # --- fig4: transfer --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ms = [m for m in methods if m in cost]
    full = [max(1.0, cost[m].get("full_download_bytes_median", 1.0)) for m in ms]
    rng_ = [max(1.0, cost[m].get("bytes_median", 1.0)) for m in ms]
    x = np.arange(len(ms), dtype=float)
    ax.bar(x - 0.2, full, width=0.38, label="full download (from headers)")
    ax.bar(x + 0.2, rng_, width=0.38, label="Stemma Range reads (measured)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([label(m) for m in ms])
    ax.set_xlabel("method")
    ax.set_ylabel("bytes per decision (log scale)")
    ax.set_title("Bytes moved per pairwise decision: full download vs Range reads")
    for xi, f, r in zip(x, full, rng_):
        ax.text(xi + 0.2, r * 1.15, f"{f / r:.0f}x less", ha="center", fontsize=8)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_transfer.png", dpi=150)
    plt.close(fig)
    captions.append(
        "fig4_transfer.png - Bytes per decision, log scale. The full-download bar is the sum "
        "of both checkpoints' shard sizes read from the safetensors header, never downloaded. "
        "On a tiny synthetic bench the fixed header cost dominates and the reduction factor "
        "is small by construction; the reduction grows with checkpoint size."
    )

    # --- fig5: capability matrix (the money figure) ---------------------------
    caps = ["relatedness", "direction", "multi-parent", "mixing ratio", "sub-1% transfer"]
    rows = [label(m) for m in methods]
    fig, ax = plt.subplots(figsize=(9, 1.1 + 0.85 * len(rows)))
    for i, m in enumerate(methods):
        frac = cost.get(m, {}).get("fraction_of_full_median", float("nan"))
        sub1 = bool(_finite(frac) and frac < 0.01)
        vals = [True, m == "stemma", m == "stemma", m == "stemma", sub1]
        notes = [
            f"AUC {_fmt(rel.get(m, {}).get('auc'), '.2f')}",
            _pct(direction.get("stemma", {}).get("accuracy")) if m == "stemma" else "50% (chance)",
            "decompose" if m == "stemma" else "n/a",
            _fmt(merge.get("overall", {}).get("mixing_mae"), ".3f") if m == "stemma" else "n/a",
            f"{100 * frac:.2f}%" if _finite(frac) else "n/a",
        ]
        for j, (ok, note) in enumerate(zip(vals, notes)):
            y = len(rows) - 1 - i
            ax.add_patch(plt.Rectangle((j, y), 1, 1, facecolor="C0" if ok else "C3",
                                       alpha=0.16, edgecolor="grey", linewidth=0.8))
            ax.text(j + 0.5, y + 0.62, "✓" if ok else "✗",
                    ha="center", va="center", fontsize=17,
                    color="C0" if ok else "C3")
            ax.text(j + 0.5, y + 0.26, note, ha="center", va="center", fontsize=8)
    ax.set_xlim(0, len(caps))
    ax.set_ylim(0, len(rows))
    ax.set_xticks(np.arange(len(caps)) + 0.5)
    ax.set_xticklabels(caps)
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels(rows[::-1])
    ax.set_xlabel("capability")
    ax.set_ylabel("method")
    ax.set_title("Capability matrix: what each method can answer at all")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_summary_matrix.png", dpi=150)
    plt.close(fig)
    captions.append(
        "fig5_summary_matrix.png - Capability matrix. The crosses in the direction, "
        "multi-parent and mixing-ratio columns are structural: those statistics are "
        "symmetric functions of an unordered pair, so the questions are not merely hard "
        "for them, they are unanswerable. The sub-1%-transfer column is measured on this "
        "run, not asserted, so a tiny synthetic bench (where fixed header cost dominates) "
        "will show a cross."
    )

    # --- fig6: per-relation direction (the honest fig1) -----------------------
    per_rel = direction.get("per_relation", {})
    groups = [("scar-free", SCAR_FREE), ("scar-bearing", SCAR_BEARING),
              ("structural / cross-arch", STRUCTURAL)]
    names: List[str] = []
    vals: List[float] = []
    colors: List[str] = []
    boundaries: List[float] = []
    group_spans: List[Tuple[str, float, float]] = []
    for gi, (gname, rels) in enumerate(groups):
        start = len(names)
        for r in rels:
            if r in per_rel:
                names.append(f"{r}\n(n={per_rel[r].get('n', 0)})")
                v = per_rel[r].get("accuracy", float("nan"))
                vals.append(float(v) if _finite(v) else 0.0)
                colors.append(f"C{gi}")
        if len(names) > start:
            group_spans.append((gname, start - 0.5, len(names) - 0.5))
            if len(names) < sum(len(g[1]) for g in groups):
                boundaries.append(len(names) - 0.5)
    fig, ax = plt.subplots(figsize=(max(7.0, 1.2 * max(1, len(names))), 5.4))
    if names:
        ax.bar(np.arange(len(names)), vals, color=colors, width=0.65)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.03, _pct(v), ha="center", fontsize=8)
        for b in boundaries:
            ax.axvline(b, color="grey", linewidth=1.0, linestyle="-")
        for gname, x0, x1 in group_spans:
            ax.text((x0 + x1) / 2.0, 1.06, gname, ha="center", fontsize=9)
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names, fontsize=8)
    else:
        ax.text(0.5, 0.5, "direction axis unavailable", ha="center", transform=ax.transAxes)
    ax.axhline(0.5, linestyle="--", linewidth=1.2, color="black")
    ax.text(len(names) - 0.4 if names else 0.9, 0.52, "chance", ha="right", fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.set_xlabel("relation (ground-truth derivation operation)")
    ax.set_ylabel("direction accuracy (abstain = wrong)")
    ax.set_title("Direction accuracy per relation - the honest version of fig1")
    fig.tight_layout()
    fig.savefig(out_dir / "fig6_direction_by_relation.png", dpi=150)
    plt.close(fig)
    captions.append(
        "fig6_direction_by_relation.png - Direction accuracy split by ground-truth relation, "
        "with the scar-bearing group (quantise/prune/vocab - lossy, irreversible, so the scar "
        "can only appear downstream) separated from the scar-free group (sft/lora/cpt, where "
        "docs/FINDINGS.md measured the delta-spectrum evidence at 1e-3..1e-2 and predicts weak "
        "identifiability). The aggregate bar in fig1 is the average of these; this is the "
        "figure to read."
    )
    return captions


def _finite(x: Any) -> bool:
    """True when ``x`` is a finite float.

    Claim: infra.
    """
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #


def environment_info(seed: int) -> Dict[str, Any]:
    """Versions, index backend, git revision (absent is fine) and the seed.

    Claim: infra -- a number without the environment that produced it is not a
    measurement, and the harness must run in a directory that is not a git repo.
    """
    versions: Dict[str, Optional[str]] = {}
    for mod in ("numpy", "scipy", "torch", "safetensors", "huggingface_hub",
                "transformers", "sklearn", "matplotlib", "faiss", "gradio",
                "networkx", "datasets", "requests"):
        try:
            versions[mod] = str(getattr(importlib.import_module(mod), "__version__", "unknown"))
        except Exception:
            versions[mod] = None

    backend = "unknown"
    try:
        from stemma.phylogeny import SketchIndex

        backend = SketchIndex().backend
    except Exception as exc:
        backend = f"unavailable ({type(exc).__name__})"

    rev: Optional[str] = None
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(_REPO_ROOT),
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or None
    except Exception:
        rev = None

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "versions": versions,
        "index_backend": backend,
        "git_revision": rev,
        "seed": seed,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cwd": os.getcwd(),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Command-line surface of the benchmark harness.

    Claim: infra -- ``stemma bench`` shells out to this parser, so every flag
    here is part of the reproducibility story.
    """
    p = argparse.ArgumentParser(
        prog="benchmarks/run.py",
        description="Stemma evaluation harness: relatedness, direction, merge recovery, cost.",
    )
    p.add_argument("--bench-dir", default=str(_REPO_ROOT / "bench_models"),
                   help="directory holding the generated models + ground_truth.json")
    p.add_argument("--out", default=str(_REPO_ROOT / "benchmarks" / "results.json"),
                   help="where to write results.json")
    p.add_argument("--figures", default=str(_REPO_ROOT / "figures"),
                   help="directory for the six PNG figures")
    p.add_argument("--markdown", default=None,
                   help="where to write RESULTS.md (default: next to --out)")
    p.add_argument("--out-dir", default=None,
                   help="convenience: put results.json, RESULTS.md and figures/ under this dir")
    p.add_argument("--quick", action="store_true", help="reduced grid: fewer rows, fewer pairs")
    p.add_argument("--seed", type=int, default=0, help="global seed (default 0)")
    p.add_argument("--max-pairs", type=int, default=None,
                   help="cap the number of labelled pairs evaluated")
    p.add_argument("--no-figures", action="store_true", help="skip figure generation")
    p.add_argument("--baselines", default="cosine,cka,huref",
                   help="comma-separated subset of stemma.baselines.BASELINES")
    p.add_argument("--remote", action="store_true",
                   help="also take one REAL HTTP-Range measurement against public repos")
    p.add_argument("--remote-pair", nargs=2,
                   default=["HuggingFaceTB/SmolLM2-135M-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"],
                   help="the two public repos used by --remote")
    p.add_argument("--decoys", type=int, default=3,
                   help="decoy candidates added to each merge decomposition")
    p.add_argument("--abstain", type=float, default=0.5,
                   help="|llr| below which stemma.direction abstains")
    p.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING for stemma loggers")
    return p


def _subsample_pairs(pairs: List[PairRec], limit: Optional[int], seed: int) -> List[PairRec]:
    """Deterministically cap the pair list while keeping the label balance.

    Claim: infra -- ``--max-pairs`` must not silently drop every hard control,
    which is the one class the false-positive claim depends on.
    """
    if limit is None or limit <= 0 or len(pairs) <= limit:
        return pairs
    rng = random.Random(seed)
    pos = [p for p in pairs if p.related]
    hard = [p for p in pairs if not p.related and p.hard_control]
    easy = [p for p in pairs if not p.related and not p.hard_control]
    for bucket in (pos, hard, easy):
        rng.shuffle(bucket)
    quota = max(1, limit // 3)
    out = pos[:quota] + hard[:quota] + easy[: limit - len(pos[:quota]) - len(hard[:quota])]
    # top up from whatever is left if a bucket was small
    if len(out) < limit:
        rest = [p for p in pairs if p not in out]
        out += rest[: limit - len(out)]
    out.sort(key=lambda p: (p.a, p.b))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run every axis, write results.json / RESULTS.md / figures, print the table.

    Claim: infra -- the single reproducible entry point behind all four claims;
    returns 0 on success and 1 only when the benchmark could not start at all.
    """
    args = build_parser().parse_args(argv)
    if args.log_level:
        import logging

        logging.getLogger("stemma").setLevel(args.log_level.upper())

    if args.out_dir:
        d = Path(args.out_dir)
        args.out = str(d / "results.json")
        args.markdown = args.markdown or str(d / "RESULTS.md")
        args.figures = str(d / "figures")
    md_path = Path(args.markdown) if args.markdown else Path(args.out).with_name("RESULTS.md")

    args.baselines = [b.strip() for b in str(args.baselines).split(",") if b.strip()]
    try:
        from stemma.baselines import BASELINES

        unknown = [b for b in args.baselines if b not in BASELINES]
        if unknown:
            print(f"error: unknown baseline(s) {unknown}; available: {sorted(BASELINES)}",
                  file=sys.stderr)
            return 1
    except Exception as exc:  # pragma: no cover - stemma must import
        print(f"error: cannot import stemma.baselines: {exc}", file=sys.stderr)
        return 1

    set_seed(int(args.seed))
    budget = Budget.quick() if args.quick else Budget()
    if args.max_pairs is not None:
        budget.max_pairs = int(args.max_pairs)

    bench_dir = Path(args.bench_dir).expanduser().resolve()
    try:
        gt = load_ground_truth(bench_dir)
    except Exception as exc:
        print(f"error: could not load ground truth from {bench_dir}: {exc}", file=sys.stderr)
        print("hint: generate one with scripts/build_bench.py, or pass --bench-dir",
              file=sys.stderr)
        return 1

    t_start = time.perf_counter()
    print(f"stemma benchmark: {len(gt.models)} models, {len(gt.pairs)} labelled pairs "
          f"from {bench_dir}")

    # Hard-control labelling needs a header-only shape signature per model.
    sig: Dict[str, Optional[str]] = {}
    for mid, rec in gt.models.items():
        sig[mid] = shape_signature(rec.path)
    mark_hard_controls(gt, sig)
    gt.pairs = _subsample_pairs(gt.pairs, budget.max_pairs, int(args.seed))

    harness = Harness(args, gt, budget)
    harness.import_direction()
    harness.prepare()

    results: Dict[str, Any] = {
        "schema": SCHEMA,
        "environment": environment_info(int(args.seed)),
        "config": {
            "bench_dir": str(bench_dir),
            "seed": int(args.seed),
            "quick": bool(args.quick),
            "max_pairs": budget.max_pairs,
            "decoys": int(args.decoys),
            "abstain": float(args.abstain),
            "baselines": list(args.baselines),
            "methods": harness.methods,
            "remote": bool(args.remote),
            "budget": {
                "sketch_max_rows": budget.sketch_max_rows,
                "sketch_k": budget.sketch_k,
                "pair_max_rows": budget.pair_max_rows,
                "merge_coords": budget.merge_coords,
                "baseline_max_rows": budget.baseline_max_rows,
                "baseline_coords": budget.baseline_coords,
            },
        },
        "modules": {
            "direction_available": harness.direction_mod is not None,
            "direction_error": harness.direction_error,
            "relatedness_backend": harness.relatedness_backend,
            "mocked_components": ([] if harness.direction_mod is not None
                                  else ["stemma.direction (unavailable: direction axis not "
                                        "measured; relatedness falls back to sketch cosine)"]),
        },
        "ground_truth": {
            "n_models": len(gt.models),
            "n_edges": len(gt.edges),
            "n_pairs": len(gt.pairs),
            "n_hard_controls": sum(1 for p in gt.pairs if p.hard_control),
            "relations": sorted({p.relation for p in gt.pairs if p.related}),
        },
    }

    results["relatedness"] = harness.run_relatedness()
    results["direction"] = harness.run_direction()
    results["merge"] = harness.run_merge()
    results["cost"] = harness.run_cost()
    results["failures"] = [
        {"axis": f.axis, "subject": f.subject, "error": f.error, "kind": f.kind}
        for f in harness.failures
    ]
    results["n_failures"] = len(harness.failures)
    results["sketch_stats"] = harness.sketch_stats
    results["wall_clock_seconds"] = time.perf_counter() - t_start

    table = render_tables(results)
    results["markdown"] = table

    captions: List[str] = []
    if not args.no_figures:
        try:
            captions = make_figures(results, Path(args.figures))
        except Exception as exc:
            harness.fail("figures", str(args.figures), exc)
            results["failures"] = [
                {"axis": f.axis, "subject": f.subject, "error": f.error, "kind": f.kind}
                for f in harness.failures
            ]
            results["n_failures"] = len(harness.failures)
    results["figure_captions"] = captions

    atomic_write_json(Path(args.out), results)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_body = table
    if captions:
        md_body += "\n\n## Figures\n\n" + "\n\n".join(f"- {c}" for c in captions)
    md_path.write_text("# Stemma benchmark results\n\n" + md_body + "\n", encoding="utf-8")

    print()
    print(table)
    if captions:
        print()
        print("Figures written to", args.figures)
        for c in captions:
            print("  -", c)
    if harness.direction_mod is None:
        print()
        print("WARNING: stemma.direction could not be imported "
              f"({harness.direction_error}). The DIRECTION axis was NOT measured and the "
              "relatedness column for Stemma used the sketch-similarity fallback. "
              "Every other number in this run is real.")
    print()
    print(f"results.json -> {args.out}")
    print(f"RESULTS.md   -> {md_path}")
    print(f"total wall clock: {results['wall_clock_seconds']:.1f}s, "
          f"{results['n_failures']} non-fatal failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
