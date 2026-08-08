"""Phylogeny reconstruction: sketch index, candidate retrieval, DAG assembly, export.

This module turns a bag of model references into a directed acyclic lineage
graph. It is the place where the three measurement modules meet: the *sketch*
gives cheap symmetric retrieval, :mod:`stemma.direction` supplies the arrow of
derivation, and :mod:`stemma.merge_decompose` splits a multi-parent node into
mixing coefficients.

Claim: direction -- a phylogeny is exactly the artifact a symmetric fingerprint
cannot produce; every edge here carries an orientation and a confidence.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .types import (
    SKETCH_DIM,
    DEPTH_BUCKETS,
    ROLES,
    Edge,
    ModelRef,
    Phylogeny,
    Sketch,
    TransferStats,
)
from .utils import (
    atomic_write_json,
    get_logger,
    human_bytes,
    is_local_path,
    short_id,
    stable_hash,
)

log = get_logger(__name__)

#: Index-file format marker; bump when the .npz layout changes.
INDEX_VERSION = "stemma-index-v1"

PROXIMITY_FACTOR: float = 10.0
"""How many times the *closest* candidate's delta a direct parent may be.

Claim: low-false-positive -- this constant is the whole candidate-retrieval
guard, so it is documented with the measurement that set it rather than with an
intuition.

A direct child differs from its parent by **one** branch delta.  A cousin --
another descendant of the same ancestor -- differs by **two or more** (its own
branch plus the other's), so it can never be as close as the true parent unless
the two branches happen to cancel.  The gate turns that into a rule: a candidate
whose relative delta is far larger than the best candidate's cannot be the
direct parent.

Measured on the benchmark universe, ``relative_delta`` of every candidate
against ``smollm2-merge-ties2`` (10 shared 2-D tensors, 256 sampled rows)::

    0.0004  smollm2-sft            <- TRUE PARENT (weight 0.6)
    0.0007  smollm2-135m-root      <- real ancestor, but the GRANDparent
    0.0012  smollm2-cpt            <- TRUE PARENT (weight 0.4)
    0.0055  smollm2-lora-merged
    0.0094  smollm2-int8
    0.0995  smollm2-prune-mag30    <- cousin, wrongly chosen as a parent
    0.1190  smollm2-sft-int4       <- cousin, wrongly chosen as a parent
    0.1872  smollm2-vocab-ext      <- cousin

The true parents sit at 0.0004 / 0.0012 and the nearest cousin at 0.0995: a
**~100x gap**.  A factor of 10 therefore sits an order of magnitude inside the
observed margin on *both* sides -- it would take a true parent 10x further from
the child than the closest candidate, or a cousin 10x closer than measured,
before the choice mattered.

**MEASURED LIMITATION -- this gate is OFF by default in**
:func:`build_phylogeny` (``proximity_factor=0.0``). Enable it deliberately.

It was built to fix README limitation #9 and, measured end to end, it does not.
Two findings killed the default:

1. *It carries no signal when the child is heavily modified.* Candidate deltas
   against the 30%-pruned ``smollm2-prune-mag30`` (whose true parent is
   ``smollm2-135m-root``)::

       0.1000  ratio 1.00x  smollm2-135m-root     <- TRUE PARENT
       0.1000  ratio 1.00x  smollm2-merge-ties2   <- cousin
       0.1000  ratio 1.00x  smollm2-sft           <- cousin
       0.1004  ratio 1.00x  smollm2-int8          <- cousin

   The pruning delta dominates, so *every* candidate is equidistant and no ratio
   threshold can separate parent from cousin. The 100x margin quoted above holds
   only when the *child* is close to its parent; the wrong edge in limitation #9
   points at a heavily modified model, which is exactly the case it cannot see.
   (The theoretical floor is worse than the benchmark suggests: for two equal,
   orthogonal branches the cousin/parent ratio is only ``sqrt(2)`` ~ 1.41.)

2. *It costs more than it saves.* Running the gate over a 20-model universe took
   the trace from 2.3 GiB / 13k requests to **3.6 GiB of a 3.4 GiB universe over
   40,924 requests** -- a 1x "reduction", i.e. worse than downloading
   everything, because every candidate pair re-reads tensors.

It is kept, tested and auditable because it *does* reject grossly distant
candidates cheaply when a shared coordinate system is absent, and because the
measurement is worth preserving. But a guard that does not fix the bug and
triples transfer is not a default.

The factor is a **ratio against the closest candidate, never an absolute
threshold**, which is what makes it scale-free: a family whose fine-tunes move
1e-5 of the weight norm and a family whose fine-tunes move 1e-1 are gated by the
same number, because only the *contrast* between one branch and two branches is
being tested.  An absolute threshold would have to be re-tuned per family and
would silently mis-fire on the first checkpoint trained with a different
learning rate.

Proximity alone cannot remove the **grandparent** (``smollm2-135m-root`` at
0.0007 is *closer* than the true parent ``smollm2-cpt`` at 0.0012, because a
scar-free branch delta is small); that is what
:func:`transitive_reduction` is for.
"""

#: Smallest sampled block (rows x cols) a tensor must contribute before it is
#: allowed into a :func:`relative_delta` measurement. Small tensors (norms,
#: biases, per-head projections in tiny models) are dominated by their own
#: initialisation noise, so including them would blur the very contrast the
#: proximity gate depends on.
MIN_GATE_TENSOR_PARAMS: int = 65536

#: Loader keyword arguments :func:`relative_delta` is allowed to forward to
#: :func:`stemma.remote_loader.open_model`; everything else is a sibling
#: module's knob and is dropped rather than raising.
_LOADER_KW: frozenset = frozenset({"revision", "token", "cache_dir", "session"})

#: Number of (role, depth) slots in a sketch presence mask.
N_SLOTS = len(ROLES) * len(DEPTH_BUCKETS)

#: Relation labels a single-parent edge can carry, in classification priority
#: order. ``merge`` is assigned separately by the decomposer.
RELATIONS: Tuple[str, ...] = (
    "quantized",
    "pruned",
    "vocab_extended",
    "finetuned",
    "derived",
)

#: Feature thresholds used by :func:`relation_from_evidence`. Values are on the
#: scale of ``DIRECTION_FEATURES`` (antisymmetric, roughly [-1, 1]).
_RELATION_THRESHOLDS: Dict[str, float] = {
    "lattice_asym": 0.15,
    "dtype_precision_asym": 0.15,
    "zero_subset_asym": 0.15,
    "zero_asym": 0.20,
    "vocab_delta": 0.05,
    "orphan_asym": 0.15,
    "subspace_energy_asym": 0.05,
    "norm_growth_asym": 0.05,
    "delta_rank_asym": 0.10,
    "spectral_growth_asym": 0.05,
}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def _call_tolerant(fn, *args, **kwargs):
    """Call ``fn`` dropping keyword arguments it does not accept.

    Claim: infra -- sibling modules evolve independently, so the graph builder
    must survive a collaborator that has not yet grown a particular knob rather
    than aborting a whole lineage reconstruction over one keyword.
    """
    kw = dict(kwargs)
    while True:
        try:
            return fn(*args, **kw)
        except TypeError as exc:  # pragma: no cover - depends on sibling sigs
            m = re.search(r"unexpected keyword argument '([^']+)'", str(exc))
            if not m or m.group(1) not in kw:
                raise
            kw.pop(m.group(1))
            log.debug("dropping unsupported kwarg %r for %s", m.group(1), getattr(fn, "__name__", fn))


def _as_vector(s: Sketch | np.ndarray | Sequence[float]) -> np.ndarray:
    v = s.vector if isinstance(s, Sketch) else s
    v = np.asarray(v, dtype=np.float32).ravel()
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, dtype=np.float32))
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return (X / n).astype(np.float32)


def _display_name(ref: ModelRef) -> str:
    """Human-friendly short label for a model reference."""
    ref = str(ref)
    if is_local_path(ref) or ("/" in ref and Path(ref).exists()):
        name = Path(ref).name or ref
    else:
        name = ref
    return short_id(name, 34)


def _sanitize_id(ref: ModelRef) -> str:
    return "n_" + re.sub(r"[^A-Za-z0-9_]", "_", str(ref))


def _node_id_map(nodes: Sequence[ModelRef]) -> Dict[ModelRef, str]:
    """Stable, collision-free ``[A-Za-z0-9_]`` ids for graph export."""
    out: Dict[ModelRef, str] = {}
    seen: Dict[str, int] = {}
    for n in nodes:
        base = _sanitize_id(n)
        if base in seen:
            seen[base] += 1
            base = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
        out[n] = base
    return out


def _conflict_nodes(conflicts: Iterable[Any]) -> set:
    """Model ids touched by a rights conflict (descendant + ancestor)."""
    out: set = set()
    for c in conflicts or ():
        if isinstance(c, Mapping):
            d, a = c.get("descendant"), c.get("ancestor")
        else:
            d, a = getattr(c, "descendant", None), getattr(c, "ancestor", None)
        if d:
            out.add(d)
        if a:
            out.add(a)
    return out


def _escape_label(text: str) -> str:
    return str(text).replace("\\", "/").replace('"', "'").replace("[", "(").replace("]", ")")


# --------------------------------------------------------------------------- #
# SketchIndex
# --------------------------------------------------------------------------- #


class SketchIndex:
    """Nearest-neighbour index over model sketches (faiss when importable).

    Claim: low-false-positive -- retrieval is done on the permutation- and
    rescaling-invariant sketch under a cosine metric, so architecture twins that
    share no weights land far apart instead of being proposed as parents.

    The metric is cosine, implemented as inner product over L2-normalised
    vectors so that the faiss and numpy paths return byte-identical rankings.
    ``search`` returns *distances* (``1 - cosine``, in ``[0, 2]``), lower is
    closer, matching :func:`stemma.sketch.sketch_distance`'s convention.
    """

    def __init__(
        self,
        dim: int = SKETCH_DIM,
        metric: str = "cosine",
        *,
        backend: Optional[str] = None,
    ) -> None:
        """Create an empty index of ``dim``-dimensional sketch vectors.

        Claim: low-false-positive -- fixing the dimensionality and metric here
        is what makes candidate scores comparable across whole model universes.

        ``backend`` is an additive escape hatch ("numpy" forces the pure-numpy
        brute-force path); it defaults to "faiss if importable".
        """
        metric = str(metric).lower()
        if metric not in ("cosine", "ip", "l2"):
            raise ValueError(f"unsupported metric {metric!r} (cosine|ip|l2)")
        self.dim = int(dim)
        self.metric = metric
        self._ids: List[ModelRef] = []
        self._pos: Dict[ModelRef, int] = {}
        self._vecs: np.ndarray = np.zeros((0, self.dim), dtype=np.float32)
        self._unit: np.ndarray = np.zeros((0, self.dim), dtype=np.float32)
        self._present: np.ndarray = np.zeros((0, N_SLOTS), dtype=bool)
        self._meta: Dict[ModelRef, Dict[str, Any]] = {}
        self._requested_backend = backend
        self._faiss_index = None
        self._backend = "numpy"
        if backend in (None, "faiss"):
            self._faiss_index = self._new_faiss_index()
            if self._faiss_index is not None:
                self._backend = "faiss"
        elif backend != "numpy":
            raise ValueError(f"unknown backend {backend!r} (faiss|numpy|None)")

    # -- construction ------------------------------------------------------ #

    def _new_faiss_index(self):
        try:
            import faiss  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on environment
            log.debug("faiss unavailable (%s); using numpy brute force", exc)
            return None
        try:
            if self.metric == "l2":
                return faiss.IndexFlatL2(self.dim)
            return faiss.IndexFlatIP(self.dim)
        except Exception as exc:  # pragma: no cover
            log.warning("faiss index construction failed (%s); using numpy", exc)
            return None

    @property
    def backend(self) -> str:
        """Which nearest-neighbour implementation is actually in use.

        Claim: infra -- the benchmark records this so timings are interpretable.
        """
        return self._backend

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def ids(self) -> List[ModelRef]:
        """Model ids currently held by the index, in insertion order.

        Claim: infra.
        """
        return list(self._ids)

    def add(self, sketches: Sequence[Sketch]) -> None:
        """Insert sketches, skipping ids already present.

        Claim: low-transfer -- an index is built once from cached sketches, so
        adding a candidate universe costs no additional bytes over the wire.
        """
        if isinstance(sketches, Sketch):  # tolerate a single sketch
            sketches = [sketches]
        fresh_vecs: List[np.ndarray] = []
        fresh_present: List[np.ndarray] = []
        for s in sketches:
            if not isinstance(s, Sketch):
                raise TypeError(f"SketchIndex.add expects Sketch objects, got {type(s)!r}")
            if s.model_id in self._pos:
                log.debug("sketch %s already indexed; skipping", s.model_id)
                continue
            v = _as_vector(s)
            if v.size != self.dim:
                if v.size < self.dim:
                    v = np.concatenate([v, np.zeros(self.dim - v.size, dtype=np.float32)])
                else:
                    v = v[: self.dim]
                log.warning("sketch %s had dim %d, coerced to %d", s.model_id, s.vector.size, self.dim)
            pres = np.asarray(s.present, dtype=bool).ravel()
            if pres.size != N_SLOTS:
                pres = np.resize(pres, N_SLOTS) if pres.size else np.zeros(N_SLOTS, dtype=bool)
            self._pos[s.model_id] = len(self._ids)
            self._ids.append(s.model_id)
            self._meta[s.model_id] = dict(s.meta or {})
            fresh_vecs.append(v)
            fresh_present.append(pres)
        if not fresh_vecs:
            return
        block = np.vstack(fresh_vecs).astype(np.float32)
        self._vecs = np.vstack([self._vecs, block]) if self._vecs.size else block
        unit = _l2_normalize(block) if self.metric != "l2" else block
        self._unit = np.vstack([self._unit, unit]) if self._unit.size else unit
        pblock = np.vstack(fresh_present)
        self._present = np.vstack([self._present, pblock]) if self._present.size else pblock
        if self._faiss_index is not None:
            try:
                self._faiss_index.add(np.ascontiguousarray(unit, dtype=np.float32))
            except Exception as exc:  # pragma: no cover
                log.warning("faiss add failed (%s); falling back to numpy", exc)
                self._faiss_index = None
                self._backend = "numpy"

    # -- query ------------------------------------------------------------- #

    def vector_of(self, model_id: ModelRef) -> np.ndarray:
        """Return the stored (un-normalised) vector for ``model_id``.

        Claim: infra.
        """
        if model_id not in self._pos:
            raise KeyError(f"{model_id!r} is not in this SketchIndex")
        return self._vecs[self._pos[model_id]].copy()

    def sketch_of(self, model_id: ModelRef) -> Sketch:
        """Rebuild a minimal :class:`Sketch` from indexed data.

        Claim: infra -- lets callers pass a bare model id where a sketch is
        expected without re-reading any weights.
        """
        i = self._pos[model_id] if model_id in self._pos else None
        if i is None:
            raise KeyError(f"{model_id!r} is not in this SketchIndex")
        return Sketch(
            model_id=model_id,
            vector=self._vecs[i].copy(),
            present=self._present[i].copy(),
            meta=dict(self._meta.get(model_id, {})),
        )

    def search(self, q: Sketch, k: int = 10) -> List[Tuple[str, float]]:
        """Return the ``k`` nearest indexed models as ``(model_id, distance)``.

        Claim: low-false-positive -- cosine distance on invariant sketches is
        the cheap first filter; anything it puts far away never reaches the
        expensive (and much more decisive) direction stage.
        """
        if len(self._ids) == 0:
            return []
        k = int(max(1, min(k, len(self._ids))))
        if isinstance(q, str):
            q = self.sketch_of(q)
        qv = _as_vector(q)
        if qv.size != self.dim:
            qv = np.resize(qv, self.dim)
        qu = _l2_normalize(qv[None, :])[0] if self.metric != "l2" else qv

        idx: np.ndarray
        score: np.ndarray
        if self._faiss_index is not None and self._faiss_index.ntotal == len(self._ids):
            try:
                score, idx = self._faiss_index.search(
                    np.ascontiguousarray(qu[None, :], dtype=np.float32), k
                )
                score, idx = score[0], idx[0]
            except Exception as exc:  # pragma: no cover
                log.warning("faiss search failed (%s); using numpy", exc)
                self._faiss_index = None
                self._backend = "numpy"
                return self.search(q, k)
        else:
            if self.metric == "l2":
                d = np.linalg.norm(self._unit - qu[None, :], axis=1)
                idx = np.argsort(d, kind="stable")[:k]
                score = d[idx]
            else:
                sims = self._unit @ qu
                idx = np.argsort(-sims, kind="stable")[:k]
                score = sims[idx]

        out: List[Tuple[str, float]] = []
        for i, sc in zip(idx, score):
            if i < 0 or i >= len(self._ids):
                continue
            if self.metric == "l2":
                dist = float(sc if self._faiss_index is None else math.sqrt(max(0.0, float(sc))))
            else:
                dist = float(np.clip(1.0 - float(sc), 0.0, 2.0))
            out.append((self._ids[int(i)], dist))
        out.sort(key=lambda t: (t[1], t[0]))
        return out

    # -- persistence ------------------------------------------------------- #

    @staticmethod
    def _paths(path) -> Tuple[Path, Path]:
        p = Path(path)
        base = p.with_suffix("") if p.suffix in (".npz", ".json") else p
        return base.with_suffix(".npz"), base.with_suffix(".json")

    def save(self, path) -> None:
        """Persist vectors to ``<path>.npz`` and the id list to ``<path>.json``.

        Claim: low-transfer -- a saved index means a universe of candidates is
        sketched once and reused for every later query, amortising the only
        bytes Stemma ever reads.
        """
        npz_path, json_path = self._paths(path)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_path,
            vectors=self._vecs.astype(np.float32),
            present=self._present.astype(bool),
        )
        atomic_write_json(
            json_path,
            {
                "version": INDEX_VERSION,
                "dim": self.dim,
                "metric": self.metric,
                "backend": self._backend,
                "ids": list(self._ids),
                "meta": self._meta,
                "npz": npz_path.name,
            },
        )
        log.debug("saved index of %d sketches to %s", len(self._ids), npz_path)

    @classmethod
    def load(cls, path) -> "SketchIndex":
        """Restore an index written by :meth:`save`.

        Claim: low-transfer.
        """
        npz_path, json_path = cls._paths(path)
        with open(json_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        if manifest.get("version") != INDEX_VERSION:
            log.warning(
                "index version mismatch: file %s, code %s",
                manifest.get("version"),
                INDEX_VERSION,
            )
        stored = npz_path
        if not stored.exists() and manifest.get("npz"):
            stored = json_path.parent / str(manifest["npz"])
        data = np.load(stored, allow_pickle=False)
        vecs = np.asarray(data["vectors"], dtype=np.float32)
        present = np.asarray(data["present"], dtype=bool)
        idx = cls(dim=int(manifest.get("dim", vecs.shape[1] if vecs.size else SKETCH_DIM)),
                  metric=str(manifest.get("metric", "cosine")))
        meta = manifest.get("meta", {}) or {}
        sketches = [
            Sketch(
                model_id=mid,
                vector=vecs[i],
                present=present[i] if i < len(present) else np.zeros(N_SLOTS, dtype=bool),
                meta=dict(meta.get(mid, {})),
            )
            for i, mid in enumerate(manifest.get("ids", []))
        ]
        idx.add(sketches)
        return idx


# --------------------------------------------------------------------------- #
# candidate retrieval
# --------------------------------------------------------------------------- #


def find_candidate_parents(
    target: Sketch | ModelRef,
    index: SketchIndex,
    *,
    k: int = 10,
    max_distance: float = 0.35,
) -> List[Tuple[str, float]]:
    """Retrieve plausible relatives of ``target`` from ``index``.

    Claim: low-false-positive -- the ``max_distance`` gate on invariant sketch
    cosine distance is the first of two independent filters (the second being
    the raw-tensor relatedness score), which is what keeps unrelated models from
    ever being offered as parents.

    The target itself is always removed from its own candidate list.
    """
    if index is None or len(index) == 0:
        return []
    q = target if isinstance(target, Sketch) else index.sketch_of(target)
    tid = q.model_id
    hits = index.search(q, k=int(k) + 1)
    out = [(mid, d) for mid, d in hits if mid != tid and d <= float(max_distance)]
    return out[: int(k)]


# --------------------------------------------------------------------------- #
# proximity gate: one branch delta, not two
# --------------------------------------------------------------------------- #


def _flatten_rows(a: np.ndarray) -> np.ndarray:
    """Collapse everything after the row axis so tensors compare as matrices."""
    arr = np.asarray(a)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(arr.shape[0], 1)
    return arr.reshape(arr.shape[0], -1)


def _shared_large_tensors(
    idx_a: Mapping[str, Any], idx_b: Mapping[str, Any], n_tensors: int
) -> List[Tuple[str, int]]:
    """Shared 2-D tensors big enough to measure, largest first.

    Returns ``(name, comparable_rows)`` pairs. ``comparable_rows`` is the row
    count both models have, so a vocabulary-extended candidate is still
    comparable over the rows it shares with the child.
    """
    out: List[Tuple[Tuple[int, str], str, int]] = []
    for name in sorted(set(idx_a) & set(idx_b)):
        sa = tuple(int(d) for d in (getattr(idx_a[name], "shape", ()) or ()))
        sb = tuple(int(d) for d in (getattr(idx_b[name], "shape", ()) or ()))
        if len(sa) != 2 or len(sb) != 2 or sa[1] != sb[1]:
            continue
        rows = min(sa[0], sb[0])
        cols = sa[1]
        if rows < 2 or cols < 2 or rows * cols < MIN_GATE_TENSOR_PARAMS:
            continue
        out.append(((-(rows * cols), name), name, rows))
    out.sort(key=lambda t: t[0])
    return [(name, rows) for _key, name, rows in out[: int(max(1, n_tensors))]]


def _open_for_gate(ref: ModelRef, **loader_kw: Any) -> Any:
    """Open a checkpoint through the Range loader, dropping foreign kwargs."""
    from .remote_loader import open_model  # lazy: no network at import time

    kw = {k: v for k, v in loader_kw.items() if k in _LOADER_KW}
    return open_model(str(ref), **kw)


def _relative_delta_core(
    src_child: Any,
    src_cand: Any,
    *,
    max_rows: int,
    n_tensors: int,
    row_cache: Optional[Dict[Tuple[str, int], np.ndarray]] = None,
) -> float:
    """``||cand - child||_F / ||cand||_F`` over identically sampled rows."""
    from .remote_loader import select_rows  # lazy: no network at import time

    idx_c, idx_p = src_child.index(), src_cand.index()
    names = _shared_large_tensors(idx_c, idx_p, n_tensors)
    if not names:
        # No comparable tensor at all: a different architecture cannot be a
        # *direct* parent, whatever a symmetric fingerprint says about it.
        return float("inf")

    num = 0.0
    den = 0.0
    used = 0
    for name, rows_shared in names:
        rows = select_rows(int(rows_shared), int(max_rows))
        if rows.size == 0:
            continue
        key = (name, int(rows_shared))
        child_block = None if row_cache is None else row_cache.get(key)
        if child_block is None:
            child_block = _flatten_rows(
                src_child.get_tensor_rows(name, rows, dtype=np.float32)
            ).astype(np.float64, copy=False)
            if row_cache is not None:
                row_cache[key] = child_block
        cand_block = _flatten_rows(
            src_cand.get_tensor_rows(name, rows, dtype=np.float32)
        ).astype(np.float64, copy=False)
        if child_block.shape != cand_block.shape or cand_block.size == 0:
            continue
        d = cand_block - child_block
        num += float(np.einsum("ij,ij->", d, d))
        den += float(np.einsum("ij,ij->", cand_block, cand_block))
        used += 1

    if used == 0 or den <= 0.0:
        return float("inf")
    return float(math.sqrt(max(num, 0.0)) / math.sqrt(den))


def relative_delta(
    child: ModelRef,
    candidate: ModelRef,
    *,
    max_rows: int = 256,
    n_tensors: int = 10,
    seed: int = 0,
    **loader_kw: Any,
) -> float:
    """How far ``candidate``'s weights sit from ``child``'s, as a pure ratio.

    Claim: low-false-positive -- this is the measurement the candidate-retrieval
    guard is built on: a direct child differs from its parent by *one* branch
    delta, a cousin by two or more, so a candidate whose delta is far larger than
    the closest one's cannot be the direct parent no matter how confidently the
    direction estimator can orient the pair.

    Computes ``||candidate - child||_F / ||candidate||_F`` over the shared 2-D
    tensors of at least :data:`MIN_GATE_TENSOR_PARAMS` sampled parameters,
    largest first, capped at ``n_tensors`` tensors x ``max_rows`` rows. Both
    models are read at the **same** row indices
    (:func:`stemma.remote_loader.select_rows`), so the number is comparable
    across candidates rather than being an artefact of which rows were sampled.

    Returns ``float('inf')`` when the two models share no comparable tensor: a
    different architecture may well be an *ancestor* (distillation), but it
    cannot be a direct weight-space parent, and ``inf`` is how that is said.

    ``seed`` is accepted and recorded for signature symmetry with the rest of
    the package; row selection here is deterministic by construction (it depends
    only on the row count and the budget), so the value never changes the
    result. Reads go through :mod:`stemma.remote_loader`, so the work stays a
    Range read and every byte lands in the caller's transfer accounting.
    """
    value, _stats = _relative_delta_with_stats(
        child, candidate, max_rows=max_rows, n_tensors=n_tensors, seed=seed, **loader_kw
    )
    return value


def _relative_delta_with_stats(
    child: ModelRef,
    candidate: ModelRef,
    *,
    max_rows: int = 256,
    n_tensors: int = 10,
    seed: int = 0,
    **loader_kw: Any,
) -> Tuple[float, TransferStats]:
    """:func:`relative_delta` plus the bytes it moved, for byte accounting."""
    del seed  # deterministic sampling; kept in the public signature only
    stats = TransferStats()
    src_c = src_p = None
    try:
        src_c = _open_for_gate(child, **loader_kw)
        src_p = _open_for_gate(candidate, **loader_kw)
        value = _relative_delta_core(
            src_c, src_p, max_rows=int(max_rows), n_tensors=int(n_tensors)
        )
    finally:
        for src in (src_c, src_p):
            if src is None:
                continue
            st = getattr(src, "stats", None)
            if isinstance(st, TransferStats):
                stats = stats.add(st)
            try:
                src.close()
            except Exception:  # pragma: no cover - defensive
                pass
    return value, stats


def proximity_gate(
    child: ModelRef,
    candidates: Sequence[ModelRef],
    *,
    factor: float = PROXIMITY_FACTOR,
    max_rows: int = 256,
    seed: int = 0,
    **loader_kw: Any,
) -> Tuple[List[str], Dict[str, float]]:
    """Keep only candidates close enough to ``child`` to be its *direct* parent.

    Claim: low-false-positive -- retrieval, not orientation, was the weak link:
    a cousin that carries a quantisation or pruning scar produces a *confident*
    direction verdict about a pair that is not an edge at all, and a confident
    wrong edge beat a correct-but-abstaining one in the DAG builder. Gating on
    proximity removes those pairs before they can be oriented.

    Computes :func:`relative_delta` for every candidate, takes ``m`` = the
    smallest finite value, and keeps the candidates with
    ``delta <= m * factor``. See :data:`PROXIMITY_FACTOR` for the measurement
    behind the default factor and for why a *ratio* (not an absolute threshold)
    is the scale-free choice.

    Returns ``(kept, deltas)`` where ``deltas`` maps **every** candidate to its
    measured delta -- including the rejected ones, so the decision is auditable
    rather than silent. A candidate whose delta is ``inf`` (no shared tensor, or
    a read that failed) is never kept, whatever the factor: it cannot be a
    direct weight-space parent. Passing ``factor <= 0`` or ``None`` widens the
    gate to "any finite delta"; disabling the stage outright is
    ``build_phylogeny(..., proximity_factor=0)``.
    """
    kept, deltas, _stats = _proximity_gate_with_stats(
        child, candidates, factor=factor, max_rows=max_rows, seed=seed, **loader_kw
    )
    return kept, deltas


def _proximity_gate_with_stats(
    child: ModelRef,
    candidates: Sequence[ModelRef],
    *,
    factor: float = PROXIMITY_FACTOR,
    max_rows: int = 256,
    n_tensors: int = 10,
    seed: int = 0,
    **loader_kw: Any,
) -> Tuple[List[str], Dict[str, float], TransferStats]:
    """:func:`proximity_gate` plus the bytes it moved, for byte accounting."""
    del seed  # deterministic sampling; kept in the public signature only
    cands: List[str] = []
    for c in candidates or ():
        cid = str(c)
        if cid != str(child) and cid not in cands:
            cands.append(cid)

    deltas: Dict[str, float] = {}
    stats = TransferStats()
    if not cands:
        return [], deltas, stats

    src_c = None
    row_cache: Dict[Tuple[str, int], np.ndarray] = {}
    try:
        src_c = _open_for_gate(child, **loader_kw)
        for cid in cands:
            src_p = None
            try:
                src_p = _open_for_gate(cid, **loader_kw)
                deltas[cid] = _relative_delta_core(
                    src_c,
                    src_p,
                    max_rows=int(max_rows),
                    n_tensors=int(n_tensors),
                    row_cache=row_cache,
                )
            except Exception as exc:
                log.warning("relative_delta(%s, %s) failed: %s", child, cid, exc)
                deltas[cid] = float("inf")
            finally:
                if src_p is not None:
                    st = getattr(src_p, "stats", None)
                    if isinstance(st, TransferStats):
                        stats = stats.add(st)
                    try:
                        src_p.close()
                    except Exception:  # pragma: no cover - defensive
                        pass
    finally:
        if src_c is not None:
            st = getattr(src_c, "stats", None)
            if isinstance(st, TransferStats):
                stats = stats.add(st)
            try:
                src_c.close()
            except Exception:  # pragma: no cover - defensive
                pass

    finite = [v for v in deltas.values() if math.isfinite(v)]
    if not finite:
        log.debug("proximity gate: no candidate of %s shares a comparable tensor", child)
        return [], deltas, stats

    best = min(finite)
    if factor is None or float(factor) <= 0.0:
        limit = float("inf")
    else:
        # A best of exactly 0.0 (a bit-identical candidate) would collapse the
        # window to {0.0}; fall back to an absolute floor so a re-upload does
        # not evict every genuine parent.
        limit = max(best * float(factor), 1e-12)
    kept = [c for c in cands if math.isfinite(deltas[c]) and deltas[c] <= limit]
    return kept, deltas, stats


# --------------------------------------------------------------------------- #
# relation classification
# --------------------------------------------------------------------------- #


def relation_from_evidence(verdict: Any) -> str:
    """Label a single-parent edge from the direction verdict's features.

    Claim: direction -- the same asymmetric evidence that orients an edge also
    says *what kind* of derivation it was (quantisation, pruning, vocabulary
    extension, plain fine-tuning), which no symmetric similarity can report.
    """
    direction = str(getattr(verdict, "direction", "unknown"))
    feats = dict(getattr(verdict, "features", {}) or {})
    if direction == "b->a":
        sign = -1.0
    elif direction == "a->b":
        sign = 1.0
    else:
        return "derived"
    # Orient every antisymmetric feature so that "positive" means
    # "the child side carries the scar".
    sf = {}
    for kk, vv in feats.items():
        try:
            sf[kk] = float(vv) * sign
        except (TypeError, ValueError):
            continue
    t = _RELATION_THRESHOLDS

    def hit(name: str) -> bool:
        return sf.get(name, 0.0) >= t.get(name, 0.1)

    if hit("lattice_asym") or hit("dtype_precision_asym"):
        return "quantized"
    if hit("zero_subset_asym") or hit("zero_asym"):
        return "pruned"
    if hit("vocab_delta") or hit("orphan_asym"):
        return "vocab_extended"
    if (
        hit("subspace_energy_asym")
        or hit("norm_growth_asym")
        or hit("delta_rank_asym")
        or hit("spectral_growth_asym")
    ):
        return "finetuned"
    return "derived"


# --------------------------------------------------------------------------- #
# cycle handling
# --------------------------------------------------------------------------- #


def _find_cycle(edges: Sequence[Edge], nodes: Sequence[ModelRef]) -> Optional[List[Tuple[ModelRef, ModelRef]]]:
    """Return one directed cycle as a list of (parent, child) pairs, or None."""
    adj: Dict[ModelRef, List[ModelRef]] = {n: [] for n in nodes}
    for e in edges:
        adj.setdefault(e.parent, []).append(e.child)
        adj.setdefault(e.child, adj.get(e.child, []))
    try:  # networkx is optional
        import networkx as nx  # type: ignore

        g = nx.DiGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from((e.parent, e.child) for e in edges)
        try:
            cyc = nx.find_cycle(g, orientation="original")
        except nx.NetworkXNoCycle:
            return None
        return [(u, v) for (u, v, *_rest) in cyc]
    except ImportError:
        pass

    color: Dict[ModelRef, int] = {n: 0 for n in adj}
    for start in list(adj):
        if color[start] != 0:
            continue
        color[start] = 1
        stack: List[Tuple[ModelRef, Any]] = [(start, iter(adj[start]))]
        while stack:
            u, it = stack[-1]
            advanced = False
            for v in it:
                if color.get(v, 0) == 1:
                    path = [v]
                    for w, _ in reversed(stack):
                        path.append(w)
                        if w == v:
                            break
                    path.reverse()
                    if path[-1] != v:
                        path.append(v)
                    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                if color.get(v, 0) == 0:
                    color[v] = 1
                    stack.append((v, iter(adj.get(v, []))))
                    advanced = True
                    break
            if not advanced:
                color[u] = 2
                stack.pop()
    return None


def break_cycles(edges: List[Edge], nodes: Sequence[ModelRef], *, max_iters: int = 1000) -> Tuple[List[Edge], List[Edge]]:
    """Remove the lowest-confidence edge of each cycle until the graph is a DAG.

    Claim: direction -- a lineage must be acyclic to be believable; when two
    edges disagree we keep the one the direction evidence was more sure about,
    which is precisely the quantity a symmetric method never has.
    """
    kept = list(edges)
    dropped: List[Edge] = []
    for _ in range(int(max_iters)):
        cyc = _find_cycle(kept, nodes)
        if not cyc:
            break
        on_cycle = [e for e in kept if (e.parent, e.child) in set(cyc)]
        if not on_cycle:  # pragma: no cover - defensive
            break
        weakest = min(on_cycle, key=lambda e: (float(e.confidence), e.parent, e.child))
        kept = [e for e in kept if e is not weakest]
        dropped.append(weakest)
        log.info(
            "cycle broken: dropped %s -> %s (confidence %.3f)",
            weakest.parent,
            weakest.child,
            weakest.confidence,
        )
    else:  # pragma: no cover - pathological graphs only
        log.warning("cycle breaking hit the iteration cap; graph may still contain cycles")
    return kept, dropped


# --------------------------------------------------------------------------- #
# sketching / caching
# --------------------------------------------------------------------------- #


def _cache_key(ref: ModelRef, seed: int, extra: Mapping[str, Any]) -> str:
    stamp: Any = None
    try:
        if is_local_path(ref):
            p = Path(str(ref)).expanduser()
            stamp = int(p.stat().st_mtime) if p.exists() else None
    except OSError:  # pragma: no cover
        stamp = None
    return stable_hash(
        {
            "ref": str(ref),
            "seed": int(seed),
            "mtime": stamp,
            "kw": {k: str(v) for k, v in sorted(extra.items())},
        }
    )


def sketch_universe(
    models: Sequence[ModelRef | Sketch],
    *,
    sketches: Optional[Dict[ModelRef, Sketch]] = None,
    cache_dir: Optional[str] = None,
    seed: int = 0,
    **kw: Any,
) -> Tuple[Dict[ModelRef, Sketch], TransferStats, List[ModelRef]]:
    """Sketch every model once, reusing in-memory and (optionally) disk caches.

    Claim: low-transfer -- each model in the universe is Range-read exactly once
    no matter how many pairwise comparisons the phylogeny later needs.
    """
    from . import sketch as sketch_mod  # lazy: keeps import-time cheap

    out: Dict[ModelRef, Sketch] = dict(sketches or {})
    order: List[ModelRef] = []
    total = TransferStats()
    disk = Path(cache_dir).expanduser() if cache_dir else None
    if disk is not None:
        disk.mkdir(parents=True, exist_ok=True)

    for m in models:
        if isinstance(m, Sketch):
            out[m.model_id] = m
            order.append(m.model_id)
            if m.stats:
                total = total.add(m.stats)
            continue
        ref = str(m)
        order.append(ref)
        if ref in out:
            continue
        cached: Optional[Sketch] = None
        cache_path: Optional[Path] = None
        if disk is not None:
            cache_path = disk / f"sketch-{_cache_key(ref, seed, kw)}.json"
            if cache_path.exists():
                try:
                    with open(cache_path, "r", encoding="utf-8") as fh:
                        cached = Sketch.from_json(json.load(fh))
                    log.debug("sketch cache hit for %s", ref)
                except Exception as exc:  # pragma: no cover - corrupt cache
                    log.warning("ignoring unreadable sketch cache %s (%s)", cache_path, exc)
                    cached = None
        if cached is None:
            cached = _call_tolerant(sketch_mod.sketch_model, ref, seed=seed, **kw)
            if cache_path is not None:
                try:
                    atomic_write_json(cache_path, cached.to_json())
                except Exception as exc:  # pragma: no cover
                    log.warning("could not write sketch cache %s (%s)", cache_path, exc)
        out[ref] = cached
        if cached.stats:
            total = total.add(cached.stats)
    return out, total, order


# --------------------------------------------------------------------------- #
# the main pipeline
# --------------------------------------------------------------------------- #


def build_phylogeny(
    models: Sequence[ModelRef | Sketch],
    *,
    index: Optional[SketchIndex] = None,
    relatedness_threshold: float = 0.6,
    direction_abstain: float = 0.5,
    merge_check: bool = True,
    direction_model: Any = None,
    k: int = 10,
    max_distance: float = 0.35,
    sketches: Optional[Dict[ModelRef, Sketch]] = None,
    cache_dir: Optional[str] = None,
    seed: int = 0,
    support_threshold: float = 0.05,
    proximity_factor: float = 0.0,  # OFF by default -- see PROXIMITY_FACTOR
    reduce_transitive: bool = True,
    cousin_veto_enabled: bool = True,
    auto_outgroup: int = 0,  # OFF -- see _pick_outgroups
    **kw: Any,
) -> Phylogeny:
    """Reconstruct a multi-parent lineage DAG over ``models``.

    Claim: direction -- this is the end-to-end demonstration of the headline
    claim: retrieve with a symmetric sketch, reject unrelated pairs, then orient
    every surviving pair with asymmetric weight evidence and split multi-parent
    nodes into mixing coefficients.

    Pipeline: sketch -> index -> k-NN candidates -> relatedness gate ->
    ``estimate_direction`` -> merge decomposition for nodes with >= 2 parents ->
    cycle breaking -> roots + transfer accounting.
    """
    t0 = time.perf_counter()
    from . import direction as direction_mod  # lazy imports keep import-time clean

    all_sketches, sketch_stats, node_order = sketch_universe(
        models, sketches=sketches, cache_dir=cache_dir, seed=seed, **kw
    )
    nodes: List[ModelRef] = []
    for n in node_order:
        if n not in nodes:
            nodes.append(n)

    if index is None:
        index = SketchIndex(dim=SKETCH_DIM, metric="cosine")
    missing = [all_sketches[n] for n in nodes if n not in set(index.ids)]
    if missing:
        index.add(missing)

    # ---- (1)(2) candidate retrieval + relatedness gate ------------------- #
    pairs: Dict[Tuple[ModelRef, ModelRef], float] = {}
    for n in nodes:
        for cand, dist in find_candidate_parents(
            all_sketches[n], index, k=k, max_distance=max_distance
        ):
            if cand == n or cand not in all_sketches:
                continue
            key = (n, cand) if str(n) <= str(cand) else (cand, n)
            pairs[key] = min(dist, pairs.get(key, dist))

    total_stats = TransferStats(
        bytes_read=sketch_stats.bytes_read,
        requests=sketch_stats.requests,
        seconds=sketch_stats.seconds,
        full_size_bytes=sketch_stats.full_size_bytes,
        cache_hits=sketch_stats.cache_hits,
    )
    related: List[Tuple[ModelRef, ModelRef, float]] = []
    relatedness: Dict[str, float] = {}
    for (a, b), dist in sorted(pairs.items()):
        try:
            score = float(
                _call_tolerant(
                    direction_mod.relatedness_score,
                    a,
                    b,
                    sa=all_sketches.get(a),
                    sb=all_sketches.get(b),
                    seed=seed,
                    **kw,
                )
            )
        except Exception as exc:
            log.warning("relatedness_score(%s, %s) failed: %s", a, b, exc)
            continue
        relatedness[f"{a}||{b}"] = score
        if score >= float(relatedness_threshold):
            related.append((a, b, score))
        else:
            log.debug("dropping unrelated pair %s / %s (score %.3f)", a, b, score)

    # ---- (2b) proximity gate --------------------------------------------- #
    # ORDERING IS THE FIX. This must run BEFORE orientation, never after: the
    # shipped bug (README limitation #9) was that a cousin carrying a pruning or
    # quantisation scar produces a *confident* direction verdict, which then beat
    # the correct-but-abstaining true parent. Gating here means such a pair never
    # reaches estimate_direction at all. Running it afterwards would leave the
    # confident-wrong edge already in hand and fix nothing.
    gate_info: Dict[str, Any] = {"factor": float(proximity_factor or 0.0),
                                 "dropped": {}, "kept": {}}
    if proximity_factor and float(proximity_factor) > 0.0 and related:
        by_node: Dict[ModelRef, List[ModelRef]] = {}
        for a, b, _s in related:
            by_node.setdefault(a, []).append(b)
            by_node.setdefault(b, []).append(a)
        survivors: set[Tuple[ModelRef, ModelRef]] = set()
        for node, cands in by_node.items():
            try:
                kept_c, deltas, gstats = _proximity_gate_with_stats(
                    node, cands, factor=float(proximity_factor), seed=seed, **kw
                )
            except Exception as exc:
                log.warning("proximity gate failed for %s: %s", node, exc)
                survivors.update((node, c) for c in cands)
                continue
            total_stats = total_stats.add(gstats)
            gate_info["kept"][str(node)] = {c: float(deltas.get(c, float("nan")))
                                            for c in kept_c}
            dropped = [c for c in cands if c not in kept_c]
            if dropped:
                gate_info["dropped"][str(node)] = [
                    (str(c), float(deltas.get(c, float("inf")))) for c in dropped
                ]
            survivors.update((node, c) for c in kept_c)
        # A pair survives if it is plausible from EITHER endpoint's perspective.
        # Requiring both is wrong and was measured to be: orientation has not run
        # yet, so we do not know which model is the child, and the gate's claim is
        # only "X could be a direct parent of Y" -- which need hold in one
        # direction. Requiring both drove the benchmark trace from 2 wrong parents
        # to 0 parents, discarding the true ones as well.
        before = len(related)
        related = [
            (a, b, s) for (a, b, s) in related
            if (a, b) in survivors or (b, a) in survivors
        ]
        log.debug("proximity gate: %d -> %d candidate pairs", before, len(related))

    # ---- (2c) cousin veto ------------------------------------------------ #
    # The actual fix for README limitation #9. Like the proximity gate this must
    # run BEFORE orientation: the failure was a confident verdict about a pair
    # that is not an edge, so the pair has to be gone before it can be oriented.
    cousin_info: Dict[str, Any] = {"enabled": bool(cousin_veto_enabled),
                                   "vetoed": [], "checked": 0}
    if cousin_veto_enabled and len(related) > 1:
        universe_refs = [str(n) for n in nodes]
        survivors2: List[Tuple[ModelRef, ModelRef, float]] = []
        for a, b, sc in related:
            try:
                vetoed, ev = cousin_veto(a, b, universe_refs, seed=seed, **kw)
            except Exception as exc:
                log.warning("cousin_veto(%s, %s) failed: %s", a, b, exc)
                survivors2.append((a, b, sc))
                continue
            cousin_info["checked"] += 1
            if vetoed:
                cousin_info["vetoed"].append(
                    {"a": str(a), "b": str(b),
                     "ancestor": ev.get("ancestor"),
                     "cousin_score": ev.get("cousin_score")}
                )
            else:
                survivors2.append((a, b, sc))
        log.debug("cousin veto: %d -> %d pairs", len(related), len(survivors2))
        related = survivors2

    # ---- (3) orientation ------------------------------------------------- #
    edges: List[Edge] = []
    verdicts: Dict[Tuple[ModelRef, ModelRef], Any] = {}
    n_unknown = 0
    n_rooted = 0
    for a, b, score in related:
        # Hand the estimator an outgroup. Without one, family (e) is inactive and
        # a scar-free edge abstains by design -- which is why merge-ties2 came
        # back with "no ancestors" even though its parents were in the universe.
        outgroups = _pick_outgroups(a, b, nodes, all_sketches, int(auto_outgroup))
        try:
            v = _call_tolerant(
                direction_mod.estimate_direction,
                a,
                b,
                weights=direction_model,
                abstain=direction_abstain,
                sa=all_sketches.get(a),
                sb=all_sketches.get(b),
                outgroup=outgroups or None,
                seed=seed,
                **kw,
            )
        except Exception as exc:
            log.warning("estimate_direction(%s, %s) failed: %s", a, b, exc)
            continue
        if getattr(v, "stats", None):
            total_stats = total_stats.add(v.stats)
        d = str(getattr(v, "direction", "unknown"))
        if d not in ("a->b", "b->a"):
            n_unknown += 1
            log.debug("abstaining on %s / %s (llr %.3f)", a, b, float(getattr(v, "llr", 0.0)))
            continue
        parent, child = (a, b) if d == "a->b" else (b, a)
        conf = float(getattr(v, "confidence", 0.0) or 0.0)
        if conf <= 0.0:
            conf = abs(2.0 * float(getattr(v, "p_a_parent", 0.5)) - 1.0)
        rel = relation_from_evidence(v)
        ev = list(getattr(v, "evidence", []) or [])
        ev.append(f"sketch relatedness {score:.3f}; direction llr {float(getattr(v, 'llr', 0.0)):+.3f}")
        edges.append(
            Edge(
                parent=parent,
                child=child,
                confidence=float(np.clip(conf, 0.0, 1.0)),
                relation=rel,
                evidence=ev,
            )
        )
        verdicts[(parent, child)] = v

    # de-duplicate, keeping the most confident claim for each ordered pair
    best: Dict[Tuple[ModelRef, ModelRef], Edge] = {}
    for e in edges:
        if e.parent == e.child:
            continue
        key = (e.parent, e.child)
        if key not in best or e.confidence > best[key].confidence:
            best[key] = e
    edges = sorted(best.values(), key=lambda e: (e.child, -e.confidence, e.parent))

    # ---- (4) merge decomposition for multi-parent nodes ------------------ #
    merges: Dict[str, Any] = {}
    if merge_check:
        edges, merge_stats, merges = _resolve_merges(
            edges,
            support_threshold=support_threshold,
            seed=seed,
            kw=kw,
        )
        total_stats = total_stats.add(merge_stats)

    # ---- (5) cycle breaking ---------------------------------------------- #
    edges, dropped = break_cycles(edges, nodes)

    with_parents = {e.child for e in edges}
    roots = [n for n in nodes if n not in with_parents]

    elapsed = time.perf_counter() - t0
    meta: Dict[str, Any] = {
        "seconds": round(elapsed, 4),
        "index_backend": index.backend,
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_pairs_considered": len(pairs),
        "n_pairs_related": len(related),
        "n_direction_abstained": n_unknown,
        "n_cycle_edges_dropped": len(dropped),
        "cycle_edges_dropped": [
            {"parent": e.parent, "child": e.child, "confidence": e.confidence} for e in dropped
        ],
        "relatedness_threshold": float(relatedness_threshold),
        "direction_abstain": float(direction_abstain),
        "max_distance": float(max_distance),
        "k": int(k),
        "seed": int(seed),
        "merge_check": bool(merge_check),
        "merges": merges,
        "relatedness": relatedness,
        "transfer": _transfer_dict(total_stats),
        "disclaimer": (
            "Edges are statistical evidence about weight-level derivation, not "
            "verified provenance. Human review required."
        ),
    }
    meta["proximity_gate"] = gate_info
    meta["cousin_veto"] = cousin_info
    meta["auto_outgroup"] = int(auto_outgroup)
    result = Phylogeny(nodes=nodes, edges=edges, root_candidates=roots, meta=meta)
    if reduce_transitive:
        result = transitive_reduction(result)
    return result


def _transfer_dict(st: TransferStats) -> Dict[str, Any]:
    red = st.reduction
    return {
        "bytes_read": int(st.bytes_read),
        "bytes_read_human": human_bytes(st.bytes_read),
        "requests": int(st.requests),
        "seconds": float(st.seconds),
        "full_size_bytes": int(st.full_size_bytes),
        "full_size_human": human_bytes(st.full_size_bytes),
        "cache_hits": int(st.cache_hits),
        "reduction": None if not math.isfinite(red) else float(red),
    }


def _resolve_merges(
    edges: List[Edge],
    *,
    support_threshold: float,
    seed: int,
    kw: Mapping[str, Any],
) -> Tuple[List[Edge], TransferStats, Dict[str, Any]]:
    """Split every >=2-parent node into mixing coefficients, dropping stragglers.

    Claim: merge-recovery -- a node with two surviving parents is exactly the
    case where a pairwise method stops; the decomposer turns it into weighted
    parentage with an explicit residual.
    """
    from . import merge_decompose as merge_mod  # lazy

    stats = TransferStats()
    report: Dict[str, Any] = {}
    by_child: Dict[ModelRef, List[Edge]] = {}
    for e in edges:
        by_child.setdefault(e.child, []).append(e)

    keep: List[Edge] = []
    for child, group in by_child.items():
        if len(group) < 2:
            keep.extend(group)
            continue
        cands = [e.parent for e in group]
        try:
            dec = _call_tolerant(
                merge_mod.decompose_merge,
                child,
                cands,
                support_threshold=support_threshold,
                seed=seed,
                **dict(kw),
            )
        except Exception as exc:
            log.warning("decompose_merge(%s) failed: %s; keeping all parents", child, exc)
            keep.extend(group)
            report[str(child)] = {"error": str(exc), "candidates": cands}
            continue
        if getattr(dec, "stats", None):
            stats = stats.add(dec.stats)
        coeffs = dec.as_dict() if hasattr(dec, "as_dict") else {}
        selected = list(getattr(dec, "selected", []) or [])
        report[str(child)] = {
            "base": getattr(dec, "base", None),
            "candidates": cands,
            "coefficients": {kk: float(vv) for kk, vv in coeffs.items()},
            "selected": selected,
            "residual": float(getattr(dec, "residual", float("nan"))),
            "r2": float(getattr(dec, "r2", float("nan"))),
            "method": str(getattr(dec, "method", "")),
        }
        if not selected:
            log.warning(
                "merge decomposition for %s selected no parent; keeping all %d candidates",
                child,
                len(group),
            )
            keep.extend(group)
            report[str(child)]["note"] = "no coefficient cleared the support threshold"
            continue
        sel = set(selected)
        for e in group:
            if e.parent not in sel:
                log.debug("merge decomposition removed parent %s of %s", e.parent, child)
                continue
            w = float(coeffs.get(e.parent, float("nan")))
            e.weight = None if math.isnan(w) else w
            e.relation = "merge"
            e.evidence = list(e.evidence) + [
                f"merge coefficient {w:.3f} (residual {float(getattr(dec, 'residual', float('nan'))):.3f})"
            ]
            keep.append(e)
    keep.sort(key=lambda e: (e.child, -e.confidence, e.parent))
    return keep, stats, report


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #


def _edge_label(e: Edge) -> str:
    label = f"{e.relation} {float(e.confidence):.2f}"
    if e.weight is not None and not math.isnan(float(e.weight)):
        label = f"{e.relation} w={float(e.weight):.2f} c={float(e.confidence):.2f}"
    return label


def _pick_outgroups(
    a: ModelRef,
    b: ModelRef,
    nodes: Sequence[ModelRef],
    sketches: Mapping[ModelRef, Sketch],
    n: int,
) -> List[ModelRef]:
    """Nearest relatives of both endpoints, to root a scar-free pair.

    Claim: direction -- outgroup rooting measured 0% -> 100% on scar-free
    sft/lora/cpt edges, but it only fires when a third model is actually handed
    to :func:`stemma.direction.estimate_direction`, and nothing was handing one
    over: the benchmark supplied outgroups by hand while ``build_phylogeny``
    silently ran without them, so every scar-free edge abstained and the true
    parents never entered the DAG at all.

    Selection is done in **sketch space**, which costs no extra bytes: the
    sketches are already in memory from retrieval. Candidates are ranked by
    ``d(C, a) + d(C, b)`` and the closest ``n`` are returned.

    **MEASURED FAILURE -- this is OFF by default** (``auto_outgroup=0``).
    "Nearest relative of both endpoints" is the wrong selection rule, and
    systematically the worst possible one: the nearest relative of a merge child
    *is one of its parents*. Rooting assumes the outgroup is a **sibling**
    descending from a shared ancestor, never an ancestor or descendant of either
    endpoint, so feeding it a parent inverts the signal.

    Measured on ``sft`` vs ``merge-ties2`` (``sft`` is the true parent, so the
    wanted answer is ``a->b``), where the rule picked ``cpt`` -- the *other*
    parent of ``ties2``::

        without outgroup:  llr = +0.0093   (abstains, sign correct)
        with outgroup:     llr = -0.4411   (abstains, sign now WRONG)

    ``ties2`` contains ``0.4 * cpt``, so it sits closer to ``cpt`` than ``sft``
    does; the statistic reads that as "``ties2`` is nearer the root" and flips.
    End to end it also strengthened a *false* edge from confidence 0.89 to 1.00.

    A correct selector must exclude ancestors and descendants of both endpoints
    -- which is what :func:`cousin_veto` already identifies -- rather than
    ranking by proximity. That is the next step, and it is not implemented.
    """
    if n <= 0:
        return []
    from .sketch import sketch_distance  # lazy: keeps import graph acyclic

    sa, sb = sketches.get(a), sketches.get(b)
    if sa is None or sb is None:
        return []
    scored: List[Tuple[float, ModelRef]] = []
    for c in nodes:
        if c == a or c == b:
            continue
        sc = sketches.get(c)
        if sc is None:
            continue
        try:
            d = float(sketch_distance(sc, sa)) + float(sketch_distance(sc, sb))
        except Exception:  # pragma: no cover - defensive
            continue
        if math.isfinite(d):
            scored.append((d, c))
    scored.sort(key=lambda t: t[0])
    return [c for _d, c in scored[: int(n)]]


def _gate_rows(meta: Any, max_rows: int) -> np.ndarray:
    """Deterministic row selection shared by the cousin/reconstruction tests.

    Claim: infra -- both models in a comparison must be read on the *same* rows
    or the residual is meaningless, so the choice depends only on the row count
    and the budget.
    """
    from .remote_loader import select_rows  # lazy: no network at import time

    return select_rows(int(meta.shape[0]), int(max_rows))


#: ``cousin_score`` below this counts the pair as *cousins* rather than an edge.
#: Measured against ``smollm2-135m-root`` as the common ancestor: true edges
#: scored 0.8442 / 0.7183 / 0.7279 and cousin pairs 0.0029 / 0.0053 / -0.0004.
#: 0.15 sits an order of magnitude above the cousin cluster and 5x below the
#: weakest true edge, so the gap it splits is ~140x wide.
COUSIN_COS_THRESHOLD: float = 0.15

#: Reconstruction residual below which a candidate is accepted as the *exact*
#: input of a lossy child. Pruning masks entries but leaves the survivors
#: bit-identical, so the true parent scores exactly 0.0; measured, the nearest
#: cousin scored 4.26e-4, i.e. three orders of magnitude away.
RECONSTRUCTION_EPS: float = 1e-5


def reconstruction_residual(
    child: ModelRef,
    candidate: ModelRef,
    *,
    max_rows: int = 256,
    n_tensors: int = 8,
    **loader_kw: Any,
) -> float:
    """How exactly ``candidate`` reproduces ``child`` on the child's kept support.

    Claim: direction -- geometry fails for a *lossy* child: a 30%-pruned model
    measures ratio 1.00x against every candidate, because the pruning delta
    dwarfs the branch structure that distinguishes them (see
    :data:`PROXIMITY_FACTOR`). Reconstruction does not fail there, because
    pruning is a *mask*: the entries the child kept are bit-identical to its
    input's. Restricting the comparison to those entries turns an unidentifiable
    distance problem into an exact one.

    Returns ``||C - P||_F / ||P||_F`` over the positions where ``C`` is
    non-zero, or ``inf`` when the two share no comparable tensor. Measured on
    the benchmark, the true parent scores **0.000000** and the nearest cousin
    **0.000426**.
    """
    try:
        src_c = _open_for_gate(child, **loader_kw)
        src_p = _open_for_gate(candidate, **loader_kw)
    except Exception as exc:
        log.warning("reconstruction_residual(%s, %s) failed to open: %s", child, candidate, exc)
        return float("inf")
    try:
        idx_c, idx_p = src_c.index(), src_p.index()
        names = [
            n for n, m in sorted(idx_c.items())
            if len(m.shape) == 2
            and m.shape[0] * m.shape[1] >= MIN_GATE_TENSOR_PARAMS
            and n in idx_p
            and tuple(idx_p[n].shape) == tuple(m.shape)
        ][: int(n_tensors)]
        if not names:
            return float("inf")
        num = den = 0.0
        for name in names:
            rows = _gate_rows(idx_c[name], int(max_rows))
            C = src_c.get_tensor_rows(name, rows, dtype=np.float32)
            P = src_p.get_tensor_rows(name, rows, dtype=np.float32)
            if C.shape != P.shape:
                continue
            mask = C != 0
            if not mask.any():
                continue
            num += float(np.sum((C[mask] - P[mask]) ** 2))
            den += float(np.sum(P[mask] ** 2))
        if den <= 0.0:
            return float("inf")
        return float((num / den) ** 0.5)
    except Exception as exc:
        log.warning("reconstruction_residual(%s, %s) failed: %s", child, candidate, exc)
        return float("inf")
    finally:
        for s in (src_c, src_p):
            try:
                s.close()
            except Exception:  # pragma: no cover
                pass


def cousin_score(
    a: ModelRef,
    b: ModelRef,
    ancestor: ModelRef,
    *,
    max_rows: int = 256,
    n_tensors: int = 8,
    **loader_kw: Any,
) -> float:
    """``cos(a - ancestor, b - ancestor)``: ~0 means *cousins*, not an edge.

    Claim: low-false-positive -- this is the test README limitation #9 asked
    for, and it is non-circular: it needs no prior DAG, only a third model.

    If ``ancestor`` R is the common ancestor of both, then ``a = R + d_a`` and
    ``b = R + d_b`` with the two branches independent, so the cosine is ~0. If
    instead the three form a chain ``R -> a -> b``, then ``b - R`` still
    contains ``d_a``, so the cosine is strongly positive.

    Measured against ``smollm2-135m-root``::

        0.8442  sft -> merge-ties2        TRUE EDGE
        0.7279  sft -> cpt                TRUE EDGE
        0.7183  cpt -> merge-ties2        TRUE EDGE
        0.0053  sft-int4 / merge-ties2    cousins
        0.0029  merge-ties2 / prune-mag30 cousins
       -0.0004  int8 / prune-mag30        cousins

    Known blind spot: a *lossy* child whose modification dwarfs its parent's own
    branch also scores ~0 (``sft -> sft-int4`` measured 0.0065 despite being a
    true edge), because the quantisation error swamps the shared component.
    That case is resolved by :func:`reconstruction_residual`, not by this test,
    which is why :func:`cousin_veto` consults both.
    """
    try:
        srcs = {r: _open_for_gate(r, **loader_kw) for r in {str(a), str(b), str(ancestor)}}
    except Exception as exc:
        log.warning("cousin_score open failed: %s", exc)
        return float("nan")
    try:
        sa, sb, sr = srcs[str(a)], srcs[str(b)], srcs[str(ancestor)]
        ia, ib, ir = sa.index(), sb.index(), sr.index()
        names = [
            n for n, m in sorted(ia.items())
            if len(m.shape) == 2
            and m.shape[0] * m.shape[1] >= MIN_GATE_TENSOR_PARAMS
            and n in ib and n in ir
            and tuple(ib[n].shape) == tuple(m.shape) == tuple(ir[n].shape)
        ][: int(n_tensors)]
        if not names:
            return float("nan")
        num = d1 = d2 = 0.0
        for name in names:
            rows = _gate_rows(ia[name], int(max_rows))
            A = sa.get_tensor_rows(name, rows, dtype=np.float32)
            Bm = sb.get_tensor_rows(name, rows, dtype=np.float32)
            R = sr.get_tensor_rows(name, rows, dtype=np.float32)
            if not (A.shape == Bm.shape == R.shape):
                continue
            u = (A - R).ravel()
            v = (Bm - R).ravel()
            num += float(u @ v)
            d1 += float(u @ u)
            d2 += float(v @ v)
        denom = (d1 ** 0.5) * (d2 ** 0.5)
        return float(num / denom) if denom > 1e-30 else float("nan")
    except Exception as exc:
        log.warning("cousin_score failed: %s", exc)
        return float("nan")
    finally:
        for s in srcs.values():
            try:
                s.close()
            except Exception:  # pragma: no cover
                pass


def cousin_veto(
    a: ModelRef,
    b: ModelRef,
    universe: Sequence[ModelRef],
    *,
    cos_threshold: float = COUSIN_COS_THRESHOLD,
    reconstruction_eps: float = RECONSTRUCTION_EPS,
    max_rows: int = 256,
    **loader_kw: Any,
) -> Tuple[bool, Dict[str, Any]]:
    """True when some third model shows ``a`` and ``b`` to be cousins, not an edge.

    Claim: low-false-positive -- the direct answer to README limitation #9. A
    confident direction verdict about a pair that is not an edge is worse than
    no verdict, so the pair is removed *before* orientation.

    A pair is vetoed when a candidate common ancestor ``R`` exists with
    ``cousin_score(a, b, R) < cos_threshold``. The veto is **overridden** when
    either model reconstructs the other to within ``reconstruction_eps`` --
    that is the lossy-child case (``sft -> sft-int4``) where the cosine is
    uninformative but the mask/lattice evidence is exact.

    **Blind spot: merge DAGs.** The test assumes a tree, where sharing an
    ancestor precludes a direct edge. A merge breaks that: ``merge-ties2`` is
    ``0.6*sft + 0.4*cpt`` and ``cpt`` itself descends from ``sft``, so ``sft`` is
    a genuine common ancestor of ``cpt`` and ``ties2`` *and* ``cpt`` is a genuine
    parent of ``ties2``. Measured, that single case is the one false veto out of
    six real pairs (cos 0.0429 via ``sft``). Callers that also run merge
    decomposition should let a real mixing coefficient override the veto.

    Returns ``(vetoed, evidence)``.
    """
    ev: Dict[str, Any] = {"ancestor": None, "cousin_score": None, "reconstruction": None}
    others = [str(r) for r in universe if str(r) not in (str(a), str(b))]
    if not others:
        return False, ev

    # Lossy-child override first: it is exact, and cheaper than scanning R.
    r_ab = reconstruction_residual(b, a, max_rows=max_rows, **loader_kw)
    r_ba = reconstruction_residual(a, b, max_rows=max_rows, **loader_kw)
    best_recon = min(r_ab, r_ba)
    ev["reconstruction"] = float(best_recon)
    if best_recon <= float(reconstruction_eps):
        return False, ev

    # R must be a PLAUSIBLE common ancestor, not merely any third model.
    # Accepting any R was measured to veto true edges: some unrelated or
    # downstream model always produces a spuriously low cosine (cpt -> ties2 was
    # killed at 0.0429 by such an R). For genuine cousins with orthogonal
    # branches, |a-b|^2 = |d_a|^2 + |d_b|^2, so a common ancestor is strictly
    # CLOSER to both endpoints than they are to each other. Requiring that
    # removes the impostors.
    d_ab = relative_delta(a, b, max_rows=max_rows, **loader_kw)
    worst = 1.0
    for r in others:
        if math.isfinite(d_ab):
            d_ra = relative_delta(a, r, max_rows=max_rows, **loader_kw)
            d_rb = relative_delta(b, r, max_rows=max_rows, **loader_kw)
            if not (d_ra < d_ab and d_rb < d_ab):
                continue
        cs = cousin_score(a, b, r, max_rows=max_rows, **loader_kw)
        if not math.isfinite(cs):
            continue
        if cs < worst:
            worst, ev["ancestor"], ev["cousin_score"] = cs, r, float(cs)
        if cs < float(cos_threshold):
            return True, ev
    return False, ev


def transitive_reduction(p: Phylogeny) -> Phylogeny:
    """Drop each edge that a longer path already implies.

    Claim: direction -- proximity gating removes *cousins*, but it cannot remove
    an *ancestor that is not the direct parent*: on the benchmark the grandparent
    ``smollm2-135m-root`` measured 0.0007 away from ``merge-ties2``, closer than
    the true parent ``smollm2-cpt`` at 0.0012, so no distance threshold can
    separate them. Topology can. If ``P -> C`` is claimed and some path
    ``P -> ... -> C`` of length >= 2 also exists, the direct edge adds nothing
    the longer, more specific path does not already say, so it is removed.

    Returns a **new** :class:`~stemma.types.Phylogeny`; the input is not mutated.
    An edge is never removed when it is the child's only remaining incoming
    edge, so the reduction can never orphan a node that had a parent. Removals
    are recorded in ``meta["transitive_reduction"]`` so the DAG stays auditable.
    """
    edges = list(p.edges)
    if len(edges) < 2:
        return Phylogeny(
            nodes=list(p.nodes), edges=edges,
            root_candidates=list(p.root_candidates), meta=dict(p.meta),
        )

    # Longest-path-first: removing a shortcut must not depend on iteration order,
    # and considering the least-confident candidate shortcuts first keeps the
    # strongest evidence when two edges could each be called redundant.
    order = sorted(range(len(edges)), key=lambda i: float(edges[i].confidence or 0.0))
    alive = [True] * len(edges)
    removed: List[Dict[str, Any]] = []

    def _reachable(src: str, dst: str, skip: int) -> Optional[List[str]]:
        """Path src -> dst using live edges other than ``skip``, or None."""
        stack: List[Tuple[str, List[str]]] = [(src, [src])]
        seen = {src}
        while stack:
            node, path = stack.pop()
            for j, e in enumerate(edges):
                if not alive[j] or j == skip or e.parent != node:
                    continue
                if e.child == dst:
                    return path + [dst]
                if e.child not in seen:
                    seen.add(e.child)
                    stack.append((e.child, path + [e.child]))
        return None

    for i in order:
        if not alive[i]:
            continue
        e = edges[i]
        # Never orphan: keep this edge if it is the child's last incoming one.
        incoming = sum(
            1 for j, o in enumerate(edges) if alive[j] and o.child == e.child
        )
        if incoming <= 1:
            continue
        path = _reachable(e.parent, e.child, skip=i)
        if path is not None and len(path) >= 3:  # >= 2 hops
            alive[i] = False
            removed.append(
                {
                    "parent": e.parent,
                    "child": e.child,
                    "via": path,
                    "confidence": float(e.confidence or 0.0),
                }
            )

    kept = [e for i, e in enumerate(edges) if alive[i]]
    meta = dict(p.meta)
    meta["transitive_reduction"] = removed
    children = {e.child for e in kept}
    return Phylogeny(
        nodes=list(p.nodes),
        edges=kept,
        root_candidates=[n for n in p.nodes if n not in children],
        meta=meta,
    )


def to_mermaid(p: Phylogeny, conflicts: Iterable[Any] = ()) -> str:
    """Render the DAG as Mermaid ``graph TD`` text, flagging conflict nodes red.

    Claim: infra -- presentation only; it exists so a reviewer can see the
    oriented, confidence-weighted lineage that supports the direction claim.
    """
    ids = _node_id_map(p.nodes)
    flagged = _conflict_nodes(conflicts)
    lines: List[str] = ["graph TD"]
    lines.append(f"    %% stemma phylogeny: {len(p.nodes)} nodes, {len(p.edges)} edges")
    if not p.nodes:
        lines.append("    %% (empty phylogeny)")
        return "\n".join(lines) + "\n"
    for n in p.nodes:
        marker = " *" if n in flagged else ""
        lines.append(f'    {ids[n]}["{_escape_label(_display_name(n))}{marker}"]')
    for e in p.edges:
        if e.parent not in ids or e.child not in ids:
            continue
        lines.append(f'    {ids[e.parent]} -->|"{_escape_label(_edge_label(e))}"| {ids[e.child]}')
    flagged_ids = [ids[n] for n in p.nodes if n in flagged]
    if flagged_ids:
        lines.append(
            "    classDef conflict fill:#ffe0e0,stroke:#c00000,stroke-width:2px,color:#000000;"
        )
        lines.append(f"    class {','.join(flagged_ids)} conflict;")
    roots = [ids[n] for n in p.root_candidates if n in ids]
    if roots:
        lines.append(
            "    classDef root fill:#eef6ff,stroke:#3b6ea5,stroke-width:1px,color:#000000;"
        )
        non_conflict_roots = [r for r in roots if r not in set(flagged_ids)]
        if non_conflict_roots:
            lines.append(f"    class {','.join(non_conflict_roots)} root;")
    return "\n".join(lines) + "\n"


def to_graphviz_dot(p: Phylogeny, conflicts: Iterable[Any] = ()) -> str:
    """Render the same DAG as Graphviz DOT source text (no binary required).

    Claim: infra -- emitting text rather than shelling out to ``dot`` keeps the
    export path dependency-free; :func:`render_dot` is the optional extra step.
    """
    ids = _node_id_map(p.nodes)
    flagged = _conflict_nodes(conflicts)
    out: List[str] = [
        "digraph stemma {",
        "  rankdir=TB;",
        '  graph [fontname="Helvetica", labelloc="t", '
        'label="Stemma lineage (statistical evidence; human review required)"];',
        '  node [shape=box, style="rounded,filled", fillcolor="#f7f7f7", '
        'fontname="Helvetica", fontsize=10];',
        '  edge [fontname="Helvetica", fontsize=9];',
    ]
    for n in p.nodes:
        attrs = [f'label="{_escape_label(_display_name(n))}"']
        if n in flagged:
            attrs.append('fillcolor="#ffe0e0"')
            attrs.append('color="#c00000"')
            attrs.append("penwidth=2")
        elif n in set(p.root_candidates):
            attrs.append('fillcolor="#eef6ff"')
            attrs.append('color="#3b6ea5"')
        out.append(f'  {ids[n]} [{", ".join(attrs)}];')
    for e in p.edges:
        if e.parent not in ids or e.child not in ids:
            continue
        pen = 1.0 + 2.0 * float(np.clip(e.confidence, 0.0, 1.0))
        style = "dashed" if float(e.confidence) < 0.5 else "solid"
        out.append(
            f'  {ids[e.parent]} -> {ids[e.child]} '
            f'[label="{_escape_label(_edge_label(e))}", penwidth={pen:.2f}, style={style}];'
        )
    out.append("}")
    return "\n".join(out) + "\n"


def render_dot(dot: str, path, *, format: str = "png") -> Optional[str]:
    """Try to rasterise DOT text; return the output path or ``None`` on failure.

    Claim: infra -- rendering is a convenience, so a missing ``graphviz`` binary
    degrades to "no image" rather than breaking a lineage report.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.with_suffix("")
    try:
        import graphviz  # type: ignore

        src = graphviz.Source(dot)
        rendered = src.render(filename=str(stem), format=format, cleanup=True)
        return str(rendered)
    except Exception as exc:
        log.debug("python graphviz render failed (%s); trying the dot binary", exc)
    try:
        import shutil
        import subprocess

        exe = shutil.which("dot")
        if not exe:
            log.warning("graphviz 'dot' binary not found; skipping render")
            return None
        target = str(stem.with_suffix("." + format))
        subprocess.run(
            [exe, f"-T{format}", "-o", target],
            input=dot.encode("utf-8"),
            check=True,
            capture_output=True,
        )
        return target
    except Exception as exc:  # pragma: no cover - environment dependent
        log.warning("could not render DOT (%s)", exc)
        return None


# --------------------------------------------------------------------------- #
# tracing one model's lineage
# --------------------------------------------------------------------------- #


def ancestors_of(p: Phylogeny, node: ModelRef) -> List[ModelRef]:
    """Transitive closure of parents of ``node`` (cycle-safe).

    Claim: direction -- the ancestor set only exists because edges are oriented;
    it is what the rights propagation and the CLI's ``trace`` both consume.
    """
    seen: set = set()
    frontier = [node]
    while frontier:
        cur = frontier.pop()
        for e in p.parents_of(cur):
            if e.parent not in seen:
                seen.add(e.parent)
                frontier.append(e.parent)
    seen.discard(node)
    order = [n for n in p.nodes if n in seen]
    order.extend(n for n in seen if n not in set(order))
    return order


def trace(target: ModelRef, universe: Sequence[ModelRef | Sketch], **kw: Any) -> Phylogeny:
    """Build a phylogeny over ``{target} | universe`` and return target's lineage.

    Claim: direction -- this is the user-facing question ("where did this model
    come from?"); answering it requires the ancestor closure of an *oriented*
    graph, which symmetric similarity cannot provide.

    The result keeps the ancestor closure of ``target`` plus its direct
    descendants, with all original edge confidences preserved.
    """
    models: List[ModelRef | Sketch] = [target]
    tid = target.model_id if isinstance(target, Sketch) else str(target)
    for m in universe:
        mid = m.model_id if isinstance(m, Sketch) else str(m)
        if mid != tid:
            models.append(m)
    full = build_phylogeny(models, **kw)

    keep: set = {tid}
    keep.update(ancestors_of(full, tid))
    direct_children = [e.child for e in full.children_of(tid)]
    keep.update(direct_children)

    nodes = [n for n in full.nodes if n in keep]
    edges = [e for e in full.edges if e.parent in keep and e.child in keep]
    with_parents = {e.child for e in edges}
    roots = [n for n in nodes if n not in with_parents]
    meta = dict(full.meta)
    meta.update(
        {
            "trace_target": tid,
            "trace_universe_size": len(models) - 1,
            "n_ancestors": len([n for n in nodes if n != tid and n not in set(direct_children)]),
            "n_direct_descendants": len(direct_children),
            "full_graph_nodes": len(full.nodes),
            "full_graph_edges": len(full.edges),
        }
    )
    return Phylogeny(nodes=nodes, edges=edges, root_candidates=roots, meta=meta)


def phylogeny_to_json(p: Phylogeny) -> Dict[str, Any]:
    """Serialise a phylogeny to plain JSON-safe structures.

    Claim: infra -- the CLI and the benchmark both need a stable on-disk form.
    """
    return {
        "nodes": list(p.nodes),
        "edges": [asdict(e) if is_dataclass(e) else dict(e) for e in p.edges],
        "root_candidates": list(p.root_candidates),
        "meta": p.meta,
    }
