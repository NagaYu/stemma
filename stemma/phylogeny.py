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

    # ---- (3) orientation ------------------------------------------------- #
    edges: List[Edge] = []
    verdicts: Dict[Tuple[ModelRef, ModelRef], Any] = {}
    n_unknown = 0
    for a, b, score in related:
        try:
            v = _call_tolerant(
                direction_mod.estimate_direction,
                a,
                b,
                weights=direction_model,
                abstain=direction_abstain,
                sa=all_sketches.get(a),
                sb=all_sketches.get(b),
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
    return Phylogeny(nodes=nodes, edges=edges, root_candidates=roots, meta=meta)


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
