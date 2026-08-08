"""Recover *which* models were merged into a child, and in *what proportions*.

The observation this module exploits is that a merged model ``M`` built from
parents ``P_1..P_k`` on top of a shared base ``B`` satisfies

    tau_M := M - B  ~=  sum_i w_i (P_i - B) =: sum_i w_i tau_i

exactly for linear / task-arithmetic / SLERP-in-the-small merges, and
approximately for TIES and DARE (which prune and rescale the task vectors but
keep their direction).  Estimating ``w`` is therefore a non-negative, sparse
least-squares problem over a handful of columns -- solvable from a few hundred
thousand *shared coordinates* rather than from whole checkpoints.

Claim: merge-recovery
Symmetric pairwise fingerprints (cosine, CKA, HuRef) return one number per pair
and structurally cannot say "0.7 of A plus 0.3 of B"; this module can, which is
the second headline result of the paper.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .types import MergeDecomposition, ModelRef, TransferStats
from .utils import get_logger, role_of, stable_hash

__all__ = [
    "pick_common_tensors",
    "sample_coordinates",
    "task_vectors",
    "nnls_l1",
    "decompose_merge",
    "mixing_mae",
    "parent_set_prf",
]

log = get_logger(__name__)

#: Tensors smaller than this are not worth a Range request for decomposition:
#: their coordinates are dominated by norm/embedding idiosyncrasies.
DEFAULT_MIN_PARAMS: int = 1 << 16  # 65_536

#: How many distinct tensors we spread the coordinate budget over. More tensors
#: means more requests but a less layer-specific (hence more robust) estimate.
DEFAULT_MAX_TENSORS: int = 8


# --------------------------------------------------------------------------- #
# Source handling (remote_loader is imported lazily: no network at import time)
# --------------------------------------------------------------------------- #


def _open_source(ref: Any, **loader_kw: Any) -> Tuple[Any, bool]:
    """Return ``(source, we_opened_it)`` for a model reference or open source.

    Claim: infra
    Accepting an already-open :class:`SafeTensorsSource` lets callers (the
    phylogeny builder) amortise one header read across many decompositions,
    which is what keeps the low-transfer number honest.
    """
    if hasattr(ref, "get_tensor") and hasattr(ref, "index"):
        return ref, False
    from .remote_loader import open_model  # lazy: no network at import time

    return open_model(ref, **loader_kw), True


def _ref_name(ref: Any) -> str:
    """Human-readable id for a model reference or an open source object.

    Claim: infra
    Keeps ``MergeDecomposition.candidates`` printable regardless of whether the
    caller passed strings or open sources.
    """
    if isinstance(ref, str):
        return ref
    for attr in ("model_id", "ref", "repo"):
        v = getattr(ref, attr, None)
        if isinstance(v, str):
            return v
    return str(ref)


def _snapshot(sources: Sequence[Any]) -> Tuple[int, int, float, int]:
    st = [getattr(s, "stats", None) for s in sources]
    return (
        sum(int(getattr(x, "bytes_read", 0) or 0) for x in st),
        sum(int(getattr(x, "requests", 0) or 0) for x in st),
        sum(float(getattr(x, "seconds", 0.0) or 0.0) for x in st),
        sum(int(getattr(x, "cache_hits", 0) or 0) for x in st),
    )


def _delta_stats(sources: Sequence[Any], before: Tuple[int, int, float, int]) -> TransferStats:
    after = _snapshot(sources)
    full = 0
    for s in sources:
        try:
            full += int(s.total_size())
        except Exception:  # pragma: no cover - loader may not know the size
            pass
    return TransferStats(
        bytes_read=max(0, after[0] - before[0]),
        requests=max(0, after[1] - before[1]),
        seconds=max(0.0, after[2] - before[2]),
        full_size_bytes=full,
        cache_hits=max(0, after[3] - before[3]),
    )


# --------------------------------------------------------------------------- #
# Choosing a common coordinate system
# --------------------------------------------------------------------------- #


#: A candidate task vector shorter than this fraction of the CHILD's own task
#: vector is numerically zero and is removed before solving.
#:
#: The reference must be the child, never the largest column. Scaling by the
#: largest column was measured to be catastrophic: the biggest task vectors
#: belong to heavily modified models (pruned, ~0.1) while a true fine-tune
#: parent sits at ~0.0008, a ratio of 0.008 -- so a 1%-of-max rule deleted the
#: true parents and merge F1 collapsed from 0.57 to 0.00 with the residual
#: exploding to 127. Magnitudes here span orders of magnitude, and a
#: relative-to-max threshold is exactly the wrong tool for that, the same trap
#: that sank the proximity gate.
#:
#: The columns actually being targeted are *exactly* zero: the inferred base
#: (``B - B``) and models that differ from the base only in tensors the sampler
#: never reads. 1e-6 catches those and nothing else.
ZERO_TASK_VECTOR_FRACTION: float = 1e-6


def pick_common_tensors(
    sources: Sequence[Any],
    min_params: int = DEFAULT_MIN_PARAMS,
    *,
    limit: Optional[int] = DEFAULT_MAX_TENSORS,
    tensor_filter: Optional[Any] = None,
) -> List[str]:
    """Names of large 2D tensors that exist with the *same shape* in every source.

    Claim: merge-recovery
    The decomposition is only meaningful if every model contributes the very
    same coordinates, so the shared-name/shared-shape intersection computed here
    is the precondition for solving ``tau_M ~= T w`` at all.

    Parameters
    ----------
    sources:
        Open safetensors sources (anything exposing ``index()``).
    min_params:
        Skip tensors with fewer elements than this (norms, biases, tiny heads).
    limit:
        Keep at most this many tensors, chosen round-robin across roles so the
        coordinate budget is spread over attention, MLP and head weights rather
        than being eaten by one giant embedding matrix.
    tensor_filter:
        Optional ``Callable[[str], bool]`` or an explicit sequence of names.
    """
    if not sources:
        return []
    indexes = []
    for s in sources:
        try:
            indexes.append(s.index())
        except Exception as exc:  # pragma: no cover - depends on loader
            log.warning("could not read tensor index for %s: %s", _ref_name(s), exc)
            return []

    allow: Optional[Callable[[str], bool]]
    if tensor_filter is None:
        allow = None
    elif callable(tensor_filter):
        allow = tensor_filter
    else:
        wanted = set(str(x) for x in tensor_filter)
        allow = lambda n: n in wanted  # noqa: E731

    first = indexes[0]
    common: List[Tuple[str, Tuple[int, ...]]] = []
    for name, meta in first.items():
        shape = tuple(int(d) for d in meta.shape)
        if len(shape) != 2 or min(shape) < 2:
            continue
        if shape[0] * shape[1] < int(min_params):
            continue
        if allow is not None and not allow(name):
            continue
        ok = True
        for idx in indexes[1:]:
            other = idx.get(name)
            if other is None or tuple(int(d) for d in other.shape) != shape:
                ok = False
                break
        if ok:
            common.append((name, shape))

    if not common:
        return []

    # Group by role, biggest first inside each role, then interleave roles.
    buckets: Dict[str, List[Tuple[str, Tuple[int, ...]]]] = {}
    for name, shape in common:
        buckets.setdefault(role_of(name) or "other", []).append((name, shape))
    for entries in buckets.values():
        entries.sort(key=lambda t: (-(t[1][0] * t[1][1]), t[0]))

    order = sorted(buckets.keys())
    picked: List[str] = []
    depth = 0
    while True:
        added = False
        for role in order:
            entries = buckets[role]
            if depth < len(entries):
                picked.append(entries[depth][0])
                added = True
                if limit is not None and len(picked) >= int(limit):
                    return picked
        if not added:
            break
        depth += 1
    return picked


def sample_coordinates(shape: Sequence[int], n: int, seed: int = 0) -> np.ndarray:
    """Deterministic set of ``n`` flat coordinate indices inside ``shape``.

    Claim: merge-recovery
    Every model must be probed at *identical* positions or the differences we
    regress on are noise; making the sample a pure function of
    ``(shape, n, seed)`` guarantees that without any cross-model communication.

    Returns a sorted ``int64`` array (sorted order also makes the subsequent
    gather cache-friendly). If ``n`` >= the number of elements, all of them are
    returned.
    """
    total = 1
    for d in shape:
        total *= int(d)
    if total <= 0:
        return np.zeros(0, dtype=np.int64)
    n = int(max(0, min(int(n), total)))
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if n == total:
        return np.arange(total, dtype=np.int64)

    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    if total <= 4_000_000:
        idx = rng.permutation(total)[:n].astype(np.int64)
    else:
        # Avoid materialising a full permutation of a very large tensor.
        picked = np.zeros(0, dtype=np.int64)
        while picked.size < n:
            draw = rng.integers(0, total, size=int((n - picked.size) * 1.4) + 32, dtype=np.int64)
            picked = np.unique(np.concatenate([picked, draw]))
        idx = picked[rng.permutation(picked.size)[:n]]
    return np.sort(idx)


def _tensor_seed(seed: int, name: str) -> int:
    """Per-tensor seed that depends on the tensor name, not on iteration order.

    Claim: infra
    Order-independence means adding a candidate model cannot silently change
    which coordinates were sampled for the others.
    """
    h = int(stable_hash(name)[:8], 16)
    return int((int(seed) * 1_000_003 + h) & 0xFFFFFFFF)


# --------------------------------------------------------------------------- #
# Reading the shared coordinate matrix
# --------------------------------------------------------------------------- #


def _collect_matrix(
    sources: Sequence[Any],
    *,
    coords: int,
    seed: int,
    min_params: int,
    max_tensors: Optional[int],
    tensor_filter: Optional[Any],
) -> Tuple[np.ndarray, List[str]]:
    """Gather the same ``coords`` weight coordinates from every source.

    Claim: low-transfer
    Only ``ceil(budget / n_cols)`` rows of each chosen tensor are requested, so
    a decomposition costs a few megabytes of Range reads instead of the tens of
    gigabytes a full download of every candidate would need.
    """
    names = pick_common_tensors(
        sources, min_params, limit=max_tensors, tensor_filter=tensor_filter
    )
    if not names:
        raise ValueError(
            "no 2D tensor with identical shape and >= %d parameters is present in all "
            "%d models; cannot build a shared coordinate system" % (min_params, len(sources))
        )

    budget = max(1, int(math.ceil(int(coords) / len(names))))
    parts: List[List[np.ndarray]] = [[] for _ in sources]
    used: List[str] = []

    for name in names:
        meta = sources[0].index()[name]
        n_cols = int(meta.shape[1])
        n_rows_full = int(meta.shape[0])
        rows = int(min(n_rows_full, max(1, math.ceil(budget / max(1, n_cols)))))
        try:
            blocks = [s.get_tensor(name, dtype=np.float32, max_rows=rows) for s in sources]
        except Exception as exc:
            log.warning("skipping tensor %s: %s", name, exc)
            continue
        blocks = [np.asarray(b, dtype=np.float32) for b in blocks]
        shapes = {b.shape for b in blocks}
        if len(shapes) != 1:
            # Loud, because a silent mismatch would make the whole solve garbage.
            log.warning("shape mismatch for %s across models (%s); skipping", name, shapes)
            continue
        shape = blocks[0].shape
        if blocks[0].size == 0:
            continue
        assert all(b.shape == shape for b in blocks), "coordinate systems diverged"
        idx = sample_coordinates(shape, min(budget, blocks[0].size), _tensor_seed(seed, name))
        if idx.size == 0:
            continue
        for i, b in enumerate(blocks):
            parts[i].append(np.ascontiguousarray(b).reshape(-1)[idx])
        used.append(name)

    if not used:
        raise ValueError("every shared tensor failed to read; cannot decompose")

    M = np.stack([np.concatenate(p).astype(np.float64, copy=False) for p in parts], axis=0)
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    log.debug("collected %d coordinates from %d tensors x %d models", M.shape[1], len(used), M.shape[0])
    return M, used


def task_vectors(
    base: Optional[ModelRef],
    candidates: Sequence[ModelRef],
    child: ModelRef,
    *,
    coords: int = 200_000,
    seed: int = 0,
    tensor_filter: Optional[Any] = None,
    **loader_kw: Any,
) -> Tuple[np.ndarray, np.ndarray, List[str], TransferStats]:
    """Build the task-vector design matrix ``T`` and target ``tau_child``.

    Claim: merge-recovery
    Subtracting a common base turns "which models look alike" into a linear
    algebra problem whose *coefficients* are the mixing ratios we want to
    recover.

    Returns ``(T, t_child, used_tensor_names, stats)`` where ``T`` has shape
    ``(n_candidates, n_coords)`` and ``t_child`` has shape ``(n_coords,)``. When
    ``base`` is ``None`` the raw sampled weights are returned instead of
    differences (the fallback path documented in CONTRACT.md).
    """
    min_params = int(loader_kw.pop("min_params", DEFAULT_MIN_PARAMS))
    max_tensors = loader_kw.pop("max_tensors", DEFAULT_MAX_TENSORS)

    refs: List[Any] = [child, *candidates]
    if base is not None:
        refs.append(base)

    sources: List[Any] = []
    owned: List[bool] = []
    try:
        for r in refs:
            s, mine = _open_source(r, **loader_kw)
            sources.append(s)
            owned.append(mine)
        before = _snapshot(sources)
        M, used = _collect_matrix(
            sources,
            coords=coords,
            seed=seed,
            min_params=min_params,
            max_tensors=max_tensors,
            tensor_filter=tensor_filter,
        )
        stats = _delta_stats(sources, before)
    finally:
        for s, mine in zip(sources, owned):
            if mine:
                try:
                    s.close()
                except Exception:  # pragma: no cover
                    pass

    n_c = len(candidates)
    child_vec = M[0]
    cand = M[1 : 1 + n_c]
    if base is None:
        return np.ascontiguousarray(cand), np.ascontiguousarray(child_vec), used, stats
    base_vec = M[1 + n_c]
    T = cand - base_vec[None, :]
    y = child_vec - base_vec
    return np.ascontiguousarray(T), np.ascontiguousarray(y), used, stats


# --------------------------------------------------------------------------- #
# The solver
# --------------------------------------------------------------------------- #


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection of ``v`` onto ``{w >= 0, sum w = 1}``.

    Claim: merge-recovery
    Used by the scipy-free fallback so mixing ratios stay interpretable as
    proportions even on a minimal install.
    """
    v = np.asarray(v, dtype=np.float64).ravel()
    n = v.size
    if n == 0:
        return v
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1, dtype=np.float64)
    cond = u - css / ind > 0
    if not np.any(cond):
        return np.full(n, 1.0 / n)
    rho = int(np.nonzero(cond)[0][-1])
    theta = css[rho] / float(rho + 1)
    return np.maximum(v - theta, 0.0)


def _projected_gradient(
    G: np.ndarray, c: np.ndarray, w0: np.ndarray, *, sum_to_one: bool, nonneg: bool, iters: int = 800
) -> np.ndarray:
    """Pure-numpy fallback solver for ``min w'Gw - 2c'w`` under simple constraints.

    Claim: merge-recovery
    Guarantees the decomposition still runs (and still returns sane ratios)
    when scipy is unavailable, so the merge-recovery result is not gated on an
    optional dependency.
    """
    w = np.asarray(w0, dtype=np.float64).ravel().copy()
    if w.size == 0:
        return w
    lam = float(np.max(np.abs(np.linalg.eigvalsh(G)))) if G.size else 1.0
    step = 1.0 / (2.0 * lam + 1e-12)
    best, best_obj = w.copy(), float("inf")
    for _ in range(int(iters)):
        grad = 2.0 * (G @ w - c)
        w = w - step * grad
        if sum_to_one and nonneg:
            w = _project_simplex(w)
        elif nonneg:
            w = np.maximum(w, 0.0)
        elif sum_to_one:
            w = w + (1.0 - w.sum()) / w.size
        obj = float(w @ G @ w - 2.0 * c @ w)
        if obj < best_obj:
            best_obj, best = obj, w.copy()
    return best


def nnls_l1(
    T: np.ndarray,
    y: np.ndarray,
    *,
    l1: float = 0.0,
    sum_to_one: bool = False,
    nonneg: bool = True,
) -> np.ndarray:
    """Solve ``min_w ||y - T'w||^2 + l1 ||w||_1`` with ``w >= 0`` (optionally ``sum w = 1``).

    Claim: merge-recovery
    Sparsity plus non-negativity is what turns a dense least-squares fit into a
    *parent set* -- decoy candidates are driven to exactly zero instead of
    picking up small explanatory crumbs.

    The L1 trick
    ------------
    On the non-negative orthant ``||w||_1 == sum(w) == 1'w``, which is *linear*
    in ``w``. So the penalty can be folded into the least-squares system by
    appending **one** extra row ``sqrt(l1) * ||y|| * 1'`` with target ``0``:
    that row contributes ``l1 * ||y||^2 * (1'w)^2``, a strictly increasing
    function of ``||w||_1``. (Appending ``sqrt(l1) * I`` with zero targets --
    the usual reflex -- would give a *ridge* penalty ``l1 ||w||_2^2``, which
    shrinks but never selects, so it is deliberately not used here.) The
    ``||y||`` factor makes ``l1`` scale-free: it is measured relative to the
    energy of the target task vector.

    Parameters
    ----------
    T : (n_candidates, n_coords)
    y : (n_coords,)
    """
    T = np.atleast_2d(np.asarray(T, dtype=np.float64))
    y = np.asarray(y, dtype=np.float64).ravel()
    k, m = T.shape
    if k == 0:
        return np.zeros(0, dtype=np.float64)
    if m != y.size:
        raise ValueError(f"T has {m} coordinates but y has {y.size}")

    A = np.ascontiguousarray(T.T)  # (n_coords, n_candidates)
    y_norm = float(np.linalg.norm(y))
    scale = y_norm if y_norm > 0 else 1.0

    if float(l1) > 0.0 and nonneg:
        aug_row = math.sqrt(float(l1)) * scale * np.ones((1, k), dtype=np.float64)
        A_aug = np.vstack([A, aug_row])
        y_aug = np.concatenate([y, np.zeros(1, dtype=np.float64)])
    else:
        A_aug, y_aug = A, y

    w = None
    if nonneg:
        try:
            from scipy.optimize import nnls  # lazy import, optional dependency

            w = np.asarray(nnls(A_aug, y_aug, maxiter=200 * max(k, 1))[0], dtype=np.float64)
        except TypeError:  # older scipy without maxiter
            try:
                from scipy.optimize import nnls

                w = np.asarray(nnls(A_aug, y_aug)[0], dtype=np.float64)
            except Exception as exc:  # pragma: no cover
                log.debug("scipy nnls unavailable/failed (%s); using projected gradient", exc)
        except Exception as exc:
            log.debug("scipy nnls unavailable/failed (%s); using projected gradient", exc)

    G = A.T @ A
    c = A.T @ y
    if w is None:
        if nonneg:
            w0 = np.full(k, 1.0 / k)
            w = _projected_gradient(
                G + float(l1) * scale**2 * np.ones((k, k)), c, w0, sum_to_one=False, nonneg=True
            )
        else:
            w = np.linalg.lstsq(A_aug, y_aug, rcond=None)[0]
    w = np.nan_to_num(np.asarray(w, dtype=np.float64).ravel(), nan=0.0, posinf=0.0, neginf=0.0)

    if not sum_to_one:
        return w

    # ---- refine under the equality constraint sum(w) == 1 -------------------
    s = float(w.sum())
    x0 = w / s if s > 1e-12 else np.full(k, 1.0 / k)
    if nonneg:
        x0 = np.clip(x0, 0.0, 1.0)
        s0 = float(x0.sum())
        x0 = x0 / s0 if s0 > 1e-12 else np.full(k, 1.0 / k)

    def obj(v: np.ndarray) -> float:
        return float(v @ G @ v - 2.0 * c @ v)

    def jac(v: np.ndarray) -> np.ndarray:
        return 2.0 * (G @ v - c)

    best = x0
    try:
        from scipy.optimize import minimize  # lazy import, optional dependency

        res = minimize(
            obj,
            x0,
            jac=jac,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * k if nonneg else [(None, None)] * k,
            constraints=({"type": "eq", "fun": lambda v: float(v.sum() - 1.0),
                          "jac": lambda v: np.ones_like(v)},),
            options={"maxiter": 300, "ftol": 1e-12},
        )
        cand = np.asarray(res.x, dtype=np.float64).ravel()
        if np.all(np.isfinite(cand)) and obj(cand) <= obj(best) + 1e-12:
            best = cand
    except Exception as exc:
        log.debug("SLSQP unavailable/failed (%s); using projected gradient", exc)
        cand = _projected_gradient(G, c, x0, sum_to_one=True, nonneg=nonneg)
        if obj(cand) <= obj(best):
            best = cand

    if nonneg:
        best = np.maximum(best, 0.0)
    s = float(best.sum())
    if s > 1e-12:
        best = best / s
    return best


# --------------------------------------------------------------------------- #
# The public decomposition
# --------------------------------------------------------------------------- #


def _pairwise_scaled_distance(M: np.ndarray) -> np.ndarray:
    """Scale-free Euclidean distance matrix between the rows of ``M``."""
    n = M.shape[0]
    norms = np.linalg.norm(M, axis=1)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            denom = 0.5 * (norms[i] + norms[j]) + 1e-12
            d = float(np.linalg.norm(M[i] - M[j]) / denom)
            D[i, j] = D[j, i] = d
    return D


def decompose_merge(
    child: ModelRef,
    candidates: Sequence[ModelRef],
    *,
    base: Optional[ModelRef] = None,
    l1: float = 1e-3,
    sum_to_one: bool = True,
    support_threshold: float = 0.05,
    coords: int = 200_000,
    seed: int = 0,
    **loader_kw: Any,
) -> MergeDecomposition:
    """Explain ``child`` as a non-negative mixture of ``candidates`` over a base.

    Claim: merge-recovery
    This is the function the headline "parent-set F1 / mixing MAE" row of the
    benchmark table is computed from: it names the actual parents and reports
    their weights, which no symmetric similarity statistic can do.

    Procedure
    ---------
    1. Read the *same* seeded coordinate subsample from child, candidates and
       base (one Range read per tensor per model).
    2. If ``base is None``, infer it as the candidate with the smallest mean
       scaled distance to the other candidates; if that is impossible or
       degenerate, fall back to decomposing the raw weights with
       ``sum_to_one=True`` and report ``base=None``.
    3. Solve the sparse NNLS problem, threshold the support, then **re-solve
       restricted to the surviving support** and renormalise, which removes the
       shrinkage bias the discarded columns had introduced.
    """
    min_params = int(loader_kw.pop("min_params", DEFAULT_MIN_PARAMS))
    max_tensors = loader_kw.pop("max_tensors", DEFAULT_MAX_TENSORS)
    tensor_filter = loader_kw.pop("tensor_filter", None)

    cand_names = [_ref_name(c) for c in candidates]
    child_name = _ref_name(child)
    n_c = len(candidates)
    if n_c == 0:
        return MergeDecomposition(
            child=child_name,
            base=None,
            candidates=[],
            coefficients=np.zeros(0, dtype=np.float64),
            selected=[],
            residual=1.0,
            r2=0.0,
            method="nnls+l1",
            stats=TransferStats(),
        )

    refs: List[Any] = [child, *candidates]
    explicit_base = base is not None
    if explicit_base:
        refs.append(base)

    sources: List[Any] = []
    owned: List[bool] = []
    try:
        for r in refs:
            s, mine = _open_source(r, **loader_kw)
            sources.append(s)
            owned.append(mine)
        # Reject architecture-incompatible candidates instead of aborting.
        # _collect_matrix intersects tensors across *every* source, so a single
        # decoy of a different architecture raised "no 2D tensor ... present in
        # all N models" and killed the whole decomposition (it cost us 2 of 4
        # ground-truth merges). A candidate that shares no coordinate system
        # with the child cannot be a linear parent, so the correct result is to
        # keep it with a zero coefficient -- exactly what a decoy deserves.
        incompatible: List[str] = []
        keep_idx: List[int] = list(range(n_c))
        if n_c > 1:
            probe_keep: List[int] = []
            for i in range(n_c):
                probe = [sources[0], sources[1 + i]]
                if explicit_base:
                    probe.append(sources[-1])
                if pick_common_tensors(
                    probe, min_params, limit=max_tensors, tensor_filter=tensor_filter
                ):
                    probe_keep.append(i)
                else:
                    incompatible.append(cand_names[i])
            if incompatible and probe_keep:
                log.info(
                    "%s: %d candidate(s) share no coordinate system and cannot be "
                    "linear parents (coefficient forced to 0): %s",
                    child_name, len(incompatible), ", ".join(incompatible),
                )
                keep_idx = probe_keep

        solve_sources = [sources[0]] + [sources[1 + i] for i in keep_idx]
        if explicit_base:
            solve_sources.append(sources[-1])

        before = _snapshot(solve_sources)
        M, used = _collect_matrix(
            solve_sources,
            coords=coords,
            seed=seed,
            min_params=min_params,
            max_tensors=max_tensors,
            tensor_filter=tensor_filter,
        )
        stats = _delta_stats(solve_sources, before)
    finally:
        for s, mine in zip(sources, owned):
            if mine:
                try:
                    s.close()
                except Exception:  # pragma: no cover
                    pass

    # M is aligned with solve_sources, i.e. only the compatible candidates.
    # Everything below solves in that reduced space; w is expanded back to the
    # full candidate list (zeros for the rejected ones) before being returned.
    kept_names = [cand_names[i] for i in keep_idx]
    n_k = len(keep_idx)

    child_vec = M[0]
    cand_mat = M[1 : 1 + n_k]

    base_name: Optional[str] = None
    base_vec: Optional[np.ndarray] = None
    if explicit_base:
        base_name = _ref_name(base)
        base_vec = M[1 + n_k]
    elif n_k >= 2:
        D = _pairwise_scaled_distance(cand_mat)
        mean_d = (D.sum(axis=1)) / max(1, n_k - 1)
        b_idx = int(np.argmin(mean_d))
        if float(np.max(D)) > 1e-9:
            base_name = kept_names[b_idx]
            base_vec = cand_mat[b_idx]
            log.debug("inferred base %s (mean scaled distance %.4f)", base_name, mean_d[b_idx])

    used_task_vectors = base_vec is not None
    if used_task_vectors:
        T = cand_mat - base_vec[None, :]
        y = child_vec - base_vec
        # A base that happens to *be* the child, or candidates identical to the
        # base, leave nothing to regress on -- fall back to raw weights.
        col_norms = np.linalg.norm(T, axis=1)
        if float(np.linalg.norm(y)) <= 1e-12 or float(np.max(col_norms)) <= 1e-12:
            log.debug("degenerate task vectors; falling back to raw-weight decomposition")
            used_task_vectors = False

    # Drop candidates whose task vector is ~zero relative to the base. Such a
    # column cannot explain any of the child's deviation, but under the
    # sum-to-one constraint it happily ABSORBS weight, which both invents false
    # parents and drags the true ones down. Measured on smollm2-merge-ties2
    # (truth sft 0.6 / cpt 0.4) with all 19 other models as candidates: the
    # inferred base itself (task vector identically zero) plus vocab-ext and
    # vocab-ext-trained -- which differ from the base only in embedding rows the
    # sampler never reads -- each took exactly 0.1197, while sft fell to 0.5410
    # and cpt to 0.1000. They are degenerate columns, not parents.
    degenerate: List[str] = []
    if used_task_vectors:
        col_norms = np.linalg.norm(T, axis=1)
        y_norm = float(np.linalg.norm(y))
        if y_norm > 0.0 and col_norms.size:
            keep_cols = col_norms >= ZERO_TASK_VECTOR_FRACTION * y_norm
            if not keep_cols.all():
                degenerate = [kept_names[i] for i in np.nonzero(~keep_cols)[0].tolist()]
                log.info(
                    "%s: dropping %d degenerate candidate(s) whose task vector is ~0 "
                    "relative to the base (they cannot explain the child, but the "
                    "simplex constraint would hand them weight): %s",
                    child_name, len(degenerate), ", ".join(degenerate),
                )
                T = T[keep_cols]
                kept_names = [n for n, k in zip(kept_names, keep_cols.tolist()) if k]
                keep_idx = [i for i, k in zip(keep_idx, keep_cols.tolist()) if k]
                n_k = len(keep_idx)
                if n_k == 0:
                    used_task_vectors = False

    if used_task_vectors:
        method = "nnls+l1"
    else:
        T = cand_mat
        y = child_vec
        base_name = None
        sum_to_one = True  # raw weights only mix sensibly as a convex combination
        method = "nnls+l1(raw)"
    if sum_to_one:
        method += "+simplex"

    w = nnls_l1(T, y, l1=float(l1), sum_to_one=bool(sum_to_one), nonneg=True)

    # ---- support selection --------------------------------------------------
    if sum_to_one:
        thr = float(support_threshold)  # coefficients already sum to 1
    else:
        peak = float(np.max(w)) if w.size else 0.0
        thr = float(support_threshold) * peak
    support = np.nonzero(w >= thr)[0] if w.size else np.zeros(0, dtype=int)
    if support.size == 0 and w.size:
        support = np.array([int(np.argmax(w))])

    # ---- re-solve restricted to the support --------------------------------
    w_final = np.zeros(n_k, dtype=np.float64)
    if support.size:
        w_sub = nnls_l1(
            T[support], y, l1=float(l1), sum_to_one=bool(sum_to_one), nonneg=True
        )
        if sum_to_one:
            s = float(w_sub.sum())
            w_sub = w_sub / s if s > 1e-12 else np.full(support.size, 1.0 / support.size)
        w_final[support] = w_sub

    selected = [kept_names[i] for i in support.tolist() if w_final[i] > 0.0]

    y_norm = float(np.linalg.norm(y))
    resid_vec = y - (w_final @ T)
    residual = float(np.linalg.norm(resid_vec) / y_norm) if y_norm > 1e-12 else 0.0
    r2 = float(1.0 - residual**2)
    if not np.isfinite(residual):
        residual, r2 = 1.0, 0.0

    log.debug(
        "decomposed %s over %d candidates: residual=%.4f r2=%.4f coeffs=%s",
        child_name, n_c, residual, r2, np.round(w_final, 4).tolist(),
    )

    # Expand back to the full candidate list. Candidates rejected as
    # architecture-incompatible keep an explicit 0.0: they were considered and
    # ruled out, which is a different statement from "not evaluated", and the
    # benchmark scores precision over the full candidate set.
    w_full = np.zeros(n_c, dtype=np.float64)
    for slot, i in enumerate(keep_idx):
        w_full[i] = w_final[slot]

    return MergeDecomposition(
        child=child_name,
        base=base_name,
        candidates=cand_names,
        coefficients=w_full,
        selected=selected,
        residual=residual,
        r2=r2,
        method=method,
        stats=stats,
    )


# --------------------------------------------------------------------------- #
# Scoring helpers used by the benchmark
# --------------------------------------------------------------------------- #


def mixing_mae(pred: Dict[str, float], truth: Dict[str, float]) -> float:
    """Mean absolute error between predicted and true mixing ratios.

    Claim: merge-recovery
    Scored over the **union** of parent names with missing entries treated as
    ``0`` so that inventing a parent is penalised exactly as hard as missing
    one -- otherwise a decomposer could game the metric by predicting nothing.
    """
    keys = set(pred or {}) | set(truth or {})
    if not keys:
        return 0.0
    total = 0.0
    for k in keys:
        total += abs(float((pred or {}).get(k, 0.0)) - float((truth or {}).get(k, 0.0)))
    return float(total / len(keys))


def parent_set_prf(pred: Sequence[str], truth: Sequence[str]) -> Tuple[float, float, float]:
    """Precision, recall and F1 of the recovered parent *set*.

    Claim: merge-recovery
    The set metric is reported alongside the ratio MAE because a decomposition
    that finds the right parents with slightly wrong weights is far more useful
    than one with tidy weights on the wrong models.
    """
    p_set = {str(x) for x in (pred or [])}
    t_set = {str(x) for x in (truth or [])}
    if not p_set and not t_set:
        return 1.0, 1.0, 1.0
    tp = len(p_set & t_set)
    precision = tp / len(p_set) if p_set else 0.0
    recall = tp / len(t_set) if t_set else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)
