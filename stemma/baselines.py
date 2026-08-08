"""Symmetric weight-similarity baselines: the comparators Stemma is measured against.

Three families are implemented, one per line of the headline table:

* ``cosine_baseline``  -- flattened-weight cosine over a shared coordinate sample.
* ``cka_baseline``     -- REEF-style linear CKA on weight matrices, rows as samples.
* ``huref_baseline``   -- HuRef-style invariant terms (``W_q W_k^T`` and adjacent
  MLP products), which are stable under the permutation/scaling symmetries of a
  transformer, then compared by cosine.

Claim: direction
Every statistic here obeys ``f(a, b) == f(b, a)`` by construction, so none of
them can answer "which one came first". That symmetry is not an implementation
shortcut -- it is the ceiling the paper's direction claim is measured against,
and :func:`baseline_direction` makes it explicit by always abstaining.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .types import ModelRef
from .utils import get_logger, layer_index_of, role_of, stable_hash

__all__ = [
    "cosine_baseline",
    "cka_baseline",
    "huref_baseline",
    "BASELINES",
    "baseline_direction",
]

log = get_logger(__name__)

#: Default budget knobs. Baselines are allowed to read a little more than the
#: sketcher does -- that asymmetry is itself a result (see the bytes-per-decision
#: column of the benchmark table).
DEFAULT_MAX_ROWS = 512
DEFAULT_COORDS = 100_000
DEFAULT_MAX_TENSORS = 8
DEFAULT_MIN_PARAMS = 1 << 14  # 16_384


# --------------------------------------------------------------------------- #
# Shared plumbing
# --------------------------------------------------------------------------- #


def _open(ref: Any, **loader_kw: Any) -> Tuple[Any, bool]:
    """Open ``ref`` (or pass through an already-open source).

    Claim: infra
    Baselines and Stemma must read through the same loader or the
    bytes-per-decision comparison in the benchmark would be meaningless.
    """
    if hasattr(ref, "get_tensor") and hasattr(ref, "index"):
        return ref, False
    from .remote_loader import open_model  # lazy: no network at import time

    return open_model(ref, **loader_kw), True


def _close(sources: Sequence[Any], owned: Sequence[bool]) -> None:
    for s, mine in zip(sources, owned):
        if mine:
            try:
                s.close()
            except Exception:  # pragma: no cover
                pass


def _shared_tensors(
    ia: Dict[str, Any],
    ib: Dict[str, Any],
    *,
    min_params: int = DEFAULT_MIN_PARAMS,
    limit: Optional[int] = DEFAULT_MAX_TENSORS,
    roles: Optional[Sequence[str]] = None,
) -> List[str]:
    """Names of 2D tensors present in both indexes with identical shapes.

    Claim: low-false-positive
    Requiring exact shape agreement stops the baselines (and Stemma) from
    scoring two unrelated checkpoints as similar merely because both happen to
    contain a matrix called ``model.layers.0.mlp.down_proj.weight``.
    """
    out: List[Tuple[int, str]] = []
    for name, meta in ia.items():
        other = ib.get(name)
        if other is None:
            continue
        sa = tuple(int(d) for d in meta.shape)
        sb = tuple(int(d) for d in other.shape)
        if len(sa) != 2 or sa != sb or min(sa) < 2:
            continue
        if sa[0] * sa[1] < int(min_params):
            continue
        if roles is not None and (role_of(name) not in roles):
            continue
        out.append((-(sa[0] * sa[1]), name))
    out.sort()
    names = [n for _, n in out]
    if limit is not None:
        names = names[: int(limit)]
    return names


def _sample_idx(shape: Sequence[int], n: int, seed: int) -> np.ndarray:
    """Deterministic flat-index sample, identical for both models being compared.

    Claim: infra
    """
    from .merge_decompose import sample_coordinates

    return sample_coordinates(shape, n, seed)


def _tensor_seed(seed: int, name: str) -> int:
    return int((int(seed) * 1_000_003 + int(stable_hash(name)[:8], 16)) & 0xFFFFFFFF)


def _cos01(x: np.ndarray, y: np.ndarray) -> float:
    """Cosine of two vectors mapped monotonically from [-1, 1] into [0, 1]."""
    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    if nx <= 0.0 or ny <= 0.0:
        return 0.0
    c = float(np.dot(x, y) / (nx * ny))
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.clip(0.5 * (1.0 + c), 0.0, 1.0))


# --------------------------------------------------------------------------- #
# 1. flattened-weight cosine
# --------------------------------------------------------------------------- #


def cosine_baseline(
    a: ModelRef,
    b: ModelRef,
    *,
    coords: int = DEFAULT_COORDS,
    max_rows: int = DEFAULT_MAX_ROWS,
    min_params: int = DEFAULT_MIN_PARAMS,
    max_tensors: int = DEFAULT_MAX_TENSORS,
    seed: int = 0,
    **loader_kw: Any,
) -> float:
    """Cosine similarity of the two models' weights on a shared coordinate sample.

    Claim: direction
    This is the simplest thing a practitioner reaches for, and it is exactly
    symmetric: ``cosine(a, b) == cosine(b, a)``, so it can rank relatedness but
    never orient an edge. Returned as ``(1 + cos) / 2`` so the value lives in
    ``[0, 1]`` as the contract requires (a strictly increasing transform, so
    AUC/ranking results are unaffected).

    Returns ``0.0`` (with a logged reason) when the two models share no tensor
    of identical shape.
    """
    sources: List[Any] = []
    owned: List[bool] = []
    try:
        for ref in (a, b):
            s, mine = _open(ref, **loader_kw)
            sources.append(s)
            owned.append(mine)
        sa, sb = sources
        names = _shared_tensors(
            sa.index(), sb.index(), min_params=min_params, limit=max_tensors
        )
        if not names:
            log.warning("cosine_baseline: no shared same-shape 2D tensor between the models")
            return 0.0

        budget = max(1, int(coords) // len(names))
        va: List[np.ndarray] = []
        vb: List[np.ndarray] = []
        for name in names:
            try:
                Wa = np.asarray(sa.get_tensor(name, dtype=np.float32, max_rows=max_rows))
                Wb = np.asarray(sb.get_tensor(name, dtype=np.float32, max_rows=max_rows))
            except Exception as exc:
                log.warning("cosine_baseline: skipping %s (%s)", name, exc)
                continue
            if Wa.shape != Wb.shape or Wa.size == 0:
                log.warning("cosine_baseline: shape mismatch on %s (%s vs %s)", name, Wa.shape, Wb.shape)
                continue
            idx = _sample_idx(Wa.shape, min(budget, Wa.size), _tensor_seed(seed, name))
            if idx.size == 0:
                continue
            va.append(Wa.reshape(-1)[idx].astype(np.float64))
            vb.append(Wb.reshape(-1)[idx].astype(np.float64))
        if not va:
            log.warning("cosine_baseline: every shared tensor failed to read")
            return 0.0
        x = np.nan_to_num(np.concatenate(va))
        y = np.nan_to_num(np.concatenate(vb))
        return _cos01(x, y)
    finally:
        _close(sources, owned)


# --------------------------------------------------------------------------- #
# 2. REEF-style linear CKA
# --------------------------------------------------------------------------- #


def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two centred sample-by-feature matrices.

    Claim: direction
    ``||Y'X||_F^2 / (||X'X||_F ||Y'Y||_F)`` is invariant to orthogonal
    transforms and isotropic scaling of either side -- and, being a ratio of
    Frobenius norms of Gram matrices, it is exactly symmetric in ``X`` and
    ``Y``, hence direction-blind.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    num = float(np.linalg.norm(Y.T @ X, ord="fro") ** 2)
    dx = float(np.linalg.norm(X.T @ X, ord="fro"))
    dy = float(np.linalg.norm(Y.T @ Y, ord="fro"))
    if dx <= 0.0 or dy <= 0.0:
        return 0.0
    return float(np.clip(num / (dx * dy), 0.0, 1.0))


def cka_baseline(
    a: ModelRef,
    b: ModelRef,
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
    n_rows: int = 512,
    n_cols: int = 1024,
    min_params: int = DEFAULT_MIN_PARAMS,
    max_tensors: int = DEFAULT_MAX_TENSORS,
    seed: int = 0,
    **loader_kw: Any,
) -> float:
    """REEF-style representation similarity, computed directly on weight matrices.

    Claim: direction
    Rows of ``W`` are treated as samples and columns as features, so linear CKA
    measures whether the two models' weight matrices span the same subspace --
    a strong relatedness signal and a completely symmetric one. It is the
    fairest "representation similarity" stand-in we can compute without running
    the models on data.

    Both models are subsampled at the **same** row and column indices (seeded by
    tensor name), because CKA compares aligned samples; mismatched shapes return
    ``0.0`` with a logged reason.
    """
    sources: List[Any] = []
    owned: List[bool] = []
    try:
        for ref in (a, b):
            s, mine = _open(ref, **loader_kw)
            sources.append(s)
            owned.append(mine)
        sa, sb = sources
        names = _shared_tensors(
            sa.index(), sb.index(), min_params=min_params, limit=max_tensors
        )
        if not names:
            log.warning("cka_baseline: no shared same-shape 2D tensor between the models")
            return 0.0

        scores: List[float] = []
        weights: List[float] = []
        for name in names:
            try:
                Wa = np.asarray(sa.get_tensor(name, dtype=np.float32, max_rows=max_rows))
                Wb = np.asarray(sb.get_tensor(name, dtype=np.float32, max_rows=max_rows))
            except Exception as exc:
                log.warning("cka_baseline: skipping %s (%s)", name, exc)
                continue
            if Wa.shape != Wb.shape or Wa.ndim != 2 or min(Wa.shape) < 2:
                log.warning("cka_baseline: shape mismatch on %s (%s vs %s)", name, Wa.shape, Wb.shape)
                continue
            rng = np.random.default_rng(_tensor_seed(seed, name))
            r, c = Wa.shape
            ri = np.arange(r) if r <= n_rows else np.sort(rng.permutation(r)[:n_rows])
            ci = np.arange(c) if c <= n_cols else np.sort(rng.permutation(c)[:n_cols])
            Xa = np.nan_to_num(Wa[np.ix_(ri, ci)].astype(np.float64))
            Xb = np.nan_to_num(Wb[np.ix_(ri, ci)].astype(np.float64))
            s_ = _linear_cka(Xa, Xb)
            if np.isfinite(s_):
                scores.append(s_)
                weights.append(float(ri.size * ci.size))
        if not scores:
            log.warning("cka_baseline: no usable tensor pair")
            return 0.0
        w = np.asarray(weights, dtype=np.float64)
        return float(np.clip(np.dot(np.asarray(scores), w) / w.sum(), 0.0, 1.0))
    finally:
        _close(sources, owned)


# --------------------------------------------------------------------------- #
# 3. HuRef-style invariant terms
# --------------------------------------------------------------------------- #


def _fused_qkv_split(W: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Split a GPT-2 style fused ``c_attn`` block into its q and k halves."""
    if W.ndim != 2:
        return None
    r, c = W.shape
    if c % 3 != 0:
        return None
    third = c // 3
    if third < 2:
        return None
    return W[:, :third], W[:, third : 2 * third]


def _adjacent_product(A: np.ndarray, B: np.ndarray) -> Optional[np.ndarray]:
    """``A[:, :r] @ B[:r, :]`` -- the product of two adjacent weight matrices.

    Claim: direction
    Products of adjacent layers cancel the permutation (and, for a diagonal
    rescaling pair, the scaling) freedom of the intermediate neurons, which is
    the invariance HuRef relies on. The contraction is truncated to the ``r``
    rows we actually fetched, identically on both sides, so the comparison stays
    a Range read; symmetry of the final cosine is untouched.
    """
    if A.ndim != 2 or B.ndim != 2:
        return None
    r = int(min(A.shape[1], B.shape[0]))
    if r < 2:
        return None
    return A[:, :r] @ B[:r, :]


def _huref_invariants(
    src: Any, names_by_layer: Dict[int, Dict[str, str]], max_rows: int
) -> Dict[str, np.ndarray]:
    """Compute the per-layer invariant blocks for one model."""
    out: Dict[str, np.ndarray] = {}
    for layer, roles in sorted(names_by_layer.items()):
        # --- attention: W_q W_k^T --------------------------------------------
        q_name, k_name = roles.get("attn_q"), roles.get("attn_k")
        if q_name is not None:
            try:
                Wq = np.asarray(src.get_tensor(q_name, dtype=np.float32, max_rows=max_rows),
                                dtype=np.float64)
                if k_name is None:
                    split = _fused_qkv_split(Wq)
                    if split is not None:
                        Q, K = split
                    else:
                        Q = K = None
                else:
                    Q = Wq
                    K = np.asarray(src.get_tensor(k_name, dtype=np.float32, max_rows=max_rows),
                                   dtype=np.float64)
                if Q is not None and K is not None and Q.shape[1] == K.shape[1]:
                    out[f"attn/{layer}"] = np.nan_to_num(Q @ K.T)
            except Exception as exc:
                log.debug("huref: attention invariant failed at layer %d (%s)", layer, exc)
        # --- mlp: product of adjacent matrices --------------------------------
        in_name, out_name = roles.get("mlp_in"), roles.get("mlp_out")
        if in_name is not None and out_name is not None:
            try:
                Win = np.asarray(src.get_tensor(in_name, dtype=np.float32, max_rows=max_rows),
                                 dtype=np.float64)
                Wout = np.asarray(src.get_tensor(out_name, dtype=np.float32, max_rows=max_rows),
                                  dtype=np.float64)
                P = _adjacent_product(Wout, Win)
                if P is None:
                    P = _adjacent_product(Win, Wout)
                if P is not None:
                    out[f"mlp/{layer}"] = np.nan_to_num(P)
            except Exception as exc:
                log.debug("huref: mlp invariant failed at layer %d (%s)", layer, exc)
    return out


def huref_baseline(
    a: ModelRef,
    b: ModelRef,
    *,
    max_rows: int = 256,
    n_layers: int = 6,
    seed: int = 0,
    **loader_kw: Any,
) -> float:
    """HuRef-style fingerprint similarity from permutation-invariant weight products.

    Claim: direction
    HuRef identifies a base LLM by comparing quantities that survive the
    symmetries of a transformer (``W_q W_k^T`` for attention, products of
    adjacent MLP matrices). Those invariants are excellent at saying *related /
    not related* and, being a cosine between two fingerprints, are perfectly
    symmetric -- they cannot tell parent from child, which is precisely the gap
    :mod:`stemma.direction` fills.

    Returns ``0.0`` (with a logged reason) if no layer yields a comparable pair
    of invariant blocks.
    """
    sources: List[Any] = []
    owned: List[bool] = []
    try:
        for ref in (a, b):
            s, mine = _open(ref, **loader_kw)
            sources.append(s)
            owned.append(mine)
        sa, sb = sources
        ia, ib = sa.index(), sb.index()

        def layer_map(index: Dict[str, Any]) -> Dict[int, Dict[str, str]]:
            m: Dict[int, Dict[str, str]] = {}
            for name, meta in index.items():
                if len(tuple(meta.shape)) != 2:
                    continue
                role = role_of(name)
                if role not in ("attn_q", "attn_k", "mlp_in", "mlp_out"):
                    continue
                li = layer_index_of(name)
                if li is None:
                    continue
                slot = m.setdefault(int(li), {})
                # deterministic choice when both gate_proj and up_proj match
                if role not in slot or name < slot[role]:
                    slot[role] = name
            return m

        ma, mb = layer_map(ia), layer_map(ib)
        shared_layers = sorted(set(ma) & set(mb))
        if not shared_layers:
            log.warning("huref_baseline: models share no comparable transformer layer")
            return 0.0
        # Spread the probe evenly over depth rather than taking the first N.
        if len(shared_layers) > n_layers:
            sel = np.linspace(0, len(shared_layers) - 1, int(n_layers)).round().astype(int)
            shared_layers = [shared_layers[i] for i in sorted(set(sel.tolist()))]

        # Only keep layers where both sides name the same tensors with equal shapes.
        keep_a: Dict[int, Dict[str, str]] = {}
        keep_b: Dict[int, Dict[str, str]] = {}
        for li in shared_layers:
            ra, rb = ma[li], mb[li]
            common_roles = {}
            for role in ("attn_q", "attn_k", "mlp_in", "mlp_out"):
                na, nb = ra.get(role), rb.get(role)
                if na is None or nb is None:
                    continue
                if tuple(ia[na].shape) != tuple(ib[nb].shape):
                    log.warning(
                        "huref_baseline: shape mismatch for %s at layer %d (%s vs %s)",
                        role, li, tuple(ia[na].shape), tuple(ib[nb].shape),
                    )
                    continue
                common_roles[role] = (na, nb)
            if common_roles:
                keep_a[li] = {r: v[0] for r, v in common_roles.items()}
                keep_b[li] = {r: v[1] for r, v in common_roles.items()}
        if not keep_a:
            log.warning("huref_baseline: no layer with matching invariant inputs")
            return 0.0

        inv_a = _huref_invariants(sa, keep_a, max_rows)
        inv_b = _huref_invariants(sb, keep_b, max_rows)
        keys = sorted(set(inv_a) & set(inv_b))
        fa: List[np.ndarray] = []
        fb: List[np.ndarray] = []
        for k in keys:
            A, B = inv_a[k], inv_b[k]
            if A.shape != B.shape or A.size == 0:
                log.warning("huref_baseline: invariant %s has mismatched shape; skipping", k)
                continue
            # Normalise each block so one huge layer cannot dominate the cosine.
            na = float(np.linalg.norm(A)) or 1.0
            nb = float(np.linalg.norm(B)) or 1.0
            fa.append((A / na).reshape(-1))
            fb.append((B / nb).reshape(-1))
        if not fa:
            log.warning("huref_baseline: no comparable invariant block")
            return 0.0
        return _cos01(np.concatenate(fa), np.concatenate(fb))
    finally:
        _close(sources, owned)


# --------------------------------------------------------------------------- #
# Registry + the deliberate abstention
# --------------------------------------------------------------------------- #

#: Name -> callable, all with the signature ``f(a, b, **kw) -> float in [0, 1]``.
BASELINES: Dict[str, Callable[..., float]] = {
    "cosine": cosine_baseline,
    "cka": cka_baseline,
    "huref": huref_baseline,
}


def baseline_direction(*args: Any, **kwargs: Any) -> str:
    """Always returns ``"unknown"`` -- baselines cannot orient an edge.

    Claim: direction
    Every statistic in :data:`BASELINES` satisfies ``f(a, b) == f(b, a)``
    exactly, so no decision rule built on top of one can do better than chance
    at "is A the parent of B, or B the parent of A": guessing gives 50% and the
    honest answer is abstention. This function exists so the benchmark can
    report that 50% ceiling as a *measured* baseline row rather than as an
    assertion, and so nobody is tempted to bolt a coin flip onto a symmetric
    score and call it provenance.
    """
    return "unknown"
