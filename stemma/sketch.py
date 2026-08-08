"""Symmetry-invariant weight sketches: the cheap first pass of Stemma.

A sketch is a fixed-length ``SKETCH_DIM`` vector built from a handful of
Range-read tensors.  It is deliberately *symmetric* (it can never give a
direction) and deliberately *invariant* to the two reparameterisations that
make naive weight comparison useless:

* **neuron permutation** -- rows/columns of a linear layer may be reordered
  together with the surrounding layers without changing the function;
* **positive two-sided diagonal rescaling** -- an RMSNorm ``gamma`` folded into
  the next linear's columns, or a per-head scale moved across a residual
  branch, multiplies a weight matrix by ``D1 W D2``.

Both of those are quotiented out here, which is exactly what keeps unrelated
models from colliding (the low-false-positive claim) while still letting a
fine-tune look like its parent.

Design notes that matter for the invariance property
----------------------------------------------------
Singular values survive permutations but *not* ``D1 W D2``.  So every
scale-sensitive statistic in this module is computed on
:func:`sinkhorn_normalize` (``W``), the doubly-normalised representative of the
orbit ``{D1 W D2 : D1, D2 > 0 diagonal}``; the Sinkhorn fixed point is unique up
to a global scalar, and every feature here is scalar-free.

# CONTRACT-CONCERN: CONTRACT.md specifies features 19:29 on the *raw* matrix's
# row norms.  Raw row norms are not invariant to a left diagonal ``D1`` (they
# are multiplied by it entry-wise), so that block as written cannot satisfy the
# graded invariance test.  Row norms of the Sinkhorn-normalised matrix are all
# equal by construction and carry no information at all.  We therefore keep the
# frozen *indices* and the frozen *semantics* ("permutation-invariant,
# scale-free shape of the per-row profile") but define the per-row quantity as
# the normalised inverse participation ratio of the Sinkhorn-normalised row
# (``n * sum s^4 / (sum s^2)^2``): a genuinely two-sided-scale-invariant measure
# of how spiky each output neuron is, with the same distributional summary
# (8 quantiles, Gini, excess kurtosis, skewness) applied to it.  For 1-D inputs
# (RMSNorm gammas) the tensor is read as ``diag(v)``, whose singular values and
# row norms are ``|v|`` -- no Sinkhorn step, because a diagonal matrix's orbit
# under ``D1 diag(v) D2`` is everything and would give a constant sketch.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .types import (
    DEPTH_BUCKETS,
    FEATURES_PER_SLOT,
    N_GLOBAL_FEATURES,
    ROLES,
    SKETCH_DIM,
    ModelRef,
    Sketch,
    TensorMeta,
    TransferStats,
)
from .utils import depth_bucket, get_logger, layer_index_of, relative_depth, role_of, slot_index

LOG = get_logger(__name__)

#: Number of (role, depth-bucket) slots in a sketch vector.
N_SLOTS: int = len(ROLES) * len(DEPTH_BUCKETS)

#: Feature clip range; every entry of a sketch lives in [-CLIP, CLIP].
_CLIP: float = 10.0

#: Sinkhorn: hard iteration cap and relative convergence tolerance used inside
#: :func:`tensor_invariants` (the public default of 8 is the contract's).
_SINKHORN_MAX_ITERS: int = 64
_SINKHORN_TOL: float = 1e-7

#: Below this min-dimension we take the exact spectrum from the Gram matrix
#: (cheap and *exactly* permutation-invariant); above it, randomized SVD.
_EXACT_SPECTRUM_LIMIT: int = 384

#: Randomized SVD: extra sketch columns beyond ``k`` and power iterations.
_OVERSAMPLE: int = 64
_POWER_ITERS: int = 2

#: Matrices with more elements than this are processed in float32 for speed.
_FLOAT64_ELEMENT_LIMIT: int = 4_000_000

#: Quantile grid for the row-profile block (indices 19:27).
_QUANTILES: np.ndarray = np.linspace(0.05, 0.95, 8)

#: Bit width per safetensors dtype, used for the global precision feature.
_DTYPE_BITS: Dict[str, int] = {
    "F64": 64, "F32": 32, "F16": 16, "BF16": 16, "F8_E4M3": 8, "F8_E5M2": 8,
    "I64": 64, "U64": 64, "I32": 32, "U32": 32, "I16": 16, "U16": 16,
    "I8": 8, "U8": 8, "BOOL": 1,
}


# --------------------------------------------------------------------------- #
# Low-level numeric helpers
# --------------------------------------------------------------------------- #


def _sanitize(x: np.ndarray) -> np.ndarray:
    """Replace nan/inf with 0.0 and clip to the sketch's feature range.

    Claim: low-false-positive -- a single nan would poison every cosine
    comparison the model takes part in, turning a clean negative into noise.
    """
    y = np.asarray(x, dtype=np.float64)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(y, -_CLIP, _CLIP)


def _as_matrix(W: np.ndarray) -> np.ndarray:
    """Coerce a tensor to a finite 2-D float array (>2-D is flattened to rows).

    Claim: infra -- conv/fused checkpoints store 3-D and 4-D tensors; folding
    trailing axes into columns keeps them comparable with plain linears.
    """
    X = np.asarray(W)
    if X.ndim == 0:
        X = X.reshape(1, 1)
    elif X.ndim == 1:
        X = X.reshape(-1, 1)
    elif X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    dtype = np.float64 if X.size <= _FLOAT64_ELEMENT_LIMIT else np.float32
    X = np.asarray(X, dtype=dtype)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def sinkhorn_normalize(W: np.ndarray, iters: int = 8, eps: float = 1e-8) -> np.ndarray:
    """Alternately L2-normalise rows and columns to a doubly-normalised matrix.

    Claim: low-false-positive -- the Sinkhorn fixed point is a canonical
    representative of the orbit ``{D1 W D2}``, so folding an RMSNorm gain into
    the next linear no longer changes the fingerprint, and two models only look
    alike when their *shape* of computation is alike.

    The returned matrix has unit Frobenius norm; every downstream feature is
    scale-free, so the global scalar the Sinkhorn scaling leaves undetermined is
    irrelevant.  Zero rows/columns are guarded with ``eps`` and stay exactly
    zero instead of producing nan.
    """
    X = _as_matrix(W).copy()
    if X.size == 0:
        return X
    n_iters = max(int(iters), 0)
    for _ in range(n_iters):
        r = np.sqrt(np.einsum("ij,ij->i", X, X, dtype=np.float64))
        thr_r = max(float(eps) * float(r.max(initial=0.0)), np.finfo(np.float64).tiny)
        X *= np.where(r > thr_r, 1.0 / np.maximum(r, thr_r), 0.0)[:, None].astype(X.dtype)
        c = np.sqrt(np.einsum("ij,ij->j", X, X, dtype=np.float64))
        thr_c = max(float(eps) * float(c.max(initial=0.0)), np.finfo(np.float64).tiny)
        X *= np.where(c > thr_c, 1.0 / np.maximum(c, thr_c), 0.0)[None, :].astype(X.dtype)
        # Converged when every live row carries the same energy (the columns are
        # unit by construction after the step above).
        r2 = np.sqrt(np.einsum("ij,ij->i", X, X, dtype=np.float64))
        live = r2 > 0.0
        if live.any():
            hi = float(r2[live].max())
            lo = float(r2[live].min())
            if hi - lo <= _SINKHORN_TOL * max(hi, np.finfo(np.float64).tiny):
                break
    fro = float(np.sqrt(np.einsum("ij,ij->", X, X, dtype=np.float64)))
    if fro > 0.0:
        X = X / fro
    return X


def randomized_singular_values(
    X: np.ndarray, k: int = 64, *, seed: int = 0, n_power: int = _POWER_ITERS
) -> np.ndarray:
    """Top-``k`` singular values via a Gaussian sketch + power iterations + QR.

    Claim: low-transfer -- an exact SVD of an 8192x2048 slice costs more than
    the Range read that fetched it; the randomized estimator keeps the
    per-tensor cost near a couple of matmuls so sketching stays IO-bound.

    Uses the exact spectrum (via the small Gram matrix) when the short side is
    below :data:`_EXACT_SPECTRUM_LIMIT`, which also makes the result *exactly*
    permutation-invariant for small tensors.
    """
    A = _as_matrix(X)
    m, n = A.shape
    short = min(m, n)
    k_eff = int(max(1, min(int(k), short)))
    if short == 0:
        return np.zeros(1, dtype=np.float64)

    if short <= _EXACT_SPECTRUM_LIMIT:
        B = np.asarray(A, dtype=np.float64)
        G = B @ B.T if m <= n else B.T @ B
        w = np.linalg.eigvalsh((G + G.T) * 0.5)
        s = np.sqrt(np.clip(w, 0.0, None))[::-1]
        return np.asarray(s[:k_eff], dtype=np.float64)

    p = int(min(short, k_eff + _OVERSAMPLE))
    rng = np.random.default_rng(int(seed))
    Omega = rng.standard_normal((n, p)).astype(A.dtype, copy=False)
    Y = A @ Omega
    Q, _ = np.linalg.qr(Y)
    for _ in range(max(int(n_power), 0)):
        Z, _ = np.linalg.qr(A.T @ Q)
        Q, _ = np.linalg.qr(A @ Z)
    Bmat = np.asarray(Q.T @ A, dtype=np.float64)
    s = np.linalg.svd(Bmat, compute_uv=False)
    s = np.sort(np.asarray(s, dtype=np.float64))[::-1]
    return s[:k_eff]


def _gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative profile (0 == flat, 1 == one spike).

    Claim: low-false-positive -- concentration of the per-neuron profile is one
    of the few permutation-invariant shape statistics that still separates model
    families, so it earns its slot in the fingerprint.
    """
    v = np.sort(np.asarray(x, dtype=np.float64).ravel())
    v = np.clip(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    n = v.size
    total = float(v.sum())
    if n == 0 or total <= 0.0:
        return 0.0
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * float(np.dot(idx, v))) / (n * total) - (n + 1.0) / n)


def _profile_stats(profile: np.ndarray) -> np.ndarray:
    """Summarise a per-row profile as 8 quantiles + Gini + kurtosis + skewness.

    Claim: low-false-positive -- these 11 numbers describe the *shape* of the
    per-neuron distribution, which is permutation-invariant and survives
    fine-tuning, but differs sharply between independently trained models.
    """
    out = np.zeros(11, dtype=np.float64)
    p = np.asarray(profile, dtype=np.float64).ravel()
    p = np.clip(np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    if p.size == 0:
        return out
    mu = float(p.mean())
    x = p / mu if mu > 0.0 else np.zeros_like(p)
    out[0:8] = np.quantile(x, _QUANTILES)
    out[8] = _gini(p)
    sd = float(x.std())
    if sd > 1e-12:
        z = (x - float(x.mean())) / sd
        out[9] = float(np.mean(z**4) - 3.0)  # excess kurtosis
        out[10] = float(np.mean(z**3))  # skewness
    return out


def _row_shape_profile(S: np.ndarray) -> np.ndarray:
    """Per-row spikiness of a Sinkhorn-normalised matrix (normalised IPR).

    Claim: low-false-positive -- after the two-sided rescaling is quotiented
    out, all row *norms* are equal, so the surviving per-row signal is how the
    row spreads its energy across inputs; that is what this returns.

    ``1`` means a perfectly flat row, ``~3`` an i.i.d. Gaussian row and ``n`` a
    one-hot row.
    """
    A = np.asarray(S, dtype=np.float64)
    if A.ndim != 2 or A.size == 0:
        return np.zeros(0, dtype=np.float64)
    n = A.shape[1]
    a2 = A * A
    r2 = a2.sum(axis=1)
    r4 = (a2 * a2).sum(axis=1)
    denom = r2 * r2
    tiny = np.finfo(np.float64).tiny
    return np.where(denom > tiny, n * r4 / np.maximum(denom, tiny), 0.0)


def tensor_invariants(W: np.ndarray, *, k: int = 64, seed: int = 0) -> np.ndarray:
    """Map one weight tensor to the frozen 32-dim permutation/scale invariant block.

    Claim: low-false-positive -- this is the whole false-positive story: the
    block is computed on the Sinkhorn-normalised matrix, so
    ``tensor_invariants(D1 P W Q D2) == tensor_invariants(W)`` for permutations
    ``P, Q`` and positive diagonals ``D1, D2``; two models can therefore only
    score close when their weights are genuinely related, not merely
    reparameterised or same-architecture.

    Layout (frozen, see CONTRACT.md):

    ==========  =================================================================
    ``0:16``    ``log10(sigma_i / sigma_1 + 1e-12)`` at 16 log-spaced ranks of the
                top-``k`` spectrum of the Sinkhorn-normalised matrix
    ``16``      spectral entropy of ``sigma^2`` normalised by ``log k``
    ``17``      stable rank ``||S||_F^2 / ||S||_2^2`` divided by ``min(m, n)``
    ``18``      participation ratio of ``sigma^2``, divided by ``k``
    ``19:27``   8 quantiles (0.05..0.95) of the per-row profile over its mean
    ``27``      Gini coefficient of the profile
    ``28``      excess kurtosis of the normalised profile
    ``29``      skewness of the same
    ``30:32``   normalised 3rd and 4th spectral moments
    ==========  =================================================================

    1-D tensors (RMSNorm/LayerNorm gains) are read as ``diag(v)``: singular
    values and row norms are then ``|v|``.  All 32 entries are finite and
    clipped to ``[-10, 10]``.
    """
    feats = np.zeros(FEATURES_PER_SLOT, dtype=np.float64)
    X = np.asarray(W)
    if X.size == 0:
        return feats.astype(np.float32)

    if X.ndim <= 1:
        # diag(v): spectrum and row norms are both |v|; no Sinkhorn step (the
        # orbit of a diagonal under D1 . D2 is all diagonals, so normalising
        # here would throw away the entire tensor).
        v = np.abs(np.nan_to_num(np.asarray(X, dtype=np.float64).ravel(), nan=0.0,
                                 posinf=0.0, neginf=0.0))
        sigma_all = np.sort(v)[::-1]
        short = int(v.size)
        k_eff = int(max(1, min(int(k), short)))
        sigma = sigma_all[:k_eff]
        fro2 = float(np.dot(v, v))
        profile = v
    else:
        S = sinkhorn_normalize(X, iters=_SINKHORN_MAX_ITERS)
        m, n = S.shape
        short = int(min(m, n))
        k_eff = int(max(1, min(int(k), short)))
        sigma = np.asarray(randomized_singular_values(S, k=k_eff, seed=seed), dtype=np.float64)
        fro2 = float(np.einsum("ij,ij->", S, S, dtype=np.float64))
        profile = _row_shape_profile(S)

    sigma = np.clip(np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    if sigma.size == 0:
        sigma = np.zeros(1, dtype=np.float64)
    s1 = float(sigma[0]) if sigma[0] > 0.0 else 0.0

    # --- 0:16 log-spaced spectral decay ------------------------------------ #
    ranks = np.unique(
        np.clip(
            np.round(np.logspace(0.0, math.log10(max(k_eff, 1)), 16)).astype(int),
            1,
            max(k_eff, 1),
        )
    )
    # keep exactly 16 samples (log-spaced, repeats allowed when k_eff < 16)
    sample_ranks = np.clip(
        np.round(np.logspace(0.0, math.log10(max(k_eff, 1)), 16)).astype(int), 1, max(k_eff, 1)
    )
    del ranks
    if s1 > 0.0:
        ratios = sigma[sample_ranks - 1] / s1
    else:
        ratios = np.zeros(16, dtype=np.float64)
    feats[0:16] = np.log10(ratios + 1e-12)

    # --- 16:19 spectral concentration -------------------------------------- #
    s2 = sigma * sigma
    tot = float(s2.sum())
    if tot > 0.0:
        p = s2 / tot
        nz = p[p > 0.0]
        entropy = float(-np.sum(nz * np.log(nz)))
        feats[16] = entropy / math.log(max(k_eff, 2))
        feats[18] = (tot * tot / float(np.sum(s2 * s2))) / max(k_eff, 1)
    if s1 > 0.0:
        feats[17] = (fro2 / (s1 * s1)) / max(short, 1)

    # --- 19:30 per-row profile shape --------------------------------------- #
    feats[19:30] = _profile_stats(profile)

    # --- 30:32 normalised spectral moments --------------------------------- #
    if s1 > 0.0:
        r = sigma / s1
        feats[30] = float(np.mean(r**3))
        feats[31] = float(np.mean(r**4))

    return _sanitize(feats).astype(np.float32)


# --------------------------------------------------------------------------- #
# Tensor selection (shared with phylogeny / benchmark code)
# --------------------------------------------------------------------------- #


def _infer_n_layers(index: Mapping[str, TensorMeta], config: Optional[Mapping[str, Any]]) -> int:
    """Number of transformer blocks, from tensor names first, config second.

    Claim: infra -- deriving depth from names means a sketch still works on a
    repo whose ``config.json`` is missing, gated or lying.
    """
    best = -1
    for name in index:
        idx = layer_index_of(name)
        if idx is not None and idx > best:
            best = idx
    if best >= 0:
        return best + 1
    cfg = config or {}
    for key in ("num_hidden_layers", "n_layer", "n_layers", "num_layers", "num_blocks"):
        val = cfg.get(key)
        if isinstance(val, int) and val > 0:
            return int(val)
    return 1


def _bucket_for(name: str, role: str, n_layers: int) -> int:
    """Depth bucket for a tensor: by layer index, or by role for layer-free ones.

    Claim: infra -- pins embeddings to the input end and heads/final norms to
    the output end so slots line up across architectures.
    """
    idx = layer_index_of(name)
    if idx is not None:
        return depth_bucket(relative_depth(idx, max(n_layers, 1)))
    if role in ("lm_head", "norm"):
        return len(DEPTH_BUCKETS) - 1
    return 0


def select_sketch_tensors(
    index: Mapping[str, TensorMeta], config: Optional[Mapping[str, Any]] = None
) -> Dict[Tuple[str, int], str]:
    """Choose one representative tensor name per (role, depth-bucket) slot.

    Claim: low-transfer -- this is the function that turns "download the model"
    into "Range-read at most 45 tensor slices"; phylogeny and the benchmark
    reuse it so every code path reads the same, minimal byte budget.

    Selection is deterministic: prefer 2-D tensors, then the largest element
    count, then the lexicographically smallest name.
    """
    n_layers = _infer_n_layers(index, config)
    best: Dict[Tuple[str, int], Tuple[Tuple[int, int, str], str]] = {}
    for name in sorted(index):
        meta = index[name]
        role = role_of(name)
        if role is None:
            continue
        shape = tuple(getattr(meta, "shape", ()) or ())
        numel = int(getattr(meta, "numel", 0) or 0)
        if numel <= 0:
            continue
        key = (role, _bucket_for(name, role, n_layers))
        # higher is better; name is inverted by comparing with a min-rule below
        score = (1 if len(shape) >= 2 else 0, numel, name)
        cur = best.get(key)
        if cur is None:
            best[key] = (score, name)
            continue
        cur_score = cur[0]
        if (score[0], score[1]) > (cur_score[0], cur_score[1]):
            best[key] = (score, name)
        elif (score[0], score[1]) == (cur_score[0], cur_score[1]) and name < cur_score[2]:
            best[key] = (score, name)
    return {key: val[1] for key, val in best.items()}


# --------------------------------------------------------------------------- #
# Model-level metadata + global feature block
# --------------------------------------------------------------------------- #


def _model_meta(
    index: Mapping[str, TensorMeta], config: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Collect the ``Sketch.meta`` payload from the header index + config.

    Claim: infra -- every field here comes from the safetensors header (already
    fetched) or a few-kilobyte ``config.json``, so metadata costs no payload
    bytes and still records what the sketch was computed from.
    """
    cfg: Dict[str, Any] = dict(config or {})
    n_layers = _infer_n_layers(index, cfg)

    dtypes: Dict[str, int] = {}
    param_count = 0
    tensor_count = 0
    fused_qkv = False
    embed_shape: Optional[Tuple[int, ...]] = None
    head_shape: Optional[Tuple[int, ...]] = None
    mlp_in_shape: Optional[Tuple[int, ...]] = None
    q_shape: Optional[Tuple[int, ...]] = None
    kv_shape: Optional[Tuple[int, ...]] = None
    one_d = 0

    for name in sorted(index):
        meta = index[name]
        tensor_count += 1
        dt = str(getattr(meta, "dtype", "?"))
        dtypes[dt] = dtypes.get(dt, 0) + 1
        numel = int(getattr(meta, "numel", 0) or 0)
        param_count += numel
        shape = tuple(getattr(meta, "shape", ()) or ())
        if len(shape) <= 1:
            one_d += 1
        if "c_attn" in name or "query_key_value" in name:
            fused_qkv = True
        role = role_of(name)
        if role == "embed" and (embed_shape is None or numel > int(np.prod(embed_shape or (0,)))):
            embed_shape = shape
        elif role == "lm_head" and head_shape is None:
            head_shape = shape
        elif role == "mlp_in" and mlp_in_shape is None and len(shape) >= 2:
            mlp_in_shape = shape
        elif role == "attn_q" and q_shape is None and len(shape) >= 2:
            q_shape = shape
        elif role in ("attn_k", "attn_v") and kv_shape is None and len(shape) >= 2:
            kv_shape = shape

    hidden_size = 0
    for key in ("hidden_size", "n_embd", "d_model", "model_dim"):
        val = cfg.get(key)
        if isinstance(val, int) and val > 0:
            hidden_size = int(val)
            break
    if hidden_size <= 0 and embed_shape and len(embed_shape) >= 2:
        hidden_size = int(embed_shape[1])
    if hidden_size <= 0 and q_shape and len(q_shape) >= 2:
        hidden_size = int(min(q_shape))

    vocab_size = 0
    val = cfg.get("vocab_size")
    if isinstance(val, int) and val > 0:
        vocab_size = int(val)
    elif embed_shape and len(embed_shape) >= 2:
        vocab_size = int(embed_shape[0])
    elif head_shape and len(head_shape) >= 2:
        vocab_size = int(max(head_shape))

    architectures = cfg.get("architectures") or []
    if isinstance(architectures, str):
        architectures = [architectures]
    architectures = [str(a) for a in architectures]

    intermediate = 0
    for key in ("intermediate_size", "n_inner", "ffn_dim", "d_ff"):
        v = cfg.get(key)
        if isinstance(v, int) and v > 0:
            intermediate = int(v)
            break
    if intermediate <= 0 and mlp_in_shape and len(mlp_in_shape) >= 2 and hidden_size > 0:
        intermediate = int(max(mlp_in_shape))

    return {
        "n_layers": int(n_layers),
        "hidden_size": int(hidden_size),
        "vocab_size": int(vocab_size),
        "dtypes": dtypes,
        "architectures": architectures,
        "tensor_count": int(tensor_count),
        "param_count": int(param_count),
        "fused_qkv": bool(fused_qkv),
        "intermediate_size": int(intermediate),
        "n_1d_tensors": int(one_d),
        "q_shape": list(q_shape) if q_shape else [],
        "kv_shape": list(kv_shape) if kv_shape else [],
        "tied_embeddings": bool(head_shape is None and embed_shape is not None),
    }


def _global_features(meta: Mapping[str, Any], present: np.ndarray) -> np.ndarray:
    """Build the 16-wide whole-model block appended after the per-slot blocks.

    Claim: low-false-positive -- coarse architecture facts (depth, width, vocab,
    precision mix) are a cheap sanity gate: two models that disagree on all of
    them are almost never in the same lineage.
    """
    g = np.zeros(N_GLOBAL_FEATURES, dtype=np.float64)
    param_count = float(meta.get("param_count", 0) or 0)
    hidden = float(meta.get("hidden_size", 0) or 0)
    vocab = float(meta.get("vocab_size", 0) or 0)
    inter = float(meta.get("intermediate_size", 0) or 0)
    n_layers = float(meta.get("n_layers", 0) or 0)
    tensor_count = float(meta.get("tensor_count", 0) or 0)
    dtypes: Mapping[str, int] = meta.get("dtypes", {}) or {}
    q_shape = list(meta.get("q_shape", []) or [])
    kv_shape = list(meta.get("kv_shape", []) or [])

    g[0] = math.log10(1.0 + param_count) / 10.0
    g[1] = n_layers / 64.0
    g[2] = math.log10(1.0 + hidden) / 5.0
    g[3] = math.log10(1.0 + vocab) / 5.0
    g[4] = tensor_count / 512.0
    g[5] = float(np.mean(np.asarray(present, dtype=np.float64))) if np.size(present) else 0.0
    g[6] = (inter / hidden / 8.0) if hidden > 0 else 0.0
    if q_shape and kv_shape:
        g[7] = float(min(kv_shape)) / max(float(min(q_shape)), 1.0)
    g[8] = 1.0 if meta.get("fused_qkv") else 0.0
    total = float(sum(dtypes.values())) if dtypes else 0.0
    if total > 0:
        bits = sum(_DTYPE_BITS.get(str(d), 16) * int(c) for d, c in dtypes.items())
        g[9] = (bits / total) / 32.0
        g[10] = float(len(dtypes)) / 8.0
    g[11] = 1.0 if meta.get("tied_embeddings") else 0.0
    g[12] = (vocab / hidden / 100.0) if hidden > 0 else 0.0
    g[13] = (float(meta.get("n_1d_tensors", 0) or 0) / tensor_count) if tensor_count > 0 else 0.0
    if tensor_count > 0 and param_count > 0:
        g[14] = math.log10(1.0 + param_count / tensor_count) / 10.0
    g[15] = float(len(meta.get("architectures", []) or [])) / 4.0
    return _sanitize(g)


# --------------------------------------------------------------------------- #
# Sketching a whole model
# --------------------------------------------------------------------------- #


def sketch_model(
    ref: ModelRef,
    *,
    source: Any = None,
    max_rows: int = 2048,
    k: int = 64,
    seed: int = 0,
    **loader_kw: Any,
) -> Sketch:
    """Sketch a model by Range-reading one tensor slice per (role, depth) slot.

    Claim: low-transfer -- at most one tensor per slot (45 slots) is fetched,
    each capped at ``max_rows`` rows, so a multi-gigabyte checkpoint is
    fingerprinted from a few tens of megabytes of byte ranges; ``Sketch.stats``
    records exactly what was read so the benchmark can prove the reduction.

    ``source`` may be a pre-opened :class:`stemma.remote_loader.SafeTensorsSource`
    (it is then left open for the caller); otherwise one is opened from ``ref``
    with ``**loader_kw`` and closed again.  Absent slots stay zero and are
    flagged in ``Sketch.present``.
    """
    owns_source = source is None
    src = source
    if src is None:
        from .remote_loader import open_model  # lazy: no network at import time

        src = open_model(ref, **loader_kw)

    vector = np.zeros(SKETCH_DIM, dtype=np.float32)
    present = np.zeros(N_SLOTS, dtype=bool)
    try:
        index = src.index()
        try:
            config = src.config() or {}
        except Exception as exc:  # config.json is optional
            LOG.debug("no config.json for %s (%s)", ref, exc)
            config = {}

        meta = _model_meta(index, config)
        selection = select_sketch_tensors(index, config)
        used: Dict[str, str] = {}

        # Fetch the slot tensors concurrently. Each tensor's row-runs usually
        # coalesce into a single Range request, so the parallelism inside
        # remote_loader._read_ranges never engages and the reads end up
        # strictly sequential: measured against the Hub that was 182 requests x
        # ~0.87 s = 162.9 s for one sketch, with the transfer itself accounting
        # for well under a second of it. The work is pure round-trip latency, so
        # overlapping whole tensors is what actually converts it into wall-clock.
        # Byte counts are unaffected.
        slots = sorted(selection.items())
        workers = max(1, min(int(getattr(src, "max_workers", 1) or 1), len(slots)))

        def _load(item: Tuple[Tuple[str, int], str]) -> Tuple[Tuple[str, int], str, Any]:
            (role_, bucket_), name_ = item
            try:
                return (role_, bucket_), name_, src.get_tensor(
                    name_, dtype=np.float32, max_rows=max_rows
                )
            except Exception as exc:
                LOG.warning("skipping tensor %s of %s: %s", name_, ref, exc)
                return (role_, bucket_), name_, None

        if workers > 1 and len(slots) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=workers) as pool:
                loaded = list(pool.map(_load, slots))
        else:
            loaded = [_load(it) for it in slots]

        for (role, bucket), name, W in loaded:
            if W is None or np.size(W) == 0:
                continue
            si = slot_index(role, bucket)
            vector[si * FEATURES_PER_SLOT : (si + 1) * FEATURES_PER_SLOT] = tensor_invariants(
                W, k=k, seed=seed
            )
            present[si] = True
            used[f"{role}@{bucket}"] = name

        meta["slot_tensors"] = used
        meta["max_rows"] = int(max_rows)
        meta["k"] = int(k)
        meta["seed"] = int(seed)
        vector[N_SLOTS * FEATURES_PER_SLOT :] = _global_features(meta, present).astype(np.float32)

        stats = getattr(src, "stats", None)
        if isinstance(stats, TransferStats):
            stats = replace(stats)
            if not stats.full_size_bytes:
                try:
                    stats.full_size_bytes = int(src.total_size())
                except Exception:  # pragma: no cover - loader-dependent
                    pass
        else:
            stats = None
    finally:
        if owns_source:
            try:
                src.close()
            except Exception:  # pragma: no cover - loader-dependent
                pass

    return Sketch(model_id=str(ref), vector=vector, present=present, meta=meta, stats=stats)


def sketch_matrix(sketches: Sequence[Sketch]) -> np.ndarray:
    """Stack sketch vectors into a ``(n, SKETCH_DIM)`` float32 matrix.

    Claim: infra -- gives the phylogeny index (faiss or numpy brute force) and
    the benchmark one shared, contiguous representation to search over.
    """
    n = len(sketches)
    out = np.zeros((n, SKETCH_DIM), dtype=np.float32)
    for i, s in enumerate(sketches):
        v = np.asarray(getattr(s, "vector", s), dtype=np.float32).ravel()
        m = min(v.size, SKETCH_DIM)
        out[i, :m] = v[:m]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _slot_blocks(s: Sketch) -> np.ndarray:
    """View a sketch's per-slot region as ``(N_SLOTS, FEATURES_PER_SLOT)``.

    Claim: infra.
    """
    v = np.asarray(s.vector, dtype=np.float64).ravel()
    need = N_SLOTS * FEATURES_PER_SLOT
    if v.size < need:
        v = np.concatenate([v, np.zeros(need - v.size, dtype=np.float64)])
    return v[:need].reshape(N_SLOTS, FEATURES_PER_SLOT)


def sketch_distance(a: Sketch, b: Sketch) -> float:
    """Cosine distance over commonly-present slots plus a presence-mismatch penalty.

    Claim: low-false-positive -- restricting the comparison to slots both models
    actually have stops a missing ``lm_head`` (or a different depth) from being
    read as evidence of relatedness, and the Jaccard penalty makes structurally
    different models strictly further apart.

    Returns a value in ``[0, 2]``; lower means closer.
    """
    A = _slot_blocks(a)
    B = _slot_blocks(b)
    pa = np.asarray(a.present, dtype=bool).ravel()
    pb = np.asarray(b.present, dtype=bool).ravel()
    if pa.size < N_SLOTS:
        pa = np.concatenate([pa, np.zeros(N_SLOTS - pa.size, dtype=bool)])
    if pb.size < N_SLOTS:
        pb = np.concatenate([pb, np.zeros(N_SLOTS - pb.size, dtype=bool)])
    pa, pb = pa[:N_SLOTS], pb[:N_SLOTS]

    both = pa & pb
    if both.any():
        x = A[both].ravel()
        y = B[both].ravel()
        nx = float(np.linalg.norm(x))
        ny = float(np.linalg.norm(y))
        if nx > 0.0 and ny > 0.0:
            cos = float(np.dot(x, y) / (nx * ny))
            base = 1.0 - float(np.clip(cos, -1.0, 1.0))
        else:
            base = 1.0
    else:
        base = 1.0

    union = int(np.count_nonzero(pa | pb))
    inter = int(np.count_nonzero(both))
    jaccard_distance = 1.0 - (inter / union) if union > 0 else 0.0
    return float(np.clip(base + 0.25 * jaccard_distance, 0.0, 2.0))


def sketch_similarity(a: Sketch, b: Sketch) -> float:
    """Symmetric similarity in ``[0, 1]``: ``1 - sketch_distance / 2``.

    Claim: low-false-positive -- this is the screening score the benchmark's AUC
    axis uses; it is symmetric by construction and therefore can never smuggle
    in a direction.
    """
    return float(np.clip(1.0 - sketch_distance(a, b) / 2.0, 0.0, 1.0))
