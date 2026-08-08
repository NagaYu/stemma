"""Asymmetric weight evidence: which of two related checkpoints is the ancestor.

This is the module the whole project rests on.  A sketch, a cosine, a CKA and a
HuRef-style invariant are all **symmetric** functions of two checkpoints: they
can say *how close* two models are, never *which way the arrow points*.  Every
statistic in this file is deliberately **anti-symmetric**, so swapping the two
arguments flips the sign of the answer instead of leaving it unchanged.

Claim: direction -- ``estimate_direction(a, b).llr == -estimate_direction(b, a).llr``
by construction, and its sign is the derivation order.  That is exactly the
quantity a symmetric fingerprint is structurally incapable of producing.

How anti-symmetry is guaranteed
-------------------------------
Never "compare A to B and then compare B to A with different code".  Instead,
:func:`collect_pair_evidence` measures

* a **per-side** statistic ``g(X)`` for ``X in {A, B}`` (lattice fit, zero
  fraction, dtype precision, dead-row fraction, spectral mass, ...), and
* a **per-ordered-pair** statistic ``h(X, Y)`` evaluated *both ways*
  (``h(A, B)`` and ``h(B, A)``: zero-set containment, dead-row containment,
  outlier-channel containment, outgroup distances),

and :func:`direction_features` emits every feature through :func:`_antisym`,
which returns ``h_a - h_b`` (optionally normalised by a symmetric denominator).
Since IEEE-754 subtraction satisfies ``y - x == -(x - y)`` exactly and addition
is commutative, ``f(b, a) == -f(a, b)`` holds to the last bit, not merely to the
1e-6 the contract asks for.

Positive ``llr`` means **A is the parent of B**.

What the evidence families are worth (measured; see ``docs/FINDINGS.md``)
------------------------------------------------------------------------
The five families are *not* equally strong, and this module is weighted to say
so rather than to blend them into one flattering number:

===========================  ==========================  ==================
family                       mechanism                   strength
===========================  ==========================  ==================
(c) quantisation/pruning     value lattice, zero subset  near-deterministic
(b) orphan embeddings        untrained rows, vocab grew  near-deterministic
(d) fossils                  dead rows, outlier channels strong when present
(a) delta spectrum           subspace energy, norms      weak (1e-3..1e-2)
(e) outgroup rooting         sibling distance to root    strong *if* supplied
===========================  ==========================  ==================

Families (b), (c) and (d) work because the underlying operation is **lossy and
irreversible**: you cannot un-quantise, un-prune, un-kill a neuron or shrink a
vocabulary, so the scar can only ever appear *downstream*.  Family (a) is a
tiebreaker and nothing more -- see :meth:`DirectionModel.default` for the
measured reason ``norm_growth_asym`` gets a hand-set weight of exactly zero.

Family (e) is the answer to the honest limitation family (a) exposes: for a
scar-free SFT/LoRA edge, both models have the same shapes, dtype, vocabulary and
no lossy scar, so two-model direction is only weakly identifiable.  Given a
third relative ``C`` (an outgroup), ``d(A, C) < d(B, C)`` consistently over
tensors puts ``A`` closer to the root, because a sibling's distance is dominated
by the shared ancestral component.  It is a *separate additive term* -- see
:attr:`DirectionModel.outgroup_weight` -- because ``DIRECTION_FEATURES`` is
frozen and has no key for it, and because keeping it separate is what lets the
benchmark report its contribution instead of hiding it inside a blend.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .types import (
    DIRECTION_FEATURES,
    DirectionVerdict,
    ModelRef,
    Sketch,
    TransferStats,
    sigmoid,
)
from .utils import atomic_write_json, get_logger, role_of

LOG = get_logger(__name__)

#: Wire-format version stamped into every saved :class:`DirectionModel`.
DIRECTION_MODEL_VERSION: str = "stemma-direction-v1"

#: Contract defaults for the work cap: ``n_tensors`` shared tensors, each read
#: at ``max_rows`` rows through :mod:`stemma.remote_loader`.
DEFAULT_MAX_ROWS: int = 1024
DEFAULT_N_TENSORS: int = 6

#: Key under which the outgroup-rooting term appears in ``contributions`` (and,
#: optionally, in the mapping handed to :meth:`DirectionModel.llr`).  It is
#: deliberately *not* a member of ``DIRECTION_FEATURES``, which is frozen.
OUTGROUP_KEY: str = "outgroup_rooting"

#: Value the raw outgroup gap takes for a *clean* sibling rooting, derived
#: geometrically rather than tuned.  If a parent P has two descendants B and C
#: whose task vectors have comparable magnitude ``s`` and are near-orthogonal
#: (independent fine-tuning runs), then
#:
#:     d(P, C) = s          (one branch)
#:     d(B, C) = sqrt(2)*s  (two orthogonal branches)
#:
#: so the normalised gap ``(d(B,C) - d(P,C)) / (d(B,C) + d(P,C))`` equals
#: ``(sqrt(2) - 1) / (sqrt(2) + 1) = 3 - 2*sqrt(2) ~= 0.1716``.  Measured on the
#: test fixtures the raw statistic came out at 0.17034, i.e. 99.3% of this
#: value, confirming the derivation.  Dividing by it puts "textbook clean
#: rooting" at exactly 1.0, which is what makes :attr:`
#: DirectionModel.outgroup_weight` an interpretable log-odds-per-clean-rooting
#: rather than an arbitrary scale factor.
CANONICAL_OUTGROUP_GAP: float = 3.0 - 2.0 * math.sqrt(2.0)

#: Ceiling applied to :func:`relatedness_score` when the two models share no
#: comparable tensor, so no raw-coordinate evidence exists.  Deliberately below
#: the 0.6 relatedness threshold used by :func:`stemma.phylogeny.build_phylogeny`
#: so that "we cannot substantiate lineage from the weights" reads as an
#: abstention instead of a positive claim.  See the long comment in
#: :func:`relatedness_score` for the measurements behind the number.
NO_SHARED_TENSOR_CEILING: float = 0.45

#: Cap on the normalised outgroup statistic.  A gap far above the canonical
#: value means the "outgroup" is not a sibling at all (e.g. it is a descendant
#: of B, or unrelated), so we refuse to let it dominate the verdict.
OUTGROUP_CLIP: float = 3.0

# --- family (a): delta spectrum ------------------------------------------- #
#: Rank of the right-singular subspace used by :func:`subspace_energy`.
SUBSPACE_K: int = 48
#: Both dimensions are subsampled to this before any eigendecomposition; the
#: same row/column subset is used for A, B and Delta so the comparison is exact.
_SPECTRAL_MAX_DIM: int = 768
#: Scale applied inside ``tanh`` for ``norm_growth_asym``; measured log-norm
#: growth for a real base->instruct pair is ~0.011, which maps to ~0.11.
_NORM_GROWTH_SCALE: float = 10.0

# --- family (b): orphan embeddings ---------------------------------------- #
#: Row-count scale inside ``tanh`` for ``vocab_delta``.  A 256-row vocabulary
#: extension saturates to 0.96; an 8-row padding artefact stays at 0.06.
_VOCAB_SCALE: float = 128.0
#: Minimum number of orphan rows before the "untrained" statistics are trusted.
_MIN_ORPHAN_ROWS: int = 8
#: Orphan rows are capped at this many for the norm-distribution statistics.
_MAX_ORPHAN_ROWS: int = 4096

# --- family (c): quantisation / pruning scars ------------------------------ #
#: Rows/columns sampled per tensor for the lattice fit (it is O(rows * cols)).
_LATTICE_MAX_ROWS: int = 96
_LATTICE_MAX_COLS: int = 512
#: A value counts as "on the lattice" when ``|v/s - round(v/s)| <= _LATTICE_TOL``.
#: Expressing the tolerance *in units of the step* is what stops a vanishingly
#: small candidate step from trivially explaining any float row.
_LATTICE_TOL: float = 1e-3
#: Candidate steps must imply between 3 and 4096 levels across the row's range.
_LATTICE_MIN_LEVELS: float = 3.0
_LATTICE_MAX_LEVELS: float = 4096.0
#: Zero fraction (on either side) below which the zero-structure features are
#: gated off: three stray zeros must not manufacture a confident verdict.
_ZERO_GATE: float = 0.01
#: Bit widths per safetensors dtype, for ``dtype_precision_asym``.
_DTYPE_BITS: Dict[str, float] = {
    "F64": 64.0, "F32": 32.0, "F16": 16.0, "BF16": 16.0,
    "F8_E4M3": 8.0, "F8_E4M3FN": 8.0, "F8_E4M3FNUZ": 8.0,
    "F8_E5M2": 8.0, "F8_E5M2FNUZ": 8.0,
    "I64": 64.0, "U64": 64.0, "I32": 32.0, "U32": 32.0,
    "I16": 16.0, "U16": 16.0, "I8": 8.0, "U8": 8.0, "BOOL": 1.0,
}

# --- family (d): fossils --------------------------------------------------- #
#: A row is "dead" when its L2 norm is below this absolute floor or this
#: fraction of the tensor's median row norm.
_DEAD_ABS: float = 1e-6
_DEAD_REL: float = 1e-3
#: A row is an "outlier channel" when its norm exceeds this multiple of the median.
_OUTLIER_MULT: float = 8.0
_DEAD_GATE: float = 0.002
_OUTLIER_GATE: float = 0.005
#: Tensor-name fragments that mark an *auxiliary* tensor a quantiser adds.
_QUANT_NAME_HINTS: Tuple[str, ...] = (
    "scale", "zero_point", "qzeros", "qweight", "absmax", "g_idx", "quant", "zeros",
)

#: Keyword arguments forwarded to :func:`stemma.remote_loader.open_model`.
#: Everything else in ``**loader_kw`` is ignored, so a caller (the phylogeny
#: builder) may pass through unrelated knobs without a TypeError.
_LOADER_KW: frozenset[str] = frozenset({"revision", "token", "cache_dir", "session"})

_EPS: float = 1e-12


# --------------------------------------------------------------------------- #
# Anti-symmetry primitives
# --------------------------------------------------------------------------- #


def _antisym(h_a: float, h_b: float, *, scale: Optional[float] = None) -> float:
    """Combine a statistic measured on each side into an anti-symmetric feature.

    Claim: direction -- this three-line helper *is* the anti-symmetry guarantee.
    Every feature in :func:`direction_features` is routed through it, so
    ``f(b, a) == -f(a, b)`` is true by construction rather than by test.

    ``h_a`` is the statistic oriented so that **larger means more ancestral**
    (either a per-side ``g(A)`` or a per-ordered-pair ``h(A, B)``); ``h_b`` is
    the mirror-image measurement.  With ``scale=None`` the raw difference is
    returned; otherwise the difference is divided by the *symmetric* denominator
    ``|h_a| + |h_b| + scale``, which keeps the feature in ``[-1, 1]`` while the
    ``scale`` floor stops two near-zero measurements from saturating it.

    Exactness: IEEE-754 makes ``h_b - h_a`` the exact negation of ``h_a - h_b``
    and makes the denominator commutative, so the sign flip is bit-for-bit.
    """
    x = float(h_a)
    y = float(h_b)
    if not (math.isfinite(x) and math.isfinite(y)):
        return 0.0
    num = x - y
    if scale is None:
        return float(num)
    den = abs(x) + abs(y) + float(scale) + _EPS
    return float(num / den)


def _antisym_child(c_a: float, c_b: float, *, scale: Optional[float] = None) -> float:
    """Anti-symmetric feature from a *child-likeness* score measured per side.

    Claim: direction -- most of the strong evidence is a scar, i.e. a property
    the **derived** model has and its ancestor does not, so it is natural to
    measure "how much does this side look like the downstream model" and then
    flip the sign: if B looks more derived, A is the parent.

    Equivalent to ``_antisym(-c_a, -c_b)``; written as its own helper only so
    the call sites read in the direction the statistic was defined.
    """
    return _antisym(c_b, c_a, scale=scale)


def _odd_tanh(x: float) -> float:
    """``tanh`` forced to be *exactly* odd, for squashing an unbounded feature.

    Claim: direction -- libm is not contractually odd-symmetric, and a 1-ulp
    asymmetry in a squashing function would be the one place the anti-symmetry
    guarantee could leak; computing ``sign(x) * tanh(|x|)`` removes the risk.
    """
    v = float(x)
    if not math.isfinite(v):
        return 0.0
    return math.copysign(math.tanh(abs(v)), v)


def _finite(x: Any, default: float = 0.0) -> float:
    """Coerce anything to a finite float, falling back to ``default``.

    Claim: infra -- a nan anywhere in the feature vector would silently turn a
    verdict into "unknown"; we would rather record a zero and say why.
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float(default)
    return v if math.isfinite(v) else float(default)


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #


def _flatten2d(W: np.ndarray) -> np.ndarray:
    """View a tensor as a finite 2-D float32 matrix (trailing axes -> columns).

    Claim: infra -- conv/fused checkpoints store 3-D and 4-D tensors; folding
    the trailing axes keeps them comparable with plain linear layers.
    """
    X = np.asarray(W)
    if X.ndim == 0:
        X = X.reshape(1, 1)
    elif X.ndim == 1:
        X = X.reshape(-1, 1)
    elif X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    X = np.asarray(X, dtype=np.float32)
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def _even_indices(n: int, cap: int) -> np.ndarray:
    """Deterministically pick at most ``cap`` evenly spaced indices from ``n``.

    Claim: low-transfer -- the same rule the loader uses for rows is reused for
    the (free) in-memory subsampling, so A, B and Delta are always compared on
    identical coordinates and no randomness enters the comparison.
    """
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    if cap is None or n <= int(cap):
        return np.arange(n, dtype=np.int64)
    pick = np.linspace(0.0, float(n - 1), int(cap))
    return np.unique(np.rint(pick).astype(np.int64))


def _row_norms(X: np.ndarray) -> np.ndarray:
    """L2 norm of every row of a 2-D matrix, in float64.

    Claim: infra -- row norms are the substrate of the fossil family (dead rows,
    outlier channels) and of the orphan-embedding test.
    """
    A = np.asarray(X, dtype=np.float64)
    if A.ndim != 2 or A.size == 0:
        return np.zeros(0, dtype=np.float64)
    return np.sqrt(np.einsum("ij,ij->i", A, A))


def _dead_mask(X: np.ndarray) -> np.ndarray:
    """Boolean mask of rows that carry (essentially) no signal.

    Claim: direction -- a dead unit stays dead in every descendant, so the set
    of dead rows can only grow downstream; that monotonicity is a fossil and
    therefore an arrow.
    """
    r = _row_norms(X)
    if r.size == 0:
        return np.zeros(0, dtype=bool)
    med = float(np.median(r))
    thr = max(_DEAD_ABS, _DEAD_REL * med)
    return r <= thr


def _outlier_mask(X: np.ndarray) -> np.ndarray:
    """Boolean mask of rows whose norm exceeds ``_OUTLIER_MULT`` x the median.

    Claim: direction -- massive-activation / outlier channels are created during
    pre-training and then persist (and sharpen) through every fine-tune, so the
    outlier set is another monotone fossil.
    """
    r = _row_norms(X)
    if r.size == 0:
        return np.zeros(0, dtype=bool)
    med = float(np.median(r))
    if med <= 0.0:
        return np.zeros(r.size, dtype=bool)
    return r > _OUTLIER_MULT * med


def _containment(mask_x: np.ndarray, mask_y: np.ndarray) -> float:
    """Fraction of ``X``'s flagged positions that are also flagged in ``Y``.

    Claim: direction -- this is the ordered-pair statistic ``h(X, Y)`` behind
    every "superset" feature.  ``h(A, B) == 1`` with ``h(B, A) < 1`` says B's
    flagged set strictly contains A's, i.e. B is downstream.  An empty set is
    contained in everything, so ``h`` is 1.0 when X flags nothing -- callers
    gate the feature on the pair having enough flagged positions for that
    vacuous truth to mean something.
    """
    mx = np.asarray(mask_x, dtype=bool).ravel()
    my = np.asarray(mask_y, dtype=bool).ravel()
    n = min(mx.size, my.size)
    if n == 0:
        return 1.0
    mx, my = mx[:n], my[:n]
    nx = int(np.count_nonzero(mx))
    if nx == 0:
        return 1.0  # vacuous: the empty set is a subset of anything
    return float(np.count_nonzero(mx & my) / nx)


def _ks_one_sided(sample: np.ndarray, reference: np.ndarray) -> float:
    """One-sided two-sample Kolmogorov-Smirnov statistic ``max(F_s - F_r)``.

    Claim: direction -- freshly initialised embedding rows have a norm
    distribution that is tightly concentrated where trained rows are
    heavy-tailed; the KS gap between "rows only in the wider model" and "rows
    both models share" is what turns that into a number.

    Implemented with ``searchsorted`` so no scipy import is needed.
    """
    s = np.sort(np.asarray(sample, dtype=np.float64).ravel())
    r = np.sort(np.asarray(reference, dtype=np.float64).ravel())
    if s.size == 0 or r.size == 0:
        return 0.0
    grid = np.concatenate([s, r])
    fs = np.searchsorted(s, grid, side="right") / float(s.size)
    fr = np.searchsorted(r, grid, side="right") / float(r.size)
    return float(np.clip(np.max(np.abs(fs - fr)), 0.0, 1.0))


def _gram_eig(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Descending eigenvalues (== squared singular values) and right vectors.

    Claim: infra -- one symmetric eigendecomposition of ``X^T X`` yields
    everything family (a) needs (effective rank, spectral mass, and the top-k
    right-singular subspace used by :func:`subspace_energy`), so the delta
    spectrum costs one decomposition per matrix rather than three.
    """
    A = np.asarray(X, dtype=np.float64)
    if A.ndim != 2 or A.size == 0 or A.shape[1] == 0:
        return np.zeros(1, dtype=np.float64), np.zeros((max(A.shape[1:2] or [1]), 1))
    G = A.T @ A
    G = 0.5 * (G + G.T)
    try:
        w, V = np.linalg.eigh(G)
    except np.linalg.LinAlgError:  # pragma: no cover - numerically pathological
        LOG.debug("eigh failed on a %s Gram matrix; falling back to zeros", G.shape)
        return np.zeros(G.shape[0], dtype=np.float64), np.eye(G.shape[0])
    order = np.argsort(w)[::-1]
    return np.clip(w[order], 0.0, None), V[:, order]


def _effective_rank(eigenvalues: np.ndarray) -> float:
    """Entropy-based effective rank ``exp(-sum p_i log p_i)`` with ``p = s^2/sum``.

    Claim: direction -- effective rank is the scale-free way to ask "how many
    directions does this matrix actually use", which is what family (a) compares
    between a delta and the two sides it connects.
    """
    w = np.clip(np.asarray(eigenvalues, dtype=np.float64).ravel(), 0.0, None)
    total = float(w.sum())
    if total <= 0.0:
        return 0.0
    p = w / total
    p = p[p > 0.0]
    if p.size == 0:
        return 0.0
    return float(np.exp(-float(np.sum(p * np.log(p)))))


def subspace_energy(delta: np.ndarray, basis: np.ndarray, k: int = SUBSPACE_K) -> float:
    """Fraction of ``||delta||_F^2`` captured by ``basis``'s top-``k`` right subspace.

    Claim: direction -- family (a)'s headline statistic.  Measured behaviour
    (``docs/FINDINGS.md`` section 3) is the *opposite* of the naive intuition:
    because the child has already absorbed the delta, the child's own top-k
    right-singular subspace aligns marginally better with it, and the effect is
    only ~1e-3..1e-2 wide against a per-tensor spread of the same order.  It is
    a tiebreaker, weighted accordingly, and never a decisive signal.

    ``basis`` may be either a matrix (its Gram is decomposed here) or a
    precomputed ``(n, k)`` orthonormal basis.
    """
    D = np.asarray(delta, dtype=np.float64)
    if D.ndim != 2 or D.size == 0:
        return 0.0
    den = float(np.einsum("ij,ij->", D, D))
    if den <= 0.0:
        return 0.0
    B = np.asarray(basis, dtype=np.float64)
    if B.ndim != 2 or B.size == 0:
        return 0.0
    if B.shape[0] != D.shape[1]:
        _, V = _gram_eig(B)
    else:
        # Already an (n, r) basis if its columns are orthonormal; otherwise treat
        # it as a data matrix with n columns and decompose.
        gram = B.T @ B
        if B.shape[1] <= B.shape[0] and np.allclose(gram, np.eye(B.shape[1]), atol=1e-6):
            V = B
        else:
            _, V = _gram_eig(B)
    kk = int(max(1, min(int(k), V.shape[1])))
    P = D @ V[:, :kk]
    num = float(np.einsum("ij,ij->", P, P))
    return float(np.clip(num / den, 0.0, 1.0))


def lattice_fit(
    W: np.ndarray, *, max_rows: int = _LATTICE_MAX_ROWS, max_cols: int = _LATTICE_MAX_COLS
) -> Tuple[float, float, float]:
    """Estimate how strongly a matrix's values sit on a per-row value lattice.

    Claim: direction -- quantisation is lossy and irreversible: once weights
    have been snapped to ``k*s`` they never come back off the lattice, so a
    lattice scar can only ever appear *downstream*.  This is the single
    strongest direction signal Stemma has.

    Returns ``(on_lattice_fraction, mean_levels, mean_distinct_ratio)``:

    * candidate steps come from the sorted unique ``|values|`` of each row (its
      smallest non-zero magnitude, the smallest positive gap, and low quantiles
      of the gaps), restricted to steps implying between 3 and 4096 levels
      across the row's range;
    * a value is on the lattice when ``|v/s - round(v/s)| <= 1e-3`` -- the
      tolerance is expressed *in units of the step*, which is what stops a
      vanishingly small ``s`` from explaining an arbitrary float row (a random
      row scores ~2e-3, an int8 round trip scores ~1.0);
    * exact zeros are excluded from the fit, because a zero lies on every
      lattice and would otherwise let a *pruning* scar masquerade as a
      *quantisation* scar.
    """
    X = _flatten2d(W)
    if X.ndim != 2 or X.size == 0:
        return 0.0, 0.0, 0.0
    ridx = _even_indices(X.shape[0], max_rows)
    cidx = _even_indices(X.shape[1], max_cols)
    if ridx.size == 0 or cidx.size == 0:
        return 0.0, 0.0, 0.0
    block = np.asarray(X[np.ix_(ridx, cidx)], dtype=np.float64)

    fracs: List[float] = []
    levels: List[float] = []
    distinct: List[float] = []
    for row in block:
        v = row[row != 0.0]
        if v.size < 8:
            continue
        a = np.abs(v)
        amax = float(a.max())
        if not math.isfinite(amax) or amax <= 0.0:
            continue
        u = np.unique(a)
        distinct.append(float(u.size) / float(v.size))
        if u.size < 3:
            # A row with fewer than three distinct magnitudes is degenerate: it
            # sits on a trivial lattice regardless of provenance.
            continue
        gaps = np.diff(u)
        gaps = gaps[gaps > 0.0]
        cands: List[float] = [float(u[0])]
        if gaps.size:
            cands.append(float(gaps.min()))
            cands.append(float(np.quantile(gaps, 0.05)))
            cands.append(float(np.quantile(gaps, 0.25)))
            cands.append(float(np.median(gaps)))
        lo = amax / _LATTICE_MAX_LEVELS
        hi = amax / _LATTICE_MIN_LEVELS
        cands = sorted({round(s, 18) for s in cands if lo <= s <= hi})
        if not cands:
            fracs.append(0.0)
            continue
        best = 0.0
        best_s = 0.0
        for s in cands:
            q = v / s
            resid = np.abs(q - np.rint(q))
            frac = float(np.count_nonzero(resid <= _LATTICE_TOL)) / float(v.size)
            if frac > best:
                best, best_s = frac, s
        fracs.append(best)
        if best_s > 0.0:
            levels.append(amax / best_s)

    on_lattice = float(np.mean(fracs)) if fracs else 0.0
    mean_levels = float(np.mean(levels)) if levels else 0.0
    mean_distinct = float(np.mean(distinct)) if distinct else 0.0
    return on_lattice, mean_levels, mean_distinct


def _mean_dtype_bits(index: Mapping[str, Any]) -> float:
    """Parameter-weighted mean stored bit width over a checkpoint's tensors.

    Claim: direction -- precision only ever goes down.  A BF16 copy of an F32
    model can exist; the reverse cannot recover the discarded mantissa bits, so
    the lower-precision side of a related pair is the descendant.
    """
    total_w = 0.0
    total_b = 0.0
    for name, meta in index.items():
        shape = tuple(getattr(meta, "shape", ()) or ())
        if len(shape) < 2:
            continue
        numel = float(getattr(meta, "numel", 0) or 0)
        if numel <= 0:
            continue
        bits = _DTYPE_BITS.get(str(getattr(meta, "dtype", "F32")).upper(), 16.0)
        total_w += numel
        total_b += numel * bits
    if total_w <= 0.0:
        return 0.0
    return float(total_b / total_w)


def _extra_tensor_score(only_mine: Sequence[str], n_mine: int) -> float:
    """How much "extra apparatus" one side carries that the other lacks.

    Claim: direction -- a quantiser bolts on ``*_scale`` / ``*_zero_point`` /
    ``qweight`` tensors; a merge or fine-tune does not remove tensors.  Extra
    auxiliary tensors therefore point downstream, and are weighted double
    against generic extras.
    """
    if n_mine <= 0:
        return 0.0
    score = 0.0
    for name in only_mine:
        low = str(name).lower()
        score += 2.0 if any(h in low for h in _QUANT_NAME_HINTS) else 1.0
    return float(min(1.0, score / float(n_mine)))


# --------------------------------------------------------------------------- #
# Evidence container
# --------------------------------------------------------------------------- #


@dataclass
class PairEvidence:
    """Raw, mirrored measurements for one ordered pair, before any combination.

    Claim: direction -- the layout is the anti-symmetry contract made explicit:
    ``side_a`` / ``side_b`` hold the *same* per-side statistics ``g(X)`` and
    ``pair_ab`` / ``pair_ba`` hold the *same* ordered-pair statistics evaluated
    both ways, so :func:`direction_features` only ever has to subtract.
    Everything here is JSON-serialisable via :meth:`to_json`.

    ``shared`` carries the genuinely symmetric context (tensor names, sampled
    row counts, bit-identical fraction, relative delta norm) that the evidence
    strings and :func:`relatedness_score` quote but which can never contribute a
    direction.
    """

    a: ModelRef
    b: ModelRef
    side_a: Dict[str, float] = field(default_factory=dict)
    side_b: Dict[str, float] = field(default_factory=dict)
    pair_ab: Dict[str, float] = field(default_factory=dict)
    pair_ba: Dict[str, float] = field(default_factory=dict)
    orphan_a: Dict[str, float] = field(default_factory=dict)
    orphan_b: Dict[str, float] = field(default_factory=dict)
    shared: Dict[str, Any] = field(default_factory=dict)
    outgroup: Dict[str, Any] = field(default_factory=dict)
    stats: Optional[TransferStats] = None

    @property
    def outgroup_stat(self) -> float:
        """Signed outgroup-rooting statistic; ``0.0`` when no outgroup was used.

        Claim: direction -- positive means A sits closer to every outgroup, and
        therefore closer to the root.  Exactly zero when the term is inactive,
        so the additive llr term vanishes rather than guessing.

        The raw gap is divided by :data:`CANONICAL_OUTGROUP_GAP` so that a clean
        sibling rooting scores 1.0 regardless of how far the branches ran; see
        that constant for the derivation.  The result is clipped to
        +/-:data:`OUTGROUP_CLIP` so a mis-specified "outgroup" (one that is
        actually a descendant, or unrelated) cannot dominate the verdict.
        """
        raw = _finite(self.outgroup.get("stat", 0.0))
        norm = raw / CANONICAL_OUTGROUP_GAP
        return float(np.clip(norm, -OUTGROUP_CLIP, OUTGROUP_CLIP))

    @property
    def outgroup_stat_raw(self) -> float:
        """Un-normalised outgroup gap, as measured. Useful for the benchmark.

        Claim: direction -- reported separately so the evaluation can show the
        measured geometry rather than only the calibrated log-odds term.
        """
        return _finite(self.outgroup.get("stat", 0.0))

    @property
    def has_outgroup(self) -> bool:
        """Whether at least one usable outgroup comparison was made.

        Claim: direction -- the UI and the benchmark must be able to say
        "rooted with an outgroup" versus "two-model evidence only".
        """
        return int(self.outgroup.get("n_terms", 0) or 0) > 0

    def to_json(self) -> Dict[str, Any]:
        """JSON-serialisable dict of the whole evidence record.

        Claim: infra -- the Gradio Space and the benchmark both persist raw
        evidence so a verdict can be re-derived without re-reading weights.
        """
        out: Dict[str, Any] = {
            "a": str(self.a),
            "b": str(self.b),
            "side_a": {k: _finite(v) for k, v in self.side_a.items()},
            "side_b": {k: _finite(v) for k, v in self.side_b.items()},
            "pair_ab": {k: _finite(v) for k, v in self.pair_ab.items()},
            "pair_ba": {k: _finite(v) for k, v in self.pair_ba.items()},
            "orphan_a": {k: _finite(v) for k, v in self.orphan_a.items()},
            "orphan_b": {k: _finite(v) for k, v in self.orphan_b.items()},
            "shared": json.loads(json.dumps(self.shared, default=str)),
            "outgroup": json.loads(json.dumps(self.outgroup, default=str)),
            "stats": asdict(self.stats) if self.stats is not None else None,
        }
        return out


# --------------------------------------------------------------------------- #
# Loader plumbing
# --------------------------------------------------------------------------- #


def _open_source(ref: ModelRef, **loader_kw: Any) -> Any:
    """Open a checkpoint through the Range-reading loader.

    Claim: low-transfer -- direction never opens a file any other way, so its
    byte budget is accounted for by :class:`stemma.types.TransferStats` like
    every other stage.
    """
    from .remote_loader import open_model  # lazy: keeps import time network-free

    kw = {k: v for k, v in loader_kw.items() if k in _LOADER_KW}
    dropped = sorted(set(loader_kw) - set(kw))
    if dropped:
        LOG.debug("ignoring non-loader kwargs %s for %s", dropped, ref)
    return open_model(ref, **kw)


def _accumulate(total: TransferStats, src: Any) -> TransferStats:
    """Fold one source's transfer accounting into a running total.

    Claim: low-transfer -- a direction verdict must report every byte it cost,
    including the outgroup models it consulted.
    """
    st = getattr(src, "stats", None)
    if not isinstance(st, TransferStats):
        return total
    return total.add(st)


def _embed_tensor_name(index: Mapping[str, Any]) -> Optional[str]:
    """Name of the largest 2-D embedding tensor, or ``None``.

    Claim: direction -- family (b) lives entirely in this one tensor: the rows
    a wider vocabulary added are the orphans whose statistics give the arrow.
    """
    best: Optional[str] = None
    best_numel = -1
    for name in sorted(index):
        meta = index[name]
        if role_of(name) != "embed":
            continue
        if len(tuple(getattr(meta, "shape", ()) or ())) != 2:
            continue
        numel = int(getattr(meta, "numel", 0) or 0)
        if numel > best_numel:
            best, best_numel = name, numel
    return best


def _vocab_rows(index: Mapping[str, Any]) -> int:
    """Vocabulary row count from the embedding (or the lm_head) tensor shape.

    Claim: low-transfer -- read from the safetensors header, so the vocabulary
    comparison costs no payload bytes at all.
    """
    best = 0
    for name in sorted(index):
        role = role_of(name)
        if role not in ("embed", "lm_head"):
            continue
        shape = tuple(getattr(index[name], "shape", ()) or ())
        if len(shape) == 2:
            best = max(best, int(shape[0]))
    return int(best)


def _select_shared_tensors(
    idx_a: Mapping[str, Any], idx_b: Mapping[str, Any], n_tensors: int, *, prefer: Sequence[str] = ()
) -> List[str]:
    """Pick up to ``n_tensors`` comparable 2-D tensors present in both models.

    Claim: low-transfer -- this is the work cap.  Selection depends only on the
    *set* of shared names and their shapes, never on argument order, so
    ``select(A, B) == select(B, A)`` and the anti-symmetry survives; role
    diversity is enforced so a verdict is not decided by six slices of the same
    MLP.
    """
    prefer_set = {str(p) for p in prefer}
    cands: List[Tuple[Tuple[int, int, str], str, str]] = []
    for name in sorted(set(idx_a) & set(idx_b)):
        sa = tuple(getattr(idx_a[name], "shape", ()) or ())
        sb = tuple(getattr(idx_b[name], "shape", ()) or ())
        if len(sa) < 2 or len(sb) < 2 or sa[1:] != sb[1:]:
            continue
        rows = min(int(sa[0]), int(sb[0]))
        cols = 1
        for d in sa[1:]:
            cols *= int(d)
        if rows < 2 or cols < 2:
            continue
        role = role_of(name) or "other"
        key = (0 if name in prefer_set else 1, -(rows * cols), name)
        cands.append((key, name, role))
    cands.sort(key=lambda t: t[0])

    n_tensors = int(max(1, n_tensors))
    per_role_cap = max(1, math.ceil(n_tensors / 3.0))
    chosen: List[str] = []
    used: Dict[str, int] = {}
    for _, name, role in cands:
        if len(chosen) >= n_tensors:
            break
        if used.get(role, 0) >= per_role_cap:
            continue
        chosen.append(name)
        used[role] = used.get(role, 0) + 1
    if len(chosen) < n_tensors:  # relax the diversity cap to fill the budget
        for _, name, _role in cands:
            if len(chosen) >= n_tensors:
                break
            if name not in chosen:
                chosen.append(name)
    return chosen


# --------------------------------------------------------------------------- #
# Evidence collection
# --------------------------------------------------------------------------- #


def _orphan_stats(orphan: np.ndarray, shared_norms: np.ndarray) -> Dict[str, float]:
    """Score how "untrained" the rows only one model has actually look.

    Claim: direction -- vocabularies only ever grow, and the rows an extension
    adds start life as i.i.d. draws from a single init scale.  That is visible
    without any tokenizer: their norms concentrate at ``sigma*sqrt(d)`` with
    coefficient of variation ``1/sqrt(2d)``, where trained rows are heavy-tailed.

    Returns ``n_rows``, ``untrained_frac``, ``ks``, ``dup_frac``, ``zero_frac``,
    ``cv`` and a combined ``score`` in ``[0, 1]``.
    """
    out = {
        "n_rows": 0.0, "untrained_frac": 0.0, "ks": 0.0, "dup_frac": 0.0,
        "zero_frac": 0.0, "cv": 0.0, "score": 0.0,
    }
    X = _flatten2d(orphan)
    if X.ndim != 2 or X.shape[0] == 0:
        return out
    n, d = X.shape
    out["n_rows"] = float(n)
    norms = _row_norms(X)
    out["zero_frac"] = float(np.count_nonzero(norms <= _DEAD_ABS)) / float(n)

    # Exactly duplicated rows (classic padding artefact) count as untrained.
    try:
        uniq = np.unique(np.ascontiguousarray(X), axis=0)
        out["dup_frac"] = float(max(0, n - uniq.shape[0])) / float(n)
    except Exception:  # pragma: no cover - np.unique(axis=) on odd dtypes
        out["dup_frac"] = 0.0

    med = float(np.median(norms))
    mean = float(np.mean(norms))
    out["cv"] = float(np.std(norms) / mean) if mean > 0.0 else 0.0
    if n >= _MIN_ORPHAN_ROWS and med > 0.0:
        # i.i.d. rows of width d concentrate with cv = 1/sqrt(2d); allow 3x that.
        band = 3.0 * med / math.sqrt(max(2.0 * float(d), 2.0))
        tight = np.abs(norms - med) <= band
        untrained = tight | (norms <= _DEAD_ABS)
        out["untrained_frac"] = float(np.count_nonzero(untrained)) / float(n)
    if shared_norms is not None and np.size(shared_norms) > 0:
        out["ks"] = _ks_one_sided(norms, shared_norms)

    if n >= _MIN_ORPHAN_ROWS:
        blended = 0.5 * out["untrained_frac"] + 0.5 * out["ks"]
        out["score"] = float(np.clip(max(blended, out["zero_frac"], out["dup_frac"]), 0.0, 1.0))
    return out


def _empty_side() -> Dict[str, float]:
    """Zero-filled per-side statistics dict (all keys present).

    Claim: infra -- both sides must always carry the same keys or the
    subtraction in :func:`direction_features` would silently skip a feature.
    """
    return {
        "lattice_frac": 0.0, "lattice_levels": 0.0, "distinct_ratio": 0.0,
        "zero_frac": 0.0, "dead_frac": 0.0, "outlier_frac": 0.0, "outlier_sharpness": 0.0,
        "subspace_energy": 0.0, "erank": 0.0, "spectral_mass": 0.0, "log_norm": 0.0,
        "bits": 0.0, "vocab_rows": 0.0, "extra_tensor_score": 0.0, "n_tensors_total": 0.0,
    }


def collect_pair_evidence(
    a: ModelRef,
    b: ModelRef,
    *,
    sa: Optional[Sketch] = None,
    sb: Optional[Sketch] = None,
    max_rows: int = DEFAULT_MAX_ROWS,
    n_tensors: int = DEFAULT_N_TENSORS,
    seed: int = 0,
    outgroup: Optional[Sequence[ModelRef]] = None,
    source_a: Any = None,
    source_b: Any = None,
    **loader_kw: Any,
) -> PairEvidence:
    """Range-read a handful of shared tensors and measure every direction family.

    Claim: direction -- this is where all five evidence families are actually
    measured: (a) delta spectrum, (b) orphan embeddings, (c) quantisation and
    pruning scars, (d) fossils, (e) outgroup rooting.  Every measurement is
    taken *twice in mirror image* (once per side, or once per ordered pair in
    each order) so that the features derived from it are anti-symmetric.

    Work is capped at ``n_tensors`` shared tensors x ``max_rows`` rows, chosen
    with the loader's deterministic row rule so A and B are compared
    row-for-row.  ``sa`` / ``sb`` are optional pre-computed sketches, used only
    for metadata; they never change a measurement.

    ``outgroup`` enables family (e): every listed model that shares tensor
    shapes with **both** A and B contributes
    ``(d(B, C) - d(A, C)) / (d(A, C) + d(B, C))`` per shared tensor, averaged.
    Positive means A is closer to every outgroup and therefore closer to the
    root.  Supplying no outgroup leaves the statistic at exactly ``0.0``.
    """
    del seed  # every statistic here is deterministic; kept for signature parity
    max_rows = int(max(1, max_rows))
    n_tensors = int(max(1, n_tensors))

    src_a = source_a if source_a is not None else _open_source(a, **loader_kw)
    src_b = source_b if source_b is not None else _open_source(b, **loader_kw)
    owns_a = source_a is None
    owns_b = source_b is None

    side_a = _empty_side()
    side_b = _empty_side()
    pair_ab: Dict[str, float] = {"zero_containment": 1.0, "dead_containment": 1.0,
                                 "outlier_containment": 1.0}
    pair_ba: Dict[str, float] = dict(pair_ab)
    orphan_a: Dict[str, float] = {}
    orphan_b: Dict[str, float] = {}
    shared: Dict[str, Any] = {}
    outgroup_info: Dict[str, Any] = {"models": [], "n_terms": 0, "stat": 0.0, "per_model": {}}
    stats = TransferStats()

    try:
        idx_a = src_a.index()
        idx_b = src_b.index()

        # ---- header-only features (cost: no payload bytes) ---------------- #
        side_a["bits"] = _mean_dtype_bits(idx_a)
        side_b["bits"] = _mean_dtype_bits(idx_b)
        side_a["vocab_rows"] = float(_vocab_rows(idx_a))
        side_b["vocab_rows"] = float(_vocab_rows(idx_b))
        only_a = sorted(set(idx_a) - set(idx_b))
        only_b = sorted(set(idx_b) - set(idx_a))
        side_a["n_tensors_total"] = float(len(idx_a))
        side_b["n_tensors_total"] = float(len(idx_b))
        side_a["extra_tensor_score"] = _extra_tensor_score(only_a, len(idx_a))
        side_b["extra_tensor_score"] = _extra_tensor_score(only_b, len(idx_b))
        shared["only_a"] = only_a[:16]
        shared["only_b"] = only_b[:16]
        shared["n_only_a"] = len(only_a)
        shared["n_only_b"] = len(only_b)
        shared["dtypes_a"] = sorted({str(getattr(m, "dtype", "?")) for m in idx_a.values()})
        shared["dtypes_b"] = sorted({str(getattr(m, "dtype", "?")) for m in idx_b.values()})

        # ---- tensor selection --------------------------------------------- #
        prefer: List[str] = []
        emb_a = _embed_tensor_name(idx_a)
        emb_b = _embed_tensor_name(idx_b)
        if emb_a and emb_a == emb_b:
            prefer.append(emb_a)
        names = _select_shared_tensors(idx_a, idx_b, n_tensors, prefer=prefer)
        shared["tensors"] = list(names)
        shared["n_shared_tensors"] = len(names)
        shared["max_rows"] = int(max_rows)

        from .remote_loader import select_rows  # lazy import, no network

        blocks: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        rows_used: Dict[str, np.ndarray] = {}

        # per-tensor accumulators, parameter-weighted
        w_total = 0.0
        acc: Dict[str, float] = {k: 0.0 for k in (
            "lat_a", "lat_b", "lev_a", "lev_b", "dis_a", "dis_b",
            "zf_a", "zf_b", "df_a", "df_b", "of_a", "of_b", "sh_a", "sh_b",
            "se_a", "se_b", "er_a", "er_b", "er_d", "sm_a", "sm_b",
            "ln_a", "ln_b", "zc_ab", "zc_ba", "dc_ab", "dc_ba", "oc_ab", "oc_ba",
        )}
        n_identical = 0
        n_compared = 0
        delta_rel: List[float] = []

        for name in names:
            meta_a, meta_b = idx_a[name], idx_b[name]
            n_rows = min(int(meta_a.shape[0]), int(meta_b.shape[0]))
            rows = select_rows(n_rows, max_rows)
            if rows.size == 0:
                continue
            try:
                A = _flatten2d(src_a.get_tensor_rows(name, rows, dtype=np.float32))
                B = _flatten2d(src_b.get_tensor_rows(name, rows, dtype=np.float32))
            except Exception as exc:
                LOG.warning("skipping tensor %s (%s)", name, exc)
                continue
            if A.shape != B.shape or A.size == 0:
                continue
            blocks[name] = (A, B)
            rows_used[name] = rows
            n_compared += 1
            weight = float(A.size)
            w_total += weight

            if np.array_equal(A, B):
                n_identical += 1

            # --- (c) quantisation lattice ---------------------------------- #
            la, lva, da = lattice_fit(A)
            lb, lvb, db = lattice_fit(B)
            acc["lat_a"] += weight * la
            acc["lat_b"] += weight * lb
            acc["lev_a"] += weight * lva
            acc["lev_b"] += weight * lvb
            acc["dis_a"] += weight * da
            acc["dis_b"] += weight * db

            # --- (c) pruning zeros ----------------------------------------- #
            za = A == 0.0
            zb = B == 0.0
            acc["zf_a"] += weight * float(np.count_nonzero(za)) / weight
            acc["zf_b"] += weight * float(np.count_nonzero(zb)) / weight
            acc["zc_ab"] += weight * _containment(za, zb)
            acc["zc_ba"] += weight * _containment(zb, za)

            # --- (d) fossils ------------------------------------------------ #
            da_mask, db_mask = _dead_mask(A), _dead_mask(B)
            oa_mask, ob_mask = _outlier_mask(A), _outlier_mask(B)
            n_r = float(max(1, A.shape[0]))
            acc["df_a"] += weight * float(np.count_nonzero(da_mask)) / n_r
            acc["df_b"] += weight * float(np.count_nonzero(db_mask)) / n_r
            acc["of_a"] += weight * float(np.count_nonzero(oa_mask)) / n_r
            acc["of_b"] += weight * float(np.count_nonzero(ob_mask)) / n_r
            acc["dc_ab"] += weight * _containment(da_mask, db_mask)
            acc["dc_ba"] += weight * _containment(db_mask, da_mask)
            acc["oc_ab"] += weight * _containment(oa_mask, ob_mask)
            acc["oc_ba"] += weight * _containment(ob_mask, oa_mask)
            acc["sh_a"] += weight * _sharpness(A, oa_mask)
            acc["sh_b"] += weight * _sharpness(B, ob_mask)

            # --- (a) delta spectrum ----------------------------------------- #
            ridx = _even_indices(A.shape[0], _SPECTRAL_MAX_DIM)
            cidx = _even_indices(A.shape[1], _SPECTRAL_MAX_DIM)
            As = np.asarray(A[np.ix_(ridx, cidx)], dtype=np.float64)
            Bs = np.asarray(B[np.ix_(ridx, cidx)], dtype=np.float64)
            Ds = Bs - As
            wa, Va = _gram_eig(As)
            wb, Vb = _gram_eig(Bs)
            wd, _ = _gram_eig(Ds)
            kk = int(max(1, min(SUBSPACE_K, Va.shape[1], Vb.shape[1])))
            acc["se_a"] += weight * subspace_energy(Ds, Va, kk)
            acc["se_b"] += weight * subspace_energy(Ds, Vb, kk)
            short = float(max(1, min(As.shape)))
            acc["er_a"] += weight * (_effective_rank(wa) / short)
            acc["er_b"] += weight * (_effective_rank(wb) / short)
            acc["er_d"] += weight * (_effective_rank(wd) / short)
            acc["sm_a"] += weight * _top_mass(wa, kk)
            acc["sm_b"] += weight * _top_mass(wb, kk)
            fa = float(np.sqrt(np.einsum("ij,ij->", np.asarray(A, dtype=np.float64),
                                         np.asarray(A, dtype=np.float64))))
            fb = float(np.sqrt(np.einsum("ij,ij->", np.asarray(B, dtype=np.float64),
                                         np.asarray(B, dtype=np.float64))))
            fd = float(np.sqrt(np.einsum("ij,ij->", Bs - As, Bs - As)))
            acc["ln_a"] += weight * math.log(max(fa, _EPS))
            acc["ln_b"] += weight * math.log(max(fb, _EPS))
            if max(fa, fb) > 0.0:
                delta_rel.append(fd / (0.5 * (fa + fb)))

        if w_total > 0.0:
            for key in acc:
                acc[key] /= w_total
        side_a.update({
            "lattice_frac": acc["lat_a"], "lattice_levels": acc["lev_a"],
            "distinct_ratio": acc["dis_a"], "zero_frac": acc["zf_a"],
            "dead_frac": acc["df_a"], "outlier_frac": acc["of_a"],
            "outlier_sharpness": acc["sh_a"], "subspace_energy": acc["se_a"],
            "erank": acc["er_a"], "spectral_mass": acc["sm_a"], "log_norm": acc["ln_a"],
        })
        side_b.update({
            "lattice_frac": acc["lat_b"], "lattice_levels": acc["lev_b"],
            "distinct_ratio": acc["dis_b"], "zero_frac": acc["zf_b"],
            "dead_frac": acc["df_b"], "outlier_frac": acc["of_b"],
            "outlier_sharpness": acc["sh_b"], "subspace_energy": acc["se_b"],
            "erank": acc["er_b"], "spectral_mass": acc["sm_b"], "log_norm": acc["ln_b"],
        })
        pair_ab = {"zero_containment": acc["zc_ab"], "dead_containment": acc["dc_ab"],
                   "outlier_containment": acc["oc_ab"]}
        pair_ba = {"zero_containment": acc["zc_ba"], "dead_containment": acc["dc_ba"],
                   "outlier_containment": acc["oc_ba"]}
        shared["erank_delta"] = float(acc["er_d"])
        shared["n_compared"] = int(n_compared)
        shared["identical_frac"] = float(n_identical / n_compared) if n_compared else 0.0
        shared["n_identical"] = int(n_identical)
        shared["rel_delta_norm"] = float(np.mean(delta_rel)) if delta_rel else 0.0

        # ---- (b) orphan embeddings ---------------------------------------- #
        orphan_a, orphan_b = _collect_orphans(
            src_a, src_b, idx_a, idx_b, emb_a, emb_b, blocks, rows_used, max_rows
        )

        # ---- (e) outgroup rooting ------------------------------------------ #
        if outgroup:
            outgroup_info = _collect_outgroup(
                blocks, rows_used, idx_a, idx_b, outgroup, loader_kw, stats
            )

        stats = _accumulate(stats, src_a)
        stats = _accumulate(stats, src_b)
        try:
            stats.full_size_bytes = int(src_a.total_size()) + int(src_b.total_size())
        except Exception:  # pragma: no cover - loader dependent
            pass
    finally:
        if owns_a:
            try:
                src_a.close()
            except Exception:  # pragma: no cover
                pass
        if owns_b:
            try:
                src_b.close()
            except Exception:  # pragma: no cover
                pass

    if sa is not None:
        shared.setdefault("meta_a", dict(getattr(sa, "meta", {}) or {}).get("architectures", []))
    if sb is not None:
        shared.setdefault("meta_b", dict(getattr(sb, "meta", {}) or {}).get("architectures", []))

    return PairEvidence(
        a=str(a), b=str(b), side_a=side_a, side_b=side_b,
        pair_ab=pair_ab, pair_ba=pair_ba,
        orphan_a=orphan_a, orphan_b=orphan_b,
        shared=shared, outgroup=outgroup_info, stats=stats,
    )


def _sharpness(X: np.ndarray, outliers: np.ndarray) -> float:
    """Mean outlier-row norm divided by the median row norm (0 when none).

    Claim: direction -- outlier channels do not merely persist downstream, they
    *sharpen*; the ratio is the "and sharpen" half of family (d).
    """
    r = _row_norms(X)
    m = np.asarray(outliers, dtype=bool)
    if r.size == 0 or m.size != r.size or not m.any():
        return 0.0
    med = float(np.median(r))
    if med <= 0.0:
        return 0.0
    return float(np.mean(r[m]) / med)


def _top_mass(eigenvalues: np.ndarray, k: int) -> float:
    """Fraction of squared-singular-value mass in the top ``k`` directions.

    Claim: direction -- "spectral mass accumulation": training concentrates
    energy into the dominant directions, so a descendant's top-k share tends to
    be marginally larger.  Family (a), therefore weak by measurement.
    """
    w = np.clip(np.asarray(eigenvalues, dtype=np.float64).ravel(), 0.0, None)
    total = float(w.sum())
    if total <= 0.0:
        return 0.0
    kk = int(max(1, min(int(k), w.size)))
    return float(np.sum(w[:kk]) / total)


def _collect_orphans(
    src_a: Any,
    src_b: Any,
    idx_a: Mapping[str, Any],
    idx_b: Mapping[str, Any],
    emb_a: Optional[str],
    emb_b: Optional[str],
    blocks: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    rows_used: Mapping[str, np.ndarray],
    max_rows: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Measure the rows only the wider-vocabulary side has (family (b)).

    Claim: direction -- vocabulary extension is monotone: a token can be added
    but the rows for it cannot be un-added.  Reading only the *extra* row range
    of the wider model turns that into evidence for a few hundred kilobytes.
    """
    empty = {"n_rows": 0.0, "untrained_frac": 0.0, "ks": 0.0, "dup_frac": 0.0,
             "zero_frac": 0.0, "cv": 0.0, "score": 0.0}
    if not emb_a or not emb_b:
        return dict(empty), dict(empty)
    sa_shape = tuple(getattr(idx_a[emb_a], "shape", ()) or ())
    sb_shape = tuple(getattr(idx_b[emb_b], "shape", ()) or ())
    if len(sa_shape) != 2 or len(sb_shape) != 2 or sa_shape[1] != sb_shape[1]:
        return dict(empty), dict(empty)
    rows_a, rows_b = int(sa_shape[0]), int(sb_shape[0])
    if rows_a == rows_b:
        return dict(empty), dict(empty)

    n_shared = min(rows_a, rows_b)
    wider_is_a = rows_a > rows_b
    src_w = src_a if wider_is_a else src_b
    name_w = emb_a if wider_is_a else emb_b
    n_extra = max(rows_a, rows_b) - n_shared

    # Reference distribution: the shared rows of the *wider* model, reused from
    # the already-read block when the embedding was one of the sampled tensors.
    shared_norms = np.zeros(0, dtype=np.float64)
    if emb_a == emb_b and emb_a in blocks:
        A_blk, B_blk = blocks[emb_a]
        shared_norms = _row_norms(A_blk if wider_is_a else B_blk)
    else:
        try:
            from .remote_loader import select_rows

            rows = select_rows(n_shared, min(max_rows, 2048))
            blk = src_w.get_tensor_rows(name_w, rows, dtype=np.float32)
            shared_norms = _row_norms(_flatten2d(blk))
        except Exception as exc:  # pragma: no cover - loader dependent
            LOG.debug("could not read shared embedding rows: %s", exc)

    extra_idx = np.arange(n_shared, n_shared + n_extra, dtype=np.int64)
    if extra_idx.size > _MAX_ORPHAN_ROWS:
        extra_idx = extra_idx[_even_indices(extra_idx.size, _MAX_ORPHAN_ROWS)]
    try:
        orphan_rows = _flatten2d(src_w.get_tensor_rows(name_w, extra_idx, dtype=np.float32))
    except Exception as exc:
        LOG.warning("could not read orphan embedding rows: %s", exc)
        return dict(empty), dict(empty)

    stats = _orphan_stats(orphan_rows, shared_norms)
    if wider_is_a:
        return stats, dict(empty)
    return dict(empty), stats


def _as_ref_list(refs: Any) -> List[ModelRef]:
    """Normalise an outgroup argument to a list of model references.

    Claim: direction -- ``str`` satisfies ``Sequence[ModelRef]``, so passing a
    single reference instead of a one-element list type-checks and then
    silently iterates *character by character*. That failure is loud but
    absurd: it sent requests to ``huggingface.co/s/``, ``/m/``, ``/o/`` ... one
    per letter of the model name, each 401ing, and the outgroup term quietly
    fell back to zero. Coercing here makes the ergonomic call correct instead
    of subtly wrong.
    """
    if refs is None:
        return []
    if isinstance(refs, (str, os.PathLike)):
        return [str(refs)]
    return [str(r) for r in refs]


def _collect_outgroup(
    blocks: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    rows_used: Mapping[str, np.ndarray],
    idx_a: Mapping[str, Any],
    idx_b: Mapping[str, Any],
    outgroup: Sequence[ModelRef],
    loader_kw: Mapping[str, Any],
    stats: TransferStats,
) -> Dict[str, Any]:
    """Family (e): root the pair against one or more sibling models.

    Claim: direction -- for a scar-free SFT/LoRA edge, families (a)-(d) have
    nothing lossy to grip and Stemma would honestly abstain.  Classical
    phylogenetics solves exactly this with an outgroup: a sibling's distance is
    dominated by the shared *ancestral* component, so whichever of A and B is
    consistently closer to every outgroup sits closer to the root.

    Per outgroup ``C`` and per shared tensor, using the **same sampled rows**,
    the term is ``(d(B, C) - d(A, C)) / (d(A, C) + d(B, C))`` with ``d`` the
    Frobenius distance normalised by ``||C||_F``; the reported statistic is the
    mean over all terms, in ``[-1, 1]``, positive when A is nearer the root.
    """
    outgroup = _as_ref_list(outgroup)
    info: Dict[str, Any] = {"models": [], "n_terms": 0, "stat": 0.0, "per_model": {}}
    if not blocks:
        return info
    terms: List[float] = []
    for ref in outgroup:
        ref = str(ref)
        src_c = None
        per_terms: List[float] = []
        try:
            src_c = _open_source(ref, **dict(loader_kw))
            idx_c = src_c.index()
            for name, (A, B) in blocks.items():
                if name not in idx_c:
                    continue
                shape_c = tuple(getattr(idx_c[name], "shape", ()) or ())
                shape_a = tuple(getattr(idx_a[name], "shape", ()) or ())
                if len(shape_c) < 2 or shape_c[1:] != shape_a[1:]:
                    continue
                rows = np.asarray(rows_used.get(name, np.zeros(0, dtype=np.int64)))
                if rows.size == 0 or int(rows.max()) >= int(shape_c[0]):
                    continue
                C = _flatten2d(src_c.get_tensor_rows(name, rows, dtype=np.float32))
                if C.shape != A.shape:
                    continue
                Cd = np.asarray(C, dtype=np.float64)
                nc = float(np.sqrt(np.einsum("ij,ij->", Cd, Cd)))
                if nc <= 0.0:
                    continue
                da = float(np.linalg.norm(np.asarray(A, dtype=np.float64) - Cd)) / nc
                db = float(np.linalg.norm(np.asarray(B, dtype=np.float64) - Cd)) / nc
                den = da + db
                if den <= 0.0:
                    continue
                per_terms.append((db - da) / den)
            if per_terms:
                info["models"].append(ref)
                info["per_model"][ref] = {
                    "n_tensors": len(per_terms),
                    "stat": float(np.mean(per_terms)),
                }
                terms.extend(per_terms)
        except Exception as exc:
            LOG.warning("outgroup %s unusable: %s", ref, exc)
        finally:
            if src_c is not None:
                try:
                    st = getattr(src_c, "stats", None)
                    if isinstance(st, TransferStats):
                        merged = stats.add(st)
                        stats.bytes_read = merged.bytes_read
                        stats.requests = merged.requests
                        stats.seconds = merged.seconds
                        stats.cache_hits = merged.cache_hits
                    src_c.close()
                except Exception:  # pragma: no cover
                    pass
    if terms:
        info["n_terms"] = len(terms)
        info["stat"] = float(np.clip(np.mean(terms), -1.0, 1.0))
    return info


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #


def direction_features(ev: PairEvidence) -> Dict[str, float]:
    """Turn mirrored evidence into the frozen, anti-symmetric feature vector.

    Claim: direction -- the keys are exactly
    :data:`stemma.types.DIRECTION_FEATURES`, in that order, and every value is
    produced by :func:`_antisym` from a pair of mirror-image measurements.  It
    follows that ``direction_features(collect(b, a))`` is the exact negation of
    ``direction_features(collect(a, b))``, feature by feature, which is what
    makes ``llr(b, a) == -llr(a, b)`` a structural property rather than a test
    that happens to pass.

    Sign convention throughout: **positive means A is the parent of B**.
    """
    sa, sb = dict(ev.side_a), dict(ev.side_b)
    ab, ba = dict(ev.pair_ab), dict(ev.pair_ba)
    oa, ob = dict(ev.orphan_a), dict(ev.orphan_b)
    shared = dict(ev.shared)

    g = lambda d, k: _finite(d.get(k, 0.0))  # noqa: E731 - local shorthand

    # ---- symmetric gates: identical on both sides, so anti-symmetry holds -- #
    zero_gate = min(1.0, max(g(sa, "zero_frac"), g(sb, "zero_frac")) / _ZERO_GATE)
    dead_gate = min(1.0, max(g(sa, "dead_frac"), g(sb, "dead_frac")) / _DEAD_GATE)
    out_gate = min(1.0, max(g(sa, "outlier_frac"), g(sb, "outlier_frac")) / _OUTLIER_GATE)
    tie_gate = 0.5 + 0.5 * _finite(shared.get("identical_frac", 0.0))

    feats: Dict[str, float] = {}

    # --- (b) vocabulary ---------------------------------------------------- #
    feats["vocab_delta"] = _odd_tanh(
        _antisym_child(g(sa, "vocab_rows"), g(sb, "vocab_rows")) / _VOCAB_SCALE
    )
    feats["orphan_asym"] = _antisym_child(g(oa, "score"), g(ob, "score"))

    # --- (c) quantisation / pruning scars ---------------------------------- #
    feats["lattice_asym"] = _antisym_child(g(sa, "lattice_frac"), g(sb, "lattice_frac"))
    feats["zero_asym"] = _antisym_child(g(sa, "zero_frac"), g(sb, "zero_frac"))
    feats["zero_subset_asym"] = zero_gate * _antisym(
        g(ab, "zero_containment"), g(ba, "zero_containment")
    )
    feats["dtype_precision_asym"] = float(
        np.clip(_antisym(g(sa, "bits"), g(sb, "bits")) / 32.0, -1.0, 1.0)
    )

    # --- (a) delta spectrum (weak: see docs/FINDINGS.md sections 2 and 3) --- #
    feats["subspace_energy_asym"] = _antisym_child(
        g(sa, "subspace_energy"), g(sb, "subspace_energy"), scale=0.0
    )
    # erank(delta) cancels in the normalisation, leaving (erank_a - erank_b) /
    # (erank_a + erank_b): positive when B uses fewer directions than A.
    e_d = _finite(shared.get("erank_delta", 0.0))
    r_a = e_d / max(g(sa, "erank"), _EPS)
    r_b = e_d / max(g(sb, "erank"), _EPS)
    feats["delta_rank_asym"] = _antisym_child(r_a, r_b, scale=0.0)
    feats["spectral_growth_asym"] = _antisym_child(
        g(sa, "spectral_mass"), g(sb, "spectral_mass"), scale=0.0
    )
    feats["norm_growth_asym"] = _odd_tanh(
        _NORM_GROWTH_SCALE * _antisym_child(g(sa, "log_norm"), g(sb, "log_norm"))
    )

    # --- (d) fossils -------------------------------------------------------- #
    feats["dead_fossil_asym"] = dead_gate * _antisym(
        g(ab, "dead_containment"), g(ba, "dead_containment")
    )
    feats["outlier_fossil_asym"] = 0.5 * out_gate * _antisym(
        g(ab, "outlier_containment"), g(ba, "outlier_containment")
    ) + 0.5 * _antisym_child(g(sa, "outlier_sharpness"), g(sb, "outlier_sharpness"), scale=0.0)
    feats["exact_tie_asym"] = tie_gate * _antisym_child(
        g(sa, "extra_tensor_score"), g(sb, "extra_tensor_score")
    )

    ordered = {k: float(np.clip(_finite(feats.get(k, 0.0)), -1.0, 1.0)) for k in DIRECTION_FEATURES}
    if set(ordered) != set(DIRECTION_FEATURES):  # pragma: no cover - guarded above
        raise RuntimeError("direction_features must emit exactly DIRECTION_FEATURES")
    return ordered


# --------------------------------------------------------------------------- #
# The combiner
# --------------------------------------------------------------------------- #


@dataclass
class DirectionModel:
    """Linear log-likelihood-ratio combiner over the anti-symmetric features.

    Claim: direction -- the model is deliberately *odd*: no intercept, and a
    standardiser with a **zero mean**.  Under those two constraints
    ``llr(-x) == -llr(x)`` exactly, so the anti-symmetry the features guarantee
    survives the combination step; a non-zero bias or a non-zero ``scaler_mean``
    would make ``estimate_direction(a, b).llr != -estimate_direction(b, a).llr``
    and quietly destroy the headline property.  :meth:`default` and :meth:`fit`
    both force ``bias = 0.0`` and ``scaler_mean = 0``.

    :attr:`outgroup_weight` multiplies the separate family-(e) statistic, which
    has no slot in the frozen ``DIRECTION_FEATURES`` tuple; it only ever
    contributes when an outgroup was supplied (the statistic is exactly ``0.0``
    otherwise).
    """

    weights: np.ndarray
    bias: float = 0.0
    feature_names: Tuple[str, ...] = DIRECTION_FEATURES
    scaler_mean: np.ndarray = field(default_factory=lambda: np.zeros(len(DIRECTION_FEATURES)))
    scaler_scale: np.ndarray = field(default_factory=lambda: np.ones(len(DIRECTION_FEATURES)))
    outgroup_weight: float = 2.0
    version: str = DIRECTION_MODEL_VERSION
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Coerce the arrays to the right length/dtype and warn on a broken prior.

        Claim: direction -- a loaded model with a non-zero bias would silently
        break anti-symmetry, so it is logged loudly rather than tolerated in
        silence.
        """
        n = len(self.feature_names)
        self.weights = np.asarray(self.weights, dtype=np.float64).ravel()
        if self.weights.size != n:
            self.weights = np.resize(self.weights, n).astype(np.float64)
        self.scaler_mean = np.asarray(self.scaler_mean, dtype=np.float64).ravel()
        if self.scaler_mean.size != n:
            self.scaler_mean = np.zeros(n, dtype=np.float64)
        self.scaler_scale = np.asarray(self.scaler_scale, dtype=np.float64).ravel()
        if self.scaler_scale.size != n:
            self.scaler_scale = np.ones(n, dtype=np.float64)
        self.scaler_scale = np.where(
            np.abs(self.scaler_scale) < 1e-12, 1.0, self.scaler_scale
        )
        self.bias = float(self.bias)
        self.outgroup_weight = float(self.outgroup_weight)
        if not self.is_antisymmetric:
            LOG.warning(
                "DirectionModel has bias=%.4g and |scaler_mean|max=%.4g; llr will NOT be "
                "anti-symmetric", self.bias, float(np.max(np.abs(self.scaler_mean)))
            )

    # ------------------------------------------------------------------ core

    @property
    def is_antisymmetric(self) -> bool:
        """True when ``llr(-x) == -llr(x)``, i.e. no intercept and zero mean.

        Claim: direction -- the one invariant the whole module depends on.
        """
        return self.bias == 0.0 and not np.any(np.abs(self.scaler_mean) > 0.0)

    def vectorize(self, feats: Mapping[str, float]) -> np.ndarray:
        """Order a feature mapping into this model's feature vector.

        Claim: infra -- missing keys become 0.0 so a partially measurable pair
        (headers only, no shared tensors) still produces a usable verdict.
        """
        return np.asarray(
            [_finite(feats.get(name, 0.0)) for name in self.feature_names], dtype=np.float64
        )

    def standardize(self, feats: Mapping[str, float]) -> np.ndarray:
        """Return ``z = (x - scaler_mean) / scaler_scale``.

        Claim: direction -- with ``scaler_mean == 0`` this map is odd, which is
        what carries the features' anti-symmetry through to the llr.
        """
        x = self.vectorize(feats)
        return (x - self.scaler_mean) / self.scaler_scale

    def contributions(self, feats: Mapping[str, float]) -> Dict[str, float]:
        """Per-feature signed contribution ``w_i * z_i`` (plus the outgroup term).

        Claim: direction -- the UI's evidence table is exactly this dict; being
        able to say *which* asymmetry produced the arrow is the difference
        between a provenance tool and an oracle.
        """
        z = self.standardize(feats)
        out = {name: float(self.weights[i] * z[i]) for i, name in enumerate(self.feature_names)}
        if OUTGROUP_KEY in feats:
            out[OUTGROUP_KEY] = float(self.outgroup_weight * _finite(feats.get(OUTGROUP_KEY, 0.0)))
        return out

    def llr(self, feats: Mapping[str, float]) -> float:
        """Log-likelihood ratio that A is the parent of B; positive means yes.

        Claim: direction -- this is the number the entire project reports.  It
        is ``w . z + bias`` over the 13 frozen features, plus
        ``outgroup_weight * outgroup_rooting`` when (and only when) the caller
        included that key, which is how family (e) contributes without altering
        the frozen ``DIRECTION_FEATURES`` wire format.
        """
        z = self.standardize(feats)
        total = float(np.dot(self.weights, z)) + float(self.bias)
        if OUTGROUP_KEY in feats:
            total += float(self.outgroup_weight) * _finite(feats.get(OUTGROUP_KEY, 0.0))
        return float(total)

    # ------------------------------------------------------------ (de)serialise

    def to_json(self) -> Dict[str, Any]:
        """Serialise to the single-file JSON wire format.

        Claim: infra -- one small JSON is what gets pushed to the Hub next to
        the Space, so a fitted combiner is reproducible.
        """
        return {
            "version": self.version,
            "feature_names": list(self.feature_names),
            "weights": [float(w) for w in self.weights],
            "bias": float(self.bias),
            "scaler_mean": [float(v) for v in self.scaler_mean],
            "scaler_scale": [float(v) for v in self.scaler_scale],
            "outgroup_weight": float(self.outgroup_weight),
            "meta": self.meta,
        }

    def save(self, path: os.PathLike | str) -> None:
        """Write the model as one JSON file (atomically).

        Claim: infra.
        """
        atomic_write_json(Path(path), self.to_json())

    @classmethod
    def load(cls, path: os.PathLike | str) -> "DirectionModel":
        """Load a model from JSON, re-ordering weights onto the frozen features.

        Claim: infra -- a checkpoint fitted before a feature was appended must
        still load; unknown names are dropped and missing ones get weight 0.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        names = tuple(str(n) for n in data.get("feature_names", DIRECTION_FEATURES))
        w = np.asarray(data.get("weights", []), dtype=np.float64).ravel()
        mean = np.asarray(data.get("scaler_mean", []), dtype=np.float64).ravel()
        scale = np.asarray(data.get("scaler_scale", []), dtype=np.float64).ravel()

        n = len(DIRECTION_FEATURES)
        weights = np.zeros(n, dtype=np.float64)
        smean = np.zeros(n, dtype=np.float64)
        sscale = np.ones(n, dtype=np.float64)
        for i, name in enumerate(names):
            if name not in DIRECTION_FEATURES:
                continue
            j = DIRECTION_FEATURES.index(name)
            if i < w.size:
                weights[j] = float(w[i])
            if i < mean.size:
                smean[j] = float(mean[i])
            if i < scale.size and abs(float(scale[i])) > 1e-12:
                sscale[j] = float(scale[i])
        return cls(
            weights=weights,
            bias=float(data.get("bias", 0.0)),
            feature_names=DIRECTION_FEATURES,
            scaler_mean=smean,
            scaler_scale=sscale,
            outgroup_weight=float(data.get("outgroup_weight", 2.0)),
            version=str(data.get("version", DIRECTION_MODEL_VERSION)),
            meta=dict(data.get("meta", {}) or {}),
        )

    # --------------------------------------------------------------- factories

    @classmethod
    def default(cls) -> "DirectionModel":
        """Hand-set priors: heavy on lossy scars, near-zero on the spectrum.

        Claim: direction -- the weights encode what ``docs/FINDINGS.md``
        actually measured, not what would be flattering:

        * **Large** on ``lattice_asym``, ``zero_subset_asym``, ``orphan_asym``,
          ``vocab_delta`` and ``dtype_precision_asym``.  Every one of these
          comes from a *lossy, irreversible* operation -- quantisation,
          pruning, vocabulary extension, precision reduction -- so the scar can
          only appear downstream and the sign is not a guess.
        * **Small** on the family-(a) spectral features.  Section 3 measured the
          delta-subspace effect at ~1e-3..1e-2 against a per-tensor spread of
          the same order, and found its sign to be the *opposite* of the naive
          intuition (the child has already absorbed the delta, so the child's
          own top-k subspace aligns marginally better).  They are tiebreakers.
        * **Exactly zero** on ``norm_growth_asym``.  Section 2 measured
          ``log||B|| - log||A||`` at -0.0171 (0/8 tensors positive) for
          ``Qwen2.5-0.5B -> -Instruct`` and +0.0113 (8/8 positive) for
          ``SmolLM2-135M -> -Instruct``.  Both edges are unambiguously
          base -> instruct, so the statistic is consistent *within* a pair and
          sign-flipped *across* pairs -- a weight-decay/recipe artefact.  Any
          hand-set sign would be right on one family and wrong on the other, so
          it gets none: the weight is 0.0 and only :meth:`fit` may move it.

        ``outgroup_weight`` is 2.0, and that number is derived rather than
        tuned.  The statistic it multiplies is normalised by
        :data:`CANONICAL_OUTGROUP_GAP`, so a *textbook clean rooting* -- a
        sibling reached by an independent branch of comparable length -- scores
        exactly 1.0.  A weight of 2.0 therefore maps "one clean rooting" to
        ``llr = 2.0``, i.e. ``p = 0.88``: confident enough to clear the 0.5
        abstain threshold and root the edge, but still an order of magnitude
        below what a quantisation lattice or a block of untrained vocabulary
        rows contributes, so a disagreeing lossy scar still wins.  Family (e) is
        the only strong signal available on a scar-free SFT/LoRA edge, but it
        needs a third model, so it must be able to carry a verdict on its own
        without swamping evidence that is genuinely near-deterministic.
        """
        priors: Dict[str, float] = {
            # (b) vocabulary -- monotone, near-deterministic
            "vocab_delta": 2.0,
            "orphan_asym": 3.0,
            # (c) lossy scars -- the strongest evidence Stemma has
            "lattice_asym": 3.0,
            "zero_asym": 1.5,
            "zero_subset_asym": 2.5,
            "dtype_precision_asym": 3.0,
            # (a) delta spectrum -- measured weak; tiebreakers only
            "subspace_energy_asym": 0.25,
            "delta_rank_asym": 0.20,
            "spectral_growth_asym": 0.20,
            "norm_growth_asym": 0.0,  # sign-flips across families: see above
            # (d) fossils -- strong when present, gated when not
            "dead_fossil_asym": 1.0,
            "outlier_fossil_asym": 0.5,
            "exact_tie_asym": 0.5,
        }
        w = np.asarray([priors[name] for name in DIRECTION_FEATURES], dtype=np.float64)
        n = len(DIRECTION_FEATURES)
        return cls(
            weights=w,
            bias=0.0,  # forced: an intercept would break anti-symmetry
            feature_names=DIRECTION_FEATURES,
            scaler_mean=np.zeros(n),  # forced: a non-zero mean would break it too
            scaler_scale=np.ones(n),  # features are already O(1) by construction
            outgroup_weight=2.0,
            meta={"source": "hand-set priors (docs/FINDINGS.md sections 1-4)"},
        )

    @classmethod
    def fit(
        cls,
        X: Any,
        y: Any,
        *,
        l2: float = 1.0,
        symmetrize: bool = True,
        max_iter: int = 200,
    ) -> "DirectionModel":
        """Fit the combiner by L2-regularised logistic regression, **no intercept**.

        Claim: direction -- fitting is how ``norm_growth_asym`` is allowed to
        earn a sign that no hand-set prior could justify (see
        :meth:`default`).  Two constraints are enforced rather than fitted:

        * ``bias`` is forced to ``0.0`` -- an intercept ``c`` would make
          ``llr(-x) = -llr(x) + 2c``, so ordered pairs would no longer negate;
        * ``scaler_mean`` is forced to **zeros** -- centring by a non-zero mean
          ``m`` adds the constant ``-w.m/s``, which is the same failure.

        ``scaler_scale`` is the per-feature standard deviation (zeros guarded to
        1.0), which is a pure odd rescaling and therefore safe.  With
        ``symmetrize=True`` (default) the training set is augmented with
        ``(-X, 1 - y)``, the exact statement that direction is anti-symmetric;
        it also makes the empirical mean exactly zero, so the forced zero mean
        costs nothing.

        ``X`` may be an ``(n, 13)`` array or a sequence of feature dicts; ``y``
        is 1 when "a is the parent of b".
        """
        Xa = _as_feature_matrix(X)
        ya = np.asarray(y, dtype=np.float64).ravel()
        if Xa.shape[0] != ya.size:
            raise ValueError(f"X has {Xa.shape[0]} rows but y has {ya.size} labels")
        ya = (ya > 0.5).astype(np.float64)

        if symmetrize:
            Xa = np.vstack([Xa, -Xa])
            ya = np.concatenate([ya, 1.0 - ya])

        n = len(DIRECTION_FEATURES)
        scale = np.std(Xa, axis=0)
        scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
        Z = Xa / scale

        if ya.size == 0 or float(ya.min()) == float(ya.max()):
            LOG.warning("fit() got a single-class label set; returning zero weights")
            return cls(
                weights=np.zeros(n), bias=0.0, feature_names=DIRECTION_FEATURES,
                scaler_mean=np.zeros(n), scaler_scale=scale, outgroup_weight=2.0,
                meta={"source": "fit", "n_samples": int(ya.size), "degenerate": True},
            )

        w, backend = _fit_logistic(Z, ya, l2=float(l2), max_iter=int(max_iter))
        return cls(
            weights=w,
            bias=0.0,  # forced: see docstring
            feature_names=DIRECTION_FEATURES,
            scaler_mean=np.zeros(n),  # forced: see docstring
            scaler_scale=scale,
            outgroup_weight=2.0,
            meta={
                "source": "fit", "backend": backend, "l2": float(l2),
                "n_samples": int(ya.size), "symmetrized": bool(symmetrize),
            },
        )


def _as_feature_matrix(X: Any) -> np.ndarray:
    """Coerce dicts / arrays / lists into an ``(n, len(DIRECTION_FEATURES))`` matrix.

    Claim: infra -- the benchmark collects features as dicts and stores them as
    arrays; both must train the same model.
    """
    if isinstance(X, Mapping):
        X = [X]
    if isinstance(X, (list, tuple)) and X and isinstance(X[0], Mapping):
        return np.asarray(
            [[_finite(row.get(name, 0.0)) for name in DIRECTION_FEATURES] for row in X],
            dtype=np.float64,
        )
    M = np.atleast_2d(np.asarray(X, dtype=np.float64))
    n = len(DIRECTION_FEATURES)
    if M.shape[1] != n:
        raise ValueError(f"expected {n} feature columns, got {M.shape[1]}")
    return np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)


def _fit_logistic(Z: np.ndarray, y: np.ndarray, *, l2: float, max_iter: int) -> Tuple[np.ndarray, str]:
    """Logistic regression without an intercept: sklearn if present, else IRLS.

    Claim: infra -- the pure-numpy Newton/IRLS fallback keeps the fitted
    combiner reproducible on a machine without scikit-learn, which matters
    because the fitted weights are part of the published artifact.
    """
    try:  # lazy: sklearn is optional
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(
            fit_intercept=False,
            C=1.0 / max(float(l2), 1e-9),
            solver="lbfgs",
            max_iter=int(max_iter) * 5,
        )
        clf.fit(Z, y)
        return np.asarray(clf.coef_, dtype=np.float64).ravel(), "sklearn"
    except Exception as exc:
        LOG.debug("sklearn unavailable or failed (%s); using IRLS", exc)

    n_features = Z.shape[1]
    w = np.zeros(n_features, dtype=np.float64)
    reg = float(max(l2, 1e-9)) * np.eye(n_features)
    for _ in range(int(max_iter)):
        eta = np.clip(Z @ w, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        grad = Z.T @ (y - p) - float(max(l2, 1e-9)) * w
        s = np.clip(p * (1.0 - p), 1e-9, None)
        H = (Z.T * s) @ Z + reg
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:  # pragma: no cover
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        w = w + step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    return w, "irls"


def _coerce_model(weights: Any) -> DirectionModel:
    """Interpret the ``weights=`` argument as a :class:`DirectionModel`.

    Claim: infra -- the CLI passes a path, the phylogeny builder passes a model
    or ``None``, and a notebook may pass a raw array; all three must work.
    """
    if weights is None:
        return DirectionModel.default()
    if isinstance(weights, DirectionModel):
        return weights
    if isinstance(weights, (str, os.PathLike, Path)):
        return DirectionModel.load(weights)
    if isinstance(weights, Mapping):
        if "weights" in weights:
            model = DirectionModel.default()
            data = dict(weights)
            return DirectionModel(
                weights=np.asarray(data["weights"], dtype=np.float64),
                bias=float(data.get("bias", 0.0)),
                feature_names=DIRECTION_FEATURES,
                scaler_mean=np.asarray(
                    data.get("scaler_mean", np.zeros(len(DIRECTION_FEATURES))), dtype=np.float64
                ),
                scaler_scale=np.asarray(
                    data.get("scaler_scale", np.ones(len(DIRECTION_FEATURES))), dtype=np.float64
                ),
                outgroup_weight=float(data.get("outgroup_weight", model.outgroup_weight)),
            )
        vec = np.asarray([_finite(weights.get(k, 0.0)) for k in DIRECTION_FEATURES])
        return DirectionModel(weights=vec)
    return DirectionModel(weights=np.asarray(weights, dtype=np.float64))


# --------------------------------------------------------------------------- #
# Human-readable evidence
# --------------------------------------------------------------------------- #


def _evidence_strings(ev: PairEvidence, feats: Mapping[str, float]) -> List[str]:
    """Render the firing evidence families as sentences for the CLI and Space.

    Claim: direction -- a provenance claim nobody can read is not auditable, and
    this project's disclaimer requires a human to review the reasoning, not just
    the number.  Each line names the measurement, the number, and the arrow it
    implies.
    """
    a_name = "A"
    b_name = "B"
    out: List[str] = []
    sa, sb = ev.side_a, ev.side_b
    shared = ev.shared

    def arrow(value: float) -> str:
        if value > 0:
            return f"{a_name} is upstream of {b_name}"
        return f"{b_name} is upstream of {a_name}"

    # (b) vocabulary + orphans
    va, vb = _finite(sa.get("vocab_rows")), _finite(sb.get("vocab_rows"))
    if va != vb:
        wider, narrower = (b_name, a_name) if vb > va else (a_name, b_name)
        orph = ev.orphan_b if vb > va else ev.orphan_a
        n_extra = int(abs(vb - va))
        if _finite(orph.get("n_rows")) > 0:
            out.append(
                f"{wider} has {n_extra} embedding rows absent from {narrower}, "
                f"{100.0 * _finite(orph.get('untrained_frac')):.0f}% of which look untrained "
                f"(KS={_finite(orph.get('ks')):.2f}, cv={_finite(orph.get('cv')):.3f}) "
                f"-> {wider} extends {narrower}'s vocabulary"
            )
        else:
            out.append(
                f"{wider} has {n_extra} more embedding rows than {narrower}; vocabularies only "
                f"grow -> {wider} is downstream"
            )
        if _finite(orph.get("dup_frac")) > 0.01:
            out.append(
                f"{100.0 * _finite(orph.get('dup_frac')):.0f}% of those rows are exact duplicates "
                f"of one another (padding artefact)"
            )

    # (c) lattice
    la, lb = _finite(sa.get("lattice_frac")), _finite(sb.get("lattice_frac"))
    if abs(la - lb) > 0.05:
        hi, lo = (b_name, a_name) if lb > la else (a_name, b_name)
        lev = _finite(sb.get("lattice_levels") if lb > la else sa.get("lattice_levels"))
        out.append(
            f"{100.0 * max(la, lb):.0f}% of {hi}'s sampled weights sit on a per-row value lattice "
            f"(~{lev:.0f} levels) versus {100.0 * min(la, lb):.0f}% for {lo}; quantisation is "
            f"irreversible -> {hi} is downstream"
        )

    # (c) zeros
    za, zb = _finite(sa.get("zero_frac")), _finite(sb.get("zero_frac"))
    if max(za, zb) > _ZERO_GATE and abs(za - zb) > 0.01:
        hi, lo = (b_name, a_name) if zb > za else (a_name, b_name)
        cont = _finite(ev.pair_ab.get("zero_containment") if zb > za
                       else ev.pair_ba.get("zero_containment"))
        out.append(
            f"{hi} is {100.0 * max(za, zb):.1f}% exactly zero versus "
            f"{100.0 * min(za, zb):.1f}% for {lo}, and {100.0 * cont:.0f}% of {lo}'s zeros are "
            f"also zero in {hi} (zero-set superset) -> {hi} was pruned from {lo}"
        )

    # (c) dtype
    ba_, bb_ = _finite(sa.get("bits")), _finite(sb.get("bits"))
    if abs(ba_ - bb_) > 0.5:
        lower, higher = (b_name, a_name) if bb_ < ba_ else (a_name, b_name)
        out.append(
            f"mean stored precision is {min(ba_, bb_):.1f} bits for {lower} versus "
            f"{max(ba_, bb_):.1f} for {higher}; precision only ever goes down "
            f"-> {lower} is downstream"
        )

    # (d) fossils
    dfa, dfb = _finite(sa.get("dead_frac")), _finite(sb.get("dead_frac"))
    if max(dfa, dfb) > _DEAD_GATE:
        out.append(
            f"dead rows: {100.0 * dfa:.2f}% in {a_name}, {100.0 * dfb:.2f}% in {b_name}; "
            f"{100.0 * _finite(ev.pair_ab.get('dead_containment')):.0f}% of {a_name}'s dead rows "
            f"are dead in {b_name} and "
            f"{100.0 * _finite(ev.pair_ba.get('dead_containment')):.0f}% the other way"
        )
    if _finite(shared.get("n_identical")) > 0:
        out.append(
            f"{int(_finite(shared.get('n_identical')))} of "
            f"{int(_finite(shared.get('n_compared')))} sampled tensors are bit-identical "
            f"(strong relatedness, no direction on its own)"
        )
    if int(_finite(shared.get("n_only_a"))) or int(_finite(shared.get("n_only_b"))):
        out.append(
            f"tensor inventory differs: {int(_finite(shared.get('n_only_a')))} only in {a_name}, "
            f"{int(_finite(shared.get('n_only_b')))} only in {b_name}"
        )

    # (a) spectrum -- always reported, always labelled weak
    out.append(
        f"delta spectrum (weak, ~1e-2): subspace energy {_finite(sa.get('subspace_energy')):.4f} "
        f"in {a_name}'s top-{SUBSPACE_K} basis vs {_finite(sb.get('subspace_energy')):.4f} in "
        f"{b_name}'s; log-norm growth {_finite(sb.get('log_norm')) - _finite(sa.get('log_norm')):+.4f} "
        f"(unsigned prior: measured to flip sign across model families)"
    )

    # (e) outgroup
    if ev.has_outgroup:
        stat = ev.outgroup_stat
        out.append(
            f"outgroup rooting over {len(ev.outgroup.get('models', []))} model(s) "
            f"({', '.join(str(m) for m in ev.outgroup.get('models', []))}) across "
            f"{int(ev.outgroup.get('n_terms', 0))} tensor comparisons: normalised distance gap "
            f"{stat:+.3f} -> {arrow(stat)} (closer to every outgroup == closer to the root)"
        )
    else:
        out.append(
            "no outgroup supplied: family (e) inactive, so a scar-free fine-tune edge may "
            "legitimately abstain"
        )
    return out


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #


def estimate_direction(
    a: ModelRef,
    b: ModelRef,
    *,
    weights: Any = None,
    abstain: float = 0.5,
    evidence: Optional[PairEvidence] = None,
    outgroup: Optional[Sequence[ModelRef]] = None,
    **kw: Any,
) -> DirectionVerdict:
    """Decide which of two checkpoints is the ancestor, with per-feature evidence.

    Claim: direction -- the headline entry point.  ``llr > abstain`` reports
    ``"a->b"``, ``llr < -abstain`` reports ``"b->a"`` and anything between is
    ``"unknown"``; abstention is a first-class outcome because a confident wrong
    provenance claim is worse than no claim.  Every symmetric baseline
    (:mod:`stemma.baselines`) is pinned at ``"unknown"`` by construction, which
    is the 50% structural ceiling this function is measured against.

    ``outgroup`` activates family (e): supplying one or more sibling models lets
    a *scar-free* SFT/LoRA edge be rooted (see :func:`_collect_outgroup`).
    Without it the term is exactly zero and such an edge will often, correctly,
    abstain.

    Anti-symmetry: ``estimate_direction(b, a).llr == -estimate_direction(a, b).llr``
    to within floating-point exactness, because the features are anti-symmetric
    and the combiner is odd.
    """
    model = _coerce_model(weights)
    ev = evidence if evidence is not None else collect_pair_evidence(
        a, b, outgroup=outgroup, **kw
    )
    feats = direction_features(ev)

    scored: Dict[str, float] = dict(feats)
    if ev.has_outgroup:
        scored[OUTGROUP_KEY] = ev.outgroup_stat
    llr = model.llr(scored)
    contributions = model.contributions(scored)

    thr = float(abs(abstain))
    if llr > thr:
        direction = "a->b"
    elif llr < -thr:
        direction = "b->a"
    else:
        direction = "unknown"

    p = sigmoid(llr)
    return DirectionVerdict(
        a=str(a),
        b=str(b),
        direction=direction,
        llr=float(llr),
        p_a_parent=float(p),
        confidence=float(abs(2.0 * p - 1.0)),
        features=feats,
        contributions=contributions,
        evidence=_evidence_strings(ev, feats),
        stats=ev.stats,
    )


def relatedness_score(
    a: ModelRef,
    b: ModelRef,
    *,
    sa: Optional[Sketch] = None,
    sb: Optional[Sketch] = None,
    max_rows: int = 512,
    n_tensors: int = DEFAULT_N_TENSORS,
    seed: int = 0,
    **kw: Any,
) -> float:
    """Symmetric, order-independent relatedness in ``[0, 1]``.

    Claim: low-false-positive -- this is the gate that keeps unrelated models
    out of the DAG, and it is the benchmark's AUC axis.  It is *deliberately*
    symmetric: it can never smuggle in a direction.

    The critical design point: **the sketch alone is not enough.**  Two models
    with the same architecture but independent initialisations have near
    identical spectra (both are essentially random matrices), so every
    permutation/scale-invariant fingerprint puts them close together.  What
    actually separates "same architecture" from "same lineage" is the *raw*
    weights on identical sampled coordinates: a fine-tune keeps a cosine of
    ~1.0 and a relative delta norm of ~0.01, while two independent inits give a
    cosine of ~0 and a relative delta norm of ~sqrt(2).  Both terms are
    included, and the raw term carries the majority of the weight.

    Composition: ``0.30 * sketch_similarity + 0.55 * raw_similarity +
    0.15 * bit_identical_fraction``, where ``raw_similarity`` is the mean of the
    clipped cosine and ``1 - relative_delta_norm / sqrt(2)``.  When the two
    models share no comparable tensor the raw term is unavailable and the score
    is capped at ``0.8 * sketch_similarity`` rather than trusted outright.

    Order independence is enforced by sorting the two references first, so the
    result is bit-for-bit identical either way round.
    """
    ref_x, ref_y = str(a), str(b)
    sk_x, sk_y = sa, sb
    if ref_y < ref_x:
        ref_x, ref_y = ref_y, ref_x
        sk_x, sk_y = sk_y, sk_x

    # --- symmetric fingerprint term ------------------------------------- #
    sketch_sim = 0.0
    try:
        from .sketch import sketch_distance, sketch_model  # lazy

        if sk_x is None:
            sk_x = sketch_model(ref_x, seed=seed, **{k: v for k, v in kw.items() if k in _LOADER_KW})
        if sk_y is None:
            sk_y = sketch_model(ref_y, seed=seed, **{k: v for k, v in kw.items() if k in _LOADER_KW})
        sketch_sim = float(np.clip(1.0 - sketch_distance(sk_x, sk_y) / 2.0, 0.0, 1.0))
    except Exception as exc:
        LOG.warning("sketch comparison failed for %s / %s: %s", ref_x, ref_y, exc)

    # --- raw-weight term (the part that actually rejects a same-arch pair) - #
    src_x = src_y = None
    cos = 0.0
    rel = float("nan")
    identical_frac = 0.0
    have_raw = False
    try:
        loader_kw = {k: v for k, v in kw.items() if k in _LOADER_KW}
        src_x = _open_source(ref_x, **loader_kw)
        src_y = _open_source(ref_y, **loader_kw)
        idx_x, idx_y = src_x.index(), src_y.index()
        names = _select_shared_tensors(idx_x, idx_y, int(n_tensors))
        if names:
            from .remote_loader import select_rows

            dot = 0.0
            nx2 = 0.0
            ny2 = 0.0
            dd = 0.0
            n_identical = 0
            n_used = 0
            for name in names:
                n_rows = min(int(idx_x[name].shape[0]), int(idx_y[name].shape[0]))
                rows = select_rows(n_rows, int(max_rows))
                if rows.size == 0:
                    continue
                X = _flatten2d(src_x.get_tensor_rows(name, rows, dtype=np.float32))
                Y = _flatten2d(src_y.get_tensor_rows(name, rows, dtype=np.float32))
                if X.shape != Y.shape or X.size == 0:
                    continue
                Xd = np.asarray(X, dtype=np.float64)
                Yd = np.asarray(Y, dtype=np.float64)
                dot += float(np.einsum("ij,ij->", Xd, Yd))
                nx2 += float(np.einsum("ij,ij->", Xd, Xd))
                ny2 += float(np.einsum("ij,ij->", Yd, Yd))
                dd += float(np.einsum("ij,ij->", Yd - Xd, Yd - Xd))
                n_used += 1
                if np.array_equal(X, Y):
                    n_identical += 1
            if n_used:
                have_raw = True
                identical_frac = n_identical / float(n_used)
                denom = math.sqrt(max(nx2, _EPS)) * math.sqrt(max(ny2, _EPS))
                cos = float(dot / denom) if denom > 0.0 else 0.0
                mean_norm = 0.5 * (math.sqrt(max(nx2, 0.0)) + math.sqrt(max(ny2, 0.0)))
                rel = math.sqrt(max(dd, 0.0)) / max(mean_norm, _EPS)
    except Exception as exc:
        LOG.warning("raw-weight comparison failed for %s / %s: %s", ref_x, ref_y, exc)
    finally:
        for src in (src_x, src_y):
            if src is not None:
                try:
                    src.close()
                except Exception:  # pragma: no cover
                    pass

    if not have_raw:
        # No comparable tensor at all.  This is *absence of evidence*, not weak
        # evidence, and the distinction matters: measured on the test fixtures,
        # the invariant sketch scores 0.857 between a model and a structurally
        # different stranger, and 0.877 between a model and a same-architecture
        # model trained from an independent seed.  Both are unrelated.  A
        # permutation/scale-invariant fingerprint simply cannot reject either,
        # because every transformer weight matrix looks approximately like a
        # random matrix with a similar spectrum -- which is the whole reason
        # Stemma does not rest its relatedness gate on a fingerprint.
        #
        # So when the raw-coordinate term is unavailable we refuse to let the
        # sketch alone carry a model over the gate.  The score is capped
        # strictly below NO_SHARED_TENSOR_CEILING < the shipped operating
        # threshold, which turns this case into an abstention rather than a
        # claim.  Cross-architecture distillation lands here on purpose:
        # docs/FINDINGS.md documents it as a case where weight-level lineage is
        # expected to be weak, and the benchmark scores it as a known miss
        # instead of quietly excluding it.
        return float(np.clip(0.8 * sketch_sim, 0.0, NO_SHARED_TENSOR_CEILING))

    close = float(np.clip(1.0 - (rel if math.isfinite(rel) else 2.0) / math.sqrt(2.0), 0.0, 1.0))
    raw_sim = 0.5 * float(np.clip(cos, 0.0, 1.0)) + 0.5 * close
    score = 0.30 * sketch_sim + 0.55 * raw_sim + 0.15 * float(np.clip(identical_frac, 0.0, 1.0))
    return float(np.clip(score, 0.0, 1.0))


__all__ = [
    "DIRECTION_MODEL_VERSION",
    "NO_SHARED_TENSOR_CEILING",
    "OUTGROUP_KEY",
    "SUBSPACE_K",
    "DirectionModel",
    "PairEvidence",
    "collect_pair_evidence",
    "direction_features",
    "estimate_direction",
    "lattice_fit",
    "relatedness_score",
    "subspace_energy",
]
