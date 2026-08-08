"""Shared dataclasses and type aliases for Stemma.

Every other module in the package codes against the structures defined here, so
this file is the single source of truth for the wire format of sketches,
direction verdicts, merge decompositions, phylogeny DAGs and AI-BOM records.

Claim support (see CONTRACT.md): this module itself proves nothing; it exists so
that the four headline claims -- *direction*, *merge recovery*, *low transfer*,
*low false-positive rate* -- are all reported through one comparable schema.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Versioning: bump when a stored sketch / BOM can no longer be compared to old
# ones. The benchmark records these strings so results stay reproducible.
# --------------------------------------------------------------------------- #
SKETCH_VERSION = "stemma-sketch-v1"
BOM_VERSION = "stemma-bom-v1"

#: A model reference is either a Hugging Face repo id ("org/name") or a local
#: filesystem path to a directory containing *.safetensors.
ModelRef = str


# --------------------------------------------------------------------------- #
# Roles / depth buckets: the fixed coordinate system a sketch is written in.
# --------------------------------------------------------------------------- #

#: Canonical tensor roles. Order is part of the sketch wire format -- never
#: reorder, only append (and bump SKETCH_VERSION).
ROLES: Tuple[str, ...] = (
    "embed",
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_o",
    "mlp_in",
    "mlp_out",
    "lm_head",
    "norm",
)

#: Relative-depth buckets. A tensor at layer i of L lands in the bucket whose
#: centre is nearest to i / max(L - 1, 1). Using *relative* depth is what lets
#: us compare a 24-layer child to its 24-layer parent and still say something
#: about a 12-layer distilled cousin.
DEPTH_BUCKETS: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

#: Per-(role, depth) invariant feature width. See sketch.tensor_invariants.
FEATURES_PER_SLOT: int = 32

#: Global (whole-model) features appended after the per-slot block.
N_GLOBAL_FEATURES: int = 16

#: Total sketch dimensionality: 9 roles x 5 depths x 32 + 16 = 1456.
SKETCH_DIM: int = len(ROLES) * len(DEPTH_BUCKETS) * FEATURES_PER_SLOT + N_GLOBAL_FEATURES


# --------------------------------------------------------------------------- #
# safetensors remote access
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TensorMeta:
    """Location of one tensor inside a safetensors file, from the header alone.

    Supports the *low transfer* claim: knowing ``(dtype, shape, byte range)``
    without downloading payload is what makes selective Range reads possible.
    """

    name: str
    dtype: str  # safetensors dtype string, e.g. "BF16", "F32", "I8"
    shape: Tuple[int, ...]
    begin: int  # absolute byte offset in the file (header size already added)
    end: int  # exclusive
    file: str = "model.safetensors"  # shard filename this tensor lives in

    @property
    def nbytes(self) -> int:
        return self.end - self.begin

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


@dataclass
class TransferStats:
    """Byte/latency accounting for one model access.

    Supports the *low transfer* claim: ``bytes_read`` is compared against
    ``full_size_bytes`` (sum of all shard sizes) to compute the reduction
    factor reported in the benchmark table.
    """

    bytes_read: int = 0
    requests: int = 0
    seconds: float = 0.0
    full_size_bytes: int = 0
    cache_hits: int = 0

    def add(self, other: "TransferStats") -> "TransferStats":
        return TransferStats(
            bytes_read=self.bytes_read + other.bytes_read,
            requests=self.requests + other.requests,
            seconds=self.seconds + other.seconds,
            full_size_bytes=max(self.full_size_bytes, other.full_size_bytes),
            cache_hits=self.cache_hits + other.cache_hits,
        )

    @property
    def reduction(self) -> float:
        """How many times less data than a full download. 1.0 == no saving."""
        if self.bytes_read <= 0:
            return float("inf")
        return self.full_size_bytes / self.bytes_read


# --------------------------------------------------------------------------- #
# Sketches
# --------------------------------------------------------------------------- #


@dataclass
class Sketch:
    """Fixed-length, symmetry-invariant fingerprint of a model's weights.

    Supports the *low false-positive* and *low transfer* claims: the vector is
    computed from a handful of Range-read tensors and is invariant to neuron
    permutation and to per-row/per-column rescaling, so unrelated models do not
    collide just because they share an architecture.

    It is deliberately **symmetric** -- a sketch alone can never give direction.
    That job belongs to :mod:`stemma.direction`.
    """

    model_id: ModelRef
    vector: np.ndarray  # float32, shape (SKETCH_DIM,)
    present: np.ndarray  # bool, shape (len(ROLES) * len(DEPTH_BUCKETS),)
    version: str = SKETCH_VERSION
    meta: Dict[str, Any] = field(default_factory=dict)
    stats: Optional[TransferStats] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "vector": [float(x) for x in np.asarray(self.vector).ravel()],
            "present": [bool(x) for x in np.asarray(self.present).ravel()],
            "meta": self.meta,
            "stats": asdict(self.stats) if self.stats else None,
        }

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "Sketch":
        st = d.get("stats")
        return Sketch(
            model_id=d["model_id"],
            vector=np.asarray(d["vector"], dtype=np.float32),
            present=np.asarray(d["present"], dtype=bool),
            version=d.get("version", SKETCH_VERSION),
            meta=d.get("meta", {}),
            stats=TransferStats(**st) if st else None,
        )


# --------------------------------------------------------------------------- #
# Direction
# --------------------------------------------------------------------------- #

#: Names of the asymmetric direction features, in the order produced by
#: ``direction.direction_features``. Part of the fitted-model wire format.
DIRECTION_FEATURES: Tuple[str, ...] = (
    "vocab_delta",  # (b) vocabulary growth, signed
    "orphan_asym",  # (b) untrained "orphan" embedding rows only in one side
    "lattice_asym",  # (c) quantisation lattice scar
    "zero_asym",  # (c) pruning zero-structure scar
    "zero_subset_asym",  # (c) zeros of one side are a superset of the other's
    "dtype_precision_asym",  # (c) precision only ever goes down
    "subspace_energy_asym",  # (a) delta lives in the parent's dominant subspace
    "delta_rank_asym",  # (a) effective rank of delta relative to each side
    "spectral_growth_asym",  # (a) spectral mass accumulation
    "norm_growth_asym",  # (a) Frobenius norm growth under training
    "dead_fossil_asym",  # (d) dead-neuron fossils persist downstream
    "outlier_fossil_asym",  # (d) outlier channels persist and sharpen
    "exact_tie_asym",  # (d) bit-identical tensor fossils + extra-tensor sign
)


@dataclass
class DirectionVerdict:
    """Which of two related models is the ancestor, and how sure we are.

    Supports the headline *direction* claim: symmetric baselines (cosine, CKA,
    HuRef-style invariants) are structurally incapable of filling this in and
    score 50% by construction.
    """

    a: ModelRef
    b: ModelRef
    #: "a->b", "b->a", or "unknown" when |llr| is under the abstain threshold.
    direction: str
    #: Log-likelihood ratio, positive means a is the parent of b.
    llr: float
    #: sigmoid(llr) -- probability that a is the parent of b.
    p_a_parent: float
    confidence: float
    features: Dict[str, float] = field(default_factory=dict)
    #: Per-feature signed contribution to the llr, for the UI's evidence table.
    contributions: Dict[str, float] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    stats: Optional[TransferStats] = None


# --------------------------------------------------------------------------- #
# Merge decomposition
# --------------------------------------------------------------------------- #


@dataclass
class MergeDecomposition:
    """Sparse non-negative decomposition of a merged model over task vectors.

    Supports the *merge recovery* claim: reports which candidate parents were
    selected and with what coefficients, both of which symmetric pairwise
    fingerprints cannot produce at all.
    """

    child: ModelRef
    base: Optional[ModelRef]
    candidates: List[ModelRef]
    #: Non-negative coefficients aligned with ``candidates``.
    coefficients: np.ndarray
    #: Subset of ``candidates`` whose coefficient survived the support threshold.
    selected: List[ModelRef]
    residual: float  # ||tau_child - sum w_i tau_i|| / ||tau_child||
    r2: float
    method: str = "nnls+l1"
    stats: Optional[TransferStats] = None

    def as_dict(self) -> Dict[str, float]:
        return {c: float(w) for c, w in zip(self.candidates, np.asarray(self.coefficients))}


# --------------------------------------------------------------------------- #
# Phylogeny + rights
# --------------------------------------------------------------------------- #


@dataclass
class Edge:
    """One directed parent -> child claim in the reconstructed DAG."""

    parent: ModelRef
    child: ModelRef
    confidence: float
    relation: str = "derived"  # derived | merge | quantized | pruned | vocab_extended | finetuned
    weight: Optional[float] = None  # merge mixing coefficient when known
    evidence: List[str] = field(default_factory=list)


@dataclass
class Phylogeny:
    """Reconstructed multi-parent lineage DAG."""

    nodes: List[ModelRef]
    edges: List[Edge]
    root_candidates: List[ModelRef] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def parents_of(self, node: ModelRef) -> List[Edge]:
        return [e for e in self.edges if e.child == node]

    def children_of(self, node: ModelRef) -> List[Edge]:
        return [e for e in self.edges if e.parent == node]


@dataclass
class LicenseFacts:
    """License / usage-restriction facts scraped from a model card."""

    model_id: ModelRef
    license: Optional[str] = None
    license_name: Optional[str] = None
    commercial_use: Optional[bool] = None  # None == unknown
    share_alike: bool = False
    attribution_required: bool = False
    gated: bool = False
    restrictions: List[str] = field(default_factory=list)
    source: str = "unknown"  # "model_card" | "declared" | "unknown"
    raw_tags: List[str] = field(default_factory=list)


@dataclass
class RightsConflict:
    """A detected license inconsistency along the DAG.

    Deliberately phrased as *statistical evidence*, never as a legal finding.
    """

    descendant: ModelRef
    ancestor: ModelRef
    kind: str  # "noncommercial_ancestor" | "share_alike_broken" | "unknown_ancestor" | "gated_ancestor"
    severity: str  # "info" | "warning" | "high"
    message: str
    path: List[ModelRef] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class BOM:
    """AI Bill of Materials with per-edge confidence.

    Supports all four claims by carrying them into one auditable artifact: each
    component records how it was inferred, at what confidence, and how many
    bytes were read to infer it.
    """

    root: ModelRef
    version: str = BOM_VERSION
    generated_by: str = "stemma"
    components: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    transfer: Dict[str, Any] = field(default_factory=dict)
    disclaimer: str = (
        "Stemma reports statistical evidence about weight-level similarity and "
        "derivation direction. It does NOT establish provenance as fact and "
        "does NOT constitute a legal determination of license compliance or "
        "infringement. A human must review before any action is taken."
    )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Benchmark records
# --------------------------------------------------------------------------- #


@dataclass
class GroundTruthEdge:
    parent: ModelRef
    child: ModelRef
    relation: str
    weight: Optional[float] = None


@dataclass
class BenchmarkPair:
    """One labelled ordered pair used by the benchmark harness."""

    a: ModelRef
    b: ModelRef
    related: bool
    #: "a->b", "b->a", "sibling", or "none".
    direction: str
    relation: str = "none"


def sigmoid(x: float) -> float:
    """Numerically stable logistic function used for llr -> probability."""
    if x >= 0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    z = np.exp(x)
    return float(z / (1.0 + z))
