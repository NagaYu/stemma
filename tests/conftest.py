"""Fixtures that build tiny, *real* safetensors checkpoints on disk.

Claim: infra -- the whole suite is offline and self-contained: every model used
by the four required tests (false positive, direction, merge recovery, transfer)
is generated here from a seed, so no test depends on the network, on
``bench_models/`` or on the Hugging Face cache.

The models are deliberately Llama-shaped (``model.embed_tokens.weight``,
``model.layers.N.self_attn.{q,k,v,o}_proj.weight``,
``model.layers.N.mlp.{gate,up,down}_proj.weight``,
``model.layers.N.input_layernorm.weight``, ``model.norm.weight``,
``lm_head.weight``) so that :func:`stemma.utils.role_of` and
:func:`stemma.utils.layer_index_of` resolve every tensor into the same
(role, depth-bucket) coordinate system the real sketcher uses.

Sizes are chosen so the whole suite runs in well under a minute:
hidden 128, 4 layers, vocab 2048, intermediate 256 -> ~1.2 M parameters,
~4.7 MB per checkpoint in float32.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest
from safetensors.numpy import save_file

# --------------------------------------------------------------------------- #
# Geometry of the tiny models. Small enough to be fast, large enough that the
# spectral invariants in stemma.sketch are not dominated by finite-size noise.
# --------------------------------------------------------------------------- #
HIDDEN: int = 128
N_LAYERS: int = 4
VOCAB: int = 2048
INTERMEDIATE: int = 256

#: Extra vocabulary rows added by the vocab-extension child.
EXTRA_VOCAB: int = 512
#: Standard deviation of the freshly initialised (untrained) embedding rows.
FRESH_INIT_STD: float = 0.02
#: Relative Frobenius magnitude of a simulated SFT update (low-rank + noise).
SFT_DELTA_SCALE: float = 0.03
#: Rank of that update.
SFT_DELTA_RANK: int = 8
#: Fraction of weights zeroed by the magnitude-pruning child.
PRUNE_FRACTION: float = 0.30

#: Tensors that carry the "training signal"; 1-D norms are handled separately.
_MATRIX_MIN_DIM: int = 2


# --------------------------------------------------------------------------- #
# Weight synthesis
# --------------------------------------------------------------------------- #


def _structured_matrix(rng: np.random.Generator, m: int, n: int, std: float) -> np.ndarray:
    """A matrix that *looks* trained: decaying spectrum + heavy-tailed row norms.

    Claim: low-false-positive -- an i.i.d. Gaussian block would give every model
    of the same shape an identical spectral fingerprint, which would make the
    same-shape/different-seed control trivially "similar". Giving each model its
    own decaying spectrum and its own heavy-tailed row-norm profile reproduces
    the structure a real checkpoint has, so the control is a fair one.
    """
    r = min(m, n)
    U = rng.standard_normal((m, r))
    V = rng.standard_normal((r, n))
    decay = np.exp(-np.arange(r, dtype=np.float64) / max(1.0, r / 3.0))
    W = (U * decay) @ V
    # per-row (channel) gain -> heavy-tailed row-norm distribution
    W *= np.exp(0.4 * rng.standard_normal((m, 1)))
    W /= max(float(W.std()), 1e-12)
    return (W * std).astype(np.float32)


def _norm_vector(rng: np.random.Generator, n: int) -> np.ndarray:
    return (1.0 + 0.01 * rng.standard_normal(n)).astype(np.float32)


def base_weights(
    seed: int,
    *,
    hidden: int = HIDDEN,
    n_layers: int = N_LAYERS,
    vocab: int = VOCAB,
    intermediate: int = INTERMEDIATE,
) -> Dict[str, np.ndarray]:
    """Generate a complete Llama-style float32 state dict from ``seed``.

    Claim: infra -- deterministic given the seed, so every fixture in the suite
    is reproducible and two runs of the test session compare identical bytes.
    """
    rng = np.random.default_rng(seed)
    w: Dict[str, np.ndarray] = {}
    w["model.embed_tokens.weight"] = _structured_matrix(rng, vocab, hidden, 0.02)
    for i in range(n_layers):
        p = f"model.layers.{i}."
        for nm in ("q", "k", "v", "o"):
            w[f"{p}self_attn.{nm}_proj.weight"] = _structured_matrix(
                rng, hidden, hidden, 1.0 / np.sqrt(hidden)
            )
        w[f"{p}mlp.gate_proj.weight"] = _structured_matrix(rng, intermediate, hidden, 1.0 / np.sqrt(hidden))
        w[f"{p}mlp.up_proj.weight"] = _structured_matrix(rng, intermediate, hidden, 1.0 / np.sqrt(hidden))
        w[f"{p}mlp.down_proj.weight"] = _structured_matrix(
            rng, hidden, intermediate, 1.0 / np.sqrt(intermediate)
        )
        w[f"{p}input_layernorm.weight"] = _norm_vector(rng, hidden)
    w["model.norm.weight"] = _norm_vector(rng, hidden)
    w["lm_head.weight"] = _structured_matrix(rng, vocab, hidden, 0.02)
    return w


def matrix_names(w: Dict[str, np.ndarray]) -> List[str]:
    """Names of the 2-D tensors in a state dict, in deterministic order."""
    return sorted(n for n, t in w.items() if t.ndim >= _MATRIX_MIN_DIM)


def apply_sft_delta(
    w: Dict[str, np.ndarray],
    seed: int,
    *,
    scale: float = SFT_DELTA_SCALE,
    rank: int = SFT_DELTA_RANK,
) -> Dict[str, np.ndarray]:
    """Return a copy of ``w`` plus a seeded low-rank update on every matrix.

    Claim: direction -- this is the deliberately *scar-free* edge: same shapes,
    same dtype, same vocabulary, no lossy operation. docs/FINDINGS.md section 4
    records that such an edge is only weakly identifiable from two models alone,
    and the direction tests assert exactly that weaker property.
    """
    rng = np.random.default_rng(seed)
    out = {k: v.copy() for k, v in w.items()}
    for name in matrix_names(w):
        W = out[name]
        m, n = W.shape
        r = max(1, min(rank, m, n))
        A = rng.standard_normal((m, r))
        B = rng.standard_normal((r, n))
        D = A @ B
        D *= scale * float(np.linalg.norm(W)) / max(float(np.linalg.norm(D)), 1e-12)
        out[name] = (W + D).astype(np.float32)
    return out


def apply_int8_roundtrip(w: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Per-output-channel symmetric int8 quantise -> dequantise of every matrix.

    Claim: direction -- quantisation is lossy and irreversible, so the value
    lattice it leaves behind can only ever appear *downstream*. FINDINGS section
    1 records this as the near-deterministic evidence family.
    """
    out = {k: v.copy() for k, v in w.items()}
    for name in matrix_names(w):
        W = out[name].astype(np.float64)
        s = np.max(np.abs(W), axis=1, keepdims=True) / 127.0
        s = np.where(s <= 0, 1.0, s)
        q = np.clip(np.rint(W / s), -127, 127)
        out[name] = (q * s).astype(np.float32)
    return out


def apply_magnitude_pruning(
    w: Dict[str, np.ndarray], fraction: float = PRUNE_FRACTION
) -> Dict[str, np.ndarray]:
    """Zero the smallest-magnitude ``fraction`` of entries of every matrix.

    Claim: direction -- pruning only ever adds exact zeros, so the child's zero
    set is a superset of the parent's; that asymmetry is the ``zero_subset``
    evidence family.
    """
    out = {k: v.copy() for k, v in w.items()}
    for name in matrix_names(w):
        W = out[name]
        flat = np.abs(W).ravel()
        k = int(round(fraction * flat.size))
        if k <= 0:
            continue
        thresh = np.partition(flat, k - 1)[k - 1]
        W = np.where(np.abs(W) <= thresh, np.float32(0.0), W)
        out[name] = W.astype(np.float32)
    return out


def apply_vocab_extension(
    w: Dict[str, np.ndarray], seed: int, extra: int = EXTRA_VOCAB
) -> Dict[str, np.ndarray]:
    """Append ``extra`` freshly initialised N(0, 0.02) rows to embed and lm_head.

    Claim: direction -- vocabulary only grows, and fresh rows are statistically
    distinguishable from trained ones (tight norm concentration vs a heavy tail),
    so the wider model must be the descendant.
    """
    rng = np.random.default_rng(seed)
    out = {k: v.copy() for k, v in w.items()}
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        W = out[name]
        fresh = (rng.standard_normal((extra, W.shape[1])) * FRESH_INIT_STD).astype(np.float32)
        out[name] = np.concatenate([W, fresh], axis=0).astype(np.float32)
    return out


def linear_merge(
    base: Dict[str, np.ndarray], parts: List[Tuple[Dict[str, np.ndarray], float]]
) -> Dict[str, np.ndarray]:
    """``base + sum_i w_i * (part_i - base)`` -- a textbook task-vector merge.

    Claim: merge-recovery -- this is the exact generative model
    ``stemma.merge_decompose.decompose_merge`` inverts, so the mixing ratios it
    recovers can be scored against a known ground truth.
    """
    out = {k: v.astype(np.float64).copy() for k, v in base.items()}
    for part, coef in parts:
        for k in out:
            out[k] = out[k] + float(coef) * (part[k].astype(np.float64) - base[k].astype(np.float64))
    return {k: v.astype(np.float32) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# On-disk model directories
# --------------------------------------------------------------------------- #


def write_model(directory: Path, weights: Dict[str, np.ndarray], **config_extra) -> Path:
    """Write ``weights`` as ``model.safetensors`` plus a matching ``config.json``.

    Claim: infra -- the loader under test is exercised on real safetensors bytes
    (real header, real ``data_offsets``), not on a mock, so the byte accounting
    the low-transfer claim rests on is measured against a genuine file.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {k: np.ascontiguousarray(v) for k, v in weights.items()}
    save_file(tensors, str(directory / "model.safetensors"), metadata={"format": "pt"})

    vocab = int(tensors["model.embed_tokens.weight"].shape[0])
    hidden = int(tensors["model.embed_tokens.weight"].shape[1])
    n_layers = 1 + max(
        int(n.split(".")[2]) for n in tensors if n.startswith("model.layers.")
    )
    inter = int(tensors["model.layers.0.mlp.gate_proj.weight"].shape[0])
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": hidden,
        "intermediate_size": inter,
        "num_hidden_layers": n_layers,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "vocab_size": vocab,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
    }
    config.update(config_extra)
    (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return directory


@dataclass
class MergeCase:
    """Everything the merge-recovery test needs, with the ground truth attached."""

    merged: str
    base: str
    parents: Dict[str, float] = field(default_factory=dict)
    decoys: List[str] = field(default_factory=list)

    @property
    def candidates(self) -> List[str]:
        return list(self.parents) + list(self.decoys)

    @property
    def truth(self) -> Dict[str, float]:
        return dict(self.parents)


# --------------------------------------------------------------------------- #
# Fixtures. Session-scoped: the models are read-only, and regenerating them per
# test would dominate the suite's runtime.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def models_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-wide temp directory holding every generated checkpoint."""
    return tmp_path_factory.mktemp("stemma_models")


@pytest.fixture(scope="session")
def parent_weights() -> Dict[str, np.ndarray]:
    """The reference state dict every derived fixture is built from."""
    return base_weights(seed=0)


@pytest.fixture(scope="session")
def tiny_parent(models_root: Path, parent_weights: Dict[str, np.ndarray]) -> str:
    """A tiny but structurally realistic 'ancestor' checkpoint."""
    return str(write_model(models_root / "tiny_parent", parent_weights))


@pytest.fixture(scope="session")
def tiny_child_sft(models_root: Path, parent_weights: Dict[str, np.ndarray]) -> str:
    """Scar-free descendant: parent + a seeded low-rank update. The hard case."""
    w = apply_sft_delta(parent_weights, seed=11)
    return str(write_model(models_root / "tiny_child_sft", w))


@pytest.fixture(scope="session")
def tiny_sibling_sft(models_root: Path, parent_weights: Dict[str, np.ndarray]) -> str:
    """A second, independent SFT descendant -- usable as an outgroup for rooting."""
    w = apply_sft_delta(parent_weights, seed=12)
    return str(write_model(models_root / "tiny_sibling_sft", w))


@pytest.fixture(scope="session")
def tiny_child_int8(models_root: Path, parent_weights: Dict[str, np.ndarray]) -> str:
    """Descendant carrying a per-channel int8 quantisation lattice scar."""
    w = apply_int8_roundtrip(parent_weights)
    return str(write_model(models_root / "tiny_child_int8", w))


@pytest.fixture(scope="session")
def tiny_child_pruned(models_root: Path, parent_weights: Dict[str, np.ndarray]) -> str:
    """Descendant with 30% of every matrix zeroed by magnitude pruning."""
    w = apply_magnitude_pruning(parent_weights)
    return str(write_model(models_root / "tiny_child_pruned", w))


@pytest.fixture(scope="session")
def tiny_child_vocab(models_root: Path, parent_weights: Dict[str, np.ndarray]) -> str:
    """Descendant with 512 freshly initialised (untrained) vocabulary rows."""
    w = apply_sft_delta(parent_weights, seed=13, scale=0.01)
    w = apply_vocab_extension(w, seed=13)
    return str(write_model(models_root / "tiny_child_vocab", w))


@pytest.fixture(scope="session")
def tiny_merged(models_root: Path, parent_weights: Dict[str, np.ndarray]) -> MergeCase:
    """A 0.7 / 0.3 task-vector merge of two children over the parent base.

    Also materialises two decoy candidates (independent children of the same
    parent) whose true mixing coefficient is exactly zero.
    """
    base_dir = str(write_model(models_root / "merge_base", parent_weights))

    a_w = apply_sft_delta(parent_weights, seed=101)
    b_w = apply_sft_delta(parent_weights, seed=202)
    a_dir = str(write_model(models_root / "merge_src_a", a_w))
    b_dir = str(write_model(models_root / "merge_src_b", b_w))

    decoys = []
    for i, s in enumerate((303, 404)):
        d_w = apply_sft_delta(parent_weights, seed=s)
        decoys.append(str(write_model(models_root / f"merge_decoy_{i}", d_w)))

    merged_w = linear_merge(parent_weights, [(a_w, 0.7), (b_w, 0.3)])
    merged_dir = str(write_model(models_root / "merge_merged", merged_w))

    return MergeCase(
        merged=merged_dir,
        base=base_dir,
        parents={a_dir: 0.7, b_dir: 0.3},
        decoys=decoys,
    )


@pytest.fixture(scope="session")
def tiny_unrelated(models_root: Path) -> str:
    """The hard control: identical shapes and generator, a different seed.

    Claim: low-false-positive -- an independently initialised model of the same
    architecture is exactly the case a shape-only or spectrum-only fingerprint
    gets wrong, so this is the fixture the false-positive test is built on.
    """
    return str(write_model(models_root / "tiny_unrelated", base_weights(seed=98765)))


@pytest.fixture(scope="session")
def tiny_unrelated_arch(models_root: Path) -> str:
    """An unrelated model with a different architecture (shapes differ)."""
    w = base_weights(seed=4242, hidden=192, n_layers=3, vocab=1536, intermediate=384)
    return str(write_model(models_root / "tiny_unrelated_arch", w))
