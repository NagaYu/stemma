#!/usr/bin/env python3
"""Build Stemma's ground-truth, labelled lineage DAG as real safetensors models.

Claim: infra -- every other claim in this project (direction, merge-recovery,
low-transfer, low-false-positive) is measured against the labels this script
writes, so the labels have to come from operations that were *actually applied
to real weights*, never from a synthetic annotation file.

What it produces under ``--out-dir``:

* one directory per model (``model.safetensors`` + ``config.json`` + the
  tokenizer files copied from the base model), and
* ``ground_truth.json`` in exactly the schema at the end of ``CONTRACT.md``
  (keys ``models``, ``edges``, ``pairs``).

The DAG covers, over locally cached base checkpoints and with no network and no
token required:

  (i)    a real short SFT child and a real LoRA-merged child (LoRA implemented
         here -- ``peft`` is deliberately not a dependency),
  (ii)   a continued-pretraining child *of the SFT child*, giving depth >= 3,
  (iii)  INT8 and INT4 fake-quantisation round trips on two different parents,
  (iv)   global-magnitude and structured (whole-row MLP) pruned children,
  (v)    a vocabulary-extended child with genuinely untrained orphan rows, plus
         a harder variant whose new rows got a little training,
  (vi)   SLERP / TIES-2 / TIES-3 / DARE merges over a *single shared base* with
         the mixing ratios recorded,
  (vii)  distillation: the real ``gpt2 -> distilgpt2`` pair plus a depth-halved
         student of the SmolLM2 root, both flagged in the ground truth as cases
         where weight-level lineage is EXPECTED TO BE WEAK (docs/FINDINGS.md
         section 5.3 -- scored honestly, not hidden), and
  (viii) false-positive controls: cross-family pairs and, most importantly, two
         same-architecture / different-initialisation models built with
         ``AutoModelForCausalLM.from_config`` -- identical shapes, zero shared
         lineage.

A licence conflict is planted deliberately (a ``cc-by-nc-4.0`` intermediate and
a ``cc-by-sa-4.0`` sibling, both feeding ``apache-2.0`` merges) so that
``stemma.rights`` has something real to find.

Run::

    python scripts/build_bench.py                       # full, ~10 min on an M-series Mac
    python scripts/build_bench.py --quick               # shorter training
    python scripts/build_bench.py --skip-train          # ~1 minute, for CI
    python scripts/build_bench.py --quick --skip-train --limit 8 --out-dir /tmp/bench_smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from stemma.utils import atomic_write_json, human_bytes  # noqa: E402

LOG = logging.getLogger("build_bench")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Base checkpoints. All four are in the local Hugging Face cache in the dev
#: environment and resolve offline; ``--base`` overrides the primary root.
DEFAULT_BASES: Dict[str, str] = {
    "root": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "gpt2": "openai-community/gpt2",
    "distilgpt2": "distilgpt2",
    "qwen": "Qwen/Qwen2.5-0.5B-Instruct",
}

#: Files copied verbatim from the base snapshot into every generated model dir.
ASSET_FILES: Tuple[str, ...] = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "generation_config.json",
    "chat_template.jinja",
)

#: Config keys scrubbed before writing: they would leak the parent's identity
#: into the model directory and hand the detector the answer for free.
CONFIG_LEAK_KEYS: Tuple[str, ...] = ("_name_or_path", "_attn_implementation_autoset", "name_or_path")

#: LoRA hyper-parameters (r x in and out x r factors, scale = alpha / r).
LORA_RANK = 8
LORA_ALPHA = 16
LORA_TARGETS: Tuple[str, ...] = ("q_proj", "v_proj", "gate_proj", "down_proj")

#: Number of embedding rows appended by the vocabulary-extension cases.
VOCAB_EXTRA_ROWS = 512
VOCAB_INIT_STD = 0.02

#: INT4 group size along the input dimension (per-group absmax).
INT4_GROUP = 64

#: Fraction of weights removed by the two pruning cases.
MAGNITUDE_PRUNE_FRACTION = 0.30
STRUCTURED_PRUNE_FRACTION = 0.10

#: TIES trim: keep this fraction of each task vector by magnitude.
TIES_TOP_FRACTION = 0.20
#: DARE drop probability (survivors are rescaled by 1 / (1 - p)).
DARE_DROP_P = 0.50

#: Relative Frobenius size of the seeded low-rank stand-in used by --skip-train.
SKIP_TRAIN_DELTA = 0.03
SKIP_TRAIN_RANK = 8

#: A ~200-line fallback corpus so training works with *no* dataset cache at all.
#: Original prose; deliberately domain-flavoured so a few dozen optimizer steps
#: actually move the weights in a consistent direction.
EMBEDDED_CORPUS: str = """
Model provenance is the question of where a set of weights came from.
A checkpoint is a long list of numbers, and the numbers remember their history.
Fine-tuning moves weights a little; quantisation moves them onto a lattice.
Pruning sets weights to exactly zero, and exact zeros are hard to undo.
A derived model inherits the shape of its parent before it inherits anything else.
The direction of derivation is not visible to any symmetric similarity score.
Cosine similarity between two checkpoints cannot say which one came first.
Linear centred kernel alignment is symmetric, so it cannot order a pair either.
Invariant fingerprints identify a model; they do not date it.
Lossy operations are the friend of the provenance analyst.
You cannot un-quantise a tensor and recover the values you threw away.
You cannot un-prune a row and recover the weights you set to zero.
A vocabulary grows when new tokens are added, and it rarely shrinks.
Freshly initialised embedding rows look nothing like trained embedding rows.
Untrained rows have norms concentrated around a single initialisation scale.
Trained rows have heavy tails, because frequent tokens accumulate updates.
A dead neuron stays dead in every descendant of the model that killed it.
An outlier channel tends to persist and to sharpen as training continues.
Bit-identical tensors between two models are a fossil, not a coincidence.
Merging two fine-tunes of one base produces a model with two parents.
Task vectors are differences between a fine-tuned model and its base.
The sum of two task vectors is a merge, and the coefficients are the recipe.
Spherical interpolation follows the arc between two directions.
When two directions are nearly colinear, spherical interpolation degenerates.
Trimming a task vector to its largest entries removes most of the interference.
Electing a sign per coordinate resolves disagreement between parents.
Dropping half the coordinates and rescaling the rest preserves the expectation.
Distillation copies behaviour, not weights, so the weight trail goes cold.
A student with half the layers of its teacher is a different architecture.
Weight-level lineage is expected to be weak across a distillation edge.
Honest evaluation reports the weak cases alongside the strong ones.
An aggregate accuracy number lets the easy edges hide the hard ones.
Abstention is a legitimate answer when the evidence is genuinely thin.
A confident wrong answer is worse than an admission of uncertainty.
Two models with the same architecture and different seeds share nothing.
Identical shapes are not evidence of a shared ancestor.
The false-positive rate is measured on exactly those look-alike pairs.
Range requests let a client read a few rows of a tensor and nothing else.
The safetensors header describes every tensor before any payload is fetched.
Reading a header costs kilobytes; reading a checkpoint costs gigabytes.
Selective reading is what makes auditing a whole hub plausible.
An audit that requires a full download is an audit nobody runs.
A bill of materials lists the components that went into a build.
An AI bill of materials should list the checkpoints that went into a model.
Every edge in a provenance graph deserves a confidence, not a claim.
Evidence is not proof, and a statistical finding is not a legal finding.
A human has to review the graph before anyone acts on it.
Licences propagate along derivation edges whether or not anyone notices.
A non-commercial ancestor casts a long shadow over its descendants.
A share-alike ancestor asks its descendants to keep sharing alike.
An unknown licence upstream is itself a finding worth surfacing.
Rights questions are downstream of provenance questions.
You cannot reason about a licence until you know what the model is made of.
The order of layers in a transformer is a coordinate system.
Relative depth makes a twelve-layer model comparable to a twenty-four-layer one.
Roles group tensors that do the same job in different naming schemes.
Attention projections, feed-forward matrices, embeddings, and norms.
A fused query-key-value matrix is three matrices wearing one name.
Permutation of neurons leaves the function of a layer unchanged.
A fingerprint that survives permutation is a fingerprint worth keeping.
Rescaling a row and its matching column leaves the function unchanged too.
Singular values are invariant to orthogonal changes of basis.
Spectral entropy summarises how the energy is spread across directions.
Stable rank counts the directions that actually carry the signal.
The participation ratio is another way of counting effective directions.
Row norms have a distribution, and the distribution has a shape.
Skewness and kurtosis describe that shape in two numbers.
The Gini coefficient measures how unequal the row norms are.
Quantiles are robust where moments are fragile.
A benchmark is only as good as the labels it is scored against.
Labels that were generated by the method under test are worthless.
Ground truth here means an operation that was really applied to real weights.
Every derived model in this benchmark was produced by running the operation.
The recipe is recorded at the moment the recipe is executed.
Mixing ratios are written down before the merged weights are saved.
A reproducible benchmark takes a seed and returns the same graph.
Determinism is a property you have to design for, not one you get for free.
Random number generators need explicit seeds in every branch.
A build that skips training should still build the same graph shape.
Continuous integration needs a fast path that exercises the slow path's code.
Resumability turns a ten-minute build into a one-second no-op.
Progress logging is what makes a long build tolerable.
Text is the input to a language model and text is what it predicts.
Causal language modelling predicts the next token given the previous tokens.
The loss is the negative log likelihood of the observed continuation.
A few dozen optimizer steps will not make a model good.
A few dozen optimizer steps will make a model measurably different.
Measurably different is exactly what a provenance benchmark needs.
Weight decay pulls weights towards zero over the course of training.
Whether a norm grows or shrinks depends on the recipe, not on the direction.
A statistic that flips sign between families is not a usable prior.
It can still be a useful feature once it has been fitted.
Fitted weights beat hand-set signs whenever labels are available.
Labels are available here because we generated the lineage ourselves.
The outgroup is a relative that sits outside the pair under consideration.
Phylogenetics has used outgroups to root trees for a very long time.
If a third relative is closer to one of the pair, that one is closer to the root.
Rooting a tree is a different problem from building a tree.
An unrooted tree records relationships without recording ancestry.
A directed acyclic graph records ancestry and admits multiple parents.
Cycles in a provenance graph mean at least one edge is wrong.
Breaking the least confident edge on a cycle is a reasonable repair.
Confidence should come from evidence, and evidence should be inspectable.
The user interface should show the reasons, not only the verdict.
An evidence table is more useful than a single number.
Transfer accounting belongs in the report next to the accuracy.
Bytes per decision is a first-class metric for an auditing tool.
Seconds per decision matters when the universe has ten thousand models.
An index over fingerprints turns a quadratic search into a lookup.
Approximate nearest neighbours are enough to shortlist candidate parents.
Exact scoring can then be reserved for the shortlist.
The shortlist is where the expensive tensor reads happen.
Most pairs in a large universe are unrelated and can be dismissed cheaply.
Dismissing unrelated pairs cheaply is most of the engineering.
Numerical work should be done in float32 or float64, not in bfloat16.
Storage precision and compute precision are different decisions.
A bfloat16 checkpoint has about three significant decimal digits.
A float16 checkpoint has about four significant decimal digits.
An eight-bit integer lattice has two hundred and fifty six levels.
A four-bit integer lattice has sixteen levels per group.
Group-wise scales make four-bit quantisation survivable.
The scale is recorded per group and the residual is the scar.
Counting distinct values in a row detects a low-bit lattice immediately.
Fitting a step size to a row detects it even after a dtype conversion.
Precision goes down over time and essentially never goes back up.
That asymmetry is a direction signal all by itself.
Zero sets grow under pruning and never shrink under fine-tuning.
A superset relation between zero sets is therefore directional.
Structured pruning removes whole rows and leaves an obvious hole.
Unstructured pruning scatters zeros and leaves a subtler one.
Sparse checkpoints compress well, which is often the point.
Compression and provenance interact in ways nobody has mapped yet.
A merge of merges is a real thing that people publish.
Depth in a provenance graph accumulates faster than anyone expects.
Three generations is enough to lose track of the original licence.
Model cards are written by people and people forget things.
Weights do not forget, which is the entire premise of this project.
The premise is testable, and this benchmark is the test.
Some derivations will be recovered and some will not.
Reporting both is the difference between a tool and a demonstration.
A tool that overclaims will be trusted once and then discarded.
Calibration is worth more than headline accuracy.
The threshold that trades false positives for false negatives is a choice.
That choice belongs to the auditor, not to the library.
Exposing the threshold is therefore part of the design.
Defaults should be conservative because the cost of a false accusation is high.
Nobody wants to be told their model was derived from something it was not.
The disclaimer is not boilerplate; it is the honest summary of the method.
Statistical evidence about weight-level similarity is what this produces.
It does not establish provenance as fact.
It does not constitute a legal determination of anything.
A human must review before any action is taken.
That sentence is printed on every report the tool emits.
Benchmarks age badly when the models they use disappear.
Local generation keeps the benchmark alive without a network.
The base checkpoints are small enough to keep on a laptop.
A hundred and thirty five million parameters fits comfortably in memory.
Half a gigabyte per checkpoint adds up across twenty checkpoints.
Disk is cheaper than a download, and a download is cheaper than a mistake.
The build writes each model once and skips it on the next run.
Forcing a rebuild is an explicit flag, never the default.
An interrupted build should be resumable from where it stopped.
Atomic writes keep partial results from being mistaken for finished ones.
The ground truth file is written last, after every model exists.
Schema validation runs before the file is considered complete.
A malformed label file would poison every number downstream.
Validation is cheap insurance against a very expensive mistake.
The pairs list is derived from the edges, not written by hand.
Transitive closure turns a parent edge into an ancestor relation.
Siblings share an ancestor without descending from one another.
Unrelated pairs come from disjoint components of the graph.
Balance between related and unrelated pairs keeps the metric honest.
An unbalanced test set flatters whichever answer is more common.
The most valuable negative pair is the one that looks most positive.
Same architecture, same shapes, different seed, no shared history.
If the method survives that pair it has earned some trust.
If it does not, the honest thing to do is to say so in the report.
Reporting failures is how a research prototype becomes a method.
Methods that only report successes do not survive contact with reality.
Reality here is a hub with a million checkpoints and no reliable metadata.
Metadata is missing, wrong, or copied from whatever was fashionable.
Weights are the only artefact that cannot lie about itself.
Reading them carefully is the whole idea.
Careful reading means reading the right bytes, not all of them.
The right bytes are chosen by role and by depth.
Two tensors per role and depth bucket is enough for a fingerprint.
A few hundred rows per tensor is enough for a spectrum.
Subsampling rows preserves the singular value profile surprisingly well.
Randomised singular value decomposition makes the spectrum cheap.
A seed makes the randomised decomposition reproducible.
Reproducible numbers are the ones you can argue about productively.
This corpus exists so that the benchmark can be built without a dataset.
It is short, original, and repetitive on purpose.
Repetition gives a short training run something to latch onto.
The resulting model is not useful, and it is not supposed to be.
It is supposed to be a genuine descendant of its parent.
That is the only property the benchmark requires of it.
Everything else about it is irrelevant to the measurement.
The measurement is whether the lineage can be recovered from weights.
That is the question this project exists to answer.
""".strip()


# --------------------------------------------------------------------------- #
# Environment shims
# --------------------------------------------------------------------------- #


def import_transformers() -> Any:
    """Import ``transformers``, tolerating a too-new ``huggingface_hub``.

    Claim: infra -- the dev environment ships transformers 4.57 next to
    huggingface_hub 1.x, whose pinned version check aborts ``import
    transformers`` outright. The check is advisory (the APIs this script uses --
    ``AutoConfig``/``AutoTokenizer``/``AutoModelForCausalLM`` -- all work), so we
    neutralise just that module rather than pinning the user's environment.
    """
    try:
        import transformers  # noqa: PLC0415

        return transformers
    except ImportError as exc:
        if "transformers.dependency_versions_check" in sys.modules:
            raise
        LOG.warning("transformers import failed (%s); installing version-check shim", exc)
        shim = types.ModuleType("transformers.dependency_versions_check")
        shim.dep_version_check = lambda *a, **k: None  # type: ignore[attr-defined]
        shim.__file__ = "<stemma-build-bench-shim>"
        for key in [k for k in list(sys.modules) if k == "transformers" or k.startswith("transformers.")]:
            del sys.modules[key]
        sys.modules["transformers.dependency_versions_check"] = shim
        import transformers  # noqa: PLC0415

        return transformers


def pick_device(requested: str = "auto") -> str:
    """Choose the torch device for the short training runs.

    Claim: infra -- MPS keeps the default build inside its ~10 minute budget on
    an M-series Mac; CPU is always a working fallback and CUDA is never required.
    """
    if requested and requested != "auto":
        return requested
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # pragma: no cover - platform dependent
        pass
    if torch.cuda.is_available():  # pragma: no cover - not the dev machine
        return "cuda"
    return "cpu"


def resolve_base(ref: str) -> Path:
    """Resolve a base model reference to a local snapshot directory.

    Claim: infra -- the benchmark must build with no network and no token, so
    local paths and cached Hub snapshots are both accepted and a missing cache
    produces one clear error instead of a stack of HTTP failures.
    """
    p = Path(ref).expanduser()
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    patterns = ["*.json", "*.txt", "*.model", "*.jinja", "model.safetensors"]
    try:
        return Path(snapshot_download(ref, local_files_only=True, allow_patterns=patterns))
    except Exception as exc:
        LOG.warning("%s not in local cache (%s); trying the network", ref, type(exc).__name__)
        try:
            return Path(snapshot_download(ref, allow_patterns=patterns))
        except Exception as exc2:  # pragma: no cover - depends on connectivity
            raise SystemExit(
                f"cannot resolve base model {ref!r}: not a local directory, not in the "
                f"Hugging Face cache, and the download failed ({exc2}). Pass --base with a "
                f"local path, or pre-populate the cache."
            ) from exc2


# --------------------------------------------------------------------------- #
# Small IO helpers
# --------------------------------------------------------------------------- #


def load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    """Read a model directory's ``model.safetensors`` into a torch state dict.

    Claim: infra.
    """
    from safetensors.torch import load_file  # noqa: PLC0415

    return load_file(str(Path(path) / "model.safetensors"))


def read_config(path: Path) -> Dict[str, Any]:
    """Read a model directory's ``config.json``.

    Claim: infra.
    """
    with open(Path(path) / "config.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def scrub_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Remove keys that would leak the parent's identity into a child's config.

    Claim: low-false-positive -- ``_name_or_path`` names the parent directory.
    Leaving it in place would let a detector "recover" lineage from a string
    instead of from weights, which is exactly the failure this benchmark exists
    to rule out.
    """
    out = dict(cfg)
    for key in CONFIG_LEAK_KEYS:
        out.pop(key, None)
    return out


def copy_assets(src: Path, dst: Path) -> List[str]:
    """Copy tokenizer/generation files from a base snapshot into a model dir.

    Claim: infra -- every generated directory must be loadable by
    ``AutoTokenizer``/``AutoModelForCausalLM`` on its own.
    """
    copied: List[str] = []
    for name in ASSET_FILES:
        s = src / name
        if s.is_file():
            shutil.copyfile(s, dst / name)
            copied.append(name)
    return copied


def write_model_dir(
    out: Path,
    state: Dict[str, torch.Tensor],
    config: Dict[str, Any],
    asset_src: Path,
    *,
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, Any]:
    """Write one complete model directory (weights + config + tokenizer files).

    Claim: infra -- the benchmark reads these directories through the very same
    ``SafeTensorsSource`` path it uses for Hub repos, so they must be real
    safetensors checkpoints, not pickles or npz files.
    """
    from safetensors.torch import save_file  # noqa: PLC0415

    out.mkdir(parents=True, exist_ok=True)
    clean: Dict[str, torch.Tensor] = {}
    n_params = 0
    for k, v in state.items():
        t = v.detach().cpu()
        if dtype is not None and t.is_floating_point():
            t = t.to(dtype)
        clean[k] = t.clone().contiguous()
        n_params += int(t.numel())
    save_file(clean, str(out / "model.safetensors"), metadata={"format": "pt"})

    cfg = scrub_config(config)
    if dtype is not None:
        cfg["torch_dtype"] = str(dtype).replace("torch.", "")
    with open(out / "config.json", "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, sort_keys=False)
    copy_assets(asset_src, out)

    size = (out / "model.safetensors").stat().st_size
    dtypes = sorted({str(t.dtype).replace("torch.", "") for t in clean.values()})
    return {"n_params": n_params, "bytes": int(size), "dtypes": dtypes, "n_tensors": len(clean)}


def is_built(path: Path) -> bool:
    """True when a model directory already holds a finished checkpoint.

    Claim: infra -- makes the build resumable, which matters because the full
    run takes minutes and is routinely interrupted.
    """
    return (path / "model.safetensors").is_file() and (path / "config.json").is_file()


def layer_of(name: str) -> Optional[int]:
    """Transformer block index encoded in a tensor name, or ``None``.

    Claim: infra.
    """
    m = re.search(r"\.(?:layers|h|blocks|layer)\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def is_matrix(t: torch.Tensor, min_dim: int = 32) -> bool:
    """True for the 2D float tensors the weight-surgery operations act on.

    Claim: infra -- norms, biases and attention masks are deliberately excluded
    so that "touched tensors" in the ground truth means what it says.
    """
    return t.dim() == 2 and t.is_floating_point() and min(t.shape) >= min_dim


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #


def load_corpus(max_lines: int = 6000) -> Tuple[List[str], str]:
    """Return training lines and the name of the source they came from.

    Claim: infra -- wikitext-2-raw-v1 loads from the local datasets cache; the
    embedded fallback keeps the whole benchmark buildable on a machine with no
    dataset cache and no network at all.
    """
    for repo in ("wikitext", "Salesforce/wikitext"):
        try:
            from datasets import load_dataset  # noqa: PLC0415

            ds = load_dataset(repo, "wikitext-2-raw-v1", split="train")
            lines = [t.strip() for t in ds["text"][: max_lines * 4] if len(t.strip()) > 80]
            if len(lines) >= 200:
                LOG.info("corpus: %s/wikitext-2-raw-v1, %d usable lines", repo, len(lines))
                return lines[:max_lines], f"{repo}:wikitext-2-raw-v1"
        except Exception as exc:
            LOG.debug("dataset %s unavailable: %s", repo, exc)
    lines = [ln.strip() for ln in EMBEDDED_CORPUS.splitlines() if ln.strip()]
    LOG.warning("wikitext-2-raw-v1 unavailable; falling back to the embedded %d-line corpus", len(lines))
    return lines, "embedded"


def corpus_slice(lines: Sequence[str], part: int, n_parts: int) -> List[str]:
    """Take a disjoint slice of the corpus so two children see different text.

    Claim: direction -- the continued-pretraining child must be trained on
    genuinely different text from the SFT child, otherwise the two task vectors
    would be near-duplicates and the merge decomposition would be ill-posed.
    """
    n = len(lines)
    lo = (n * part) // n_parts
    hi = (n * (part + 1)) // n_parts
    chunk = list(lines[lo:hi])
    return chunk if chunk else list(lines)


def make_batches(
    tokenizer: Any,
    lines: Sequence[str],
    *,
    seq_len: int,
    batch_size: int,
    n_batches: int,
    seed: int,
) -> List[torch.Tensor]:
    """Tokenise text into fixed-length causal-LM batches.

    Claim: infra -- a deterministic batch list keeps ``--seed`` meaningful all
    the way down to the individual optimizer step.
    """
    need_tokens = max(n_batches * batch_size * seq_len * 2, seq_len * batch_size * 4)
    text_parts: List[str] = []
    approx = 0
    idx = 0
    while approx < need_tokens * 5 and lines:
        ln = lines[idx % len(lines)]
        text_parts.append(ln)
        approx += len(ln)
        idx += 1
        if idx > 200_000:
            break
    text = "\n\n".join(text_parts)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    blocks = [ids[i : i + seq_len] for i in range(0, max(len(ids) - seq_len, 1), seq_len)]
    blocks = [b for b in blocks if len(b) == seq_len]
    if not blocks:
        pad = (ids + ids)[:seq_len] if ids else [0] * seq_len
        blocks = [pad]
    rng = random.Random(seed)
    rng.shuffle(blocks)
    batches: List[torch.Tensor] = []
    cursor = 0
    for _ in range(n_batches):
        rows = []
        for _ in range(batch_size):
            rows.append(blocks[cursor % len(blocks)])
            cursor += 1
        batches.append(torch.tensor(rows, dtype=torch.long))
    return batches


# --------------------------------------------------------------------------- #
# Real training
# --------------------------------------------------------------------------- #


def train_causal_lm(
    model: Any,
    batches: Sequence[torch.Tensor],
    *,
    device: str,
    lr: float,
    tag: str,
    grad_filter: Optional[Callable[[Any], None]] = None,
) -> Dict[str, float]:
    """Run a short real causal-LM fine-tune and return loss statistics.

    Claim: direction -- an SFT/continued-pretraining edge is the *hard* case in
    docs/FINDINGS.md (no lossy scar), so the benchmark's copy of it has to be a
    real optimizer trajectory rather than an injected perturbation.
    """
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    first = last = float("nan")
    t0 = time.time()
    for i, batch in enumerate(batches):
        ids = batch.to(device)
        out = model(input_ids=ids, labels=ids)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        if grad_filter is not None:
            grad_filter(model)
        opt.step()
        opt.zero_grad(set_to_none=True)
        val = float(loss.detach().float().cpu())
        if i == 0:
            first = val
        last = val
        if i % 10 == 0:
            LOG.info("    %s step %d/%d loss=%.4f", tag, i, len(batches), val)
    opt.zero_grad(set_to_none=True)
    model.eval()
    return {
        "steps": float(len(batches)),
        "loss_first": first,
        "loss_last": last,
        "train_seconds": round(time.time() - t0, 2),
    }


class LoRALinear(torch.nn.Module):
    """A frozen ``nn.Linear`` plus a trainable rank-``r`` update ``B @ A``.

    Claim: direction -- ``peft`` is not installed, and a merged LoRA child is a
    distinct relation type from a full fine-tune (its task vector is genuinely
    low rank), so the adapter is implemented here and merged explicitly.
    """

    def __init__(self, base: torch.nn.Linear, *, r: int, alpha: int, generator: torch.Generator) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        out_f, in_f = base.weight.shape
        self.r = int(r)
        self.scale = float(alpha) / float(r)
        a = torch.randn(self.r, in_f, generator=generator, dtype=torch.float32) / math.sqrt(in_f)
        self.lora_A = torch.nn.Parameter(a.to(base.weight.dtype))
        self.lora_B = torch.nn.Parameter(torch.zeros(out_f, self.r, dtype=base.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102 - see class docstring
        return self.base(x) + self.scale * torch.nn.functional.linear(
            torch.nn.functional.linear(x, self.lora_A), self.lora_B
        )

    def merged_weight(self) -> torch.Tensor:
        """Return ``W + (alpha/r) * B @ A`` -- the merged child's weight.

        Claim: direction.
        """
        delta = self.scale * (self.lora_B.detach().float() @ self.lora_A.detach().float())
        return (self.base.weight.detach().float() + delta).to(self.base.weight.dtype)


def inject_lora(model: Any, *, targets: Sequence[str], r: int, alpha: int, seed: int) -> List[str]:
    """Wrap the target ``nn.Linear`` layers with :class:`LoRALinear`.

    Claim: direction -- returns the parameter names that the adapter will move,
    which the ground truth records as ``touched_tensors``.
    """
    gen = torch.Generator().manual_seed(seed)
    touched: List[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, torch.nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in targets:
            continue
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, LoRALinear(module, r=r, alpha=alpha, generator=gen))
        touched.append(f"{name}.weight")
    for p in model.parameters():
        p.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)
    return touched


def merge_lora(model: Any) -> None:
    """Fold every :class:`LoRALinear` back into a plain ``nn.Linear``.

    Claim: direction -- the published artefact of a LoRA run is the *merged*
    checkpoint, so that is what the benchmark stores.
    """
    for name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue
        merged = module.merged_weight()
        base = module.base
        base.weight.data.copy_(merged)
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        leaf = name.rsplit(".", 1)[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, base)
    for p in model.parameters():
        p.requires_grad_(True)


# --------------------------------------------------------------------------- #
# Weight surgery: the lossy, directional operations
# --------------------------------------------------------------------------- #


def lowrank_perturb(
    state: Dict[str, torch.Tensor],
    *,
    keys: Sequence[str],
    seed: int,
    scale: float = SKIP_TRAIN_DELTA,
    rank: int = SKIP_TRAIN_RANK,
) -> List[str]:
    """Add a seeded low-rank delta to selected tensors (the ``--skip-train`` path).

    Claim: infra -- stands in for a real fine-tune so CI can build the full DAG
    in about a minute. It is a *shape-preserving stand-in only*: the ground
    truth records ``training="synthetic"`` so no result computed from a
    ``--skip-train`` build can be reported as a fine-tuning measurement.
    """
    gen = torch.Generator().manual_seed(int(seed))
    touched: List[str] = []
    for k in keys:
        w = state.get(k)
        if w is None or not is_matrix(w):
            continue
        w32 = w.float()
        m, n = w32.shape
        u = torch.randn(m, rank, generator=gen)
        v = torch.randn(rank, n, generator=gen)
        d = u @ v
        dn = torch.linalg.norm(d)
        wn = torch.linalg.norm(w32)
        if float(dn) <= 0 or float(wn) <= 0:
            continue
        d = d * (scale * wn / dn)
        state[k] = (w32 + d).to(w.dtype)
        touched.append(k)
    return touched


def fake_quantize_int8(w: torch.Tensor, *, channel_axis: int = 0) -> torch.Tensor:
    """Per-output-channel absmax INT8 round trip (quantise then dequantise).

    Claim: direction -- quantisation is irreversible, so the value lattice it
    leaves behind can only ever appear on the downstream side of an edge. This
    is the strongest single direction signal in docs/FINDINGS.md table 1.
    """
    w32 = w.float()
    reduce_dim = 1 - channel_axis
    absmax = w32.abs().amax(dim=reduce_dim, keepdim=True).clamp_min(1e-12)
    scale = absmax / 127.0
    q = torch.round(w32 / scale).clamp_(-127, 127)
    return q * scale


def fake_quantize_int4(w: torch.Tensor, *, channel_axis: int = 0, group: int = INT4_GROUP) -> torch.Tensor:
    """Group-wise (absmax, group=64) INT4 round trip along the input dimension.

    Claim: direction -- sixteen levels per group is coarse enough that a row's
    distinct-value count alone identifies the child side of the edge.
    """
    w32 = w.float()
    transposed = channel_axis == 1
    if transposed:
        w32 = w32.t().contiguous()
    out_f, in_f = w32.shape
    g = min(group, in_f)
    pad = (-in_f) % g
    if pad:
        w32 = torch.cat([w32, torch.zeros(out_f, pad)], dim=1)
    blocks = w32.view(out_f, -1, g)
    absmax = blocks.abs().amax(dim=2, keepdim=True).clamp_min(1e-12)
    scale = absmax / 7.0
    q = torch.round(blocks / scale).clamp_(-7, 7)
    deq = (q * scale).view(out_f, -1)
    if pad:
        deq = deq[:, :in_f]
    if transposed:
        deq = deq.t().contiguous()
    return deq


def quantize_state(
    state: Dict[str, torch.Tensor], *, bits: int, conv1d: bool
) -> List[str]:
    """Apply an INT8 or INT4 fake-quantisation round trip in place.

    Claim: direction.
    """
    axis = 1 if conv1d else 0
    touched: List[str] = []
    for k, v in list(state.items()):
        if not is_matrix(v):
            continue
        ax = axis if (conv1d and _is_conv1d_name(k)) else 0
        if bits == 8:
            state[k] = fake_quantize_int8(v, channel_axis=ax).to(v.dtype)
        elif bits == 4:
            state[k] = fake_quantize_int4(v, channel_axis=ax).to(v.dtype)
        else:  # pragma: no cover - guarded by the caller
            raise ValueError(f"unsupported bit width {bits}")
        touched.append(k)
    return touched


def _is_conv1d_name(name: str) -> bool:
    """True for GPT-2 ``Conv1D`` weights, which are stored ``(in, out)``.

    Claim: infra -- getting the channel axis wrong would put the quantisation
    lattice on the wrong side of the matrix and weaken a signal the benchmark is
    supposed to be measuring.
    """
    return bool(re.search(r"\.(c_attn|c_proj|c_fc)\.weight$", name))


def magnitude_prune(
    state: Dict[str, torch.Tensor], *, fraction: float, seed: int, samples_per_tensor: int = 400_000
) -> Tuple[List[str], float, float]:
    """Global unstructured magnitude pruning; leaves exact zeros.

    Claim: direction -- a zero set can grow but essentially never shrinks, so
    "A's zeros are a subset of B's" is a directional statement. The global
    threshold is estimated from a seeded subsample (exact sorting of ~10^8
    values would dominate the build time); the sampling error on the 30%
    quantile is far below the sparsity resolution the benchmark reports.
    """
    gen = torch.Generator().manual_seed(int(seed))
    candidates = [k for k, v in state.items() if is_matrix(v) and not _is_embedding_name(k)]
    pool: List[torch.Tensor] = []
    for k in candidates:
        flat = state[k].float().abs().reshape(-1)
        if flat.numel() > samples_per_tensor:
            idx = torch.randint(0, flat.numel(), (samples_per_tensor,), generator=gen)
            flat = flat[idx]
        pool.append(flat)
    if not pool:
        return [], 0.0, 0.0
    allvals = torch.cat(pool)
    # NOTE: torch.quantile() refuses inputs beyond ~2^24 elements
    # ("quantile() input tensor is too large"). Even after the per-tensor
    # subsampling above, a 30-layer model contributes ~10^2 tensors x 4*10^5
    # samples = 4*10^7 values, which trips that limit. numpy has no such cap and
    # computes the same statistic, so the pool goes through np.quantile.
    thr = float(np.quantile(allvals.double().cpu().numpy(), float(fraction)))
    touched: List[str] = []
    zeroed = 0
    total = 0
    for k in candidates:
        w = state[k]
        mask = w.float().abs() >= thr
        state[k] = (w.float() * mask).to(w.dtype)
        zeroed += int((~mask).sum())
        total += int(mask.numel())
        touched.append(k)
    return touched, thr, (zeroed / total if total else 0.0)


def _is_embedding_name(name: str) -> bool:
    """True for embedding / output-head matrices.

    Claim: infra.
    """
    return bool(re.search(r"(embed_tokens|\bwte\b|\bwpe\b|lm_head)", name))


def structured_prune_mlp(
    state: Dict[str, torch.Tensor], *, fraction: float
) -> Tuple[List[str], int]:
    """Zero the lowest-norm whole rows of every MLP block (structured pruning).

    Claim: direction -- whole-row removal creates dead-neuron fossils that
    persist in every descendant, evidence family (d) in the contract.
    """
    touched: List[str] = []
    n_rows = 0
    groups: Dict[str, Dict[str, str]] = {}
    for k in state:
        m = re.match(r"^(.*)\.mlp\.(gate_proj|up_proj|down_proj|c_fc|c_proj)\.weight$", k)
        if m:
            groups.setdefault(m.group(1), {})[m.group(2)] = k
    for prefix, names in sorted(groups.items()):
        if "gate_proj" in names or "up_proj" in names:
            ins = [names[n] for n in ("gate_proj", "up_proj") if n in names]
            out_name = names.get("down_proj")
            inter = state[ins[0]].shape[0]
            score = torch.zeros(inter, dtype=torch.float32)
            for n in ins:
                score += state[n].float().pow(2).sum(dim=1)
            n_cut = max(1, int(round(inter * fraction)))
            victims = torch.argsort(score)[:n_cut]
            for n in ins:
                w = state[n].float()
                w[victims, :] = 0.0
                state[n] = w.to(state[n].dtype)
                touched.append(n)
            if out_name is not None:
                w = state[out_name].float()
                w[:, victims] = 0.0
                state[out_name] = w.to(state[out_name].dtype)
                touched.append(out_name)
            n_rows += int(n_cut)
        elif "c_fc" in names:  # GPT-2 Conv1D: weights are (in, out)
            in_name = names["c_fc"]
            out_name = names.get("c_proj")
            inter = state[in_name].shape[1]
            score = state[in_name].float().pow(2).sum(dim=0)
            n_cut = max(1, int(round(inter * fraction)))
            victims = torch.argsort(score)[:n_cut]
            w = state[in_name].float()
            w[:, victims] = 0.0
            state[in_name] = w.to(state[in_name].dtype)
            touched.append(in_name)
            if out_name is not None:
                w = state[out_name].float()
                w[victims, :] = 0.0
                state[out_name] = w.to(state[out_name].dtype)
                touched.append(out_name)
            n_rows += int(n_cut)
    return sorted(set(touched)), n_rows


def extend_vocab(
    state: Dict[str, torch.Tensor], config: Dict[str, Any], *, extra: int, seed: int, std: float = VOCAB_INIT_STD
) -> Tuple[List[str], int]:
    """Append freshly initialised embedding rows and grow ``config.vocab_size``.

    Claim: direction -- vocabularies only grow, and untrained rows are
    statistically distinguishable from trained ones (evidence family (b)). The
    new rows are drawn from N(0, 0.02) and left untrained, which is exactly what
    an un-retrained ``resize_token_embeddings`` produces.
    """
    gen = torch.Generator().manual_seed(int(seed))
    touched: List[str] = []
    old_vocab = int(config.get("vocab_size", 0))
    for key in list(state):
        if not re.search(r"(embed_tokens\.weight$|\bwte\.weight$|lm_head\.weight$)", key):
            continue
        w = state[key]
        if w.dim() != 2 or w.shape[0] != old_vocab:
            continue
        new = torch.randn(extra, w.shape[1], generator=gen, dtype=torch.float32) * std
        state[key] = torch.cat([w.float(), new], dim=0).to(w.dtype)
        touched.append(key)
    config["vocab_size"] = old_vocab + extra
    return touched, old_vocab


def halve_depth(state: Dict[str, torch.Tensor], config: Dict[str, Any]) -> Tuple[Dict[str, torch.Tensor], int]:
    """Build a student state dict from every other layer of the teacher.

    Claim: direction -- this is the DistilBERT recipe; the resulting student has
    a *different architecture* from its teacher, which is precisely the case
    docs/FINDINGS.md flags as one where weight-level lineage is expected to be
    weak.
    """
    n_layers = int(config.get("num_hidden_layers") or config.get("n_layer") or 0)
    keep = list(range(0, n_layers, 2))
    remap = {src: dst for dst, src in enumerate(keep)}
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        li = layer_of(k)
        if li is None:
            out[k] = v
            continue
        if li in remap:
            out[re.sub(r"\.(\d+)\.", f".{remap[li]}.", k, count=1)] = v
    return out, len(keep)


# --------------------------------------------------------------------------- #
# Merges
# --------------------------------------------------------------------------- #


def _mergeable_keys(base: Dict[str, torch.Tensor], parents: Sequence[Dict[str, torch.Tensor]]) -> List[str]:
    """Float tensors present with identical shape in the base and all parents.

    Claim: merge-recovery.
    """
    keys = []
    for k, v in base.items():
        if not v.is_floating_point():
            continue
        if all(k in p and p[k].shape == v.shape for p in parents):
            keys.append(k)
    return keys


def merge_slerp(
    base: Dict[str, torch.Tensor],
    parents: Sequence[Dict[str, torch.Tensor]],
    weights: Sequence[float],
    *,
    eps: float = 5e-4,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Per-tensor spherical interpolation of two task vectors, lerp near colinear.

    Claim: merge-recovery -- SLERP is the merge whose effective coefficients are
    *not* the recipe ratios, so it tests whether the decomposer recovers the
    realised mixture rather than parroting a label.
    """
    if len(parents) != 2:
        raise ValueError("SLERP takes exactly two parents")
    t = float(weights[1]) / float(weights[0] + weights[1])
    out = dict(base)
    keys = _mergeable_keys(base, parents)
    n_lerp = 0
    for k in keys:
        b = base[k].float()
        d0 = parents[0][k].float() - b
        d1 = parents[1][k].float() - b
        f0, f1 = d0.reshape(-1), d1.reshape(-1)
        n0 = float(torch.linalg.norm(f0))
        n1 = float(torch.linalg.norm(f1))
        if n0 < 1e-12 or n1 < 1e-12:
            merged = (1.0 - t) * d0 + t * d1
            n_lerp += 1
        else:
            cos = float(torch.dot(f0, f1) / (n0 * n1))
            cos = max(-1.0, min(1.0, cos))
            if abs(cos) > 1.0 - eps:
                merged = (1.0 - t) * d0 + t * d1
                n_lerp += 1
            else:
                omega = math.acos(cos)
                s = math.sin(omega)
                merged = (math.sin((1.0 - t) * omega) / s) * d0 + (math.sin(t * omega) / s) * d1
        out[k] = (b + merged).to(base[k].dtype)
    return out, {"method": "slerp", "t": t, "tensors": len(keys), "lerp_fallbacks": n_lerp}


def merge_ties(
    base: Dict[str, torch.Tensor],
    parents: Sequence[Dict[str, torch.Tensor]],
    weights: Sequence[float],
    *,
    top_fraction: float = TIES_TOP_FRACTION,
    seed: int = 0,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """TIES merge: trim to top-k% magnitude, elect a sign, disjoint weighted mean.

    Claim: merge-recovery -- trimming makes the parents' contributions nearly
    disjoint per coordinate, which is what a naive correlation-based attribution
    gets wrong and a non-negative least-squares decomposition gets right.
    """
    gen = torch.Generator().manual_seed(int(seed))
    out = dict(base)
    keys = _mergeable_keys(base, parents)
    w = torch.tensor([float(x) for x in weights], dtype=torch.float32)
    for k in keys:
        b = base[k].float()
        taus = [p[k].float() - b for p in parents]
        trimmed = []
        for tau in taus:
            flat = tau.reshape(-1).abs()
            if flat.numel() > 1_000_000:
                idx = torch.randint(0, flat.numel(), (1_000_000,), generator=gen)
                sample = flat[idx]
            else:
                sample = flat
            thr = float(torch.quantile(sample.double(), 1.0 - top_fraction)) if sample.numel() else 0.0
            trimmed.append(tau * (tau.abs() >= thr))
        stack = torch.stack(trimmed, dim=0)
        wv = w.view(-1, *([1] * (stack.dim() - 1)))
        elected = torch.sign((stack * wv).sum(dim=0))
        agree = (torch.sign(stack) == elected) & (stack != 0)
        count = agree.sum(dim=0).clamp_min(1).float()
        merged = (stack * wv * agree.float()).sum(dim=0) / count
        out[k] = (b + merged).to(base[k].dtype)
    return out, {"method": "ties", "top_fraction": top_fraction, "tensors": len(keys)}


def merge_dare(
    base: Dict[str, torch.Tensor],
    parents: Sequence[Dict[str, torch.Tensor]],
    weights: Sequence[float],
    *,
    p: float = DARE_DROP_P,
    seed: int = 0,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """DARE merge: drop task-vector entries with probability p, rescale, sum.

    Claim: merge-recovery -- the 1/(1-p) rescale keeps the expectation of each
    task vector intact, so the recorded ratios remain the right answer even
    though half of every parent's coordinates are gone.
    """
    gen = torch.Generator().manual_seed(int(seed))
    out = dict(base)
    keys = _mergeable_keys(base, parents)
    keep = 1.0 - float(p)
    for k in keys:
        b = base[k].float()
        acc = torch.zeros_like(b)
        for wi, parent in zip(weights, parents):
            tau = parent[k].float() - b
            mask = (torch.rand(tau.shape, generator=gen) < keep).float()
            acc += float(wi) * tau * mask / keep
        out[k] = (b + acc).to(base[k].dtype)
    return out, {"method": "dare", "drop_p": float(p), "tensors": len(keys)}


# --------------------------------------------------------------------------- #
# Build context and model specifications
# --------------------------------------------------------------------------- #


@dataclass
class BuildContext:
    """Everything a builder needs: paths, budgets, corpus and RNG seed.

    Claim: infra.
    """

    out_dir: Path
    seed: int
    device: str
    quick: bool
    skip_train: bool
    force: bool
    base_refs: Dict[str, str]
    base_paths: Dict[str, Path] = field(default_factory=dict)
    corpus: List[str] = field(default_factory=list)
    corpus_source: str = "none"
    seq_len: int = 256
    batch_size: int = 4
    steps: Dict[str, int] = field(default_factory=dict)
    _tokenizers: Dict[str, Any] = field(default_factory=dict)

    def path(self, model_id: str) -> Path:
        """Directory a model id is written to.

        Claim: infra.
        """
        return self.out_dir / model_id

    def tokenizer(self, base_key: str = "root") -> Any:
        """Cached tokenizer for one of the base models.

        Claim: infra.
        """
        if base_key not in self._tokenizers:
            tf = import_transformers()
            self._tokenizers[base_key] = tf.AutoTokenizer.from_pretrained(str(self.base_paths[base_key]))
        return self._tokenizers[base_key]

    def load_model(self, path: Path) -> Any:
        """Load a generated (or base) model directory in float32 on ``device``.

        Claim: infra -- training happens in float32 and is cast back to the
        parent's storage dtype on save, so a short run's delta is not rounded
        away by bfloat16 storage.
        """
        tf = import_transformers()
        model = tf.AutoModelForCausalLM.from_pretrained(str(path), dtype=torch.float32)
        return model.to(self.device)


@dataclass
class ModelSpec:
    """One node of the ground-truth DAG plus the callable that materialises it.

    Claim: infra -- id, parents, relation, mixing ratios and licence are all
    declared next to the code that actually performs the operation, so the label
    cannot drift away from the artefact.
    """

    id: str
    family: str
    op: str
    relation: str
    parents: List[str]
    weights: Dict[str, float]
    license: str
    build: Callable[["BuildContext", "ModelSpec"], Dict[str, Any]]
    weak_weight_lineage: bool = False
    notes: str = ""


def save_trained(
    ctx: BuildContext, model: Any, out: Path, *, asset_src: Path, dtype: torch.dtype
) -> Dict[str, Any]:
    """Persist a trained ``transformers`` model as a scrubbed model directory.

    Claim: infra.
    """
    out.mkdir(parents=True, exist_ok=True)
    model = model.to("cpu").to(dtype)
    model.save_pretrained(str(out), safe_serialization=True)
    cfg = scrub_config(read_config(out))
    with open(out / "config.json", "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    copy_assets(asset_src, out)
    size = (out / "model.safetensors").stat().st_size
    n_params = sum(int(p.numel()) for p in model.parameters())
    return {"n_params": n_params, "bytes": int(size), "n_tensors": len(list(model.state_dict()))}


def dtype_of_dir(path: Path) -> torch.dtype:
    """Storage dtype of an existing checkpoint (first floating tensor wins).

    Claim: infra.
    """
    sd = load_state_dict(path)
    for v in sd.values():
        if v.is_floating_point():
            return v.dtype
    return torch.float32


# --- individual builders ---------------------------------------------------- #


def _build_base_copy(base_key: str) -> Callable[[BuildContext, ModelSpec], Dict[str, Any]]:
    """Return a builder that copies a cached base checkpoint into the bench dir.

    Claim: infra -- the roots live inside ``--out-dir`` so the benchmark reads
    every node, root or derived, through one identical local code path.
    """

    def _build(ctx: BuildContext, spec: ModelSpec) -> Dict[str, Any]:
        src = ctx.base_paths[base_key]
        state = load_state_dict(src)
        cfg = read_config(src)
        info = write_model_dir(ctx.path(spec.id), state, cfg, src)
        info["source_repo"] = ctx.base_refs[base_key]
        info["arch"] = (cfg.get("architectures") or ["unknown"])[0]
        info["touched_tensors"] = []
        return info

    return _build


def _build_finetune(
    *, parent_id: str, corpus_part: int, steps_key: str, lr: float, base_key: str = "root"
) -> Callable[[BuildContext, ModelSpec], Dict[str, Any]]:
    """Return a builder for a real full-parameter fine-tune of ``parent_id``.

    Claim: direction -- covers cases (i) SFT and (ii) continued pretraining, the
    two scar-free relations that docs/FINDINGS.md identifies as hard.
    """

    def _build(ctx: BuildContext, spec: ModelSpec) -> Dict[str, Any]:
        parent = ctx.path(parent_id)
        cfg = read_config(parent)
        dtype = dtype_of_dir(parent)
        src = ctx.base_paths[base_key]
        if ctx.skip_train:
            state = load_state_dict(parent)
            keys = [k for k, v in state.items() if is_matrix(v)]
            touched = lowrank_perturb(state, keys=keys, seed=ctx.seed + abs(hash(spec.id)) % 10_000)
            info = write_model_dir(ctx.path(spec.id), state, cfg, src, dtype=dtype)
            info.update({"training": "synthetic", "touched_tensors": touched})
        else:
            model = ctx.load_model(parent)
            lines = corpus_slice(ctx.corpus, corpus_part, 4)
            batches = make_batches(
                ctx.tokenizer(base_key),
                lines,
                seq_len=ctx.seq_len,
                batch_size=ctx.batch_size,
                n_batches=ctx.steps[steps_key],
                seed=ctx.seed + corpus_part,
            )
            stats = train_causal_lm(model, batches, device=ctx.device, lr=lr, tag=spec.id)
            touched = [n for n, _ in model.named_parameters()]
            info = save_trained(ctx, model, ctx.path(spec.id), asset_src=src, dtype=dtype)
            info.update({"training": "real", "touched_tensors": touched, **stats})
            del model
        info["arch"] = (cfg.get("architectures") or ["unknown"])[0]
        info["corpus"] = f"{ctx.corpus_source}[{corpus_part}/4]"
        return info

    return _build


def _build_lora(
    *, parent_id: str, corpus_part: int, steps_key: str, lr: float, base_key: str = "root"
) -> Callable[[BuildContext, ModelSpec], Dict[str, Any]]:
    """Return a builder for a LoRA fine-tune that is merged back into the weights.

    Claim: direction -- a merged LoRA child's task vector is low rank by
    construction, a different fingerprint from a full fine-tune of the same base.
    """

    def _build(ctx: BuildContext, spec: ModelSpec) -> Dict[str, Any]:
        parent = ctx.path(parent_id)
        cfg = read_config(parent)
        dtype = dtype_of_dir(parent)
        src = ctx.base_paths[base_key]
        if ctx.skip_train:
            state = load_state_dict(parent)
            keys = [k for k, v in state.items() if is_matrix(v) and any(t in k for t in LORA_TARGETS)]
            touched = lowrank_perturb(
                state, keys=keys, seed=ctx.seed + 991, rank=LORA_RANK, scale=SKIP_TRAIN_DELTA
            )
            info = write_model_dir(ctx.path(spec.id), state, cfg, src, dtype=dtype)
            info.update({"training": "synthetic", "touched_tensors": touched})
        else:
            model = ctx.load_model(parent)
            touched = inject_lora(model, targets=LORA_TARGETS, r=LORA_RANK, alpha=LORA_ALPHA, seed=ctx.seed + 7)
            model = model.to(ctx.device)
            lines = corpus_slice(ctx.corpus, corpus_part, 4)
            batches = make_batches(
                ctx.tokenizer(base_key),
                lines,
                seq_len=ctx.seq_len,
                batch_size=ctx.batch_size,
                n_batches=ctx.steps[steps_key],
                seed=ctx.seed + corpus_part,
            )
            stats = train_causal_lm(model, batches, device=ctx.device, lr=lr, tag=spec.id)
            merge_lora(model)
            info = save_trained(ctx, model, ctx.path(spec.id), asset_src=src, dtype=dtype)
            info.update({"training": "real", "touched_tensors": touched, **stats})
            del model
        info["arch"] = (cfg.get("architectures") or ["unknown"])[0]
        info["lora"] = {"r": LORA_RANK, "alpha": LORA_ALPHA, "targets": list(LORA_TARGETS)}
        return info

    return _build


def _build_quantized(*, parent_id: str, bits: int, base_key: str = "root") -> Callable[..., Dict[str, Any]]:
    """Return a builder for an INT8 / INT4 fake-quantised child.

    Claim: direction -- case (iii). Stored in float16 (not bfloat16) so the
    lattice survives the round trip to disk: bfloat16's ~4e-3 relative
    resolution is coarser than the 1e-3 tolerance the lattice detector uses.
    """

    def _build(ctx: BuildContext, spec: ModelSpec) -> Dict[str, Any]:
        parent = ctx.path(parent_id)
        state = load_state_dict(parent)
        cfg = read_config(parent)
        conv1d = str(cfg.get("model_type", "")).startswith("gpt2")
        touched = quantize_state(state, bits=bits, conv1d=conv1d)
        info = write_model_dir(ctx.path(spec.id), state, cfg, ctx.base_paths[base_key], dtype=torch.float16)
        info.update(
            {
                "touched_tensors": touched,
                "quantization": {"bits": bits, "scheme": "absmax", "group": INT4_GROUP if bits == 4 else None},
                "arch": (cfg.get("architectures") or ["unknown"])[0],
            }
        )
        return info

    return _build


def _build_pruned(
    *, parent_id: str, kind: str, base_key: str = "root"
) -> Callable[[BuildContext, ModelSpec], Dict[str, Any]]:
    """Return a builder for a magnitude-pruned or structurally pruned child.

    Claim: direction -- case (iv); exact zeros are left in place on purpose.
    """

    def _build(ctx: BuildContext, spec: ModelSpec) -> Dict[str, Any]:
        parent = ctx.path(parent_id)
        state = load_state_dict(parent)
        cfg = read_config(parent)
        extra: Dict[str, Any] = {}
        if kind == "magnitude":
            touched, thr, sparsity = magnitude_prune(
                state, fraction=MAGNITUDE_PRUNE_FRACTION, seed=ctx.seed + 13
            )
            extra = {
                "pruning": {
                    "kind": "global_magnitude",
                    "fraction": MAGNITUDE_PRUNE_FRACTION,
                    "threshold": thr,
                    "achieved_sparsity": round(sparsity, 5),
                }
            }
        elif kind == "structured":
            touched, n_rows = structured_prune_mlp(state, fraction=STRUCTURED_PRUNE_FRACTION)
            extra = {
                "pruning": {
                    "kind": "structured_mlp_rows",
                    "fraction": STRUCTURED_PRUNE_FRACTION,
                    "rows_removed": n_rows,
                }
            }
        else:  # pragma: no cover - guarded by the plan
            raise ValueError(kind)
        info = write_model_dir(ctx.path(spec.id), state, cfg, ctx.base_paths[base_key])
        info.update({"touched_tensors": touched, "arch": (cfg.get("architectures") or ["unknown"])[0]})
        info.update(extra)
        return info

    return _build


def _build_vocab_extended(
    *, parent_id: str, train_new_rows: bool, base_key: str = "root"
) -> Callable[[BuildContext, ModelSpec], Dict[str, Any]]:
    """Return a builder for a vocabulary-extended child (case (v)).

    Claim: direction -- ``train_new_rows=False`` gives real orphan rows; the
    harder variant trains *only* the new rows for a few steps so the benchmark
    contains a case where the orphan signal is degraded but not absent.
    """

    def _build(ctx: BuildContext, spec: ModelSpec) -> Dict[str, Any]:
        parent = ctx.path(parent_id)
        state = load_state_dict(parent)
        cfg = read_config(parent)
        dtype = dtype_of_dir(parent)
        src = ctx.base_paths[base_key]
        touched, old_vocab = extend_vocab(
            state, cfg, extra=VOCAB_EXTRA_ROWS, seed=ctx.seed + (77 if train_new_rows else 33)
        )
        out = ctx.path(spec.id)
        info = write_model_dir(out, state, cfg, src, dtype=dtype)
        extra: Dict[str, Any] = {
            "vocab": {"old": old_vocab, "new": old_vocab + VOCAB_EXTRA_ROWS, "init_std": VOCAB_INIT_STD},
            "touched_tensors": touched,
            "arch": (cfg.get("architectures") or ["unknown"])[0],
        }
        if train_new_rows:
            if ctx.skip_train:
                state2 = load_state_dict(out)
                gen = torch.Generator().manual_seed(ctx.seed + 555)
                for key in touched:
                    w = state2[key].float()
                    noise = torch.randn(VOCAB_EXTRA_ROWS, w.shape[1], generator=gen) * (VOCAB_INIT_STD * 0.5)
                    w[old_vocab:, :] = w[old_vocab:, :] + noise
                    state2[key] = w.to(state2[key].dtype)
                write_model_dir(out, state2, cfg, src, dtype=dtype)
                extra["training"] = "synthetic"
            else:
                stats = _train_only_new_rows(ctx, out, old_vocab=old_vocab, base_key=base_key, dtype=dtype)
                extra["training"] = "real"
                extra.update(stats)
            extra["vocab"]["new_rows_trained"] = True
        else:
            extra["vocab"]["new_rows_trained"] = False
        info.update(extra)
        return info

    return _build


def _train_only_new_rows(
    ctx: BuildContext, path: Path, *, old_vocab: int, base_key: str, dtype: torch.dtype
) -> Dict[str, float]:
    """Train a vocab-extended model's *new* embedding rows and nothing else.

    Claim: direction -- injecting the new token ids into the batches is what
    gives those rows a real gradient; masking the gradient of the first
    ``old_vocab`` rows keeps every other weight bit-identical to the parent.
    """
    model = ctx.load_model(path)
    for p in model.parameters():
        p.requires_grad_(False)
    emb = model.get_input_embeddings()
    emb.weight.requires_grad_(True)

    def _mask_grad(_m: Any) -> None:
        if emb.weight.grad is not None:
            emb.weight.grad[:old_vocab].zero_()

    lines = corpus_slice(ctx.corpus, 3, 4)
    batches = make_batches(
        ctx.tokenizer(base_key),
        lines,
        seq_len=ctx.seq_len,
        batch_size=ctx.batch_size,
        n_batches=ctx.steps["vocab"],
        seed=ctx.seed + 4,
    )
    gen = torch.Generator().manual_seed(ctx.seed + 5)
    new_total = int(emb.weight.shape[0])
    for b in batches:
        hit = torch.rand(b.shape, generator=gen) < 0.05
        rnd = torch.randint(old_vocab, new_total, b.shape, generator=gen)
        b[hit] = rnd[hit]
    stats = train_causal_lm(
        model, batches, device=ctx.device, lr=1e-3, tag=f"{path.name}:new-rows", grad_filter=_mask_grad
    )
    save_trained(ctx, model, path, asset_src=ctx.base_paths[base_key], dtype=dtype)
    del model
    return stats


def _build_merge(
    *, base_id: str, parent_ids: Sequence[str], ratios: Sequence[float], method: str, base_key: str = "root"
) -> Callable[[BuildContext, ModelSpec], Dict[str, Any]]:
    """Return a builder for a 2- or 3-parent merge over a shared base (case (vi)).

    Claim: merge-recovery -- every merge here uses the *same* base, so the task
    vectors are well defined and the recorded ratios are the exact quantity
    ``merge_decompose.decompose_merge`` is asked to recover.
    """

    def _build(ctx: BuildContext, spec: ModelSpec) -> Dict[str, Any]:
        base_sd = load_state_dict(ctx.path(base_id))
        parents = [load_state_dict(ctx.path(p)) for p in parent_ids]
        cfg = read_config(ctx.path(base_id))
        if method == "slerp":
            merged, meta = merge_slerp(base_sd, parents, ratios)
        elif method == "ties":
            merged, meta = merge_ties(base_sd, parents, ratios, seed=ctx.seed + 21)
        elif method == "dare":
            merged, meta = merge_dare(base_sd, parents, ratios, seed=ctx.seed + 31)
        else:  # pragma: no cover - guarded by the plan
            raise ValueError(method)
        dtype = dtype_of_dir(ctx.path(base_id))
        info = write_model_dir(ctx.path(spec.id), merged, cfg, ctx.base_paths[base_key], dtype=dtype)
        info.update(
            {
                "merge": {**meta, "base": base_id, "parents": list(parent_ids), "ratios": list(map(float, ratios))},
                "touched_tensors": sorted(_mergeable_keys(base_sd, parents)),
                "arch": (cfg.get("architectures") or ["unknown"])[0],
            }
        )
        return info

    return _build


def _build_distilled_student(*, teacher_id: str, base_key: str = "root") -> Callable[..., Dict[str, Any]]:
    """Return a builder for a depth-halved KL-distilled student (case (vii)).

    Claim: direction -- labelled ``weak_weight_lineage=True``: the student has a
    different architecture from its teacher and docs/FINDINGS.md requires this
    case to be reported honestly rather than excluded.
    """

    def _build(ctx: BuildContext, spec: ModelSpec) -> Dict[str, Any]:
        teacher_path = ctx.path(teacher_id)
        t_state = load_state_dict(teacher_path)
        cfg = read_config(teacher_path)
        dtype = dtype_of_dir(teacher_path)
        s_state, n_kept = halve_depth(t_state, cfg)
        s_cfg = dict(cfg)
        s_cfg["num_hidden_layers"] = n_kept
        out = ctx.path(spec.id)
        info = write_model_dir(out, s_state, s_cfg, ctx.base_paths[base_key], dtype=dtype)
        extra: Dict[str, Any] = {
            "distillation": {"teacher": teacher_id, "student_layers": n_kept, "teacher_layers": len(t_state) and int(cfg.get("num_hidden_layers", 0))},
            "arch": (cfg.get("architectures") or ["unknown"])[0],
            "touched_tensors": sorted(s_state),
        }
        del t_state, s_state
        if ctx.skip_train:
            state = load_state_dict(out)
            keys = [k for k, v in state.items() if is_matrix(v)]
            lowrank_perturb(state, keys=keys, seed=ctx.seed + 4242)
            write_model_dir(out, state, s_cfg, ctx.base_paths[base_key], dtype=dtype)
            extra["training"] = "synthetic"
        else:
            extra.update(_distil_kl(ctx, teacher_path, out, base_key=base_key, dtype=dtype))
            extra["training"] = "real"
        info.update(extra)
        return info

    return _build


def _distil_kl(
    ctx: BuildContext, teacher_path: Path, student_path: Path, *, base_key: str, dtype: torch.dtype
) -> Dict[str, float]:
    """Train the student against the teacher's logits with a KL objective.

    Claim: direction.
    """
    teacher = ctx.load_model(teacher_path)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = ctx.load_model(student_path)
    student.train()
    lines = corpus_slice(ctx.corpus, 2, 4)
    batch_size = max(1, ctx.batch_size // 2)
    batches = make_batches(
        ctx.tokenizer(base_key),
        lines,
        seq_len=ctx.seq_len,
        batch_size=batch_size,
        n_batches=ctx.steps["distil"],
        seed=ctx.seed + 6,
    )
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=5e-5)
    temp = 2.0
    first = last = float("nan")
    t0 = time.time()
    for i, batch in enumerate(batches):
        ids = batch.to(ctx.device)
        with torch.no_grad():
            t_logits = teacher(input_ids=ids).logits
        s_logits = student(input_ids=ids).logits
        loss = torch.nn.functional.kl_div(
            torch.log_softmax(s_logits / temp, dim=-1),
            torch.log_softmax(t_logits / temp, dim=-1),
            log_target=True,
            reduction="batchmean",
        ) * (temp * temp)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        val = float(loss.detach().float().cpu())
        if i == 0:
            first = val
        last = val
        if i % 5 == 0:
            LOG.info("    distil step %d/%d kl=%.4f", i, len(batches), val)
    save_trained(ctx, student, student_path, asset_src=ctx.base_paths[base_key], dtype=dtype)
    del teacher, student
    return {"steps": float(len(batches)), "kl_first": first, "kl_last": last, "train_seconds": round(time.time() - t0, 2)}


def _build_random_init(*, template_id: str, init_seed: int, base_key: str = "root") -> Callable[..., Dict[str, Any]]:
    """Return a builder for a same-architecture / different-init control (case (viii)).

    Claim: low-false-positive -- identical shapes, identical config, zero shared
    lineage. This is the pair that decides whether the relatedness score is
    measuring history or merely architecture.
    """

    def _build(ctx: BuildContext, spec: ModelSpec) -> Dict[str, Any]:
        tf = import_transformers()
        cfg = tf.AutoConfig.from_pretrained(str(ctx.path(template_id)))
        torch.manual_seed(int(init_seed))
        np.random.seed(int(init_seed))
        model = tf.AutoModelForCausalLM.from_config(cfg)
        dtype = dtype_of_dir(ctx.path(template_id))
        info = save_trained(ctx, model, ctx.path(spec.id), asset_src=ctx.base_paths[base_key], dtype=dtype)
        info.update(
            {
                "init_seed": int(init_seed),
                "arch": (read_config(ctx.path(template_id)).get("architectures") or ["unknown"])[0],
                "touched_tensors": [],
            }
        )
        del model
        return info

    return _build


# --------------------------------------------------------------------------- #
# The plan
# --------------------------------------------------------------------------- #


def plan_models(ctx: BuildContext, *, large: bool) -> List[ModelSpec]:
    """Declare the whole ground-truth DAG in topological order.

    Claim: infra -- topological order is what makes ``--limit`` safe: truncating
    the list can never orphan a child whose parent was not built.

    The licence assignment is deliberate: ``smollm2-cpt`` is ``cc-by-nc-4.0``
    and ``smollm2-lora`` is ``cc-by-sa-4.0``, while the merges downstream of
    them declare ``apache-2.0``. That plants one real non-commercial conflict
    and one real share-alike conflict for ``stemma.rights`` to find.
    """
    root = "smollm2-135m-root"
    sft = "smollm2-sft"
    lora = "smollm2-lora-merged"
    cpt = "smollm2-cpt"
    gpt2 = "gpt2-root"

    specs: List[ModelSpec] = [
        ModelSpec(
            id=root,
            family="smollm2",
            op="base",
            relation="base",
            parents=[],
            weights={},
            license="apache-2.0",
            build=_build_base_copy("root"),
            notes="primary root, copied verbatim from the local cache",
        ),
        ModelSpec(
            id=gpt2,
            family="gpt2",
            op="base",
            relation="base",
            parents=[],
            weights={},
            license="mit",
            build=_build_base_copy("gpt2"),
            notes="second family; cross-family false-positive control against smollm2",
        ),
        ModelSpec(
            id="distilgpt2",
            family="gpt2",
            op="distilled",
            relation="distilled",
            parents=[gpt2],
            weights={},
            license="apache-2.0",
            build=_build_base_copy("distilgpt2"),
            weak_weight_lineage=True,
            notes="REAL published distillation of gpt2 (6 layers vs 12); headline case (vii)",
        ),
        ModelSpec(
            id=sft,
            family="smollm2",
            op="sft",
            relation="sft",
            parents=[root],
            weights={},
            license="apache-2.0",
            build=_build_finetune(parent_id=root, corpus_part=0, steps_key="sft", lr=2e-5),
        ),
        ModelSpec(
            id=lora,
            family="smollm2",
            op="lora_merged",
            relation="lora_merged",
            parents=[root],
            weights={},
            license="cc-by-sa-4.0",
            build=_build_lora(parent_id=root, corpus_part=1, steps_key="lora", lr=5e-4),
            notes="planted share-alike ancestor for the rights demo",
        ),
        ModelSpec(
            id=cpt,
            family="smollm2",
            op="continued_pretrain",
            relation="continued_pretrain",
            parents=[sft],
            weights={},
            license="cc-by-nc-4.0",
            build=_build_finetune(parent_id=sft, corpus_part=2, steps_key="cpt", lr=2e-5),
            notes="depth-3 node; planted non-commercial ancestor for the rights demo",
        ),
        ModelSpec(
            id="smollm2-int8",
            family="smollm2",
            op="quantized_int8",
            relation="quantized_int8",
            parents=[root],
            weights={},
            license="apache-2.0",
            build=_build_quantized(parent_id=root, bits=8),
        ),
        ModelSpec(
            id="smollm2-sft-int4",
            family="smollm2",
            op="quantized_int4",
            relation="quantized_int4",
            parents=[sft],
            weights={},
            license="apache-2.0",
            build=_build_quantized(parent_id=sft, bits=4),
        ),
        ModelSpec(
            id="gpt2-int8",
            family="gpt2",
            op="quantized_int8",
            relation="quantized_int8",
            parents=[gpt2],
            weights={},
            license="mit",
            build=_build_quantized(parent_id=gpt2, bits=8, base_key="gpt2"),
        ),
        ModelSpec(
            id="smollm2-prune-mag30",
            family="smollm2",
            op="pruned_magnitude",
            relation="pruned_magnitude",
            parents=[root],
            weights={},
            license="apache-2.0",
            build=_build_pruned(parent_id=root, kind="magnitude"),
        ),
        ModelSpec(
            id="smollm2-sft-prune-struct10",
            family="smollm2",
            op="pruned_structured",
            relation="pruned_structured",
            parents=[sft],
            weights={},
            license="unknown",
            build=_build_pruned(parent_id=sft, kind="structured"),
            notes="licence deliberately left unknown so rights.detect_conflicts sees an unknown ancestor",
        ),
        ModelSpec(
            id="smollm2-vocab-ext",
            family="smollm2",
            op="vocab_extended",
            relation="vocab_extended",
            parents=[root],
            weights={},
            license="apache-2.0",
            build=_build_vocab_extended(parent_id=root, train_new_rows=False),
        ),
        ModelSpec(
            id="smollm2-vocab-ext-trained",
            family="smollm2",
            op="vocab_extended_trained",
            relation="vocab_extended_trained",
            parents=[root],
            weights={},
            license="apache-2.0",
            build=_build_vocab_extended(parent_id=root, train_new_rows=True),
            notes="harder variant: the 512 new rows received a few real optimizer steps",
        ),
        ModelSpec(
            id="smollm2-merge-slerp",
            family="smollm2",
            op="merge_slerp",
            relation="merge_slerp",
            parents=[sft, lora],
            weights={sft: 0.7, lora: 0.3},
            license="apache-2.0",
            build=_build_merge(base_id=root, parent_ids=[sft, lora], ratios=[0.7, 0.3], method="slerp"),
            notes="apache-2.0 downstream of a cc-by-sa-4.0 parent (planted share-alike conflict)",
        ),
        ModelSpec(
            id="smollm2-merge-ties2",
            family="smollm2",
            op="merge_ties",
            relation="merge_ties",
            parents=[sft, cpt],
            weights={sft: 0.6, cpt: 0.4},
            license="apache-2.0",
            build=_build_merge(base_id=root, parent_ids=[sft, cpt], ratios=[0.6, 0.4], method="ties"),
            notes="apache-2.0 downstream of a cc-by-nc-4.0 parent (planted non-commercial conflict)",
        ),
        ModelSpec(
            id="smollm2-merge-ties3",
            family="smollm2",
            op="merge_ties",
            relation="merge_ties",
            parents=[sft, lora, cpt],
            weights={sft: 0.5, lora: 0.3, cpt: 0.2},
            license="apache-2.0",
            build=_build_merge(
                base_id=root, parent_ids=[sft, lora, cpt], ratios=[0.5, 0.3, 0.2], method="ties"
            ),
            notes="three-parent merge; both planted licence conflicts are upstream",
        ),
        ModelSpec(
            id="smollm2-merge-dare2",
            family="smollm2",
            op="merge_dare",
            relation="merge_dare",
            parents=[sft, lora],
            weights={sft: 0.5, lora: 0.5},
            license="apache-2.0",
            build=_build_merge(base_id=root, parent_ids=[sft, lora], ratios=[0.5, 0.5], method="dare"),
        ),
        ModelSpec(
            id="smollm2-distil-half",
            family="smollm2",
            op="distilled",
            relation="distilled",
            parents=[root],
            weights={},
            license="apache-2.0",
            build=_build_distilled_student(teacher_id=root),
            weak_weight_lineage=True,
            notes="every-other-layer student trained on KL to the teacher logits",
        ),
        ModelSpec(
            id="control-rand-init-a",
            family="control",
            op="random_init",
            relation="none",
            parents=[],
            weights={},
            license="apache-2.0",
            build=_build_random_init(template_id=root, init_seed=1),
            notes="same architecture as the root, different init seed, NO shared lineage",
        ),
        ModelSpec(
            id="control-rand-init-b",
            family="control",
            op="random_init",
            relation="none",
            parents=[],
            weights={},
            license="apache-2.0",
            build=_build_random_init(template_id=root, init_seed=2),
            notes="same architecture as the root, different init seed, NO shared lineage",
        ),
    ]

    if large:
        qwen = "qwen25-05b-root"
        specs.extend(
            [
                ModelSpec(
                    id=qwen,
                    family="qwen2.5",
                    op="base",
                    relation="base",
                    parents=[],
                    weights={},
                    license="apache-2.0",
                    build=_build_base_copy("qwen"),
                    notes="third family (--large)",
                ),
                ModelSpec(
                    id="qwen25-sft",
                    family="qwen2.5",
                    op="sft",
                    relation="sft",
                    parents=[qwen],
                    weights={},
                    license="apache-2.0",
                    build=_build_finetune(parent_id=qwen, corpus_part=0, steps_key="sft", lr=2e-5, base_key="qwen"),
                ),
                ModelSpec(
                    id="qwen25-int8",
                    family="qwen2.5",
                    op="quantized_int8",
                    relation="quantized_int8",
                    parents=[qwen],
                    weights={},
                    license="apache-2.0",
                    build=_build_quantized(parent_id=qwen, bits=8, base_key="qwen"),
                ),
            ]
        )
    return specs


# --------------------------------------------------------------------------- #
# Ground truth assembly
# --------------------------------------------------------------------------- #


def _ancestors(edges: Sequence[Dict[str, Any]], nodes: Sequence[str]) -> Dict[str, set]:
    """Transitive closure: every node mapped to the set of all its ancestors.

    Claim: infra -- the benchmark's related-pair set *is* this closure, so it is
    computed rather than hand-listed.
    """
    parents: Dict[str, List[str]] = {n: [] for n in nodes}
    for e in edges:
        parents[e["child"]].append(e["parent"])
    memo: Dict[str, set] = {}

    def rec(n: str, stack: Tuple[str, ...] = ()) -> set:
        if n in memo:
            return memo[n]
        if n in stack:  # pragma: no cover - the plan is acyclic by construction
            return set()
        acc: set = set()
        for p in parents.get(n, []):
            acc.add(p)
            acc |= rec(p, stack + (n,))
        memo[n] = acc
        return acc

    return {n: rec(n) for n in nodes}


def _components(edges: Sequence[Dict[str, Any]], nodes: Sequence[str]) -> Dict[str, int]:
    """Connected components of the undirected lineage graph.

    Claim: low-false-positive -- two models in different components share no
    ancestor at all, which is exactly the definition of an unrelated pair.
    """
    parent = {n: n for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for e in edges:
        union(e["parent"], e["child"])
    roots = sorted({find(n) for n in nodes})
    return {n: roots.index(find(n)) for n in nodes}


def build_pairs(
    models: Sequence[Dict[str, Any]], edges: Sequence[Dict[str, Any]], *, seed: int
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Derive the labelled pair list from the edge list.

    Claim: low-false-positive -- related pairs come from the transitive closure
    (direction ``a->b``), sibling pairs from a shared ancestor without descent
    (direction ``sibling``), and unrelated pairs from disjoint components
    (direction ``none``), sampled to roughly balance the two classes so the
    reported AUC and FPR@95TPR are not flattered by class imbalance.
    """
    ids = [m["id"] for m in models]
    direct = {(e["parent"], e["child"]): e["relation"] for e in edges}
    anc = _ancestors(edges, ids)
    comp = _components(edges, ids)

    pairs: List[Dict[str, Any]] = []
    for child in ids:
        for a in sorted(anc[child]):
            rel = direct.get((a, child), "ancestor")
            pairs.append({"a": a, "b": child, "related": True, "direction": "a->b", "relation": rel})
    n_ancestral = len(pairs)

    n_sibling = 0
    for i, x in enumerate(ids):
        for y in ids[i + 1 :]:
            if y in anc[x] or x in anc[y]:
                continue
            if anc[x] & anc[y] or (x in anc.get(y, set())):
                shared = anc[x] & anc[y]
                if not shared:
                    continue
                rel = "sibling" if (set(m for m in shared) and _share_direct_parent(edges, x, y)) else "cousin"
                pairs.append({"a": x, "b": y, "related": True, "direction": "sibling", "relation": rel})
                n_sibling += 1

    candidates: List[Tuple[str, str]] = []
    forced: List[Tuple[str, str]] = []
    for i, x in enumerate(ids):
        for y in ids[i + 1 :]:
            if comp[x] == comp[y]:
                continue
            if _is_hard_negative(x, y):
                forced.append((x, y))
            else:
                candidates.append((x, y))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    want = max(0, (n_ancestral + n_sibling) - len(forced))
    chosen = forced + candidates[:want]
    chosen.sort()
    for x, y in chosen:
        pairs.append({"a": x, "b": y, "related": False, "direction": "none", "relation": "none"})

    counts = {
        "ancestral": n_ancestral,
        "sibling": n_sibling,
        "unrelated": len(chosen),
        "unrelated_available": len(forced) + len(candidates),
        "total": len(pairs),
    }
    return pairs, counts


def _share_direct_parent(edges: Sequence[Dict[str, Any]], x: str, y: str) -> bool:
    """True when two models have at least one parent in common.

    Claim: infra.
    """
    px = {e["parent"] for e in edges if e["child"] == x}
    py = {e["parent"] for e in edges if e["child"] == y}
    return bool(px & py)


def _is_hard_negative(x: str, y: str) -> bool:
    """True for the unrelated pairs that matter most and must never be sampled out.

    Claim: low-false-positive -- the two different-init controls against each
    other and against the root they were configured from are the pairs that
    decide whether the method measures history or architecture.
    """
    ctrl = ("control-rand-init" in x, "control-rand-init" in y)
    if all(ctrl):
        return True
    if any(ctrl) and ("root" in x or "root" in y):
        return True
    return False


def assemble_ground_truth(
    built: Sequence[Tuple[ModelSpec, Dict[str, Any]]], *, out_dir: Path, seed: int, meta: Dict[str, Any]
) -> Dict[str, Any]:
    """Assemble ``ground_truth.json`` in the exact schema from CONTRACT.md.

    Claim: infra -- keys are exactly ``models`` / ``edges`` / ``pairs``; every
    additional per-model key (``touched_tensors``, ``weak_weight_lineage``,
    ``training``...) is additive and is what lets the benchmark report accuracy
    *per relation type* as docs/FINDINGS.md section 5 requires.
    """
    built_ids = {spec.id for spec, _ in built}
    models: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    for spec, info in built:
        parents = [p for p in spec.parents if p in built_ids]
        rec: Dict[str, Any] = {
            "id": spec.id,
            "path": str((out_dir / spec.id).resolve()),
            "family": spec.family,
            "op": spec.op,
            "parents": parents,
            "weights": {k: float(v) for k, v in spec.weights.items() if k in built_ids},
            "license": spec.license,
            "relation": spec.relation,
            "weak_weight_lineage": bool(spec.weak_weight_lineage),
            "notes": spec.notes,
        }
        for key in (
            "arch",
            "n_params",
            "n_tensors",
            "bytes",
            "dtypes",
            "training",
            "touched_tensors",
            "corpus",
            "quantization",
            "pruning",
            "vocab",
            "merge",
            "lora",
            "distillation",
            "init_seed",
            "source_repo",
            "loss_first",
            "loss_last",
            "kl_first",
            "kl_last",
            "steps",
            "train_seconds",
            "build_seconds",
        ):
            if key in info:
                rec[key] = info[key]
        models.append(rec)
        for p in parents:
            edges.append(
                {
                    "parent": p,
                    "child": spec.id,
                    "relation": spec.relation,
                    "weight": float(spec.weights[p]) if p in spec.weights else None,
                }
            )

    pairs, counts = build_pairs(models, edges, seed=seed)
    return {
        "models": models,
        "edges": edges,
        "pairs": pairs,
        "meta": {**meta, "pair_counts": counts, "n_models": len(models), "n_edges": len(edges)},
    }


def validate_ground_truth(gt: Dict[str, Any], *, check_files: bool = True) -> List[str]:
    """Check the ground truth against the CONTRACT.md schema; return problems.

    Claim: infra -- a malformed label file would silently poison every number in
    the benchmark, so validation runs before the file is considered finished.
    """
    problems: List[str] = []
    for key in ("models", "edges", "pairs"):
        if key not in gt or not isinstance(gt[key], list):
            problems.append(f"missing or non-list key {key!r}")
    if problems:
        return problems
    ids = set()
    for i, m in enumerate(gt["models"]):
        for key in ("id", "path", "family", "op", "parents", "weights", "license"):
            if key not in m:
                problems.append(f"models[{i}] missing key {key!r}")
        if "id" in m:
            if m["id"] in ids:
                problems.append(f"duplicate model id {m['id']!r}")
            ids.add(m["id"])
        if not isinstance(m.get("parents", []), list):
            problems.append(f"models[{i}].parents is not a list")
        if not isinstance(m.get("weights", {}), dict):
            problems.append(f"models[{i}].weights is not a dict")
        if check_files and "path" in m:
            p = Path(m["path"])
            if not (p / "model.safetensors").is_file():
                problems.append(f"{m.get('id')}: missing model.safetensors at {p}")
            if not (p / "config.json").is_file():
                problems.append(f"{m.get('id')}: missing config.json at {p}")
    for i, m in enumerate(gt["models"]):
        for p in m.get("parents", []):
            if p not in ids:
                problems.append(f"models[{i}] parent {p!r} is not a known model")
        for p in m.get("weights", {}):
            if p not in m.get("parents", []):
                problems.append(f"models[{i}] weight key {p!r} is not a parent")
    for i, e in enumerate(gt["edges"]):
        for key in ("parent", "child", "relation"):
            if key not in e:
                problems.append(f"edges[{i}] missing key {key!r}")
        if e.get("parent") not in ids or e.get("child") not in ids:
            problems.append(f"edges[{i}] references an unknown model")
    seen_pairs = set()
    for i, pr in enumerate(gt["pairs"]):
        for key in ("a", "b", "related", "direction"):
            if key not in pr:
                problems.append(f"pairs[{i}] missing key {key!r}")
        if pr.get("a") not in ids or pr.get("b") not in ids:
            problems.append(f"pairs[{i}] references an unknown model")
        if pr.get("direction") not in ("a->b", "b->a", "sibling", "none"):
            problems.append(f"pairs[{i}] bad direction {pr.get('direction')!r}")
        if not isinstance(pr.get("related"), bool):
            problems.append(f"pairs[{i}].related is not a bool")
        if pr.get("related") and pr.get("direction") == "none":
            problems.append(f"pairs[{i}] related but direction 'none'")
        if not pr.get("related") and pr.get("direction") != "none":
            problems.append(f"pairs[{i}] unrelated but direction {pr.get('direction')!r}")
        key2 = (pr.get("a"), pr.get("b"), pr.get("direction"))
        if key2 in seen_pairs:
            problems.append(f"pairs[{i}] duplicate {key2}")
        seen_pairs.add(key2)
    return problems


# --------------------------------------------------------------------------- #
# Dataset card + Hub push
# --------------------------------------------------------------------------- #


def dataset_card(gt: Dict[str, Any], repo: str = "<org>/<name>") -> str:
    """Render the dataset card shipped alongside a ``--push-to-hub`` upload.

    Claim: infra -- the card states plainly that the labels are generated by
    running the operations, and repeats the FINDINGS caveat about distillation,
    so nobody can read a number off this dataset without its caveat.
    """
    counts = gt.get("meta", {}).get("pair_counts", {})
    rels = sorted({e["relation"] for e in gt["edges"]})
    weak = [m["id"] for m in gt["models"] if m.get("weak_weight_lineage")]
    lines = [
        "---",
        "license: apache-2.0",
        "task_categories:",
        "- other",
        "tags:",
        "- model-provenance",
        "- lineage",
        "- ai-bom",
        "- model-merging",
        "---",
        "",
        f"# Stemma lineage benchmark ({repo})",
        "",
        "Labelled ordered pairs over a **generated** model-lineage DAG. Every edge was",
        "produced by really running the operation on real safetensors weights (short",
        "fine-tunes, a hand-written LoRA, INT8/INT4 fake-quantisation round trips,",
        "magnitude and structured pruning, vocabulary extension, SLERP/TIES/DARE merges",
        "with recorded mixing ratios, and distillation), so the labels are recipes, not",
        "annotations.",
        "",
        f"- models: {gt.get('meta', {}).get('n_models')}",
        f"- edges: {gt.get('meta', {}).get('n_edges')}",
        f"- pairs: ancestral {counts.get('ancestral')}, sibling {counts.get('sibling')}, "
        f"unrelated {counts.get('unrelated')}",
        f"- relations: {', '.join(rels)}",
        "",
        "## Columns",
        "",
        "`a`, `b` (model ids), `related` (bool), `direction` (`a->b` | `sibling` | `none`),",
        "`relation`, plus the per-side `family`/`op` of each model.",
        "",
        "## Caveats",
        "",
        "- Distillation edges are marked `weak_weight_lineage=true` "
        f"({', '.join(weak) if weak else 'none in this build'}). Weight-level lineage across a",
        "  distillation edge is *expected to be weak*; it is included so it can be scored",
        "  honestly rather than quietly excluded.",
        "- Direction accuracy must be reported per relation type. Scar-free relations",
        "  (SFT, LoRA, continued pretraining) are much harder than lossy ones",
        "  (quantisation, pruning, vocabulary extension).",
        "- Unrelated pairs include same-architecture / different-initialisation models.",
        "  Those are the pairs a false-positive rate should be measured on.",
        "- Builds made with `--skip-train` carry `training: synthetic` and must not be",
        "  reported as fine-tuning measurements.",
    ]
    return "\n".join(lines)


def push_to_hub(gt: Dict[str, Any], repo: str, *, token: Optional[str] = None, private: bool = False) -> str:
    """Push the labelled pairs to the Hub as a ``datasets.Dataset``.

    Claim: infra -- lazily imported, and a missing token produces one clear
    instruction instead of an HTTP 401 traceback. Nothing about the local build
    depends on this path.
    """
    try:
        from datasets import Dataset  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - datasets is installed in dev
        raise SystemExit(f"--push-to-hub needs the 'datasets' package: {exc}") from exc

    tok = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        try:
            from huggingface_hub import get_token  # noqa: PLC0415

            tok = get_token()
        except Exception:  # pragma: no cover - old hub versions
            tok = None
    if not tok:
        raise SystemExit(
            "--push-to-hub needs a Hugging Face token. Run `huggingface-cli login`, or set "
            "HF_TOKEN=... in the environment. The local build itself never needs a token."
        )

    by_id = {m["id"]: m for m in gt["models"]}
    rows = [
        {
            "a": p["a"],
            "b": p["b"],
            "related": bool(p["related"]),
            "direction": p["direction"],
            "relation": p.get("relation", "none"),
            "a_family": by_id[p["a"]]["family"],
            "b_family": by_id[p["b"]]["family"],
            "a_op": by_id[p["a"]]["op"],
            "b_op": by_id[p["b"]]["op"],
            "weak_weight_lineage": bool(
                by_id[p["a"]].get("weak_weight_lineage") or by_id[p["b"]].get("weak_weight_lineage")
            ),
        }
        for p in gt["pairs"]
    ]
    ds = Dataset.from_list(rows)
    ds.push_to_hub(repo, token=tok, private=private)

    card = dataset_card(gt, repo)
    try:
        from huggingface_hub import HfApi  # noqa: PLC0415

        HfApi().upload_file(
            path_or_fileobj=card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo,
            repo_type="dataset",
            token=tok,
        )
    except Exception as exc:  # pragma: no cover - network dependent
        LOG.warning("dataset pushed but the card upload failed: %s", exc)
    return f"https://huggingface.co/datasets/{repo}"


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def make_context(args: argparse.Namespace) -> BuildContext:
    """Resolve bases, corpus and step budgets into a :class:`BuildContext`.

    Claim: infra.
    """
    refs = dict(DEFAULT_BASES)
    if args.base:
        refs["root"] = args.base
    device = pick_device(args.device)
    steps = (
        {"sft": 6, "lora": 8, "cpt": 8, "vocab": 4, "distil": 4}
        if args.quick
        else {"sft": 40, "lora": 60, "cpt": 70, "vocab": 20, "distil": 30}
    )
    ctx = BuildContext(
        out_dir=Path(args.out_dir).expanduser().resolve(),
        seed=int(args.seed),
        device=device,
        quick=bool(args.quick),
        skip_train=bool(args.skip_train),
        force=bool(args.force),
        base_refs=refs,
        seq_len=128 if args.quick else 256,
        batch_size=2 if args.quick else 4,
        steps=steps,
    )
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    needed = ["root", "gpt2", "distilgpt2"] + (["qwen"] if args.large else [])
    for key in needed:
        t0 = time.time()
        ctx.base_paths[key] = resolve_base(refs[key])
        LOG.info("base %-11s -> %s (%.2fs)", key, ctx.base_paths[key], time.time() - t0)
    if not ctx.skip_train:
        ctx.corpus, ctx.corpus_source = load_corpus(400 if args.quick else 6000)
    else:
        ctx.corpus, ctx.corpus_source = ["(training skipped)"], "none"
    return ctx


def run_build(ctx: BuildContext, specs: Sequence[ModelSpec]) -> List[Tuple[ModelSpec, Dict[str, Any]]]:
    """Materialise every planned model, skipping the ones already on disk.

    Claim: infra.
    """
    built: List[Tuple[ModelSpec, Dict[str, Any]]] = []
    for i, spec in enumerate(specs, 1):
        path = ctx.path(spec.id)
        t0 = time.time()
        if is_built(path) and not ctx.force:
            LOG.info("[%2d/%2d] %-30s skip (already built)", i, len(specs), spec.id)
            info = _info_from_disk(path)
            info["reused"] = True
        else:
            if ctx.force and path.exists():
                shutil.rmtree(path)
            LOG.info("[%2d/%2d] %-30s building (%s)", i, len(specs), spec.id, spec.op)
            info = spec.build(ctx, spec)
            info["reused"] = False
        info["build_seconds"] = round(time.time() - t0, 2)
        LOG.info(
            "[%2d/%2d] %-30s %s  %s  %.1fs",
            i,
            len(specs),
            spec.id,
            human_bytes(info.get("bytes", 0)),
            ",".join(info.get("dtypes", [])) or "-",
            info["build_seconds"],
        )
        built.append((spec, info))
    return built


def _info_from_disk(path: Path) -> Dict[str, Any]:
    """Recover the size/dtype summary of an already-built model directory.

    Claim: infra -- keeps a resumed build's ground truth identical in shape to a
    fresh one, minus the training statistics that only exist at build time.
    """
    from safetensors import safe_open  # noqa: PLC0415

    size = (path / "model.safetensors").stat().st_size
    n_params = 0
    dtypes = set()
    with safe_open(str(path / "model.safetensors"), framework="pt") as f:
        keys = list(f.keys())
        for k in keys:
            sl = f.get_slice(k)
            shape = sl.get_shape()
            n = 1
            for d in shape:
                n *= int(d)
            n_params += n
            dtypes.add(str(sl.get_dtype()))
    cfg = read_config(path)
    return {
        "bytes": int(size),
        "n_params": n_params,
        "n_tensors": len(keys),
        "dtypes": sorted(d.replace("torch.", "").lower() for d in dtypes),
        "arch": (cfg.get("architectures") or ["unknown"])[0],
    }


def summarise(gt: Dict[str, Any]) -> str:
    """Render the end-of-run summary table.

    Claim: infra.
    """
    rows = ["", f"{'model':<32} {'family':<9} {'op':<22} {'params':>12} {'size':>10}  parents"]
    rows.append("-" * 104)
    for m in gt["models"]:
        rows.append(
            f"{m['id']:<32} {m['family']:<9} {m['op']:<22} {m.get('n_params', 0):>12,} "
            f"{human_bytes(m.get('bytes', 0)):>10}  {','.join(m['parents']) or '-'}"
        )
    counts = gt["meta"]["pair_counts"]
    rows.append("-" * 104)
    rows.append(
        f"models={gt['meta']['n_models']}  edges={gt['meta']['n_edges']}  "
        f"pairs={counts['total']} (ancestral={counts['ancestral']}, sibling={counts['sibling']}, "
        f"unrelated={counts['unrelated']}/{counts['unrelated_available']} available)"
    )
    return "\n".join(rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Command-line interface for the benchmark builder.

    Claim: infra.
    """
    ap = argparse.ArgumentParser(
        description="Build Stemma's ground-truth lineage DAG as real safetensors models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--out-dir", default=str(_REPO_ROOT / "bench_models"), help="where to write the models")
    ap.add_argument("--base", default=None, help="override the primary root checkpoint (repo id or local dir)")
    ap.add_argument("--limit", type=int, default=0, help="build only the first N models (topological order)")
    ap.add_argument("--quick", action="store_true", help="far fewer optimizer steps and shorter sequences")
    ap.add_argument("--skip-train", action="store_true", help="seeded low-rank perturbations instead of training")
    ap.add_argument("--large", action="store_true", help="also build the Qwen2.5-0.5B family")
    ap.add_argument("--seed", type=int, default=0, help="master seed")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"], help="training device")
    ap.add_argument("--force", action="store_true", help="rebuild models that already exist")
    ap.add_argument("--push-to-hub", default=None, metavar="ORG/NAME", help="push the pair table as a dataset")
    ap.add_argument("--hub-private", action="store_true", help="make the pushed dataset private")
    ap.add_argument("--hub-token", default=None, help="token for --push-to-hub (else HF_TOKEN / cached login)")
    ap.add_argument("--log-level", default="INFO", help="logging level")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Build the benchmark and write ``ground_truth.json``.

    Claim: infra -- this is the entry point every reported number depends on.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    t_start = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ctx = make_context(args)
    LOG.info(
        "out-dir=%s device=%s quick=%s skip-train=%s seed=%d corpus=%s",
        ctx.out_dir,
        ctx.device,
        ctx.quick,
        ctx.skip_train,
        ctx.seed,
        ctx.corpus_source,
    )
    specs = plan_models(ctx, large=bool(args.large))
    if args.limit and args.limit > 0:
        specs = specs[: args.limit]
        LOG.info("--limit %d -> building %d of the planned models", args.limit, len(specs))

    built = run_build(ctx, specs)

    meta = {
        "generator": "scripts/build_bench.py",
        "seed": ctx.seed,
        "device": ctx.device,
        "quick": ctx.quick,
        "skip_train": ctx.skip_train,
        "training": "synthetic" if ctx.skip_train else "real",
        "corpus": ctx.corpus_source,
        "bases": {k: ctx.base_refs[k] for k in ctx.base_paths},
        "steps": ctx.steps,
        "seq_len": ctx.seq_len,
        "batch_size": ctx.batch_size,
        "limit": int(args.limit or 0),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    gt = assemble_ground_truth(built, out_dir=ctx.out_dir, seed=ctx.seed, meta=meta)
    problems = validate_ground_truth(gt)
    if problems:
        for p in problems[:20]:
            LOG.error("ground truth: %s", p)
        raise SystemExit(f"ground truth failed validation ({len(problems)} problems)")

    gt_path = ctx.out_dir / "ground_truth.json"
    atomic_write_json(gt_path, gt)
    card_path = ctx.out_dir / "DATASET_CARD.md"
    card_path.write_text(dataset_card(gt), encoding="utf-8")
    meta["elapsed_seconds"] = round(time.time() - t_start, 2)
    gt["meta"] = meta | gt["meta"]
    atomic_write_json(gt_path, gt)

    print(summarise(gt))
    print(f"\nground truth : {gt_path}")
    print(f"dataset card : {card_path}")
    print(f"elapsed      : {meta['elapsed_seconds']:.1f}s  (validation: OK)")
    if args.push_to_hub:
        url = push_to_hub(gt, args.push_to_hub, token=args.hub_token, private=bool(args.hub_private))
        print(f"pushed       : {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
