#!/usr/bin/env python3
"""Package Stemma's fitted artifacts as a Hugging Face **model** repository.

What lands in the repo is deliberately tiny: a fitted :class:`DirectionModel`
(a few dozen floats), the frozen sketch coordinate system, a prebuilt
:class:`~stemma.phylogeny.SketchIndex` over the benchmark universe, a fit report
and a model card. No checkpoint weights are ever redistributed.

Claim: infra -- this script ships nothing new; it packages the direction and
low-transfer machinery so a third party can reproduce a verdict without
refitting or re-sketching a universe.

**Dry run by default.** Without an explicit ``--push`` the script builds the
artifacts locally into ``--out-dir`` and prints exactly what *would* be
uploaded. It contacts the Hub only when ``--push`` is given.

Usage::

    python scripts/push_model.py --repo-id org/stemma-direction \\
        --bench-dir bench_models --fit --out-dir hf_model_export
    python scripts/push_model.py --repo-id org/stemma-direction --push
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Make `import stemma` work when the script is run from anywhere.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stemma.types import (  # noqa: E402  (path bootstrap must come first)
    DEPTH_BUCKETS,
    DIRECTION_FEATURES,
    FEATURES_PER_SLOT,
    N_GLOBAL_FEATURES,
    ROLES,
    SKETCH_DIM,
    SKETCH_VERSION,
)
from stemma.utils import atomic_write_json, human_bytes, set_seed, short_id  # noqa: E402

#: Files the packaged repo is expected to contain, in presentation order.
ARTIFACT_ORDER: Tuple[str, ...] = (
    "README.md",
    "direction_model.json",
    "sketch_config.json",
    "fit_report.json",
    "sketch_index.json",
    "sketch_index.npz",
)

#: Threshold used when reporting abstention in the fit report. Matches the CLI
#: default (``stemma direction --abstain``).
ABSTAIN = 0.5


# --------------------------------------------------------------------------- #
# ground truth
# --------------------------------------------------------------------------- #


def load_ground_truth(bench_dir: str | os.PathLike[str]) -> Dict[str, Any]:
    """Read ``<bench_dir>/ground_truth.json`` produced by ``scripts/build_bench.py``.

    Claim: infra -- the fitted model is only meaningful relative to labels that
    were *constructed* rather than inferred, so the ground-truth file is the one
    input this script refuses to synthesise.
    """
    path = Path(bench_dir).expanduser()
    if path.is_dir():
        path = path / "ground_truth.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"ground truth not found at {path}. Build it first with:\n"
            f"    python scripts/build_bench.py --out-dir {Path(bench_dir)}"
        )
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    if not isinstance(doc, dict):
        raise ValueError(f"{path} is not a JSON object")
    return doc


def universe_refs(gt: Dict[str, Any], bench_dir: str | os.PathLike[str]) -> List[str]:
    """Resolve every benchmark model to a reference the loader can open.

    Claim: low-transfer -- a local directory and a Hub repo id go down the same
    ``SafeTensorsSource`` path, so an index built here is byte-comparable with
    one built over the wire.

    Prefers each record's on-disk ``path`` (absolute, or relative to
    ``bench_dir``) and falls back to its ``id``.
    """
    base = Path(bench_dir).expanduser()
    if base.is_file():
        base = base.parent
    refs: List[str] = []
    seen: set[str] = set()
    for entry in gt.get("models", []) or []:
        if isinstance(entry, str):
            ref = entry
        elif isinstance(entry, dict):
            raw = entry.get("path") or entry.get("id") or entry.get("model_id")
            if not raw:
                continue
            p = Path(str(raw)).expanduser()
            if not p.is_absolute():
                candidate = base / p
                ref = str(candidate) if candidate.exists() else str(raw)
            else:
                ref = str(p)
        else:
            continue
        if ref not in seen:
            seen.add(ref)
            refs.append(ref)
    return refs


def _ref_alias(gt: Dict[str, Any], bench_dir: str | os.PathLike[str]) -> Dict[str, str]:
    """Map every spelling of a benchmark model (id or path) to one canonical ref.

    Claim: infra -- ``ground_truth.json`` labels pairs by ``id`` while the models
    live at ``path``; without this map every pair would fail to open.
    """
    base = Path(bench_dir).expanduser()
    if base.is_file():
        base = base.parent
    alias: Dict[str, str] = {}
    for entry in gt.get("models", []) or []:
        if not isinstance(entry, dict):
            continue
        raw_path = entry.get("path")
        canonical = None
        if raw_path:
            p = Path(str(raw_path)).expanduser()
            if not p.is_absolute():
                cand = base / p
                p = cand if cand.exists() else p
            canonical = str(p)
        canonical = canonical or str(entry.get("id") or "")
        if not canonical:
            continue
        for key in (entry.get("id"), entry.get("model_id"), raw_path, canonical):
            if key:
                alias[str(key)] = canonical
    return alias


# --------------------------------------------------------------------------- #
# feature extraction + fitting
# --------------------------------------------------------------------------- #


def _import_direction():
    """Import :mod:`stemma.direction` with an actionable error when it is absent.

    Claim: direction -- the whole point of this repo is the fitted combiner, so
    a missing direction module must fail loudly rather than silently shipping a
    default prior labelled as fitted.
    """
    import importlib

    try:
        return importlib.import_module("stemma.direction")
    except Exception as exc:  # pragma: no cover - depends on checkout state
        raise SystemExit(
            f"error: could not import stemma.direction ({type(exc).__name__}: {exc}).\n"
            "       DirectionModel lives there, so it is required both to --fit and to "
            "ship the\n       hand-set DirectionModel.default() priors. Fix that module "
            "first; there is\n       nothing meaningful to package without it."
        ) from exc


def labelled_pairs(gt: Dict[str, Any], alias: Dict[str, str]) -> List[Dict[str, Any]]:
    """Select the ordered, oriented pairs usable as supervision.

    Claim: direction -- only pairs whose ground truth states an *order*
    (``a->b`` / ``b->a``) can teach the combiner anything; ``sibling`` and
    ``none`` rows carry no direction label and are dropped here rather than
    being folded in as noise.
    """
    out: List[Dict[str, Any]] = []
    for p in gt.get("pairs", []) or []:
        if not isinstance(p, dict):
            continue
        d = str(p.get("direction", ""))
        if d not in ("a->b", "b->a"):
            continue
        a = alias.get(str(p.get("a")), str(p.get("a")))
        b = alias.get(str(p.get("b")), str(p.get("b")))
        if not a or not b or a == b:
            continue
        out.append({"a": a, "b": b, "direction": d, "relation": str(p.get("relation", "none"))})
    return out


def extract_features(
    pairs: Sequence[Dict[str, Any]],
    *,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[List[Dict[str, float]], List[Dict[str, Any]]]:
    """Range-read each labelled pair and reduce it to the antisymmetric features.

    Claim: direction -- the features are built as ``g(A,B) - g(B,A)``, so
    ``f(b, a) == -f(a, b)`` and the combiner cannot learn a positional bias from
    the order the pairs happen to be written in.

    Returns ``(feature_dicts, kept_pairs)``; pairs whose evidence could not be
    collected are dropped and reported, never imputed.
    """
    direction = _import_direction()
    feats: List[Dict[str, float]] = []
    kept: List[Dict[str, Any]] = []
    for i, p in enumerate(pairs, 1):
        try:
            ev = direction.collect_pair_evidence(p["a"], p["b"], seed=seed)
            f = direction.direction_features(ev)
        except Exception as exc:
            if verbose:
                print(f"  [{i}/{len(pairs)}] {short_id(p['a'], 28)} | {short_id(p['b'], 28)}"
                      f"  SKIPPED ({type(exc).__name__}: {exc})")
            continue
        feats.append({k: float(f.get(k, 0.0)) for k in DIRECTION_FEATURES})
        kept.append(p)
        if verbose:
            print(f"  [{i}/{len(pairs)}] {short_id(p['a'], 28)} | {short_id(p['b'], 28)}"
                  f"  ok  ({p['relation']})")
    return feats, kept


def _row(feat: Dict[str, float]) -> np.ndarray:
    """Order one feature dict into the frozen ``DIRECTION_FEATURES`` layout.

    Claim: direction -- the fitted weight vector is only interpretable against a
    fixed feature order, which is why the order lives in ``stemma.types``.
    """
    return np.array([float(feat.get(k, 0.0)) for k in DIRECTION_FEATURES], dtype=np.float64)


def split_pairs(n: int, *, test_size: float = 0.25, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic train/test split over *pairs* (never over mirrored rows).

    Claim: low-false-positive -- because each pair is also used in its mirrored
    form, splitting over rows would leak a pair's mirror into the test set and
    inflate held-out accuracy. Splitting over pairs is what keeps the reported
    number honest.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_test = int(round(max(0.0, min(0.9, float(test_size))) * n))
    n_test = min(max(n_test, 1 if n >= 4 else 0), max(n - 1, 0))
    return order[n_test:], order[:n_test]


def fit_direction_model(
    feats: Sequence[Dict[str, float]],
    pairs: Sequence[Dict[str, Any]],
    *,
    l2: float = 1.0,
    test_size: float = 0.25,
    seed: int = 0,
) -> Tuple[Any, Dict[str, Any]]:
    """Fit the combiner on the training pairs and score it on the held-out ones.

    Claim: direction -- this is the only place a number attached to the word
    "accuracy" is produced, and it is produced on pairs the fit never saw, per
    relation type, with the abstention rate alongside it (a confident wrong
    answer is worse than "unknown" for this application).
    """
    direction = _import_direction()
    DirectionModel = direction.DirectionModel

    n = len(feats)
    if n < 4:
        raise SystemExit(
            f"error: only {n} usable labelled pair(s); need at least 4 to fit. "
            "Rebuild the benchmark or run without --fit."
        )
    set_seed(seed)
    train_idx, test_idx = split_pairs(n, test_size=test_size, seed=seed)

    def _stack(idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rows: List[np.ndarray] = []
        ys: List[float] = []
        for i in idx:
            x = _row(feats[int(i)])
            y = 1.0 if pairs[int(i)]["direction"] == "a->b" else 0.0
            rows.append(x)
            ys.append(y)
            # Mirrored copy: features are antisymmetric, so the mirror is -x
            # with the opposite label. Included so the fit sees a balanced,
            # sign-symmetric design matrix and cannot learn an intercept bias.
            rows.append(-x)
            ys.append(1.0 - y)
        return np.asarray(rows, dtype=np.float64), np.asarray(ys, dtype=np.float64)

    X_tr, y_tr = _stack(train_idx)
    model = DirectionModel.fit(X_tr, y_tr, l2=float(l2))

    def _score(idx: np.ndarray) -> Dict[str, Any]:
        correct = 0
        decided = 0
        decided_correct = 0
        per_rel: Dict[str, List[int]] = {}
        for i in idx:
            i = int(i)
            llr = float(model.llr(feats[i]))
            truth_pos = pairs[i]["direction"] == "a->b"
            pred_pos = llr > 0.0
            ok = int(pred_pos == truth_pos)
            correct += ok
            if abs(llr) >= ABSTAIN:
                decided += 1
                decided_correct += ok
            rel = pairs[i]["relation"]
            bucket = per_rel.setdefault(rel, [0, 0, 0])  # ok, n, decided
            bucket[0] += ok
            bucket[1] += 1
            bucket[2] += int(abs(llr) >= ABSTAIN)
        n_idx = max(len(idx), 1)
        return {
            "n": int(len(idx)),
            "accuracy": correct / n_idx,
            "abstain_rate": 1.0 - decided / n_idx,
            "accuracy_on_decided": (decided_correct / decided) if decided else None,
            "per_relation": {
                rel: {
                    "n": v[1],
                    "accuracy": v[0] / max(v[1], 1),
                    "abstain_rate": 1.0 - v[2] / max(v[1], 1),
                }
                for rel, v in sorted(per_rel.items())
            },
        }

    report: Dict[str, Any] = {
        "fitted": True,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": int(seed),
        "l2": float(l2),
        "abstain_threshold": ABSTAIN,
        "label_convention": "y=1 means A is the parent of B (llr > 0)",
        "feature_names": list(DIRECTION_FEATURES),
        "n_pairs_total": int(n),
        "n_pairs_train": int(len(train_idx)),
        "n_pairs_test": int(len(test_idx)),
        "n_rows_train": int(X_tr.shape[0]),
        "train": _score(train_idx),
        "test": _score(test_idx),
        "weights": {k: float(w) for k, w in zip(DIRECTION_FEATURES, np.asarray(model.weights).ravel())},
        "bias": float(getattr(model, "bias", 0.0)),
        "note": (
            "Accuracy is reported per relation type as well as in aggregate: an "
            "aggregate would let lossy, near-deterministically orientable edges "
            "(quantisation, pruning, vocab extension) hide the scar-free "
            "fine-tuning edges. See docs/FINDINGS.md."
        ),
    }
    return model, report


def default_direction_model() -> Tuple[Any, Dict[str, Any]]:
    """Fall back to the hand-set priors when no benchmark is available.

    Claim: direction -- the packaged artifact must always say whether its
    weights were *fitted* or *hand-set*; shipping priors labelled as fitted
    would misrepresent the evidence behind every downstream verdict.
    """
    direction = _import_direction()
    model = direction.DirectionModel.default()
    report = {
        "fitted": False,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "feature_names": list(DIRECTION_FEATURES),
        "abstain_threshold": ABSTAIN,
        "note": (
            "These are DirectionModel.default() hand-set priors, NOT fitted "
            "weights. No held-out accuracy is claimed. Refit with "
            "`python scripts/push_model.py --fit --bench-dir bench_models`."
        ),
    }
    return model, report


# --------------------------------------------------------------------------- #
# artifacts
# --------------------------------------------------------------------------- #


def write_sketch_config(out_dir: Path) -> Path:
    """Emit the frozen sketch coordinate system next to the fitted model.

    Claim: low-false-positive -- two sketches are only comparable when they were
    written in the same (role, depth-bucket) coordinate system, so the version
    and layout travel with the artifacts instead of being assumed.
    """
    payload = {
        "version": SKETCH_VERSION,
        "ROLES": list(ROLES),
        "DEPTH_BUCKETS": list(DEPTH_BUCKETS),
        "FEATURES_PER_SLOT": int(FEATURES_PER_SLOT),
        "SKETCH_DIM": int(SKETCH_DIM),
        "N_GLOBAL_FEATURES": int(N_GLOBAL_FEATURES),
        "n_slots": len(ROLES) * len(DEPTH_BUCKETS),
        "metric": "cosine",
    }
    path = out_dir / "sketch_config.json"
    atomic_write_json(path, payload)
    return path


def build_sketch_index(
    refs: Sequence[str],
    out_dir: Path,
    *,
    seed: int = 0,
    max_rows: int = 2048,
    verbose: bool = True,
) -> Tuple[Optional[Any], Dict[str, Any]]:
    """Sketch the benchmark universe once and persist a nearest-neighbour index.

    Claim: low-transfer -- shipping the index means a consumer answers
    "who might the parents be?" with zero bytes over the wire; the universe is
    Range-read once, here, and never again.
    """
    from stemma.phylogeny import SketchIndex
    from stemma.sketch import sketch_model
    from stemma.types import TransferStats

    sketches = []
    failures: List[Dict[str, str]] = []
    total = TransferStats()
    for i, ref in enumerate(refs, 1):
        try:
            sk = sketch_model(ref, max_rows=max_rows, seed=seed)
        except Exception as exc:
            failures.append({"ref": ref, "error": f"{type(exc).__name__}: {exc}"})
            if verbose:
                print(f"  [{i}/{len(refs)}] {short_id(ref, 44)}  FAILED ({type(exc).__name__})")
            continue
        sketches.append(sk)
        if getattr(sk, "stats", None) is not None:
            total = total.add(sk.stats)
        if verbose:
            print(f"  [{i}/{len(refs)}] {short_id(ref, 44)}  ok")

    if not sketches:
        return None, {"indexed": 0, "failed": failures, "backend": None}

    idx = SketchIndex()
    idx.add(sketches)
    idx.save(out_dir / "sketch_index")
    return idx, {
        "indexed": len(sketches),
        "failed": failures,
        "backend": idx.backend,
        "dim": int(idx.dim),
        "transfer": {
            "bytes_read": int(total.bytes_read),
            "requests": int(total.requests),
            "full_size_bytes": int(total.full_size_bytes),
        },
    }


def render_model_card(
    out_dir: Path,
    *,
    repo_id: str,
    report: Dict[str, Any],
    index_info: Dict[str, Any],
) -> Path:
    """Write the repo's ``README.md``: the project model card plus this build.

    Claim: infra -- the card must carry HF frontmatter and must state whether
    the shipped weights were fitted or hand-set, so that no consumer can mistake
    a prior for a measurement.
    """
    source = REPO_ROOT / "MODEL_CARD.md"
    if source.is_file():
        body = source.read_text(encoding="utf-8").rstrip() + "\n"
    else:  # pragma: no cover - MODEL_CARD.md ships with the repo
        body = (
            "---\n"
            "license: apache-2.0\n"
            "library_name: stemma\n"
            "tags: [model-provenance, lineage, safetensors, ai-bom, model-merging, supply-chain]\n"
            "---\n\n"
            "# Stemma direction model\n\n"
            "Fitted artifacts for the Stemma model-provenance tool. This repository "
            "contains no language model and no third-party weights.\n"
        )

    lines: List[str] = ["", "---", "", "## This build", "", f"- Repository: `{repo_id}`",
                        f"- Generated (UTC): `{report.get('generated_utc', 'unknown')}`",
                        f"- Sketch format: `{SKETCH_VERSION}` (dim {SKETCH_DIM})"]
    if report.get("fitted"):
        test = report.get("test", {}) or {}
        train = report.get("train", {}) or {}
        acc = test.get("accuracy")
        dec = test.get("accuracy_on_decided")
        lines += [
            f"- Direction weights: **fitted** (l2={report.get('l2')}, seed={report.get('seed')})",
            f"- Pairs: {report.get('n_pairs_train')} train / {report.get('n_pairs_test')} held out"
            f" (of {report.get('n_pairs_total')} labelled ordered pairs)",
            f"- Held-out accuracy: **{acc:.3f}**" if isinstance(acc, float) else "- Held-out accuracy: n/a",
            f"- Held-out accuracy on non-abstained: "
            + (f"**{dec:.3f}**" if isinstance(dec, float) else "n/a")
            + f" (abstain rate {float(test.get('abstain_rate', 0.0)):.3f}, |llr| < {ABSTAIN})",
            f"- Training accuracy: "
            + (f"{float(train.get('accuracy', 0.0)):.3f}" if train else "n/a"),
            "",
            "Held-out accuracy **per relation type** (an aggregate would let the lossy, "
            "near-deterministically orientable edges hide the scar-free ones):",
            "",
            "| relation | n | accuracy | abstain rate |",
            "|---|---:|---:|---:|",
        ]
        for rel, v in (test.get("per_relation", {}) or {}).items():
            lines.append(
                f"| `{rel}` | {v.get('n', 0)} | {float(v.get('accuracy', 0.0)):.3f} "
                f"| {float(v.get('abstain_rate', 0.0)):.3f} |"
            )
        if not (test.get("per_relation") or {}):
            lines.append("| _(no held-out pairs)_ | 0 | - | - |")
    else:
        lines += [
            "- Direction weights: **hand-set priors** (`DirectionModel.default()`), **not fitted**,",
            "  and that is a deliberate, measured choice rather than a missing step.",
            "",
            "### Why the priors and not a fit",
            "",
            "Fitting an L2 logistic combiner on the benchmark's labelled ordered pairs was tried",
            "and **lost**. On the same held-out split the priors scored **1.000** accuracy on",
            "decided pairs against the fit's **0.500** — chance. Read that with its sample size:",
            "the priors abstained on 5 of 7 and decided only 2, so the accuracy gap rests on 2",
            "decisions against 4 and is suggestive, not conclusive.",
            "",
            "The decisive evidence is *what the fit learned*. With 13 features and 21 training",
            "pairs the problem is underdetermined, and the fit assigned `lattice_asym` a",
            "**negative** weight — asserting that the quantised model is the parent. That is",
            "physically impossible: dequantisation cannot restore what rounding destroyed, so the",
            "scar can only ever appear downstream. It also put its largest weight on the statistic",
            "already measured as the weakest. A prior encoding a physical impossibility beats a",
            "coefficient fitted on 21 examples.",
            "",
            "`--fit` remains available for anyone with a substantially larger labelled corpus:",
            "  `python scripts/push_model.py --repo-id <id> --bench-dir bench_models --fit`",
        ]

    lines += [
        "",
        f"- Prebuilt index: {index_info.get('indexed', 0)} sketches"
        f" (backend `{index_info.get('backend')}`,"
        f" {len(index_info.get('failed', []) or [])} model(s) unreadable)",
        "",
        "Full numbers, including the exact split and per-relation breakdown, are in "
        "`fit_report.json` in this repository. The wider benchmark (relatedness AUC / "
        "FPR@95TPR, merge F1 and mixing MAE, bytes per decision) is regenerated with "
        "`python benchmarks/run.py`.",
        "",
        "## What weight geometry cannot do",
        "",
        "Two structural limits were measured after this repository was first published, and they",
        "bound how the artifacts here should be used:",
        "",
        "1. **Direction is near-deterministic only for *lossy* operations.** Quantisation,",
        "   pruning and vocabulary extension score 100%; scar-free SFT/LoRA/CPT edges abstain",
        "   (mean |llr| ~0.02). The estimator declines rather than guessing, which is the correct",
        "   failure mode for provenance.",
        "2. **Outgroup rooting is invalid for merge children.** Rooting assumes descendants drift",
        "   monotonically away from the root, but merging is a *contraction toward the centroid*:",
        "   `0.6*sft + 0.4*cpt` partly cancels two perturbations and lands **closer to the root",
        "   than either parent** (root→sft 0.000820, root→cpt 0.001610, root→merge 0.000678).",
        "   Every correctly chosen sibling outgroup then pushes the answer the *wrong* way.",
        "   Direction for a merged model must come from the **decomposition**, not from distance",
        "   geometry — merge precision **1.000**, DARE mixing MAE **0.0004**.",
        "",
        "Full derivations, with the measurements that produced them, are in",
        "[`docs/FINDINGS.md`](https://github.com/NagaYu/stemma/blob/main/docs/FINDINGS.md).",
        "",
        "## Scope and ethics",
        "",
        "Stemma reports **statistical evidence with a confidence**, never a determination of",
        "infringement or licence non-compliance. Weight-level similarity and derivation direction",
        "are inferences from a small sample of tensors and can be wrong. A human must review every",
        "finding before any action is taken.",
        "",
    ]

    path = out_dir / "README.md"
    path.write_text(body + "\n".join(lines), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# upload plan
# --------------------------------------------------------------------------- #


def upload_plan(out_dir: Path) -> List[Tuple[str, int]]:
    """List every file that would be uploaded, with its size, in stable order.

    Claim: infra -- a dry run is only useful if it is exhaustive, so this walks
    the staging directory rather than echoing the files the script *meant* to
    write.
    """
    files: List[Tuple[str, int]] = []
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            files.append((str(p.relative_to(out_dir)), p.stat().st_size))
    rank = {name: i for i, name in enumerate(ARTIFACT_ORDER)}
    files.sort(key=lambda kv: (rank.get(kv[0], len(ARTIFACT_ORDER)), kv[0]))
    return files


def print_plan(plan: Sequence[Tuple[str, int]], *, repo_id: str, private: bool,
               out_dir: Path, pushing: bool) -> None:
    """Print the exact upload manifest.

    Claim: infra -- publishing is a side effect on someone else's namespace, so
    the default path shows the manifest and stops.
    """
    verb = "UPLOADING" if pushing else "WOULD UPLOAD (dry run)"
    print("")
    print("=" * 72)
    print(f"{verb} -> https://huggingface.co/{repo_id}")
    print(f"  repo_type = model    private = {bool(private)}")
    print(f"  staged in = {out_dir}")
    print("-" * 72)
    total = 0
    for name, size in plan:
        total += size
        print(f"  {human_bytes(size):>10}  {name}")
    print("-" * 72)
    print(f"  {len(plan)} file(s), {human_bytes(total)} total")
    print("=" * 72)
    if not pushing:
        print("")
        print("Dry run: nothing was uploaded and no repo was created.")
        print("Re-run with --push to create the repo and upload the files above.")


def do_push(out_dir: Path, *, repo_id: str, private: bool, token: Optional[str]) -> str:
    """Create the model repo if needed and upload the staged folder.

    Claim: infra -- ``create_repo(exist_ok=True)`` + ``upload_folder`` makes
    republishing idempotent, which is what lets the benchmark and the published
    artifacts be regenerated together without hand-managed repo state.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=bool(private), exist_ok=True)
    url = api.upload_folder(
        folder_path=str(out_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Publish Stemma direction model, sketch config and prebuilt index",
        ignore_patterns=["__pycache__/*", "*.pyc", ".DS_Store"],
    )
    return str(url)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Command-line surface for the model packager.

    Claim: infra.
    """
    p = argparse.ArgumentParser(
        prog="push_model.py",
        description=(
            "Package Stemma's fitted DirectionModel, sketch config and prebuilt "
            "SketchIndex as a Hugging Face model repo. Dry run by default."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Nothing is uploaded unless --push is given.",
    )
    p.add_argument("--repo-id", required=True, metavar="ORG/NAME",
                   help="target Hugging Face model repo id")
    p.add_argument("--bench-dir", default="bench_models", metavar="DIR",
                   help="benchmark directory containing ground_truth.json (default: bench_models)")
    p.add_argument("--out-dir", default="hf_model_export", metavar="DIR",
                   help="local staging directory for the packaged repo (default: hf_model_export)")
    p.add_argument("--fit", action="store_true",
                   help="fit the DirectionModel from the benchmark's labelled ordered pairs "
                        "(otherwise DirectionModel.default() hand-set priors are shipped)")
    p.add_argument("--l2", type=float, default=1.0, help="L2 regularisation for the fit (default: 1.0)")
    p.add_argument("--test-size", type=float, default=0.25, dest="test_size",
                   help="held-out fraction of labelled pairs (default: 0.25)")
    p.add_argument("--max-pairs", type=int, default=0, dest="max_pairs",
                   help="cap the number of labelled pairs used (0 = no cap)")
    p.add_argument("--max-rows", type=int, default=2048, dest="max_rows",
                   help="rows sub-sampled per tensor when sketching (default: 2048)")
    p.add_argument("--seed", type=int, default=0, help="seed for sampling and the split (default: 0)")
    p.add_argument("--skip-index", action="store_true",
                   help="do not build the prebuilt SketchIndex")
    p.add_argument("--private", action="store_true", help="create the repo as private")
    p.add_argument("--token", default=None, metavar="TOKEN",
                   help="Hugging Face token; defaults to $HF_TOKEN / $HUGGINGFACE_HUB_TOKEN "
                        "or your cached `huggingface-cli login` credentials")
    p.add_argument("--push", action="store_true",
                   help="actually create the repo and upload (without this the script is a dry run)")
    p.add_argument("--quiet", action="store_true", help="less progress output")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Build the artifacts, print the upload manifest, and upload only if asked.

    Claim: infra -- one reproducible command turns a benchmark run into a
    published, versioned artifact carrying its own fit report, so a downstream
    verdict can always be traced back to the measurement that produced it.
    """
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    verbose = not args.quiet

    out_dir = Path(args.out_dir).expanduser()
    if out_dir.exists() and not out_dir.is_dir():
        print(f"error: --out-dir {out_dir} exists and is not a directory", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    gt: Dict[str, Any] = {}
    refs: List[str] = []
    bench_available = True
    try:
        gt = load_ground_truth(args.bench_dir)
        refs = universe_refs(gt, args.bench_dir)
    except FileNotFoundError as exc:
        bench_available = False
        if args.fit:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"note: {exc}")
        print("note: continuing without a benchmark (hand-set priors, no index).")

    # --- (a) direction model ------------------------------------------------
    if args.fit:
        pairs = labelled_pairs(gt, _ref_alias(gt, args.bench_dir))
        if args.max_pairs and len(pairs) > args.max_pairs:
            pairs = pairs[: args.max_pairs]
        if verbose:
            print(f"collecting direction evidence for {len(pairs)} labelled ordered pair(s)")
        feats, kept = extract_features(pairs, seed=args.seed, verbose=verbose)
        if verbose:
            print(f"  usable pairs: {len(kept)}/{len(pairs)}")
        model, report = fit_direction_model(
            feats, kept, l2=args.l2, test_size=args.test_size, seed=args.seed
        )
        test = report.get("test", {})
        acc = test.get("accuracy")
        dec = test.get("accuracy_on_decided")
        print("")
        print(f"fit: {report['n_pairs_train']} train / {report['n_pairs_test']} held-out pairs")
        if isinstance(acc, float):
            print(f"     held-out accuracy          {acc:.3f}")
        if isinstance(dec, float):
            print(f"     held-out acc (non-abstain) {dec:.3f}"
                  f"   abstain rate {float(test.get('abstain_rate', 0.0)):.3f}")
        for rel, v in (test.get("per_relation", {}) or {}).items():
            print(f"     {rel:<22} n={v['n']:<4} acc={float(v['accuracy']):.3f}"
                  f"  abstain={float(v['abstain_rate']):.3f}")
    else:
        model, report = default_direction_model()
        print("note: shipping DirectionModel.default() hand-set priors (no --fit).")

    model.save(out_dir / "direction_model.json")
    atomic_write_json(out_dir / "fit_report.json", report)

    # --- (b) sketch config + prebuilt index ---------------------------------
    write_sketch_config(out_dir)
    index_info: Dict[str, Any] = {"indexed": 0, "failed": [], "backend": None}
    if args.skip_index or not bench_available or not refs:
        if verbose:
            print("note: prebuilt SketchIndex skipped "
                  f"({'--skip-index' if args.skip_index else 'no benchmark universe'})")
        for stale in ("sketch_index.npz", "sketch_index.json"):
            p = out_dir / stale
            if p.exists():
                p.unlink()
    else:
        if verbose:
            print(f"sketching {len(refs)} benchmark model(s) for the prebuilt index")
        _, index_info = build_sketch_index(
            refs, out_dir, seed=args.seed, max_rows=args.max_rows, verbose=verbose
        )
        if index_info.get("indexed"):
            tr = index_info.get("transfer", {})
            if tr.get("full_size_bytes"):
                print(f"  index: {index_info['indexed']} sketches, read "
                      f"{human_bytes(tr['bytes_read'])} of {human_bytes(tr['full_size_bytes'])}")

    # --- (c) model card -----------------------------------------------------
    render_model_card(out_dir, repo_id=args.repo_id, report=report, index_info=index_info)

    # --- (d) push (or not) --------------------------------------------------
    plan = upload_plan(out_dir)
    print_plan(plan, repo_id=args.repo_id, private=args.private, out_dir=out_dir,
               pushing=bool(args.push))
    if not args.push:
        return 0

    try:
        url = do_push(out_dir, repo_id=args.repo_id, private=args.private, token=token)
    except Exception as exc:
        print(f"error: upload failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("")
    print(f"pushed: {url}")
    print(f"        https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
