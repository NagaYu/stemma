---
license: apache-2.0
pretty_name: Stemma provenance benchmark
size_categories:
  - n<1K
tags:
  - model-provenance
  - lineage
  - safetensors
  - ai-bom
  - model-merging
  - supply-chain
  - benchmark
configs:
  - config_name: default
    data_files:
      - split: pairs
        path: ground_truth.json
---

# Stemma provenance benchmark

A small, fully reproducible benchmark of **model lineages with known ground truth**, used to
measure whether derivation *direction*, *multi-parent merges* and *mixing ratios* can be recovered
from weights alone.

It is not a text dataset. Its records describe relationships **between checkpoints**: which model
was derived from which, by what operation, and — for merges — in what proportion.

## Contents

`ground_truth.json`:

```json
{"models": [{"id": "...", "path": "...", "family": "...", "op": "...",
             "parents": ["..."], "weights": {"parent": 0.7}, "license": "..."}],
 "edges":  [{"parent": "...", "child": "...", "relation": "...", "weight": 0.7}],
 "pairs":  [{"a": "...", "b": "...", "related": true, "direction": "a->b",
             "relation": "sft"}]}
```

- **`models`** — one record per checkpoint, with the operation that produced it and its true
  parents (and mixing weights for merges).
- **`edges`** — the true DAG, one record per parent→child relation.
- **`pairs`** — labelled **ordered** pairs. `direction` is `"a->b"`, `"b->a"`, `"sibling"` or
  `"none"`; `related` is the label for the relatedness/AUC axis; `relation` names the operation so
  accuracy can be reported **per relation type** rather than as a single aggregate.

Relation types cover fine-tuning (SFT/LoRA-style), continued pretraining, quantisation, pruning,
vocabulary extension, linear / TIES / DARE merges, cross-architecture distillation, and unrelated
negatives (including *architecture twins*: same shapes, independent weights — the hardest
false-positive case).

The checkpoints themselves are real `*.safetensors` directories written next to
`ground_truth.json`. They are **not** committed to git (`bench_models/` is in `.gitignore`) because
they run to hundreds of megabytes and are deterministically regenerable.

## Generation procedure

```bash
python scripts/build_bench.py --out-dir bench_models
```

The builder starts from small locally cached public checkpoints (e.g. `openai-community/gpt2`,
`distilgpt2`, `gpt2-medium`, `HuggingFaceTB/SmolLM2-135M-Instruct`, `Qwen/Qwen2.5-0.5B-Instruct`,
`Qwen/Qwen3-0.6B`) and applies **known** operations to them — brief fine-tunes on
`wikitext-2-raw-v1`, quantisation, magnitude pruning, vocabulary extension, and merges with
prescribed mixing ratios — recording each operation as ground truth as it goes. Every randomised
step takes an explicit `seed` (default `0`), so a rebuild reproduces the same lineages.

Ground truth is *constructed*, never inferred: the labels come from the operation that was applied,
not from any Stemma output. Stemma never sees the labels at feature-extraction time.

## Intended use

**In scope.** Measuring provenance-recovery methods: direction accuracy per relation type,
abstention rate, relatedness AUC and FPR@95TPR, merge parent-set precision/recall/F1, mixing MAE,
and bytes/seconds per decision. Comparing against symmetric fingerprint baselines, which are
structurally pinned at 50% on direction.

```bash
python benchmarks/run.py            # -> benchmarks/results.json + figures/*.png
stemma trace <model> --universe bench_models/ground_truth.json
```

**Out of scope.** Training a production provenance classifier and deploying it without refitting.
Any use of a benchmark score as evidence about a specific real-world model. Any enforcement,
takedown or procurement decision.

## Evaluation

Consumed by `benchmarks/run.py`, which writes `benchmarks/results.json` and the table in the
project README. No scores are quoted in this card — they live in the generated artifacts so they
cannot drift from what was measured.

## Limitations

- **Small and synthetic-by-construction.** The lineages are built by applying known operations to
  a handful of small public checkpoints. It is a *controlled* benchmark, which is what makes the
  labels trustworthy and also what limits external validity.
- **Skewed toward the easy half of the problem.** Lossy operations (quantisation, pruning, vocab
  extension) are near-deterministically orientable; scar-free fine-tunes are not. An aggregate
  accuracy over this dataset would let the easy cases hide the hard ones, which is exactly why
  `benchmarks/run.py` reports **per relation type**. See `docs/FINDINGS.md`.
- **Small models only.** Everything is ≤ ~1B parameters so the benchmark runs on CPU in reasonable
  time. Behaviour at frontier scale is untested here.
- **Short fine-tunes.** The fine-tuning edges use brief runs on `wikitext-2-raw-v1`; a
  production-scale SFT run moves weights further and may be *easier* to detect but also further
  from these labels.
- **Non-linear merge recipes are approximations for the linear decomposer.** TIES/DARE ground-truth
  weights are the recipe's nominal coefficients; the linear task-vector model cannot reproduce
  sign-election and trimming exactly, so a non-zero mixing MAE on those rows is expected.
- **License field is metadata, not law.** The `license` field on each record is a synthetic label
  used to exercise rights propagation and conflict detection. It is not legal advice and does not
  describe the licensing of any real derived work.

## Ethics and framing

Records in this dataset are **labels of constructed experiments**, not accusations about anyone's
model. Downstream, Stemma reports statistical evidence with a confidence and **never** a
determination of infringement; a human must review every finding before any action is taken.
Do not use this benchmark, or a score obtained on it, to assert that a real third-party model is a
derivative of another.

## License

Apache-2.0 for `ground_truth.json`, the builder script and the harness. The generated checkpoints
are derived from third-party base models and remain governed by **those** models' licenses — check
each base model's terms before redistributing any generated checkpoint.
