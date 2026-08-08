# Stemma benchmark results

## Headline table

Bench: `/Users/nagaoyuta/Desktop/Claude code/24-Stemma/bench_models` - 190 labelled pairs (108 related, 82 unrelated, 25 of them HARD same-shape-different-init controls); 28 ordered pairs; 4 merges with 3 decoys each. seed=0, quick=False.

| method | AUC | AP | FPR@95TPR | **FPR hard controls** | DIRECTION ACC | MERGE F1 | MIXING MAE | BYTES/DECISION |
|---|---|---|---|---|---|---|---|---|
| **Stemma** | 0.994 | 1.000 | 0.000 | **0.000** | 32.1% (100.0% on answered, 67.9% abstain) | 0.900 | 0.066 | 64.4 MiB |
| cosine | 0.987 | 0.996 | 0.000 | **0.000** | 50% (chance) | n/a | n/a | 15.8 MiB |
| CKA/REEF-style | 0.987 | 0.996 | 0.000 | **0.000** | 50% (chance) | n/a | n/a | 15.8 MiB |
| HuRef-style | 0.981 | 0.994 | 0.000 | **0.000** | 50% (chance) | n/a | n/a | 37.2 MiB |

`n/a` in the merge columns is not a zero: a symmetric fingerprint produces **no mixing coefficients at all**, so there is nothing to score. `50% (chance)` in the direction column is a **structural ceiling** - `cosine(a,b) == cosine(b,a)` by construction, so `baselines.baseline_direction` abstains on every pair and a coin flip is the best a symmetric statistic can do. It is not a tuning failure.

Full download of both checkpoints in a median decision: **513.2 MiB** (from safetensors headers - nothing was downloaded).

## Direction accuracy per relation (Stemma)

docs/FINDINGS.md 5.1 makes this table mandatory: an aggregate would let the easy scar-bearing edges hide the hard scar-free ones.

| relation | group | n | accuracy | acc. on answered | abstained | mean \|llr\| |
|---|---|---|---|---|---|---|
| sft | scar-free | 1 | 0.0% | n/a | 100.0% | 0.02 |
| lora | scar-free | 1 | 0.0% | n/a | 100.0% | 0.01 |
| cpt | scar-free | 1 | 0.0% | n/a | 100.0% | 0.02 |
| quant | scar-bearing | 3 | 100.0% | 100.0% | 0.0% | 2.80 |
| prune | scar-bearing | 2 | 100.0% | 100.0% | 0.0% | 2.65 |
| vocab | scar-bearing | 2 | 100.0% | 100.0% | 0.0% | 4.98 |
| merge | structural / cross-arch | 9 | 0.0% | n/a | 100.0% | 0.01 |
| distilled | structural / cross-arch | 2 | 0.0% | n/a | 100.0% | 0.07 |
| other | other | 7 | 28.6% | 100.0% | 71.4% | 0.75 |

## Outgroup rooting (scar-free relations only)

| variant | n | accuracy | acc. on answered | abstained |
|---|---|---|---|---|
| without outgroup | 3 | 0.0% | n/a | 100.0% |
| **with outgroup** | 3 | 100.0% | 100.0% | 0.0% |

## Merge recovery breakdown

| slice | n | precision | recall | F1 | mixing MAE | residual |
|---|---|---|---|---|---|---|
| all | 4 | 0.917 | 0.917 | 0.900 | 0.066 | 0.583 |
| method=dare | 1 | 1.000 | 1.000 | 1.000 | 0.001 | 0.702 |
| method=slerp | 1 | 1.000 | 1.000 | 1.000 | 0.027 | 0.386 |
| method=ties | 2 | 0.833 | 0.833 | 0.800 | 0.117 | 0.621 |
| parents=2 | 3 | 0.889 | 1.000 | 0.933 | 0.057 | 0.538 |
| parents=3 | 1 | 1.000 | 0.667 | 0.800 | 0.091 | 0.716 |

Baselines: **n/a - symmetric fingerprints produce no mixing coefficients**.

## Transfer cost per decision

| method | decisions | median bytes | total bytes | median s | median full download | reduction |
|---|---|---|---|---|---|---|
| **Stemma** | 190 | 64.4 MiB | 12.6 GiB | 0.034 | 513.2 MiB | 8.0x |
| cosine | 190 | 15.8 MiB | 2.1 GiB | 0.059 | 513.2 MiB | 32.5x |
| CKA/REEF-style | 190 | 15.8 MiB | 2.1 GiB | 0.124 | 513.2 MiB | 32.5x |
| HuRef-style | 190 | 37.2 MiB | 4.9 GiB | 0.102 | 513.2 MiB | 13.8x |
| stemma-direction | 28 | 64.4 MiB | 1.8 GiB | 0.873 | 513.2 MiB | 8.0x |

Non-fatal failures: **0** (none).

Stemma reports statistical evidence about weight-level similarity and derivation direction. It does not establish provenance as fact.

## Figures

- fig1_direction_accuracy.png - Direction accuracy per method. Every baseline sits exactly on the 0.50 chance line because cosine/CKA/HuRef statistics are symmetric in their two arguments; no amount of tuning moves them. Stemma's bar is the aggregate only - see fig6 for the per-relation breakdown that matters.

- fig2_roc.png - Relatedness ROC curves. All four methods can rank related above unrelated; this axis is where the symmetric baselines are genuinely competitive. The dotted 95%-TPR line is the operating point at which the hard-control false-positive rate in the headline table is measured.

- fig3_merge_recovery.png - Recovered vs true mixing coefficient for every candidate of every ground-truth merge, coloured by merge method. Points on the y=x line are correctly weighted parents; points on the x=0 axis are decoys, and a decoy above zero is a false parent. No symmetric baseline can produce a single point on this plot.

- fig4_transfer.png - Bytes per decision, log scale. The full-download bar is the sum of both checkpoints' shard sizes read from the safetensors header, never downloaded. On a tiny synthetic bench the fixed header cost dominates and the reduction factor is small by construction; the reduction grows with checkpoint size.

- fig5_summary_matrix.png - Capability matrix. The crosses in the direction, multi-parent and mixing-ratio columns are structural: those statistics are symmetric functions of an unordered pair, so the questions are not merely hard for them, they are unanswerable. The sub-1%-transfer column is measured on this run, not asserted, so a tiny synthetic bench (where fixed header cost dominates) will show a cross.

- fig6_direction_by_relation.png - Direction accuracy split by ground-truth relation, with the scar-bearing group (quantise/prune/vocab - lossy, irreversible, so the scar can only appear downstream) separated from the scar-free group (sft/lora/cpt, where docs/FINDINGS.md measured the delta-spectrum evidence at 1e-3..1e-2 and predicts weak identifiability). The aggregate bar in fig1 is the average of these; this is the figure to read.
