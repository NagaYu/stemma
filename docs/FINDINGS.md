# Measured findings that shaped Stemma's design

These are real measurements taken during development, not projections. They are
recorded here because two of them **narrow what the project can honestly claim**,
and one of them motivated an extra component (outgroup rooting).

All measurements below were taken over HTTP Range reads against public Hugging
Face repos — no checkpoint was fully downloaded.

## 1. Direction evidence is not uniformly strong. It is strong exactly where an operation is *lossy*.

Stemma's four evidence families do not contribute equally:

| family | mechanism | strength | why |
|---|---|---|---|
| (c) quantisation / pruning scars | value lattice, exact-zero superset | **near-deterministic** | lossy and irreversible: you cannot un-quantise or un-prune, so the scar can only ever *appear* downstream |
| (b) orphan embeddings | untrained rows only in the wider vocab | **near-deterministic** | freshly initialised rows are statistically distinguishable from trained ones, and vocabulary only grows |
| (d) fossils | dead neurons, bit-identical tensors | **strong when present** | a dead unit stays dead downstream |
| (a) Δ-spectrum | subspace energy, effective rank, norm growth | **weak** | measured below |

## 2. Norm growth is consistent *within* a pair and sign-flipped *across* pairs. It is not a usable prior.

8 shared 2D tensors per pair, 768 sampled rows each, `Δ = child − parent`:

| statistic | `Qwen2.5-0.5B → -Instruct` | `SmolLM2-135M → -Instruct` |
|---|---|---|
| `log‖B‖_F − log‖A‖_F` | **−0.0171 (0/8 positive)** | **+0.0113 (8/8 positive)** |
| max\|value\| growth | −0.0064 (1/8) | −0.0006 (2/8) |
| row-norm Gini growth | −0.0009 (3/8) | −0.0015 (0/8) |
| outlier-channel growth | 0.0 (0/8) | 0.0 (0/8) |
| near-dead-row growth | 0.0 (0/8) | 0.0 (0/8) |

Both pairs are unambiguously base → instruct-tuned. The norm statistic is
perfectly consistent inside each pair and points in **opposite directions**
between them — almost certainly a weight-decay/recipe artefact. A hand-set
prior on `norm_growth_asym` would therefore have been right on one family and
wrong on the other. It is kept as a *fitted* feature with a small weight, never
as a hand-set sign.

## 3. The Δ-subspace statistic is real but small, and its sign is the opposite of the naive intuition.

The intuition "a child preserves its parent's dominant subspace, so Δ should lie
more inside the *parent's* top subspace" is **not** what happens. Because the
child has already absorbed Δ, the child's own top-k right-singular subspace is
marginally better aligned with Δ:

```
synthetic low-rank + noise delta, k=64:  E(Δ|parent)=0.1165  E(Δ|child)=0.1365  diff=-0.0200
real Qwen base→instruct, k=48, mean over 8 tensors:                            diff=+0.0022 (6/8 positive)
```

Magnitude `~10⁻³–10⁻²` against a per-tensor spread of the same order. It is a
genuine tiebreaker, not a decisive signal, and it is weighted accordingly.

## 4. Consequence: two-model, scar-free direction is only weakly identifiable — so Stemma adds outgroup rooting.

For a pure SFT/LoRA/continued-pretraining edge, both models are the same dtype,
the same shape, the same vocabulary, and carry no lossy scar. Everything that
distinguishes them is a soft spectral statistic. We state this plainly rather
than reporting one blended direction number.

Stemma is not restricted to pairs, though — it indexes a **universe** of models.
That makes the classical phylogenetic fix available: **outgroup rooting**. If a
third relative `C` descends from the same ancestor, then for the pair `(A, B)`

```
d(A, C) < d(B, C)  consistently over tensors  ⇒  A sits closer to the root
```

because a sibling's distance is dominated by the shared ancestral component.
This is implemented as evidence family **(e)** and used automatically whenever
the candidate universe supplies a usable outgroup. It is reported as a separate
column in the benchmark so its contribution is visible rather than blended away.

## 5. Reporting rules this imposes on the benchmark

1. Direction accuracy is reported **per relation type**, never only as one
   aggregate — an aggregate would let the easy scar-bearing edges hide the hard
   scar-free ones.
2. Abstention is a first-class outcome. Accuracy-on-non-abstained and the
   abstention rate are both reported; a confident wrong answer is worse than
   "unknown" for this application.
3. Cross-architecture distillation is reported as a case where weight-level
   lineage is **expected to be weak**, and is scored honestly rather than
   quietly excluded.
4. The baselines' 50% direction score is labelled a **structural ceiling**
   (these statistics are symmetric by construction), not a tuning failure.

## 5b. The predictions in this document were then confirmed by the benchmark

Sections 1–4 were written *before* the direction estimator existed. The full run over 20 real
checkpoints (190 labelled pairs) reproduced them:

| prediction made here | measured |
|---|---|
| lossy operations carry direction | quant **100%**, prune **100%**, vocab **100%** |
| scar-free edges are weakly identifiable | sft / lora / cpt **0%, 100% abstained**, mean \|llr\| ≈ 0.02 |
| outgroup rooting closes that gap | **0% → 100%** on the same three edges |
| symmetric baselines cannot answer | cosine / CKA / HuRef pinned at the 50% structural ceiling |
| hard same-shape controls must not false-positive | **FPR = 0.000** on 25 controls |

The abstention behaviour is the part worth emphasising: on scar-free edges Stemma is never
confidently *wrong* in this run — it declines. For a provenance tool that is the correct failure
mode, and it is why the benchmark reports accuracy-on-answered and abstention rate side by side
rather than one blended accuracy.

## 6. Transfer cost, measured

Reading 8 tensors × 768 rows from **both** models of a pair:

```
Qwen2.5-0.5B + Qwen2.5-0.5B-Instruct: 77.1 MB read of 1976.2 MB total = 3.90%
```

That figure is a deliberately *pessimistic* upper bound: it uses one coalesced
Range request spanning the sampled row block per tensor.

The shipped loader was then measured end-to-end against the public Hub:

| model | checkpoint | header only | full 45-slot sketch | reduction |
|---|---|---|---|---|
| SmolLM2-135M-Instruct | 269.1 MB | 31,397 B (**0.012%**) | 17.07 MB (6.34%) | 16× |
| Qwen2.5-7B-Instruct | 15,231.3 MB | 27,752 B (**0.0002%**) | 98.14 MB (**0.644%**) | **155×** |

The sampling budget is fixed while checkpoints grow, so **the reduction improves
with scale**. Sub-1% is therefore true at realistic model sizes and false on a
toy one — which is why the benchmark's capability matrix *measures* that column
per run instead of asserting it.

## 7. Latency, not bandwidth, is the binding constraint

Measured against the Hub:

- a redirect-following Range request costs **~0.87 s**; `/resolve/` 302s to a
  signed CDN URL
- that signed URL **cannot be cached and replayed** — 0 of 6 reuse attempts
  returned anything but 403, so the redirect is paid on every request
- bandwidth is **~24 MB/s**, so one round trip is worth roughly 24 MB of payload

Two consequences shaped the loader:

1. `select_rows` samples **block-stratified**, not evenly spaced. Spreading 256
   rows across a 49152-row embedding leaves ~220 KiB gaps, above the coalescing
   threshold, so the reader issued *one request per row*. Sixteen contiguous
   runs read the same bytes in 16 requests. The strata still span the whole
   tensor, which matters because row order is not meaningless — embedding rows
   are token-id and therefore roughly frequency ordered.
2. Reads are **overlapped**. That cut one sketch from **162.9 s to 27.8 s
   (5.9×) with the byte count unchanged** (17.04 → 17.07 MB). The first attempt
   parallelised the wrong level: row-runs inside a tensor almost always coalesce
   to a single range, so the concurrency never engaged until it was moved up to
   whole tensors.
