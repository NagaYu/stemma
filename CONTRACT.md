# Stemma internal API contract (frozen)

Every module codes against this. Do not change a signature listed here; if you
believe one is wrong, implement it as specified and note the concern in a
`# CONTRACT-CONCERN:` comment.

Shared types live in `stemma/types.py` (already written — read it first).

## Global conventions

- Python >= 3.10, `from __future__ import annotations` at the top of every module.
- Numeric work in **numpy float32/float64**; torch is used only to *read*
  safetensors and for the benchmark's model construction. Never require CUDA.
- Every public function has a docstring whose **first line** states what it does
  and which contains a line of the form:
  `Claim: <direction|merge-recovery|low-transfer|low-false-positive|infra>` —
  this is a hard requirement of the project spec (tests grep for it).
- No network at import time. No `print()` in library code — use
  `stemma.utils.get_logger(__name__)`.
- Optional deps (`faiss`, `gradio`, `datasets`, `graphviz`) must be imported
  lazily inside functions, with a working pure-numpy/pure-python fallback where
  the contract says so.
- Determinism: any randomness takes an explicit `seed: int = 0`.

## `stemma/utils.py`  (owner: infra)

```python
def get_logger(name: str) -> logging.Logger
def set_seed(seed: int) -> None
def is_local_path(ref: ModelRef) -> bool
def role_of(tensor_name: str) -> str | None          # -> one of types.ROLES
def layer_index_of(tensor_name: str) -> int | None   # transformer block index
def relative_depth(idx: int, n_layers: int) -> float
def depth_bucket(rel_depth: float) -> int            # index into DEPTH_BUCKETS
def human_bytes(n: int) -> str
def stable_hash(obj) -> str                          # sha256 hex[:16]
def atomic_write_json(path, obj) -> None
```

`role_of` maps by regex over common naming schemes (Llama/Qwen/Mistral:
`model.embed_tokens`, `self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`,
`*layernorm*`; GPT-2: `wte`, `wpe`, `attn.c_attn`, `attn.c_proj`, `mlp.c_fc`,
`mlp.c_proj`, `ln_*`). GPT-2's fused `c_attn` maps to role `attn_q` and is
noted in `Sketch.meta["fused_qkv"] = True`. Unrecognised names -> `None`.

## `stemma/remote_loader.py`  (owner: loader)  — **hard requirement**

Reads safetensors over HTTP **Range** requests: header first, then only the
byte ranges of the tensors actually requested. Must never download a whole
shard unless the caller asks for every tensor in it.

```python
class SafeTensorsSource:
    def __init__(self, ref: ModelRef, *, revision: str = "main",
                 token: str | None = None, cache_dir: str | None = None,
                 session: requests.Session | None = None) -> None
    @property
    def stats(self) -> TransferStats
    def index(self) -> dict[str, TensorMeta]     # name -> meta (all shards)
    def total_size(self) -> int                  # sum of shard file sizes
    def get_tensor(self, name: str, *, dtype=np.float32,
                   max_rows: int | None = None,
                   row_stride: int | None = None) -> np.ndarray
    def get_tensors(self, names: Sequence[str], **kw) -> dict[str, np.ndarray]
    def config(self) -> dict                     # config.json (small GET)
    def close(self) -> None
```

Requirements:
- Local dirs: resolve `ref` as a path when `utils.is_local_path(ref)`; read via
  `mmap` and still populate `TransferStats` (`bytes_read` = bytes actually
  touched) so the benchmark can compare like with like.
- Remote: `https://huggingface.co/{repo}/resolve/{revision}/{file}`.
  1. `GET` bytes `0-7` -> little-endian u64 header length `N`.
  2. `GET` bytes `8..8+N-1` -> JSON header. Convert `data_offsets` to absolute
     by adding `8 + N`.
  3. Sharded repos: try `model.safetensors.index.json` first (small GET); fall
     back to `model.safetensors`; then to `*.safetensors` listed by
     `huggingface_hub.list_repo_files`.
- `max_rows` / `row_stride`: for a 2D row-major tensor, fetch **only** the
  needed row byte-ranges (contiguous rows are one request; strided rows are
  coalesced into runs, merging gaps smaller than `COALESCE_GAP = 64 * 1024`).
  This is the main lever for the low-transfer claim.
- dtype support: F64 F32 F16 BF16 I64 I32 I16 I8 U8 BOOL, plus F8_E4M3 /
  F8_E5M2 (decode via a lookup table; if unsupported, raise
  `UnsupportedDtype`). BF16 -> float32 by bit-shifting (no torch dependency).
- Cache header JSON + config on disk under `cache_dir` (default
  `~/.cache/stemma`), keyed by `stable_hash((repo, revision, file))`.
- Retries: 3 attempts, exponential backoff, honour `Retry-After`. If the server
  answers 200 instead of 206, raise `RangeUnsupported` (do not silently
  download the file).
- Also expose module-level helpers:
  ```python
  def open_model(ref, **kw) -> SafeTensorsSource
  def measure_full_download_bytes(ref, **kw) -> int   # header only, no payload
  ```

## `stemma/sketch.py`  (owner: sketch)

```python
def tensor_invariants(W: np.ndarray, *, k: int = 64, seed: int = 0) -> np.ndarray
    # -> float32 (FEATURES_PER_SLOT,)  == 32
def sinkhorn_normalize(W: np.ndarray, iters: int = 8, eps: float = 1e-8) -> np.ndarray
def sketch_model(ref, *, source=None, max_rows: int = 2048, k: int = 64,
                 seed: int = 0, **loader_kw) -> Sketch
def sketch_distance(a: Sketch, b: Sketch) -> float      # in [0, 2], lower == closer
def sketch_similarity(a: Sketch, b: Sketch) -> float    # 1 - distance/2
```

`tensor_invariants` layout (indices are frozen):
- `0:16`  log10(sigma_i / sigma_1 + 1e-12) sampled at 16 log-spaced ranks of
  the top-`k` singular values of the **sinkhorn-normalised** matrix
  (randomized SVD, `seed`).
- `16`    spectral entropy of sigma^2 (normalised to [0,1] by log k).
- `17`    stable rank ||W||_F^2 / ||W||_2^2, divided by min(m, n).
- `18`    participation ratio of sigma^2.
- `19:27` 8 quantiles (0.05..0.95) of sorted row norms divided by their mean
  (computed on the **raw** matrix — permutation-invariant, globally scale-free).
- `27`    Gini coefficient of row norms.
- `28`    excess kurtosis of the normalised row-norm distribution.
- `29`    skewness of same.
- `30`    normalised 3rd spectral moment; `31` normalised 4th.
All entries must be finite (replace nan/inf with 0.0) and clipped to [-10, 10].

Invariance requirement (tested): for a random permutation P, Q and positive
diagonal D1, D2, `tensor_invariants(D1 P W Q D2)` must match
`tensor_invariants(W)` to within 5e-2 in L-inf.

`sketch_model` reads at most **two tensors per (role, depth-bucket) slot** and
subsamples to `max_rows` rows via `SafeTensorsSource.get_tensor(max_rows=...)`.
Fills `Sketch.present` and leaves absent slots as zeros. `Sketch.meta` must
carry `n_layers`, `hidden_size`, `vocab_size`, `dtypes`, `architectures`,
`tensor_count`, `param_count`, `fused_qkv`.

`sketch_distance`: cosine distance restricted to slots present in **both**
sketches, plus a small penalty for mismatched presence masks.

## `stemma/direction.py`  (owner: direction) — **most important module**

```python
@dataclass
class PairEvidence:      # raw per-pair measurements, before combination
    ...                  # free-form, must be JSON-serialisable
def collect_pair_evidence(a, b, *, sa=None, sb=None, max_rows=1024,
                          n_tensors=6, seed=0, **loader_kw) -> PairEvidence
def direction_features(ev: PairEvidence) -> dict[str, float]   # keys == DIRECTION_FEATURES
def estimate_direction(a, b, *, weights=None, abstain: float = 0.5,
                       **kw) -> DirectionVerdict
class DirectionModel:                 # the fitted combiner pushed to the Hub
    weights: np.ndarray               # (len(DIRECTION_FEATURES),)
    bias: float
    feature_names: tuple[str, ...]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    def llr(self, feats: dict[str, float]) -> float
    def save(self, path) -> None      # single .json
    @classmethod
    def load(cls, path) -> "DirectionModel"
    @classmethod
    def default(cls) -> "DirectionModel"   # hand-set priors, no fitting needed
    @classmethod
    def fit(cls, X, y, *, l2: float = 1.0) -> "DirectionModel"
```

Anti-symmetry requirement (tested): `direction_features` must satisfy
`f(b, a) == -f(a, b)` for every feature, to within 1e-6. Therefore
`estimate_direction(a, b).llr == -estimate_direction(b, a).llr`. Implement each
feature as `g(A, B) - g(B, A)` for a per-side statistic `g`, or as an explicitly
signed quantity.

The four evidence families, each of which MUST be implemented:
- **(a) Δ-spectrum**: with `Δ = B - A` per shared tensor, compute
  `subspace_energy(Δ, X, k)` = fraction of `||Δ||_F^2` captured by the top-`k`
  right singular subspace of `X`; effective rank of `Δ` (entropy-based);
  Frobenius norm growth; spectral-mass growth. Aggregate over tensors by
  parameter-weighted mean.
- **(b) Orphan embeddings**: when vocab sizes differ, examine the rows present
  only in the larger model. Score how "untrained" they look: near-zero norm,
  or norms tightly concentrated around a single init scale while trained rows
  are heavy-tailed (compare to the shared rows' norm distribution via a
  one-sided KS statistic). Also detect *exactly duplicated* pad rows.
- **(c) Quantisation / pruning scars**: per row, estimate the smallest lattice
  step that explains the values (fit `s = argmin` residual over candidate step
  sizes derived from the sorted unique |values|; report the fraction of entries
  within 1e-3 relative of `k*s`); count distinct values per row (INT4 -> <= 16);
  exact-zero fraction and whether one side's zero set is a superset of the
  other's; dtype precision ordering (F32 > BF16/F16 > I8 > I4).
- **(d) Fossils**: dead rows (all-zero / norm below 1e-6) present in both;
  outlier channels (rows whose norm exceeds 8x the median) that persist; count
  of bit-identical tensors; tensors present in one model but not the other.

`estimate_direction` returns `direction="unknown"` when `abs(llr) < abstain`.
It must populate `contributions` (per-feature `w_i * z_i`) and human-readable
`evidence` strings for the Gradio UI.

Also expose:
```python
def relatedness_score(a, b, *, sa=None, sb=None, **kw) -> float   # symmetric, [0,1]
```
built from the sketch distance plus a small number of raw-tensor checks; the
benchmark's AUC axis uses this.

## `stemma/merge_decompose.py`  (owner: merge)

```python
def task_vectors(base, candidates, child, *, coords: int = 200_000,
                 seed: int = 0, tensor_filter=None, **loader_kw)
        -> tuple[np.ndarray, np.ndarray, list[str], TransferStats]
    # -> (T [n_candidates, n_coords], t_child [n_coords], used_tensor_names, stats)
def nnls_l1(T: np.ndarray, y: np.ndarray, *, l1: float = 0.0,
            sum_to_one: bool = False, nonneg: bool = True) -> np.ndarray
def decompose_merge(child, candidates, *, base=None, l1: float = 1e-3,
                    sum_to_one: bool = True, support_threshold: float = 0.05,
                    coords: int = 200_000, seed: int = 0,
                    **loader_kw) -> MergeDecomposition
def mixing_mae(pred: dict[str, float], truth: dict[str, float]) -> float
def parent_set_prf(pred: Sequence[str], truth: Sequence[str]) -> tuple[float, float, float]
```

Notes:
- Coordinate subsampling: pick a fixed pseudo-random set of `coords` positions
  (seeded, identical across all models being compared) inside a fixed set of
  large 2D tensors that exist in **all** models involved. Use
  `get_tensor(max_rows=...)` so this stays a Range read.
- `base=None`: infer a base as the candidate with the smallest average distance
  to the others, or fall back to decomposing the raw weights (no task vectors)
  with `sum_to_one=True`.
- `nnls_l1`: solve via `scipy.optimize.nnls` on the L1/simplex-augmented system;
  when `sum_to_one`, solve the constrained problem with
  `scipy.optimize.minimize(method="SLSQP")` warm-started from the NNLS solution.
  Pure-numpy projected-gradient fallback if scipy is missing.
- `mixing_mae` compares over the **union** of keys, treating missing as 0.

## `stemma/baselines.py`  (owner: bench)

Symmetric comparators for the headline table. Each returns a similarity in
[0, 1] and **must not** be able to produce a direction.

```python
def cosine_baseline(a, b, **kw) -> float       # flattened-weight cosine
def cka_baseline(a, b, **kw) -> float          # REEF-style linear CKA on weight rows
def huref_baseline(a, b, **kw) -> float        # HuRef-style invariant terms
BASELINES: dict[str, Callable]
def baseline_direction(*args, **kw) -> str     # always "unknown" (documents the 50% ceiling)
```

## `stemma/phylogeny.py`  (owner: graph)

```python
class SketchIndex:                # faiss if available, numpy brute force otherwise
    def __init__(self, dim: int = SKETCH_DIM, metric: str = "cosine")
    def add(self, sketches: Sequence[Sketch]) -> None
    def search(self, q: Sketch, k: int = 10) -> list[tuple[str, float]]
    def save(self, path) -> None
    @classmethod
    def load(cls, path) -> "SketchIndex"
    @property
    def backend(self) -> str      # "faiss" | "numpy"
def find_candidate_parents(target, index, *, k=10, max_distance=0.35) -> list[tuple[str, float]]
def build_phylogeny(models, *, index=None, relatedness_threshold=0.6,
                    direction_abstain=0.5, merge_check=True,
                    direction_model=None, **kw) -> Phylogeny
def to_mermaid(p: Phylogeny, conflicts=()) -> str
def to_graphviz_dot(p: Phylogeny, conflicts=()) -> str
def trace(target, universe, **kw) -> Phylogeny     # lineage of one model
```

`build_phylogeny` must: (1) retrieve candidates via the index; (2) drop pairs
below `relatedness_threshold`; (3) orient with `direction.estimate_direction`;
(4) when a node has >= 2 surviving parents, call the merge decomposer and keep
only parents with non-negligible coefficients, recording `Edge.weight`;
(5) break cycles by removing the lowest-confidence edge on the cycle.

## `stemma/rights.py`  (owner: graph)

```python
def fetch_license_facts(ref, *, token=None, offline=False) -> LicenseFacts
def propagate(p: Phylogeny, facts: dict[str, LicenseFacts]) -> dict[str, LicenseFacts]
def detect_conflicts(p: Phylogeny, facts) -> list[RightsConflict]
def build_bom(p: Phylogeny, facts, conflicts, *, root, transfer=None,
              sketches=None) -> BOM
def bom_to_spdx(bom: BOM) -> dict          # SPDX-2.3-flavoured dict
```

Known-license table must cover at least: apache-2.0, mit, bsd-3-clause,
llama2/llama3/llama3.1/llama3.2, gemma, qwen (tongyi-qianwen), cc-by-nc-4.0,
cc-by-sa-4.0, cc-by-4.0, openrail / bigscience-openrail-m, gpl-3.0, agpl-3.0,
cc0-1.0, unknown. Conflict kinds are exactly those in `RightsConflict.kind`.
Wording must stay evidential — never assert infringement.

## `stemma/cli.py`  (owner: cli)

`stemma trace org/model [--universe FILE] [--out bom.json] [--format json|spdx|mermaid|dot] [--offline] [--k 10]`
plus subcommands `sketch`, `direction A B`, `decompose CHILD --candidates ...`,
`bench`, `index`. Entry point `stemma.cli:main`, also `python -m stemma`.
Rich/plain text output, `--json` for machine-readable. Must print the
human-review disclaimer on every `trace`.

## Benchmark & data

`scripts/build_bench.py` writes to `--out-dir` (default `bench_models/`) a set
of real safetensors model directories plus `ground_truth.json`:
```json
{"models": [{"id": "...", "path": "...", "family": "...", "op": "...",
             "parents": [...], "weights": {...}, "license": "..."}],
 "edges": [{"parent": "...", "child": "...", "relation": "...", "weight": 0.7}],
 "pairs": [{"a": "...", "b": "...", "related": true, "direction": "a->b",
            "relation": "sft"}]}
```

`benchmarks/run.py` produces `benchmarks/results.json` + `figures/*.png` and a
markdown table with: relatedness AUC / FPR@95TPR, direction accuracy (Stemma vs
each baseline), merge parent-set F1 and mixing MAE, bytes-per-decision and
seconds-per-decision.
