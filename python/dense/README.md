# Dense bedrock

Multi-engine **dense layer** training bedrock for Planet Bridging.

Each **planet** (PyTorch, TensorFlow, JAX, sklearn, PaddlePaddle) trains the same 12 MLP specs on the same deterministic fixture, saves checkpoints, runs inference on 100 held-out samples, and POSTs results to the Go compare-host. The UI compares steps **within** each planet's pipeline — not PyTorch vs TensorFlow.

```
native  →  export  →  loom
  ✓          ✓         (next — Go side)
```

---

## Planet-side status

**Done for the current scope.** All five enabled engines train, infer, report, and skip retrain on rerun.

| Planet | Native format | Export formats | Pipeline reports |
|--------|---------------|----------------|------------------|
| **PyTorch** | `model.pt` | `model.safetensors`, `model.onnx` | 3 per model |
| **TensorFlow** | `model.keras` | `saved_model/` (Keras 3 export) | 2 per model |
| **JAX** | `params.msgpack` | — | 1 per model |
| **sklearn** | `model.pkl` | — | 1 per model |
| **Paddle** | `model.pdparams` | — | 1 per model |

Full successful run → **96 reports** (12 models × 8 steps: PyTorch 3 + TF 2 + three native-only planets).

What's **not** on the planet side yet (and doesn't block starting Loom import work):

- **Export steps** for JAX, sklearn, Paddle (no safetensors/ONNX yet — only native artifacts)
- **TF → ONNX** export (SavedModel only today)
- **`stage=loom`** reports (posted from Go once import lands)
- Standalone **`onnxruntime`** engine — folded into PyTorch's ONNX export step; `engines/onnxruntime/` is legacy/unwired from `run_dense.sh`

---

## Quick start

From repo root:

```bash
# Compare host (separate terminal)
go run .

# Or stop a background host
./killserver.sh

# One-time conda envs (pb-dense-*)
./python/dense/setup_conda.sh

# Train all planets + push reports (starts host if needed)
./python/dense/run_dense.sh

# Single planet
./python/dense/run_engine.sh pytorch
```

Open http://localhost:9876/ for the Dense tab.

Environment variables:

| Var | Default | Purpose |
|-----|---------|---------|
| `PLANETBRIDGING_HOST` | `http://127.0.0.1:9876` | Compare-host base URL |
| `PLANETBRIDGING_PORT` | `9876` | Used by `killserver.sh` only |

---

## Shared fixture

All planets read the same deterministic dataset (`dense_bedrock_v2`):

| Setting | Value |
|---------|-------|
| Seed | 42 |
| Train samples | 5000 |
| Test samples | 100 (reported to host) |
| Max input dim | 128 |
| Max output dim | 8 (models slice `y` to their width) |

Fixture npz is generated on first run under `fixtures/` (gitignored). Model specs live in [`manifest.yaml`](./manifest.yaml) — 12 dense configs from single linear layers through deep ReLU MLPs, including a no-bias variant.

Training hyperparams (shared): 40 epochs, batch 64, Adam lr 0.01.

---

## Per-planet pipelines

### PyTorch (`pb-dense-pytorch`)

1. **native / pytorch** — train or load `model.pt`, infer in PyTorch
2. **export / safetensors** — reload weights from safetensors, infer in PyTorch
3. **export / onnx** — infer via ONNX Runtime on `model.onnx`

Exports are written at train time. Rerun skips training if `meta.json` marks artifacts complete.

### TensorFlow (`pb-dense-tensorflow`)

1. **native / keras** — train or load `model.keras`
2. **export / saved_model** — infer via `TFSMLayer(..., call_endpoint="serve")`

Keras 3 cannot `load_model()` a SavedModel directory; export inference uses `TFSMLayer` with endpoint `serve` (from `model.export()`, not `serving_default`).

### JAX, sklearn, Paddle

Native checkpoint only. Each posts one report per model. Artifacts are planet-specific and ready for a future export pass or direct Loom bridge experiments.

---

## Observations (from bedrock runs)

These are expected behaviours, not bugs:

**Safetensors is bit-exact with PyTorch native.** Same weights, same runtime — `EXACT` in the UI.

**ONNX shows tiny float32 drift** (~`1e-7` max, ~`1e-8` mean). Different graph lowering and ORT kernels; normal for FP32 export. The UI labels anything non-zero as `DIFF` even when the error is negligible. For pipeline validation, treat `< 1e-5` as pass.

**TensorFlow native → SavedModel is exact** after the `TFSMLayer` fix. Same float path, same weights.

**sklearn uses `MLPRegressor`** — one linear output head only; hidden layers share one activation. Matches manifest constraints, not arbitrary Keras-style per-layer activations.

**Retrain skip works.** Delete a model dir under `models/<planet>/<model_id>/` to force retrain for that checkpoint only.

**Conda env drift.** If PyTorch ONNX step fails with `No module named 'onnxruntime'`, either re-run `setup_conda.sh` or `conda run -n pb-dense-pytorch pip install onnxruntime`. It's listed in `engines/pytorch/environment.yml` but existing envs won't pick it up until updated.

### Numerical types — planets vs Loom (FML tier)

Loom runs Dense across **21 DTypes** (FP64/32/16, BF16, FP8, ints, ternary, binary, …) natively and per-layer. **None of the Python bedrock planets do anything like that.**

| Planet | What bedrock actually uses | What the framework allows in theory |
|--------|----------------------------|-------------------------------------|
| **PyTorch** | `float32` train + infer | FP64, FP16, BF16, FP8, quant ints — but training is almost always FP32 or mixed FP16/BF16 |
| **TensorFlow/Keras** | `float32` | Mixed precision, post-hoc int8 quant — not arbitrary per-layer dtype menus |
| **JAX** | `float32` | Very flexible per-array typing — still FP32 in practice here |
| **sklearn** | **`float64`** internally (`MLPRegressor` / numpy double) | Basically FP64 only; no FP16/int/low-bit training path |
| **Paddle** | `float32` | FP16/BF16, deploy quant — same story as PyTorch-lite |

Bedrock casts test outputs to **float64 for reports** so the compare-host can diff apples-to-apples. Checkpoints on disk are **FP32 weights** (sklearn excepted — FP64 math, still “one type for everything”).

**Implication for stage 2 (Loom import):**

- Bridge validation is **FP32 weights → Loom FP32 (or FP64) infer → compare floats**. That's the realistic cross-planet bar.
- Don't expect to import a PyTorch checkpoint and replay Loom's int4/ternary/binary dtype sweep against planet ground truth — **the planets never produced those runs**.
- Loom's multi-dtype story is a **Loom-native superpower**, not something the AI solar system shares. Planet Bridging proves **weight/topology fidelity at FP32**; dtype exploration stays on the Go side after import.

If we ever want cross-planet dtype parity tests, we'd need new bedrock runs per engine per dtype — and most engines won't cooperate for half of Loom's type list anyway.

---

## Layout

```
python/dense/
  manifest.yaml          # models + engines + fixture version
  setup_conda.sh         # create pb-dense-* envs
  run_dense.sh           # all engines → host
  run_engine.sh          # single engine
  shared/
    fixtures.py          # deterministic npz
    manifest.py          # YAML loader
    runner.py              # per-model loop + POST
    reporter.py            # HTTP client
    artifacts.py           # meta.json completion markers
    variants.py            # VariantResult (stage/format/outputs)
  engines/
    pytorch/             # native + safetensors + onnx
    tensorflow/            # native + saved_model
    jax/
    sklearn/
    paddle/
    onnxruntime/         # legacy; not in run_dense.sh
  models/                # gitignored — trained checkpoints
  reports/               # gitignored — JSON copies (host also stores)
  fixtures/              # gitignored — generated npz
```

Checkpoint path: `models/<planet>/<model_id>/`

Each complete model has `meta.json`:

```json
{
  "framework_version": "2.x.x",
  "artifacts": ["model.pt", "model.safetensors", "model.onnx"]
}
```

---

## Report schema

Each pipeline step POSTs to `POST /api/v1/report`:

| Field | Example |
|-------|---------|
| `planet` | `pytorch` |
| `stage` | `native` \| `export` \| `loom` |
| `format` | `pytorch`, `safetensors`, `onnx`, `keras`, `saved_model`, … |
| `model_id` | `mlp_32_16_4_relu` |
| `fixture_version` | `dense_bedrock_v2` |
| `outputs` | 100 × output_dim float64 vectors |

Stored as `{planet}__{model_id}__{stage}__{format}.json`.

Compare-host diffs **native → each export** and **native → loom** (pending until Go import exists). It does **not** diff across planets.

---

## Conda environments

| Env | Engine |
|-----|--------|
| `pb-dense-pytorch` | PyTorch + safetensors + onnxscript + onnxruntime |
| `pb-dense-tensorflow` | TensorFlow/Keras |
| `pb-dense-jax` | JAX + Flax |
| `pb-dense-sklearn` | scikit-learn |
| `pb-dense-paddle` | PaddlePaddle |

`setup_conda.sh` creates missing envs from each `engines/*/environment.yml`. It does not upgrade existing envs — recreate or pip-install manually when deps change.

---

## Stage 2: Loom import — file types to build

Planet bedrock is done. **Stage 2 is Go:** read checkpoint files from `python/dense/models/`, build a Loom `VolumetricNetwork`, infer on the same 100 test inputs, POST `stage=loom` reports, diff against each planet's **native** report.

**Every format below needs a pure-Go importer built from scratch** (or a generic mapper extended for dense bedrock). Loom has partial Safetensors ingest for HF transformers — not a drop-in for these twelve MLP checkpoints. There is no existing bridge in this repo yet; `POST /api/v1/loom/import` is still a stub.

Topology for weight-only files comes from [`manifest.yaml`](./manifest.yaml) (`model_id` → layers, activations, bias). Graph-bearing files (ONNX, Keras) can cross-check topology against the manifest.

### Pure Go, stdlib-only — topology and weights without third-party libs

**Policy (for now):** `planetbridging` `go.mod` has **no external dependencies** — same spirit as Loom's pure-Go, zero-CGO core. The bridge should prefer **`encoding/json`**, **`encoding/binary`**, **`archive/zip`**, **`compress/*`**, **`io`**, **`os`** only. Pulling in `google.golang.org/protobuf`, `scigolib/hdf5`, etc. is a deliberate policy change, not a requirement.

Read this before writing `bridge/`. Question: for each bedrock file type, can we read **topology** and **weights** from scratch in Go with no third-party libs?

#### Topology vs weights (what's in the file)

| Format | Topology in file? | Weights in file? | Bedrock sidecar |
|--------|-------------------|------------------|-----------------|
| **`.safetensors`** | No — tensor names + shapes only | Yes — raw bytes | `manifest.yaml` supplies layers/activations |
| **`.onnx`** | Yes — `GraphProto` nodes/edges | Yes — inline tensors or `.data` sidecar | Manifest can cross-check |
| **`.keras`** | Yes — `config.json` inside ZIP | Yes — usually HDF5 blob inside ZIP | Optional cross-check |
| **`saved_model/`** | Yes — `saved_model.pb` | Yes — `variables/` | Optional cross-check |
| **`params.msgpack`** | No — Flax param tree | Yes — msgpack bytes | `manifest.yaml` |
| **`.pkl` / `.pt` / `.pdparams`** | No / buried in pickle | Yes | `manifest.yaml` (if you parse at all) |

#### Stdlib-only feasibility per format

| Format | Weights (stdlib) | Topology (stdlib) | Verdict |
|--------|------------------|-------------------|---------|
| **`.safetensors`** | ✅ 8-byte LE header + JSON + raw buffer read | ⚠️ Use `manifest.yaml` | **Start here.** Easiest path. No protobuf, no HDF5, no pickle. |
| **`.onnx`** | ✅ `TensorProto` bytes in protobuf wire format | ✅ `GraphProto` / `NodeProto` in same file | **Feasible** for bedrock scope. Hand-decode protobuf wire format for a *subset* of `onnx.proto` — only ops our MLPs use (`Gemm`, `MatMul`, `Add`, `Relu`, `Tanh`, `Sigmoid`). Do **not** need full ONNX schema or `google.golang.org/protobuf`. Real work, but bounded. |
| **`.keras`** | 🟡 Often HDF5 inside ZIP — HDF5 is a filesystem-in-a-file; readable from scratch but a substantial parser | ✅ `archive/zip` + `encoding/json` on config | **Partial.** Topology easy; weights are the lift. Alternative if export uses `.npz`/raw arrays: simpler, still annoying. |
| **`saved_model/`** | 🟡 `variables/` layout + protobuf graph | 🟡 `saved_model.pb` — hand protobuf subset (different schema than ONNX) | **Hard.** Defer until `.keras` or ONNX TF export exists. Keras 3 export is simpler than legacy SSTable checkpoints but still more moving parts than `.keras`. |
| **`params.msgpack`** (JAX) | 🟡 Msgpack spec is small; decode from scratch doable | ⚠️ `manifest.yaml` + Flax param naming | Defer — export to safetensors in Python first. |
| **`model.pkl`** (sklearn) | 🛑 Pickle | 🛑 Pickle | **Do not.** Mini Python VM. Export weights in Python. |
| **`model.pt`** (PyTorch) | 🛑 Pickle/ZIP state_dict | 🛑 Pickle | **Do not.** Use `.safetensors`. |
| **`model.pdparams`** (Paddle) | 🟠 Opaque binary | ⚠️ `manifest.yaml` | Defer — export to ONNX/safetensors first. |

#### What "from scratch" means in practice

- **Safetensors:** ~100 lines — header JSON, slice byte ranges, cast FP32. Map tensor keys → Loom Dense using manifest layer order.
- **ONNX:** wire-format protobuf decoder (varint, length-delimited fields) + structs for ~6 message types + tiny graph walker. Topology and weights both from file — first format where no sidecar is strictly required.
- **Keras 3:** ZIP open + JSON config → layer stack; then either hand-roll HDF5 dataset reader for weight arrays or add a Python export that writes safetensors instead (reuses Tier 1).
- **Pickle formats:** out of scope for a clean Go runtime — ecosystem consensus in [`../../legacy/docs/import_export_todo_list.md`](../../legacy/docs/import_export_todo_list.md) is convert → ONNX or Safetensors.

#### Recommended stdlib-only build order

```
1. .safetensors + manifest.yaml   weights from file, topology from YAML   ✅ stdlib
2. .onnx                          topology + weights from file           ✅ stdlib (subset protobuf)
3. .keras                         topology stdlib; weights need HDF5 work 🟡
4. saved_model/                   defer                                  🟠
— jax / sklearn / paddle native — export to (1) or (2) in Python first   🛑/🟠
```

Third-party libs (`google.golang.org/protobuf`, HDF5 packages) would speed development but break the current zero-dependency `go.mod`. If we add them later, document why and keep safetensors/ONNX-subset paths working stdlib-only as the reference implementation.

---

| Planet | Bedrock path | File type | Stage | On disk now |
|--------|--------------|-----------|-------|-------------|
| PyTorch | `models/pytorch/<id>/model.safetensors` | Safetensors | export | ✅ |
| PyTorch | `models/pytorch/<id>/model.onnx` | ONNX | export | ✅ |
| PyTorch | `models/pytorch/<id>/model.pt` | PyTorch state_dict (ZIP/pickle) | native | ✅ (prefer safetensors for import) |
| TensorFlow | `models/tensorflow/<id>/model.keras` | Keras 3 `.keras` (ZIP + HDF5/JSON) | native | ✅ |
| TensorFlow | `models/tensorflow/<id>/saved_model/` | TF SavedModel dir | export | ✅ |
| JAX | `models/jax/<id>/params.msgpack` | Flax msgpack bytes | native | ✅ |
| sklearn | `models/sklearn/<id>/model.pkl` | Python pickle | native | ✅ |
| Paddle | `models/paddle/<id>/model.pdparams` | Paddle state dict blob | native | ✅ |

Success criterion per import: **native → loom** matches within tolerance on the compare-host (same bar as native → export: EXACT or `< 1e-5` for FP32 paths).

### Import queue (build order)

Work top to bottom. Each row is a **separate Go implementation** (parser + dense layer mapper + infer hook).

#### Tier 1 — do first (hub formats, already exported)

| Priority | File type | Source | Graph? | Build in Go from scratch | Effort | Notes |
|----------|-----------|--------|--------|--------------------------|--------|-------|
| **1** | **`.safetensors`** | PyTorch export | Weights only | Safetensors header parser (8-byte len + JSON + raw tensors); map keys → Loom Dense via manifest | 🟢 Easiest | **Stdlib-only ✅** Bedrock already EXACT vs PyTorch native. |
| **2** | **`.onnx`** | PyTorch export | Yes (protobuf) | Hand-decode protobuf wire subset + scoped op executor (`Gemm`/`MatMul`, `Relu`, `Tanh`, `Sigmoid`) | 🟡 Medium | **Stdlib-only ✅** (subset). ~1e-7 drift vs native expected. Topology + weights in file. |

#### Tier 2 — TensorFlow native + export

| Priority | File type | Source | Graph? | Build in Go from scratch | Effort | Notes |
|----------|-----------|--------|--------|--------------------------|--------|-------|
| **3** | **`.keras`** | TF native | Yes (inside ZIP) | Keras 3 archive reader (ZIP + JSON config); weight arrays often in HDF5 inside ZIP — HDF5 from scratch is the hard part (stdlib-only 🟡) | 🟡 Medium | Prefer ONNX/safetensors TF export if we add it; else hand-roll HDF5 reader or revisit third-party policy. |
| **4** | **`saved_model/`** | TF export | Yes (protobuf + variables/) | SavedModel dir; hand protobuf subset + `variables/` — stdlib-only 🟠 | 🟠 Hard | Defer behind `.keras` or TF→ONNX export. Bedrock already EXACT native → export in Python. |

#### Tier 3 — deferred (no hub export yet; bad Go targets)

These exist on disk for native reports only. **Do not import directly first** — add a Python export-to-safetensors/ONNX step in bedrock, *then* reuse Tier 1 importers.

| File type | Source | Why defer | If you insist on Go anyway |
|-----------|--------|-----------|----------------------------|
| **`params.msgpack`** | JAX/Flax | Flax serialization format; no graph; tree-shaped param names | Msgpack parser + Flax name → Dense mapper 🟠 |
| **`model.pkl`** | sklearn | Python pickle — unsafe, not a serious Go target | Allowlisted unpickler or don't; export weights via Python 🛑 |
| **`model.pdparams`** | Paddle | Paddle-specific binary; often paired with `.pdmodel` for graph | Paddle format reverse-engineering 🟠 |
| **`model.pt`** | PyTorch native | Pickle/ZIP state_dict | Same pickle problem; **use `.safetensors` instead** 🛑 |

### What each importer must do

1. **Detect** — planet + format from path / magic bytes / extension  
2. **Parse** — file format only (no Python, no CGO)  
3. **Map** — tensors + activations → Loom volumetric Dense layers (manifest fills gaps for weight-only)  
4. **Infer** — run 100 bedrock test inputs through Loom  
5. **Report** — `POST /api/v1/report` with `stage=loom`, `format=loom` (or `loom/safetensors` if tracking provenance)  
6. **Compare** — host diffs native → loom; shows on Dense tab  

Planned home: `bridge/` package (or under `host/` initially) wired to `POST /api/v1/loom/import`.

### Formats we are *not* targeting in stage 2

| Format | Reason |
|--------|--------|
| `.gguf` | LLM inference planet; no dense bedrock artifacts |
| `.h5` (legacy Keras) | Bedrock uses Keras 3 `.keras` only |
| `.tflite` / `.ort` | Runtime-specific; not produced by bedrock |
| `.ckpt` / TF checkpoint index | SSTable index pain; `.keras` is the TF native path |
| HuggingFace `config.json` + sharded safetensors | Different naming/layout; later phase for MHA/CNN layers |

### Summary

```
Stage 2 build list (Go, from scratch):
  1. .safetensors     ← start here
  2. .onnx
  3. .keras
  4. saved_model/
  — later via export hub or new parsers —
  5. params.msgpack / .pdparams / .pkl (avoid .pt pickle)
```

One Go toolchain, one Loom runtime — vs five conda envs that produced the files.

See [`../../README.md`](../../README.md) for the broader Loom bridge roadmap.

---

## Why conda? The Python dependency problem

The dense bedrock is intentionally small — twelve MLPs, one layer type, shared CSV-grade fixture data. The **orchestration cost** is wildly disproportionate to the math being done. That is the point. This section documents what we hit in practice and why the Go/Loom side exists partly as an escape hatch.

### You cannot use one Python environment

We tried to structure this sanely (`shared/`, one manifest, one shell runner). The runtime reality still forces **one isolated conda env per engine**:

| Env | Why it must be alone |
|-----|----------------------|
| `pb-dense-pytorch` | PyTorch stack + pip add-ons (safetensors, onnxscript, onnxruntime) |
| `pb-dense-tensorflow` | TensorFlow pulls its own numpy/scipy/protobuf ecosystem |
| `pb-dense-jax` | JAX + jaxlib + Flax + Optax — version-locked to each other |
| `pb-dense-sklearn` | Could share numpy with others in theory; kept separate for consistency and to avoid contaminating sklearn's scipy/sklearn pin |
| `pb-dense-paddle` | Paddle installed via **pip inside conda** — another resolver, another wheel story |

There is no workable `requirements.txt` that installs PyTorch + TensorFlow + JAX + Paddle in one venv. They fight over numpy versions, CUDA stubs (even on CPU), protobuf, and compiled extensions. The AI ecosystem assumes **one framework per environment**. Planet Bridging dense bedrock mirrors that fracture literally: five planets, five envs.

A sixth env (`pb-dense-onnxruntime`) exists for legacy standalone ONNX inference. ONNX Runtime is now folded into the PyTorch env instead — but the folder and yml remain as a reminder that even "just run ONNX" wanted its own env.

### What conda buys us (and what it costs)

**Buys:**

- Pre-built binary wheels for heavy C++/CUDA stacks without compiling from source
- Per-engine isolation so TF 2.x and torch don't corrupt each other
- `conda run -n pb-dense-* python ...` — repeatable invocation from shell scripts

**Costs:**

- **Miniconda/Anaconda is mandatory.** No conda → `run_engine.sh` exits immediately. This is not `pip install -r requirements.txt` and go.
- **Disk:** five envs × (Python + numpy + framework) ≈ many gigabytes. Each env duplicates Python 3.11 and numpy.
- **Time:** first `setup_conda.sh` run downloads and solves environments for every planet. Slow on a fresh machine.
- **Channel politics:** Anaconda/conda-forge ToS acceptance, channel config, `conda init zsh` — extra onboarding friction we hit on macOS setup.
- **Two package managers:** PyTorch env uses conda + pip (`safetensors`, `onnxscript`, `onnxruntime`). Paddle env uses conda + pip (`paddlepaddle`). Conda solved the env; pip deps can still drift independently.
- **No auto-upgrade:** `setup_conda.sh` only **creates missing** envs. Change `environment.yml` (e.g. add `onnxruntime`) and existing envs stay stale until you manually `conda env update`, recreate, or `pip install`. We learned this when PyTorch ONNX failed with `No module named 'onnxruntime'` on envs created before that dep was added.
- **Shell glue everywhere:** every engine run is `conda run --no-capture-output -n pb-dense-$ENGINE python engines/$ENGINE/run.py ...`. The shared Python code is portable; the **interpreter is not**.
- **Process overhead:** `conda run` spawns a wrapper per engine per run. Fine for bedrock; annoying for tight iteration loops.
- **Framework API churn:** even with pinned envs, framework *code* breaks across versions — Keras 3 dropped `load_model()` on SavedModel dirs; we needed `TFSMLayer(call_endpoint="serve")`. That's not a conda bug, but it's the same class of problem: the planet moves under you.

### Failure modes we actually saw

| Symptom | Cause |
|---------|-------|
| `No module named 'onnxruntime'` | Env created before pip dep added to yml; setup script doesn't upgrade |
| `Keras 3 ... SavedModel ... use TFSMLayer` | TF API change, not env-related — but only shows up at export infer time |
| `compare-host not reachable` | Go host is separate process; Python side hard-fails if it's down |
| Engine exits 1, others continue | `run_dense.sh` tolerates per-engine failure — partial report sets are easy to produce |
| Paddle ccache warning | Harmless noise from Paddle's C++ extension path inside conda |

### The orchestration stack (Python side)

To run the full bedrock once:

```
Miniconda installed
  → conda init + accept channel ToS
    → ./setup_conda.sh  (5 env creates)
      → go run .        (compare host — different language, different toolchain)
        → ./run_dense.sh
          → conda run pb-dense-pytorch   (12 models)
          → conda run pb-dense-tensorflow (12 models)
          → conda run pb-dense-jax       (12 models)
          → conda run pb-dense-sklearn   (12 models)
          → conda run pb-dense-paddle    (12 models)
```

Six tools, five Python interpreters, one Go server — to train tiny MLPs and POST 100 float vectors each.

---

## Go / Loom: the contrast

The compare-host and (eventually) Loom import live on the other side of this wall. Same repo, different universe.

### Planetbridging host (what runs today)

```bash
go run .
```

| | Python bedrock | Go compare-host |
|--|----------------|-----------------|
| **Toolchain** | Python 3.11 × 5 + conda + pip | Go 1.26 |
| **External deps** | PyTorch, TF, JAX, sklearn, Paddle, ORT, … | **None** (`go.mod` is stdlib-only) |
| **Install step** | Miniconda + `setup_conda.sh` | Install Go (you likely already have it) |
| **Run command** | `conda run -n pb-dense-* python ...` per engine | `go run .` |
| **Disk** | Multi-GB env tree | Compiled binary + report JSON |
| **Port** | N/A | `:9876` |

No virtualenv. No conda. No pip. No CUDA toolkit for the dashboard. The host reads JSON reports, diffs pipelines, serves HTML — all stdlib `net/http`.

### Loom (where import is heading)

Loom is the reason Planet Bridging exists: **one deterministic runtime** in pure Go for Dense, CNN, MHA, LSTM, RNN across 21 dtypes — no CGO on the core path, no Python interpreter at inference time, no per-framework conda env.

| | Python planets | Loom |
|--|----------------|------|
| **Runtimes to install** | 5+ (one per engine) | 1 (Go binary) |
| **Dependency graph** | Incommensurable per planet | `go.mod` |
| **Train + infer** | Each planet reimplements Adam, MSE, activations | Native volumetric layers |
| **Save format** | `.pt`, `.keras`, `.pkl`, `.pdparams`, msgpack, … | `model.json` (+ optional safetensors ingest) |
| **Reproducibility** | Seed + env + framework version + export path | Deterministic DNVM semantics |

The Python bedrock is **ground truth generation** — we need real PyTorch/TF/etc. outputs to validate that Loom import didn't butcher weights. But the *product* is Loom eating checkpoints and running them without this circus.

### Why we still need Python (for now)

Honest scope:

- **Training on every planet** — Loom can train native dense layers, but Planet Bridging's job is to prove **cross-planet import**, which requires actual planet runtimes producing real artifacts.
- **Export paths** — `torch.onnx.export`, `model.export()` SavedModel, etc. are mature in their home frameworks. Rebuilding export in Go is the long game; consuming exports is shorter.
- **Pickle planets** — sklearn's `.pkl` will never be a Go-native format. Python (or a one-shot export script) stays the source of truth there.

The end state is not "replace Go with five condas." It's **condense five condas into one Go runtime** after weights cross the bridge.

### The lesson (why this doc exists)

```
Python bedrock  = 5 planets, 5 envs, 5 weight formats, shell orchestration, API drift
Go compare-host = 1 toolchain, 0 deps, `go run .`
Loom target     = 1 runtime, 1 model.json, deterministic infer, no conda ever
```

Dependency hell on the Python side isn't a setup mistake — it's a faithful model of the AI ecosystem. Every major engine is its own island. Conda is the ferry service. Loom is the attempt to build a continent.

When something breaks in dense bedrock, check conda first: wrong env, stale env, missing pip dep, host not running. When something breaks on the Go side, it's usually `go run .` not listening or a bad report JSON — not five package managers.
