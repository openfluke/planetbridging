# Planet Bridging — Progress

Living doc for what works, what does not, and what we are building next — **layer type by layer type**.

Update this file when a planet/step/layer type moves status. The compare UI at http://localhost:9876/ is the live scoreboard; this file is the narrative.

> **Scope today:** only **Dense** is in active bridging. CNN1/2/3, MHA, LSTM, RNN (and everything below) are **not started** — each is roughly another full bedrock program like `python/dense/`. Long road ahead.

---

## How to read status

| Symbol | Meaning |
|--------|---------|
| ✅ | Verified on shared fixture (`dense_bedrock_v2`, 100 test inputs) |
| 🟡 | Partially working or only spot-checked |
| ⬜ | Not built / not run yet |
| ❌ | Built but failing — add a note in **Known issues** |

**Compare labels:** `EXACT` = bit-identical outputs · `PASS` = within fp32 tolerance (`< 1e-5` max abs diff) · `DIFF` = investigate.

---

## Important: parallel pipelines, not a chain

Each planet pipeline branches from **native**. Steps are compared **to native**, not to each other.

```
                         ┌─→ export / onnx          (checkpoint fidelity)
native / <planet> ───────┼─→ export / safetensors   (checkpoint fidelity)
                         └─→ loom / entity          (layer stream → .entity)
```

**This is NOT:**

```
native → onnx → safetensors → loom entity   ❌
```

| Path | Status | Notes |
|------|--------|-------|
| native → export (on disk) | ✅ dense bedrock | Proves save/reload did not corrupt weights |
| native → loom / entity (layer stream) | 🟡 dense only | Python reads **live in-memory** weights; Go builds `.stream.entity` |
| onnx file → loom entity | ⬜ | No Go importer wired in compare-host |
| safetensors file → loom entity | ⬜ | No Go importer wired in compare-host (`.safetensors.entity` files on disk are experiments, not reported) |
| loom entity → export | ⬜ | Export back to planets not started |

So when the UI shows native ✓, export ✓, and loom ✓ together, that means **three independent checks passed** — not that data flowed onnx → safetensors → Loom.

---

## Layer types (roadmap)

We bridge **one Loom volumetric layer type at a time**. Dense bedrock is step 1. Every other row needs its **own** fixture, manifest, Python bedrock tree, stream schema extension, Go builder, and compare-tab work — not a small tweak to dense.

| Loom layer | Bedrock | Planet extractors | Go bridge | Compare UI | Overall |
|------------|---------|-------------------|-----------|------------|---------|
| **Dense** | ✅ `python/dense/` · 12×5 | ✅ live extractors | ✅ stream → `.entity` | ✅ dense tab | 🟡 POC |
| **CNN1** | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| **CNN2** | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| **CNN3** | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| **MHA** | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| **LSTM** | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| **RNN** | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |

**Also in Loom (not on planet-bridge roadmap yet):** SwiGLU, Embedding, Residual — needed for full transformers but deferred until MHA + Dense paths exist.

### Suggested order after Dense is green

1. **CNN1** — smallest conv surface; teaches layout (NCHW vs NHWC) before 2D/3D
2. **CNN2** — vision stacks; ONNX `Conv` + Keras `Conv2D` export paths
3. **MHA** — unlocks transformers; hardest mapping (heads, KV, masks, GQA)
4. **LSTM** → **RNN** — recurrent state + time dimension; export pain on TF/JAX
5. **CNN3** — niche (video/volume); do when 1D/2D conv bridge is boring

---

## Beyond Dense — what each layer needs

Each section is the **same checklist** dense is finishing now. Nothing here exists yet except Loom’s native CPU execution (see [Loom bedrock validation](../docs/bedrock_validation.md) Lucy **[7]** suite).

| Work item | Dense (now) | Every other layer |
|-----------|-------------|-------------------|
| Shared fixture + manifest | `dense_bedrock_v2` | New `*_bedrock_v1` per layer family |
| Python bedrock (`python/<layer>/`) | ✅ five conda engines | Clone pattern; new model zoo |
| Native train + infer + report | ✅ | ⬜ |
| Export checkpoint compare | ✅ PyTorch ONNX/ST, TF SavedModel | Per-format per planet |
| Live weight extractor | ✅ `extract_*` dense | ⬜ planet-specific module walk |
| Stream JSON schema | `layers[]` dense only | Extend `bridge` with conv/MHA/RNN payloads |
| Go `BuildNetworkFromStream` | Dense only | Map to `VolumetricLayer` CNN/MHA/… |
| `.entity` save/load + infer | ✅ dense MLP path | Per-layer infer helper |
| Compare host UI tab | Dense pipeline view | New tab or filter per layer type |

---

### CNN1 (1D convolution) — ⬜ not started

**Loom:** `VolumetricLayer` CNN1 · volumetric grid (e.g. sequence × 1 × 1).

**Planet ops:**

| Planet | Native op | Export |
|--------|-----------|--------|
| PyTorch | `nn.Conv1d` | ONNX `Conv` (1D) |
| TensorFlow/Keras | `Conv1D` | SavedModel |
| JAX | `lax.conv` / Flax `Conv` | ⬜ export TBD |
| Paddle | `Conv1D` | ⬜ |
| sklearn | N/A (no conv) | skip or tiny custom |

**Hard parts:** kernel layout `[out_ch, in_ch, k]` vs ONNX ordering; `padding`, `stride`, `dilation`; channel-first vs channel-last; bias; activation after conv; matching **multi-cell** Loom grid dims to planet tensor shapes.

**Bedrock idea:** short 1D signals (audio-ish or synthetic), 2–3 conv blocks + dense head, same 5 planets where applicable.

---

### CNN2 (2D convolution) — ⬜ not started

**Loom:** `VolumetricLayer` CNN2 · H×W grids (Lucy suite uses 1³, 2³, 3³ morphs).

**Planet ops:** `nn.Conv2d`, Keras `Conv2D`, ONNX `Conv` 2D, Paddle `Conv2D`.

**Hard parts:** everything in CNN1, plus **spatial layout** (NCHW PyTorch vs NHWC TF historical default), pooling layers (not native Loom layer — may need flatten + dense or extend bedrock), image-sized fixtures and memory.

**Bedrock idea:** tiny MNIST-style 8×8 or 16×16 grayscale, small CNN + dense classifier.

---

### CNN3 (3D convolution) — ⬜ not started

**Loom:** `VolumetricLayer` CNN3 · depth×H×W (Lucy often 1³ only).

**Planet ops:** `nn.Conv3d`, Keras `Conv3D`, ONNX `Conv` 3D.

**Hard parts:** large tensors, fewer planet examples in the wild, JAX/Paddle coverage thin.

**Bedrock idea:** micro volume (e.g. 4×4×4 synthetic), single conv block — prove layout before scaling.

---

### MHA (multi-head attention) — ⬜ not started

**Loom:** `VolumetricLayer` MHA · Q/K/V/O projections, heads, RoPE, KV cache semantics (see Loom 0.79 MHA fixes).

**Planet ops:**

| Planet | API |
|--------|-----|
| PyTorch | `nn.MultiheadAttention`, custom HF blocks |
| TensorFlow | `MultiHeadAttention`, Keras attention layers |
| ONNX | `Attention`, `MultiHeadAttention`, `GroupQueryAttention` |
| JAX/Flax | `nn.MultiHeadDotProductAttention` |
| HF + safetensors | weight naming conventions, not a single op |

**Hard parts:** **Not one weight matrix** — Q/K/V/O, num_heads, head_dim, optional biases; causal masks vs bidirectional; **GQA/MQA**; layer norm / RMS norm placement (often separate Loom layers); sequence length and batch in stream protocol; export graphs that fuse ops differently per planet.

**Bedrock idea:** single attention block + projection, fixed seq len (e.g. 8 or 16), tiny d_model — compare one forward pass before full GPT blocks.

**Depends on:** Dense (projections are dense-like); often **SwiGLU + RMSNorm** for full transformer parity (future).

---

### LSTM — ⬜ not started

**Loom:** `VolumetricLayer` LSTM · gates, cell state, time steps.

**Planet ops:** `nn.LSTM`, Keras `LSTM`, ONNX `LSTM` (opset-dependent), JAX `lax.lstm` / Flax `LSTMCell`.

**Hard parts:** **stateful inference** (h, c) across timesteps; `batch_first` vs `seq_first`; bidirectional = two cells; TF vs PyTorch gate ordering conventions; streaming weights for **four** gate matrices + biases per direction; aligning unrolled vs dynamic ONNX graphs.

**Bedrock idea:** small seq2seq or sequence classifier, 1-layer LSTM, fixed length — native vs export vs loom on same `x_test` trajectories.

---

### RNN — ⬜ not started

**Loom:** `VolumetricLayer` RNN · simpler than LSTM but same time/state issues.

**Planet ops:** `nn.RNN`, `nn.GRU` (GRU is not Loom RNN — scope call needed), Keras `SimpleRNN`, ONNX `RNN`.

**Hard parts:** same recurrent streaming as LSTM with different weight count; some planets push you toward LSTM/GRU only; TF deprecated `SimpleRNN` in places.

**Bedrock idea:** minimal tanh RNN, short sequences — prove state threading before LSTM.

---

### Layer types not listed above (Loom-native, bridge later)

| Loom layer | Why deferred |
|------------|--------------|
| **SwiGLU** | FFN block inside transformers; bridge after MHA + Dense |
| **Embedding** | Integer token input + lookup table; different fixture shape |
| **Residual** | Topology wiring, not standalone planet op |

---

## Dense bedrock — per-planet pipeline

## Dense bedrock — per-planet pipeline

Fixture: `dense_bedrock_v2` · seed 42 · 12 model ids · 5 planets.

### Export formats (native → checkpoint)

| Planet | Export format | Status | Notes |
|--------|---------------|--------|-------|
| **PyTorch** | ONNX | ✅ EXACT | All 12 models |
| **PyTorch** | Safetensors | ✅ EXACT | All 12 models |
| **TensorFlow** | SavedModel | ✅ EXACT | All 12 models |
| **JAX** | — | ⬜ | Native infer only; no export step yet |
| **sklearn** | — | ⬜ | Native infer only; pickle on disk |
| **Paddle** | — | ⬜ | Native infer only |

### Loom entity (native → layer stream → `.stream.entity`)

Mechanism: Python `extract_*` loops dense layers → JSON `layers[]` → Go `bridge.BuildNetworkFromStream` → `bridge.SaveEntity` → Loom infer on shared `x_test`.

| Planet | Extractor | Stream wired | Loom reports | Status |
|--------|-----------|--------------|--------------|--------|
| **PyTorch** | `extract_pytorch_sequential` | ✅ | 1 / 12 | 🟡 `mlp_32_16_4_relu` PASS (~6e-7); rest pending re-run |
| **TensorFlow** | `extract_keras_dense` | ✅ | 1 / 12 | 🟡 `mlp_32_16_4_relu` — re-verify after host/fixture fixes |
| **JAX** | `extract_jax_mlp` | ✅ | 0 / 12 | ⬜ run with host up |
| **sklearn** | `extract_sklearn_mlp` | ✅ | 0 / 12 | ⬜ run with host up |
| **Paddle** | `extract_paddle_mlp` | ✅ | 0 / 12 | ⬜ run with host up |

**To fill pending rows:** `go run .` then `./python/dense/run_engine.sh <planet>` (omit `--skip-loom`).

### Verified example (PyTorch `mlp_32_16_4_relu`)

| Compare | Result | Max abs diff | Plain English |
|---------|--------|--------------|---------------|
| native → export / onnx | EXACT | 0 | ONNX export is faithful |
| native → export / safetensors | EXACT | 0 | Safetensors export is faithful |
| native → loom / entity | PASS | ~5.96e-7 (~0.0000006) | Loom matches native; fp32 vs fp64 noise only |

Artifact: `python/dense/models/pytorch/mlp_32_16_4_relu/mlp_32_16_4_relu.stream.entity`

---

## Go bridge package (`bridge/`)

| Piece | Status | Notes |
|-------|--------|-------|
| `StreamRequest` / layer JSON | ✅ | Row-major `[out × in]` weights + optional bias |
| `BuildNetworkFromStream` | ✅ | Dense layers only |
| `SaveEntity` / `LoadEntity` | ✅ | Biases in `bridge.dense.N.biases` blobs |
| `InferDenseMLP` | ✅ | Bias before activation |
| Fixture loader (`fixtures/*.npz`) | ✅ | Shared `x_test` for compare |
| File importers (ONNX, safetensors, keras, …) | ⬜ | Deliberately dropped for POC; use live stream instead |

---

## Compare host (`host/`)

| Feature | Status |
|---------|--------|
| Dashboard http://localhost:9876/ | ✅ |
| Per-planet pipeline compare | ✅ |
| Plain-decimal diff readout + scale hints | ✅ |
| `POST /api/v1/loom/stream` | ✅ |
| Loom entity catalog table | ✅ |
| Pending loom rows when stream not run | ✅ |

---

## Known issues / caveats

- **Host must be restarted** after Go changes (`go run .` or `./killserver.sh` first).
- **Background `go run` from agents** may get `signal: terminated` — run in your own terminal for long sessions.
- **fp32 planets vs fp64 Loom** — expect PASS not EXACT on loom rows for PyTorch/TF/JAX/Paddle; sklearn (fp64) may be closer to EXACT.
- **TensorFlow loom diff** — one run showed ~1.25 max diff; may be stale run — re-run after fixture `io.ReadAll` fix and host restart.
- **`.onnx.entity` / `.safetensors.entity` files** — experimental artifacts on disk; not the current compare pipeline and not listed in reports.

---

## Next steps (suggested order)

**Dense (finish step 1):**

1. Re-run all five engines with host up → 60 loom reports (12 × 5).
2. Document per-model PASS/EXACT/DIFF in the table below as results land.
3. Add file-based importers (safetensors → entity) if we want chain validation without live Python.

**Then layer-by-layer (steps 2–7):**

4. Scaffold `python/cnn1/` bedrock + extend `bridge` stream schema for Conv1d.
5. CNN2 bedrock (tiny vision).
6. MHA bedrock (single attention block — largest jump).
7. LSTM → RNN bedrock.
8. CNN3 if needed for 3D models.

Do **not** assume dense stream API generalizes — each layer type gets explicit schema fields and Go builder code.

---

## Per-model loom log (fill in as we go)

| Model ID | PyTorch | TensorFlow | JAX | sklearn | Paddle |
|----------|---------|------------|-----|---------|--------|
| `linear_16_4` | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| `linear_32_4` | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| `mlp_32_16_4_relu` | ✅ PASS | 🟡 verify | ⬜ | ⬜ | ⬜ |
| `mlp_32_16_4_tanh` | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| `mlp_32_16_4_sigmoid` | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| `mlp_32_16_8_no_bias` | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| `mlp_16_16_16_16_4_relu` | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| `mlp_32_32_32_8_relu` | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| `mlp_32_64_32_16_8_relu` | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| `mlp_32_128_32_8_relu` | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| `mlp_64_32_16_4_relu` | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| `mlp_128_64_32_16_4_relu` | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |

---

## Related docs

- [`README.md`](./README.md) — project overview
- [`BRIDGE.md`](./BRIDGE.md) — architecture diagrams
- [`python/dense/README.md`](./python/dense/README.md) — conda bedrock, run commands
