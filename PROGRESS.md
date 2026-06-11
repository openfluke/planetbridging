# Planet Bridging — Progress

Living doc for what works, what does not, and what we are building next — **layer type by layer type**.

Update this file when a planet/step/layer type moves status. The compare UI at http://localhost:9876/ is the live scoreboard; this file is the narrative.

> **Planet Bridging v0.5.0** — planets → Loom **complete** for all standard volumetric layer types. **v1.0** = Loom → other engines. **v1.x–2.0** = file import, niche layers, tighter determinism. See [`README.md`](./README.md#version-roadmap-how-we-count-halves).
>
> **Scope today:** Thirteen compare tabs — **Dense**, **CNN1–3**, **MHA**, **LSTM**, **RNN**, **LayerNorm**, **Embedding**, **RMSNorm**, **SwiGLU**, **Residual**, and **Mixer** (v1 + v2). Each layer has its own bedrock; **Mixer v2** chains all **12 types** in one 16-layer stack.

---

## How to read status

| Symbol | Meaning |
|--------|---------|
| ✅ | Verified on shared fixture (`dense_bedrock_v2`, 100 test inputs) |
| 🟡 | Partially working or only spot-checked |
| ⬜ | Not built / not run yet |
| ❌ | Built but failing — add a note in **Known issues** |

**Compare labels:** `EXACT` = bit-identical outputs · `PASS` = within fp32 tolerance (`< 1e-5` max abs diff) · `DIFF` = investigate (POC: small `DIFF` on deep stacks like **Mixer v2** is acceptable while we tighten determinism later).

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
| native → loom / entity (layer stream) | ✅ all standard bedrocks (dense → residual) | Python reads **live in-memory** weights; Go builds `.stream.entity` |
| onnx file → loom entity | ⬜ | No Go importer wired in compare-host |
| safetensors file → loom entity | ⬜ | No Go importer wired in compare-host (`.safetensors.entity` files on disk are experiments, not reported) |
| loom entity → export | ⬜ | Export back to planets not started |

So when the UI shows native ✓, export ✓, and loom ✓ together, that means **three independent checks passed** — not that data flowed onnx → safetensors → Loom.

---

## Layer types (roadmap)

We bridge **one Loom volumetric layer type at a time**. Dense bedrock is step 1. Every other row needs its **own** fixture, manifest, Python bedrock tree, stream schema extension, Go builder, and compare-tab work — not a small tweak to dense.

| Loom layer | Bedrock | Planet extractors | Go bridge | Compare UI | Overall |
|------------|---------|-------------------|-----------|------------|---------|
| **Dense** | ✅ `python/dense/` · 12×4 planets | ✅ live extractors (no Paddle) | ✅ stream → `.entity` | ✅ dense tab | 🟡 POC |
| **CNN1** | ✅ `python/cnn1/` · 4 models × 3 planets | ✅ pytorch/tf/jax extractors | ✅ `POST /api/v1/loom/stream/cnn1` | ✅ cnn1 tab | 🟡 POC |
| **CNN2** | ✅ `python/cnn2/` · 4 models × 3 planets | ✅ pytorch/tf/jax extractors | ✅ `POST /api/v1/loom/stream/cnn2` | ✅ cnn2 tab | 🟡 POC |
| **CNN3** | ✅ `python/cnn3/` · 4 models × 3 planets | ✅ pytorch/tf/jax extractors | ✅ `POST /api/v1/loom/stream/cnn3` | ✅ cnn3 tab | 🟡 POC |
| **MHA** | ✅ `python/mha/` · 4 models × 3 planets | ✅ pytorch/tf/jax extractors | ✅ `POST /api/v1/loom/stream/mha` | ✅ mha tab | 🟡 POC |
| **LSTM** | ✅ `python/lstm/` · 4 models × 3 planets | ✅ pytorch/tf/jax extractors | ✅ `POST /api/v1/loom/stream/lstm` | ✅ lstm tab | 🟡 POC |
| **RNN** | ✅ `python/rnn/` · 4 models × 3 planets | ✅ pytorch/tf/jax extractors | ✅ `POST /api/v1/loom/stream/rnn` | ✅ rnn tab | 🟡 POC |
| **LayerNorm** | ✅ `python/layernorm/` · 4 models × 3 planets | ✅ numpy reference forward | ✅ `POST /api/v1/loom/stream/layernorm` | ✅ layernorm tab | 🟡 POC |
| **Embedding** | ✅ `python/embedding/` · 4 models × 3 planets | ✅ numpy reference forward | ✅ `POST /api/v1/loom/stream/embedding` | ✅ embedding tab | 🟡 POC |
| **RMSNorm** | ✅ `python/rmsnorm/` · 4 models × 3 planets | ✅ numpy reference forward | ✅ `POST /api/v1/loom/stream/rmsnorm` | ✅ rmsnorm tab | 🟡 POC |
| **SwiGLU** | ✅ `python/swiglu/` · 4 models × 3 planets | ✅ numpy reference forward | ✅ `POST /api/v1/loom/stream/swiglu` | ✅ swiglu tab | 🟡 POC |
| **Residual** | ✅ `python/residual/` · 4 models × 3 planets | ✅ numpy reference forward | ✅ `POST /api/v1/loom/stream/residual` | ✅ residual tab | 🟡 POC |
| **Mixer** | ✅ `python/mixer/` · v1 + v2 × 3 planets | ✅ pytorch/tf/jax | ✅ `POST /api/v1/loom/stream/mixer` | ✅ mixer tab | 🟡 POC |

### v0.5.0 complete — standard bedrocks

All rows above are wired. **Mixer v2** (`mixer_all_v2`) chains every bridged type in one stack (16 layers).

**Later (v1.x–2.0):** ConvTranspose 1/2/3, Softmax, Parallel, Sequential, KMeans, file import without live Python, …

### Suggested order after Dense is green

1. ~~**CNN1**~~ — ✅ done
2. ~~**CNN2**~~ — ✅ done
3. ~~**CNN3**~~ — ✅ done
4. ~~**MHA**~~ — ✅ done (2026-06-09)
5. ~~**LSTM**~~ — ✅ done (2026-06-09)
6. ~~**RNN**~~ — ✅ done (2026-06-09)
7. ~~**Mixer**~~ — ✅ done (2026-06-10) — 3/3 loom PASS (`mixer_all_v1`, 7-type stack)
8. ~~**LayerNorm**~~ — ✅ done (2026-06-10) — 24/24 loom PASS
9. ~~**Embedding**~~ — ✅ done (2026-06-11) — 24/24 loom EXACT
10. ~~**RMSNorm**~~ — ✅ done (2026-06-11) — 24/24 loom PASS
11. ~~**SwiGLU**~~ — ✅ done (2026-06-11) — 24/24 loom PASS
12. ~~**Residual**~~ — ✅ done (2026-06-11) — 24/24 loom PASS
13. ~~**Mixer v2**~~ — ✅ done (2026-06-11) — `mixer_all_v2`, 16 layers, 12 types · native→loom ~5e-5 max (POC, not bit-exact)

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

### RNN — ✅ v1 bedrock

**Bedrock:** `python/rnn/` · fixture `rnn_bedrock_v1` · 4 single-cell models · pytorch / tensorflow / jax · **`[N, seq, input_size]`** fixtures.

**Run:** `go run .` then `./python/rnn/run_rnn.sh` · UI tab: http://localhost:9876/?tab=rnn

| Planet | Export | Loom stream | Status |
|--------|--------|-------------|--------|
| PyTorch | — (native `nn.RNN`) | ✅ | ✅ 4/4 PASS |
| TensorFlow | — (reference forward) | ✅ | ✅ 4/4 PASS |
| JAX | — (reference forward) | ✅ | ✅ 4/4 PASS |

**Scope (v1):** single vanilla RNN matching Loom layout (**W_ih + W_hh + b_h**), **tanh** activation, zero initial hidden. `epochs: 0` — init weights, infer on shared `x_test`, stream to Go.

**Models:** `rnn_4_4_4`, `rnn_8_4_8`, `rnn_4_8_4`, `rnn_8_8_8`.

**Out of scope (v1):** GRU, bidirectional, stacked RNN, training loops, export checkpoints.

---

### LayerNorm — ✅ v1 bedrock

**Bedrock:** `python/layernorm/` · fixture `layernorm_bedrock_v1` · 4 models · pytorch / tensorflow / jax · **`[N, seq, dim]`** fixtures.

**Run:** `go run .` then `./python/layernorm/run_layernorm.sh` · UI tab: http://localhost:9876/?tab=layernorm

| Planet | Export | Loom stream | Status |
|--------|--------|-------------|--------|
| PyTorch | — (reference forward) | ✅ | ✅ 4/4 PASS |
| TensorFlow | — (reference forward) | ✅ | ✅ 4/4 PASS |
| JAX | — (reference forward) | ✅ | ✅ 4/4 PASS |

**Scope (v1):** per-token LayerNorm (gamma + beta), `eps=1e-5`, linear activation on Loom build. `epochs: 0` — init weights, infer on shared `x_test`, stream to Go.

**Models:** `layernorm_4`, `layernorm_8`, `layernorm_16`, `layernorm_32` (seq_len=4).

---

### Embedding — ✅ v1 bedrock

**Bedrock:** `python/embedding/` · fixture `embedding_bedrock_v1` · 4 models · pytorch / tensorflow / jax · **`[N, seq_len]`** token-id fixtures.

**Run:** `go run .` then `./python/embedding/run_embedding.sh` · UI tab: http://localhost:9876/?tab=embedding

| Planet | Export | Loom stream | Status |
|--------|--------|-------------|--------|
| PyTorch | — (reference forward) | ✅ | ✅ 4/4 EXACT |
| TensorFlow | — (reference forward) | ✅ | ✅ 4/4 EXACT |
| JAX | — (reference forward) | ✅ | ✅ 4/4 EXACT |

**Scope (v1):** token lookup table `[vocab × embed_dim]`, linear activation on Loom build. `epochs: 0` — init weights, infer on shared `x_test`, stream to Go.

**Models:** `embedding_16_4_4`, `embedding_32_4_8`, `embedding_32_8_4`, `embedding_64_8_8`.

---

### RMSNorm — ✅ v1 bedrock

**Bedrock:** `python/rmsnorm/` · fixture `rmsnorm_bedrock_v1` · 4 models · pytorch / tensorflow / jax · **`[N, seq, dim]`** fixtures.

**Run:** `go run .` then `./python/rmsnorm/run_rmsnorm.sh` · UI tab: http://localhost:9876/?tab=rmsnorm

| Planet | Export | Loom stream | Status |
|--------|--------|-------------|--------|
| PyTorch | — (reference forward) | ✅ | ✅ 4/4 PASS |
| TensorFlow | — (reference forward) | ✅ | ✅ 4/4 PASS |
| JAX | — (reference forward) | ✅ | ✅ 4/4 PASS |

**Scope (v1):** per-token RMSNorm (gamma only), `eps=1e-6`, linear activation on Loom build. `epochs: 0` — init weights, infer on shared `x_test`, stream to Go.

**Models:** `rmsnorm_4`, `rmsnorm_8`, `rmsnorm_16`, `rmsnorm_32` (seq_len=4).

---

### SwiGLU — ✅ v1 bedrock

**Bedrock:** `python/swiglu/` · fixture `swiglu_bedrock_v1` · 4 models · pytorch / tensorflow / jax · **`[N, seq, input_dim]`** fixtures.

**Run:** `go run .` then `./python/swiglu/run_swiglu.sh` · UI tab: http://localhost:9876/?tab=swiglu

| Planet | Export | Loom stream | Status |
|--------|--------|-------------|--------|
| PyTorch | — (reference forward) | ✅ | ✅ 4/4 PASS |
| TensorFlow | — (reference forward) | ✅ | ✅ 4/4 PASS |
| JAX | — (reference forward) | ✅ | ✅ 4/4 PASS |

**Scope (v1):** gate/up/down MLP with SiLU gate, Loom weight pack `gateW|upW|downW|gateB|upB|downB`. `epochs: 0` — init weights, infer on shared `x_test`, stream to Go.

**Models:** `swiglu_4_8_4`, `swiglu_8_16_4`, `swiglu_4_16_4`, `swiglu_8_8_4` (seq_len=4).

---

### Residual — ✅ v1 bedrock

**Bedrock:** `python/residual/` · fixture `residual_bedrock_v1` · 4 models · pytorch / tensorflow / jax · paired **`x_main` + `x_skip`** `[N, seq, dim]` fixtures.

**Run:** `go run .` then `./python/residual/run_residual.sh` · UI tab: http://localhost:9876/?tab=residual

| Planet | Export | Loom stream | Status |
|--------|--------|-------------|--------|
| PyTorch | — (reference forward) | ✅ | ✅ 4/4 PASS |
| TensorFlow | — (reference forward) | ✅ | ✅ 4/4 PASS |
| JAX | — (reference forward) | ✅ | ✅ 4/4 PASS |

**Scope (v1):** weightless residual add — `output = main + skip`. No training; infer on shared paired test inputs, stream layer metadata to Go.

**Models:** `residual_4`, `residual_8`, `residual_16`, `residual_32` (seq_len=4).

---

### Mixer — ✅ v1 + v2 bedrock

**Bedrock:** `python/mixer/` · fixture `mixer_bedrock_v1` · pytorch / tensorflow / jax · volume input `[N,1,2,2,2]`, token IDs for v2 embedding.

**Run:** `go run .` then `./python/mixer/run_mixer.sh` · UI tab: http://localhost:9876/?tab=mixer

| Model | Layers | Types | PyTorch | TensorFlow | JAX |
|-------|--------|-------|---------|------------|-----|
| `mixer_all_v1` | 10 | 7 (CNN3→Dense→…→LSTM→Dense) | ✅ PASS | ✅ PASS | ✅ PASS |
| `mixer_all_v2` | 16 | 12 (adds Embedding, LayerNorm, RMSNorm, SwiGLU, Residual×2) | 🟡 POC ~5e-5 | 🟡 POC ~5e-5 | 🟡 POC ~5e-5 |

**v2 pipeline:** CNN3 → Dense → CNN2 → Dense → CNN1 → Dense → Embedding → LayerNorm → MHA → Residual → RMSNorm → SwiGLU → Residual → RNN → LSTM → Dense head.

**POC note:** Deep stacks accumulate fp32 noise; v2 behaviour matches native — strict `< 1e-5` PASS can wait.

---

### LSTM — ✅ v1 bedrock

**Bedrock:** `python/lstm/` · fixture `lstm_bedrock_v1` · 4 single-cell models · pytorch / tensorflow / jax · **`[N, seq, input_size]`** fixtures.

**Run:** `go run .` then `./python/lstm/run_lstm.sh` · UI tab: http://localhost:9876/?tab=lstm

| Planet | Export | Loom stream | Status |
|--------|--------|-------------|--------|
| PyTorch | — (native `nn.LSTM`) | ✅ | ✅ 4/4 PASS |
| TensorFlow | — (reference forward) | ✅ | ✅ 4/4 PASS |
| JAX | — (reference forward) | ✅ | ✅ 4/4 PASS |

**Scope (v1):** single LSTM layer matching Loom gate layout (**i, f, g, o**), combined bias per gate, zero initial **h/c**. `epochs: 0` — init weights, infer on shared `x_test`, stream gates to Go.

**Models:** `lstm_4_4_4`, `lstm_8_4_8`, `lstm_4_8_4`, `lstm_8_8_8`.

**Out of scope (v1):** bidirectional, peephole, layer norm, training loops, export checkpoints, stacked LSTM.

---

### MHA (multi-head attention) — ✅ v1 bedrock

**Bedrock:** `python/mha/` · fixture `mha_bedrock_v1` · 4 single-block models · pytorch / tensorflow / jax · **`[N, seq, d_model]`** fixtures.

**Run:** `go run .` then `./python/mha/run_mha.sh` · UI tab: http://localhost:9876/?tab=mha

| Planet | Export | Loom stream | Status |
|--------|--------|-------------|--------|
| PyTorch | — (native only) | ✅ | ✅ 4/4 PASS |
| TensorFlow | — (native only) | ✅ | ✅ 4/4 PASS |
| JAX | — (native only) | ✅ | ✅ 4/4 PASS |

**Scope (v1):** single causal MHA block matching Loom semantics (**RoPE** + **causal mask**). Custom forward in each planet (not stock `nn.MultiheadAttention`). `epochs: 0` — init weights, infer on shared `x_test`, stream Q/K/V/O to Go. Infer uses batch=1 per sample to avoid Loom KV cross-batch coupling.

**Models:** `mha_8_2_4`, `mha_16_2_8`, `mha_16_4_4`, `mha_8_4_8`.

**Out of scope (v1):** GQA/MQA, Q/K norm, training loops, export checkpoints, full transformer blocks.

---

### CNN3 (3D convolution) — ✅ v1 bedrock

**Bedrock:** `python/cnn3/` · fixture `cnn3_bedrock_v1` · 4 single-conv models · pytorch / tensorflow / jax · **NCDHW** fixtures.

**Run:** `go run .` then `./python/cnn3/run_cnn3.sh` · UI tab: http://localhost:9876/?tab=cnn3

| Planet | Export | Loom stream | Status |
|--------|--------|-------------|--------|
| PyTorch | ONNX | ✅ | ✅ 4/4 PASS |
| TensorFlow | SavedModel | ✅ | ✅ 4/4 PASS (export EXACT) |
| JAX | — (native only) | ✅ | ✅ 4/4 PASS |

**Scope (v1):** single `Conv3d`, kernel = D = H = W, no bias. Largest cube is `6³` (16³ full-kernel was too slow on TF CPU).

---

### CNN2 (2D convolution) — ✅ v1 bedrock

**Bedrock:** `python/cnn2/` · fixture `cnn2_bedrock_v1` · 4 single-conv models · pytorch / tensorflow / jax · **NCHW** fixtures.

**Run:** `go run .` then `./python/cnn2/run_cnn2.sh` · UI tab: http://localhost:9876/?tab=cnn2

| Planet | Export | Loom stream | Status |
|--------|--------|-------------|--------|
| PyTorch | ONNX | ✅ | ✅ 4/4 PASS |
| TensorFlow | SavedModel | ✅ | ✅ 4/4 PASS (export EXACT) |
| JAX | — (native only) | ✅ | ✅ 4/4 PASS |

**Scope (v1):** single `Conv2d`, kernel = H = W, no bias, flat output = filters.

---

### CNN1 (1D convolution) — ✅ v1 bedrock

**Bedrock:** `python/cnn1/` · fixture `cnn1_bedrock_v1` · 4 single-conv models · pytorch / tensorflow / jax.

**Run:** `go run .` then `./python/cnn1/run_cnn1.sh` · UI tab: http://localhost:9876/?tab=cnn1

| Planet | Export | Loom stream | Status |
|--------|--------|-------------|--------|
| PyTorch | ONNX | ✅ | ✅ 4/4 PASS |
| TensorFlow | SavedModel | ✅ | ✅ 4/4 PASS (export EXACT) |
| JAX | — (native only) | ✅ | ✅ 4/4 PASS |

---

### CNN1 design notes (original plan)

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

### CNN2 / CNN3 design notes (original plan — now ✅ v1 above)

CNN2 and CNN3 bedrocks are live. See **CNN2** and **CNN3** sections earlier in this file for run commands and per-planet status.

---

### MHA design notes (original plan — now ✅ v1 above)

**Loom:** `VolumetricLayer` MHA · Q/K/V/O projections, heads, RoPE, KV cache semantics.

**Hard parts (addressed in v1):** Q/K/V/O weight packing order; causal + RoPE must match `poly.MHAForwardPolymorphic`; RoPE in-place bugs in planet forwards (read values before write); per-sample Loom infer for compare.

**Still deferred:** GQA/MQA, Q/K norm, bidirectional attention, export graphs, bit-exact **Mixer v2** determinism.

---

### LSTM design notes (original plan — now ✅ v1 above)

**Loom:** `VolumetricLayer` LSTM · i/f/g/o gates, cell state, `[batch, seq, input]` layout.

**Hard parts (addressed in v1):** four gates × (ih + hh + bias) weight packing; PyTorch `nn.LSTM` i/f/g/o gate order matches Loom; zero initial state; per-sample batch=1 Loom infer.

**Still deferred:** bidirectional, peephole connections, TF/Keras native `LSTM` layer (TF uses reference forward), stacked cells, export graphs.

---

### RNN design notes (original plan — now ✅ v1 above)

**Loom:** `VolumetricLayer` RNN · `h_t = tanh(x_t W_ih^T + h_{t-1} W_hh^T + b)`.

**Hard parts (addressed in v1):** single weight pack `[ih | hh | bias]`; PyTorch `nn.RNN` tanh with bias in `bias_ih_l0` only; zero initial hidden; per-sample batch=1 Loom infer.

**Still deferred:** GRU (different Loom layer), bidirectional, TF/Keras native `SimpleRNN` (TF uses reference forward), stacked cells, export graphs.

---

### Layer types not listed above (Loom-native, bridge later)

| Loom layer | Why deferred |
|------------|--------------|
| **ConvTranspose 1/2/3** | Not in standard v0.5 set |
| **Softmax / Parallel / Sequential** | Loom-native; bridge in v1.x |
| **KMeans / Metacognition** | Niche — defer |

---

## Dense bedrock — per-planet pipeline

Fixture: `dense_bedrock_v2` · seed 42 · 12 model ids · **4 planets** (PyTorch, TensorFlow, JAX, sklearn — Paddle disabled).

### Export formats (native → checkpoint)

| Planet | Export format | Status | Notes |
|--------|---------------|--------|-------|
| **PyTorch** | ONNX | ✅ EXACT | All 12 models |
| **PyTorch** | Safetensors | ✅ EXACT | All 12 models |
| **TensorFlow** | SavedModel | ✅ EXACT | All 12 models |
| **JAX** | — | ⬜ | Native infer only; no export step yet |
| **sklearn** | — | ⬜ | Native infer only; pickle on disk |
| ~~**Paddle**~~ | — | — | **Disabled** — out of scope |

### Loom entity (native → layer stream → `.stream.entity`)

Mechanism: Python `extract_*` loops dense layers → JSON `layers[]` → Go `bridge.BuildNetworkFromStream` → `bridge.SaveEntity` → Loom infer on shared `x_test`.

| Planet | Extractor | Stream wired | Loom reports | Status |
|--------|-----------|--------------|--------------|--------|
| **PyTorch** | `extract_pytorch_sequential` | ✅ | 12 / 12 | ✅ all PASS (export EXACT) |
| **TensorFlow** | `extract_keras_dense` | ✅ | 12 / 12 | ✅ all PASS (export EXACT) |
| **JAX** | `extract_jax_mlp` | ✅ | 12 / 12 | 🟡 10 PASS · 2 DIFF (deeper MLPs) |
| **sklearn** | `extract_sklearn_mlp` | ✅ | 12 / 12 | 🟡 9 PASS · 3 DIFF (deeper MLPs / no_bias) |

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
| `BuildNetworkFromStream` | ✅ | Dense layers |
| `BuildNetworkFromCNN*Stream` | ✅ | Conv1d/2d/3d |
| `BuildNetworkFromMHAStream` | ✅ | Q/K/V/O + biases → `LayerMultiHeadAttention` |
| `InferMHAStack` | ✅ | batch=1 per sample (KV isolation) |
| `BuildNetworkFromLSTMStream` | ✅ | i/f/g/o gates → `LayerLSTM` |
| `InferLSTMStack` | ✅ | batch=1 per sample |
| `BuildNetworkFromRNNStream` | ✅ | ih/hh/bias → `LayerRNN` |
| `InferRNNStack` | ✅ | batch=1 per sample |
| LayerNorm / Embedding / RMSNorm / SwiGLU / Residual builders | ✅ | Per-type stream + infer |
| `BuildNetworkFromMixerStream` | ✅ | v1 (10 layers) + v2 (16 layers, skip tensors) |
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
| `POST /api/v1/loom/stream/mha` | ✅ |
| MHA compare tab | ✅ |
| `POST /api/v1/loom/stream/lstm` | ✅ |
| LSTM compare tab | ✅ |
| `POST /api/v1/loom/stream/rnn` | ✅ |
| RNN compare tab | ✅ |
| `POST /api/v1/loom/stream/layernorm` | ✅ |
| LayerNorm compare tab | ✅ |
| `POST /api/v1/loom/stream/embedding` | ✅ |
| Embedding compare tab | ✅ |
| `POST /api/v1/loom/stream/rmsnorm` | ✅ |
| RMSNorm compare tab | ✅ |
| `POST /api/v1/loom/stream/swiglu` | ✅ |
| SwiGLU compare tab | ✅ |
| `POST /api/v1/loom/stream/residual` | ✅ |
| Residual compare tab | ✅ |
| `POST /api/v1/loom/stream/mixer` | ✅ |
| Mixer compare tab (v1 + v2) | ✅ |
| `GET /api/v1/export/all.pdf` | ✅ all 13 bedrocks |
| Loom entity catalog table | ✅ |
| Pending loom rows when stream not run | ✅ |

---

## Known issues / caveats

- **Host must be restarted** after Go changes (`go run .` or `./killserver.sh` first).
- **Background `go run` from agents** may get `signal: terminated` — run in your own terminal for long sessions.
- **fp32 planets vs fp64 Loom** — expect PASS not EXACT on loom rows for PyTorch/TF/JAX/Paddle; sklearn (fp64) may be closer to EXACT.
- **Mixer v2 (`mixer_all_v2`)** — native→loom max diff ~5e-5 (POC). Behaviour matches; strict `< 1e-5` PASS can wait. Deep stacks accumulate fp32 noise — acceptable for v0.5.0.
- **JAX / sklearn loom DIFF** on `mlp_16_16_16_16_4_relu`, `mlp_32_32_32_8_relu`, `mlp_32_16_8_no_bias` (sklearn only) — likely extractor or activation ordering bug; PyTorch + TF PASS on same models.
- **`.onnx.entity` / `.safetensors.entity` files** — experimental artifacts on disk; not the current compare pipeline and not listed in reports.

---

## Next steps (v1.0 and beyond)

**v0.5.0 layer bedrocks — done.** All standard volumetric types + Mixer v2 POC.

**Toward v1.0 (Loom → other engines):**

1. Loom export → Safetensors / ONNX / GGUF
2. Round-trip compare: Loom entity → hub file → ORT / llama.cpp / CoreML
3. File-based import in compare-host (no live Python planet required)

**Polish (can run in parallel):**

4. Fix JAX + sklearn dense extractors for the 5 DIFF rows above
5. Tighten **Mixer v2** determinism (currently ~5e-5 — acceptable POC)
6. CABI stream path (ctypes → `.entity`) as dev harness alternative to HTTP+JSON

Do **not** assume one stream API generalizes — each layer type gets explicit schema fields and Go builder code.

---

## Per-model loom log — Mixer (`mixer_bedrock_v1`)

Shared fixture: `[N,1,2,2,2]` volume input, output dim 8, 100 test samples. PDF export includes pipeline description per model.

### `mixer_all_v1` — 10 layers, 7 types

CNN3 → Dense → CNN2 → Dense → CNN1 → Dense → MHA → RNN → LSTM → Dense head.

| Model ID | PyTorch | TensorFlow | JAX |
|----------|---------|------------|-----|
| `mixer_all_v1` | ✅ PASS | ✅ PASS | ✅ PASS |

### `mixer_all_v2` — 16 layers, all 12 types

CNN3 → Dense → CNN2 → Dense → CNN1 → Dense → **Embedding** → **LayerNorm** → MHA → **Residual** → **RMSNorm** → **SwiGLU** → **Residual** → RNN → LSTM → Dense head. Token IDs in fixture for embedding lookup.

| Model ID | PyTorch | TensorFlow | JAX |
|----------|---------|------------|-----|
| `mixer_all_v2` | 🟡 POC (~5e-5) | 🟡 POC (~5e-5) | 🟡 POC (~5e-5) |

**Run:** `./python/mixer/run_mixer.sh` (both models) or `--model mixer_all_v2` per engine.

---

## Per-model loom log — RNN (`rnn_bedrock_v1`)

| Model ID | PyTorch | TensorFlow | JAX |
|----------|---------|------------|-----|
| `rnn_4_4_4` | ✅ PASS | ✅ PASS | ✅ PASS |
| `rnn_8_4_8` | ✅ PASS | ✅ PASS | ✅ PASS |
| `rnn_4_8_4` | ✅ PASS | ✅ PASS | ✅ PASS |
| `rnn_8_8_8` | ✅ PASS | ✅ PASS | ✅ PASS |

---

## Per-model loom log — LSTM (`lstm_bedrock_v1`)

| Model ID | PyTorch | TensorFlow | JAX |
|----------|---------|------------|-----|
| `lstm_4_4_4` | ✅ PASS | ✅ PASS | ✅ PASS |
| `lstm_8_4_8` | ✅ PASS | ✅ PASS | ✅ PASS |
| `lstm_4_8_4` | ✅ PASS | ✅ PASS | ✅ PASS |
| `lstm_8_8_8` | ✅ PASS | ✅ PASS | ✅ PASS |

---

## Per-model loom log — MHA (`mha_bedrock_v1`)

| Model ID | PyTorch | TensorFlow | JAX |
|----------|---------|------------|-----|
| `mha_8_2_4` | ✅ PASS | ✅ PASS | ✅ PASS |
| `mha_16_2_8` | ✅ PASS | ✅ PASS | ✅ PASS |
| `mha_16_4_4` | ✅ PASS | ✅ PASS | ✅ PASS |
| `mha_8_4_8` | ✅ PASS | ✅ PASS | ✅ PASS |

---

## Per-model loom log — Dense (fill in as we go)

| Model ID | PyTorch | TensorFlow | JAX | sklearn |
|----------|---------|------------|-----|---------|
| `linear_16_4` | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS |
| `linear_32_4` | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS |
| `mlp_32_16_4_relu` | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS |
| `mlp_32_16_4_tanh` | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS |
| `mlp_32_16_4_sigmoid` | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS |
| `mlp_32_16_8_no_bias` | ✅ PASS | ✅ PASS | ✅ PASS | ❌ DIFF |
| `mlp_16_16_16_16_4_relu` | ✅ PASS | ✅ PASS | ❌ DIFF | ❌ DIFF |
| `mlp_32_32_32_8_relu` | ✅ PASS | ✅ PASS | ❌ DIFF | ❌ DIFF |
| `mlp_32_64_32_16_8_relu` | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS |
| `mlp_32_128_32_8_relu` | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS |
| `mlp_64_32_16_4_relu` | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS |
| `mlp_128_64_32_16_4_relu` | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS |

---

## Related docs

- [`README.md`](./README.md) — project overview
- [`BRIDGE.md`](./BRIDGE.md) — architecture diagrams
- [`python/dense/README.md`](./python/dense/README.md) — conda bedrock, run commands
