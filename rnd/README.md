# Planet Bridging — R&D

Automated research and development on **talking between planets' AI engines** — how different model runtimes, file formats, and inference stacks can communicate and interoperate. This folder holds exploratory notes; the long-term goal is to **build a bridge between these planets** inside [Loom](../README.md).

Sources consolidated here: [chatgpt.pdf](./chatgpt.pdf), [google.pdf](./google.pdf), [claude.pdf](./claude.pdf), [grok.md](./grok.md).

---

## The problem

Each AI "planet" speaks its own dialect: PyTorch, TensorFlow, llama.cpp, ONNX Runtime, CoreML, TensorRT, and others each have native serialization, operator sets, and runtime assumptions. Models do not flow freely — they pass through conversion tools, lose fidelity, or get locked to a single engine.

Loom is a pure-Go, zero-CGO neural runtime (a Deterministic Neural Virtual Machine). To load models from other planets without abandoning that constraint, we need to understand every major format, map conversion paths, and decide what to parse natively vs. what to reject with a clear conversion message.

---

## Executive summary

All four research passes converge on the same picture:

| Finding | Detail |
|---------|--------|
| **Three hub formats** | **Safetensors** (HF weight distribution), **ONNX** (cross-framework graph interchange), **GGUF** (local/edge LLM inference). Everything else feeds into or out of these hubs. |
| **Loom today** | Native **Safetensors** loader + **Loom JSON** persistence (`model.json` — full architecture + bit-packed weights across 21 DTypes). Config-driven transformer loading for specific HF families. |
| **Pure-Go sweet spot** | Safetensors (solved), GGUF parsing + dequant (feasible, high ROI), ONNX protobuf parsing + scoped executor (mechanical but large), NumPy/Flax msgpack (trivial). |
| **Hard blockers (historical → current)** | PyTorch pickle and Keras HDF5 were blockers; both now have pure-Go paths with caveats (`nlpodyssey/gopickle`, `scigolib/hdf5`). **TF SavedModel checkpoint v2** (LevelDB SSTable index) remains the one genuinely unsolved pure-Go gap. |
| **Runtime-locked targets** | TensorRT `.engine`, CoreML execution, OpenVINO IR, ExecuTorch `.pte`, MLC/TVM native bundles — parse-only at best; no execution in Loom. Document conversion paths and reject with clear errors. |
| **Strategic differentiator** | Pure-Go **GGUF TQ1_0/TQ2_0 dequant** for BitNet b1.58 — no existing pure-Go library dequantizes K-quants, I-quants, or T-quants today; aligns with Loom's existing BitNet execution path. |

---

## Format spectrum

Formats sit on a spectrum from **weights-only** (need separate architecture code) to **full computation graph** (self-contained runnable model).

| Format | Extension | Graph? | Weights | Primary engines | Pure-Go load |
|--------|-----------|--------|---------|-----------------|--------------|
| **Safetensors** | `.safetensors` | No | Yes (raw/quant tensors) | HF Transformers, PyTorch, Loom | **Easy** — `nlpodyssey/safetensors` |
| **GGUF** | `.gguf` | Implicit (metadata KV) | Yes (quantized) | llama.cpp, Ollama, GPT4All | **Medium** — parsers exist; dequant is the work |
| **ONNX** | `.onnx` (+ optional `.data`) | Yes (protobuf nodes) | Embedded or external | ONNX Runtime, TensorRT, OpenVINO | **Medium–Hard** — parse easy; executing all ops is years |
| **PyTorch checkpoint** | `.pt`, `.pth`, `.bin` | Sometimes (pickle) | Yes | PyTorch, HF | **Hard/unsafe** — pickle; state_dict via allowlisted unpickler is tractable |
| **TorchScript** | `.pt` | Yes (JIT IR) | Yes | PyTorch LibTorch | Hard — no Go parser |
| **TF SavedModel** | directory (`.pb`) | Yes | Yes (SSTable shards) | TensorFlow, TFLite | **Very hard** — SSTable reader unsolved |
| **TF GraphDef** | `.pb` | Yes (frozen) | Embedded in Const nodes | TensorFlow | Medium — protobuf only |
| **Keras HDF5** | `.h5`, `.keras` | Yes (JSON in HDF5) | Yes | Keras/TF | **Medium** — `scigolib/hdf5` (beta) |
| **TFLite** | `.tflite` | Yes (FlatBuffers) | Yes | TF Lite, mobile | Medium — `flatc --go` on schema |
| **CoreML** | `.mlmodel`, `.mlpackage` | Yes (protobuf/MIL) | Yes | Apple Neural Engine | Hard parse; execution pointless in Go |
| **OpenVINO IR** | `.xml` + `.bin` | Yes (XML) | Yes | Intel OpenVINO | Low parse effort; proprietary semantics |
| **Loom JSON** | `model.json` | Yes | Yes (bit-packed) | Loom | **Native** |

Other formats surveyed but lower priority: MXNet (`.json` + `.params`), Caffe (`.prototxt` + `.caffemodel`), Paddle (`.pdmodel` + `.pdiparams`), Darknet (`.weights` + `.cfg`), llamafile (APE + GGUF), NumPy (`.npy`/`.npz`), Flax msgpack, Orbax/Zarr (multi-month effort — defer), MLC-LLM, EXL2/EXL3.

---

## Safetensors (primary hub)

- **Layout**: 8-byte LE u64 header length → UTF-8 JSON header → raw tensor bytes (C-order, LE).
- **Header**: `{tensor_name: {dtype, shape, data_offsets: [start, end]}, __metadata__: {...}}`.
- **No version field** — format frozen since 2022; compatibility via dtype enum extension (F32, F16, BF16, FP8 variants, INT4/8, etc.).
- **Sharding**: `model-NNNNN-of-MMMMM.safetensors` + `model.safetensors.index.json`.
- **Quantization is out-of-band**: GPTQ/AWQ/bitsandbytes store extra tensors (`qweight`, `scales`, `qzeros`, `g_idx`) with naming conventions + `quantization_config` in companion `config.json`. No universal contract — loader must pair tensors per architecture.
- **Security**: No code execution; zero-copy mmap safe.
- **Loom gap**: Finish writer, sharded index reader, FP8/FP4 dtype handling (~1 week).

---

## GGUF (second hub)

- **Magic**: `'G','G','U','F'` (LE u32 `0x46554747`). Versions v1→v3 (v3 adds big-endian support).
- **Layout**: magic | version | tensor_count | metadata_kv_count | metadata KVs | tensor_infos | padding | tensor_data (32-byte aligned by default).
- **Metadata**: Typed KV store (`general.architecture`, `llama.block_count`, tokenizer keys, `tokenizer.chat_template`, etc.). Architecture inferred from metadata, not explicit graph nodes.
- **Quant types**: F32/F16/BF16, legacy block quants (Q4_0, Q8_0), K-quants (Q4_K, Q6_K — 256-element super-blocks), I-quants (codebook-based), **TQ1_0/TQ2_0** (BitNet ternary, PR #8151).
- **Pure-Go parsers**: `gpustack/gguf-parser-go`, `abrander/gguf`, Ollama's `fs/ggml` — metadata + raw bytes, **no dequant** for K/I/T-quants.
- **Effort**: v1/v2/v3 reader ~1 week; F32/F16 + legacy quants ~1–2 days; K-quants ~4–6 days; TQ2_0 ~0.5 day; TQ1_0 ~1–2 days; full llama.cpp parity ~2–3 months.

---

## ONNX (third hub)

- **Container**: Protobuf `ModelProto` → `GraphProto` (nodes, initializers, inputs/outputs).
- **Versioning**: IR version (currently through IR 13) + per-domain opset (ai.onnx through ~23–25). LLM-critical milestones: LayerNorm (opset 17), FP8 Q/DQ (19), INT4 (21), Attention/RoPE/RMSNorm (23).
- **Weights**: Embedded, `raw_data`, or external `.data` files (>2 GB); sub-byte types packed LSB-first.
- **Production LLM ONNX** still relies heavily on `com.microsoft` contrib ops: `MatMulNBits`, `GroupQueryAttention`, `RotaryEmbedding`, `SkipSimplifiedLayerNormalization`, etc.
- **Pure-Go**: `gonnx`, `oramasearch/onnx-go`, `gomlx/onnx-gomlx` — parse well; execution coverage ~25% of ops. Full ORT rewrite = 50–150 engineer-weeks.
- **Scoped strategy**: Parse any IR 1–13 (~1–2 weeks); external-data mmap resolver; executor for ~30 LLM-shaped ops (~8–16 weeks). Reject unsupported opsets with clear errors.

---

## Interoperability: where weights flow

```
Training frameworks          Hub formats              Inference engines
─────────────────           ───────────              ─────────────────
PyTorch/HF  ──────────────► Safetensors ◄──────────► Loom (native)
                │                │
                ├─ torch.onnx ──► ONNX ◄────────────► ONNX Runtime, TensorRT, OpenVINO
                └─ convert_hf ──► GGUF ◄────────────► llama.cpp, Ollama
TensorFlow    ── tf2onnx ─────► ONNX
              ── tflite conv ─► TFLite ─────────────► mobile/edge
CoreML        ◄── coremltools (from PyTorch/ONNX) ── Apple ANE (runtime-locked)
```

**Routine paths**: HF ↔ Safetensors (default since 2023); PyTorch ↔ ONNX (`torch.onnx.export`); HF ↔ GGUF (`convert_hf_to_gguf.py`); TF ↔ ONNX (`tf2onnx`).

**Lossy or one-way**: GGUF → Safetensors (dequant recoverable, packing lost); GPTQ ↔ AWQ ↔ GGUF Q4_K (different group sizes — dequant + requant); ONNX → PyTorch (loses native ops); TensorRT/CoreML/OpenVINO engines have no useful reverse paths.

**Security**: Never load arbitrary pickle (`.pt`/`.bin`) without an allowlist mirroring PyTorch 2.6's `_weights_only_unpickler`. Safetensors and ONNX protobuf are safe. Sandbox any external Python conversion.

---

## What Loom has today

- **Safetensors**: Native loader (`safetensors.go`, `universal_loader.go`, transformer-specific loaders) for supported HF architectures.
- **Loom JSON**: Full serialize/deserialize — architecture + bit-packed weights for all 21 DTypes (Int8, Int4, Binary, FP4, etc.).
- **Gap**: Safetensors is weights-only; Loom must supply architecture from `config.json` and map tensor names to volumetric layers. Same pattern needed for GGUF (metadata-driven) and ONNX (explicit graph → layer mapping).

For the full ecosystem diagram (formats ↔ engines ↔ Loom), see the Mermaid chart in [grok.md](./grok.md) (paste into [mermaid.live](https://mermaid.live)).

---

## Pure-Go library inventory

| Need | Library | Notes |
|------|---------|-------|
| Safetensors | `nlpodyssey/safetensors` | Read + write + streaming |
| GGUF parse | `gpustack/gguf-parser-go` | Production-used; no dequant |
| ONNX protobuf | `gonnx`, `onnx-go`, `gomlx/onnx-gomlx` | Parse + partial exec |
| HDF5/Keras | `scigolib/hdf5` | v0.13+; beta, validate against real Keras files |
| PyTorch state_dict | `nlpodyssey/gopickle`, `kisielk/og-rek` | Allowlist-only; no TorchScript |
| NumPy | `codeberg.org/sbinet/npyio` | Trivial integration |
| Flax legacy | `vmihailenco/msgpack/v5` | Custom ext types ~300 LoC |
| Protobuf | `google.golang.org/protobuf` | ONNX, TF protos, CoreML, Caffe |
| FlatBuffers | `google/flatbuffers` | TFLite, `.ort` |
| mmap | `golang.org/x/exp/mmap` | Zero-copy weight loading |
| Tokenizers | `sugarme/tokenizer` (pure Go) vs CGO wrappers | Fidelity vs purity tradeoff |

**Unsolved in pure Go (mid-2026)**: Zarr v2/v3 (Orbax/JAX checkpoints), TF checkpoint v2 SSTable reader (~2–3 week LevelDB Table port).

---

## Recommended integration roadmap

Merged priority order from all four research passes — sequenced for maximum ecosystem coverage per engineer-week:

### Phase 1 — Safetensors (≤1 week)
Complete writer, sharded index reader, FP8/FP4 dtypes. Validates against `nlpodyssey/safetensors` for byte parity. Buys the entire HF Transformers ecosystem since 2023 and MLX models.

### Phase 2 — GGUF reader + strategic dequant (2–3 weeks)
Header/KV metadata, tensor info, alignment, mmap. Dequant kernels in order: F32/F16/BF16 → **TQ2_0 → TQ1_0** (BitNet differentiator) → Q8_0, Q4_0, Q4_K, Q6_K. Architecture registry per family (llama, qwen2/3, mistral, phi3, gemma, bitnet) — ~2–3 days per family.

### Phase 3 — Small-format wins (1–2 weeks)
GGUF writer, llamafile extraction (~50 LoC over GGUF), Darknet `.weights` (~150 LoC), `.npy`/`.npz` via npyio, Flax msgpack.

### Phase 4 — ONNX scoped executor (2–4 weeks)
Generate Go bindings from `onnx.proto`; external-data mmap resolver; sub-byte/FP8 unpacking. Execute ~30 LLM-shaped ops (Attention-23, GQA, RoPE, RMSNorm, MatMulNBits, Gather, Q/DQ-INT4, etc.). Parse-only `.ort` via `flatc --go ort.fbs` (+1 week).

### Phase 5 — PyTorch state_dict (opportunistic, ~1 week)
Integrate `gopickle` with hard-coded allowlist. Reject TorchScript, `.pt2`, arbitrary `nn.Module` subclasses — document conversion to Safetensors via `transformers` first.

### Phase 6 — Keras (contingent)
`.keras` outer ZIP (1 day) + inner HDF5 via `scigolib/hdf5` (1–2 weeks with real-file validation). Python fallback converter for edge cases.

### Phase 7 — TF SavedModel (defer)
Frozen `.pb` GraphDef with Const weights = 2-day protobuf path. Full SavedModel requires SSTable port — only if user demand emerges.

### Skip in v1
TensorRT engines, CoreML execution, TVM/MLC native bundles, arbitrary pickled models, Stable Diffusion raw `.ckpt`, Paddle dygraph `.pdparams`, Orbax/Zarr, EXL2 kernels. Detect and reject with documented conversion paths.

---

## Loader architecture

All sources agree on a detect → load → integrate pipeline:

```
Model input
    │
    ▼
Detect format (extension + magic bytes)
    │
    ├─ .safetensors ──► Native Go loader (mmap) ──────────────┐
    ├─ .gguf ─────────► Native Go parser + dequant ───────────┤
    ├─ .onnx ─────────► Native Go protobuf + scoped executor ─┤
    ├─ .npy/.npz ─────► npyio ────────────────────────────────┤
    ├─ Loom JSON ─────► Native deserialize ───────────────────┤
    ├─ .pt/.pth ──────► Allowlisted gopickle OR ext convert ──┤
    ├─ .h5/.keras ────► scigolib/hdf5 OR ext convert ─────────┤
    └─ everything else ► External convert → hub format ───────┤
                                                                ▼
                                              Build universal topology AST
                                              (explicit graph OR heuristic from
                                               tensor names + config/metadata)
                                                                │
                                                                ▼
                                              Map to Loom VolumetricNetwork
                                                                │
                                                                ▼
                                                     Inference in Loom
```

**Google's addition**: compile all ingested formats into a **universal intermediate AST** before mapping to Loom's volumetric grid. ONNX/TFLite give explicit nodes; Safetensors/GGUF require **heuristic graph assembly** from tensor key patterns (`blk.12.ffn_down.weight`) and metadata KVs.

**Memory strategy**: mmap multi-GB files; avoid GC scanning huge heap allocations (`unsafe` pointer casts or off-heap buffers). Critical for edge/WASM deployments.

**External conversion fallback** (ChatGPT): sandboxed Python subprocess for formats with no pure-Go path — `.pt` → Safetensors, SavedModel → ONNX, `.mlmodel` → ONNX, `.gguf` → Safetensors (via llama.cpp tools), etc. Limit to offline/trusted workflows.

---

## Runtime-locked planets (bridge = conversion only)

These formats are consumed by exactly one inference engine. Loom should not attempt execution:

| Format | Engine | Loom action |
|--------|--------|-------------|
| TensorRT `.engine` | NVIDIA GPU (arch-specific) | Detect, reject, suggest ONNX → trtexec |
| CoreML `.mlpackage` | Apple Neural Engine | Parse-only optional; suggest coremltools |
| OpenVINO IR | Intel | Parse XML+weights optional; suggest mo |
| TFLite `.tflite` | TF Lite interpreter | Parse feasible; exec = months |
| ExecuTorch `.pte` | PyTorch Edge | Skip |
| MLC-LLM bundles | MLC runtime | Skip |

---

## BitNet b1.58 — Loom-specific opportunity

BitNet models ship in three forms, all relevant to planet bridging:

1. **Safetensors BF16/F32** — ternary values in full-precision tensors (LoRA-fine-tunable). Loom reads these today.
2. **Safetensors packed ternary** — Microsoft's custom packing, locked to bitnet.cpp.
3. **GGUF with TQ1_0/TQ2_0** — llama.cpp ternary quants (PR #8151).

**Highest-leverage move**: GGUF reader + pure-Go TQ2_0/TQ1_0 dequant makes Loom the only pure-Go engine that can load Microsoft's reference BitNet GGUFs. TQ2_0 ≈ half a day; TQ1_0 ≈ 1–2 days.

---

## What comes close to Loom?

Nothing matches the full Loom combination — but several systems overlap on one axis. Useful for planet bridging: know which "planet" is nearest when importing or comparing formats.

Loom is **not** the only system that saves topology + weights together, and **not** the only one with many numeric/quant representations. What's unusual is the **combination**: deterministic DNVM, volumetric 3D mesh topology, 21-type morphing, bit-packed native JSON, pure-Go/WASM, and edge training. Pick any two axes and you'll find a closer peer; pick all of them and you're mostly describing Loom.

### Closest by dimension

**Native format (topology + weights in one bundle)**

| Peer | Overlap | Gap vs Loom |
|------|---------|-------------|
| **ONNX** | Explicit graph + embedded/external weights; broad dtype support in spec | Interchange standard, not Loom-native; runtimes implement subsets |
| **GGUF** | Architecture in metadata KVs + quantized weights; mmap-friendly | LLM/inference-centric; weak on training and exotic numerics |
| **TFLite** | FlatBuffers graph + weights | Mobile-focused |
| **CoreML `.mlpackage`** | Graph + externalized weights | Apple-runtime-locked |
| **TorchScript `.pt`** | Serialized graph + weights | PyTorch ecosystem |

**Many numeric / quant representations**

| Peer | Overlap | Gap vs Loom |
|------|---------|-------------|
| **GGUF + llama.cpp** | Widest practical quant zoo (Q4_K, Q8_0, TQ1_0/TQ2_0, I-quants) | Inference-only; llama-centric |
| **ONNX** | Many element types on paper (FP8, INT4, etc.) | Execution support varies by runtime |
| **BitNet.cpp / Microsoft BitNet** | Ternary weights | Ternary-focused, not 21 dtypes end-to-end |
| **MLX** | Safetensors + custom quant packing (2–8 bit groups) | Apple-local; not a universal runtime |

Loom's 21 types (FP64 down to binary/ternary, morph per layer, train + save) is a **runtime design choice**, not a claim no other format exists.

**Deterministic, bit-identical cross-platform execution**

Almost nothing mainstream claims what Loom targets: same bits on CPU vs GPU, x86 vs ARM, native vs WASM. **ONNX Runtime** can be made deterministic with care but isn't guaranteed bitwise-identical across backends. **JAX/XLA** can be reproducible with fixed seeds — not the same product guarantee. This is likely Loom's most genuinely unusual axis.

**Embeddable, local, "SQLite of AI"**

| Peer | Overlap | Gap vs Loom |
|------|---------|-------------|
| **llama.cpp + GGUF** | Single binary + one model file; runs fully local | Inference-only LLMs; simpler and more proven for "load and run" |
| **ONNX Runtime** | Embeddable C++ library; runs everywhere | You bring the model; no volumetric mesh or edge training story |
| **TensorFlow Lite** | Mobile/edge embeddable | Google/mobile stack |

**Pure Go / zero-CGO runtime**

Very few peers at serious scale: **GoMLX**, **gorgonia** (partial overlap); **gonnx** / **onnx-go** (ONNX parsing only). Most real inference stacks are C++, Rust, or CUDA (ORT, llama.cpp, PyTorch).

**Volumetric 3D mesh topology (M-POLY-VTD)**

Nothing mainstream comes close. Standard stacks use layer graphs (PyTorch modules, ONNX nodes, GGUF metadata). Neuromorphic or spiking simulators share *spatial/temporal* vibes but not the same model.

### Closest overall (one line)

| If you mean… | Closest thing |
|--------------|---------------|
| Universal file format | **ONNX** |
| LLM-native single file | **GGUF** |
| Local embeddable inference | **llama.cpp** |
| Exotic low-bit weights | **GGUF quants** or **BitNet.cpp** |
| Cross-platform deterministic VM | **Nothing direct** — few optimize for this as a requirement |
| Full stack in Go without CGO | **Nothing direct** at Loom's ambition |

For planet bridging: treat **ONNX**, **GGUF**, and **Safetensors** as the three interchange hubs; treat **Loom JSON** as Loom's native checkpoint format — comparable in *role* to GGUF or ONNX for a single engine, not a global standard (yet).

---

## Open questions / next steps

- [ ] Prototype generic Safetensors + `config.json` loader for additional HF arch families beyond current transformer support
- [ ] Spike GGUF parser (`gpustack/gguf-parser-go`) + TQ2_0 dequant against a BitNet GGUF
- [ ] Spike ONNX protobuf parse + identify ops in a sample HF-exported LLM ONNX
- [ ] Define the universal topology AST struct hierarchy in Loom
- [ ] Roundtrip test suite: load reference models, compare outputs vs native framework runners
- [ ] Security audit: fuzz malformed headers, ban pickle without allowlist, sandbox external converters

---

## Source documents

| File | Focus |
|------|-------|
| [chatgpt.pdf](./chatgpt.pdf) | Exhaustive format inventory (~20 formats), framework R/W matrix, Go library survey, loader flow diagram, security/testing plan |
| [google.pdf](./google.pdf) | Architectural paradigms, format taxonomy table, Loom DNVM context, four-phase universal AST pipeline, mmap/GC strategy, WebGPU angle |
| [claude.pdf](./claude.pdf) | Pure-Go feasibility deep-dive per format, effort estimates, BitNet TQ quant spec, consolidated library state, phased roadmap with engineer-week math |
| [grok.md](./grok.md) | Format comparison table, Loom integration path, ecosystem Mermaid diagram |
