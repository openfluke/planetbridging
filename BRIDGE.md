# Planet Bridging — architecture diagram

**End goal:** **bidirectional** — planets ↔ hub formats ↔ **Loom** (import and export).

**Step 1 (now):** **one direction only** → planets train in Python, checkpoints hit the bridge, **everything lands in Loom**. Export back to planets is dashed / not built.

---

## Main diagram (read left → center → right)

| Column | What you’re looking at |
|--------|-------------------------|
| **Left** | Python AI engines as **stacked lego** — shared base overlaps every island; each engine is still its **own conda env** |
| **Center** | **Numerical types** listed top-to-bottom — what planets actually use, then all **21 Loom DTypes** |
| **Right** | **Loom** — one Go hub; solid arrows = step 1 today · dashed = bidirectional later |

```mermaid
flowchart LR

  %% ── LEFT: lego planets (overlapping shared base) ──────────────────────────
  subgraph LEFT["Python AI engines · dense bedrock"]
    direction TB

    subgraph ISLANDS["engine blocks — separate conda env each"]
      direction TB
      PT["🧱 PyTorch<br/>.pt · .safetensors · .onnx"]
      TF["🧱 TensorFlow<br/>.keras · saved_model/"]
      JAX["🧱 JAX / Flax<br/>params.msgpack"]
      SK["🧱 sklearn<br/>model.pkl"]
      PD["🧱 Paddle<br/>.pdparams"]
    end

    subgraph BASE["shared lego base — overlaps all islands"]
      direction TB
      B_PY["Python 3.11"]
      B_CO["conda-forge"]
      B_NP["numpy · pyyaml"]
      B_FIX["fixture dense_bedrock_v2<br/>12 models · 5000 train · 100 test"]
    end

    PT --> B_PY
    TF --> B_PY
    JAX --> B_PY
    SK --> B_PY
    PD --> B_PY
    PT & TF & JAX & SK & PD --> B_CO
    PT & TF & JAX & SK & PD --> B_NP
    PT & TF & JAX & SK & PD --> B_FIX
  end

  %% ── CENTER: numerical types (long column down) ───────────────────────────
  subgraph MID["Numerical types"]
    direction TB

    MID_H["── planets use ONE type each ──"]
    P32["FP32 ⭐ bedrock default<br/>PyTorch · TF · JAX · Paddle"]
    P64["FP64 ⭐ sklearn only"]

    MID_L["── Loom: 21 DTypes per layer ──"]
    D01["1 · Float64"]
    D02["2 · Float32"]
    D03["3 · Float16"]
    D04["4 · BFloat16"]
    D05["5 · FP8 E4M3"]
    D06["6 · FP8 E5M2"]
    D07["7 · Int64"]
    D08["8 · Uint64"]
    D09["9 · Int32"]
    D10["10 · Uint32"]
    D11["11 · Int16"]
    D12["12 · Uint16"]
    D13["13 · Int8"]
    D14["14 · Uint8"]
    D15["15 · Int4"]
    D16["16 · Uint4"]
    D17["17 · FP4"]
    D18["18 · Int2"]
    D19["19 · Uint2"]
    D20["20 · Ternary"]
    D21["21 · Binary"]

    MID_H --> P32 --> P64 --> MID_L
    MID_L --> D01 --> D02 --> D03 --> D04 --> D05 --> D06
    D06 --> D07 --> D08 --> D09 --> D10 --> D11 --> D12
    D12 --> D13 --> D14 --> D15 --> D16 --> D17 --> D18
    D18 --> D19 --> D20 --> D21
  end

  %% ── RIGHT: Loom hub ───────────────────────────────────────────────────────
  subgraph RIGHT["Loom · pure Go"]
    direction TB
    LOOM["🏛️ LOOM<br/>VolumetricNetwork<br/>Dense · CNN · MHA · LSTM · RNN"]
    HOST["Compare host ✅<br/>go run . :9876"]
    BRG["bridge/ import ⬜<br/>.safetensors · .onnx · .keras …"]
    JSON["model.json ✅"]
    OUT["export → planets ⬜<br/>endgoal"]

    LOOM --> HOST
    LOOM --> BRG
    LOOM --> JSON
    LOOM --> OUT
  end

  %% planets → their dtype
  PT --> P32
  TF --> P32
  JAX --> P32
  PD --> P32
  SK --> P64

  %% bridge boundary: FP32 weights in → Loom picks any dtype
  P32 --> BRG
  P64 --> BRG

  %% all Loom numerics → hub
  D01 & D02 & D03 & D04 & D05 & D06 & D07 & D08 & D09 & D10 --> LOOM
  D11 & D12 & D13 & D14 & D15 & D16 & D17 & D18 & D19 & D20 & D21 --> LOOM

  %% checkpoints → bridge
  PT & TF --> BRG
  JAX & SK & PD -.->|"export hub first ⬜"| BRG

  %% step 1 reports
  PT & TF & JAX & SK & PD -->|"100-sample reports"| HOST
  BRG -->|"stage=loom ⬜"| HOST

  %% bidirectional endgoal
  OUT -.-> PT & TF & JAX & SK & PD

  classDef lego fill:#334155,stroke:#94a3b8,color:#f1f5f9
  classDef base fill:#475569,stroke:#cbd5e1,color:#f8fafc
  classDef planetdtype fill:#0c4a6e,stroke:#38bdf8,color:#e0f2fe
  classDef loomdtype fill:#1e3a5f,stroke:#7dd3fc,color:#e0f2fe
  classDef loom fill:#fffbeb,stroke:#b45309,color:#78350f,stroke-width:3px
  classDef todo stroke-dasharray:6 6

  class PT,TF,JAX,SK,PD lego
  class B_PY,B_CO,B_NP,B_FIX base
  class P32,P64 planetdtype
  class D01,D02,D03,D04,D05,D06,D07,D08,D09,D10,D11,D12,D13,D14,D15,D16,D17,D18,D19,D20,D21 loomdtype
  class LOOM,HOST,JSON loom
  class BRG,OUT todo
```

**How to read it**

- **Left:** engines look like separate bricks, but they all sit on the same overlapping base (Python, conda, numpy, shared fixture). You still cannot merge them into one venv.
- **Center:** two **planet** dtypes at the top (FP32 / FP64); below that, the full **Loom** ladder (21 types) — planets do not train those; Loom does after import.
- **Right:** every numeric path and every engine path ends at **Loom**. Dashed lines = not built yet (export back, JAX/sklearn/Paddle import without hub export).

---

## Step 1 vs endgoal (simple)

```mermaid
flowchart LR
  A["Python planets ✅"] --> B["checkpoint files ✅"]
  B --> C["Go bridge ⬜"]
  C --> D["LOOM ✅/⬜"]
  D -.->|"bidirectional later"| A

  style C stroke-dasharray:6 6
  style D fill:#fffbeb,stroke:#b45309
```

| | Step 1 (now) | Endgoal |
|--|--------------|---------|
| Direction | Planets → Loom | Loom ↔ planets |
| Python | 5 conda envs | Still needed for training / export |
| Compare | native / export ✅ · loom ⬜ | full round-trip |
| Numerics | FP32 (FP64 sklearn) at boundary | Loom runs any of 21 DTypes |

---

## One planet pipeline (not cross-planet)

```mermaid
flowchart LR
  N["native"] --> E["export"]
  E --> L["loom ⬜"]
  N --> L

  style L stroke-dasharray:6 6
```

We never diff PyTorch vs TensorFlow — only **native → export → loom** within one planet.

---

## Engine → file → Loom (cheat sheet)

| Engine | Conda | Planet dtype | Example file | → Loom step 1 |
|--------|-------|--------------|--------------|---------------|
| PyTorch | `pb-dense-pytorch` | FP32 | `model.safetensors` | ✅ first import target |
| TensorFlow | `pb-dense-tensorflow` | FP32 | `model.keras` | 🟡 |
| JAX | `pb-dense-jax` | FP32 | `params.msgpack` | ⬜ export first |
| sklearn | `pb-dense-sklearn` | FP64 | `model.pkl` | ⬜ export first |
| Paddle | `pb-dense-paddle` | FP32 | `model.pdparams` | ⬜ export first |

---

## Related docs

- [`python/dense/README.md`](./python/dense/README.md) — conda hell, stdlib parsers, dtype notes
- [`README.md`](./README.md) — full format matrix & roadmap
- [`rnd/README.md`](./rnd/README.md) — future planets (GGUF, ONNX Runtime, …)

Preview: [mermaid.live](https://mermaid.live) · paste the diagram block if your renderer misbehaves.
