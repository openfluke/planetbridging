# Planet Bridging examples

Stream live AI engine weights into Loom `.entity` checkpoints — **no HTTP server required**.

## Setup (once)

```bash
cd planetbridging

# 1. Build the Go bridge (reads JSON on stdin, writes .entity + outputs)
go build -o bin/loom-stream ./cmd/loom-stream/

# 2. Install the Python package (pick your engine)
pip install -e ".[pytorch]"          # PyTorch only
pip install -e ".[pytorch,tensorflow,jax]"   # cross-engine compare
pip install -e ".[all]"              # everything + pytest

# 3. Optional: welvet reload (monorepo checkout)
pip install -e ../welvet/python
```

Set `PLANETBRIDGING_LOOM_STREAM=/path/to/loom-stream` if the binary is not in `bin/`.

## Run the examples

| Script | What it shows |
|--------|----------------|
| `01_hello_stream.py` | One bedrock, one engine — smallest possible demo |
| `02_all_layer_types.py` | All 13 Loom layer types from PyTorch |
| `03_cross_engine.py` | Same layer on PyTorch / TensorFlow / JAX |
| `04_multi_layer_models.py` | 4-layer MLP, 2-layer CNNs, 16-layer Mixer v2 |
| `05_welvet_ladder.py` | native → loom-stream → welvet reload (where supported) |
| `06_showcase_everything.py` | **Full tour** — every API (smoke, ladder, engines, absorb, welvet) |

```bash
python examples/01_hello_stream.py
python examples/02_all_layer_types.py
python examples/03_cross_engine.py layernorm
python examples/04_multi_layer_models.py
python examples/05_welvet_ladder.py cnn1
python examples/06_showcase_everything.py
python examples/06_showcase_everything.py --quick
```

### Run all examples + save transcripts

```bash
chmod +x examples/run_all_examples.sh
./examples/run_all_examples.sh           # full showcase
./examples/run_all_examples.sh --quick   # faster 06_showcase

# Writes gitignored text logs under examples/outputs/
#   run_all.txt              — combined transcript
#   01_hello_stream.txt      — per-example output
#   …
```

## Core API

```python
from planetbridging import engines

# Stream a live PyTorch model → .entity, compare native vs Loom
result = engines.stream("mha", "pytorch")
print(result.native_vs_loom)   # PASS / EXACT
print(result.entity_path)      # path to .stream.entity

# All 13 bedrocks from one engine
results = engines.stream_all_bedrocks("pytorch")

# One bedrock on every installed engine
results = engines.stream_all_planets("cnn2")

# Optional welvet reload (needs welvet installed)
result = engines.stream("layernorm", "pytorch", try_welvet=True)
```

Compare labels: **EXACT** (bit-identical), **PASS** (fp32 tolerance), **DIFF** (investigate).
