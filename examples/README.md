# Planet Bridging examples

Stream live AI engine weights into Loom `.entity` checkpoints — **no HTTP, no git clone required**.

## Setup (pip — recommended)

```bash
pip install planetbridging[pytorch] welvet

# Optional extras
pip install planetbridging[tensorflow,jax]   # cross-engine compare
```

The wheel includes **loom-stream** and all bedrock data. No `go build` needed.

## Setup (git checkout — developers)

```bash
cd planetbridging
pip install -e ".[pytorch,welvet]"
go build -o bin/loom-stream ./cmd/loom-stream/   # optional; pip wheel bundles this
```

## Run examples

From a git clone:

```bash
python examples/01_hello_stream.py
```

From pip (examples ship inside the installed package):

```bash
EXAMPLES=$(python -c "import pathlib, planetbridging as pb; print(pathlib.Path(pb.__file__).parent / 'examples')")
python "$EXAMPLES/01_hello_stream.py"
```

| Script | What it shows |
|--------|----------------|
| `01_hello_stream.py` | One bedrock, one engine |
| `02_all_layer_types.py` | All 13 Loom layer types (PyTorch) |
| `03_cross_engine.py` | PyTorch / TensorFlow / JAX |
| `04_multi_layer_models.py` | 4-layer MLP, 2-layer CNNs, Mixer v2 |
| `05_welvet_ladder.py` | native → loom-stream → welvet reload |
| `06_showcase_everything.py` | Full API tour |

```bash
python examples/02_all_layer_types.py
python examples/06_showcase_everything.py --quick
```

Entity output and logs go to `./.planetbridging/examples/` (gitignored).

### Run all + save transcripts (git checkout)

```bash
chmod +x examples/run_all_examples.sh
./examples/run_all_examples.sh
./examples/run_all_examples.sh --quick
```

Writes `examples/outputs/*.txt` (gitignored).

## Core API

```python
from planetbridging import engines

result = engines.stream("mha", "pytorch")
print(result.native_vs_loom, result.entity_path)

# Reload with welvet
from welvet import Network
net = Network.deserialize_entity(open(result.entity_path, "rb").read())
```

Compare labels: **EXACT**, **PASS**, **DIFF**.
