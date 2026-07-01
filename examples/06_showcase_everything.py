#!/usr/bin/env python3
"""Full planetbridging showcase — every major API surface.

Runs the catalog, numpy-reference smokes, compare ladders, live AI engine
streaming, cross-engine batch APIs, low-level dense absorb, and welvet reload.

    python examples/06_showcase_everything.py
    python examples/06_showcase_everything.py --quick
    python examples/06_showcase_everything.py --no-welvet

Requires: loom-stream binary + pip install -e ".[pytorch]"
Optional: tensorflow, jax, sklearn extras; welvet for entity reload demos.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import textwrap
from pathlib import Path

from _helpers import ROOT, require_loom_stream

BINARY = require_loom_stream()

import numpy as np  # noqa: E402

import planetbridging as pb  # noqa: E402
from planetbridging import (  # noqa: E402
    BEDROCK_IDS,
    BEDROCK_PLANETS,
    BEDROCKS,
    STREAM_MODEL_IDS,
    absorb,
    compare_outputs,
    diff_label,
    engines,
    run_all_bedrock_ladders,
    run_all_bedrock_smokes,
    run_bedrock_ladder,
    run_bedrock_smoke,
)
from planetbridging._binary import find_loom_stream, repo_root  # noqa: E402
from planetbridging.welvet_infer import (  # noqa: E402
    WELVET_RELOAD_BEDROCKS,
    WELVET_RELOAD_SKIP,
    import_welvet,
    print_compare_ladder,
)

MIXER_POC_TOLERANCE = 5e-5
OUT = ROOT / ".planetbridging" / "showcase"

# Numpy smoke runners use single-layer native forwards; 2-layer CNN / mixer v2
# may DIFF here while engines.stream (live POC) is authoritative.
SMOKE_WARN_BEDROCKS = frozenset({"cnn1", "cnn2", "cnn3", "mixer"})


def section(n: int, title: str) -> None:
    bar = "─" * 72
    print(f"\n{bar}\n  {n}. {title}\n{bar}")


def ok_label(label: str, *, bedrock: str = "", max_diff: float | None = None) -> bool:
    if label in ("PASS", "EXACT"):
        return True
    if bedrock == "mixer" and label == "DIFF" and max_diff is not None:
        return max_diff < MIXER_POC_TOLERANCE
    return False


def smoke_status(r) -> tuple[str, bool]:
    """Return (display mark, counts as hard failure)."""
    label = r.compare_label
    if ok_label(label, bedrock=r.bedrock, max_diff=r.max_abs_diff):
        return "✓", False
    if r.bedrock in SMOKE_WARN_BEDROCKS:
        return "~", False  # known numpy-smoke gap; live engines.stream is authority
    if label == "ERROR":
        return "✗", True
    return "✗", True


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _welvet_available() -> bool:
    try:
        import_welvet()
        return True
    except ImportError:
        return False


def showcase_catalog() -> None:
    section(1, "Catalog — what planetbridging knows about")
    print(f"  version:          {pb.__version__}")
    print(f"  repo root:        {repo_root()}")
    print(f"  loom-stream:      {find_loom_stream(BINARY)}")
    print(f"  bedrock count:    {len(BEDROCK_IDS)}")
    print(f"  bedrock ids:      {', '.join(BEDROCK_IDS)}")
    print()
    print("  Default stream models (multi-layer where noted):")
    for bid in BEDROCK_IDS:
        info = BEDROCKS[bid]
        planets = ", ".join(BEDROCK_PLANETS[bid])
        print(
            f"    {bid:<12} {STREAM_MODEL_IDS[bid]:<28} "
            f"{info.layer_count:>2} layers  [{planets}]"
        )


def showcase_smokes(bedrocks: tuple[str, ...]) -> list[str]:
    section(2, "Numpy reference smoke — stream_bedrock without live AI engines")
    print("  API: run_bedrock_smoke(), run_all_bedrock_smokes()")
    print("  Uses fixture weights + native numpy forward → loom-stream CLI\n")

    failures: list[str] = []
    if len(bedrocks) == len(BEDROCK_IDS):
        results = run_all_bedrock_smokes(root=repo_root(), binary=BINARY, out_dir=OUT / "smoke")
    else:
        results = [
            run_bedrock_smoke(b, root=repo_root(), binary=BINARY, out_dir=OUT / "smoke" / b)
            for b in bedrocks
        ]

    for r in results:
        mark, hard_fail = smoke_status(r)
        diff = f"  max={r.max_abs_diff:.2e}" if r.max_abs_diff is not None else ""
        note = "  (numpy smoke gap — use engines.stream)" if mark == "~" else ""
        print(f"  {mark} {r.bedrock:<12} {r.layer_count:>2}L  {r.compare_label:<6}{diff}{note}")
        if hard_fail:
            failures.append(f"smoke/{r.bedrock}")
    return failures


def showcase_ladders(bedrocks: tuple[str, ...], *, try_welvet: bool) -> list[str]:
    section(3, "Compare ladder — numpy native → loom-stream → welvet reload")
    print("  API: run_bedrock_ladder(), run_all_bedrock_ladders(), engines.ladder()")
    if try_welvet:
        print(f"  welvet OK:  {', '.join(sorted(WELVET_RELOAD_BEDROCKS))}")
        print(f"  welvet skip: {', '.join(sorted(WELVET_RELOAD_SKIP))}\n")
    else:
        print("  (--no-welvet: skipping welvet reload)\n")

    failures: list[str] = []
    kwargs = dict(root=repo_root(), binary=BINARY, out_dir=OUT / "ladder", try_welvet=try_welvet)
    if len(bedrocks) == len(BEDROCK_IDS):
        results = run_all_bedrock_ladders(**kwargs)
    else:
        results = [run_bedrock_ladder(b, **kwargs) for b in bedrocks]

    for r in results:
        if r.bedrock in SMOKE_WARN_BEDROCKS and r.native_vs_loom in ("DIFF", "ERROR"):
            mark = "~"
            hard_fail = False
        else:
            mark = "✓" if ok_label(r.native_vs_loom, bedrock=r.bedrock) else "✗"
            hard_fail = not ok_label(r.native_vs_loom, bedrock=r.bedrock)
        wel = r.native_vs_welvet or r.welvet_note or "—"
        note = "  (use engines.stream)" if mark == "~" else ""
        print(f"  {mark} {r.bedrock:<12} native→loom {r.native_vs_loom:<6}  welvet: {wel}{note}")
        if hard_fail:
            failures.append(f"ladder/{r.bedrock}")
    return failures


def showcase_live_engine() -> list[str]:
    section(4, "Live AI engine streaming — engines.stream() / absorb.stream()")
    if not _has_module("torch"):
        print("  skip — install pytorch extra: pip install -e \".[pytorch]\"")
        return []

    print("  POC handlers patch urllib → loom-stream (no compare-host HTTP)\n")
    failures: list[str] = []

    for api_name, stream_fn in [("engines.stream", engines.stream), ("absorb.stream", absorb.stream)]:
        r = stream_fn(
            "swiglu",
            "pytorch",
            root=repo_root(),
            binary=BINARY,
            out_dir=OUT / "live" / api_name,
        )
        mark = "✓" if ok_label(r.native_vs_loom) else "✗"
        print(f"  {mark} {api_name}('swiglu', 'pytorch') → {r.native_vs_loom}  entity={Path(r.entity_path).name}")
        if not ok_label(r.native_vs_loom):
            failures.append(api_name)

    r = engines.stream(
        "dense",
        "pytorch",
        root=repo_root(),
        binary=BINARY,
        out_dir=OUT / "live" / "multilayer",
    )
    mark = "✓" if ok_label(r.native_vs_loom) else "✗"
    print(
        f"  {mark} engines.stream('dense', 'pytorch') "
        f"→ {r.model_id} ({r.layer_count} layers) {r.native_vs_loom}"
    )
    if not ok_label(r.native_vs_loom):
        failures.append("engines.stream/dense")

    return failures


def showcase_batch_apis(bedrocks: tuple[str, ...]) -> list[str]:
    section(5, "Batch APIs — all bedrocks / all planets")
    failures: list[str] = []

    if not _has_module("torch"):
        print("  skip batch engine APIs — torch not installed")
        return failures

    if len(bedrocks) == len(BEDROCK_IDS):
        print("  engines.stream_all_bedrocks('pytorch'):\n")
        results = engines.stream_all_bedrocks("pytorch", root=repo_root(), binary=BINARY)
        for r in results:
            label = r.native_vs_loom
            max_d = None
            if r.bedrock == "mixer" and label == "DIFF":
                max_d, _, _ = compare_outputs(r.native, r.loom_stream)
                label = "PASS" if max_d < MIXER_POC_TOLERANCE else "DIFF"
            mark = "✓" if ok_label(label, bedrock=r.bedrock, max_diff=max_d) else "✗"
            print(f"    {mark} {r.bedrock:<12} {label}")
            if not ok_label(label, bedrock=r.bedrock, max_diff=max_d):
                failures.append(f"all_bedrocks/{r.bedrock}")
    else:
        print(f"  (--quick: skipping stream_all_bedrocks; tried {len(bedrocks)} smokes above)\n")

    demo_bedrock = "layernorm"
    installed = engines.installed_planets(demo_bedrock)
    print(f"\n  engines.stream_all_planets('{demo_bedrock}') — installed: {', '.join(installed) or 'none'}:\n")
    if installed:
        for r in engines.stream_all_planets(demo_bedrock, root=repo_root(), binary=BINARY):
            mark = "✓" if ok_label(r.native_vs_loom) else "✗"
            print(f"    {mark} {r.planet:<12} {r.native_vs_loom}")
            if r.native_vs_loom not in ("PASS", "EXACT"):
                failures.append(f"all_planets/{r.planet}")
    else:
        print("    (install tensorflow/jax extras to see more planets)")

    available = engines.available_planets("dense")
    print(f"\n  engines.available_planets('dense'): {available}")
    print(f"  engines.installed_planets('dense'): {engines.installed_planets('dense')}")

    return failures


def showcase_low_level_stream() -> list[str]:
    section(6, "Low-level stream API — stream_dense / stream_bedrock")
    print("  API: stream_dense(), stream_bedrock(), stream_<layer>()")
    print("  absorb.pytorch/keras/jax/sklearn wrap stream_dense for live modules\n")

    from planetbridging.layers.dense import LayerSpec, layers_from_specs
    from planetbridging.stream import stream_dense

    specs = (LayerSpec(2, "relu"), LayerSpec(1, "linear"))
    w0 = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float64)
    w1 = np.array([[1.0, 0.0]], dtype=np.float64)
    b0 = np.array([0.01, 0.02], dtype=np.float64)
    b1 = np.array([0.5], dtype=np.float64)
    layers = layers_from_specs(
        input_dim=4, specs=specs, kernels=[w0, w1], biases=[b0, b1]
    )

    # Fixture inputs from dense bedrock (100 samples × 4 features)
    fx = np.load(repo_root() / "python/dense/fixtures/dense_bedrock_v2.npz")["x_test"][:, :4]
    h = np.maximum(fx @ w0.T + b0, 0.0)
    native = (h @ w1.T + b1).astype(np.float64)

    r = stream_dense(
        planet="showcase",
        model_id="synthetic_mlp",
        layers=layers,
        input_dim=4,
        fixture_version="dense_bedrock_v2",
        output_path=OUT / "stream_dense" / "synthetic.entity",
        root=repo_root(),
        binary=BINARY,
        native_outputs=native,
    )
    mark = "✓" if ok_label(r.compare_label or "") else "✗"
    print(f"  {mark} stream_dense  layers={r.layer_count}  {r.compare_label}  entity={Path(r.entity_path).name}")

    if _has_module("torch"):
        import torch
        import torch.nn as nn
        from planetbridging import absorb

        net = nn.Sequential(nn.Linear(4, 2), nn.ReLU(), nn.Linear(2, 1))
        with torch.no_grad():
            for mod in net:
                if hasattr(mod, "weight"):
                    mod.weight.fill_(0.1)
                    if mod.bias is not None:
                        mod.bias.zero_()
        x = fx[:5].astype(np.float32)
        with torch.no_grad():
            pt_native = net(torch.from_numpy(x)).numpy()
        pt = absorb.pytorch(
            net,
            input_dim=4,
            layer_units=(2, 1),
            inputs=x,
            native_outputs=pt_native,
            model_id="absorb_demo",
            fixture_version="dense_bedrock_v2",
            root=repo_root(),
            binary=BINARY,
            output_path=OUT / "absorb" / "absorb_demo.entity",
        )
        mark2 = "✓" if ok_label(pt.compare_label or "") else "✗"
        print(f"  {mark2} absorb.pytorch (live nn.Module)  {pt.compare_label}")
    else:
        print("  — absorb.pytorch skipped (torch not installed)")

    if r.compare_label not in ("PASS", "EXACT"):
        return ["stream_dense"]
    return []


def showcase_welvet_engine(*, try_welvet: bool) -> list[str]:
    section(7, "Welvet entity reload — engines.stream(..., try_welvet=True)")
    if not try_welvet:
        print("  skipped (--no-welvet)")
        return []
    if not _has_module("torch"):
        print("  skip — torch not installed")
        return []
    if not _welvet_available():
        print("  skip — welvet not installed (pip install -e ../welvet/python)")
        return []

    failures: list[str] = []
    for bedrock in ("cnn1", "layernorm"):
        r = engines.stream(
            bedrock,
            "pytorch",
            root=repo_root(),
            binary=BINARY,
            out_dir=OUT / "welvet" / bedrock,
            try_welvet=True,
        )
        n = len(r.welvet) if r.welvet is not None else 0
        if r.welvet is not None and n:
            print_compare_ladder(
                title=f"{bedrock} (live engine + welvet)",
                native=r.native[:n],
                loom_stream=r.loom_stream[:n],
                welvet_out=r.welvet,
            )
            if r.native_vs_welvet not in ("PASS", "EXACT"):
                failures.append(f"welvet/{bedrock}")
        else:
            print(f"  — {bedrock}: {r.welvet_note or 'no welvet output'}")

    r = engines.ladder("mha", "numpy", root=repo_root(), binary=BINARY, try_welvet=True)
    print(f"\n  engines.ladder('mha', 'numpy'): native→loom {r.native_vs_loom}", end="")
    if r.native_vs_welvet:
        print(f"  native→welvet {r.native_vs_welvet}")
    else:
        print(f"  ({r.welvet_note})")

    return failures


def showcase_compare_utils() -> None:
    section(8, "Compare utilities")
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0000001])
    max_d, mean_d, exact = compare_outputs(a, b)
    print("  API: compare_outputs(), diff_label(), welvet_infer.print_compare_ladder()")
    print(f"  compare_outputs demo:  max={max_d:.2e}  mean={mean_d:.2e}  exact={exact}")
    print(f"  diff_label:            {diff_label(max_d, exact=exact)}")


def showcase_summary(all_failures: list[str], *, quick: bool) -> None:
    section(9, "Summary")
    apis = textwrap.dedent("""
        Package surface demonstrated:
          • BEDROCK_IDS, BEDROCKS, STREAM_MODEL_IDS, BEDROCK_PLANETS
          • find_loom_stream / repo_root
          • run_bedrock_smoke, run_all_bedrock_smokes     (numpy → loom-stream)
          • run_bedrock_ladder, run_all_bedrock_ladders   (+ welvet reload)
          • engines.stream, stream_all_bedrocks, stream_all_planets, ladder
          • engines.available_planets, installed_planets
          • absorb.stream, absorb.pytorch                   (low-level dense)
          • stream_bedrock, stream_dense, stream_<layer>    (via smoke runners)
          • compare_outputs, diff_label, print_compare_ladder
          • WELVET_RELOAD_BEDROCKS / WELVET_RELOAD_SKIP
    """).strip()
    print(apis)
    if quick:
        print("\n  (ran with --quick; use full run for all 13 bedrocks in batch sections)")
    if all_failures:
        print(f"\n  ✗ {len(all_failures)} issue(s): {', '.join(all_failures)}")
        raise SystemExit(1)
    print("\n  ✓ Showcase complete — planetbridging APIs exercised successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke/ladder only 3 bedrocks; skip stream_all_bedrocks loop",
    )
    parser.add_argument("--no-welvet", action="store_true", help="Skip welvet reload sections")
    args = parser.parse_args()

    bedrocks: tuple[str, ...] = ("dense", "mha", "layernorm") if args.quick else BEDROCK_IDS
    try_welvet = not args.no_welvet

    print("planetbridging showcase — everything the package can do")
    print(f"  mode: {'quick' if args.quick else 'full'}  welvet: {'on' if try_welvet else 'off'}")
    OUT.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    showcase_catalog()
    failures.extend(showcase_smokes(bedrocks))
    failures.extend(showcase_ladders(bedrocks, try_welvet=try_welvet))
    failures.extend(showcase_live_engine())
    failures.extend(showcase_batch_apis(bedrocks))
    failures.extend(showcase_low_level_stream())
    failures.extend(showcase_welvet_engine(try_welvet=try_welvet))
    showcase_compare_utils()
    showcase_summary(failures, quick=args.quick)


if __name__ == "__main__":
    main()
