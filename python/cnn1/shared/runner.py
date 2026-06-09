"""Common CLI for CNN1 engine runners."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from .fixtures import ensure_fixtures
from .manifest import Manifest, load_manifest, model_output_dim
from .reporter import host_reachable, post_report
from .spec import CNN1_ROOT, DEFAULT_HOST
from .variants import VariantResult


def build_parser(planet: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"planetbridging cnn1 runner ({planet})")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--models-dir", type=Path, default=CNN1_ROOT / "models")
    p.add_argument("--manifest", type=Path, default=CNN1_ROOT / "manifest.yaml")
    p.add_argument("--skip-report", action="store_true")
    p.add_argument("--skip-loom", action="store_true")
    p.add_argument("--model", action="append", dest="models")
    return p


def run_planet(
    planet: str,
    framework_version: str,
    handler: Callable[..., list[VariantResult]],
    argv: list[str] | None = None,
) -> int:
    args = build_parser(planet).parse_args(argv)
    manifest = load_manifest(args.manifest)
    data = ensure_fixtures(manifest)

    if not args.skip_report and not host_reachable(args.host):
        print(f"[{planet}] compare-host not reachable at {args.host}", file=sys.stderr)
        return 2

    selected = set(args.models) if args.models else None
    failures = 0

    for model in manifest.models:
        if selected and model.id not in selected:
            continue
        try:
            variants = handler(
                model=model,
                manifest=manifest,
                data=data,
                models_dir=args.models_dir,
                host=args.host,
                skip_loom=args.skip_loom,
            )
            out_dim = model_output_dim(model)
            in_dim = model.input_channels * model.seq_len
            for v in variants:
                if not args.skip_report:
                    try:
                        post_report(
                            host=args.host,
                            planet=v.planet,
                            stage=v.stage,
                            format=v.format,
                            model_id=model.id,
                            framework_version=framework_version,
                            fixture_version=manifest.fixture_version,
                            input_dim=in_dim,
                            output_dim=out_dim,
                            outputs=v.outputs,
                            artifact_paths=v.artifact_paths,
                            train_skipped=v.train_skipped,
                        )
                    except RuntimeError as exc:
                        failures += 1
                        print(f"[{planet}] {model.id} REPORT FAILED: {exc}", file=sys.stderr)
                        continue
                print(
                    f"[{planet}] {model.id} {v.stage}/{v.format} ok "
                    f"samples={v.outputs.shape[0]} skipped_train={v.train_skipped}"
                )
        except Exception as exc:
            failures += 1
            print(f"[{planet}] {model.id} FAILED: {exc}", file=sys.stderr)

    return 1 if failures else 0
