#!/usr/bin/env python3
"""Resolve the supported official LLaVA checkpoints and their layer counts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


OFFICIAL_MODELS = {
    "llava1.5-7b": "llava-v1.5-7b",
    "llava-v1.5-7b": "llava-v1.5-7b",
    "llava1.5-13b": "llava-v1.5-13b",
    "llava-v1.5-13b": "llava-v1.5-13b",
    "llava-next": "llava-v1.6-vicuna-7b",
    "llava-next-7b": "llava-v1.6-vicuna-7b",
    "llava-v1.6-vicuna-7b": "llava-v1.6-vicuna-7b",
}
DEFAULT_MODELS = ("llava-v1.5-7b", "llava-v1.5-13b", "llava-next")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path
    num_hidden_layers: int


def resolve_models(model_root: Path, models: str) -> list[ModelSpec]:
    requested = [item.strip() for item in models.replace(" ", ",").split(",") if item.strip()]
    if not requested or requested == ["all"]:
        requested = list(DEFAULT_MODELS)

    specs: list[ModelSpec] = []
    seen_paths: set[Path] = set()
    for requested_name in requested:
        directory_name = OFFICIAL_MODELS.get(requested_name)
        if directory_name is None:
            choices = ", ".join(sorted(set(OFFICIAL_MODELS) | {"all"}))
            raise ValueError(f"Unsupported model {requested_name!r}; choose from {choices}")
        model_path = (model_root / directory_name).resolve()
        config_path = model_path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Official checkpoint is incomplete: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        architectures = config.get("architectures") or []
        if config.get("model_type") != "llava" or not architectures or architectures[0] != "LlavaLlamaForCausalLM":
            raise ValueError(f"Not an official LLaVA checkpoint: {config_path}")
        num_layers = config.get("num_hidden_layers")
        if num_layers is None and isinstance(config.get("text_config"), dict):
            num_layers = config["text_config"].get("num_hidden_layers")
        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError(f"Missing positive num_hidden_layers in {config_path}")
        if model_path in seen_paths:
            continue
        seen_paths.add(model_path)
        specs.append(ModelSpec(directory_name, model_path, num_layers))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--models", default="all")
    args = parser.parse_args()
    for spec in resolve_models(args.model_root, args.models):
        print(f"{spec.name}\t{spec.path}\t{spec.num_hidden_layers}")


if __name__ == "__main__":
    main()
