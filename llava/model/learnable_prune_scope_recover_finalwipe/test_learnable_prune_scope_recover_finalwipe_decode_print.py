#!/usr/bin/env python3
# coding: utf-8
"""Standalone smoke/demo for learnable-prune SCOPE recover final-wipe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.model.learnable_prune_scope_recover_finalwipe.modeling_llava_learnable_prune_scope_recover_finalwipe import DEFAULT_CHECKPOINT


BASE_MODEL = "/data1/chenzixuan/model/liuhaotian/llava-v1.5-7b"
IMAGE_PATH = "/data1/chenzixuan/MLLM_Token_Compression_Workdir/data/docvqa.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small learnable-prune SCOPE recover final-wipe smoke test.")
    parser.add_argument("--model_name_or_path", type=str, default=BASE_MODEL)
    parser.add_argument("--learnable_prune_checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--image_path", type=str, default=IMAGE_PATH)
    parser.add_argument("--prompt", type=str, default="How many nomination committee meetings has Y. C. Deveshwar attended?")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--conv_mode", type=str, default="llava_v1")
    parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    return parser.parse_args()


def dtype_from_arg(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def main() -> None:
    args = parse_args()
    model_name = get_model_name_from_path(args.model_name_or_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_name_or_path,
        None,
        model_name,
        device_map=args.device_map,
        device=args.device,
        learnable_prune_scope_recover_finalwipe_model=True,
        dtype=str(dtype_from_arg(args.dtype)).rsplit(".", 1)[-1],
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    model.load_learnable_prune_checkpoint(args.learnable_prune_checkpoint)

    image = Image.open(args.image_path).convert("RGB") if args.image_path else Image.new("RGB", (336, 336), "black")
    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [tensor.to(device=model.device, dtype=getattr(model, "dtype", dtype_from_arg(args.dtype))) for tensor in image_tensor]
    else:
        image_tensor = image_tensor.to(device=model.device, dtype=getattr(model, "dtype", dtype_from_arg(args.dtype)))
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{args.prompt}")
    conv.append_message(conv.roles[1], None)
    input_ids = tokenizer_image_token(
        conv.get_prompt(),
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)

    model.reset_learnable_prune_stats()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[:, input_ids.shape[-1] :] if output_ids.shape[-1] > input_ids.shape[-1] else output_ids
    print("prompt:", args.prompt)
    print("input_ids shape:", tuple(input_ids.shape))
    print("generated_ids shape:", tuple(generated.shape))
    print("learnable_prune_stats:", model.get_learnable_prune_stats())
    print("decoded:", tokenizer.batch_decode(generated, skip_special_tokens=True)[0])


if __name__ == "__main__":
    main()
