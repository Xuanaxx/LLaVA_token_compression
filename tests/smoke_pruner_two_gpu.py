"""Two-GPU CUDA/DDP smoke test for the learnable-pruner training hot path."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import MethodType

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from learnable_pruner_lightweight.predictor import LearnablePrunePredictor
from learnable_pruner_lightweight.train_learnable_pruner import LearnablePruneWrapper
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
from llava.model.learnable_prune_lightweight_scope_finalwipe import official_modeling


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    torch.manual_seed(1234)

    config = LlavaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        pad_token_id=0,
    )
    config._attn_implementation = "sdpa"
    config.tokenizer_padding_side = "right"
    llava = LlavaLlamaForCausalLM(config)
    predictor = LearnablePrunePredictor(
        input_size=64,
        hidden_size=32,
        rank=16,
        num_heads=2,
    )
    wrapper = LearnablePruneWrapper(
        llava,
        predictor,
        teacher_layer=2,
        budget_profiles=[
            {
                "avg_token_budget": 20,
                "keep_k": 16,
                "scope_target_count": 48,
                "mid_pruning_layer_idx": 1,
                "mid_target_count": 16,
                "final_wipe_layer_idx": 3,
                "enable_final_wipe": True,
            }
        ],
        budgeted_soft_topk_iters=8,
    ).to(device=device, dtype=torch.bfloat16)
    wrapper.gradient_checkpointing_enable()

    batch_size, seq_len = 2, 88
    expanded_ids = torch.arange(1, seq_len + 1, device=device)[None, :].repeat(batch_size, 1)
    expanded_ids[:, 4:68] = IMAGE_TOKEN_INDEX
    expanded_attention = torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)
    expanded_positions = torch.arange(seq_len, device=device)[None, :].repeat(batch_size, 1)
    expanded_labels = torch.full_like(expanded_ids, IGNORE_INDEX)
    expanded_labels[:, 76:] = torch.arange(20, 32, device=device)[None, :]
    expanded_embeds = torch.randn(
        batch_size, seq_len, config.hidden_size, device=device, dtype=torch.bfloat16
    )
    expanded_coordinates = torch.zeros(
        batch_size, seq_len, 2, device=device, dtype=torch.bfloat16
    )
    grid = torch.linspace(0, 1, 8, device=device, dtype=torch.bfloat16)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    expanded_coordinates[:, 4:68] = torch.stack((xx, yy), dim=-1).reshape(1, 64, 2)

    def fake_multimodal_inputs(_self, **_kwargs):
        return (
            expanded_embeds,
            expanded_ids,
            expanded_attention,
            expanded_positions,
            expanded_labels,
            expanded_coordinates,
        )

    wrapper._build_multimodal_inputs = MethodType(fake_multimodal_inputs, wrapper)
    ddp = DistributedDataParallel(
        wrapper,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )
    optimizer = torch.optim.AdamW(wrapper.predictor.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = ddp(
            input_ids=torch.ones((batch_size, 1), device=device, dtype=torch.long),
            attention_mask=torch.ones((batch_size, 1), device=device, dtype=torch.bool),
            labels=torch.full((batch_size, 1), IGNORE_INDEX, device=device, dtype=torch.long),
            images=torch.zeros((batch_size, 3, 2, 2), device=device, dtype=torch.bfloat16),
        )
    outputs["loss"].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(wrapper.predictor.parameters(), 1.0)
    if not torch.isfinite(outputs["loss"]) or not torch.isfinite(grad_norm):
        raise RuntimeError(f"non-finite loss/grad: loss={outputs['loss']}, grad_norm={grad_norm}")
    optimizer.step()
    dist.all_reduce(grad_norm, op=dist.ReduceOp.SUM)

    captured_graphs = sum(
        isinstance(value, official_modeling._SeededScopeCudaGraph)
        for value in official_modeling._SCOPE_GRAPH_CACHE.values()
    )
    if captured_graphs == 0:
        failures = [repr(value) for value in official_modeling._SCOPE_GRAPH_CACHE.values()]
        raise RuntimeError(f"SCOPE CUDA graph was not captured: {failures}")

    if local_rank == 0:
        print(
            "two_gpu_smoke_ok",
            f"loss={outputs['loss'].detach().float().item():.6f}",
            f"summed_grad_norm={grad_norm.float().item():.6f}",
            f"scope_graphs={captured_graphs}",
            f"max_memory_mib={torch.cuda.max_memory_allocated(device) / 2**20:.1f}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
