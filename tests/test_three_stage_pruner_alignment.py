import os
import sys
import tempfile
from pathlib import Path
from types import MethodType
from unittest.mock import patch

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from learnable_pruner_lightweight.predictor import LearnablePrunePredictor
from learnable_pruner_lightweight.train_learnable_pruner import (
    LearnablePruneTrainer,
    LearnablePruneWrapper,
    _seeded_residual_scope_indices,
    parse_budget_profiles,
    validate_distributed_cuda_mapping,
)
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
from llava.model.learnable_prune_lightweight_scope_finalwipe.official_modeling import (
    LlavaLearnablePruneLightweightScopeFinalwipeForCausalLM,
    SeededResidualSCOPE,
)


def test_training_scope_matches_inference_scope():
    torch.manual_seed(7)
    visual = torch.randn(1, 23, 12)
    seeds = torch.tensor([4, 11, 2, 19])

    training_indices = _seeded_residual_scope_indices(visual[0], seeds, target_keep=13)
    inference_indices, _ = SeededResidualSCOPE(visual, seeds, target_keep=13)

    assert torch.equal(training_indices, inference_indices[0])


def test_training_scope_enables_thread_local_cuda_graph_path():
    mocked_result = (torch.tensor([[1, 3]]), torch.empty(0))
    with patch(
        "learnable_pruner_lightweight.train_learnable_pruner._inference_seeded_residual_scope",
        return_value=mocked_result,
    ) as scope_mock:
        selected = _seeded_residual_scope_indices(
            torch.randn(5, 4), torch.tensor([1]), target_keep=2
        )

    assert selected.tolist() == [1, 3]
    assert scope_mock.call_args.kwargs["use_cuda_graph"] is True


def test_authoritative_budget_profiles_and_balanced_schedule():
    next_profiles = parse_budget_profiles(
        "160:272:340:12:80:25;320:544:680:12:160:25;640:1088:1360:12:320:25"
    )
    assert next_profiles is not None
    assert [profile["avg_token_budget"] for profile in next_profiles] == [160, 320, 640]
    assert [profile["keep_k"] for profile in next_profiles] == [272, 544, 1088]
    assert [profile["scope_target_count"] for profile in next_profiles] == [340, 680, 1360]
    assert [profile["mid_target_count"] for profile in next_profiles] == [80, 160, 320]

    v15_profiles = parse_budget_profiles(
        "64:110:137:12:32:25;128:218:272:12:64:25;192:326:408:12:96:25"
    )
    assert v15_profiles is not None
    assert [profile["avg_token_budget"] for profile in v15_profiles] == [64, 128, 192]
    assert [profile["keep_k"] for profile in v15_profiles] == [110, 218, 326]
    assert [profile["scope_target_count"] for profile in v15_profiles] == [137, 272, 408]
    assert [profile["mid_target_count"] for profile in v15_profiles] == [32, 64, 96]
    assert all(
        abs(profile["keep_k"] - 0.8 * profile["scope_target_count"]) <= 0.5
        for profile in v15_profiles
    )

    v15_13b_profiles = parse_budget_profiles(
        "64:143:179:12:32:25;128:286:357:12:64:25;192:429:536:12:96:25"
    )
    assert v15_13b_profiles is not None
    assert [profile["keep_k"] for profile in v15_13b_profiles] == [143, 286, 429]
    assert [profile["scope_target_count"] for profile in v15_13b_profiles] == [179, 357, 536]
    assert all(
        abs(profile["keep_k"] - 0.8 * profile["scope_target_count"]) <= 0.5
        for profile in v15_13b_profiles
    )

    config = LlavaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    tiny_profiles = [
        {
            "avg_token_budget": budget,
            "keep_k": 2,
            "scope_target_count": scope,
            "mid_pruning_layer_idx": 1,
            "mid_target_count": mid,
            "final_wipe_layer_idx": 3,
            "enable_final_wipe": True,
        }
        for budget, scope, mid in ((1, 3, 1), (2, 4, 2), (3, 5, 3))
    ]

    def make_wrapper():
        return LearnablePruneWrapper(
            LlavaLlamaForCausalLM(config),
            LearnablePrunePredictor(input_size=16, hidden_size=8, rank=4, num_heads=1),
            teacher_layer=2,
            budget_profiles=tiny_profiles,
            budget_schedule_seed=123,
        )

    first = make_wrapper()
    second = make_wrapper()
    first_schedule = [first.budget_profile_index_for_micro_step(step) for step in range(12)]
    second_schedule = [second.budget_profile_index_for_micro_step(step) for step in range(12)]
    ignored = set(first._ddp_params_and_buffers_to_ignore)
    assert ignored
    assert all(name.startswith("llava.") for name in ignored)
    assert not any(name.startswith("predictor.") for name in ignored)
    assert first_schedule == second_schedule
    assert all(sorted(first_schedule[start : start + 3]) == [0, 1, 2] for start in range(0, 12, 3))
    for micro_step, expected_index in enumerate(first_schedule):
        profile = first.set_budget_micro_step(micro_step)
        assert first._active_budget_profile_index == expected_index
        assert first.scope_target_count == profile["scope_target_count"]
        assert first.mid_target_count == profile["mid_target_count"]


def test_predictor_budget_topk_handles_packed_rows():
    scores = torch.tensor(
        [
            [0.1, 0.9, 0.2, 0.8, 0.7],
            [5.0, 4.0, 3.0, 100.0, 200.0],
        ]
    )
    valid = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, False, False],
        ]
    )
    indices, selected_valid = LearnablePrunePredictor.budget_topk_indices(scores, 4, valid)
    selected_mask = LearnablePrunePredictor.budget_topk_mask(scores, 4, valid)

    assert indices.shape == selected_valid.shape == (2, 4)
    assert selected_mask[0].nonzero(as_tuple=False).flatten().tolist() == [1, 2, 3, 4]
    assert selected_mask[1].nonzero(as_tuple=False).flatten().tolist() == [0, 1, 2]
    assert selected_valid.sum(dim=-1).tolist() == [4, 3]


def test_distributed_cuda_preflight_rejects_rank_gpu_aliasing():
    keys = ("LOCAL_WORLD_SIZE", "LOCAL_RANK", "CUDA_VISIBLE_DEVICES")
    original = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["LOCAL_WORLD_SIZE"] = "4"
        os.environ["LOCAL_RANK"] = "2"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        with patch("torch.cuda.device_count", return_value=2):
            try:
                validate_distributed_cuda_mapping()
            except RuntimeError as exc:
                assert "more local processes than visible CUDA devices" in str(exc)
            else:
                raise AssertionError("Expected rank/GPU aliasing preflight to fail")
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_three_stage_forward_packs_tokens_and_keeps_predictor_gradient():
    torch.manual_seed(3)
    config = LlavaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        pad_token_id=0,
    )
    config._attn_implementation = "sdpa"
    llava = LlavaLlamaForCausalLM(config)
    predictor = LearnablePrunePredictor(
        input_size=32,
        hidden_size=16,
        rank=8,
        num_heads=1,
    )
    wrapper = LearnablePruneWrapper(
        llava,
        predictor,
        budget_profiles=[
            {
                "avg_token_budget": 2,
                "keep_k": 2,
                "scope_target_count": 5,
                "mid_pruning_layer_idx": 1,
                "mid_target_count": 2,
                "final_wipe_layer_idx": 3,
                "enable_final_wipe": True,
            }
        ],
        teacher_layer=2,
    )
    wrapper.gradient_checkpointing_enable()

    input_ids = torch.tensor(
        [
            [5, -200, -200, -200, -200, -200, -200, 6, 7, 8],
            [9, -200, -200, -200, -200, 10, 11, 12, 0, 0],
        ]
    )
    assert IMAGE_TOKEN_INDEX == -200
    attention_mask = input_ids.ne(0)
    position_ids = torch.arange(input_ids.shape[1])[None].repeat(2, 1)
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    labels[0, -2:] = torch.tensor([20, 21])
    labels[1, 6:8] = torch.tensor([22, 23])
    inputs_embeds = torch.randn(2, input_ids.shape[1], config.hidden_size)
    visual_mask = input_ids.eq(IMAGE_TOKEN_INDEX) & attention_mask
    visual_positions, visual_valid = wrapper._pack_mask_positions(visual_mask)
    predictor_logits = torch.randn(2, visual_positions.shape[1], requires_grad=True)
    topk_indices, topk_selected_valid, hard_topk_mask = (
        wrapper._predictor_topk_selection(predictor_logits, visual_valid)
    )
    retention_probs = wrapper._retention_probabilities(
        predictor_logits, visual_valid, hard_topk_mask
    )
    exact_one_gate = torch.ones_like(retention_probs) + (
        retention_probs - retention_probs.detach()
    )
    assert torch.equal(exact_one_gate, torch.ones_like(exact_one_gate))

    with wrapper._temporary_decoder_checkpoint_training():
        hidden, final_attention, final_labels, stats = wrapper._three_stage_student_forward(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            retention_probs=retention_probs,
            topk_indices=topk_indices,
            topk_selected_valid=topk_selected_valid,
            visual_positions=visual_positions,
            visual_valid=visual_valid,
        )

    assert stats["scope_visual_tokens"].item() == 4.5
    assert stats["mid_visual_tokens"].item() == 2.0
    assert stats["final_visual_tokens"].item() == 0.0
    assert final_attention.sum(dim=-1).tolist() == [4, 4]
    answer_mask = final_attention[:, :-1] & final_labels[:, 1:].ne(IGNORE_INDEX)
    hidden[:, :-1][answer_mask].square().mean().backward()

    assert torch.all(predictor_logits.grad[visual_valid].ne(0))
    assert all(parameter.grad is None for parameter in llava.parameters())


def test_topk_only_forward_prunes_directly_and_keeps_predictor_gradient():
    torch.manual_seed(5)
    config = LlavaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        pad_token_id=0,
    )
    config._attn_implementation = "sdpa"
    llava = LlavaLlamaForCausalLM(config)
    wrapper = LearnablePruneWrapper(
        llava,
        LearnablePrunePredictor(
            input_size=32,
            hidden_size=16,
            rank=8,
            num_heads=1,
        ),
        budget_profiles=[
            {
                "avg_token_budget": 2,
                "keep_k": 2,
                "scope_target_count": 5,
                "mid_pruning_layer_idx": 1,
                "mid_target_count": 2,
                "final_wipe_layer_idx": 3,
                "enable_final_wipe": True,
            }
        ],
        teacher_layer=2,
        enable_training_three_stage_prune=False,
    )
    wrapper.gradient_checkpointing_enable()

    input_ids = torch.tensor(
        [
            [5, -200, -200, -200, -200, -200, -200, 6, 7, 8],
            [9, -200, -200, -200, -200, 10, 11, 12, 0, 0],
        ]
    )
    attention_mask = input_ids.ne(0)
    position_ids = torch.arange(input_ids.shape[1])[None].repeat(2, 1)
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    labels[0, -2:] = torch.tensor([20, 21])
    labels[1, 6:8] = torch.tensor([22, 23])
    inputs_embeds = torch.randn(2, input_ids.shape[1], config.hidden_size)
    visual_mask = input_ids.eq(IMAGE_TOKEN_INDEX) & attention_mask
    visual_positions, visual_valid = wrapper._pack_mask_positions(visual_mask)
    predictor_logits = torch.randn(2, visual_positions.shape[1], requires_grad=True)
    _, _, hard_topk_mask = wrapper._predictor_topk_selection(
        predictor_logits, visual_valid
    )
    retention_probs = wrapper._retention_probabilities(
        predictor_logits, visual_valid, hard_topk_mask
    )

    with wrapper._temporary_decoder_checkpoint_training():
        hidden, final_attention, final_labels, stats = wrapper._topk_student_forward(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            retention_probs=retention_probs,
            hard_topk_mask=hard_topk_mask,
            visual_positions=visual_positions,
            visual_valid=visual_valid,
        )

    assert wrapper.enable_training_three_stage_prune is False
    assert stats["scope_visual_tokens"].item() == 2.0
    assert stats["mid_visual_tokens"].item() == 2.0
    assert stats["final_visual_tokens"].item() == 2.0
    assert stats["avg_visual_tokens_budget"].item() == 2.0
    assert final_attention.sum(dim=-1).tolist() == [6, 6]
    answer_mask = final_attention[:, :-1] & final_labels[:, 1:].ne(IGNORE_INDEX)
    hidden[:, :-1][answer_mask].square().mean().backward()

    assert torch.all(predictor_logits.grad[visual_valid].ne(0))
    assert all(parameter.grad is None for parameter in llava.parameters())


def test_js_and_ce_independently_reach_every_valid_predictor_logit():
    """Physical hard pruning must retain an end-to-end ST gradient path."""

    torch.manual_seed(19)
    config = LlavaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        pad_token_id=0,
    )
    config._attn_implementation = "sdpa"
    wrapper = LearnablePruneWrapper(
        LlavaLlamaForCausalLM(config),
        LearnablePrunePredictor(input_size=32, hidden_size=16, rank=8, num_heads=1),
        teacher_layer=2,
        budget_profiles=[
            {
                "avg_token_budget": 2,
                "keep_k": 2,
                "scope_target_count": 5,
                "mid_pruning_layer_idx": 1,
                "mid_target_count": 2,
                "final_wipe_layer_idx": 3,
                "enable_final_wipe": True,
            }
        ],
    )
    input_ids = torch.tensor(
        [
            [5, -200, -200, -200, -200, -200, -200, 6, 7, 8],
            [9, -200, -200, -200, -200, 10, 11, 12, 0, 0],
        ]
    )
    attention = input_ids.ne(0)
    positions = torch.arange(input_ids.shape[1])[None].repeat(2, 1)
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    labels[0, -2:] = torch.tensor([20, 21])
    labels[1, 6:8] = torch.tensor([22, 23])
    embeddings = torch.randn(2, input_ids.shape[1], config.hidden_size)
    visual_mask = input_ids.eq(IMAGE_TOKEN_INDEX) & attention
    visual_positions, visual_valid = wrapper._pack_mask_positions(visual_mask)
    predictor_logits = torch.randn(
        2, visual_positions.shape[1], requires_grad=True
    )
    topk_indices, topk_selected_valid, hard_topk_mask = (
        wrapper._predictor_topk_selection(predictor_logits, visual_valid)
    )
    retention = wrapper._retention_probabilities(
        predictor_logits, visual_valid, hard_topk_mask
    )
    hidden, student_attention, student_labels, _ = wrapper._three_stage_student_forward(
        inputs_embeds=embeddings,
        input_ids=input_ids,
        attention_mask=attention,
        position_ids=positions,
        labels=labels,
        retention_probs=retention,
        topk_indices=topk_indices,
        topk_selected_valid=topk_selected_valid,
        visual_positions=visual_positions,
        visual_valid=visual_valid,
    )
    answer_mask, _ = wrapper._answer_token_mask(student_attention, student_labels)
    teacher_logits = torch.randn(int(answer_mask.sum().item()), config.vocab_size)
    js_loss, ce_loss = wrapper._js_ce_losses(
        teacher_logits,
        hidden,
        student_attention,
        student_labels,
    )
    js_gradient = torch.autograd.grad(js_loss, predictor_logits, retain_graph=True)[0]
    ce_gradient = torch.autograd.grad(ce_loss, predictor_logits)[0]

    for gradient in (js_gradient, ce_gradient):
        assert torch.isfinite(gradient).all()
        assert torch.all(gradient[visual_valid].ne(0))


def test_vectorized_ragged_pack_matches_row_reference():
    config = LlavaConfig(
        vocab_size=64,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
        pad_token_id=0,
    )
    wrapper = LearnablePruneWrapper(
        LlavaLlamaForCausalLM(config),
        LearnablePrunePredictor(input_size=8, hidden_size=8, rank=4, num_heads=1),
        teacher_layer=1,
        budget_profiles=[
            {
                "avg_token_budget": 1,
                "keep_k": 1,
                "scope_target_count": 2,
                "mid_pruning_layer_idx": 1,
                "mid_target_count": 1,
                "final_wipe_layer_idx": 2,
                "enable_final_wipe": True,
            }
        ],
    )
    hidden = torch.randn(3, 6, 8, requires_grad=True)
    ids = torch.arange(18).view(3, 6)
    attention = torch.tensor(
        [[True, True, True, True, True, True], [True, True, True, True, False, False], [True, True, True, False, False, False]]
    )
    positions = torch.arange(6)[None].expand(3, -1)
    labels = ids + 10
    keep = torch.tensor(
        [[True, False, True, False, False, True], [False, True, False, True, False, False], [True, False, True, False, False, False]]
    )

    packed = wrapper._pack_by_keep_mask(hidden, ids, attention, positions, labels, keep)
    reference_positions = [
        (keep[row] & attention[row]).nonzero(as_tuple=False).flatten() for row in range(3)
    ]
    for row, indices in enumerate(reference_positions):
        count = indices.numel()
        torch.testing.assert_close(packed[0][row, :count], hidden[row].index_select(0, indices))
        assert torch.equal(packed[1][row, :count], ids[row].index_select(0, indices))
        assert torch.equal(packed[3][row, :count], positions[row].index_select(0, indices))
        assert torch.equal(packed[4][row, :count], labels[row].index_select(0, indices))
        assert packed[2][row].sum().item() == count
    packed[0].sum().backward()
    expected_grad = (keep & attention).unsqueeze(-1).expand_as(hidden)
    assert torch.equal(hidden.grad.ne(0), expected_grad)


def test_batched_teacher_scoring_matches_ragged_fallback():
    torch.manual_seed(11)
    config = LlavaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
        pad_token_id=0,
    )
    wrapper = LearnablePruneWrapper(
        LlavaLlamaForCausalLM(config),
        LearnablePrunePredictor(input_size=16, hidden_size=8, rank=4, num_heads=1),
        teacher_layer=1,
        budget_profiles=[
            {
                "avg_token_budget": 1,
                "keep_k": 1,
                "scope_target_count": 2,
                "mid_pruning_layer_idx": 1,
                "mid_target_count": 1,
                "final_wipe_layer_idx": 2,
                "enable_final_wipe": True,
            }
        ],
    )
    batch_size, seq_len = 2, 7
    projected_qkv = {
        name: torch.randn(batch_size, seq_len, config.hidden_size)
        for name in ("query", "key", "value")
    }
    attention = torch.tensor(
        [[True, True, True, True, True, True, True], [True, True, True, True, True, False, False]]
    )
    position_ids = torch.arange(seq_len)[None].expand(batch_size, -1)
    visual_mask = torch.tensor(
        [[False, True, True, False, False, False, False], [False, True, True, False, False, False, False]]
    )
    query_mask = torch.tensor(
        [[False, False, False, False, False, True, True], [False, False, False, False, True, False, False]]
    )
    visual_positions, visual_valid = wrapper._pack_mask_positions(visual_mask)
    query_positions, query_valid = wrapper._pack_mask_positions(query_mask)

    old_ratio = os.environ.get("LEARNABLE_PRUNE_BATCHED_SCORE_MAX_PADDING")
    try:
        os.environ["LEARNABLE_PRUNE_BATCHED_SCORE_MAX_PADDING"] = "0"
        reference = wrapper._get_visual_token_attention_scores(
            projected_qkv,
            attention,
            position_ids,
            1,
            visual_positions,
            visual_valid,
            query_positions,
            query_valid,
        )
        os.environ["LEARNABLE_PRUNE_BATCHED_SCORE_MAX_PADDING"] = "10"
        batched = wrapper._get_visual_token_attention_scores(
            projected_qkv,
            attention,
            position_ids,
            1,
            visual_positions,
            visual_valid,
            query_positions,
            query_valid,
        )
    finally:
        if old_ratio is None:
            os.environ.pop("LEARNABLE_PRUNE_BATCHED_SCORE_MAX_PADDING", None)
        else:
            os.environ["LEARNABLE_PRUNE_BATCHED_SCORE_MAX_PADDING"] = old_ratio

    torch.testing.assert_close(batched, reference, rtol=1e-5, atol=1e-5)


def test_metric_logging_packs_all_scalars_into_one_collective():
    class FakeAccelerator:
        def __init__(self):
            self.calls = 0

        def gather_for_metrics(self, packed):
            self.calls += 1
            return torch.cat((packed, packed + 2.0), dim=0)

    trainer = LearnablePruneTrainer.__new__(LearnablePruneTrainer)
    trainer.accelerator = FakeAccelerator()
    reduced = trainer._metric_scalars({"z": torch.tensor(3.0), "a": torch.tensor(1.0)})
    assert trainer.accelerator.calls == 1
    assert reduced == {"a": 2.0, "z": 4.0}


def test_inference_loads_stage_budgets_from_training_checkpoint():
    config = LlavaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    model = LlavaLearnablePruneLightweightScopeFinalwipeForCausalLM(config)
    predictor = LearnablePrunePredictor(
        input_size=16,
        hidden_size=8,
        rank=4,
        num_heads=1,
    )
    prune_config = {
        "predictor_input_size": 16,
        "predictor_hidden_size": 8,
        "predictor_rank": 4,
        "predictor_num_heads": 1,
        "predictor_rank_mlp_ratio": 2,
        "predictor_use_visual_position": True,
        "predictor_use_text_position": True,
        "keep_k": 3,
        "scope_target_count": 7,
        "mid_pruning_layer_idx": 1,
        "mid_target_count": 2,
        "mid_attn_anchor": "query",
        "final_wipe_layer_idx": 2,
        "enable_final_wipe": False,
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint = Path(tmp_dir)
        torch.save(prune_config, checkpoint / "learnable_prune_config.pt")
        torch.save(predictor.state_dict(), checkpoint / "predictor.pt")
        model.load_learnable_prune_checkpoint(str(checkpoint))

    assert model.learnable_prune_keep_k == 3
    assert model.learnable_prune_scope_target_count == 7
    assert model.learnable_prune_mid_pruning_layer_idx == 1
    assert model.learnable_prune_mid_target_count == 2
    assert model.learnable_prune_mid_attn_anchor == "query"
    assert model.learnable_prune_final_wipe_layer_idx == 2
    assert model.learnable_prune_enable_final_wipe is False


def test_predictor_only_inference_keeps_topk_through_all_layers_without_later_stages():
    torch.manual_seed(13)
    config = LlavaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    model = LlavaLearnablePruneLightweightScopeFinalwipeForCausalLM(config)
    model.learnable_prune_predictor = LearnablePrunePredictor(
        input_size=16,
        hidden_size=8,
        rank=4,
        num_heads=1,
    ).eval()
    model.learnable_prune_config = {"predictor_use_true_visual_coordinates": False}
    model.learnable_prune_keep_k = 2
    model.learnable_prune_active_budget_profile = {"avg_token_budget": 2}
    model.learnable_prune_checkpoint = "test-checkpoint"
    model.learnable_prune_stats = []
    model.eval()

    input_ids = torch.tensor(
        [[7, IMAGE_TOKEN_INDEX, IMAGE_TOKEN_INDEX, IMAGE_TOKEN_INDEX, IMAGE_TOKEN_INDEX, 8]]
    )
    inputs_embeds = torch.randn(1, input_ids.shape[1], config.hidden_size)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
    layer_calls = [0] * config.num_hidden_layers

    def count_layer(layer_idx):
        def hook(_module, _args, _output):
            layer_calls[layer_idx] += 1

        return hook

    handles = [
        layer.register_forward_hook(count_layer(layer_idx))
        for layer_idx, layer in enumerate(model.model.layers)
    ]
    try:
        with (
            patch.dict(
                os.environ,
                {
                    "ENABLE_PREDICTOR": "1",
                    "LEARNABLE_PRUNE_PREDICTOR_ONLY": "1",
                    "LEARNABLE_TOPK": "2",
                },
                clear=False,
            ),
            patch(
                "llava.model.learnable_prune_lightweight_scope_finalwipe.official_modeling.SeededResidualSCOPE",
                side_effect=AssertionError("SCOPE must not run in predictor-only mode"),
            ) as scope_mock,
            patch.object(
                model,
                "_get_visual_token_attention_scores",
                side_effect=AssertionError("mid-layer scoring must not run in predictor-only mode"),
            ) as mid_score_mock,
            patch.object(
                model,
                "_prune_by_mid_attention_scores",
                side_effect=AssertionError("mid-layer pruning must not run in predictor-only mode"),
            ) as mid_prune_mock,
            patch.object(
                model,
                "_wipe_visual_tokens",
                side_effect=AssertionError("final wipe must not run in predictor-only mode"),
            ) as final_wipe_mock,
        ):
            outputs = model._prefill_with_finalwipe(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                visual_coordinates=None,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    hidden_states, past_key_values, _, _, pruned_input_ids, pruned_attention_mask, _ = outputs
    assert layer_calls == [1, 1, 1]
    assert hidden_states.shape[1] == 4
    assert past_key_values.get_seq_length() == 4
    assert int(pruned_input_ids.eq(IMAGE_TOKEN_INDEX).sum().item()) == 2
    assert pruned_attention_mask.shape == (1, 4)
    scope_mock.assert_not_called()
    mid_score_mock.assert_not_called()
    mid_prune_mock.assert_not_called()
    final_wipe_mock.assert_not_called()

    stats = model.learnable_prune_stats[-1]
    assert stats["pruning_mode"] == "predictor_topk_only"
    assert stats["scope_enabled"] is False
    assert stats["scope_target_visual_tokens"] == 0
    assert stats["mid_layer_prune_enabled"] is False
    assert stats["enable_finalwipe"] is False
    assert stats["avg_visual_tokens_budget"] == 2.0


def test_scope_only_prefill_fills_from_empty_seed_to_avg64_scope_target_without_loading_predictor():
    torch.manual_seed(17)
    config = LlavaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=192,
        pad_token_id=0,
    )
    model = LlavaLearnablePruneLightweightScopeFinalwipeForCausalLM(config)
    model.eval()

    input_ids = torch.tensor([[7] + [IMAGE_TOKEN_INDEX] * 160 + [8]])
    inputs_embeds = torch.randn(1, input_ids.shape[1], config.hidden_size)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)

    with (
        patch.dict(
            os.environ,
            {
                "ENABLE_PREDICTOR": "0",
                "LEARNABLE_PRUNE_PREDICTOR_ONLY": "0",
                "LEARNABLE_TOPK": "0",
                "SCOPE_TARGET_COUNT": "137",
            },
            clear=False,
        ),
        patch.object(
            model,
            "_ensure_learnable_prune_loaded",
            side_effect=AssertionError("scope-only mode must not load a predictor"),
        ) as load_predictor_mock,
        patch(
            "llava.model.learnable_prune_lightweight_scope_finalwipe.official_modeling.SeededResidualSCOPE",
            wraps=SeededResidualSCOPE,
        ) as scope_mock,
    ):
        _, pruned_input_ids, pruned_attention_mask, _, _ = model._learnable_scope_prune_prefill(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            visual_coordinates=None,
        )

    load_predictor_mock.assert_not_called()
    scope_mock.assert_called_once()
    assert scope_mock.call_args.kwargs["seed_relative"].numel() == 0
    assert scope_mock.call_args.kwargs["target_keep"] == 137
    assert int(pruned_input_ids.eq(IMAGE_TOKEN_INDEX).sum().item()) == 137
    assert pruned_attention_mask.shape == (1, 139)

    stats = model.learnable_prune_stats[-1]
    assert stats["enable_predictor"] is False
    assert stats["learnable_topk_visual_tokens"] == 0
    assert stats["scope_target_visual_tokens"] == 137
    assert stats["kept_visual_tokens"] == 137


def test_inference_selects_saved_multi_budget_profile():
    config = LlavaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    predictor = LearnablePrunePredictor(
        input_size=16,
        hidden_size=8,
        rank=4,
        num_heads=1,
    )
    profiles = parse_budget_profiles(
        "160:272:340:12:80:25;320:544:680:12:160:25;640:1088:1360:12:320:25"
    )
    prune_config = {
        "predictor_input_size": 16,
        "predictor_hidden_size": 8,
        "predictor_rank": 4,
        "predictor_num_heads": 1,
        "predictor_rank_mlp_ratio": 2,
        "predictor_use_visual_position": True,
        "predictor_use_text_position": True,
        "keep_k": 272,
        "scope_target_count": 340,
        "mid_pruning_layer_idx": 12,
        "mid_target_count": 80,
        "final_wipe_layer_idx": 25,
        "enable_final_wipe": True,
        "budget_profiles": profiles,
        "default_avg_token_budget": 160,
    }
    old_budget = os.environ.get("LEARNABLE_PRUNE_AVG_TOKEN_BUDGET")
    old_topk = os.environ.get("LEARNABLE_TOPK")
    try:
        os.environ["LEARNABLE_PRUNE_AVG_TOKEN_BUDGET"] = "640"
        os.environ.pop("LEARNABLE_TOPK", None)
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint = Path(tmp_dir)
            torch.save(prune_config, checkpoint / "learnable_prune_config.pt")
            torch.save(predictor.state_dict(), checkpoint / "predictor.pt")
            model = LlavaLearnablePruneLightweightScopeFinalwipeForCausalLM(config)
            model.load_learnable_prune_checkpoint(str(checkpoint))
    finally:
        if old_budget is None:
            os.environ.pop("LEARNABLE_PRUNE_AVG_TOKEN_BUDGET", None)
        else:
            os.environ["LEARNABLE_PRUNE_AVG_TOKEN_BUDGET"] = old_budget
        if old_topk is None:
            os.environ.pop("LEARNABLE_TOPK", None)
        else:
            os.environ["LEARNABLE_TOPK"] = old_topk

    assert model.learnable_prune_keep_k == 1088
    assert model.learnable_prune_scope_target_count == 1360
    assert model.learnable_prune_mid_target_count == 320
    assert model.learnable_prune_active_budget_profile["avg_token_budget"] == 640


def test_full_loss_backward_updates_predictor_through_three_stage_student():
    torch.manual_seed(9)
    config = LlavaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        pad_token_id=0,
    )
    config._attn_implementation = "sdpa"
    config.tokenizer_padding_side = "right"
    llava = LlavaLlamaForCausalLM(config)
    predictor = LearnablePrunePredictor(
        input_size=32,
        hidden_size=16,
        rank=8,
        num_heads=1,
    )
    wrapper = LearnablePruneWrapper(
        llava,
        predictor,
        teacher_layer=2,
        budget_profiles=[
            {
                "avg_token_budget": budget,
                "keep_k": 2,
                "scope_target_count": scope,
                "mid_pruning_layer_idx": 1,
                "mid_target_count": mid,
                "final_wipe_layer_idx": 3,
                "enable_final_wipe": True,
            }
        for budget, scope, mid in ((1, 3, 1), (2, 4, 2), (3, 5, 3))
        ],
    )
    wrapper._apply_budget_profile(1)
    expanded_ids = torch.tensor([[5, -200, -200, -200, -200, -200, 6, 7, 8]])
    expanded_attention = torch.ones_like(expanded_ids, dtype=torch.bool)
    expanded_positions = torch.arange(expanded_ids.shape[1])[None]
    expanded_labels = torch.full_like(expanded_ids, IGNORE_INDEX)
    expanded_labels[0, -2:] = torch.tensor([10, 11])
    expanded_embeds = torch.randn(1, expanded_ids.shape[1], config.hidden_size)
    expanded_coordinates = torch.randn(1, expanded_ids.shape[1], 2)

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
    outputs = wrapper(
        input_ids=torch.tensor([[1]]),
        attention_mask=torch.ones(1, 1, dtype=torch.bool),
        labels=torch.tensor([[IGNORE_INDEX]]),
        images=torch.zeros(1),
    )
    outputs["loss"].backward()

    assert torch.isfinite(outputs["loss"])
    assert outputs["scope_visual_tokens"].item() == 4.0
    assert outputs["mid_visual_tokens"].item() == 2.0
    assert outputs["final_visual_tokens"].item() == 0.0
    assert outputs["active_avg_token_budget"].item() == 2.0
    assert "budget_2_total_loss" in outputs
    assert all(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0
        for parameter in predictor.parameters()
    )


if __name__ == "__main__":
    test_training_scope_matches_inference_scope()
    test_training_scope_enables_thread_local_cuda_graph_path()
    test_authoritative_budget_profiles_and_balanced_schedule()
    test_predictor_budget_topk_handles_packed_rows()
    test_distributed_cuda_preflight_rejects_rank_gpu_aliasing()
    test_three_stage_forward_packs_tokens_and_keeps_predictor_gradient()
    test_topk_only_forward_prunes_directly_and_keeps_predictor_gradient()
    test_js_and_ce_independently_reach_every_valid_predictor_logit()
    test_vectorized_ragged_pack_matches_row_reference()
    test_batched_teacher_scoring_matches_ragged_fallback()
    test_metric_logging_packs_all_scalars_into_one_collective()
    test_inference_loads_stage_budgets_from_training_checkpoint()
    test_inference_selects_saved_multi_budget_profile()
    test_full_loss_backward_updates_predictor_through_three_stage_student()
    print("three-stage alignment tests passed")
