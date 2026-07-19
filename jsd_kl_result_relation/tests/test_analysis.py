from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from best_layer_sweep.modeling import LlavaBestLayerSweepForCausalLM
from jsd_kl_result_relation.correlate import analyze_correlations, load_layer_rows
from jsd_kl_result_relation.evaluate_layer import (
    full_reference_prefill,
    parse_layers,
    paired_greedy_generate,
    sampled_indices,
    score_and_prune_many,
)
from jsd_kl_result_relation.finalize_shards import finalize_shards
from jsd_kl_result_relation.metrics import DIVERGENCE_METRICS, distribution_divergence
from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.language_model.llava_llama import LlavaConfig


class DistributionMetricsTest(unittest.TestCase):
    def test_identical_logits_have_zero_divergence(self) -> None:
        logits = torch.tensor([[1.0, -0.5, 2.0]])
        result = distribution_divergence(logits, logits.clone())
        self.assertAlmostEqual(result["jsd"], 0.0, places=7)
        self.assertAlmostEqual(result["kl_full_to_pruned"], 0.0, places=7)
        self.assertTrue(result["top1_match"])

    def test_jsd_is_symmetric_but_kl_is_directional(self) -> None:
        full = torch.tensor([[4.0, 0.0, -1.0]])
        pruned = torch.tensor([[-2.0, 1.0, 2.0]])
        forward = distribution_divergence(full, pruned)
        reverse = distribution_divergence(pruned, full)
        self.assertAlmostEqual(forward["jsd"], reverse["jsd"], places=7)
        self.assertAlmostEqual(
            forward["kl_full_to_pruned"], reverse["kl_pruned_to_full"], places=7
        )
        self.assertNotAlmostEqual(
            forward["kl_full_to_pruned"], forward["kl_pruned_to_full"], places=4
        )


class SamplingTest(unittest.TestCase):
    def test_layer_spec_supports_ranges_order_and_deduplication(self) -> None:
        self.assertEqual(parse_layers("0-2, 5 2 4-3"), [0, 1, 2, 5, 4, 3])

    def test_sampling_is_exact_and_reproducible(self) -> None:
        first = sampled_indices(1000, 500, 42)
        second = sampled_indices(1000, 500, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 500)
        self.assertEqual(len(set(first)), 500)

    def test_sampling_rejects_short_dataset(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 500"):
            sampled_indices(499, 500, 42)


class InferenceEquivalenceTest(unittest.TestCase):
    @staticmethod
    def _model() -> LlavaBestLayerSweepForCausalLM:
        torch.manual_seed(7)
        config = LlavaConfig(
            vocab_size=37,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        return LlavaBestLayerSweepForCausalLM(config).eval()

    def test_single_pass_layer_scores_match_independent_prefix_forwards(self) -> None:
        model = self._model()
        torch.manual_seed(11)
        embeddings = torch.randn(1, 9, model.config.hidden_size)
        input_ids = torch.tensor(
            [
                [
                    1,
                    IMAGE_TOKEN_INDEX,
                    IMAGE_TOKEN_INDEX,
                    IMAGE_TOKEN_INDEX,
                    IMAGE_TOKEN_INDEX,
                    5,
                    6,
                    7,
                    8,
                ]
            ]
        )
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
        visual_positions = (
            input_ids[0].eq(IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).flatten()
        )
        query_indices = model._build_query_indices(
            input_ids, attention_mask, visual_positions
        )

        with torch.inference_mode():
            optimized = score_and_prune_many(
                model,
                input_ids,
                embeddings,
                attention_mask,
                position_ids,
                scoring_layers=[0, 1, 2],
                topk=2,
            )
            for layer in range(3):
                layer_input = model._forward_to_layer_input(
                    embeddings,
                    attention_mask,
                    position_ids,
                    scoring_layer_idx=layer,
                )
                scores = model._visual_token_scores(
                    layer_input,
                    attention_mask,
                    position_ids,
                    scoring_layer_idx=layer,
                    visual_positions=visual_positions,
                    query_indices=query_indices,
                )
                keep = model._topk_keep_positions(input_ids, attention_mask, scores, 2)
                self.assertTrue(
                    torch.equal(
                        optimized[layer]["pruned_position_ids"],
                        position_ids.index_select(1, keep),
                    )
                )
                self.assertTrue(
                    torch.equal(
                        optimized[layer]["pruned_embeds"],
                        embeddings.index_select(1, keep),
                    )
                )

    def test_static_cache_generation_matches_dynamic_reference(self) -> None:
        model = self._model()
        torch.manual_seed(13)
        full_embeds = torch.randn(1, 7, model.config.hidden_size)
        full_mask = torch.ones(1, 7, dtype=torch.bool)
        full_positions = torch.arange(7).unsqueeze(0)
        keep = torch.tensor([0, 1, 4, 5, 6])
        pruned_embeds = full_embeds.index_select(1, keep)
        pruned_positions = full_positions.index_select(1, keep)

        class Tokenizer:
            eos_token_id = None

        def dynamic_reference():
            full_hidden, dynamic_full, _, _ = model._manual_decode(
                full_embeds,
                full_mask,
                full_positions,
                cache_position=torch.arange(7),
                use_cache=True,
            )
            pruned_mask = torch.ones(1, 5, dtype=torch.bool)
            pruned_hidden, dynamic_pruned, _, _ = model._manual_decode(
                pruned_embeds,
                pruned_mask,
                pruned_positions,
                cache_position=torch.arange(5),
                use_cache=True,
            )
            reference_full_logits = model.lm_head(full_hidden[:, -1])
            reference_pruned_logits = model.lm_head(pruned_hidden[:, -1])
            generated = []
            rows = []
            for step in range(4):
                rows.append(
                    distribution_divergence(
                        reference_full_logits, reference_pruned_logits
                    )
                )
                next_token = int(torch.argmax(reference_pruned_logits, dim=-1).item())
                generated.append(next_token)
                if step == 3:
                    break
                token_embed = model.get_model().embed_tokens(
                    torch.tensor([[next_token]])
                )
                semantic_position = torch.tensor([[7 + step]])
                full_hidden, dynamic_full, _, _ = model._manual_decode(
                    token_embed,
                    torch.ones(1, 8 + step, dtype=torch.bool),
                    semantic_position,
                    past_key_values=dynamic_full,
                    cache_position=torch.tensor([7 + step]),
                    use_cache=True,
                )
                pruned_hidden, dynamic_pruned, _, _ = model._manual_decode(
                    token_embed,
                    torch.ones(1, 6 + step, dtype=torch.bool),
                    semantic_position,
                    past_key_values=dynamic_pruned,
                    cache_position=torch.tensor([5 + step]),
                    use_cache=True,
                )
                reference_full_logits = model.lm_head(full_hidden[:, -1])
                reference_pruned_logits = model.lm_head(pruned_hidden[:, -1])
            return generated, {
                "first_token_jsd": rows[0]["jsd"],
                "first_token_kl_full_to_pruned": rows[0]["kl_full_to_pruned"],
                "first_token_kl_pruned_to_full": rows[0]["kl_pruned_to_full"],
                "first_token_top1_match": rows[0]["top1_match"],
                "generation_mean_jsd": sum(row["jsd"] for row in rows) / len(rows),
                "generation_mean_kl_full_to_pruned": sum(
                    row["kl_full_to_pruned"] for row in rows
                )
                / len(rows),
                "generation_mean_kl_pruned_to_full": sum(
                    row["kl_pruned_to_full"] for row in rows
                )
                / len(rows),
                "generation_top1_match_rate": sum(
                    1.0 if row["top1_match"] else 0.0 for row in rows
                )
                / len(rows),
                "num_compared_steps": len(rows),
            }

        with torch.inference_mode():
            generated_reference, metrics_reference = dynamic_reference()
            full_logits, full_cache = full_reference_prefill(
                model,
                full_embeds,
                full_mask,
                full_positions,
            )
            generated_a, metrics_a = paired_greedy_generate(
                model=model,
                tokenizer=Tokenizer(),
                full_logits=full_logits,
                full_cache=full_cache,
                full_length=7,
                full_attention_mask=full_mask,
                next_semantic_position=7,
                pruned_embeds=pruned_embeds,
                pruned_position_ids=pruned_positions,
                max_new_tokens=4,
            )

        self.assertEqual(int(full_cache.get_seq_length()), 7)
        self.assertEqual(generated_a, generated_reference)
        for key, value in metrics_reference.items():
            if isinstance(value, bool) or isinstance(value, int):
                self.assertEqual(metrics_a[key], value)
            else:
                self.assertAlmostEqual(metrics_a[key], value, places=6)


class ShardFinalizationTest(unittest.TestCase):
    def test_merges_strided_shards_in_global_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            global_indices = [10, 20, 30, 40]
            expected_records: dict[int, list[dict]] = {}
            for layer in (0, 1):
                config = {
                    "model": "tiny",
                    "task": "refcoco",
                    "scoring_layer": layer,
                    "topk": 64,
                    "sample_size": 4,
                    "caption_metrics": ["CIDEr"],
                }
                fingerprint = hashlib.sha256(
                    json.dumps(config, sort_keys=True, ensure_ascii=False).encode()
                ).hexdigest()
                layer_records = []
                for dataset_index in global_indices:
                    record = {
                        "model": "tiny",
                        "task": "refcoco",
                        "scoring_layer": layer,
                        "topk": 64,
                        "config_sha256": fingerprint,
                        "dataset_index": dataset_index,
                        "answer": [f"answer-{dataset_index}"],
                        "prediction": f"prediction-{layer}-{dataset_index}",
                        **{
                            metric: float(layer + dataset_index)
                            for metric in DIVERGENCE_METRICS
                        },
                        "first_token_top1_match": True,
                        "generation_top1_match_rate": 1.0,
                        "num_compared_steps": 2,
                    }
                    layer_records.append(record)
                expected_records[layer] = layer_records
                layer_dir = root / f"scoring_layer_{layer}"
                layer_dir.mkdir(parents=True)
                for shard_index in range(2):
                    shard_number = shard_index + 1
                    suffix = f".shard_{shard_number}_of_2"
                    shard_indices = global_indices[shard_index::2]
                    manifest = {"config": config, "indices": shard_indices}
                    (layer_dir / f"sample_indices{suffix}.json").write_text(
                        json.dumps(manifest), encoding="utf-8"
                    )
                    shard_records = layer_records[shard_index::2]
                    (layer_dir / f"records{suffix}.jsonl").write_text(
                        "".join(json.dumps(record) + "\n" for record in shard_records),
                        encoding="utf-8",
                    )

            performance = {0: {"CIDEr": 0.1}, 1: {"CIDEr": 0.2}}
            with patch(
                "jsd_kl_result_relation.finalize_shards._caption_performance_many",
                return_value=performance,
            ):
                finalize_shards(root, [0, 1], shard_count=2, expected_samples=4)

            for layer in (0, 1):
                layer_dir = root / f"scoring_layer_{layer}"
                merged = [
                    json.loads(line)
                    for line in (layer_dir / "records.jsonl").read_text().splitlines()
                ]
                self.assertEqual(merged, expected_records[layer])
                manifest = json.loads((layer_dir / "sample_indices.json").read_text())
                self.assertEqual(manifest["indices"], global_indices)
                summary = json.loads((layer_dir / "layer_summary.json").read_text())
                self.assertEqual(summary["num_records"], 4)
                self.assertEqual(summary["performance"], performance[layer])


class CorrelationAnalysisTest(unittest.TestCase):
    @staticmethod
    def _write_summary(
        root: Path, task: str, layer: int, divergence: float, cider: float
    ) -> None:
        layer_dir = root / "llava-next" / task / f"scoring_layer_{layer}"
        layer_dir.mkdir(parents=True)
        payload = {
            "config": {"model": "llava-next", "task": task, "scoring_layer": layer},
            "num_records": 500,
            "sample_indices_sha256": f"same-{task}",
            "mean_divergence": {metric: divergence for metric in DIVERGENCE_METRICS},
            "performance": {"CIDEr": cider, "Bleu_4": cider / 2.0},
        }
        (layer_dir / "layer_summary.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def test_reports_negative_layer_correlation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for task in ("refcoco", "refcoco_plus"):
                for layer, divergence in enumerate((0.3, 0.2, 0.1)):
                    self._write_summary(
                        root, task, layer, divergence, cider=1.0 - divergence
                    )
            rows = load_layer_rows(root, expected_samples=500)
            output = analyze_correlations(rows)
            result = output["models"]["llava-next"]["tasks"]["refcoco"][
                "correlations_divergence_vs_pruned_performance"
            ]["CIDEr"]["first_token_jsd"]
            self.assertAlmostEqual(result["pearson"], -1.0)
            self.assertAlmostEqual(result["spearman"], -1.0)
            pooled = output["models"]["llava-next"][
                "pooled_task_standardized_correlations"
            ]["CIDEr"]["generation_mean_jsd"]
            self.assertEqual(pooled["n"], 6)
            self.assertAlmostEqual(pooled["pearson"], -1.0)

    def test_rejects_misaligned_samples_across_layers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_summary(root, "refcoco", 0, 0.2, 0.8)
            self._write_summary(root, "refcoco", 1, 0.1, 0.9)
            path = root / "llava-next/refcoco/scoring_layer_1/layer_summary.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["sample_indices_sha256"] = "different"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different sampled examples"):
                analyze_correlations(load_layer_rows(root, expected_samples=500))


if __name__ == "__main__":
    unittest.main()
