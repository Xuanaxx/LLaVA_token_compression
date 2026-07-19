from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from best_layer_sweep.model_registry import resolve_models
from best_layer_sweep.summarize import summarize


MODEL_ROOT = Path("/data1/chenzixuan/model/liuhaotian")


class ModelRegistryTest(unittest.TestCase):
    def test_official_models_and_layer_counts(self) -> None:
        specs = resolve_models(MODEL_ROOT, "all")
        self.assertEqual(
            [(spec.name, spec.num_hidden_layers) for spec in specs],
            [
                ("llava-v1.5-7b", 32),
                ("llava-v1.5-13b", 40),
                ("llava-v1.6-vicuna-7b", 32),
            ],
        )


class SummaryTest(unittest.TestCase):
    @staticmethod
    def _write_result(root: Path, model: str, task: str, layer: int, score: float) -> None:
        result_dir = root / model / task / f"scoring_layer_{layer}" / "checkpoint"
        result_dir.mkdir(parents=True)
        result = {
            "date": "20260101_000000",
            "results": {task: {"exact_match,none": score}},
            "n-samples": {task: {"effective": 500, "original": 1000}},
        }
        (result_dir / "20260101_000000_results.json").write_text(json.dumps(result), encoding="utf-8")

    def test_best_layer_across_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for task, layer_scores in {
                "gqa": {0: 0.4, 1: 0.6},
                "textvqa_val": {0: 0.5, 1: 0.5},
            }.items():
                for layer, score in layer_scores.items():
                    self._write_result(root, "llava-v1.5-7b", task, layer, score)
            output = summarize(root, expected_samples=500)
            self.assertEqual(output["best_layer"], 1)
            self.assertEqual(output["models"]["llava-v1.5-7b"]["tasks"]["gqa"]["best_layer"], 1)

    def test_rejects_wrong_sample_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_result(root, "llava-v1.5-7b", "gqa", 0, 0.4)
            with self.assertRaisesRegex(ValueError, "Expected 1 evaluated samples"):
                summarize(root, expected_samples=1)


if __name__ == "__main__":
    unittest.main()
