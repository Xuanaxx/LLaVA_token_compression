"""Compatibility aliases for the relocated scoring-layer sweep model."""

from best_layer_sweep.modeling import (
    LlavaBestLayerSweepForCausalLM,
    LlavaBestLayerSweepModel,
    LlavaForConditionalGeneration,
)

LlavaImportancePruningCorrelationModel = LlavaBestLayerSweepModel
LlavaImportancePruningCorrelationForCausalLM = LlavaBestLayerSweepForCausalLM

__all__ = [
    "LlavaForConditionalGeneration",
    "LlavaImportancePruningCorrelationForCausalLM",
    "LlavaImportancePruningCorrelationModel",
]
