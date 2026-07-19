"""Official LLaVA scoring-layer sweep package."""

from typing import Any

__all__ = ["LlavaBestLayerSweepForCausalLM", "LlavaForConditionalGeneration"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .modeling import LlavaBestLayerSweepForCausalLM, LlavaForConditionalGeneration

        return {
            "LlavaBestLayerSweepForCausalLM": LlavaBestLayerSweepForCausalLM,
            "LlavaForConditionalGeneration": LlavaForConditionalGeneration,
        }[name]
    raise AttributeError(name)
