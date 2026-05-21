#!/usr/bin/env python3
# coding: utf-8
"""Compatibility exports for the official LLaVA learnable-prune implementation."""

from .official_modeling import (
    DEFAULT_CHECKPOINT,
    FINAL_WIPE_LAYER_IDX,
    LEARNABLE_TOPK,
    SCOPE_TARGET_COUNT,
    LlavaForConditionalGeneration,
    LlavaLearnablePruneScopeFinalwipeForCausalLM,
    SeededResidualSCOPE,
)

__all__ = [
    "DEFAULT_CHECKPOINT",
    "FINAL_WIPE_LAYER_IDX",
    "LEARNABLE_TOPK",
    "SCOPE_TARGET_COUNT",
    "LlavaForConditionalGeneration",
    "LlavaLearnablePruneScopeFinalwipeForCausalLM",
    "SeededResidualSCOPE",
]
