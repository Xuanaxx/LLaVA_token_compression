#!/usr/bin/env python3
# coding: utf-8
"""Compatibility exports for the official LLaVA learnable-prune implementation."""

from .official_modeling import (
    DEFAULT_CHECKPOINT,
    ENABLE_FINALWIPE,
    FINAL_WIPE_LAYER_IDX,
    LEARNABLE_TOPK,
    MERGE_TARGET_COUNT,
    RECOVER_LAYER_IDX,
    SCOPE_TARGET_COUNT,
    LlavaForConditionalGeneration,
    LlavaLearnablePruneScopeRecoverFinalwipeForCausalLM,
    SeededResidualSCOPE,
)

__all__ = [
    "DEFAULT_CHECKPOINT",
    "ENABLE_FINALWIPE",
    "FINAL_WIPE_LAYER_IDX",
    "LEARNABLE_TOPK",
    "MERGE_TARGET_COUNT",
    "RECOVER_LAYER_IDX",
    "SCOPE_TARGET_COUNT",
    "LlavaForConditionalGeneration",
    "LlavaLearnablePruneScopeRecoverFinalwipeForCausalLM",
    "SeededResidualSCOPE",
]
