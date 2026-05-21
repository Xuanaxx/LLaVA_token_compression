#!/usr/bin/env bash
set -euo pipefail

if command -v conda >/dev/null 2>&1; then
  conda deactivate >/dev/null 2>&1 || true
fi
source /data1/chenzixuan/uv_env/.tokencompression/bin/activate
cd /data1/chenzixuan/open_source_projects/LLaVA_token_compression/learnable_pruner

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-"expandable_segments:True"}
export WANDB_PROJECT=${WANDB_PROJECT:-"llava_learnable_prune_dynamic_kl"}
export WANDB_API_KEY="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"

BATCH_SIZE=${BATCH_SIZE:-32}
NUM_GPUS=${NUM_GPUS:-4}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-$((BATCH_SIZE / (PER_DEVICE_BATCH_SIZE * NUM_GPUS)))}
MAX_STEPS=${MAX_STEPS:--1}
MAX_SAMPLES=${MAX_SAMPLES:-}
SAMPLE_RATE=${SAMPLE_RATE:-0.1}
LOGGING_STEPS=${LOGGING_STEPS:-10}
BF16=${BF16:-true}
FP16=${FP16:-false}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-16}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-true}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-}

# One-stage Dynamic-KL precision-at-K hinge objective weights.
RANK_LOSS_WEIGHT=${RANK_LOSS_WEIGHT:-1.0}
KL_LOSS_WEIGHT=${KL_LOSS_WEIGHT:-2.0}
CE_LOSS_WEIGHT=${CE_LOSS_WEIGHT:-0.25}
ENABLE_SCALE=${ENABLE_SCALE:-false}
ENABLE_RSS=${ENABLE_RSS:-false}

MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"/data1/chenzixuan/model/liuhaotian/llava-v1.5-7b"}
DATA_DIR=${DATA_DIR:-"/data1/czx/data/llava_1_5_mix665k_full"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"/data1/chenzixuan/train_output"}
RUN_NAME=${RUN_NAME:-"official_llava_learnable_prune_precision_at_k_hinge_top64_layers16_24_top1_iqr"}
TEACHER_TARGET_LOG_DIR=${TEACHER_TARGET_LOG_DIR:-"$OUTPUT_ROOT/$RUN_NAME/teacher_target_logs"}
TEACHER_TARGET_LOG_TO_CONSOLE=${TEACHER_TARGET_LOG_TO_CONSOLE:-false}

EXTRA_ARGS=()
if [[ -n "${MAX_SAMPLES}" ]]; then
  EXTRA_ARGS+=(--max_samples "$MAX_SAMPLES")
fi
if [[ -n "${SAMPLE_RATE}" ]]; then
  EXTRA_ARGS+=(--sample_rate "$SAMPLE_RATE")
fi
if [[ "${BF16}" == "1" || "${BF16,,}" == "true" ]]; then
  EXTRA_ARGS+=(--bf16)
fi
if [[ "${FP16}" == "1" || "${FP16,,}" == "true" ]]; then
  EXTRA_ARGS+=(--fp16)
fi

ACCELERATE_ARGS=(--num_processes "$NUM_GPUS")
if (( NUM_GPUS > 1 )); then
  ACCELERATE_ARGS=(--multi_gpu "${ACCELERATE_ARGS[@]}")
fi
if [[ -n "${MAIN_PROCESS_PORT}" ]]; then
  ACCELERATE_ARGS+=(--main_process_port "$MAIN_PROCESS_PORT")
fi

accelerate launch "${ACCELERATE_ARGS[@]}" train_learnable_pruner.py \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_ROOT/$RUN_NAME" \
  --run_name "$RUN_NAME" \
  --wandb_project "$WANDB_PROJECT" \
  --keep_k 64 \
  --candidate_layers "16-24" \
  --predictor_hidden_size 512 \
  --predictor_heads 8 \
  --predictor_layers 4 \
  --predictor_mlp_ratio 2 \
  --predictor_use_final_full_attention true \
  --rank_loss_weight "$RANK_LOSS_WEIGHT" \
  --kl_loss_weight "$KL_LOSS_WEIGHT" \
  --ce_loss_weight "$CE_LOSS_WEIGHT" \
  --enable_scale "$ENABLE_SCALE" \
  --enable_rss "$ENABLE_RSS" \
  --teacher_top_iqr_layers 1 \
  --topk_hinge_margin 1.0 \
  --kd_temperature 1.0 \
  --budgeted_soft_topk_iters 32 \
  --teacher_target_log_dir "$TEACHER_TARGET_LOG_DIR" \
  --teacher_target_log_to_console "$TEACHER_TARGET_LOG_TO_CONSOLE" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning_rate 1e-4 \
  --num_train_epochs 1 \
  --max_steps "$MAX_STEPS" \
  --logging_steps "$LOGGING_STEPS" \
  --save_steps 500 \
  --save_total_limit 4 \
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
  --attn_implementation "$ATTN_IMPLEMENTATION" \
  --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
  --ddp_find_unused_parameters false \
  "${EXTRA_ARGS[@]}"
