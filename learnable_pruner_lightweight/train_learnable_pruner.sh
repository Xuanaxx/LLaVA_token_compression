#!/usr/bin/env bash
set -euo pipefail

if command -v conda >/dev/null 2>&1; then
  conda deactivate >/dev/null 2>&1 || true
fi
source /data1/chenzixuan/uv_env/.tokencompression/bin/activate
cd /data1/chenzixuan/open_source_projects/LLaVA_token_compression/learnable_pruner_lightweight

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,7}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-"expandable_segments:True"}
export WANDB_PROJECT=${WANDB_PROJECT:-"llava_learnable_prune_lightweight_fixed_layer"}
export WANDB_API_KEY="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"
# Provide WANDB_API_KEY in the environment if needed.

BATCH_SIZE=${BATCH_SIZE:-64}
NUM_GPUS=${NUM_GPUS:-2}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-$((BATCH_SIZE / (PER_DEVICE_BATCH_SIZE * NUM_GPUS)))}
MAX_STEPS=${MAX_STEPS:--1}
MAX_SAMPLES=${MAX_SAMPLES:-}
SAMPLE_RATE=${SAMPLE_RATE:-0.2}
LOGGING_STEPS=${LOGGING_STEPS:-10}
BF16=${BF16:-true}
FP16=${FP16:-false}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-true}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-}

# One-stage fixed-layer JS-divergence precision-at-K hinge objective weights.
RANK_LOSS_WEIGHT=${RANK_LOSS_WEIGHT:-0.5}
JS_LOSS_WEIGHT=${JS_LOSS_WEIGHT:-${KL_LOSS_WEIGHT:-6.0}}
CE_LOSS_WEIGHT=${CE_LOSS_WEIGHT:-0.5}
ENABLE_SCALE=${ENABLE_SCALE:-true}
ENABLE_RSS=${ENABLE_RSS:-false}

MODEL_SIZE=${MODEL_SIZE:-13b}
if [[ "$MODEL_SIZE" != "7b" && "$MODEL_SIZE" != "13b" ]]; then
  echo "MODEL_SIZE must be 7b or 13b, got: $MODEL_SIZE" >&2
  exit 2
fi
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"/data1/chenzixuan/model/liuhaotian/llava-v1.5-$MODEL_SIZE"}
DATA_DIR=${DATA_DIR:-"/data2/czx/data/llava_1_5_mix665k_full"}
DATA_PATH=${DATA_PATH:-}
IMAGE_FOLDER=${IMAGE_FOLDER:-}
OUTPUT_ROOT=${OUTPUT_ROOT:-"/data1/chenzixuan/train_output"}
RUN_NAME=${RUN_NAME:-"official_llava_v1.5_${MODEL_SIZE}_learnable_prune_lightweight_top64_layer18_sample0.2"}
TEACHER_TARGET_LOG_DIR=${TEACHER_TARGET_LOG_DIR:-}
TEACHER_TARGET_LOG_TO_CONSOLE=${TEACHER_TARGET_LOG_TO_CONSOLE:-false}
TEACHER_LAYER=${TEACHER_LAYER:-18}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
IMAGE_ASPECT_RATIO=${IMAGE_ASPECT_RATIO:-pad}

# TCLM-RankSwiGLU predictor config.
PREDICTOR_HIDDEN_SIZE=${PREDICTOR_HIDDEN_SIZE:-512}
PREDICTOR_RANK=${PREDICTOR_RANK:-256}
PREDICTOR_NUM_HEADS=${PREDICTOR_NUM_HEADS:-4}
PREDICTOR_RANK_MLP_RATIO=${PREDICTOR_RANK_MLP_RATIO:-2}
PREDICTOR_USE_VISUAL_POSITION=${PREDICTOR_USE_VISUAL_POSITION:-true}
PREDICTOR_USE_TEXT_POSITION=${PREDICTOR_USE_TEXT_POSITION:-true}
PREDICTOR_KEEP_K=${PREDICTOR_KEEP_K:-64}

EXTRA_ARGS=()
if [[ -n "${MAX_SAMPLES}" ]]; then
  EXTRA_ARGS+=(--max_samples "$MAX_SAMPLES")
fi
if [[ -n "${SAMPLE_RATE}" ]]; then
  EXTRA_ARGS+=(--sample_rate "$SAMPLE_RATE")
fi
if [[ -n "${TEACHER_TARGET_LOG_DIR}" ]]; then
  EXTRA_ARGS+=(--teacher_target_log_dir "$TEACHER_TARGET_LOG_DIR")
fi
if [[ -n "${DATA_PATH}" ]]; then
  EXTRA_ARGS+=(--data_path "$DATA_PATH")
fi
if [[ -n "${IMAGE_FOLDER}" ]]; then
  EXTRA_ARGS+=(--image_folder "$IMAGE_FOLDER")
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
  --model_max_length "$MODEL_MAX_LENGTH" \
  --image_aspect_ratio "$IMAGE_ASPECT_RATIO" \
  --keep_k "$PREDICTOR_KEEP_K" \
  --teacher_layer "$TEACHER_LAYER" \
  --predictor_hidden_size "$PREDICTOR_HIDDEN_SIZE" \
  --predictor_rank "$PREDICTOR_RANK" \
  --predictor_num_heads "$PREDICTOR_NUM_HEADS" \
  --predictor_rank_mlp_ratio "$PREDICTOR_RANK_MLP_RATIO" \
  --predictor_use_visual_position "$PREDICTOR_USE_VISUAL_POSITION" \
  --predictor_use_text_position "$PREDICTOR_USE_TEXT_POSITION" \
  --rank_loss_weight "$RANK_LOSS_WEIGHT" \
  --js_loss_weight "$JS_LOSS_WEIGHT" \
  --ce_loss_weight "$CE_LOSS_WEIGHT" \
  --enable_scale "$ENABLE_SCALE" \
  --enable_rss "$ENABLE_RSS" \
  --topk_hinge_margin 1.0 \
  --kd_temperature 1.0 \
  --budgeted_soft_topk_iters 16 \
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
