#!/usr/bin/env bash
set -euo pipefail

if command -v conda >/dev/null 2>&1; then
  conda deactivate >/dev/null 2>&1 || true
fi
source /data1/chenzixuan/uv_env/.tokencompression/bin/activate
cd /data1/chenzixuan/open_source_projects/LLaVA_token_compression/learnable_pruner_lightweight

GPU_IDS=${GPU_IDS:-0,1}
IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPU_IDS"
declare -A GPU_ID_SEEN=()
for index in "${!GPU_ID_ARRAY[@]}"; do
  gpu_id="${GPU_ID_ARRAY[$index]//[[:space:]]/}"
  if [[ -z "$gpu_id" ]]; then
    echo "GPU_IDS contains an empty device entry: $GPU_IDS" >&2
    exit 2
  fi
  if [[ -n "${GPU_ID_SEEN[$gpu_id]:-}" ]]; then
    echo "GPU_IDS contains duplicate device '$gpu_id': $GPU_IDS" >&2
    exit 2
  fi
  GPU_ID_SEEN[$gpu_id]=1
  GPU_ID_ARRAY[$index]="$gpu_id"
done
GPU_COUNT=${#GPU_ID_ARRAY[@]}
NUM_GPUS=${NUM_GPUS:-$GPU_COUNT}
if [[ ! "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]] || (( NUM_GPUS != GPU_COUNT )); then
  echo "NUM_GPUS=$NUM_GPUS must equal the number of unique GPU_IDS entries ($GPU_COUNT): $GPU_IDS" >&2
  exit 2
fi
printf -v GPU_IDS '%s,' "${GPU_ID_ARRAY[@]}"
GPU_IDS=${GPU_IDS%,}
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-"expandable_segments:True"}
export WANDB_PROJECT=${WANDB_PROJECT:-"llava_next_learnable_prune_lightweight_fixed_layer"}
export LEARNABLE_PRUNE_SCOPE_GRAPH_CACHE_SIZE=${LEARNABLE_PRUNE_SCOPE_GRAPH_CACHE_SIZE:-4}
export LEARNABLE_PRUNE_BATCHED_SCORE_MAX_PADDING=${LEARNABLE_PRUNE_BATCHED_SCORE_MAX_PADDING:-1.5}
# Anyres batches fall back to row scoring when padding amplification is high;
# keep their batched-path perturbation temporary conservative as well.
export LEARNABLE_PRUNE_SCORE_TEMP_MIB=${LEARNABLE_PRUNE_SCORE_TEMP_MIB:-128}
export WANDB_API_KEY="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"
# Provide WANDB_API_KEY in the environment if needed.

BATCH_SIZE=${BATCH_SIZE:-128}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-16}
if [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ || ! "$PER_DEVICE_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH_SIZE and PER_DEVICE_BATCH_SIZE must be positive integers." >&2
  exit 2
fi
MICROBATCH_SIZE=$((PER_DEVICE_BATCH_SIZE * NUM_GPUS))
if [[ -z "${GRADIENT_ACCUMULATION_STEPS:-}" ]]; then
  if (( BATCH_SIZE % MICROBATCH_SIZE != 0 )); then
    echo "BATCH_SIZE=$BATCH_SIZE must be divisible by PER_DEVICE_BATCH_SIZE*NUM_GPUS=$MICROBATCH_SIZE." >&2
    exit 2
  fi
  GRADIENT_ACCUMULATION_STEPS=$((BATCH_SIZE / MICROBATCH_SIZE))
elif [[ ! "$GRADIENT_ACCUMULATION_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "GRADIENT_ACCUMULATION_STEPS must be a positive integer." >&2
  exit 2
fi
MAX_STEPS=${MAX_STEPS:--1}
MAX_SAMPLES=${MAX_SAMPLES:-}
SAMPLE_RATE=${SAMPLE_RATE:-0.2}
LOGGING_STEPS=${LOGGING_STEPS:-10}
BF16=${BF16:-true}
FP16=${FP16:-false}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
DATALOADER_PREFETCH_FACTOR=${DATALOADER_PREFETCH_FACTOR:-2}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-true}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-0}
echo "[launcher] GPU_IDS=$GPU_IDS NUM_GPUS=$NUM_GPUS gradient_accumulation=$GRADIENT_ACCUMULATION_STEPS"

RANK_LOSS_WEIGHT=${RANK_LOSS_WEIGHT:-0.5}
JS_LOSS_WEIGHT=${JS_LOSS_WEIGHT:-${KL_LOSS_WEIGHT:-6.0}}
CE_LOSS_WEIGHT=${CE_LOSS_WEIGHT:-0.5}
ENABLE_SCALE=${ENABLE_SCALE:-true}
ENABLE_RSS=${ENABLE_RSS:-false}

# BUDGET_PROFILES is the sole pruning-budget source. One complete profile is
# selected per microbatch by a balanced, deterministically shuffled cycle.
# Authoritative deployment values come from
# MLLM_Token_Compression_Workdir/test_scripts/h100/
# llava_next_learnable_prune_lightweight_scope_finalwipe_query_anchors.sh.
# Format: avg:topk:scope:mid_layer:mid_target:final_layer. Predictor TopK is
# 80% of the corresponding SCOPE target for every default budget profile.
BUDGET_PROFILES=${BUDGET_PROFILES:-"160:272:340:12:80:25;320:544:680:12:160:25;640:1088:1360:12:320:25"}
BUDGET_SCHEDULE_SEED=${BUDGET_SCHEDULE_SEED:-42}

MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"/data1/chenzixuan/model/liuhaotian/llava-v1.6-vicuna-7b"}
DATA_DIR=${DATA_DIR:-"/data2/czx/data/llava_1_5_mix665k_full"}
DATA_PATH=${DATA_PATH:-}
IMAGE_FOLDER=${IMAGE_FOLDER:-}
OUTPUT_ROOT=${OUTPUT_ROOT:-"/data1/chenzixuan/train_output"}
TEACHER_LAYER=${TEACHER_LAYER:-18}
RUN_NAME=${RUN_NAME:-"official_llava_next_7b_learnable_prune_lightweight_top80pctscope_multibudget160_320_640_layer${TEACHER_LAYER}_sample0.2"}
TEACHER_TARGET_LOG_DIR=${TEACHER_TARGET_LOG_DIR:-}
TEACHER_TARGET_LOG_TO_CONSOLE=${TEACHER_TARGET_LOG_TO_CONSOLE:-false}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
IMAGE_ASPECT_RATIO=${IMAGE_ASPECT_RATIO:-anyres}

PREDICTOR_HIDDEN_SIZE=${PREDICTOR_HIDDEN_SIZE:-512}
PREDICTOR_RANK=${PREDICTOR_RANK:-256}
PREDICTOR_NUM_HEADS=${PREDICTOR_NUM_HEADS:-4}
PREDICTOR_RANK_MLP_RATIO=${PREDICTOR_RANK_MLP_RATIO:-2}
PREDICTOR_USE_VISUAL_POSITION=${PREDICTOR_USE_VISUAL_POSITION:-true}
PREDICTOR_USE_TEXT_POSITION=${PREDICTOR_USE_TEXT_POSITION:-true}

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

MIXED_PRECISION=no
if [[ "${BF16}" == "1" || "${BF16,,}" == "true" ]]; then
  MIXED_PRECISION=bf16
elif [[ "${FP16}" == "1" || "${FP16,,}" == "true" ]]; then
  MIXED_PRECISION=fp16
fi
ACCELERATE_ARGS=(--num_processes "$NUM_GPUS" --num_machines 1 --gpu_ids "$GPU_IDS" --mixed_precision "$MIXED_PRECISION" --dynamo_backend no)
if (( NUM_GPUS > 1 )); then
  ACCELERATE_ARGS=(--multi_gpu "${ACCELERATE_ARGS[@]}")
fi
ACCELERATE_ARGS+=(--main_process_port "$MAIN_PROCESS_PORT")

accelerate launch "${ACCELERATE_ARGS[@]}" train_learnable_pruner.py \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_ROOT/$RUN_NAME" \
  --run_name "$RUN_NAME" \
  --wandb_project "$WANDB_PROJECT" \
  --model_max_length "$MODEL_MAX_LENGTH" \
  --image_aspect_ratio "$IMAGE_ASPECT_RATIO" \
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
  --budget_profiles "$BUDGET_PROFILES" \
  --budget_schedule_seed "$BUDGET_SCHEDULE_SEED" \
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
  --dataloader_prefetch_factor "$DATALOADER_PREFETCH_FACTOR" \
  --attn_implementation "$ATTN_IMPLEMENTATION" \
  --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
  --ddp_find_unused_parameters false \
  "${EXTRA_ARGS[@]}"
