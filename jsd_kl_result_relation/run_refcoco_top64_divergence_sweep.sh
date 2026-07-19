#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LLAVA_REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LMMS_EVAL_DIR="${LMMS_EVAL_DIR:-/data1/chenzixuan/open_source_projects/lmms-eval}"
MODEL_ROOT="${MODEL_ROOT:-/data1/chenzixuan/model/liuhaotian}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/data1/chenzixuan/uv_env/.tokencompression/bin/activate}"

# Every layer uses the same deterministic 500-example sample for each dataset.
MODELS="${MODELS:-llava-v1.5-7b}"
TASKS="${TASKS:-refcoco,refcoco_plus}"
DATASET_SPLIT="${DATASET_SPLIT:-val}"
TOPK="${TOPK:-64}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-500}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
CAPTION_METRICS="${CAPTION_METRICS:-Bleu_4,Bleu_3,Bleu_2,Bleu_1,METEOR,ROUGE_L,CIDEr}"
CONV_TEMPLATE="${CONV_TEMPLATE:-vicuna_v1}"
DTYPE="${DTYPE:-bfloat16}"
LAYERS="${LAYERS:-}"
# Split the deterministic sample manifest across workers.  Each worker retains
# all requested layers, so no multimodal prefix work is duplicated.
SAMPLE_SHARDS_PER_TASK="${SAMPLE_SHARDS_PER_TASK:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PROCESS_NUM="${PROCESS_NUM:-4}"
PROCESSES_PER_GPU="${PROCESSES_PER_GPU:-}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/topk_${TOPK}}"
CORRELATION_PATH="${CORRELATION_PATH:-${OUTPUT_ROOT}/correlations.json}"
LAYER_TABLE_PATH="${LAYER_TABLE_PATH:-${OUTPUT_ROOT}/layer_metrics.tsv}"
DRY_RUN="${DRY_RUN:-0}"

[[ -f "${VENV_ACTIVATE}" ]] || { echo "Missing environment: ${VENV_ACTIVATE}" >&2; exit 1; }
[[ -d "${LMMS_EVAL_DIR}" ]] || { echo "Missing lmms-eval: ${LMMS_EVAL_DIR}" >&2; exit 1; }
[[ -d "${MODEL_ROOT}" ]] || { echo "Missing official model root: ${MODEL_ROOT}" >&2; exit 1; }
[[ "${TOPK}" =~ ^[1-9][0-9]*$ ]] || { echo "TOPK must be a positive integer, got '${TOPK}'" >&2; exit 1; }
[[ "${SAMPLE_LIMIT}" =~ ^[1-9][0-9]*$ ]] || { echo "SAMPLE_LIMIT must be a positive integer." >&2; exit 1; }
[[ "${SAMPLE_SEED}" =~ ^[0-9]+$ ]] || { echo "SAMPLE_SEED must be a non-negative integer." >&2; exit 1; }
[[ "${MAX_NEW_TOKENS}" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_NEW_TOKENS must be a positive integer." >&2; exit 1; }
[[ "${SAMPLE_SHARDS_PER_TASK}" =~ ^[1-9][0-9]*$ ]] || {
    echo "SAMPLE_SHARDS_PER_TASK must be a positive integer." >&2
    exit 1
}

TASKS="$(printf '%s' "${TASKS}" | tr '[:space:]' ',' | sed -e 's/,,*/,/g' -e 's/^,//' -e 's/,$//')"
CUDA_VISIBLE_DEVICES="$(printf '%s' "${CUDA_VISIBLE_DEVICES}" | tr -d '[:space:]')"
IFS=',' read -r -a task_array <<< "${TASKS}"
IFS=',' read -r -a gpu_array <<< "${CUDA_VISIBLE_DEVICES}"
[[ ${#task_array[@]} -gt 0 ]] || { echo "TASKS must contain at least one task." >&2; exit 1; }
[[ ${#gpu_array[@]} -gt 0 ]] || { echo "CUDA_VISIBLE_DEVICES must contain at least one GPU." >&2; exit 1; }
for task in "${task_array[@]}"; do
    [[ "${task}" == "refcoco" || "${task}" == "refcoco_plus" ]] || {
        echo "Unsupported task '${task}'; choose refcoco and/or refcoco_plus." >&2
        exit 1
    }
done
if [[ -n "${PROCESSES_PER_GPU}" ]]; then
    [[ "${PROCESSES_PER_GPU}" =~ ^[1-9][0-9]*$ ]] || {
        echo "PROCESSES_PER_GPU must be a positive integer." >&2
        exit 1
    }
    PROCESS_NUM=$((PROCESSES_PER_GPU * ${#gpu_array[@]}))
fi
[[ "${PROCESS_NUM}" =~ ^[1-9][0-9]*$ ]] || { echo "PROCESS_NUM must be a positive integer." >&2; exit 1; }

source "${VENV_ACTIVATE}"
export PYTHONPATH="${LLAVA_REPO}:${LMMS_EVAL_DIR}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/data1/chenzixuan/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-warning}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

mapfile -t model_specs < <(
    python -m best_layer_sweep.model_registry --model-root "${MODEL_ROOT}" --models "${MODELS}"
)
[[ ${#model_specs[@]} -gt 0 ]] || { echo "No models resolved from MODELS=${MODELS}." >&2; exit 1; }

# Resolve layers before launching anything. Workers receive disjoint strided
# subsets of the same global deterministic sample list and write shard files.
sweep_specs=()
finalize_specs=()
for spec in "${model_specs[@]}"; do
    IFS=$'\t' read -r model_name model_path num_layers <<< "${spec}"
    if [[ -n "${LAYERS}" ]]; then
        read -r -a requested_layers <<< "$(printf '%s' "${LAYERS}" | tr ',' ' ')"
    else
        mapfile -t requested_layers < <(seq 0 $((num_layers - 1)))
    fi
    [[ ${#requested_layers[@]} -gt 0 ]] || { echo "LAYERS must contain at least one layer." >&2; exit 1; }

    layer_array=()
    declare -A seen_layers=()
    for layer in "${requested_layers[@]}"; do
        [[ "${layer}" =~ ^[0-9]+$ ]] || { echo "Invalid layer '${layer}'." >&2; exit 1; }
        if (( layer >= num_layers )); then
            echo "Layer ${layer} is outside ${model_name}'s range [0, $((num_layers - 1))]." >&2
            exit 1
        fi
        if [[ -z "${seen_layers[${layer}]+x}" ]]; then
            layer_array+=("${layer}")
            seen_layers[${layer}]=1
        fi
    done
    unset seen_layers

    layer_csv="$(IFS=,; printf '%s' "${layer_array[*]}")"
    shard_count=${SAMPLE_SHARDS_PER_TASK}
    if (( shard_count > SAMPLE_LIMIT )); then
        shard_count=${SAMPLE_LIMIT}
    fi
    for task in "${task_array[@]}"; do
        for ((shard_idx = 0; shard_idx < shard_count; shard_idx++)); do
            shard_number=$((shard_idx + 1))
            sweep_specs+=("${model_name}"$'\t'"${model_path}"$'\t'"${task}"$'\t'"${layer_csv}"$'\t'"${shard_number}"$'\t'"${shard_count}")
        done
        if (( shard_count > 1 )); then
            finalize_specs+=("${model_name}"$'\t'"${task}"$'\t'"${layer_csv}"$'\t'"${shard_count}")
        fi
    done
done

TOTAL_JOBS=${#sweep_specs[@]}
EFFECTIVE_PROCESS_NUM=${PROCESS_NUM}
if (( EFFECTIVE_PROCESS_NUM > TOTAL_JOBS )); then
    EFFECTIVE_PROCESS_NUM=${TOTAL_JOBS}
fi
MAX_PROCESSES_PER_GPU=$(((EFFECTIVE_PROCESS_NUM + ${#gpu_array[@]} - 1) / ${#gpu_array[@]}))
echo "Sweep concurrency: ${EFFECTIVE_PROCESS_NUM}/${TOTAL_JOBS} job(s) across ${#gpu_array[@]} GPU(s), up to ${MAX_PROCESSES_PER_GPU} per GPU."
if (( MAX_PROCESSES_PER_GPU > 1 )); then
    echo "Warning: each process loads a model and keeps paired full/pruned KV caches; lower PROCESS_NUM on OOM." >&2
fi
mkdir -p "${OUTPUT_ROOT}"

pids=()
active=0
next_gpu_idx=0

wait_for_batch() {
    local status=0
    local pid
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            status=1
        fi
    done
    pids=()
    active=0
    return "${status}"
}

launch_sweep() {
    local model_name="$1"
    local model_path="$2"
    local task="$3"
    local layers="$4"
    local gpu_id="$5"
    local shard_number="$6"
    local shard_count="$7"
    local task_dir="${OUTPUT_ROOT}/${model_name}/${task}"
    local log_dir="${OUTPUT_ROOT}/${model_name}/${task}/logs"
    local log_path="${log_dir}/topk_${TOPK}_sample_shard_${shard_number}_of_${shard_count}.log"
    mkdir -p "${task_dir}" "${log_dir}"
    {
        echo "model=${model_name}"
        echo "model_path=${model_path}"
        echo "task=${task}"
        echo "dataset_split=${DATASET_SPLIT}"
        echo "sample_limit=${SAMPLE_LIMIT}"
        echo "sample_seed=${SAMPLE_SEED}"
        echo "scoring_layers=${layers}"
        echo "sample_shard=${shard_number}/${shard_count}"
        echo "topk=${TOPK}"
        echo "max_new_tokens=${MAX_NEW_TOKENS}"
        echo "cuda_visible_devices=${gpu_id}"
        echo "output=${task_dir}"
    } > "${log_path}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "DRY_RUN model=${model_name} task=${task} sample_shard=${shard_number}/${shard_count} layers=${layers} topk=${TOPK} gpu=${gpu_id}" | tee -a "${log_path}"
        return 0
    fi
    CUDA_VISIBLE_DEVICES="${gpu_id}" python -m jsd_kl_result_relation.evaluate_layer \
        --model-path "${model_path}" \
        --model-name "${model_name}" \
        --task "${task}" \
        --dataset-split "${DATASET_SPLIT}" \
        --sample-size "${SAMPLE_LIMIT}" \
        --sample-seed "${SAMPLE_SEED}" \
        --sample-shard-index "$((shard_number - 1))" \
        --sample-shard-count "${shard_count}" \
        --scoring-layers "${layers}" \
        --topk "${TOPK}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --caption-metrics "${CAPTION_METRICS}" \
        --conv-template "${CONV_TEMPLATE}" \
        --dtype "${DTYPE}" \
        --output-root "${task_dir}" \
        >> "${log_path}" 2>&1
}

for sweep_spec in "${sweep_specs[@]}"; do
    IFS=$'\t' read -r model_name model_path task layer_csv shard_number shard_count <<< "${sweep_spec}"
    gpu_id="${gpu_array[${next_gpu_idx}]}"
    next_gpu_idx=$(((next_gpu_idx + 1) % ${#gpu_array[@]}))
    launch_sweep "${model_name}" "${model_path}" "${task}" "${layer_csv}" "${gpu_id}" "${shard_number}" "${shard_count}" &
    pids+=("$!")
    active=$((active + 1))
    if (( active >= EFFECTIVE_PROCESS_NUM )); then
        wait_for_batch
    fi
done
wait_for_batch

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN complete; no evaluation or correlation summary was executed."
    exit 0
fi
for finalize_spec in "${finalize_specs[@]}"; do
    IFS=$'\t' read -r model_name task layer_csv shard_count <<< "${finalize_spec}"
    python -m jsd_kl_result_relation.finalize_shards \
        --output-root "${OUTPUT_ROOT}/${model_name}/${task}" \
        --scoring-layers "${layer_csv}" \
        --shard-count "${shard_count}" \
        --expected-samples "${SAMPLE_LIMIT}"
done
if rg -n "Traceback|CUDA out of memory|Error .* generating" "${OUTPUT_ROOT}" --glob '*.log'; then
    echo "One or more sweep jobs failed; inspect logs under ${OUTPUT_ROOT}." >&2
    exit 1
fi

python -m jsd_kl_result_relation.correlate \
    --base-root "${OUTPUT_ROOT}" \
    --output "${CORRELATION_PATH}" \
    --tsv "${LAYER_TABLE_PATH}" \
    --expected-samples "${SAMPLE_LIMIT}"
echo "Saved correlation analysis to ${CORRELATION_PATH}"
echo "Saved per-layer metrics to ${LAYER_TABLE_PATH}"
