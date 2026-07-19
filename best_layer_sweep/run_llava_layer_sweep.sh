#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LLAVA_REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LMMS_EVAL_DIR="${LMMS_EVAL_DIR:-/data1/chenzixuan/open_source_projects/lmms-eval}"
MODEL_ROOT="${MODEL_ROOT:-/data1/chenzixuan/model/liuhaotian}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/data1/chenzixuan/uv_env/.tokencompression/bin/activate}"

# Sweep controls. TOPK is deliberately a hyperparameter rather than a model constant.
MODELS="${MODELS:-llava-next}"
TASKS="${TASKS:-gqa}"
TOPK="${TOPK:-160}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-500}"
LAYERS="${LAYERS:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
PROCESS_NUM="${PROCESS_NUM:-4}"
PROCESSES_PER_GPU="${PROCESSES_PER_GPU:-}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/topk_${TOPK}}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/best_layers.json}"
RANKING_PATH="${RANKING_PATH:-${OUTPUT_ROOT}/layer_ranking.tsv}"
DRY_RUN="${DRY_RUN:-0}"

[[ -f "${VENV_ACTIVATE}" ]] || { echo "Missing environment: ${VENV_ACTIVATE}" >&2; exit 1; }
[[ -d "${LMMS_EVAL_DIR}" ]] || { echo "Missing lmms-eval: ${LMMS_EVAL_DIR}" >&2; exit 1; }
[[ -d "${MODEL_ROOT}" ]] || { echo "Missing official model root: ${MODEL_ROOT}" >&2; exit 1; }
[[ "${TOPK}" =~ ^[1-9][0-9]*$ ]] || { echo "TOPK must be a positive integer, got '${TOPK}'" >&2; exit 1; }
[[ "${SAMPLE_LIMIT}" =~ ^[1-9][0-9]*$ ]] || { echo "SAMPLE_LIMIT must be a positive integer, got '${SAMPLE_LIMIT}'" >&2; exit 1; }

# Normalize comma/space lists once; no model-specific size or deprecated scoring flags remain.
TASKS="$(printf '%s' "${TASKS}" | tr '[:space:]' ',' | sed -e 's/,,*/,/g' -e 's/^,//' -e 's/,$//')"
CUDA_VISIBLE_DEVICES="$(printf '%s' "${CUDA_VISIBLE_DEVICES}" | tr -d '[:space:]')"
[[ -n "${TASKS}" ]] || { echo "TASKS must contain at least one task." >&2; exit 1; }
[[ -n "${CUDA_VISIBLE_DEVICES}" ]] || { echo "CUDA_VISIBLE_DEVICES must contain at least one GPU." >&2; exit 1; }
IFS=',' read -r -a task_array <<< "${TASKS}"
IFS=',' read -r -a gpu_array <<< "${CUDA_VISIBLE_DEVICES}"
if [[ -n "${PROCESSES_PER_GPU}" ]]; then
    [[ "${PROCESSES_PER_GPU}" =~ ^[1-9][0-9]*$ ]] || {
        echo "PROCESSES_PER_GPU must be a positive integer." >&2
        exit 1
    }
    PROCESS_NUM=$((PROCESSES_PER_GPU * ${#gpu_array[@]}))
fi
[[ "${PROCESS_NUM}" =~ ^[1-9][0-9]*$ ]] || { echo "PROCESS_NUM must be a positive integer." >&2; exit 1; }
MAX_PROCESSES_PER_GPU=$(((PROCESS_NUM + ${#gpu_array[@]} - 1) / ${#gpu_array[@]}))
echo "Sweep concurrency: ${PROCESS_NUM} processes across ${#gpu_array[@]} GPU(s), up to ${MAX_PROCESSES_PER_GPU} process(es) per GPU."
if (( MAX_PROCESSES_PER_GPU > 1 )); then
    echo "Warning: every process loads a full model replica; reduce PROCESS_NUM if GPU memory is insufficient." >&2
fi

source "${VENV_ACTIVATE}"
export PYTHONPATH="${LLAVA_REPO}:${LMMS_EVAL_DIR}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/data1/chenzixuan/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-warning}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

mapfile -t model_specs < <(
    python -m best_layer_sweep.model_registry --model-root "${MODEL_ROOT}" --models "${MODELS}"
)
[[ ${#model_specs[@]} -gt 0 ]] || { echo "No models resolved from MODELS=${MODELS}." >&2; exit 1; }
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

launch_layer() {
    local model_name="$1"
    local model_path="$2"
    local task="$3"
    local layer="$4"
    local gpu_id="$5"
    local task_dir="${OUTPUT_ROOT}/${model_name}/${task}"
    local layer_dir="${task_dir}/scoring_layer_${layer}"
    local log_dir="${task_dir}/logs"
    local log_path="${log_dir}/topk_${TOPK}_layer_${layer}.log"
    local model_args="pretrained=${model_path},best_layer_sweep_model=true,device_map=cuda,dtype=bfloat16,attn_implementation=eager,scoring_layer_idx=${layer},visual_token_target_count=${TOPK}"
    mkdir -p "${layer_dir}" "${log_dir}"
    {
        echo "model=${model_name}"
        echo "model_path=${model_path}"
        echo "task=${task}"
        echo "sample_limit=${SAMPLE_LIMIT}"
        echo "scoring_layer_idx=${layer}"
        echo "topk=${TOPK}"
        echo "process_num=${PROCESS_NUM}"
        echo "max_processes_per_gpu=${MAX_PROCESSES_PER_GPU}"
        echo "cuda_visible_devices=${gpu_id}"
        echo "output=${layer_dir}"
    } > "${log_path}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "DRY_RUN model=${model_name} task=${task} layer=${layer} topk=${TOPK} gpu=${gpu_id}" | tee -a "${log_path}"
        return 0
    fi
    CUDA_VISIBLE_DEVICES="${gpu_id}" accelerate launch \
        --num_processes=1 \
        --num_machines=1 \
        --mixed_precision=bf16 \
        --dynamo_backend=no \
        --main_process_port=0 \
        -m lmms_eval \
        --model llava \
        --model_args "${model_args}" \
        --tasks "${task}" \
        --limit "${SAMPLE_LIMIT}" \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix "official_${model_name}_topk_${TOPK}_layer_${layer}" \
        --output_path "${layer_dir}" \
        >> "${log_path}" 2>&1
}

for spec in "${model_specs[@]}"; do
    IFS=$'\t' read -r model_name model_path num_layers <<< "${spec}"
    if [[ -n "${LAYERS}" ]]; then
        read -r -a layer_array <<< "$(printf '%s' "${LAYERS}" | tr ',' ' ')"
    else
        mapfile -t layer_array < <(seq 0 $((num_layers - 1)))
    fi
    for layer in "${layer_array[@]}"; do
        [[ "${layer}" =~ ^[0-9]+$ ]] || { echo "Invalid layer '${layer}'." >&2; exit 1; }
        if (( layer >= num_layers )); then
            echo "Layer ${layer} is outside ${model_name}'s range [0, $((num_layers - 1))]." >&2
            exit 1
        fi
    done
    for task in "${task_array[@]}"; do
        for layer in "${layer_array[@]}"; do
            gpu_id="${gpu_array[${next_gpu_idx}]}"
            next_gpu_idx=$(((next_gpu_idx + 1) % ${#gpu_array[@]}))
            launch_layer "${model_name}" "${model_path}" "${task}" "${layer}" "${gpu_id}" &
            pids+=("$!")
            active=$((active + 1))
            if (( active >= PROCESS_NUM )); then
                wait_for_batch
            fi
        done
    done
done
wait_for_batch

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN complete; no evaluation or summary was executed."
    exit 0
fi
if rg -n "Error during evaluation|Traceback|CUDA out of memory|Error .* in generating" "${OUTPUT_ROOT}" --glob '*.log'; then
    echo "One or more sweep jobs failed; inspect logs under ${OUTPUT_ROOT}." >&2
    exit 1
fi

python -m best_layer_sweep.summarize \
    --base-root "${OUTPUT_ROOT}" \
    --output "${SUMMARY_PATH}" \
    --tsv "${RANKING_PATH}" \
    --expected-samples "${SAMPLE_LIMIT}"
echo "Saved best layers to ${SUMMARY_PATH}"
echo "Saved rankings to ${RANKING_PATH}"
