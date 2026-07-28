#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data1/chenzixuan/uv_env/.tokencompression/bin/python}"
PHYSICAL_GPU="${PHYSICAL_GPU:-7}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
SEED="${SEED:-42}"
WARMUP="${WARMUP:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/results}"
SAMPLE_DIR="${SAMPLE_DIR:-${SCRIPT_DIR}/samples}"

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${RESULT_DIR}" "${SAMPLE_DIR}"
exec 9>"/tmp/llava_efficiency_cuda${PHYSICAL_GPU}.lock"
if ! flock -n 9; then
  echo "Another efficiency benchmark holds the CUDA ${PHYSICAL_GPU} lock." >&2
  exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_detailcaps_sample.py" \
  --output-dir "${SAMPLE_DIR}" \
  --num-samples 500 \
  --seed "${SEED}"

SAMPLE_PARQUET="${SAMPLE_DIR}/detailcaps_seed${SEED}_n500.parquet"
nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=index,name,uuid,driver_version,memory.total \
  --format=csv,noheader >"${RESULT_DIR}/gpu_info.txt"

for method in vanilla learnpruner ours; do
  output="${RESULT_DIR}/${method}_seed${SEED}_n${NUM_SAMPLES}.tsv"
  extra_args=()
  if [[ "${method}" == "ours" ]]; then
    # Direct SDPA already ignores the materialized mask.  Skipping its
    # construction is semantics-equivalent and removes pure overhead.
    extra_args+=(--ours-implicit-causal)
  fi
  echo "Starting ${method}: ${output}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/benchmark_efficiency.py" \
    --method "${method}" \
    --sample-parquet "${SAMPLE_PARQUET}" \
    --output "${output}" \
    --num-samples "${NUM_SAMPLES}" \
    --seed "${SEED}" \
    --warmup "${WARMUP}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --expected-cuda-visible-devices "${PHYSICAL_GPU}" \
    --resume \
    "${extra_args[@]}" 2>&1 | tee "${RESULT_DIR}/${method}_seed${SEED}_n${NUM_SAMPLES}.log"
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_results.py" \
  --input-dir "${RESULT_DIR}" \
  --n "${NUM_SAMPLES}" \
  --seed "${SEED}"
