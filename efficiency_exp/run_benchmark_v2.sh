#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data1/chenzixuan/uv_env/.tokencompression/bin/python}"
FORMAL_PHYSICAL_GPU=2
PHYSICAL_GPU="${PHYSICAL_GPU:-${FORMAL_PHYSICAL_GPU}}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
FORMAL_SEED=42
SEED="${SEED:-${FORMAL_SEED}}"
FORMAL_NUM_SAMPLES=500
FORMAL_REPETITIONS=4
FORMAL_WARMUP=20
WARMUP="${WARMUP:-${FORMAL_WARMUP}}"
FORMAL_MAX_NEW_TOKENS=32
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-${FORMAL_MAX_NEW_TOKENS}}"
REPETITIONS="${REPETITIONS:-4}"
RUN_ID="${RUN_ID:-formal_seed${SEED}_n${NUM_SAMPLES}_t${MAX_NEW_TOKENS}}"
RESULT_ROOT="${RESULT_ROOT:-${SCRIPT_DIR}/results}"
SAMPLE_DIR="${SAMPLE_DIR:-${SCRIPT_DIR}/samples}"
FORMAL_LEARNPRUNER_CHECKPOINT="/data1/chenzixuan/train_output/learnpruner_llava15_7b_paper_aligned_smoke_cuda2"
FORMAL_LOCKED_SM_CLOCK_MHZ=1980
LOCKED_SM_CLOCK_MHZ="${LOCKED_SM_CLOCK_MHZ:-}"
DEFAULT_CLOCK_LOCK_MODE=unlocked
CLOCK_LOCK_MODE="${CLOCK_LOCK_MODE-${DEFAULT_CLOCK_LOCK_MODE}}"

case "${CLOCK_LOCK_MODE}" in
  unlocked)
    if [[ -n "${LOCKED_SM_CLOCK_MHZ}" ]]; then
      echo "CLOCK_LOCK_MODE=unlocked requires LOCKED_SM_CLOCK_MHZ to be empty." >&2
      exit 1
    fi
    PYTHON_CLOCK_MODE=monitored_unlocked
    ;;
  managed|external)
    LOCKED_SM_CLOCK_MHZ="${LOCKED_SM_CLOCK_MHZ:-${FORMAL_LOCKED_SM_CLOCK_MHZ}}"
    if [[ "${LOCKED_SM_CLOCK_MHZ}" != "${FORMAL_LOCKED_SM_CLOCK_MHZ}" ]]; then
      echo "Locked modes require LOCKED_SM_CLOCK_MHZ=${FORMAL_LOCKED_SM_CLOCK_MHZ}." >&2
      exit 1
    fi
    if [[ "${CLOCK_LOCK_MODE}" == "managed" ]]; then
      PYTHON_CLOCK_MODE=managed_locked
    else
      PYTHON_CLOCK_MODE=external_locked
    fi
    ;;
  *)
    echo "CLOCK_LOCK_MODE must be unlocked, managed, or external." >&2
    exit 1
    ;;
esac
if [[ "${MAX_NEW_TOKENS}" != "${FORMAL_MAX_NEW_TOKENS}" ]]; then
  echo "Formal protocol v2 requires MAX_NEW_TOKENS=${FORMAL_MAX_NEW_TOKENS}." >&2
  exit 1
fi
if [[ "${NUM_SAMPLES}" != "${FORMAL_NUM_SAMPLES}" ]]; then
  echo "Formal protocol v2 requires NUM_SAMPLES=${FORMAL_NUM_SAMPLES}." >&2
  exit 1
fi
if [[ "${REPETITIONS}" != "${FORMAL_REPETITIONS}" ]]; then
  echo "Formal protocol v2 requires REPETITIONS=${FORMAL_REPETITIONS}." >&2
  exit 1
fi
if [[ "${PHYSICAL_GPU}" != "${FORMAL_PHYSICAL_GPU}" ]]; then
  echo "Formal protocol v2 requires PHYSICAL_GPU=${FORMAL_PHYSICAL_GPU}." >&2
  exit 1
fi
if [[ "${SEED}" != "${FORMAL_SEED}" ]]; then
  echo "Formal protocol v2 requires SEED=${FORMAL_SEED}." >&2
  exit 1
fi
if [[ "${WARMUP}" != "${FORMAL_WARMUP}" ]]; then
  echo "Formal protocol v2 requires WARMUP=${FORMAL_WARMUP}." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${RESULT_ROOT}" "${SAMPLE_DIR}"
CONTROLLER_DIR="${RESULT_ROOT}/efficiency-v2/cuda${PHYSICAL_GPU}/controller/${RUN_ID}"
mkdir -p "${CONTROLLER_DIR}"

GPU_LOCK_PATH="${GPU_LOCK_PATH:-/tmp/llava_efficiency_v2_cuda${PHYSICAL_GPU}.lock}"
exec 9>"${GPU_LOCK_PATH}"
if ! flock -n 9; then
  echo "Another protocol-v2 benchmark holds the CUDA ${PHYSICAL_GPU} lock." >&2
  exit 1
fi

clock_was_locked=0
cleanup() {
  if [[ "${CLOCK_LOCK_MODE}" == "managed" && "${clock_was_locked}" == "1" ]]; then
    nvidia-smi -i "${PHYSICAL_GPU}" -rgc >/dev/null || true
  fi
}
trap cleanup EXIT

if [[ "${CLOCK_LOCK_MODE}" == "managed" ]]; then
  nvidia-smi -i "${PHYSICAL_GPU}" \
    -lgc "${LOCKED_SM_CLOCK_MHZ},${LOCKED_SM_CLOCK_MHZ}"
  clock_was_locked=1
else
  echo "Clock mode ${CLOCK_LOCK_MODE}: runner will not lock or reset the GPU clock."
fi

# Every mode records and validates positive NVML clock observations. Locked
# compatibility modes additionally require the fixed 1980 MHz observation.
clock_args=(--clock-mode "${PYTHON_CLOCK_MODE}")
if [[ -n "${LOCKED_SM_CLOCK_MHZ}" ]]; then
  clock_args+=(--locked-sm-clock-mhz "${LOCKED_SM_CLOCK_MHZ}")
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_detailcaps_sample.py" \
  --output-dir "${SAMPLE_DIR}" \
  --num-samples 500 \
  --seed "${SEED}"

SAMPLE_PARQUET="${SAMPLE_DIR}/detailcaps_seed${SEED}_n500.parquet"
nvidia-smi -i "${PHYSICAL_GPU}" \
  --query-gpu=index,name,uuid,driver_version,memory.total,power.limit,clocks.max.sm \
  --format=csv,noheader >"${CONTROLLER_DIR}/gpu_info.txt"

# Four near-balanced permutations give both Ours-vs-Vanilla and
# Ours-vs-LearnPruner a 2:2 process-order split. Position exposure differs by
# at most one occurrence for every method.
for (( repetition=0; repetition<REPETITIONS; repetition++ )); do
  case $(( repetition % 4 )) in
    0) PERFORMANCE_METHODS=(vanilla learnpruner ours) ;;
    1) PERFORMANCE_METHODS=(vanilla ours learnpruner) ;;
    2) PERFORMANCE_METHODS=(learnpruner ours vanilla) ;;
    3) PERFORMANCE_METHODS=(ours vanilla learnpruner) ;;
  esac
  for (( position=0; position<${#PERFORMANCE_METHODS[@]}; position++ )); do
    method="${PERFORMANCE_METHODS[position]}"
    extra_args=()
    if [[ "${method}" == "ours" ]]; then
      # Graph construction/capture occurs outside timing. The Python harness
      # requires one prefill plus all 31 decode replay callback pairs in every
      # diagnostic row and two cache-hit prefill replays per adjacent pair.
      extra_args+=(
        --ours-implicit-causal
        --ours-fixed-length-greedy on
        --ours-triton-rms on
        --ours-cuda-graph-prefill on
        --ours-cuda-graph-decode on
        --ours-static-kv-decode off
      )
    fi
    log="${CONTROLLER_DIR}/rep$(printf '%02d' "${repetition}")_${position}_${method}.log"
    echo "Starting v2 repetition=${repetition} position=${position} method=${method}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/benchmark_efficiency_v2.py" \
      --method "${method}" \
      --sample-parquet "${SAMPLE_PARQUET}" \
      --result-root "${RESULT_ROOT}" \
      --run-id "${RUN_ID}" \
      --repetition "${repetition}" \
      --num-samples "${NUM_SAMPLES}" \
      --seed "${SEED}" \
      --warmup "${WARMUP}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --learnpruner-checkpoint "${FORMAL_LEARNPRUNER_CHECKPOINT}" \
      --physical-gpu-id "${PHYSICAL_GPU}" \
      --expected-cuda-visible-devices "${PHYSICAL_GPU}" \
      --resume \
      "${clock_args[@]}" \
      "${extra_args[@]}" 2>&1 | tee -a "${log}"
  done

done

if [[ "${DEFER_SUMMARY:-false}" == "true" ]]; then
  echo "Raw v2 runs completed; summary deferred for post-run code integration."
else
  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_results_v2.py" \
    --result-root "${RESULT_ROOT}" \
    --physical-gpu-id "${PHYSICAL_GPU}" \
    --run-id "${RUN_ID}" \
    --expected-n "${NUM_SAMPLES}" \
    --expected-repetitions "${REPETITIONS}" \
    --expected-output-tokens "${MAX_NEW_TOKENS}" \
    --clock-mode "${PYTHON_CLOCK_MODE}"
fi
