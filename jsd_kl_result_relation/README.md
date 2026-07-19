# RefCOCO / RefCOCO+ top-64 divergence–performance sweep

This experiment follows `best_layer_sweep/run_llava_layer_sweep.sh`: it runs the
unpruned multimodal prefix to a selected scoring layer, directly selects the 64
highest-scoring visual tokens, and restarts the complete language-model prefill
from layer 0 with that compact sequence.

The deterministic sample list for one model/task is divided into disjoint
strided process shards (four by default). Every worker retains all requested
scoring layers, so image encoding, multimodal expansion, the scoring prefix, and
the unpruned reference prefill remain shared across layers without repeating any
sample across workers. Shard JSONL files are merged in the original global
sample order before metrics are computed. This allows several model replicas to
use otherwise-idle H100 memory while preserving the single-process result files.
Decoding uses preallocated static KV caches, and divergence tensors stay on the
GPU until the example is complete.

For every model, dataset, and scoring layer, the runner deterministically samples
exactly 500 validation examples (seed 42 by default). It records one JSONL row per
example with:

- first-token JSD, `KL(full || pruned)`, and `KL(pruned || full)`;
- mean JSD and both KL directions over all generated-token positions;
- the pruned prediction and standard RefCOCO caption targets;
- original/retained visual-token and sequence lengths.

Generation is greedy from the pruned branch. At every position, both the unpruned
and pruned branches consume that same generated text prefix. Thus the measured
distribution change isolates visual-token removal instead of comparing two
already-diverged strings. The first position is the model distribution used to
select the first output token; the generation mean includes every predicted
position, including EOS when emitted.

Run the configured 32-layer model sweep on RefCOCO and RefCOCO+:

```bash
cd /data1/chenzixuan/open_source_projects/LLaVA_token_compression
CUDA_VISIBLE_DEVICES=0,1 PROCESSES_PER_GPU=4 \
  bash jsd_kl_result_relation/run_refcoco_top64_divergence_sweep.sh
```

A one-layer smoke run (still use a small sample only for debugging, not for the
final correlation result):

```bash
LAYERS=16 SAMPLE_LIMIT=2 MAX_NEW_TOKENS=4 PROCESS_NUM=1 \
CUDA_VISIBLE_DEVICES=0 \
  bash jsd_kl_result_relation/run_refcoco_top64_divergence_sweep.sh
```

Useful overrides are `MODELS`, `TASKS`, `LAYERS`, `TOPK`, `SAMPLE_LIMIT`,
`SAMPLE_SEED`, `MAX_NEW_TOKENS`, `SAMPLE_SHARDS_PER_TASK`, `PROCESS_NUM`,
`PROCESSES_PER_GPU`, and `OUTPUT_ROOT`. `TASKS` accepts `refcoco`,
`refcoco_plus`, or both. Existing complete JSONL rows are resumed; a changed
configuration or sample manifest is rejected rather than silently mixed into an
old run.

`SAMPLE_SHARDS_PER_TASK=4` produces four non-overlapping sample workers per task.
`PROCESS_NUM` limits concurrent workers globally; `PROCESSES_PER_GPU` sets it
from the visible GPU count. For example, the following runs four workers on one
H100:

```bash
CUDA_VISIBLE_DEVICES=1 SAMPLE_SHARDS_PER_TASK=4 PROCESS_NUM=4 \
  bash jsd_kl_result_relation/run_refcoco_top64_divergence_sweep.sh
```

Set `SAMPLE_SHARDS_PER_TASK=1 PROCESS_NUM=1` to recover the single-process
execution schedule. A completed worker logs current and peak CUDA
allocated/reserved memory, which can be used to choose safe concurrency. On the
local 96-GiB H100, the measured LLaVA-v1.5-7B peak was about 13.9 GiB per worker,
so four concurrent workers leave substantial headroom.

The default output is `outputs/topk_64/`:

```text
<model>/<task>/scoring_layer_<n>/records.jsonl
<model>/<task>/scoring_layer_<n>/sample_indices.json
<model>/<task>/scoring_layer_<n>/layer_summary.json
<model>/<task>/scoring_layer_<n>/records.shard_<i>_of_<count>.jsonl
<model>/<task>/scoring_layer_<n>/sample_indices.shard_<i>_of_<count>.json
layer_metrics.tsv
correlations.json
```

The shard files are resumable worker state. `records.jsonl`,
`sample_indices.json`, and `layer_summary.json` are deterministic canonical
outputs produced after all shards pass completeness and non-overlap checks.

`correlations.json` reports Pearson and Spearman correlation between every
layer-level divergence statistic and every pruned caption metric. It contains
separate RefCOCO and RefCOCO+ analyses plus a pooled analysis where both axes are
z-scored within each task first. A negative coefficient means that smaller
full-vs-pruned divergence is associated with better pruned performance.
