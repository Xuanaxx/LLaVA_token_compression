# DetailCaps inference-efficiency experiment

Protocol v2 is the formal inference benchmark for Vanilla LLaVA-1.5-7B,
LearnPruner, and Ours on one cached CAPTURE / DetailCaps-4870 sample. The older
`benchmark_efficiency.py`, `run_benchmark.sh`, and `REPORT.md` remain historical
v1 artifacts. Their independently timed prefill and total calls cannot satisfy
the v2 gate.

## Formal protocol

- Hardware: physical CUDA 2, exposed as the process's only visible GPU. The
  default `monitored_unlocked` policy requires no administrator privileges and
  records positive finite NVML SM-clock observations before model load and
  before/after every authoritative generation. Formal 500×4 summaries require
  each ranked arm's clock p90/p10 to be at most 1.10 and the largest/smallest
  cross-arm median ratio to be at most 1.05.
- Data: 500 rows selected once with seed 42, cached in Parquet, and consumed in
  the identical stored order by every arm.
- Input: the same full LLaVA `vicuna_v1` prompt, batch size 1, and square
  CLIP-mean image padding on both backends.
- Compute: BF16, SDPA, KV cache, greedy decoding, one beam, and exactly 32
  generated tokens (`min_new_tokens=max_new_tokens=32`).
- Repetitions: four complete repetitions. A cyclic 4x4 Latin square balances
  whole-process load order across the four ranked arms.
- Process warmup: 20 complete 32-token generations after each independent
  model load. Seed 42, warmup 20, and output length 32 are immutable formal
  constants; the 32-token admission artifact cannot authorize another length.
- Provenance: every TSV is bound to the sample, configuration, runtime,
  physical GPU UUID, sources, checkpoints, optimization audit, and command.
  Resume accepts only the exact configuration and validates every existing row.

The ranked arms form two implementation-matched comparisons:

1. `vanilla`: official LLaVA.
2. `ours`: the same official backend with the admitted steady-state
   optimizations below.
3. `vanilla_hf`: Transformers LLaVA.
4. `learnpruner`: the same Transformers backend with LearnPruner.

The formal paths are also immutable:

```text
Official LLaVA: /data1/chenzixuan/model/liuhaotian/llava-v1.5-7b
HF LLaVA:       /data1/chenzixuan/model/llava-hf/llava-1.5-7b-hf
Ours:           /data1/chenzixuan/train_output/official_llava_v1.5_7b_learnable_prune_lightweight_top80pctscope_multibudget64_128_192_layer18_sample0.2/checkpoint-500
LearnPruner:    /data1/chenzixuan/train_output/learnpruner_llava15_7b_paper_aligned_smoke_cuda2
```

The summarizer recomputes current benchmark, adapter, runner, summarizer,
method-modeling, base-config, and applicable pruning-config/predictor hashes.
Each finalized metadata file binds the complete timing TSV bytes by SHA-256;
`gate.json` records every TSV/metadata SHA and an ordered aggregate.

`ours_slow` is an additional semantic and engineering reference. It uses the
same official model, images, prompt, and pruning checkpoint, but disables the
fixed-length loop, implicit causal specialization, Triton RMS, and both CUDA
graphs. The runner passes `--no-ours-implicit-causal` explicitly rather than
relying on a parser default. It is excluded from performance ranking. Every
Ours-fast generated tail must equal its matching slow-reference tail.

The formal LearnPruner arm loads
`/data1/chenzixuan/train_output/learnpruner_llava15_7b_paper_aligned_smoke_cuda2`.
This is a **latency-only paper-aligned timing reconstruction**, not an accuracy
checkpoint. Its fixed 111/37 visual-token schedule is used only to measure
architecture latency. No accuracy conclusion is drawn from this checkpoint.

## Adjacent measurements and timing boundaries

Each formal row contains two adjacent, token-identical 32-token generations.
Their order alternates by repetition and sample parity.

- The authoritative generation has only outer CUDA events. No benchmark
  module hook or graph-boundary callback is registered. It supplies reported
  E2E CUDA/wall latency and total throughput.
- The diagnostic generation records vision, projector, prefill, TTFT, and all
  31 direct decode intervals. Its generated IDs and hash must equal the
  authoritative generation.

Immediately before an Ours-fast pair, one synchronized, untimed, no-hook
generation makes that sample's prefill and decode graph signatures resident.
Its generated-tail hash must equal both measured calls. Any graph construction,
capture, initialization, eviction, or signature change therefore occurs
outside timing; the pair gate requires exactly two subsequent prefill cache-hit
replays with one stable runner identity. The instrumented call must also emit
one ordered `prefill_start`/`prefill_end` pair with that runner and 31 ordered
decode start/end pairs. A miss, capture, fallback, replay failure, signature
mismatch, missing callback, or runner replacement invalidates the row.

The diagnostic boundaries are:

- Multimodal prefill: outer CUDA start through first-token logits and the cache
  needed by decode. This includes vision, projector, pruning, decoder prefill,
  final normalization, and the output head.
- LLM prefill on eager arms: decoder layer-0 input normalization through the
  first output-head completion.
- LLM prefill on Ours-fast: `prefill_start`, after vision/projector and the
  eager predictor/SCOPE selection, through `prefill_end`, after the three
  decoder graph segments, eager middle selection/final wipe, final RMS/head,
  and heterogeneous-cache assembly.
- TTFT: outer CUDA start through the start of decode step 1, after selecting
  and preparing the first generated token.
- Decode: start of decode step 1 through the end of the same generation.
- TPOT: the arithmetic mean of the 31 directly recorded decode intervals. The
  final interval ends at generation completion.

Decode is never computed as `total - independent_prefill` in the formal
benchmark. Per-step CUDA and wall intervals must sum back to their direct
diagnostic totals. E2E divided by 32 is not called TPOT.

The per-row diagnostic/authoritative E2E ratio measures observer overhead.
The summarizer rejects excessive overhead or matched-method asymmetry using
paired bootstrap intervals. Token and projector tracing occurs only in untimed
validation calls.

## Admitted Ours deployment

Formal Ours-fast explicitly uses:

```text
LEARNABLE_PRUNE_DIRECT_SDPA=1
LEARNABLE_PRUNE_CACHED_BOOL_MASK=0
LEARNABLE_PRUNE_PROFILE_INTERNAL=0
LEARNABLE_PRUNE_IMPLICIT_CAUSAL=1
LEARNABLE_PRUNE_FIXED_LENGTH_GREEDY=1
LEARNABLE_PRUNE_TRITON_RMS=1
LEARNABLE_PRUNE_CUDA_GRAPH_PREFILL=1
LEARNABLE_PRUNE_CUDA_GRAPH_PREFILL_CACHE_SIZE=2
LEARNABLE_PRUNE_CUDA_GRAPH_DECODE=1
LEARNABLE_PRUNE_STATIC_KV_DECODE=0
```

The exact Triton RMS specialization is restricted to contiguous BF16
`LlamaRMSNorm` tensors with batch 1, hidden size 4096, at least 16 sequence
rows, the pinned PyTorch git revision, no hooks, and no runtime disable.
Singleton decode remains on authoritative PyTorch RMS. An untimed two-token
probe requires optimized/reference token equality, observes multi-row Triton
launches, and proves that singleton calls are neither eligible nor launched.

The prefill graph covers three decoder segments. Predictor/SCOPE selection,
middle scoring/TopK/sort/packing, and final wipe retain their authoritative
eager implementations between replay boundaries. The decode graph covers the
31 singleton steps. Both graphs use private fixed-shape buffers internally;
this is distinct from the separately controlled
`LEARNABLE_PRUNE_STATIC_KV_DECODE` optimization, which remains off.

Admission is bound to the persisted 32-image artifact
`results/audits/ours_combined_gate_prefill_seed42_n32.json`. The benchmark
pins schema `ours-combined-gate-v2`, run ID
`20260728T125445.126514Z-pid919082`, and SHA-256
`c75315c634eae9c3c724b6ede6c57a774acc486ef24e974aeff5d4992b8d88e2`,
along with its sample/model/checkpoint/source/script digests, switches, output
hashes, and checks. The audit compares:

- slow semantic generation;
- fixed-length/implicit-causal/exact-RMS eager generation;
- the same path with prefill and decode CUDA graphs.

It requires exact one-token and 32-token outputs across all three arms, one
prefill plus 31 decode callback pairs, stable prefill/decode runners, distinct
image-dependent outputs, no mask materialization on the implicit path, no
Triton singleton launch, and zero capture, replay, hook, eager, cache, or
signature failure in measured cases.

Graph capture and initialization costs are reported by the admission artifact
but excluded from latency and transient peak-memory measurements. Persistent
graph allocations remain in baseline allocated memory. Results therefore
describe a steady-state optimized deployment, not cold-start latency, and
should not be attributed to token pruning alone.

For that pinned run, combined prefill/decode graph construction took
2229.472 ms by CUDA events (2229.516 ms wall). Within it, the three prefill
segments report 29.762 ms capture and 97.548 ms initialization wall time.
These costs are admission/cold-start metadata and are not formal benchmark
latencies.

## Analytical metrics

Estimated prefill FLOPs use the full CLIP patch grid, the projector token count
observed in the same diagnostic generation, and per-layer decoder lengths.
The projector term uses 576 tokens for Vanilla/Ours and 111 tokens for the
formal LearnPruner reconstruction. Prefill KV footprint is analytical rather
than a hardware measurement. NVML clock, power, temperature, and utilization
snapshots are collected outside timed regions.

## Hard gate

`summarize_results_v2.py` rejects old, mixed, tampered, or incomplete outputs
and exits with status 2 unless all requirements hold:

- Ours-fast and Ours-slow generated hashes match every sample/repetition.
- Every Ours pair passes the prefill/decode replay, residency, runner, and
  no-fallback checks described above.
- The paired-bootstrap 95% lower bound for official Vanilla/Ours multimodal
  prefill speedup is at least 1.5x.
- The corresponding TPOT and decode lower bounds are at least 1.2x.
- The authoritative uninstrumented E2E lower bound is at least 1.2x.
- Ours is strictly fastest in the raw end-to-end system ranking against
  official Vanilla, HF Vanilla, and LearnPruner on every primary latency and
  throughput metric.
- LearnPruner's confidence-bound speedup is above 1.0x versus matched HF
  Vanilla for multimodal prefill and authoritative E2E.
- Observer overhead is bounded and symmetric, while latency distributions pass
  stability and separated-regime checks. SM-clock distribution bounds are hard
  gates for the 500×4 formal run and non-blocking warnings for smaller
  diagnostic runs such as 20×1.

Confidence intervals use a paired dataset-index cluster bootstrap: an image is
the independent sampling unit, and all four repetitions of a sampled image
remain together. Repeating the same images therefore does not create
pseudoreplication or spuriously narrow the interval.

Only official Vanilla/Ours and HF Vanilla/LearnPruner are matched-backend
algorithm comparisons. “Ours fastest overall” comparisons against HF
Vanilla/LearnPruner are raw end-to-end system rankings across different
backends, not algorithmic speedup estimates.

## Run

```bash
cd /data1/chenzixuan/open_source_projects/LLaVA_token_compression
bash efficiency_exp/run_benchmark_v2.sh
```

The default `CLOCK_LOCK_MODE=unlocked` never invokes `nvidia-smi -lgc` or
`nvidia-smi -rgc`. It records the policy as `monitored_unlocked`, leaves
`locked_sm_clock_mhz` empty/`null`, and relies on the distribution checks above:

```bash
NUM_SAMPLES=20 REPETITIONS=1 \
  RUN_ID=pilot_seed42_n20_t32 bash efficiency_exp/run_benchmark_v2.sh
```

Compatibility modes remain available. `CLOCK_LOCK_MODE=managed` asks the runner
to apply and restore 1980 MHz; `CLOCK_LOCK_MODE=external` records an
administrator-applied 1980 MHz lock but never changes it. Both locked modes
require every recorded SM-clock observation to equal 1980 MHz.

Outputs are isolated under
`results/efficiency-v2/cuda2/<arm>/<config-id>/`. Summary tables and
`gate.json` are written under
`results/efficiency-v2/cuda2/summary/<run-id>/`.

The runner fixes `PHYSICAL_GPU=2`, `SEED=42`, `WARMUP=20`, and
`MAX_NEW_TOKENS=32`. `CLOCK_LOCK_MODE` accepts `unlocked`, `managed`, or
`external`; locked modes use `LOCKED_SM_CLOCK_MHZ=1980`, while unlocked mode
requires that variable to be empty. `NUM_SAMPLES` and `REPETITIONS` remain
adjustable so a 20×1 diagnostic run can precede the 500×4 formal run, and those
values and the exact clock policy are recorded in final provenance.
