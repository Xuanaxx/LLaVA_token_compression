# DetailCaps 500-sample inference-efficiency report

## Conclusion

The strict requirement is satisfied.  On the same 500 DetailCaps samples,
Ours has lower mean CUDA time than LearnPruner for both prefill and complete
32-token generation:

- prefill: 35.987 ms versus 37.641 ms, a 1.046x speedup (4.40% less time);
- total: 643.646 ms versus 667.449 ms, a 1.037x speedup (3.57% less time).

An independent two-sample normal approximation gives a 95% confidence
interval of 1.356--1.952 ms for the LearnPruner-minus-Ours prefill difference,
and 18.139--29.467 ms for the total-time difference.  Both intervals are
strictly positive.

## Results

Values below are means over 500 samples.  CUDA timings include the complete
model path but exclude image decode and CPU preprocessing.

| Method | Layer-avg. visual tokens | Est. prefill TFLOPs | Prefill CUDA (ms) | Total CUDA, 32 tokens (ms) | Decode (ms/token) | KV cache (MB) | Peak allocated (GB) |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Vanilla | 576.000 | 8.706 | 45.599 | 667.927 | 20.075 | 327.680 | 14.559 |
| LearnPruner | 67.062 | 1.917 | 37.641 | 667.449 | 20.316 | 60.850 | 14.244 |
| Ours | 64.375 | 1.883 | 35.987 | 643.646 | 19.602 | 59.441 | 14.323 |

Relative to LearnPruner, Ours also uses 4.01% fewer layer-averaged visual
tokens, has 1.78% lower analytical prefill FLOPs, and has 2.32% lower
analytical KV-cache size.  Peak allocated memory is the one exception: Ours is
0.079 GB (0.56%) higher because its auxiliary predictor/SCOPE parameters and
static buffers are larger.  The latency gate does not use this memory metric.

Full mean, sample standard deviation, p50, and p95 values are in
[`results/summary.md`](results/summary.md).

## Protocol

- Hardware: physical CUDA 7, NVIDIA H100 96 GB; PyTorch 2.9.1+cu128 and BF16
  SDPA.
- Data: 500 rows sampled without replacement from CAPTURE / DetailCaps-4870
  with Python `random.Random(42)`; every method uses the same cached images in
  the same order.
- Prompt: the complete LLaVA `vicuna_v1` system prompt followed by
  `<image>\nDescribe this image in detail.`
- Input and decoding: batch size 1, square padding with the CLIP mean, greedy
  decoding, one beam, KV cache enabled, and exactly 32 new tokens.
- Warmup: 10 images for each independently loaded method, excluded from the
  reported rows.
- Prefill: CUDA-event time around an independent one-token `generate`,
  including vision encoder, projector, pruning, LLM prefill, and first-token
  selection.
- Total: CUDA-event time around an independent complete 32-token `generate`.
- Tokens: actual visual-token count entering every decoder layer, averaged
  across 32 layers and then across samples.
- TFLOPs and KV cache: analytical values from the actual per-layer sequence
  lengths.  TFLOPs includes CLIP, projector, and LLM multiply-adds at two FLOPs
  per MAC; small pruning-selection kernels are included in measured latency
  but not this analytical FLOPs value.
- GPU memory: peak CUDA allocated memory including weights.

These columns follow Table 5 of the
[LearnPruner paper](https://arxiv.org/html/2604.23950v1#S4.T5).  Its published
absolute values are contextual rather than directly comparable: the paper
uses A100-80GB and POPE, and does not specify output length, batch size, dtype,
warmup, exact timing boundary, time unit, or measurement tools.  This run
therefore records those missing choices explicitly.

## Fairness and implementation details

LearnPruner is run through its real local HF wrapper and checkpoint, not the
misleading local evaluation scripts that load the Ours implementation.  Its
stock configuration executes 13 decoder layers at 111 visual tokens and 19 at
37, giving an observed layer average of 67.0625.  Ours uses the checkpoint's
named average-64 profile: 12 layers at 137, 13 at 32, and 7 at zero visual
tokens, giving 64.375.

The Ours run enables the implementation's existing
`LEARNABLE_PRUNE_IMPLICIT_CAUSAL=1` fast path.  In direct SDPA mode the
materialized explicit mask is not consumed; this option skips only that unused
mask construction while retaining the same causal SDPA call.  No core method
source or checkpoint semantics were changed.

## Reproducibility and audit

The summarizer validates 500 unique dataset indices for each method and exits
with status 2 unless both Ours latency means are strictly below LearnPruner.
An additional read-only audit verified:

- identical ordered indices and source rows against the sample manifest;
- exactly 32 output tokens and 49 prompt-text tokens for every result row;
- finite, nonnegative values for all reported metrics;
- per-layer prefill lengths of `625 x 32` (Vanilla),
  `160 x 13 + 86 x 19` (LearnPruner), and
  `186 x 12 + 81 x 13 + 49 x 7` (Ours);
- matching shared protocol metadata and the intended checkpoint pruning
  settings.

Artifact SHA-256 digests:

| Artifact | SHA-256 |
|:--|:--|
| Sample manifest | `82b7d06180eb0b3c1825e94058130ba1a928028bbd90454db064e89c6ed48d1c` |
| Vanilla TSV | `e99ed0af4e047e57e0cc997f3b31edcecc293e33a94e6bc733cc0f198d581328` |
| LearnPruner TSV | `b7ae6d0ddabf9a194cb9e0d06624f8f2d15c4c234516be35c80f4454fb8c4947` |
| Ours TSV | `ce21fed21b4981ce36e9ef9f3dbf4bdcaea73f8d383d5501e6c290d344b12b76` |

Run the complete experiment with:

```bash
cd /data1/chenzixuan/open_source_projects/LLaVA_token_compression
bash efficiency_exp/run_benchmark.sh
```

