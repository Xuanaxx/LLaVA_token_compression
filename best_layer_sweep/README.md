# LLaVA best scoring-layer sweep

This directory owns the complete official-checkpoint layer-sweep workflow:

1. Expand multimodal inputs and run the unpruned sequence through layers before the selected scoring layer.
2. Compute query-conditioned visual-token importance at that layer.
3. Select `TOPK` visual tokens and restart prefill from layer 0 with the compact sequence.
4. Evaluate exactly `SAMPLE_LIMIT` samples per requested task and rank all scoring layers.

Supported local official checkpoints are resolved under `MODEL_ROOT`:

- `llava-v1.5-7b`
- `llava-v1.5-13b`
- `llava-next` (`llava-v1.6-vicuna-7b`)

Full default sweep (all three models, GQA, 500 samples per layer):

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  bash best_layer_sweep/run_llava_layer_sweep.sh
```

Multiple tasks and a non-default Top-K:

```bash
CUDA_VISIBLE_DEVICES=0,1 TASKS=gqa,textvqa_val,pope TOPK=128 \
  bash best_layer_sweep/run_llava_layer_sweep.sh
```

Multiple scoring layers can run concurrently on the same GPU. `PROCESS_NUM` is
the total number of concurrent layer jobs; `PROCESSES_PER_GPU` is a convenient
alternative that computes `PROCESS_NUM` from the visible GPU count:

```bash
# Four independent scoring-layer jobs on GPU 3.
CUDA_VISIBLE_DEVICES=3 PROCESS_NUM=4 \
  bash best_layer_sweep/run_llava_layer_sweep.sh

# Two jobs per GPU, four jobs total.
CUDA_VISIBLE_DEVICES=0,1 PROCESSES_PER_GPU=2 \
  bash best_layer_sweep/run_llava_layer_sweep.sh
```

Each job loads a complete model replica. Four 7B/LLaVA-NeXT replicas generally
fit a 96 GB H100, while four 13B replicas may not; lower `PROCESS_NUM` if needed.

For a targeted or smoke run, select models/layers and override the sample count:

```bash
MODELS=llava-next LAYERS="8 16" TASKS=gqa TOPK=96 SAMPLE_LIMIT=1 \
  CUDA_VISIBLE_DEVICES=0,1 bash best_layer_sweep/run_llava_layer_sweep.sh
```

`best_layers.json` reports the best layer for every model, each task's best layer,
and the cross-task ranking. Cross-task ranking averages each task score relative to
that task's best observed score so metrics with different numeric scales remain comparable.
