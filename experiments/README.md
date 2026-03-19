# Experiments

Each experiment is a self-contained `train_gpt.py` that tests ONE specific idea.
Baseline is untouched. All experiments can be batch-tested via `run_all.sh`.

## Naming Convention

`eXX_<short_name>/` where XX follows decision tree priority.

## Directory Structure

```
experiments/
├── run_all.sh              — batch runner: iterates all experiments, logs results
├── compare.py              — parse logs, output BPB comparison table
│
├── e01_depth_recurrence/   — [Level 3] share weights across layers for more effective depth
├── e02_deeper_narrow/      — [Level 3] more layers, smaller dim (e.g., 15L x 360d)
├── e03_swiglu/             — [Level 3] SwiGLU MLP instead of relu^2
├── e04_fewer_kv_heads/     — [Level 3] reduce KV heads from 4 to 2
├── e05_longer_eval/        — [Level 2] eval at seq_len 2048/4096 (no training change)
├── e06_kv_cache_eval/      — [Level 2] streaming eval with KV cache
├── e07_ternary/            — [Level 0] ternary weights (-1, 0, 1) with STE
├── e08_int4/               — [Level 0] int4 quantization-aware training
└── e09_larger_vocab/       — [Level 1] vocab 4096/8192 (needs tokenizer prep)
```

## Design Rules

1. Each experiment modifies exactly ONE axis — no combo experiments until individual effects are measured
2. Every `train_gpt.py` is fully self-contained and runnable standalone
3. Experiments inherit baseline defaults — only the changed parts differ
4. Each folder has a `README.md` explaining: what changed, why, expected impact

## How to Test

```bash
# Single experiment (short run for comparison)
cd experiments/e01_depth_recurrence
ITERATIONS=3000 MAX_WALLCLOCK_SECONDS=0 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py

# All experiments
bash experiments/run_all.sh
```

## Experiment Details

### e01_depth_recurrence [Level 3: Architecture]
- **Idea**: Use N unique transformer blocks, repeat them M times (e.g., 3 unique × 6 reps = 18 effective layers)
- **Why**: Same parameter count, much more effective depth. README of the challenge explicitly suggests this.
- **Expected**: 0.005-0.015 BPB improvement
- **Risk**: Training instability with many repetitions

### e02_deeper_narrow [Level 3: Architecture]
- **Idea**: Trade width for depth. E.g., 15 layers × dim 360 instead of 9 × 512
- **Why**: Deeper models tend to compress better at fixed param count
- **Expected**: 0.005-0.020 BPB improvement
- **Risk**: Narrower dim may hurt attention quality

### e03_swiglu [Level 3: Architecture]
- **Idea**: Replace relu^2 MLP with SwiGLU (3 projections, gated activation)
- **Why**: SwiGLU is consistently better in modern LLMs at same param count
- **Expected**: 0.003-0.008 BPB improvement
- **Risk**: Extra projection costs params, may not fit same budget

### e04_fewer_kv_heads [Level 3: Architecture]
- **Idea**: Reduce KV heads from 4 to 2 (or 1). Reinvest saved params elsewhere.
- **Why**: Small model may not need 4 KV heads. Freed params can add depth.
- **Expected**: 0.002-0.005 BPB improvement
- **Risk**: May hurt attention quality

### e05_longer_eval [Level 2: Eval Strategy]
- **Idea**: Train at seq_len 1024, evaluate at 2048 or 4096
- **Why**: Longer context = better predictions. Rules explicitly allow any eval seq_len.
- **Expected**: 0.005-0.020 BPB improvement
- **Risk**: RoPE extrapolation may degrade beyond 2x

### e06_kv_cache_eval [Level 2: Eval Strategy]
- **Idea**: Stream validation set continuously with KV cache instead of independent chunks
- **Why**: Tokens at chunk boundaries currently get no context
- **Expected**: 0.010-0.025 BPB improvement
- **Risk**: Implementation complexity, memory management

### e07_ternary [Level 0: Precision]
- **Idea**: Train with ternary weights (-1, 0, 1) using STE. ~5x more params in 16MB.
- **Why**: 85M ternary params could dramatically outperform 17M int8 params
- **Expected**: 0.040-0.080 BPB improvement (if it works)
- **Risk**: Training instability, quality per parameter much lower

### e08_int4 [Level 0: Precision]
- **Idea**: Train with fake int4 quantization (16 levels per weight)
- **Why**: 2x params vs int8 with better quality-per-param than ternary
- **Expected**: 0.020-0.040 BPB improvement
- **Risk**: int4 QAT adds training complexity

### e09_larger_vocab [Level 1: Tokenizer]
- **Idea**: Increase vocab to 4096 or 8192
- **Why**: More bytes per token = lower BPB even at same cross-entropy
- **Expected**: 0.010-0.030 BPB improvement
- **Risk**: Larger embedding eats param budget. Needs tokenizer training first.
- **Blocker**: Requires new tokenizer + re-tokenized dataset

## Priority for First Batch

Run these first (no external dependencies, test decision tree top-down):
1. e05_longer_eval — zero training change, instant measurement
2. e07_ternary — answers Level 0 question
3. e01_depth_recurrence — highest expected ROI in architecture
4. e02_deeper_narrow — simple architecture change
