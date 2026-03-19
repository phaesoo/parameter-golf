# Experiment Plan

## Current State (2026-03-18 baseline)

| Metric | Value |
|--------|-------|
| Baseline BPB (post-quant) | 1.2244 |
| Baseline BPB (pre-quant) | 1.2172 |
| 4-hour BPB (post-quant) | 1.2074 |
| 4-hour BPB (pre-quant) | 1.1749 |
| Quantization loss (10min) | 0.0072 |
| Quantization loss (4hr) | 0.0325 |
| Model parameters | ~17M |
| Compressed size | 15.86MB / 16MB |
| Step speed | 43.5ms |
| Steps in 10 min | ~13,780 |
| Token throughput | ~7.2B tokens |

### Baseline Learning Curve Key Points

- step 0: 4.098 BPB
- step 1000: 1.381 BPB (rapid drop)
- step 6000: 1.265 BPB (slowing)
- step 12000: 1.242 BPB (just before warmdown)
- step 13780: 1.217 BPB (end, pre-quant)
- Warmdown phase (12000→13780): 0.025 BPB drop — extremely effective
- 4-hour run at step 20000: 1.231 → only 0.014 better than 10min. Severe diminishing returns.

---

## Decision Tree (top-down)

Decision order matters. Upper decisions determine the design space for everything below.

```
Level 0: Weight Precision
├── int8 (current) → 17M params in 16MB
├── int4 → ~34M params
├── ternary (1.58bit) → ~85M params
└── binary (1bit) → ~128M params
    │
    ▼
Level 1: Tokenizer / Vocab Size
├── 1024 (current) → ~2 bytes/token
├── 4096 → ~3.5 bytes/token
├── 8192 → ~4.2 bytes/token
└── 16384 → ~4.8 bytes/token
    │
    ▼
Level 2: Evaluation Strategy
├── Standard (current, independent 1024-token chunks)
├── Longer eval seq_len (2048/4096)
├── KV-cache streaming (continuous processing)
├── Sliding window
└── TTT (Test-Time Training) — legality check required
    │
    ▼
Level 3: Architecture
├── depth/width ratio
├── depth recurrence (weight sharing)
├── attention design (GQA, MQA, linear)
├── MLP design (relu^2, SwiGLU, MoE)
└── skip connection design
    │
    ▼
Level 4: Training Hyperparameters
├── LR schedule
├── batch size
├── warmdown/warmup
├── optimizer config
└── KD (knowledge distillation)
    │
    ▼
Level 5: Compression Details
├── QAT
├── GPTQ
├── compression codec (zlib vs zstd)
└── pruning
```

---

## Phase 0: Lock Decision Points (Day 1-3)

Goal: Lock Levels 0-2. No hyperparameter tuning.

### Day 1 — Weight Precision

**Experiment 0.1: Ternary training PoC**
- Convert baseline architecture to ternary (-1, 0, 1) weights
- Train with STE (Straight-Through Estimator)
- 1xH100, same step count, compare BPB
- Decision criterion: is ternary large model > int8 small model at same 16MB?

**Experiment 0.2: int4 training PoC**
- Fake int4 quantization during training
- Compare quality/size tradeoff vs ternary

**Decision branches:**
- Ternary clearly wins → Level 0 = ternary, all subsequent experiments use ternary
- int4 is the sweet spot → Level 0 = int4
- Unclear difference → keep int8, compete on other axes

### Day 2 — Vocab Size

**Experiment 0.3: Train new tokenizers**
- Train SentencePiece tokenizers at vocab 4096, 8192
- Measure actual tokens_per_byte on FineWeb

**Experiment 0.4: Vocab-vs-BPB comparison**
- Using precision from Day 1, train with different vocab sizes only
- Measure embedding size increase vs tokens_per_byte improvement
- Decision criterion: which vocab X minimizes (cross_entropy × tokens_per_byte)?

**Decision branches:**
- Optimal vocab locked → all subsequent experiments use this vocab

### Day 3 — Evaluation Strategy

**Experiment 0.5: Longer eval seq_len**
- Test existing baseline model (no changes) at eval seq_len 2048, 4096
- Check RoPE extrapolation quality

**Experiment 0.6: KV-cache streaming eval**
- Implement KV-cache version of eval_val
- Measure BPB difference: independent chunks vs continuous streaming

**Experiment 0.7: TTT legality check**
- Ask on Discord: "Is test-time training (adapting model weights during evaluation using the validation data) allowed?"
- If possible, build a simple PoC

**Decision branches:**
- TTT allowed → model design must optimize for fast adaptation
- TTT disallowed → focus on pure model quality + eval tricks

---

## Phase 1: Architecture Search (Day 4-7)

Proceeds after Phase 0 locks (precision, vocab, eval strategy).

### Experiments (adjusted based on Phase 0 results)

**Experiment 1.1: Depth/Width sweep**
- Given locked precision + vocab, find (layers, dim) combinations that fit 16MB
- Candidates: (12, X), (15, X), (18, X), (24, X)
- X is back-calculated from 16MB budget

**Experiment 1.2: Depth recurrence**
- Sweep N unique blocks × M repetitions
- (3 blocks × 10 reps = 30 layers), (4 × 8 = 32), (6 × 5 = 30), etc.
- Find optimal unique block count vs total depth

**Experiment 1.3: MLP design**
- relu^2 vs SwiGLU vs GeGLU
- MoE (2-4 experts, top-1)

**Experiment 1.4: Attention design**
- KV head count optimization (1, 2, 4)
- Head dim optimization

---

## Phase 2: Training Optimization (Day 8-10)

Proceeds after architecture is locked.

**Experiment 2.1: LR schedule sweep**
- cosine, WSD, cyclic
- Peak LR + decay rate combinations

**Experiment 2.2: Batch size optimization**
- 524K, 1M, 2M tokens/step

**Experiment 2.3: Knowledge Distillation**
- Pre-compute soft targets from large model (offline)
- KD loss weight optimization

**Experiment 2.4: Warmdown optimization**
- Duration, start point, schedule shape

---

## Phase 3: Compression + Integration (Day 11-14)

**Experiment 3.1: QAT or GPTQ**
**Experiment 3.2: Compression codec comparison**
**Experiment 3.3: Full pipeline integration**
**Experiment 3.4: 8xH100 full run × 5+ seeds**
**Experiment 3.5: PR submission**

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Ternary training unstable | int4 as fallback |
| Large vocab BPB improvement below expectations | Measure quickly on Day 2, fallback to 1024 |
| TTT ruled illegal | Confirm on Day 3, pivot to other eval tricks |
| 8xH100 cost | Phase 0-1 on 1xH100, 8x only for final validation |
| Combinatorial explosion of experiments | Strictly follow decision tree order, lock upper levels first |
