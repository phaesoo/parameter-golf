# Experiment Plan v2 (Post-Research)

## Strategic Insight

Public knowledge (papers, competitor PRs, NanoGPT speedrun) is available to everyone.
The real edge is **optimizing known techniques to this specific challenge's constraints**.
The gap between "paper says it works" and "it works optimally at 16MB/10min/BPB" is where we win.

Every experiment below is about finding the optimal operating point within our constraints,
not about discovering new techniques.

## Challenge Constraints (shapes the optimization space)

- 16,000,000 bytes total (code + compressed model)
- 10 min training on 8xH100 (~13,780 steps at current speed)
- 10 min eval on 8xH100 (separate budget)
- BPB = bits_per_token × tokens_per_byte (tokenizer-agnostic)
- FineWeb validation set (fixed 50K documents)

---

## Track A: Quantization Precision (biggest lever for model capacity)

**Goal**: Find the optimal bits-per-weight for 16MB.

Paper insight: ParetoQ says 2-bit is sweet spot. BitNet Reloaded says ternary effective capacity ≈ 50% of FP16.

**Experiments** (each is a full train_gpt.py):

| ID | Config | Effective Params | Question |
|----|--------|-----------------|----------|
| A1 | int8 baseline (control) | ~17M | Baseline reference |
| A2 | int6 + wider MLP 3x (proven) | ~22M | Reproduce #65's approach |
| A3 | int4 QAT (STE) | ~34M | Is 2x params worth the quality-per-param drop? |
| A4 | 2-bit QAT (STE) | ~50M | ParetoQ sweet spot — does it hold at our scale? |
| A5 | Ternary QAT (STE) | ~85M | Maximum params — does it hold at our scale? |

Key optimization variables PER experiment:
- Model shape (depth × width) that maximizes quality at given param count
- STE training schedule (when to activate QAT, LR reduction)
- Which tensors get which precision (embedding always higher, per #65)

**Decision**: Pick the precision that gives best BPB at 3000 steps on 1xGPU.

---

## Track B: Tokenizer / Vocab Size (changes the BPB formula itself)

**Goal**: Find optimal vocab size for our model scale.

Paper insight: Deletang et al. says small models compress better with simpler tokenizers.
Competitor insight: PR #78 uses 8K vocab and gets 1.186 WITHOUT sliding window.

**Experiments**:

| ID | Vocab | bytes/token (est.) | Embedding cost | Question |
|----|-------|-------------------|----------------|----------|
| B1 | 256 (byte-level) | 1.0 | 256×dim = tiny | Does Deletang's finding hold on FineWeb? |
| B2 | 1024 (baseline) | ~2.0 | 1024×dim | Control |
| B3 | 4096 | ~3.5 | 4096×dim | Middle ground |
| B4 | 8192 | ~4.2 | 8192×dim | Reproduce #78's approach |

Must use the SAME model capacity (after embedding cost) for fair comparison.
Must use precision from Track A winner.

**Decision**: Pick the vocab that minimizes BPB at 3000 steps.

---

## Track C: Architecture (maximize quality per parameter)

**Goal**: Best model shape for the locked precision + vocab.

**Experiments**:

| ID | Idea | Source | Question |
|----|------|--------|----------|
| C1 | Depth recurrence (N unique × M reps) | PR #69, Relaxed Recursive Transformers | Optimal N and M? Middle layers best per slowrun. |
| C2 | Depth/width sweep | General | Optimal (layers, dim) at our param count? |
| C3 | Value embeddings | NanoGPT speedrun (major contributor) | How much at vocab 1024? |
| C4 | Partial key offset | NanoGPT speedrun | Free induction heads? |
| C5 | SwiGLU vs relu^2 | Standard | Which wins at our scale? |

---

## Track D: Eval Optimization (free BPB from eval budget)

**Goal**: Extract maximum BPB from the 10-min eval budget.

These are independent of training and apply on top of ANY model.

| ID | Idea | Source | Expected |
|----|------|--------|----------|
| D1 | Sliding window (stride=64) | PRs #65, #66 (proven 0.034) | Implement and verify |
| D2 | Document isolation | PR #77 (proven 0.011) | Implement and verify |
| D3 | Optimized TTT (multi-step, V-only LoRA, per-layer LR) | Papers: qTTT, LoRA-TTT, ScaleNet | 0.005-0.015? |
| D4 | EMA weights at eval | NanoGPT slowrun | Free, trivial |

D1+D2 are proven and must be implemented regardless.
D3 is the high-upside experiment — optimize within eval budget.
D4 is free.

---

## Track E: Training Optimization (last, after architecture is locked)

| ID | Idea | Source |
|----|------|--------|
| E1 | NorMuon optimizer | PR #78, NanoGPT speedrun |
| E2 | Batch size schedule | NanoGPT speedrun |
| E3 | Lower LR for quant-friendly weights | PR #65 finding |
| E4 | Warmdown optimization | General |
| E5 | Multi-token prediction | NanoGPT speedrun |

---

## Execution Order

```
Week 1: Track A (precision) → Track B (vocab)
         Lock: bits-per-weight, vocab size
         These determine the ENTIRE design space.

Week 2: Track C (architecture)
         Lock: model shape, recurrence config
         Track D1+D2 (proven eval tricks) — implement in parallel

Week 3: Track D3 (optimized TTT) + Track E (training optimization)
         These are fine-tuning on top of locked architecture.

Week 4: Integration + final runs
         Combine best of each track.
         8xH100 full runs × 5+ seeds.
         Submit before 4/30.
```

## What We DON'T Do

- Re-validate techniques competitors already proved (sliding window, int6, wider MLP)
- Chase NanoGPT speedrun tricks that will become common knowledge
- Implement anything we can't test on GPU within a week
- Over-plan — the plan updates as experiments produce results
