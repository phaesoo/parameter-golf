# Test-Time Training (TTT) Research

## Current State in Parameter Golf

PR #77 gets **0.003 BPB** with: rank-8 LoRA, single Adam step, Q+V+lm_head, ~10% of eval budget.

## Why 0.003 Is So Low (and How to Fix It)

| Problem | Evidence | Fix |
|---------|----------|-----|
| Single gradient step | qTTT uses 32 steps; ARC TTT uses 2 epochs | 10-32 steps |
| Uniform LR across layers | ScaleNet: layers differ by orders of magnitude | Per-layer LR |
| Wrong matrices adapted | LoRA-TTT: V-only best; qTTT: Q-only best | Test V-only, Q-only |
| No meta-learning | TTT-E2E: train initialization to be good at TTT | Train with TTT in loop |
| Score-then-train limits signal | Each chunk only benefits from previous chunks | Multi-pass: adapt first, score second |
| Only ~10% of eval budget used | 9x more compute available | Use full 10 min |
| Adam overhead | Doubles memory per adapted param | SGD+momentum or Muon (LaCT finding) |

## Key Papers

### TTT-Linear/TTT-MLP (Yu Sun et al., 2024)
- Hidden state IS a small model, updated by gradient descent on self-supervised loss
- 125M: ppl 15.91 → 11.09 with TTT layers
- Architecture component, not eval-time bolt-on
- Source: https://arxiv.org/abs/2407.04620

### qTTT (ICLR 2026) — Most directly relevant
- Single prefill pass, then 32 gradient steps on random context spans (k=128 tokens)
- Backprop **only through W_Q** (avoids corrupting K/V cache)
- LR: 1e-5 to 1e-6 optimal
- Qwen3-4B: +12.6 points on LongBench-v2
- Wall-clock: essentially free at 8K context
- Source: https://arxiv.org/abs/2512.13898

### TTT-E2E (Yu Sun, Dec 2024) — Key insight: meta-learning
- Mini-batch: 1K tokens, sliding window 8K
- Adapts last 1/4 of transformer blocks
- **Meta-learning at training time** optimizes initialization for TTT
- Improves loss by ~0.018 cross-entropy on top of full attention
- Source: https://arxiv.org/abs/2512.23675

### ScaleNet / Dynamic Layer-Wise TTA (Feb 2026)
- Hypernetwork predicts per-layer, per-step LR multipliers
- Base LR: 1e-2, LoRA rank 4
- 70B Llama: NLL 2.21 → 1.70 (dynamic) vs 11.50 (fixed LR, destroyed)
- "Adjacent layers can differ by orders of magnitude"
- Source: https://arxiv.org/abs/2602.09719

### LoRA-TTT Ablations (ICML Workshops 2025)
- **Rank**: 4-16 with proper scale. Higher rank needs lower scale.
- **Layers**: Last 2 of 12 is optimal. All layers is WORSE.
- **Matrices**: V projections best (not Q or K)
- **Steps**: Single AdamW step at LR=0.001
- Source: https://arxiv.org/abs/2502.02069

### LaCT: Test-Time Training Done Right (May 2025)
- Previous TTT achieves <5% GPU utilization
- LaCT: chunks of 2048-1M tokens → up to 70% utilization
- **Muon optimizer > Adam** for TTT
- State-to-parameter ratio: 40%
- Source: https://arxiv.org/abs/2505.23884

## Estimated Ceiling for Parameter Golf

| Approach | Estimated BPB Gain | Difficulty |
|----------|-------------------|------------|
| Optimized multi-step TTT (no training changes) | 0.005-0.010 | Medium |
| Meta-learned initialization (change training) | 0.015-0.030 | High |
| For reference: 4hr vs 10min same-arch gap | 0.017 | N/A |

## Optimal TTT Recipe for Parameter Golf

1. Multi-step LoRA: 10-32 steps per document, rank 16
2. Adapt V projections in last 2-3 layers (LoRA-TTT finding)
3. Per-layer LR (ScaleNet finding) or at minimum aggressive sweep
4. LR range: 1e-4 to 1e-2
5. Muon or SGD+momentum instead of Adam (LaCT finding)
6. Multi-pass: first pass adapts, second pass scores
7. Batched per-document (PR #77 approach, scale up to full eval budget)
8. If modifying training: add inner TTT loop (TTT-E2E meta-learning)
