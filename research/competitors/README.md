# Competitor Analysis

Each PR is analyzed once and stored as `pr_XXX.md`. Before analyzing a PR, check if the file exists.

## Index (sorted by BPB, lower = better)

| PR | BPB | Key Techniques | Validated? | File |
|----|-----|---------------|------------|------|
| #65 | 1.1630 | int6+int8 mixed quant, MLP 3x, sliding window stride=64 | Yes (3 seeds) | pr_065.md |
| #66 | 1.1652 | int6+zstd, MLP 3x, seq4096, sliding window stride=64 | Yes (3 seeds) | pr_066.md |
| #78 | 1.1858 | vocab 8192, NorMuon, w6e8 selective quant | Yes (1 run) | pr_078.md |
| #77 | 1.1928 | doc isolation, sliding window, LoRA TTT | Yes (4 seeds) | pr_077.md |
| #69 | pending | ternary QAT, depth recurrence (3×10=30L), TTT | NO RESULTS | pr_069.md |

## Cross-PR Pattern Analysis

### What actually moves the score (validated)

| Technique | Typical BPB gain | Used by |
|-----------|-----------------|---------|
| Sliding window eval | 0.033-0.034 | #65, #66, #77 |
| Wider MLP (3x) | 0.019-0.029 | #65, #66 |
| int6 quantization (size enabler) | near-zero direct, enables bigger model | #65, #66 |
| Vocab 8192 | ~0.025-0.030 | #78 |
| Document isolation | 0.011 | #77 |
| Lower LR (quant-friendly training) | indirect | #65, #66 |
| NorMuon | ~0.005-0.010 | #78 |
| LoRA TTT | 0.003 (unoptimized) | #77 |

### What nobody has combined yet

- Vocab 8192 (#78) + sliding window (#65/#66) — #78 doesn't use sliding window
- Ternary QAT (#69) + anything — no results yet
- int6 STE training (#65) + larger vocab (#78)
- TTT optimized beyond rank-8 single-step
- Depth recurrence + any validated technique
