# NanoGPT Speedrun/Slowrun Techniques

## Untried in Parameter Golf (ranked by estimated ROI)

### Tier 1: Trivial to implement, proven in speedrun

| Technique | Extra Params | Code Change | What It Does |
|-----------|-------------|-------------|-------------|
| **EMA of weights** | 0 | ~10 lines | Maintain running average during training, eval with averaged weights. Free BPB. |
| **Smear module** | ~0 (12-dim gate) | ~10 lines | Shift embeddings back 1 position + learned gate. Local bigram context without attention. |
| **Partial key offset** | 0 | ~5 lines | Shift keys forward by 1 for half of head dims. Enables single-layer induction heads. |
| **Asymmetric logit rescale** | 0 | ~3 lines | Tune softcap (30→15?) or make asymmetric for pos/neg logits. |
| **Batch size schedule** | 0 | ~15 lines | Small batches early (more updates/token), large batches late (stable convergence). |

### Tier 2: Moderate effort, proven significant in speedrun

| Technique | Extra Params | What It Does |
|-----------|-------------|-------------|
| **Value embeddings** | vocab×head_dim×num_layers (~small at vocab 1024) | Extra learnable embeddings mixed into attention values. Major speedrun contributor. |
| **Bigram hash embedding** | 5×vocab×dim (~2.6M) | XOR hash of (prev, curr) token indexes into learnable table. Captures bigram stats directly. |
| **Multi-token prediction** | small aux head | Predict next token AND token+2. Auxiliary loss improves representations. |
| **XSA (Exclusive Self Attention)** | 0 | Constrain attention to capture info orthogonal to token's own value vector. Paper: 2603.09078. |

### Tier 3: Optimizer/training tricks

| Technique | What It Does |
|-----------|-------------|
| **NorMuon** | Per-row second-moment normalization on Muon. Already in PR #78. |
| **Polar Express** | Replace Newton-Schulz iteration in Muon. More numerically stable in bf16. |
| **Cautious Weight Decay** | Apply WD only when update direction agrees with weight direction. |
| **Scheduled Weight Decay** | WD changes over training, tied to LR schedule. |
| **Heavy WD + overparameterization** | Train bigger model with extreme WD → simpler solutions. |

### From Slowrun specifically

| Technique | What It Does |
|-----------|-------------|
| **Layer looping on MIDDLE layers** | Loop layers 15-20 three times. Middle layers benefit most from recurrence, not first/last. |
| **EMA weights at eval** | Free BPB, trivial implementation. |

## What's Already Being Tried in Parameter Golf

(Do not duplicate effort on these)
- Sliding window eval — PRs #65, #66, #77
- Wider MLP 3x — PRs #65, #66
- int6 quantization — PRs #65, #66
- Vocab 8192 — PR #78
- Depth recurrence — PRs #54, #58, #69 (no validated results)
- LoRA TTT — PR #77
- NorMuon — PR #78

## Sources

- modded-nanogpt: https://github.com/KellerJordan/modded-nanogpt
- slowrun: https://github.com/qlabs-eng/slowrun
- XSA paper: https://arxiv.org/abs/2603.09078
- LessWrong analysis: https://www.lesswrong.com/posts/j3gp8tebQiFJqzBgg/
