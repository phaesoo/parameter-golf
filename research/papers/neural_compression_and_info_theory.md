# Neural Compression & Information Theory Research

## Critical Finding for Parameter Golf

**"Language Modeling Is Compression" (Deletang et al., ICLR 2024)**
- At 38M params, ASCII/byte tokenization BEAT BPE for compression
- For larger models the pattern reverses
- This CONTRADICTS the assumption that larger vocab → better BPB at our scale
- Our model (~17-22M params) is in the regime where simpler tokenizers may win

This means PR #78's approach (8K vocab) might be suboptimal. The BPB improvement
from fewer tokens_per_byte could be offset by worse cross-entropy when the model
can't learn good embeddings for 8K tokens with so few parameters.

## Key Papers

### Directly Applicable

| Paper | Key Insight | Relevance |
|-------|-------------|-----------|
| Deletang et al. (ICLR 2024) | Small models compress better with simple tokenizers | Challenges large-vocab strategy |
| Relaxed Recursive Transformers (2410.20672) | Loop layers with per-layer LoRA deltas | Depth recurrence done right |
| Vocabulary Scaling Laws (2407.13623) | V_opt ∝ N_nv^0.83; ~16-20K optimal for ~33M params | But our model is much smaller |
| PEER / Million Experts (2407.04153) | Fine-grained MoE with product-key retrieval over 1M+ single-neuron experts | Max capacity in min bytes |
| Byte Latent Transformer (2412.09871) | Tokenizer-free, dynamic byte patching based on entropy | Eliminates tokenizer entirely |

### Compression Benchmarks (what's possible)

| System | Params | BPB on enwik8 |
|--------|--------|---------------|
| ts_zip (RWKV 169M, int8) | 169M | 1.11 |
| Nacrith (SmolLM2-135M + ensemble) | 135M | 0.918 (alice29) |
| CMIX v21 | N/A (traditional) | 1.17 |
| gzip | N/A | ~6.0 |

### Tokenizer Research

| Paper | Finding |
|-------|---------|
| Info-Theoretic Tokenizers (2601.09039) | LZ-aware BPE gives 15.8% vs 11.1% compression improvement. Compression-optimized ≠ performance-optimized tokenizers. |
| BPE Gets Picky (2409.04599) | Eliminating under-trained tokens improves vocab efficiency |
| Length-MAX Tokenizer (2511.20849) | 14-18% fewer tokens-per-char vs BPE, near-zero vocab waste |

### Architecture Alternatives

| Paper | Finding |
|-------|---------|
| Mamba at 130M-370M scale | Matches or exceeds Transformer perplexity. But training throughput may be lower with current implementations — matters for 10-min constraint. |
| ReMoE (ICLR 2025) | ReLU routing universally beats TopK in MoE. Differentiable routing. |
| GLaM 130M MoE | Directly in our param range |

## Implications for Our Strategy

1. **Vocab size**: Don't blindly increase. Test byte-level (256) or current 1024 against 8K. The Deletang result suggests 1024 might already be near-optimal for our scale.

2. **Recursive Transformers with LoRA**: The Relaxed Recursive Transformer paper validates exactly what PR #69 is trying — but with per-layer LoRA deltas instead of per-layer norms+signals. This is the most parameter-efficient way to do recurrence.

3. **PEER / Fine-grained MoE**: Radical idea — replace the MLP with product-key retrieval over many tiny experts. Potentially massive effective capacity within 16MB. Nobody in the competition is trying this.

4. **BLT (Byte Latent Transformer)**: Eliminates the tokenizer question entirely. Dynamic patching based on entropy means the model allocates capacity where text is hard to predict. Novel and untested in competition.

5. **MDL framing**: Parameter Golf IS minimum description length optimization. Model artifact = program, BPB = codelength. This isn't just a metaphor — it's the literal objective.

## Sources

- Language Modeling Is Compression: https://arxiv.org/abs/2309.10668
- Relaxed Recursive Transformers: https://arxiv.org/abs/2410.20672
- Vocabulary Scaling Laws: https://arxiv.org/abs/2407.13623
- PEER / Million Experts: https://arxiv.org/abs/2407.04153
- Byte Latent Transformer: https://arxiv.org/abs/2412.09871
- Info-Theoretic Tokenizers: https://arxiv.org/abs/2601.09039
- Mamba: https://arxiv.org/abs/2312.00752
- ReMoE: ICLR 2025
- ts_zip: https://bellard.org/ts_zip/
- Nacrith: https://arxiv.org/abs/2602.19626
