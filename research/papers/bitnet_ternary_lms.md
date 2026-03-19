# BitNet & Ternary LM Research

## Key Question: Is 85M ternary > 17M int8 at our scale?

### Evidence Summary

**BitNet b1.58 Reloaded (2407.09527)** — tested 100K to 48M params:
- Effective capacity of 1.58-bit weights ≈ **50% of FP16** at small scale
- 85M ternary ≈ ~42M FP16 effective — still 2.5x more than current 17M int8
- Recommends AbsMedian quantization (not AbsMean) for stability at small scale

**Spectra / TriLM (ICLR 2025)** — 99M to 3.9B params:
- TriLM-99M ≈ FloatLM-50M on web text
- Perplexity gap persists on noisy web corpora even at 3.9B
- Ternary models absorb more data than FP models at same size

**ParetoQ (NeurIPS 2025)**:
- **2-bit may be practical sweet spot** — comparable Pareto frontier to ternary, cleaner packing
- Ternary 600M beats previous SOTA ternary 3B on commonsense
- Learning transition between 2-3 bits: below 3b, representations change drastically from pretrained

**"Low-Bit Quantization Favors Undertrained LLMs" (ACL 2025)**:
- **WARNING**: Well-trained small models suffer MORE from quantization
- Our model (~7B tokens / ~17M params = 400 tokens/param) is heavily trained
- Quantization degradation will be worse than big-model papers suggest

### Verdict

Ternary is promising but not a slam dunk. Expected effective gain: 85M ternary ≈ 42-50M FP16 params. The 2.5x effective capacity boost is real but comes with:
- Training instability risk at small scale
- Higher quant degradation for well-trained models
- Noisy web data (FineWeb) hurts ternary more than clean benchmarks

**2-bit QAT may be the practical sweet spot**: ~50M params in 16MB, simpler packing, ParetoQ-validated.

### Training Stability Tricks (from papers)

1. AbsMedian quantization (not AbsMean) — better for small models
2. Full-precision latent weights, ternary only in forward pass (STE)
3. RMSNorm before every linear layer (SubLN)
4. Activations quantized to 8-bit per-token using AbsMax
5. Phased training: 20% full BF16 warmup, then QAT with LR reduction
6. L1 regularization to encourage zeros (better compression)

## Sources

- BitNet b1.58: https://arxiv.org/abs/2402.17764
- BitNet b1.58 Reloaded: https://arxiv.org/abs/2407.09527
- Spectra / TriLM: https://arxiv.org/abs/2407.12327
- ParetoQ: https://arxiv.org/abs/2502.02631
- Low-Bit Favors Undertrained: https://arxiv.org/abs/2411.17691
- TernaryLM: https://arxiv.org/abs/2602.07374
