# Audio/Video Codecs, VQ, Rate-Distortion, Fractal Compression

## Most Actionable Ideas

### 1. AQLM / QuIP# — Vector Quantization beats Scalar Quantization
- **Scalar quant (int8/int6/int4)** ignores correlations between weights
- **AQLM (Additive Quantization)**: multi-codebook VQ, jointly optimized across blocks
  - First method Pareto-optimal BELOW 3 bits/param
  - 2-bit quantization with usable quality
- **QuIP#**: Hadamard transforms make weights "incoherent" + E8 lattice codebooks
  - Mathematically optimal 8-dim sphere packing
  - 3-bit models scale BETTER than 4-bit models
- **At 2 bits/param: 16MB → 64M parameters** vs 34M at int4
- Source: AQLM (2401.06118), QuIP# (2402.04396)

### 2. Layer-Delta Coding (from video P-frames)
- Video: I-frame (full) + P-frames (delta from previous)
- Weight analogy: store layer 0 fully, layers 1-N as deltas from previous
- Adjacent transformer layers show HIGH cosine similarity → deltas are very compressible
- Combined with depth recurrence: store 3 unique blocks + small deltas for variation
- Source: Video compression principles

### 3. Rate-Distortion Optimal Bit Allocation
- Different layers need different bit widths (Shannon's R(D) theory)
- Early layers + attention projections: more sensitive → more bits
- Middle FFN layers: more redundant → fewer bits
- **Joint optimization across all layers** >> greedy layer-by-layer
- Radio (2505.03031): SGD-based R(D) optimization for billion-param models
- For us: spend bits where sensitivity is highest

### 4. Residual VQ (from audio codecs)
- Cascade of codebooks: each stage quantizes the RESIDUAL error of previous
- Like boosting: each level focuses on what previous got wrong
- EnCodec uses this for audio at 1.5-24 kbps
- Applied to weights: first codebook captures coarse structure, subsequent ones refine
- **Structured quantization dropout**: train model to work at multiple precision levels

### 5. Fractal Self-Similarity in Weights
- **"Recursive Self-Similarity in Deep Weight Spaces" (2503.14298)**: trained neural networks
  genuinely exhibit fractal-like structure. Not metaphor — mathematical framework applies.
- Store a "seed" pattern + iterated function system (IFS) transformations
- Reconstruct full weight matrix through iteration
- **Practical version**: share submatrices between layers at different scales + affine transforms
- Combined with low-rank: small core matrix + per-layer affine transforms

### 6. Hybrid Compression (from Opus codec)
- Opus: SILK for speech, CELT for general audio, switches based on content
- Weight analog: different compression per component type:
  - Embeddings → VQ with large codebooks (they cluster well)
  - Attention weights → low-rank + quantization
  - FFN weights → aggressive scalar quantization (most redundant)
  - Control params → fp16 passthrough (tiny, sensitive)

## Key Numbers

| Method | bits/param | Params in 16MB | Quality |
|--------|-----------|----------------|---------|
| int8 scalar | 8 | ~17M | Baseline |
| int6 scalar | 6 | ~22M | Current competition standard |
| int4 scalar | 4 | ~34M | Moderate quality loss |
| AQLM 3-bit VQ | 3 | ~42M | Better than int4 scalar |
| QuIP# 2-bit VQ | 2 | ~64M | Usable, E8 lattice optimal |
| Ternary | 1.58 | ~85M | 50% effective capacity of FP16 |

**Key insight: VQ at 2-3 bits >> scalar int4.** The competition is using scalar int6.
Nobody is using learned codebook VQ.

## Sources

- AQLM: https://arxiv.org/abs/2401.06118
- QuIP#: https://arxiv.org/abs/2402.04396
- Radio R(D) optimization: https://arxiv.org/abs/2505.03031
- Fractal weight spaces: https://arxiv.org/abs/2503.14298
- EnCodec: https://arxiv.org/abs/2210.13438
- SoundStream: https://arxiv.org/abs/2107.03312
