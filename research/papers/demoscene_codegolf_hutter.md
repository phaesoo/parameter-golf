# Demoscene, Code Golf, Hutter Prize — Extreme Compression Principles

## Core Principle

**Separate what must be stored from what can be computed.**

## Transferable Ideas

### From Demoscene (4KB-64KB executables)

1. **Procedural generation > storage**: Never store what you can generate from a formula. Neural analog: weight-sharing, hypernetworks, structured matrices (low-rank factorization stores two small matrices instead of one large one).

2. **Domain-specific compression beats generic**: Crinkler/kkrunchy exploit x86 statistics. Neural analog: exploit known statistics of trained weights (approx normal, layer-wise correlated) → per-layer quantization codebooks > uniform quantization.

3. **Maximize primitive reuse**: Same SDF renderer produces different visuals via parameter variation. Neural analog: parameter sharing across layers, MoE routing over shared experts.

4. **Synth-based representation**: Compact synthesis parameters generate complex output. Neural analog: aggressive factorization — encode the *structure* of weight matrices, not raw values.

### From Code Golf

5. **Exploit implicit knowledge in the runtime**: Python golfer gets regex, math "for free." Neural analog: architectural inductive biases (attention, RoPE, skip connections) provide capability without stored parameters. The 16MB should only store what training discovers beyond the architecture.

6. **Not all parameters deserve equal precision**: Golfers use polynomials for some lookup tables, raw data for others. Neural analog: mixed-precision quantization, pruning, sensitivity-aware bit allocation.

7. **Choose the right "language"**: APL vs Java for the same algorithm. Neural analog: architecture choice IS a compression decision. The architecture whose inductive biases best match the task minimizes what must be stored.

### From Hutter Prize / CMIX

8. **Context mixing**: No single model is best in all contexts. Mixture that adapts to local statistics always wins. Neural analog: MoE where routing is context-dependent → effective capacity >> stored parameters.

9. **Preprocessing as free compression**: Dictionary coding, capitalization flags, number encoding before statistical compression. Neural analog: tokenizer design is outside the 16MB budget but directly affects needed parameter count.

10. **Depth > width for compression efficiency**: CMIX stacks models hierarchically. Neural analog: deeper narrow networks compress more per parameter than shallow wide ones.

### From Kolmogorov/MDL Theory

11. **MDL IS the challenge objective**: Minimize (model description length) + (data surprise given model) = minimize (16MB artifact) + (BPB). We're literally solving MDL.

12. **Train toward compressible solutions**: Weight decay = pressure toward shorter description length. L1 regularization → sparser weights → better compression. This isn't just regularization — it's the theoretically correct objective.

13. **Well-compressed weights should look random**: If quantized weights still have obvious statistical structure, you haven't compressed enough. High entropy in stored weights = good compression.

## Actionable Ideas for Parameter Golf

| Idea | From | Implementation |
|------|------|---------------|
| Per-layer quantization codebooks | Demoscene | Instead of uniform int6 everywhere, learn optimal quantization levels per layer |
| Hypernetwork weight generation | Demoscene | 2MB hypernetwork generates weights for main model. Total stored: 2MB, effective: 50M+ params |
| L1 regularization during training | Kolmogorov/MDL | Encourage zero weights → better zlib compression → fit more params in 16MB |
| Learned codebook VQ | Audio codecs + Demoscene | Vector quantize weight rows using learned codebook. Potentially much better than scalar int6 |
| Stronger architectural inductive bias | Code golf | Choose architecture that needs fewer params for same task quality |
| Distillation as "Kolmogorov shortcutting" | MDL | Large teacher already found the long program. Distill to find the shortest approximation. |
