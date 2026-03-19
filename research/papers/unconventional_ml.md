# Unconventional ML Approaches

## High-Potential Ideas

### 1. Deep Equilibrium Models (DEQ) — Infinite depth, finite params
- Single transformer layer applied until convergence = 15-30 "virtual" layers
- DEQ-Transformer: competitive perplexity with 1/3 the parameters
- One layer ~3MB → acts like 20-layer model
- Jacobian-free backprop = O(1) memory in depth
- **Risk**: convergence instability, inference latency
- Source: Bai et al. 2019

### 2. Tensor-Train Decomposition for Embeddings
- Reshape vocab embedding (e.g., 8192×512) into higher-dim tensor, then TT-decompose
- **20-50x compression** of embeddings specifically
- TensorGPT: 8-38x compression with minimal perplexity loss
- This is probably **highest ROI single technique** for larger vocab
- Source: TensorGPT (Xu et al. 2023)

### 3. Hypernetworks — Small net generates large weights
- 10-25x expansion achievable. 2MB hypernetwork → 50M effective params
- Chunked hypernetwork: each layer gets small embedding, hypernetwork maps → weight chunk
- **Risk**: quality degrades nonlinearly past ~15x. Generated weights lack fine-grained structure.
- Wild idea: 1MB hypernetwork + layer_id conditioning = entire model weights generated
- Source: Ha et al. 2016

### 4. Hash-Based Sparse MoE
- Hash function selects experts (zero routing params)
- 8K experts × 2KB each = 16MB for expert table
- Or: expert as codebook vector, element-wise multiply with input
- No routing overhead, all budget goes to expert capacity
- Source: Roller et al. "Hash Layers"

### 5. Retrieval-Augmented with PQ Index
- Product quantization: 500K vectors in ~12MB
- 500K chunks × 100 tokens = 50M tokens of retrievable knowledge
- kNN-LM showed retrieval-augmented models match 25x larger models
- **Risk**: only works if eval data overlaps with index content
- Wild split: 2MB model + 14MB retrieval index
- Source: Khandelwal et al. 2020 (kNN-LM), Borgeaud et al. 2022 (RETRO)

### 6. Neural Cellular Automata for LM
- Tiny update rule (<200K params) applied 64+ times over sequence
- Effective depth 64 with params of 1 layer
- **Genuinely unexplored** for language modeling
- Needs global communication mechanism (hybrid: NCA + small attention)
- Source: Mordvintsev et al. 2020

### 7. Implicit Neural Representations for Weights
- Small MLP f(layer_id, row, col) → weight value
- 20-50x compression for structured weight matrices
- Pay inference cost (generate weights at startup, then cache)
- SIREN periodic activations improve quality
- Source: COIN++ (Dupont 2022), SIREN (Sitzmann 2020)

## Ranked by Feasibility for Parameter Golf

| Rank | Technique | Expected Impact | Risk | Why |
|------|-----------|----------------|------|-----|
| 1 | TT-decomposed embeddings | High | Low | Mature, directly applicable, huge savings on embedding |
| 2 | DEQ (fixed-point depth) | High | Medium | 3x effective params, but convergence concerns |
| 3 | Hash-based MoE FFN | Medium | Medium | Better capacity routing, zero routing overhead |
| 4 | Hypernetwork weight generation | Medium | Medium-High | 10-25x expansion but quality ceiling unknown |
| 5 | PQ retrieval index | High if eval matches | High | Domain-dependent, risky for general BPB |
| 6 | NCA for LM | Unknown | Very High | Genuinely novel, nobody has tried |

## Wild Combo: DEQ-MoE-TT (proposed by research agent)

- TT-decomposed embedding (3MB)
- Single transformer block as DEQ with hash-routed MoE FFN (4MB)
- 20 convergence iterations = 20 effective layers
- Hypernetwork modulator for depth-varying behavior (1MB)
- Remaining for quantized components + possibly small retrieval index
- Total: 16MB, effective capacity of ~20-layer MoE transformer
