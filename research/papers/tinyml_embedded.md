# TinyML & Embedded AI Techniques

## Most Relevant Ideas

### Hash-Based Weight Sharing (HashedNet, HASH Layers)
- Hash function maps virtual params to smaller set of physical params
- **100K physical params → 1M+ virtual params** (10x effective multiplier)
- FFN layers: 1/8th params loses only ~1-2 perplexity points
- FFN is 2/3 of transformer params — this is where the savings are biggest
- Source: HashedNet (Chen 2015), HASH Layers (2021)

### ALBERT Cross-Layer Sharing
- All layers share same params → 108M → 12M (9x reduction)
- ALBERT-xxlarge actually EXCEEDS BERT-large despite sharing
- Combined with repetition: 12M unique params used 12+ times = massive effective compute
- Source: ALBERT (Google 2020)

### Embedding Factorization
- Decompose embedding V×H into V×E and E×H where E << H
- 32K vocab, H=768: E=128 cuts embedding from 25M to 4.2M params
- Our case (1024 vocab, 512 dim): embedding is small, but still relevant for larger vocab experiments

### Knowledge Distillation Numbers
- TinyBERT 14.5M params retains ~96% of BERT-base quality
- Baby LLaMA: 58M param model distilled from LLaMA ensemble on 10M tokens
- Key: distill attention patterns + hidden states, not just logits
- Progressive distillation (large→medium→small) better than direct

### Lottery Ticket / Sparsity
- 90% sparsity: minimal quality loss across most architectures
- Combined with quantization: 90% sparse + 4-bit = 20x compression
- Practical limit for extreme compression: sparsity + quant together
- L1 reg during training → naturally sparse → better zlib compression
- NOTE: unstructured sparsity needs index overhead. Structured (N:M) is hardware-friendly.

### NAS for Tiny Models
- Standard transformer configs are designed for large models — suboptimal at our scale
- Systematic search over (layers, dim, heads, FFN ratio) can find much better tradeoffs
- MCUNet: co-design architecture + runtime for specific hardware constraints
- Once-for-All: train one network supporting 10^19 sub-networks

## Actionable for Parameter Golf

| Idea | Potential | Difficulty |
|------|-----------|------------|
| Hash-based FFN sharing | High — FFN is 2/3 of params | Medium |
| ALBERT-style full layer sharing + more depth | High — proven 9x reduction | Low (already exploring in e01) |
| KD from larger teacher | Medium — 10-30% quality boost | High (need teacher + soft targets) |
| L1 reg for compression-friendly weights | Low-medium — better zlib ratio | Very low |
| Systematic shape search (mini-NAS) | Medium — find optimal config | Low (just sweep configs) |
| Structured N:M sparsity | Medium — 2x with HW support | Medium |
