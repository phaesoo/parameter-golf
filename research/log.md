# Experiment Log

## Format

```
### YYYY-MM-DD / Experiment ID

**Config**: (only what changed from baseline)
**Environment**: (GPU, wallclock)
**Results**: (BPB, loss, compressed size, etc.)
**Observations**: (learning curve anomalies, unexpected behavior)
**Conclusions**: (what to carry forward)
```

---

## 2026-03-19 / Project kickoff

- Challenge analysis complete
- Experiment plan established (`plan.md`)
- Decision tree defined: precision → vocab → eval strategy → architecture → hyperparameters
- Key insights from baseline analysis:
  - 10min vs 4hr same-architecture gap is only 0.017 BPB post-quant → architecture/precision change is essential
  - Quantization degradation scales with training length (0.0072 vs 0.0325) → QAT will be critical
  - Warmdown phase is extremely effective (0.025 BPB in 1780 steps)
  - Model is NOT converged at 10 min — still learning fast when wallclock cap hits
- Next: Day 1 experiments (weight precision PoC)
