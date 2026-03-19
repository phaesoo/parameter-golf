# Parameter Golf Research

## Goal

Win the OpenAI Parameter Golf challenge.
Achieve the lowest BPB within the 16MB artifact + 8xH100 10-minute constraint.

## Document Structure

- `plan.md` — Experiment plan and decision tree
- `log.md` — Experiment log (chronological)
- `decisions.md` — Key decision points with rationale
- `analysis/` — Data analysis, learning curve comparisons, etc.

## Core Principles

1. **Hard-to-reverse decisions first** — precision → vocab → eval strategy → architecture → hyperparameters
2. **Avoid local optima** — no lower-level tuning until upper-level decisions are locked
3. **Log everything** — failures are more informative than successes

## Working Rules

1. All research documentation lives in this directory (not in LLM memory)
2. All documents are written in English
3. The LLM determines the next step by carefully reasoning about the most effective direction, rather than waiting for explicit instructions
4. Commit and push after each completed unit of work
