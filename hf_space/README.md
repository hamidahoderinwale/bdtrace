---
title: SWE-bench Lite — Eval Explorer
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
---

# SWE-bench Lite — Eval Explorer

A structural embedding browser over 300 SWE-bench Lite instances. Points are positioned by UMAP applied to AST edit-operation distances — proximity reflects procedural similarity, not semantic content.

## What you can do

- **Color by** fix type, repo, pass/fail, structural coverage, task quadrant, or number of models that solved it
- **Filter** by fix type, repo, or model outcome (solved by ≥1, solved by all, unsolved)
- **Click any point** to inspect: per-model outcomes (GPT-4, Claude 3.5, GPT-4o, Qwen2.5), fix summary, structural coverage score, behavioral metrics (steps, edit retry rate)
- **Bar chart** shows live pass rate by fix type for the current filter

## Data

Companion dataset: [midah/procedural-info-theory](https://huggingface.co/datasets/midah/procedural-info-theory)
