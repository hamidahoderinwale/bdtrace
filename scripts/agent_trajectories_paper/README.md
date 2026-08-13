# Analysis scripts — *Agent trajectories as programs* paper

Reproduces the grounded figures/numbers for `docs/papers/agent_trajectories_as_programs.tex`.
Rescued from `/tmp` (2026-06-05) so they're version-controlled with the paper.

**Run from the repo root** (all scripts use cwd-relative `output/...` paths,
`sys.path.insert(0,'.')`, and `from scripts.theme import …`):

```bash
python scripts/agent_trajectories_paper/<script>.py
```

## Inputs (shared)
- `output/paper2_pilot/bpe_sequences_extended.jsonl` — 2639 rows: `{submission, agent, instance_id, canonical, bpe}`
- `output/paper2_pilot/extended_pass_fail.json` — per-submission resolved instance lists

## Script → output → paper anchor

| Script | Produces | Paper anchor | Status |
|---|---|---|---|
| `bpe_vs_prefixspan.py` | prints V-measure: BPE vs PrefixSpan | §3 sentence + `fig:bpe_vs_prefixspan` | grounded (BPE 0.606/0.626 vs PrefixSpan 0.505) |
| `make_figs.py` | `fig_jsd_matrix_full_canonical.png`, `fig_regression_length.png` | JSD matrix + length appendix | grounded; length recaptioned 45%→29% (non-monotonic) |
| `make_edittest.py` | `fig_regression_edit_test.png` | edit/test appendix | grounded; recaptioned pass 0.79 vs fail 0.67 |
| `probe_refit.py` | prints next-action / next-stage probe accuracy by agent (first-order Markov, 5-fold) | `tab:trajectory_holdouts` | grounded (table regenerated from this script 2026-06-08; the prior table's higher numbers were from an unversioned computation and had a scaffold next-action/next-stage copy artifact) |
| `reward_verifier.py` | prints `proc_score` best-of-N selection resolve-rates vs random/worst/best-agent/oracle | §reward (test-time verification) | grounded; **not** a reward-hacking run |
| `reward_fig.py` | `docs/papers/figures/fig_reward_selection.png` | §reward figure | grounded |
| `sodp.py` | `docs/papers/figures/fig12_tier1b_sodp.png` | — | **null result** (both cats ≈0.75); figure CUT from tex, kept for the record |

## Notes
- `reward_verifier.py` defines the `proc_score` heuristic inline (exploration / implementation /
  edit-then-test / completion bonuses; edit-streak and no-search penalties). It is a procedural
  *selection* / verification demo, not an adversarial reward-hacking study.
- `sodp.py`'s figure is intentionally not referenced by the paper — the analysis returned a null
  and the figure was cut. Script retained so the null is reproducible rather than lost.
- The paper's `proc_score` pass/fail means (0.902 / 0.723) are **not** reproduced by these scripts;
  that claim still needs its source located before it can be treated as grounded.
