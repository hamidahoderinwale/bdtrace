# analysis/preferences/ — scripts in this directory

Paper 2 pilot: procedural analysis of SWE-agent trajectories. Scripts group
into four roles. All are runnable as `python -m analysis.preferences.<name>`
from the repo root.

## Pipeline (data → canonical → BPE)

Run these in order when regenerating from scratch.

| script | what it does |
|---|---|
| `fetch_raw_trajectories.py` | Fetch `.traj` files from the public SWE-bench submissions S3 bucket. Writes to `output/trajectories/.cache/` with a `manifest.json` (SHA-256 per file). Idempotent. |
| `canonicalize.py` | Library. Maps one raw SWE-agent action string to one canonical atom (e.g. `EDIT_SRC_PY`). 76-atom vocabulary. Used by the BPE script. |
| `bpe.py` | Library. Pure-Python BPE implementation (`train_bpe`, `apply_bpe`). No external deps. |
| `run_bpe_analysis.py` | End-to-end: canonicalize all trajectories → train BPE at V=200 → write `bpe_sequences.jsonl`, `bpe_model.json`, `bpe_top_motifs.csv`. |

## Analyses (what the paper reports)

Each produces a figure + JSON + sometimes CSV. All output paths in
`output/paper2_pilot/`.

| script | figure | finding |
|---|---|---|
| `variance_decomp.py` | `variance_decomposition.png` (archived) | sd(group_means)/sd(agent_means) per feature across three groupings (task-id, fix-type, repo). Phase A 8-token DSL; superseded at the BPE layer. |
| `pair_features.py` | `pair_action_levenshtein.png` (archived) | Pairwise Levenshtein on tied-outcome action sequences, per agent pair. Phase A 8-token DSL; superseded by `bpe_pair_levenshtein.png`. |
| `task_diversity.py` | `task_diversity_distribution.png`, `task_diversity_by_resolved.png` (both archived) | Per-task procedural divergence across agents. Phase A 8-token DSL; reclaim candidate in cross-corpus form. |
| `motif_distributions.py` | `agent_motif_distributions.png`, `agent_jsd_matrix.png` | Per-agent motif distributions + pairwise dissimilarity matrix. |
| `bpe_vocab_sweep.py` | `bpe_vocab_sweep_jsd.png`, `bpe_vocab_sweep_compression.png` | Robustness of heritability ordering across V ∈ [100, 500]. |
| `bpe_mdl.py` | `bpe_mdl_curve.png` (archived) | MDL-based V selection. Did not converge; archived. |
| `slice_difficulty.py` | `slice_difficulty_jsd.png`, `slice_difficulty_motifs.png` | Pairwise dissimilarity by task difficulty (0/3 → 3/3). |
| `permutation_null.py` | `permutation_null.png` | Is the same-family advantage above chance? Shuffle labels × 1000. |
| `matched_pairs.py` | `matched_pairs_volcano.png`, `matched_pairs_top_motifs.png` | Per-motif Wilcoxon on tied-outcome pairs + BH-FDR. |
| `aggregate_metrics.py` | `aggregate_metrics.png`, `length_by_difficulty.png`, `novelty_top_motifs.png` | Per-agent entropy, repertoire width, length, compression; distinctive motifs. |
| `trajectory_clusters.py` | `trajectory_clusters_umap.png`, `trajectory_clusters_profile.png` | HDBSCAN on motif-frequency vectors; UMAP for display. |
| `token_cost.py` | `token_cost_per_agent.png`, `token_cost_by_difficulty.png` | Per-agent infrastructure costs from `model_stats`. |
| `step_resource_analysis.py` | `step_resources_*.png` | Per-motif cost/success profile; efficiency frontier; wasteful-motif ranking. |
| `fixtype_motif_heatmap.py` | `fixtype_motif.png` | Fix type × motif usage ratio. Shows procedural specialization by semantic fix type. |

## Case studies + viewer (thick description + interactive tool)

| script | output |
|---|---|
| `case_studies.py` | `case_studies.html` — four tasks side-by-side with phase maps and narratives. |
| `pairwise_diffs.py` | `pairwise_diffs.html` — motif-level diffs for three shared tasks (embedded in dashboard § 4.4). |
| `build_viewer_data.py` | `docs/paper2_pilot/viewer_data/*.json` — per-step payloads (atom, action text, reasoning, observation) for the interactive viewer (`docs/paper2_pilot/viewer.html`). |
| `generate_examples.py` | `examples_snippet.html` — worked examples of BPE merges and the three-view representation (embedded in dashboard § 4). |

## Utilities

| script | what it does |
|---|---|
| `enumerate_submissions.py` | Queries `github.com/swe-bench/experiments` for metadata on all 240 SWE-bench leaderboard submissions. Writes `submission_enumeration.json` + `.md` — the fetch plan for Move 1 (cross-benchmark universals). |

## Style conventions for plots in this directory

- Axis titles state the finding in plain English, not the method.
- Agent colors fixed across the project: Claude-3.5 `#009E73`, GPT-4 `#0072B2`, GPT-4o `#E69F00`.
- Avoid log-odds / z-scores / η² where a ratio or frequency-difference is
  equally informative.
- One finding per panel. Multi-panel figures use small-multiples with one
  variable on rows, one on columns, where possible.
- Annotate outliers directly on the panel instead of legends-off-to-the-side.
- No em dashes in figure titles or axis labels.

## Reproducing the whole pipeline

```bash
# 1. Fetch trajectories (one-time, ~10 min, writes 234 MB to output/trajectories/.cache/)
python -m analysis.preferences.fetch_raw_trajectories

# 2. Canonicalize + BPE
python -m analysis.preferences.run_bpe_analysis

# 3. All analyses (each depends on bpe_sequences.jsonl from step 2)
for script in variance_decomp pair_features task_diversity motif_distributions \
              bpe_vocab_sweep slice_difficulty permutation_null matched_pairs \
              aggregate_metrics trajectory_clusters token_cost \
              step_resource_analysis fixtype_motif_heatmap; do
  python -m analysis.preferences.$script
done

# 4. Viewer data + case studies
python -m analysis.preferences.build_viewer_data
python -m analysis.preferences.case_studies
python -m analysis.preferences.pairwise_diffs
python -m analysis.preferences.generate_examples
```
