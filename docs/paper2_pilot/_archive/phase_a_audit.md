# Phase A audit — 2026-04-24

Compression pass before pivoting Paper 2 to the universals direction
(cross-benchmark / cross-language transfer of the Lite-trained motif
vocabulary). The BPE vocabulary trained in Phase B is the measuring
instrument going forward; Phase A (the 8-token hand-coded DSL) is
scaffolding that motivated the methodology and does not survive the
reframe as a headline finding.

## Decisions

### Kept (instrument and committed methodology)

`bpe_model.json`, `bpe_sequences.jsonl`, `bpe_top_motifs.csv`,
`bpe_summary.json` (the instrument); `canonicalize.py`, `bpe.py`,
`run_bpe_analysis.py` (pipeline that builds it); `bpe_vocab_sweep.py`
plus its sweep PNGs and JSON (instrument robustness justification);
`submission_enumeration.json` and `.md` (Move 1 fetch plan);
`examples_snippet.html` (pedagogical anchor for what BPE is doing);
dashboard sections 3.x, 4.x, 5.1.

### Reframed in place (within-Lite finding, not universal claim)

Captions and surrounding prose to be narrowed in a follow-up pass once
cross-corpus data lands. No file moves.

`agent_motif_distributions.*`, `agent_jsd_matrix.png`,
`matched_pairs_*`, `trajectory_clusters_*`, `fixtype_motif.*`,
dashboard sections 6 (headlines) and 7 (implications).

### Demoted to appendix (diagnostic, not headline)

No file moves. Dashboard prose to mark them supplementary in the
follow-up pass.

`permutation_null.*`, `slice_difficulty_*`, `aggregate_metrics.*`,
`length_by_difficulty.png`, `novelty_top_motifs.png`,
`step_resources_*`, `token_cost_*`, `pairwise_diffs.html`,
`case_studies.html`.

### Archived (Phase A scaffolding, subsumed by Phase B)

Moved into this folder. The same heritability and diversity questions
are answered more rigorously at the BPE motif layer.

- `variance_decomposition.{png,csv,json}` and `_controlled` variants:
  the dashboard's section 2.1 ratios on the 8-token DSL. Replaced by
  per-motif matched-pairs and per-agent JSD at the BPE layer.
- `pair_action_levenshtein.png`, `pair_features.json`: dashboard
  section 2.2 pair distance on the 8-token DSL. Replaced by
  `bpe_pair_levenshtein.png` and BPE-layer matched-pairs analysis.
- `task_diversity_distribution.png`, `task_diversity_by_resolved.png`,
  `task_diversity.{csv,json}`: dashboard section 2.3 task-level
  diversity on the 8-token DSL. Candidate to reclaim later in
  cross-corpus form ("diversity per task per corpus") but archived in
  current within-Lite form.

## Not done in this pass

Dashboard prose reframe (section 2 collapse, section 6 and 7
narrowing), splitting the dashboard into instrument page versus Lite
exhibit page, and any decision about moving `case_studies.html` or
`pairwise_diffs.html` into a `gallery/` folder. Defer until the first
cross-corpus replication is in hand and the universals story has its
real headline.

## Scripts

No script in `analysis/preferences/` deleted. The scripts producing
archived outputs (`variance_decomp.py`, `pair_features.py`,
`task_diversity.py`) still run and produce their JSON and PNG; the
artifacts are now scaffolding rather than headline figures. README
entries for those scripts are flagged as archived alongside this note.
