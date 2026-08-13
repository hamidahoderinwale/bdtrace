# Procedural evaluation for LLM agents

*Draft / framing note, 2026-04-24. Companion to the Paper 2 pilot dashboard.*

## The problem

SWE-bench and its relatives ask one question: did the agent's final patch
pass the tests? Pass/fail is binary per task, and the leaderboard is a single
scalar: fraction resolved. This framing treats agents as black-box functions
from issue-text to patch-text and throws away the thing the agent actually
produced — its trajectory.

The Paper 2 pilot shows, empirically, that the trajectory carries signal the
final patch cannot:

- **Cost per resolved task varies 43% between agents** that have the same
  nominal resolve rate (Claude-3.5 $10.20, GPT-4o $16.00, GPT-4 $17.91). A
  pass/fail eval treats $10 and $18 solutions as equivalent.
- **Specific procedural patterns predict success more than the base rate**
  (the `EDIT_SRC → RUN_REPRO → SHELL_RM → SUBMIT` motif resolves 30% of the
  time vs a corpus base of 15%). Other patterns are direct anti-patterns
  (long repetitive edit bursts resolve 0–9%).
- **Same-family agents are procedurally heritable** (GPT-4 × GPT-4o motif
  dissimilarity is smaller than either with Claude, p=0.001 under a
  label-shuffle null). Backbone lineage leaves a procedural signature that
  pass/fail doesn't see.
- **Most trajectories are "generic"** (78% land in an unclusterable blob);
  the 22% that form dense clusters are almost entirely agent-typed. Agents
  have signatures invisible to outcome measurement.

Pass/fail is load-bearing but undersells the measurable differences between
agents. The argument of this note: a richer set of evaluation metrics, built
from the procedural structure already present in every SWE-agent submission,
is cheap to compute and surfaces strictly more information.

## What procedurally-aware evals should control and report

Existing evals control for task (same repo, same tests), tool set (same
agent scaffold), and prompt (same instruction). They do **not** control for:

- Which files the agent actually opens and includes in its context.
- The volume and composition of its investigation phase.
- Cursor positions and within-file navigation.
- Search queries and how many fire before a hypothesis.

Two failures look identical on the leaderboard; procedurally they can be
very different ("never opened the right file" vs "found the right spot, then
wrote a bad patch"). A procedural eval distinguishes them.

## Six concrete new metrics

All computable from data every SWE-agent submission already produces —
raw `.traj` files plus `model_stats`. No new instrumentation.

### 1. Localization rate

`P(agent opens at least one file touched by the golden patch)` — conditional on
any resolution attempt. This is the procedural analog of "gets into the
right neighborhood." If an agent's localization rate is 80% but its solve
rate is 20%, it finds the neighborhood but fumbles the edit; if localization
is 30%, the fundamental problem is search-not-edit.

### 2. Cost per resolved task (CPR)

`total_cost_usd / n_resolved`. Dollarized per-success number. Our pilot:
Claude $10.20, GPT-4o $16.00, GPT-4 $17.91 — a 75% spread that pass/fail
hides entirely. Should be a first-class leaderboard column.

### 3. Resolve rate at fixed step budget

`P(resolved | n_steps ≤ K)` for K ∈ {15, 20, 30}. Measures efficient solving
specifically. Agents that only succeed with 50-step trajectories are
different tools than agents that succeed in 10. Our data shows this maps to
distinct procedural signatures (short = cleanup-then-submit cluster; long =
edit-burst cluster).

### 4. Wasteful-motif prevalence

Fraction of trajectory atoms that fall inside a motif occurrence whose
corpus-wide resolve-rate is below the base rate. Proxy for "how much did the
agent thrash?" Per our pilot, GPT-4o has the highest prevalence (driven by
its `EDIT × 32` signature), Claude the lowest.

### 5. Phase coverage

Boolean per trajectory: did the agent visit each of {exploration,
localization, editing, verification, cleanup+submit}? Missing phases are
informative. Agents that submit without ever running a reproducer are
procedurally incomplete regardless of whether the patch happens to pass.

### 6. Procedural heritability score (PHS)

Same-family JSD minus cross-family JSD on an agent-pair's motif
distributions, averaged across the corpus. Tracks how much of an agent's
style is inherited from its backbone family versus distinctive to this
release. Useful for vendors to see how much their new model has drifted
from its predecessor, procedurally.

## Worked example on the pilot corpus

Applied to the three agents on SWE-bench Lite:

| Metric | GPT-4 | Claude-3.5 | GPT-4o |
|---|---|---|---|
| Resolve rate (pass/fail) | 14% | 16% | 16% |
| Cost per resolved | $17.91 | $10.20 | $16.00 |
| Mean trajectory length (atoms) | 21.4 | 32.7 | 39.1 |
| Resolve at ≤ 20 atoms | 11% | 8% | 6% |
| Wasteful-motif prevalence | low | low | high |
| Phase coverage (all 5 phases) | 72% | 78% | 81% |
| PHS vs same family (GPT-4 × GPT-4o) | +0.06 | — | +0.06 |

*The pass/fail column says "roughly tied." Every other row disagrees.*

## Three deeper experiments this unlocks

### Replay-then-diverge A/B

SWE-agent environments are stateful enough to serialize. Run agent A
through the investigation phase, freeze state, then have agent B resume
from there. Compare outcomes. This is a ceteris-paribus procedural A/B:
"given the same investigation, which editor is better?" Currently impossible
because procedures are not first-class.

### Counterfactual step injection

At a known pivot (e.g., just before the first EDIT), insert a synthetic
`CREATE_REPRO` atom. Does downstream performance improve? Measures the
causal effect of specific motifs. A framework that exposes procedural
positions at runtime enables this cheaply.

### Context-matched scoring

For each tied-outcome pair, compute the per-motif effect size. We did this
in the pilot (see §5.3 of the dashboard, matched-pairs analysis) and found
direction replicates across cross-family comparisons despite sparse
individual FDR-significance. More tied-outcome pairs (i.e., more agents on
the same tasks) would make this the sharpest tool in the kit.

## Why now

Three things line up:

1. **Submissions are public.** SWE-bench's S3 bucket has raw `.traj` files
   for every leaderboard submission. The data exists.
2. **Procedural vocabulary exists.** The pilot's 76 canonical atoms + 124
   BPE motifs cover 99.9% of actions in 867 trajectories. The abstraction
   layer is reusable for any SWE-agent submission.
3. **Per-token pricing is non-uniform.** Cost per resolved task is now a
   real economic variable, not a research curiosity. Teams procuring
   agents at scale (CI bots, issue triage, agentic test authoring) care
   about this number directly.

## What this doesn't do

- **It is not a causal account.** Efficient motifs are *correlated* with
  solve rates; we can't yet claim that forcing an agent to use them would
  improve outcomes. The replay-then-diverge experiment is what would turn
  correlational evidence into causal.
- **It does not replace pass/fail.** Pass/fail is still the ground-truth
  outcome. Procedural metrics complement it.
- **It needs scaffold diversity.** All our data is same-scaffold (SWE-agent)
  with different backbones. A backbone-scaffold factorial study would let us
  decompose "what the model does" from "what the scaffold constrains."

## Where this sits in the larger agenda

This is the normative arm of a four-arm program:

1. **Descriptive** (the Paper 2 pilot): what do agent procedures look like?
2. **Normative** (this note): what should procedures look like, and how do
   we measure against it?
3. **Generative** (DSPy / LangGraph / scaffold extensions): how should
   frameworks expose motifs so agents are composable/optimizable at the
   procedural layer?
4. **Interactive** (the Live Programming viewer): how do humans inspect
   agent procedures as direct-manipulation objects?

Arms 2 and 3 are mutually reinforcing. An eval that rewards procedural
efficiency only matters if a framework lets you target procedural efficiency.
Both are newly motivated by the pilot's empirical evidence that procedural
layer is real, heritable, and load-bearing for cost.

## Where it could land

- **Short paper** (6–8 pages) at an evals-focused venue (NeurIPS D&B, ML
  Evaluation Standards workshop, ICLR BPTs).
- **Section in a bigger Paper 2** — replace the "limitations / next steps"
  section with a concrete set of proposed leaderboard columns.
- **Proposal to the SWE-bench maintainers** — argue for adding CPR and
  phase-coverage as tracked columns. This is the most operational path;
  the patch is small (one-page math + a compute script).

## Status

Draft 1, 2026-04-24. Written alongside the Paper 2 pilot dashboard; all
numbers here trace to figures in that dashboard. The pilot data is at
`output/paper2_pilot/` and the dashboard at
`docs/paper2_pilot/dashboard.html`.
