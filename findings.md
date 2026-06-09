# Research Findings and Decision Traces

Running record of methodological decisions, key findings, and the reasoning behind representation choices. Updated as the project evolves.

---

## Core Thesis

Fix strategies have a latent structure measurable from edit operations but not recoverable from pass/fail outcomes or model self-report. This is the *procedural Clio* problem: eliciting what procedure an agent actually executed, from behavioral traces, without relying on the agent's account of itself.

---

## Three Stories

1. **Procedural Clio**: observe structurally, don't ask. Agents' self-descriptions of their fixes are nearly useless. You have to extract what they did from the structural trace of the patch.

2. **Procedural diversity**: same problem, different structural approaches. Agents use different edit operation vocabularies, even on the same scaffold. The LLM backbone drives fix strategy, not the wrapper.

3. **Structure predicts difficulty**: the structural pattern of a fix is the only thing that reliably predicts which instances are hard. Semantic descriptions of the problem or the fix both fail. And the reason hard patterns are hard is composition: agents have the individual parts but can't combine them.

---

## Representation Pipeline

### Level 1: Edit certificates (primary representation)

Each edit certificate is a set of (direction, AST-node-type) pairs extracted from a patch. For example, a patch that adds an if-check and removes a variable reference produces `{ADD_If, DEL_Name}`.

The format comes from GumTree (Falleri et al. 2014), stripped down deliberately. We drop Move (catching repositioned code adds complexity without meaningful cross-repository correspondence), Update (treated as delete+insert), and parent context (added back at Level 2). Starting stripped lets us measure what each dimension contributes as we add it back.

**What it abstracts over:** variable names, literal values, whitespace, line numbers. Two patches that add an if-check in completely different codebases get the same certificate if the structural operations are the same.

**What it loses:** order, multiplicity, context, scope.

**Key result:** Structural similarity of edit certificates predicts which instances an agent co-passes or co-fails better than chain-of-thought embeddings or hand-labeled fix types (F1=0.347 vs 0.266 vs 0.161 at k=5 nearest neighbors, leave-one-out). Structure carries more predictive signal than reasoning traces.

**Reference:** `analysis/procedures/ast_edit_sequences.py`, `scripts/compare_representations.py`

---

### Level 1.5: Scoped edit certificates (file + scope + edit type)

Enriches Level 1 with *where* the edit happens, not just *what* kind of edit. Each scoped certificate includes:
- **File path and module** (e.g., `django/db/models/sql/compiler.py`, module `django/db`)
- **Scopes touched** (which functions and classes the edit falls inside, e.g., `FunctionDef:get_columns`)
- **Patch size** (lines added/removed, number of hunks)
- **Level 1 certificate** (the edit operations themselves)

This decomposes "how similar are two fixes" into three layers:
1. **File-level**: did the agent find the right file?
2. **Scope-level**: did it edit the right function or class?
3. **Edit-type level**: did it make the right structural operations?

The biggest drop in agent agreement is from file to scope. Agents often find the same file (agreement 0.55-0.76) but edit different functions within it (scope Jaccard 0.25-0.31). Edit-type agreement sits between (0.43-0.54). The bottleneck is localization within the file, not edit strategy.

**Reference:** `analysis/procedures/scoped_edit_ops.py`, `scripts/build_scoped_certificates.py`, `scripts/build_agent_scoped_certs.py`

---

### Level 2: Contextual edit operations

Adds parent-node context: `ADD_For@FunctionDef` instead of just `ADD_For`. This disambiguates cases Level 1 collapses. For example, `sympy__sympy-12454` (trivial bound tightening) and `sympy__sympy-20049` (algorithmic BFS redesign) both produce `ADD_For` at Level 1, but Level 2 correctly produces `ADD_Name@comprehension` vs `ADD_Assign@While`.

Coverage: 203/300 instances (67.7%), lower than Level 1 (289/300, 96.3%) due to no fallback keyword detection.

**Reference:** `analysis/procedures/contextual_edit_ops.py`

---

### Level 3: Fix intent (semantic labels)

LLM-assigned labels from a 12-intent taxonomy per hunk (e.g., `add_guard`, `algorithm_replace`, `api_change`). Useful for *naming* what a fix does in human-readable terms. As a *grouping mechanism*, the intent-based decision tree was superseded by FIM: it assigns 147/289 instances (81%) to a single catch-all "minimal" form, collapsing most difficulty variation.

**Reference:** `representations/inferred/fix_type/chunk_intent.py`

---

## Key Empirical Findings

### 1. Structural patterns predict co-pass/co-fail better than reasoning traces

If two instances have similar edit certificates (similar structural fix patterns), agents that pass one tend to pass the other. This holds better than similarity based on chain-of-thought reasoning or hand-labeled fix types.

Measured by k=5 nearest-neighbor prediction (leave-one-out):
- **Edit certificate similarity: F1=0.347**
- Chain-of-thought embedding similarity: F1=0.266
- Hand-labeled fix type similarity: F1=0.161

F1 combines precision (of predicted co-passes, how many were real) and recall (of real co-passes, how many were predicted). 0.347 is modest but substantially above the alternatives.

**Reference:** `scripts/compare_representations.py`, `output/representation_comparison/fig2_at_k5.png`

---

### 2. Structural form and semantic topic are unrelated

Grouping instances by structural form (what kind of edit the fix requires) and grouping by semantic topic (what the bug report is about) produce essentially independent partitions.

Adjusted Rand Index = 0.010, where 0 means no more agreement than random chance and 1 means identical groupings. Knowing that a bug is in django vs sympy tells you nothing about what structural approach the fix requires.

**Reference:** `output/form_alignment/`

---

### 3. Agents' self-descriptions of their fixes are nearly useless

We asked GPT-4o to describe what edit operations it performed on a patch, then compared its claims to the actual edit certificate.

- Of the operations GPT-4o *claimed* it did, only **28% were actually in the patch** (precision)
- Of the operations *actually in the patch*, GPT-4o only mentioned **14%** (recall)
- Combined: **F1=0.20** (on a 0-1 scale where 1 is perfect, 0.20 is close to useless)

This held across prompting conditions (no context, raw logs, procedural). The model consistently misses most of what's in its own patch. This is why we extract structural patterns directly from patches rather than asking agents what they did.

**Reference:** `output/grounding_validation/`

---

### 4. More benchmark instances don't add new structural forms

All 10 structural forms appear across SWE-bench Lite, Verified, and SWE-smith. Within each benchmark, form coverage saturates early (Lite: ~60% of instances cover all forms; Verified: ~29%).

SWE-smith dramatically over-samples easy patterns: **52.7%** of its instances match the easy return-value-change pattern vs 14.5% in Lite. Hard patterns are under-represented: the hardest comparison-check pattern appears in **7.4%** of SWE-smith vs 13.5% of Lite. Collecting more instances from the same distribution repeats what we already know rather than testing what agents struggle with.

**Reference:** `output/strategy_saturation/`, `output/hard_instance_training/pattern_coverage.json`

---

### 5. Hard instances concentrate in specific structural forms

**35/289 instances (12%)** are unsolved by all 84 leaderboard agents. These concentrate in specific structural forms, not random topics.

Intent labels help name what makes hard instances different:
- `algorithm_replace + add_import + signature_change`: **75% unsolved** (full algorithmic redesign with interface changes)
- `add_iteration`: **33% unsolved**
- `add_guard + add_import`: **23% unsolved**
- `api_change` (all variants): **0% unsolved** (API-level fixes are reliably solvable)

---

### 6. Semantic descriptions of any kind cannot identify which instances are hard

We tried clustering instances by every available semantic representation: the bug report text, GPT-4o's predicted fix (before seeing any agent), and GPT-4o's description of the fix after reading the agent's actual behavioral traces. None of them separate difficulty.

| Grouping | Ease variance |
|---|---|
| Issue text, k-means k=10 | 0.0073 |
| GPT-4o predicted fix, k=10 | 0.0087 |
| GPT-4o fix from agent traces, k=10 | 0.0083 |
| AST cert decision tree, 10 forms | 0.0257 |
| **FIM closed itemsets, 15 forms** | **0.0333** |

Variance measures how much the groups differ in difficulty. Higher means the grouping captures real difficulty differences. FIM structural patterns separate difficulty **4.6x better** than the best semantic grouping.

Even the most fix-grounded semantic representation (GPT-4o describing what the fix actually does after seeing the agent's traces) clusters at 0.0083, barely above issue text (0.0073). Structural patterns are the only thing that works.

Nearest-neighbor analysis confirms: knowing an instance's semantic neighbors tells you almost nothing about its difficulty (r=0.081, essentially flat).

**Reference:** `scripts/semantic_vs_structural.py`, `scripts/cluster_fix_descriptions.py`, `output/fix_semantic_clusters/fig_variance_comparison.png`

---

### 7. Agents use different structural approaches to the same problem

When two agents both solve the same instance, they use the same structural approach only 24% of the time (identical edit certificates). The rest of the time there is meaningful structural divergence (median Jaccard similarity = 0.56).

The LLM backbone drives fix strategy, not the scaffolding. SWE-agent with Claude 3.5 Sonnet vs SWE-agent with GPT-4o (same scaffold, different model) has mean Jaccard 0.45 and only 16% identical fixes.

Claude 3.5 Sonnet has the most distinctive vocabulary (85 unique edit operations, lowest pairwise Jaccard). Claude 3 Opus is the most convergent (58 operations, highest pairwise Jaccard).

**Reference:** `output/pairwise_agent_comparison/`, `scripts/pairwise_figures_v2.py`

---

### 8. Agents diverge most at the scope level, not the edit level

Decomposing agent agreement into three layers reveals where agents actually disagree:

| Layer | Agreement |
|---|---|
| File (did they edit the same file?) | 0.55-0.76 |
| Edit type (did they make the same kind of structural change?) | 0.43-0.54 |
| Scope (did they edit the same function/class?) | 0.25-0.31 |

The biggest drop is from file to scope. Agents often find the right file but edit the wrong function within it. This suggests the bottleneck is localization (finding where in the code to apply the fix), not edit strategy (knowing what kind of change to make).

Patch size varies dramatically: Claude 3.5 Sonnet produces patches **6.2x the oracle size** (median). GPT-4o is 2.25x. Claude 3.5 almost always adds extra files when it finds the correct one (141 correct+extra vs 11 correct-only). GPT-4o is more precise (82 correct-only vs 55 correct+extra).

**Reference:** `output/scoped_certificates/`, `output/pairwise_agent_comparison/oracle_alignment.json`

---

### 9. FIM structural patterns are the best grouping for separating difficulty

FIM (frequent itemset mining) on edit certificates finds all frequently co-occurring combinations of edit operations. At support=0.10, it finds 91 closed patterns; 15 with 5+ instances.

**FIM separates difficulty 4.6x better than semantic clustering and 1.3x better than the decision tree** (variance 0.0333 vs 0.0073 vs 0.0257).

**Hardest FIM patterns (ease around 0.02, nearly unsolvable across 84 agents):**
- `{ADD_If, ADD_Compare, ADD_Constant, ADD_Attribute, ADD_Call, ADD_Name}` (n=7): adding a conditional check with a comparison into code that introduces new calls and attribute accesses
- `{ADD_If, ADD_Expr, ADD_Constant, ADD_Attribute, ADD_Call, ADD_Name}` (n=5): same shape with an expression instead of comparison

**Easiest FIM patterns:**
- `{ADD_Assign, ADD_Constant, ADD_Name, DEL_Assign, DEL_Constant, DEL_Name}` (n=5, ease=0.81): variable assignment swap
- `{ADD_Return, DEL_Return}` (n=5, ease=0.67): return value change

The FIM hard instances and the intent-labeled hard instances have **zero overlap**. Two independent methods find two non-overlapping sets of hard instances, meaning the capability gap is larger than either analysis alone shows.

**Reference:** `scripts/build_canonical_forms.py`, `scripts/fim_difficulty_analysis.py`, `output/fim_difficulty/`

---

### 10. The composition of primitives is what makes hard instances hard, not the primitives themselves

**43.8% of agent failures are composition failures**: the agent has individually demonstrated every edit operation the fix requires (in other solved instances) but can't combine them for this instance. This is the single largest failure category.

For the hardest instances (ease below 5%), it's **50.4%**.

The remaining failures split between novel primitives (24.1%, the oracle fix requires an operation the agent has never produced) and familiar patterns (32.1%, the agent has seen this exact combination before but still fails, likely a different kind of difficulty like context length or repository-specific knowledge).

**92 instances** have the specific signature of a composition failure: every required edit operation is common (over 50% of agents have each one in their library), yet ease is below 10%. The parts are universal. The combination is what's hard.

This connects to the FIM hard patterns: `{ADD_If, ADD_Compare, ADD_Constant, ADD_Attribute, ADD_Call, ADD_Name}` has ease=0.02, but ADD_If is in 99% of agents' libraries, ADD_Compare in 98%, ADD_Attribute in 100%, ADD_Call in 100%, ADD_Name in 100%. Every piece is familiar. The six-way combination is not.

**Reference:** `scripts/compositional_generalization.py`, `output/compositional_generalization/`

---

### 11. Form pass rate tracks difficulty closely

Structural form pass rate and overall instance difficulty (mean agent ease) are highly correlated (r=0.971, where 1.0 would be perfect tracking). This means forms are partially measuring problem hardness, not only agent-specific capability gaps.

The frontier claim is strongest where difficulty is controlled: forms where all agents fail but the oracle solution is well-defined, meaning the problem is solvable in principle and the gap is in the agents, not the problem.

---

### 12. Procedural fingerprints identify the agent (leakage-controlled)

A logistic-regression probe over BPE motifs identifies which of **9 agents** produced a trajectory at **85.7%** accuracy (macro-F1 0.85) vs an 11.1% chance baseline (~7.7x), **leakage-controlled** via GroupKFold by task (leakage Δ=−0.007 vs standard CV, so it is style not task memorization). Per-agent F1 is bimodal: deterministic scaffolds (Agentless, DARS F1=1.00; Moatless 0.97) and extended-thinking models (Claude-4, Claude-3.7-thinking F1=0.99) are near-perfectly identifiable; older RLHF-dense models are the most confusable (GPT-4 0.62, Claude-3 0.65, GPT-4o 0.65). Closed-set over known agents. Contrast with code stylometry, which attributes the *artifact* (Bisztray 2025, 97.6%/95.4% over generated C programs); we attribute the *process*, at the step level.

**Reference:** `scripts/backbone_probe_extended.py`, `output/paper2_pilot/backbone_probe_extended.json`

---

### 13. Failure is predictable from a short action prefix, enabling cheap early-abort

A probe on the first *k* canonical actions predicts resolution at **AUC 0.69 from just 3 actions**, rising to 0.72 by step 20 (GroupKFold by task; base resolve rate 0.33). Not a length artifact (resolved/unresolved median length 19 vs 20). A **sequential policy** (re-score the prefix every step, abort at first P(resolve)<τ) **saves ~12% of total compute while retaining ~95% of resolved tasks**, and dominates any fixed decision-step gate across the frontier. Grounds task-aware early-abort / cost control (FrugalGPT/RouteLLM analog at the procedural level).

**Reference:** `scripts/agent_trajectories_paper/early_abort_sequential.py`, `docs/papers/figures/fig_early_abort_frontier.png`

---

### 14. Forward >> reverse faithfulness; thinking models narrate least of what they do

Across families, forward coverage (says→does) is high but reverse coverage (did→says) is not, and it is **lowest for extended-thinking models** (Claude-3.7-thinking, Claude-4 ≈0.20–0.25 median reverse coverage, vs 0.50–0.71 for older models). Many actions thinking models take are never mentioned in their reasoning. (Corrects an earlier "complete reverse faithfulness" reading.)

**Reference:** `output/paper2_pilot/cot_action_alignment_embedding_6agents_summary.json`, `docs/papers/figures/cot_alignment_6agents.png`

---

### 15. The procedural reward is a weak-but-real selector whose headline milestone is empirically inert

The procgrep reward spec (`proc_score`) scored via the shipped library scorer reproduces an inline reference to **within 0.01** (a `from procgrep.reward import load_spec, score` reproducibility check). proc_score weakly predicts resolution (resolved 0.428 vs unresolved 0.396; proc_score best-of-N selection +5.4pp over random). **But its largest component — the `test_verification` milestone (edit→run_test, +0.25) — does not carry that signal:** self-verification is scaffold-determined (Simpson's paradox; within-agent flat-to-negative). proc_score also **misranks capability** across agents — Claude-4, the strongest resolver, scores near-lowest because the canonicalizer under-detects its shell-invoked test runs. So proc_score is a within-scaffold procedural-*style* measure, not a cross-agent capability ranking.

**Reference:** `scripts/agent_trajectories_paper/{proc_score_via_library,test_driven_vs_patchfirst}.py`, `reward_spec.yaml`

---

## Grounding audit (2026-06-08): claims that did NOT reproduce

A pass over the *Agent trajectories as programs* draft separated reproducible results from unsourced ones. The following were flagged and removed/marked-pending in the draft:
- **Prompt-based divergence judges:** the study is real (`output/prompting_study/`, 4 judges, 17–38% "compositionally divergent", κ<0.05), but the exact κ/% table numbers are **not traceable to any κ-computing code** — recompute or cite carefully.
- **Distillation case study numbers** (entropy 2.31→1.99, Jaccard 0.64/0.52, conditional JSD, 0.795/0.940/87.8%-`stuck_reading`): described child trajectories that had not been collected — now being regenerated from public trajectories (see below).
- **Figure 7 V-measure** (canonical 0.290 / native 0.410): no sweep data on disk, contradicts `bpe_vs_prefixspan`'s 0.606/0.626, and there is no native-alphabet field — unsupported as written.
- **Reward YAML / `stuck_reading`:** the printed spec referenced a `think` action absent from the canonicalization, so two of its rules could never fire — reconciled to the validated `reward_spec.yaml`.
- **In-task monitoring "stuck-reading" numbers** (an earlier §6.1: "21 of 23 parent-pass/child-fail fire, complete by step 12 in 80%", "runs 150+ steps without implementing", "835 vs 210 tok/read-cycle, 4×"): re-run on the 499 public child + 284 parent trajectories (`distillation_run/stuck_reading_monitor.py`, 2026-06-08). The 23-instance parent-pass/child-fail denominator is **real**, but the rest is inflated/fabricated — read-run≥4 fires **14/23 (61%)** (16/23 at ≥3, 9/23 at ≥5), onset median step 7 (≤12 in 71%); **max trajectory length 76, zero ≥150**; only **3/23 (13%)** never edit; per-step tokens are **not stored**, and aggregate tokens/step is ~9k for both stuck and non-stuck (no 4× gap, stuck slightly lower). Passage removed; current §6.1 is the grounded early-abort probe. Do **not** re-add the stuck-reading numbers without a per-step-token rollout. The only grounded residue: distilled-child failures show a modest *early read-heavy stall* (61%, onset ~step 7) — optional one-sentence behavioral note, not a headline.

**Reference:** `scripts/agent_trajectories_paper/REFERENCES.md` (per-run inspiration/hypothesis/result log)

---

## Design Decisions Not Taken

**Why not GumTree directly:** Requires AST-level alignment across versions (move tracking). Across a cross-repository benchmark like SWE-bench, "move" in django and "move" in sympy have no meaningful correspondence. The complexity isn't worth it.

**Why not semantic grouping of operations:** Pre-grouping 97 operations into 12 semantic buckets before the decision tree would make splits interpretable but loses discriminative signal at the fine-grained level.

**Why not intent-based forms as the primary grouping:** The "minimal" catch-all absorbs 81% of instances. Intent labels are useful for naming, not grouping.

---

## Open Directions

### A. Pairwise agent comparison (done, 5 agents)

Completed with 5 agents (4 SWE-agent variants + Devin). Key results: median Jaccard 0.56, 24.3% identical fixes, Claude 3.5 Sonnet most structurally diverse. Scaling to more architecturally different agents (Agentless, AutoCodeRover, Aider, OpenHands) would test whether the divergence increases with architectural diversity.

**Reference:** `output/pairwise_agent_comparison/`

### B. Compositional generalization (done, 84 agents)

The 43.8% composition failure rate is the main finding. Next steps:
- Adapt CFQ's compound divergence metric to edit certificates for a formal measure of compositional novelty
- Test on BugsInPy (493 Python bugs) to show the finding isn't SWE-bench-specific
- Decompose composition failures by layer (file/scope/edit) using scoped certificates

### C. Structurally stratified evaluation

The hard-instance training framing document (`output/hard_instance_training/framing.md`) outlines three levels:
1. **Descriptive**: filter existing benchmarks by structural form (done, pattern coverage computed)
2. **Predictive**: predict agent failure from structural distance to the agent's solved library
3. **Generative**: construct or filter instances targeting specific hard patterns

SWE-smith has 1,217 instances matching the hardest pattern (vs 39 in Lite) but proportionally under-represents hard patterns. A structurally stratified benchmark would equalize representation across FIM patterns.

---

## Literature Anchors

| Paper | What it contributes | How we extend it |
|-------|--------------------|--------------------|
| Falleri et al. 2014 (GumTree) | AST diff with Insert/Delete/Move/Update on nodes with parent context | We add parent context without Move/Update; apply to LLM patch analysis |
| Koyuncu et al. 2020 (FixMiner) | FIM on AST edit sequences for repair pattern mining | Same pipeline; evaluate on LLM agent diversity, not repair templates |
| Glassman et al. 2015 (OverCode) | Same correct output, structurally distinct procedures; behavioral observation over self-report | Same argument for LLM agents; edit certificates are our "solution paths" |
| Sobreira et al. 2018 (Defects4J dissection) | 9 repair patterns cover 95% of Defects4J | Validates that a small set of structural patterns covers most fixes; our FIM patterns are the automated equivalent |
| Anthropic Clio | Intent elicitation from behavioral signals, bypassing self-report | Procedural Clio: same argument applied to fix procedures rather than user queries |
| Ellis et al. 2021 (DreamCoder) | Library learning: grow reusable primitives from solved tasks | Each agent has an implicit primitive library; hard instances require out-of-library compositions |
| Grand et al. 2023 (LAPS) | Language-guided program synthesis with library learning | Language descriptions of fix intent; LAPS shows descriptions improve library coverage |
| Lake and Baroni 2018 (SCAN) | Models fail on novel compositions of familiar primitives | Our composition failure finding (43.8%) is the code repair analog |
| Keysers et al. 2020 (CFQ) | Compound divergence metric for compositional generalization | Adaptable to measure compositional novelty of unsolved instances vs agent library |
| Hagele et al. 2026 (Hot Mess of AI) | Bias-variance decomposition of AI errors; variance dominates on hard tasks | Our within-form agent spread (up to 0.42) is a procedural version of their variance finding; grounding failure (F1=0.20) supports incoherence |
| Widyasari et al. 2020 (BugsInPy) | 493 Python bugs with oracle patches across 17 projects | Natural extension dataset; same language, same patch format, directly pipeline-compatible |
| Bisztray et al. 2025 (I Know Which LLM Wrote Your Code) | Code stylometry attributes generated *source* to its author LLM (97.6%/95.4% over 32k C programs) | They fingerprint the artifact; we fingerprint the *process* (trajectory), step-level + leakage-controlled (Finding 12) |
| Chen, Zaharia & Zou 2023 (FrugalGPT) / Ong et al. 2024 (RouteLLM) | Cost-aware LLM cascades / learned routing; cost-vs-quality frontier | Procedural analog: early-abort from a short action prefix (Finding 13) |
| Zheng et al. 2023 (MT-Bench, LLM-as-judge) | Reference-guided, chain-of-thought judging; documents judge instability/bias | Justifies our prompt-classifier baseline design and corroborates its κ<0.05 instability (motivates the structural vocabulary) |
| Yang et al. 2025 (SWE-smith) | SWE-agent-LM-32B distilled from Claude-3.7 trajectories | The teacher→student pair for the distillation case study (Finding pending) |
