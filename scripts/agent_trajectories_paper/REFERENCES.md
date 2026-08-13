# References & run inspirations — *Agent trajectories as programs*

Durable record of the prior work motivating each planned/run case study.
**When results for a run are written up, cite the inspiration listed for it** (per
the request on 2026-06-05). IDs marked *(verify)* came from project memory and
should be confirmed against arXiv before they go in the paper.

---

## Routing / cost (case studies ① cascade, ② early-abort)
- **RouteLLM** — Ong et al., LMSYS, arXiv:2406.18665 (2024). Learned router between a
  strong/expensive and weak/cheap model; the canonical *cost-vs-quality frontier* plot
  (resolve/quality vs fraction of expensive-model calls). ~85% cost cut at 95% GPT-4 quality.
- **FrugalGPT** — Chen, Zaharia & Zou, arXiv:2305.05176 (2023). LLM *cascade*: cheap model
  first, escalate on low confidence; prompt adaptation + approximation + cascade. Matches GPT-4
  at up to 98% cost reduction. → direct analog for ① (cheap→expensive escalation).
- **CascadeDebate** — Chang et al., arXiv:2604.12262. Confidence-routed escalation with
  multi-agent deliberation at cascade boundaries; online threshold optimizer.

## Coding-agent behavioral analysis — the NEIGHBOR to position against (do not get scooped)
- **Beyond Resolution Rates: Behavioral Drivers of Coding Agent Success and Failure** —
  arXiv:2604.02547. 9,374 trajectories, **19 agents × 14 LLMs**, complete agent×task matrix on
  SWE-bench Verified; 13-symbol action+environment encoding; failure clusters + behavioral
  patterns that **predict success**. ⇒ "behavior predicts success" is taken; our novelty must be
  the **BPE-motif (subroutine) representation + the cost/route/abort decision**, not the bare claim.
- **Bouzenia & Pradel (2025)** — 8 action types (Explore, Locate, Search, Reproduce, Generate
  fix, Run tests, Refactor, Explain). Per-action-type predecessor. *(verify cite)*
- **Graphectory** — Liu et al. (2026) — 4-step abstraction (Localization, Patching, Validation,
  General). Per-step predecessor. *(verify cite)*

## LLM-as-a-judge (prompt-based-classifier baseline in §3)
- **Zheng et al. 2023** — "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena", NeurIPS 2023,
  arXiv:2306.05685. Foundational; documents position/verbosity/self-enhancement bias; recommends
  reference-guided + chain-of-thought judging + measuring agreement → justifies our judge design.
- **Panickssery, Bowman & Feng 2024** — "LLM Evaluators Recognize and Favor Their Own
  Generations", NeurIPS 2024, arXiv:2404.13076. Self-preference bias → relevant confound: GPT-4o
  is both task-model and judge in our study.
- **Wataoka, Takahashi & Ri 2024** — "Self-Preference Bias in LLM-as-a-Judge", arXiv:2410.21819
  (ICLR 2025). Quantifies self-preference (perplexity mechanism).
- **Chen et al. 2024** — "Humans or LLMs as the Judge? A Study on Judgement Bias", arXiv:2402.10669
  (EMNLP 2024). Frontier judges carry considerable bias → corroborates our κ<0.05 instability.

## Reward / process reward (§reward, and case study ③ reward-hacking)
- **AgentPRM** — arXiv:2502.10325; follow-up arXiv:2511.08325. *(verify)*
- **SWE-TRACE** — arXiv:2604.14820. *(verify)*
- **DataPRM** — arXiv:2604.24198. *(verify)*
- **Principle Process Reward** — ICLR 2026. *(verify)*
  → motivate partial/procedural reward over binary pass/fail; ③ tests whether the procedural
  reward (`proc_score`) or a divergence probe separates reward-hacking (resolve-by-test-edit)
  from genuine fixes.

---

## Planned runs → inspiration to cite (+ hypothesis, fill result later)

| Run | Inspiration to cite | Hypothesis | Result |
|---|---|---|---|
| ② Early-abort (prefix-*k* failure prediction; compute saved vs resolves lost) | FrugalGPT (2305.05176); contrast per-step predecessors 2604.02547 / Bouzenia&Pradel / Graphectory | Failing trajectories are distinguishable from a short prefix → compute can be cut before `submit` | _pending_ |
| ① Cheap→expensive cascade (SWE-LM-32B → Claude-4 on fingerprint-predicted failure) | FrugalGPT (2305.05176) + RouteLLM (2406.18665); CascadeDebate (2604.12262) | Fingerprint-triggered escalation beats fixed-tier on the resolve@cost frontier | _pending_ |
| ③ Reward-hacking detection (test-edit vs source-only resolved patches; proc_score / divergence separation) | process-reward refs above; SWE-smith cookbook | Resolve-by-test-edit trajectories are procedurally separable from genuine fixes | **scoped 2026-06-05: NOT groundable as framed.** (1) "test-file" edits are mostly the agent's own scratch reproduction scripts (test-driven dev), not repo-test tampering (Claude-4: 136 scratch vs 26 repo-test); (2) cached trajectories don't isolate the agent's final patch and embed the gold `test_patch` → repo-test counts are contaminated artifacts; (3) SWE-bench harness re-applies the gold test_patch and resets graded tests, so test-edits can't hack the reward. → reframed to ③′ |
| ③′ Test-driven vs patch-first procedures (does `proc_score` / structure separate self-verifying trajectories, and does self-verification predict resolution) | process-reward refs above; validates the `proc_score` test_verification milestone; corroborates scaffold-tribes | Trajectories that create+run their own reproduction/test resolve more often, and `proc_score` separates them | **ran 2026-06-05 (`test_driven_vs_patchfirst.py`): hypothesis NOT supported, and aggregate is confounded.** Aggregate test-driven 27.5% (n=1666) vs patch-first 43.2% (n=944), Δ=−15.7pp — but the behavior is **scaffold-determined** (Agentless 300/0 test-driven; Moatless 1/288 + sweagent-Claude-4 4/283 patch-first), so the aggregate compares scaffolds, and the strong agents sit in the patch-first bucket (Simpson's paradox). Within-agent the effect is flat-to-slightly-negative (Claude-3.5 +5.9, Claude-3.7 +0.9, DARS −3.3, Claude-3 −13.5, GPT-4 −16.5). ⇒ self-verification does **not** observationally predict resolution here; the `proc_score` edit→run_test milestone (its largest +0.25 component) is a **design assumption, not empirically validated** in this corpus. Cleanly corroborates scaffold-tribes (procedure is scaffold-set, not a free per-trajectory choice). **(a)** proc_score *does* separate the groups (test-driven 0.473 vs patch-first 0.289) but ~tautologically (the +0.25 edit→run_test term defines "test-driven"). proc_score→resolution is only **weakly** positive: resolved 0.428 vs unresolved 0.396 (Δ+0.032), consistent with the +5.4pp best-of-N selection lift. ⇒ the reward is a weak-but-real selector, yet its weak signal is **not** carried by its headline test-verification milestone. |

### Already run (reward selection, option a — 2026-06-05)
`reward_verifier.py`, n=300 instances with ≥3 agents:
proc_score best-of-N **38.3%** · random 32.9% · worst 23.7% · always-best-agent (Claude-4) **54.0%** · oracle 66.7%.
Reading: procedural reward beats random/worst but is dominated by "always pick the strongest agent" —
the FrugalGPT/RouteLLM lesson (a router only earns its keep on the *cost* axis, which ① will test).
