# EXP — Structural query vs LLM classification of trajectory properties

**Status:** registered, gated on API spend.
**Consolidates:** the existing prompt-classifier experiment (4 judges label
"compositional divergence", cross-family κ < 0.05). This subsumes it and adds
the speed / cost / accuracy axes the fuzzy-only setup couldn't measure.

## Question
Can an LLM judge replace a deterministic structural query for answering
behavioural questions about a trajectory ("did it run 5+ edits in a row?",
"did it submit without testing?", "is it compositionally divergent?")?

## Hypotheses
- **H1 (structural predicates).** For predicates computable from the action
  sequence, procgrep answers **exactly**, in ~µs/trace, at **$0**. LLM judges are
  ≥10³× slower, cost real $/trace, and are **<100% accurate** vs the deterministic
  ground truth (LLMs miscount and lose order over long traces).
- **H2 (fuzzy predicate).** For a semantic property (compositional divergence)
  there is no deterministic ground truth; LLM judges stay unreliable (κ < 0.05).
  procgrep can't answer it either — but it is at least reproducible.
- **Unified claim.** The LLM classifier is **dominated** — on speed/cost (and
  likely accuracy) for structural predicates, on reliability for fuzzy ones.

## Design
- **Corpus.** Random N = 150 trajectories from the 9-agent SWE-bench-Lite set
  (canonical atoms available; seed 0).
- **Predicates (P = 6).** Structural (deterministic ground truth from atoms):
  `edit-streak ≥5`, `tested-before-first-edit`, `submitted-without-testing`,
  `stuck-reading`, `never-searched-repo`. Fuzzy (no ground truth):
  `compositionally-divergent` (reuse the paper's existing judge prompt).
- **Method A — procgrep.** `match_patterns` / structural predicate over atoms →
  label; record total wall-time for all N (single CPU process).
- **Method B — LLM judges (J = 4).** Four models via OpenRouter; per
  (trace, predicate) a yes/no prompt with the trace's serialized actions and the
  *explicit operational definition* of the predicate; record label, latency, cost.

## Metrics
- **latency** — ms/trace (procgrep total ÷ N vs LLM per-call).
- **cost** — $/1,000 traces.
- **accuracy** — LLM vs procgrep ground truth on structural predicates
  (precision/recall/F1).
- **reliability** — inter-judge κ on every predicate (this is the only axis the
  fuzzy predicate supports).

## Ship
One consolidated table — rows = predicates (structural → fuzzy), columns =
{procgrep: latency, cost, exact?} × {LLM: latency, cost, accuracy, κ}. It
**replaces** the standalone classifier box in the paper. Headline: the
LLM-classifier route is dominated either way.

## Falsification
- If LLM accuracy ≥ 99% on *all* structural predicates **and** κ > 0.6 on the
  fuzzy one → the "unreliable/inaccurate" claim collapses; procgrep's edge
  reduces to speed/cost only (real, but weaker).
- If a reviewer calls the structural predicates "trivially defined" → they are
  the *operational* definitions, and the LLM is asked the identical operational
  question, so the comparison is fair.

## Cost / guardrails
- N=150 × P=6 × J=4 = **3,600 judge calls**. At ~$0.001–0.005/call (cheap models,
  short prompts) ≈ **$4–18**. Smoke on N=10 first; cap models + N; OpenRouter, no
  GPU. Honors the cloud-cost guardrails.

## Honest caveat
Structural ground truth *is* procgrep by construction, so "LLM accuracy" measures
whether a paid model can reproduce a free exact computation — the load-bearing
axes there are speed/cost (always) and whether the LLM is even accurate (the
empirical bet). The genuine LLM failure is the fuzzy κ.

## Artifacts (on greenlight)
- `scripts/agent_trajectories_paper/query_vs_llm.py` — runner (procgrep timing +
  OpenRouter judge loop + κ/accuracy + table emit; results to JSON).
