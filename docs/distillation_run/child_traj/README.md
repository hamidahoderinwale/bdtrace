---
license: mit
tags: [swe-bench, agents, trajectories, distillation]
---

# bidirect-distillation-traces

Rollout trajectories of SWE-agent-LM-32B (the SWE-smith distillation student) on SWE-bench,
generated for the distillation analysis in "Agent trajectories as programs" (arXiv:2606.16988).

Moved out of the companion repo
[hamidahoderinwale/bidirect-align-dev-traces](https://github.com/hamidahoderinwale/bidirect-align-dev-traces)
(`distillation_run/child_traj/`, last in-tree at commit 5dc8739) so the repo carries code and
findings, not bulk data. Derived fingerprints (`fingerprints_child.jsonl`,
`fingerprints_parent.jsonl`) remain in the repo.

One `.traj` file per SWE-bench instance attempt (SWE-agent format).
