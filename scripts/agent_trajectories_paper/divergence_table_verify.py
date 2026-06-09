"""Re-derive and verify tab:divergence_by_model from persisted judge labels.

The divergence labels ARE persisted at records[i]['conditions'][cond]['scores']
['divergence_level'] (surface|compositional|relational|none). This re-derives the
table's %compositional and inter-judge kappa from them, compares to the printed
table, and persists the result. Run from repo root.
"""
import json
from pathlib import Path
from collections import Counter
from itertools import combinations

from sklearn.metrics import cohen_kappa_score

PS = Path("output/prompting_study")
JUDGES = {
    "GPT-4o": "gpt_4o",
    "GPT-4o mini": "gpt_4o_mini",
    "Qwen 2.5 72B": "qwen_2.5_72b_instruct",
    "Llama 3.3 70B": "llama_3.3_70b_instruct",
}
# printed table: (n, %compositional, mean kappa)
TAB = {"GPT-4o": (286, 38.1, 0.124), "GPT-4o mini": (285, 36.5, 0.144),
       "Qwen 2.5 72B": (87, 17.2, 0.128), "Llama 3.3 70B": (87, 21.8, 0.123)}
COND = "no_context"  # the condition whose compositional rate matches the table


def load(slug):
    recs = json.load(open(PS / slug / "records.json"))
    # instance_id -> divergence_level (no_context)
    return {r["instance_id"]: r["conditions"].get(COND, {}).get("scores", {}).get("divergence_level", "none")
            for r in recs}


labels = {name: load(slug) for name, slug in JUDGES.items()}

# %compositional per judge
pct = {}
for name, lab in labels.items():
    n = len(lab)
    comp = sum(1 for v in lab.values() if v == "compositional")
    pct[name] = 100 * comp / n if n else 0.0

# pairwise Cohen kappa on shared instances; mean per judge
pair_kappa = {}
for a, b in combinations(JUDGES, 2):
    shared = sorted(set(labels[a]) & set(labels[b]))
    if len(shared) < 5:
        continue
    ya = [labels[a][i] for i in shared]
    yb = [labels[b][i] for i in shared]
    k = cohen_kappa_score(ya, yb)
    pair_kappa[(a, b)] = (k, len(shared))

mean_kappa = {}
for name in JUDGES:
    ks = [k for (a, b), (k, _) in pair_kappa.items() if name in (a, b)]
    mean_kappa[name] = sum(ks) / len(ks) if ks else float("nan")

print(f"condition for %%compositional = {COND}\n")
print(f"{'judge':14s} {'n':>4s} {'%comp':>7s} {'%comp_TAB':>10s} {'meanK':>7s} {'K_TAB':>7s}")
for name in JUDGES:
    n = len(labels[name])
    t = TAB[name]
    print(f"{name:14s} {n:4d} {pct[name]:7.1f} {t[1]:10.1f} {mean_kappa[name]:7.3f} {t[2]:7.3f}"
          + ("" if abs(pct[name]-t[1]) <= 0.6 else "  <-%comp DIFF")
          + ("" if abs(mean_kappa[name]-t[2]) <= 0.02 else "  <-kappa DIFF"))

print("\npairwise kappa (judge_a, judge_b): kappa (n_shared)")
for (a, b), (k, n) in sorted(pair_kappa.items()):
    print(f"  {a:14s} x {b:14s}: {k:+.3f} (n={n})")

# overall agreement among the 4 (Fleiss-style proxy: mean of all pairwise)
allk = [k for (k, _) in pair_kappa.values()]
print(f"\nmean of ALL pairwise kappa (cross-judge agreement) = {sum(allk)/len(allk):+.3f}")
print("caption claims kappa < 0.05; table column 'Mean kappa' ~0.12-0.14 -- check which the caption refers to.")

out = {"condition": COND,
       "pct_compositional": {k: round(v, 1) for k, v in pct.items()},
       "mean_kappa": {k: round(v, 3) for k, v in mean_kappa.items()},
       "pairwise_kappa": {f"{a}|{b}": round(k, 3) for (a, b), (k, _) in pair_kappa.items()},
       "mean_all_pairwise_kappa": round(sum(allk)/len(allk), 3)}
Path("output/paper2_pilot/divergence_table_verify.json").write_text(json.dumps(out, indent=2))
print("\nwrote output/paper2_pilot/divergence_table_verify.json")
