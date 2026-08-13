"""Per-agent signature transition: the step-to-step move each agent over-uses most
relative to its peers. Replaces the (ungrounded) fig13 discriminative-bigrams figure.

Discrimination factor = agent's share of a transition / mean share across OTHER agents
(leave-one-out). The 'other' catch-all bucket (shell misc.) is excluded so transitions
are interpretable; this is disclosed in the caption. Run from repo root.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

def load(f): return [json.loads(l) for l in open(f)]
par = load("distillation_run/fingerprints_parent.jsonl")
chi = load("distillation_run/fingerprints_child.jsonl")

# unambiguous fine->coarse map from the aligned parent+child fingerprint files
m = defaultdict(Counter)
for r in par + chi:
    for fa, co in zip(r["native"], r["canonical"]):
        m[fa][co] += 1
f2c = {fa: c.most_common(1)[0][0] for fa, c in m.items()}
f2c["SHELL_AWK"] = "other"

seqs = defaultdict(list)
for l in open("output/paper2_pilot/bpe_sequences_extended.jsonl"):
    r = json.loads(l)
    seqs[r["agent"]].append([f2c.get(a, "other") for a in r["canonical"]])
seqs["SWE-agent-LM-32B"] = [r["canonical"] for r in chi]
seqs["Claude-3.7-parent"] = [r["canonical"] for r in par]  # for lineage check only

def dist(ss):
    c = Counter()
    for s in ss:
        for i in range(len(s) - 1):
            if "other" in (s[i], s[i + 1]):
                continue
            c[(s[i], s[i + 1])] += 1
    t = sum(c.values()) or 1
    return {k: v / t for k, v in c.items()}

D = {a: dist(ss) for a, ss in seqs.items()}
allbg = set().union(*[set(d) for d in D.values()])

PANEL = ["Claude-3", "Claude-3.5", "Claude-3.7-thinking", "Claude-4", "GPT-4",
         "GPT-4o", "DARS+R1", "Agentless+Claude-3.5", "Moatless+V3", "SWE-agent-LM-32B"]

def top(agent, peers):
    base = {b: (sum(D[o].get(b, 0) for o in peers) / len(peers)) for b in allbg}
    cand = [(D[agent].get(b, 0) / (base[b] + 1e-9), D[agent].get(b, 0), b)
            for b in allbg if D[agent].get(b, 0) >= 0.02]
    return max(cand, default=(0, 0, None))

rows = []
for ag in PANEL:
    peers = [a for a in PANEL if a != ag]  # leave-one-out, panel agents only
    f, p, b = top(ag, peers)
    rows.append({"agent": ag, "transition": f"{b[0]}->{b[1]}", "factor": round(f, 1), "share": round(p, 3)})
    print(f"{ag:22s} {b[0]:>11s}->{b[1]:<11s}  factor={f:5.1f}x  share={p:.3f}")

Path("output/paper2_pilot/signature_transitions.json").write_text(json.dumps(rows, indent=2))
print("\nwrote output/paper2_pilot/signature_transitions.json")
