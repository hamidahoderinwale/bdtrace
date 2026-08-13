"""Pass rate vs. number of files in the agent's submitted patch, per agent.

Complements the file-breadth paragraph: shows the per-agent "tipping point" where
touching more files in the fix turns from neutral to a failure signal. Uses the
submitted diff (info['submission']) in the raw .traj cache -- consistent across all
scaffolds (incl. DARS+R1). Run from repo root.
"""
import json
import re
import glob
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import altair as alt
import sys
sys.path.insert(0, ".")
from scripts.theme import register, COPPER, BLUE, GREEN, OLIVE, MAGENTA

register()
CACHE = Path("output/trajectories/.cache")
pf = json.load(open("output/paper2_pilot/extended_pass_fail.json"))

# pass_fail key (== cache dir name) -> display agent
NAME = {
    "20240402_sweagent_claude3opus": "Claude-3",
    "20240620_sweagent_claude3.5sonnet": "Claude-3.5",
    "20250226_sweagent_claude-3-7-sonnet-20250219": "Claude-3.7-thinking",
    "20250526_sweagent_claude-4-sonnet-20250514": "Claude-4",
    "20240402_sweagent_gpt4": "GPT-4",
    "20240728_sweagent_gpt4o": "GPT-4o",
    "20250205_dars_agent_claude_3.5_sonnet_deepseek_r1": "DARS+R1",
    "20241202_agentless-1.5_claude-3.5-sonnet-20241022": "Agentless+Claude-3.5",
    "20250111_moatless_deepseek_v3": "Moatless+V3",
}
GIT = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.M)
PLUS = re.compile(r"^\+\+\+ b/(\S+)", re.M)


def patch_files(diff: str) -> int:
    if not diff or not diff.strip():
        return 0
    files = set(m.group(2) for m in GIT.finditer(diff))
    if not files:
        files = set(m.group(1) for m in PLUS.finditer(diff))
    return len(files)


def binof(n):
    if n <= 0:
        return None
    return "1" if n == 1 else ("2" if n == 2 else "3+")


rows = []
means = {}
for key, agent in NAME.items():
    d = CACHE / key
    if not d.exists():
        print(f"MISSING dir {key}")
        continue
    resolved = set(pf.get(key, {}).get("resolved", []))
    cnts = []
    cell = defaultdict(lambda: [0, 0])  # bin -> [passed, total]
    for fp in glob.glob(str(d / "*.json")):
        iid = os.path.basename(fp)[:-5]
        try:
            o = json.load(open(fp))
        except Exception:
            continue
        sub = o.get("info", {}).get("submission", "") or ""
        if not sub.strip() and isinstance(o.get("content"), dict):
            sub = o["content"].get("info", {}).get("submission", "") or ""
        if not sub.strip() and isinstance(o.get("content"), list):
            # dars_traj_list / message-list formats: final patch is the last message with a diff
            for m in reversed(o["content"]):
                t = str(m.get("content", "")) if isinstance(m, dict) else str(m)
                if "diff --git" in t:
                    sub = t
                    break
        if not sub.strip():
            sub = o.get("submission", "") or ""
        nf = patch_files(sub)
        b = binof(nf)
        if b is None:
            continue
        cnts.append(nf)
        ok = iid in resolved
        cell[b][1] += 1
        cell[b][0] += int(ok)
    means[agent] = float(np.mean(cnts)) if cnts else float("nan")
    for b in ("1", "2", "3+"):
        p, t = cell[b]
        if t >= 5:  # suppress tiny bins
            rows.append({"agent": agent, "bin": b, "pass_rate": 100 * p / t, "n": t})

df = pd.DataFrame(rows)
print("mean files-in-patch per agent:")
for a, m in sorted(means.items(), key=lambda x: -x[1]):
    print(f"  {a:22s} {m:.2f}")
print()
print(df.pivot(index="agent", columns="bin", values="pass_rate").round(1).to_string())
print()
print("bin sizes:")
print(df.pivot(index="agent", columns="bin", values="n").to_string())

# family hue, darker = newer era; only agents with data
FAM = {"Claude-3": "#fdae6b", "Claude-3.5": "#e6701a", "Claude-3.7-thinking": "#a63603", "Claude-4": "#7f2704",
       "GPT-4": "#9ecae1", "GPT-4o": "#3182bd", "DARS+R1": "#2ca25f"}
df = df[df.agent.isin(FAM)]
dom = [a for a in FAM if a in set(df.agent)]
order = ["1", "2", "3+"]
line = alt.Chart(df).mark_line(point=True, strokeWidth=2).encode(
    x=alt.X("bin:N", sort=order, title="Files in submitted patch",
            axis=alt.Axis(domain=False, ticks=False, labelAngle=0)),
    y=alt.Y("pass_rate:Q", title="Pass rate %", scale=alt.Scale(domain=[0, 70]),
            axis=alt.Axis(domain=False, ticks=False)),
    color=alt.Color("agent:N", scale=alt.Scale(domain=dom, range=[FAM[a] for a in dom]),
                    legend=alt.Legend(title="Agent, darker = newer")),
    detail="agent:N",
).properties(width=380, height=270, title="Pass rate by number of files changed")
out = Path("docs/papers/figures/fig_patch_files_passrate.png")
line.save(str(out), scale_factor=2)
Path("output/paper2_pilot/patch_files_passrate.json").write_text(
    json.dumps({"means": means, "rows": rows}, indent=2))
print(f"\nwrote {out}")
