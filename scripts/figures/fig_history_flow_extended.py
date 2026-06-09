"""History Flow figure: file-touch sequences across agents on a single instance.

Shows how each of the 8 agents in the extended corpus traverses files during
its trajectory on django__django-16041. The 4 original SWE-agent backbones all
fail; Claude-3.7-thinking, DARS+R1, Agentless+Claude-3.5, and Moatless+V3
resolve. Cells colored by whether the touched file is in the gold patch.
Inspired by Wattenberg & Viegas 2004 history flow.

Outputs:
    output/figures/fig_history_flow_extended.png
"""
from __future__ import annotations
import json, os, re, sys
from pathlib import Path

import altair as alt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, MAGENTA, OLIVE, GREEN, COPPER
register()

CACHE = ROOT / "output" / "trajectories" / ".cache"
INSTANCE = "django__django-16041"
GOLD_FILE = "django/forms/formsets.py"
OUT = ROOT / "output" / "figures" / "fig_history_flow_extended.png"

AGENTS = [
    ("Claude-3",              "20240402_sweagent_claude3opus",                      "FAIL"),
    ("Claude-3.5",            "20240620_sweagent_claude3.5sonnet",                  "FAIL"),
    ("Claude-3.7-thinking",   "20250226_sweagent_claude-3-7-sonnet-20250219",       "PASS"),
    ("Claude-4",              "20250526_sweagent_claude-4-sonnet-20250514",         "PASS"),
    ("GPT-4",                 "20240402_sweagent_gpt4",                             "FAIL"),
    ("GPT-4o",                "20240728_sweagent_gpt4o",                            "FAIL"),
    ("DARS+R1",               "20250205_dars_agent_claude_3.5_sonnet_deepseek_r1",  "PASS"),
    ("Agentless+Claude-3.5",  "20241202_agentless-1.5_claude-3.5-sonnet-20241022",  "PASS"),
    ("Moatless+V3",           "20250111_moatless_deepseek_v3",                      "PASS"),
]


def normalize(p: str | None) -> str | None:
    if not p:
        return None
    if isinstance(p, str) and p.startswith("/"):
        # strip leading container-mount prefix like /astropy__astropy/ or /testbed/
        parts = p.lstrip("/").split("/", 1)
        return parts[1] if len(parts) > 1 else parts[0]
    return p


def _extract_file_from_action(action: str) -> str | None:
    if not action:
        return None
    # str_replace_editor view/create/str_replace /path/to/file.py
    m = re.search(r"str_replace_editor\s+(?:view|create|str_replace|insert)\s+(\S+\.py)\b", action)
    if m:
        return normalize(m.group(1))
    # SWE-agent classic: "open foo.py", "edit foo.py:..."
    m = re.search(r"\b(?:open|edit|view|create|goto)\s+(\S+\.py)\b", action)
    if m:
        return normalize(m.group(1))
    # python /path/to/file.py
    m = re.search(r"\bpython\s+(\S+\.py)\b", action)
    if m:
        return normalize(m.group(1))
    return None


def extract_sweagent(content: dict) -> list[str | None]:
    out = []
    for step in content.get("trajectory", []):
        st = step.get("state", {})
        if isinstance(st, str):
            try:
                st = json.loads(st)
            except Exception:
                st = {}
        f = st.get("open_file")
        if isinstance(f, str) and f.lower() not in ("n/a", "none"):
            out.append(normalize(f))
            continue
        # fallback: parse action string for file paths
        out.append(_extract_file_from_action(step.get("action") or ""))
    return out


def extract_dars(content: list) -> list[str | None]:
    out = []
    for entry in content:
        action = entry.get("action") or ""
        # DARS actions look like "open foo/bar.py 65" or "edit foo/bar.py:12-20 ..."
        m = re.search(r"\b(?:open|edit|view|create|goto|search_file)\s+([\w./_-]+\.py)", action)
        if m:
            out.append(normalize(m.group(1)))
        else:
            out.append(None)
    return out


def extract_agentless(content: str) -> list[str | None]:
    """Parse Agentless log; capture full module paths in order, dedupe consecutive."""
    # match multi-segment paths ending in .py; include leading dir components
    paths = re.findall(r"\b([\w][\w/_.\-]*\.py)\b", content)
    # filter to plausible module paths (with at least one slash) to avoid bare filenames
    paths = [p for p in paths if "/" in p]
    seen_consecutive = []
    last = None
    for p in paths:
        n = normalize(p)
        if n != last and n is not None:
            seen_consecutive.append(n)
            last = n
    return seen_consecutive[:30]


def walk_moatless(node: dict, out: list, depth: int = 0):
    if depth > 50:
        return
    fc = node.get("file_context") or {}
    files = fc.get("files") or []
    if files:
        # pick first file as the "active" file at this node
        first = files[0]
        if isinstance(first, dict):
            out.append(normalize(first.get("file_path") or first.get("path")))
        else:
            out.append(normalize(first))
    else:
        out.append(None)
    for child in node.get("children") or []:
        walk_moatless(child, out, depth + 1)


def extract_moatless(content: dict) -> list[str | None]:
    out: list[str | None] = []
    root = content.get("root")
    if root:
        walk_moatless(root, out)
    return out


def load_sequence(submission: str) -> list[str | None]:
    path = CACHE / submission / f"{INSTANCE}.json"
    if not path.exists():
        return []
    d = json.load(path.open())
    fmt = d.get("format")
    content = d.get("content", d)
    if fmt == "sweagent_traj_subdir" or "trajectory" in content:
        return extract_sweagent(content)
    if fmt == "dars_traj_list" or isinstance(content, list):
        return extract_dars(content)
    if fmt == "agentless_log_text" or isinstance(content, str):
        return extract_agentless(content)
    if fmt == "moatless_trajectory_json" or "root" in content:
        return extract_moatless(content)
    return []


def build_dataframe() -> tuple[pd.DataFrame, dict]:
    rows = []
    first_gold_step = {}
    for agent, sub, outcome in AGENTS:
        seq = load_sequence(sub)
        # cap rendering to first 30 steps
        seq = seq[:30]
        gold_seen = False
        for step_idx, f in enumerate(seq):
            is_gold = (f == GOLD_FILE) if f else False
            if is_gold and not gold_seen:
                first_gold_step[agent] = step_idx
                gold_seen = True
            rows.append({
                "agent":   agent,
                "outcome": outcome,
                "step":    step_idx,
                "file":    f,
                "is_gold": bool(is_gold),
                "category": (
                    "gold patch file" if is_gold
                    else "other source file" if f
                    else "no file open"
                ),
            })
    return pd.DataFrame(rows), first_gold_step


def main() -> int:
    df, first_gold = build_dataframe()
    if df.empty:
        print("no rows extracted")
        return 1

    agent_order = [a for a, _, _ in AGENTS]
    cscale = alt.Scale(
        domain=["gold patch file", "other source file", "no file open"],
        range=[MAGENTA, OLIVE, "#EFEFEF"],
    )

    base = alt.Chart(df).encode(
        x=alt.X("step:O",
                axis=alt.Axis(title="Trajectory step", labelFontSize=9,
                              domain=False, ticks=False)),
        y=alt.Y("agent:N", sort=agent_order,
                axis=alt.Axis(title=None, labelFontSize=10,
                              domain=False, ticks=False)),
    )
    cells = base.mark_rect(stroke="white", strokeWidth=1.5).encode(
        color=alt.Color("category:N", scale=cscale,
                        legend=alt.Legend(orient="bottom", title=None,
                                           symbolSize=80)),
        tooltip=["agent", "step", "file", "outcome"],
    )

    outcome_df = pd.DataFrame([
        {"agent": a, "outcome": o} for a, _, o in AGENTS
    ])
    outcome_text = (
        alt.Chart(outcome_df)
        .mark_text(align="left", dx=8, fontSize=10, fontWeight=500)
        .encode(
            y=alt.Y("agent:N", sort=agent_order),
            x=alt.value(0),
            text="outcome:N",
            color=alt.Color("outcome:N",
                            scale=alt.Scale(domain=["PASS", "FAIL"],
                                            range=[GREEN, COPPER]),
                            legend=None),
        )
    )

    chart = (
        alt.layer(cells)
        .properties(
            width=480, height=200,
            title=alt.TitleParams(
                text=f"File-touch sequences across agents, instance {INSTANCE}",
                fontSize=12, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    chart.save(str(OUT), scale_factor=2)
    print(f"saved {OUT}")
    print(f"\nfirst step at which agent reached gold file:")
    for a in agent_order:
        if a in first_gold:
            print(f"  {a:24s}: step {first_gold[a]}")
        else:
            print(f"  {a:24s}: never")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
