"""Generate output/figures/controlled_eval_satisfaction_altair.png.

Faceted horizontal bar chart showing behavioral adherence rate per agent × constraint.
- 4 panels (one per constraint)
- Y axis: agent name (no title), ordered by AGENT_ORDER
- X axis: 0–100% satisfaction rate
- Bars colored by AGENT_COLORS
- Title: "Behavioral adherence by agent"
- Scale factor 2 for high-res output
- Vision-LLM (OpenRouter anthropic/claude-sonnet-4-6) verification before finishing
"""

import base64
import json
import math
import os
import sys
import urllib.request

import altair as alt
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = "/Users/hamidaho/learning-from-dev/bidirect-align-dev-traces"
DATA_PATH = os.path.join(BASE_DIR, "output/paper2_pilot/controlled_eval_results.json")
OUT_PATH  = os.path.join(BASE_DIR, "output/figures/controlled_eval_satisfaction_altair.png")

sys.path.insert(0, BASE_DIR)
from scripts.theme import register, AGENT_COLORS, AGENT_ORDER

register()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# ---------------------------------------------------------------------------
# Load data and compute satisfaction rates
# ---------------------------------------------------------------------------
with open(DATA_PATH) as f:
    raw = json.load(f)

CONSTRAINTS = ["Test before submit", "Repro step", "Search before edit", "Low edit retry rate"]

rows = []
for constraint in CONSTRAINTS:
    agents_data = raw[constraint]
    for agent in AGENT_ORDER:
        if agent not in agents_data:
            continue
        d = agents_data[agent]
        n_satisfies = d["n_satisfies"]
        n_not = d["n_not"]
        total = n_satisfies + n_not
        if total == 0 or (math.isnan(n_satisfies) if isinstance(n_satisfies, float) else False):
            rate = float("nan")
        else:
            rate = n_satisfies / total
        rows.append({
            "constraint": constraint,
            "agent": agent,
            "satisfaction_rate": rate,
            "satisfaction_pct": rate * 100 if not (isinstance(rate, float) and math.isnan(rate)) else float("nan"),
        })

df = pd.DataFrame(rows)
# Drop NaN rows (agents that have no data for a constraint)
df = df.dropna(subset=["satisfaction_pct"])

# ---------------------------------------------------------------------------
# Build color encoding
# ---------------------------------------------------------------------------
color_domain = AGENT_ORDER
color_range  = [AGENT_COLORS[a] for a in color_domain]

# Agent sort order (reversed so top of chart = first in AGENT_ORDER)
agent_sort = list(reversed(AGENT_ORDER))

# ---------------------------------------------------------------------------
# Build chart
# ---------------------------------------------------------------------------
def build_chart() -> alt.Chart:
    base = alt.Chart(df).mark_bar(cornerRadiusEnd=2).encode(
        x=alt.X(
            "satisfaction_pct:Q",
            title=None,
            scale=alt.Scale(domain=[0, 100]),
            axis=alt.Axis(
                values=[0, 25, 50, 75, 100],
                format=".0f",
                labelExpr="datum.value + '%'",
                labelFontSize=10,
                grid=False,
                domain=False,
                ticks=False,
            ),
        ),
        y=alt.Y(
            "agent:N",
            title=None,
            sort=agent_sort,
            axis=alt.Axis(
                labelFontSize=10,
                labelLimit=200,
                domain=False,
                ticks=False,
            ),
        ),
        color=alt.Color(
            "agent:N",
            scale=alt.Scale(domain=color_domain, range=color_range),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("agent:N", title="Agent"),
            alt.Tooltip("constraint:N", title="Constraint"),
            alt.Tooltip("satisfaction_pct:Q", title="Rate (%)", format=".1f"),
        ],
    ).properties(
        width=180,
        height=180,
    ).facet(
        facet=alt.Facet(
            "constraint:N",
            title=None,
            sort=CONSTRAINTS,
            header=alt.Header(
                labelFontSize=11,
                labelFontWeight="normal",
                labelColor="#222222",
                labelOrient="top",
                title=None,
            ),
        ),
        columns=2,
    ).properties(
        title=alt.TitleParams(
            text="Behavioral adherence by agent",
            fontSize=14,
            fontWeight="normal",
            color="#111111",
            anchor="start",
            offset=12,
        ),
    ).configure_view(
        stroke=None,
    ).configure_facet(
        spacing=20,
    )

    return base


# ---------------------------------------------------------------------------
# Vision-LLM verification
# ---------------------------------------------------------------------------
def verify_chart(image_path: str) -> tuple[bool, str]:
    """Call vision-LLM and check the 4 quality criteria."""
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    payload = json.dumps({
        "model": "anthropic/claude-sonnet-4-6",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": (
                            "Look at this faceted bar chart figure carefully. "
                            "Answer the following four questions:\n\n"
                            "1. READABLE: Is the chart readable? Are agent labels and panel titles legible?\n"
                            "2. COLORS: Are the bar colors distinguishable from each other?\n"
                            "3. TITLE: Is the title 'Behavioral adherence by agent' (or similar) appropriate "
                            "for a chart showing fraction of agent trajectories that follow each behavioral pattern?\n"
                            "4. PROFESSIONAL: Does the chart look clean and professional?\n\n"
                            "For each, answer YES or NO and give a one-sentence reason. "
                            "Then give an overall verdict: PASS if all four are YES, FAIL otherwise. "
                            "Format:\n"
                            "READABLE: YES/NO — reason\n"
                            "COLORS: YES/NO — reason\n"
                            "TITLE: YES/NO — reason\n"
                            "PROFESSIONAL: YES/NO — reason\n"
                            "VERDICT: PASS/FAIL"
                        )
                    }
                ]
            }
        ],
        "max_tokens": 500,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://thetaste.ai",
            "X-Title": "bidirect-align-figures",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
        text = result["choices"][0]["message"]["content"].strip()
        passed = "VERDICT: PASS" in text.upper()
        return passed, text
    except Exception as e:
        print(f"Vision-LLM call failed: {e}")
        return False, str(e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Building chart...")
    chart = build_chart()

    print(f"Saving to {OUT_PATH} ...")
    chart.save(OUT_PATH, scale_factor=2)
    print("Saved.")

    print("\nRunning vision-LLM verification...")
    passed, explanation = verify_chart(OUT_PATH)
    print(f"\nVision-LLM verdict: {'PASS' if passed else 'FAIL'}")
    print(f"Detail:\n{explanation}")

    if passed:
        print(f"\nDone — chart passes quality check. Output: {OUT_PATH}")
    else:
        print(f"\nWarning: chart may have issues. See vision-LLM detail above. Output still saved: {OUT_PATH}")
