"""Per-agent stuck-edit-loop detection rate via procgrep CLI.

Generates the §6.5 demo figure: bar chart of stuck-edit-loop rate per agent
across the 9-submission corpus. Exposes the counterintuitive finding that
the signature is base-model-specific, not paradigm-wide.

Reads:
    output/trajectories/.cache/<submission>/*.json (envelopes)
Writes:
    output/figures/fig_procgrep_results.png
    output/paper2_pilot/procgrep_results.json

Usage:
    uv run python scripts/figures/fig_procgrep_results.py
"""

from __future__ import annotations

import json
import sys
import subprocess
from pathlib import Path

import altair as alt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, GREEN, BLUE, MAGENTA, COPPER, OLIVE, GREEN_D, BLUE_D, MAGENTA_D
register()

CACHE = ROOT / "output" / "trajectories" / ".cache"
PROCGREP = ROOT / "scripts" / "tools" / "procgrep.py"
OUT_FIG = ROOT / "output" / "figures"
OUT_DAT = ROOT / "output" / "paper2_pilot"

SUBMISSION_LABEL = {
    "20240402_sweagent_claude3opus":                          "Claude-3",
    "20240402_sweagent_gpt4":                                 "GPT-4",
    "20240620_sweagent_claude3.5sonnet":                      "Claude-3.5",
    "20240728_sweagent_gpt4o":                                "GPT-4o",
    "20250226_sweagent_claude-3-7-sonnet-20250219":           "Claude-3.7-thinking",
    "20250526_sweagent_claude-4-sonnet-20250514":             "Claude-4",
    "20250205_dars_agent_claude_3.5_sonnet_deepseek_r1":      "DARS+R1",
    "20241202_agentless-1.5_claude-3.5-sonnet-20241022":      "Agentless+Claude-3.5",
    "20250111_moatless_deepseek_v3":                          "Moatless+V3",
}
AGENT_ORDER = [
    "Claude-3", "Claude-3.5", "Claude-3.7-thinking", "Claude-4",
    "GPT-4", "GPT-4o",
    "DARS+R1", "Agentless+Claude-3.5", "Moatless+V3",
]
AGENT_COLORS = {
    "Claude-3":              COPPER,
    "Claude-3.5":            GREEN,
    "Claude-3.7-thinking":   GREEN_D,
    "Claude-4":              "#187860",
    "GPT-4":                 BLUE,
    "GPT-4o":                MAGENTA,
    "DARS+R1":               MAGENTA_D,
    "Agentless+Claude-3.5":  BLUE_D,
    "Moatless+V3":           OLIVE,
}


def run_procgrep(submission_dir: Path) -> dict:
    """Run procgrep --summary on a directory; parse the summary JSON."""
    result = subprocess.run(
        ["uv", "run", "python", str(PROCGREP), str(submission_dir), "--summary"],
        capture_output=True, text=True, timeout=120,
    )
    # Summary is on stderr (per procgrep convention)
    if result.returncode != 0 and not result.stderr.strip().startswith("{"):
        return {"error": result.stderr[:200]}
    try:
        return json.loads(result.stderr)
    except Exception:
        return {"error": "could not parse summary"}


def main() -> None:
    rows = []
    for sub_id, agent in SUBMISSION_LABEL.items():
        sub_dir = CACHE / sub_id
        if not sub_dir.is_dir():
            continue
        print(f"running procgrep on {agent} ...")
        summary = run_procgrep(sub_dir)
        if "error" in summary:
            print(f"  error: {summary['error']}")
            continue
        rows.append({
            "agent": agent,
            "n_total":           summary.get("n_total", 0),
            "n_stuck_edit_loop": summary.get("n_stuck_edit_loop", 0),
            "n_normal":          summary.get("n_normal", 0),
            "n_short":           summary.get("n_short", 0),
            "n_no_localization": summary.get("n_no_localization", 0),
            "rate":              summary.get("stuck_edit_loop_rate", 0.0),
        })

    df = pd.DataFrame(rows).sort_values(
        "agent", key=lambda s: [AGENT_ORDER.index(a) if a in AGENT_ORDER else len(AGENT_ORDER) for a in s]
    )
    print("\n=== per-agent stuck-edit-loop rate ===")
    print(df.to_string(index=False))

    out_json = OUT_DAT / "procgrep_results.json"
    out_json.write_text(df.to_json(orient="records", indent=2))
    print(f"\nSaved {out_json}")

    color_scale = alt.Scale(
        domain=AGENT_ORDER,
        range=[AGENT_COLORS[a] for a in AGENT_ORDER],
    )

    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("rate:Q",
                    axis=alt.Axis(title="Stuck-edit-loop rate",
                                  domain=False, ticks=False, format=".0%", labelFontSize=10)),
            y=alt.Y("agent:N",
                    sort=AGENT_ORDER,
                    axis=alt.Axis(title=None, domain=False, ticks=False,
                                  labelFontSize=10, labelLimit=240)),
            color=alt.Color("agent:N", scale=color_scale, legend=None),
            tooltip=["agent", "n_total", "n_stuck_edit_loop", "rate"],
        )
    )
    pct_labels = (
        alt.Chart(df)
        .mark_text(align="left", dx=4, fontSize=10, color="#444444")
        .encode(
            x="rate:Q",
            y=alt.Y("agent:N", sort=AGENT_ORDER),
            text=alt.Text("rate:Q", format=".1%"),
        )
    )

    chart = (
        (bars + pct_labels)
        .properties(
            width=440,
            height=max(220, 32 * len(df)),
            title=alt.TitleParams(
                text="Stuck-edit-loop rate per agent  ·  procgrep CLI demo",
                fontSize=12, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
    )
    out_png = OUT_FIG / "fig_procgrep_results.png"
    chart.save(str(out_png), scale_factor=2)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
