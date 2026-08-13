"""Figure 1 vignette: two agents, one task, three levels.

Shows action sequence, outcome, and the contrast that pass/fail collapses.
Usage:
    python -m scripts.paper_vignette
"""

from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import altair as alt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.theme import register, STAGE_COLORS, STAGE_ORDER, GREEN, GRAY, NEAR_BLACK

register()

OUT = PROJECT_ROOT / "output" / "paper2_pilot"

ACTION_STAGE: dict[str, str] = {
    # Explore
    "SEARCH": "Explore", "FIND_FILE": "Explore", "SHELL_LS": "Explore",
    "SHELL_CD": "Explore", "SHELL_CAT": "Explore", "SHELL_GREP": "Explore",
    "SHELL_MKDIR": "Explore", "SHELL_PWD": "Explore", "SHELL_CP": "Explore",
    "SHELL_MV": "Explore", "SHELL_ECHO": "Explore", "SHELL_EXPORT": "Explore",
    "SHELL_CHMOD": "Explore", "SHELL_TOUCH": "Explore",
    # Browse
    "OPEN_SRC_PY": "Browse", "NAV_SRC_PY": "Browse",
    "OPEN_TEST_PY": "Browse", "NAV_TEST_PY": "Browse",
    "OPEN_OTHER": "Browse", "NAV_OTHER": "Browse",
    "OPEN_CONFIG_PY": "Browse", "NAV_CONFIG_PY": "Browse",
    "OPEN_REPRO_PY": "Browse", "NAV_REPRO_PY": "Browse",
    "OPEN_DOC": "Browse", "NAV_DOC": "Browse",
    "OPEN_UNKNOWN": "Browse",
    # Edit
    "EDIT_SRC_PY": "Edit", "EDIT_TEST_PY": "Edit",
    "EDIT_REPRO_PY": "Edit", "EDIT_CONFIG_PY": "Edit",
    "EDIT_OTHER": "Edit", "EDIT_DOC": "Edit",
    "CREATE_TEST_PY": "Edit", "CREATE_SRC_PY": "Edit",
    "CREATE_CONFIG_PY": "Edit", "CREATE_REPRO_PY": "Edit",
    "CREATE_OTHER": "Edit", "CREATE_UNKNOWN": "Edit",
    "CREATE_DOC": "Edit", "EDIT": "Edit",
    # Test
    "RUN_PYTHON_SRC_PY": "Test", "RUN_PYTHON_ALL": "Test",
    "RUN_PYTHON_TEST_PY": "Test", "RUN_PYTHON_REPRO_PY": "Test",
    "RUN_PYTEST_TEST_PY": "Test", "RUN_PYTEST_ALL": "Test",
    "RUN_PYTEST_SRC_PY": "Test", "RUN_PYTEST_REPRO_PY": "Test",
    "RUN_MAKE": "Test", "RUN_LINT": "Test",
    # Finish
    "SUBMIT": "Finish", "EXIT_ERROR": "Finish",
    "SHELL_RM": "Finish", "SHELL_SED": "Finish",
}


def _stage(action: str) -> str:
    return ACTION_STAGE.get(action, "Explore")


def _build_rows(agents: list[dict]) -> pd.DataFrame:
    rows = []
    for agent in agents:
        for step, action in enumerate(agent["actions"]):
            rows.append({
                "agent":    agent["label"],
                "step":     step,
                "action":   action,
                "stage":    _stage(action),
                "resolved": agent["resolved"],
            })
    return pd.DataFrame(rows)


def make_vignette(
    instance_id: str,
    agents: list[dict],
    out_path: Path,
) -> None:
    df = _build_rows(agents)
    agent_order = [a["label"] for a in agents]

    color_scale = alt.Scale(
        domain=STAGE_ORDER,
        range=[STAGE_COLORS[s] for s in STAGE_ORDER],
    )

    blocks = (
        alt.Chart(df)
        .mark_rect(cornerRadius=3, stroke="white", strokeWidth=0.8)
        .encode(
            x=alt.X("step:O",
                    axis=alt.Axis(labels=False, ticks=False, title=None, domain=False),
                    scale=alt.Scale(paddingInner=0.12)),
            y=alt.Y("agent:N",
                    sort=agent_order,
                    axis=alt.Axis(title=None, labelFontSize=12, labelColor=NEAR_BLACK)),
            color=alt.Color("stage:N",
                            sort=STAGE_ORDER,
                            scale=color_scale,
                            legend=alt.Legend(title=None, orient="bottom",
                                              direction="horizontal",
                                              symbolType="square",
                                              symbolSize=110,
                                              labelFontSize=11)),
            tooltip=["agent", "step", "action", "stage"],
        )
    )

    # Step-count labels below each row
    step_df = pd.DataFrame([
        {"agent": a["label"], "step": 0,
         "label": f"{len(a['actions'])} steps"}
        for a in agents
    ])
    # Outcome badges — render as text to the right of the last block
    max_steps = max(len(a["actions"]) for a in agents)
    outcome_df = pd.DataFrame([
        {"agent": a["label"],
         "step": max_steps - 1,
         "outcome": "Resolved" if a["resolved"] else "Not resolved",
         "color": GREEN if a["resolved"] else GRAY}
        for a in agents
    ])

    repo = instance_id.split("__")[0].replace("-", " ").title()
    issue = instance_id.split("-")[-1]

    chart = (
        blocks
        .properties(
            title=alt.TitleParams(
                text="Procedural sequences",
                subtitle=f"{repo} issue #{issue}",
                fontSize=13,
                subtitleFontSize=10,
                subtitleColor="#888888",
                fontWeight="normal",
                anchor="middle",
                color="#111111",
            ),
            width=alt.Step(28),
            height=alt.Step(52),
        )
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    chart.save(str(out_path.with_suffix(".html")))

    try:
        import vl_convert as vlc
        png = vlc.vegalite_to_png(chart.to_json(), scale=2)
        out_path.write_bytes(png)
        print(f"Saved: {out_path}")
    except ImportError:
        print(f"vl_convert not available — saved HTML: {out_path.with_suffix('.html')}")
        print("Install with: pip install vl-convert-python")


def main() -> None:
    claude_actions = [
        "FIND_FILE", "OPEN_SRC_PY", "SEARCH", "NAV_SRC_PY", "OPEN_SRC_PY",
        "EDIT_SRC_PY", "OPEN_SRC_PY", "SEARCH", "EDIT_SRC_PY", "EDIT_SRC_PY",
        "OPEN_OTHER", "SEARCH", "EDIT_OTHER", "EDIT_OTHER", "EDIT_OTHER",
        "EDIT_OTHER", "SUBMIT",
    ]
    gpt4o_actions = [
        "SHELL_LS", "SHELL_CD", "SHELL_CD", "OPEN_SRC_PY", "NAV_SRC_PY",
        "NAV_SRC_PY", "EDIT_SRC_PY", "SHELL_CD", "SHELL_CD", "SHELL_CD",
        "OPEN_TEST_PY", "NAV_TEST_PY", "EDIT_TEST_PY", "RUN_PYTHON_ALL",
        "CREATE_CONFIG_PY", "EDIT_CONFIG_PY", "RUN_PYTHON_ALL", "SHELL_MV",
        "RUN_PYTHON_ALL", "SHELL_EXPORT", "SHELL_EXPORT", "SHELL_CD",
        "CREATE_SRC_PY", "EDIT_SRC_PY", "EXIT_ERROR",
    ]

    make_vignette(
        instance_id="django__django-14608",
        agents=[
            {"label": "Claude 3.5", "actions": claude_actions, "resolved": True},
            {"label": "GPT-4o",     "actions": gpt4o_actions,  "resolved": False},
        ],
        out_path=OUT / "vignette_figure1.png",
    )


if __name__ == "__main__":
    main()
