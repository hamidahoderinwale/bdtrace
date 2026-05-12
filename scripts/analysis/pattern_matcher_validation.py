"""Pattern-matcher cross-validation against the nine-agent corpus.

Cross-validates the paper's BPE-derived qualitative claims by running
a small set of regex rules over the canonical atom sequences. The
rules live in `output/paper2_pilot/pattern_validation_rules.yaml`,
expressed in the paper's atom vocabulary.

For each (rule, cell) pair we report:

* Pass rate (fraction of trajectories that satisfy the rule).
* Per-agent breakdown within the cell.

The figure shows pass rates by cell as a small-multiple bar chart,
one panel per rule.

Reads:
    output/paper2_pilot/bpe_sequences_extended.jsonl
    output/paper2_pilot/pattern_validation_rules.yaml

Writes:
    output/paper2_pilot/pattern_validation.json
    output/figures/fig_pattern_validation.png
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import altair as alt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register

register()

SEQ_PATH = ROOT / "output" / "paper2_pilot" / "bpe_sequences_extended.jsonl"
RULES_PATH = ROOT / "output" / "paper2_pilot" / "pattern_validation_rules.yaml"
OUT_JSON = ROOT / "output" / "paper2_pilot" / "pattern_validation.json"
OUT_FIG = ROOT / "output" / "figures" / "fig_pattern_validation.png"

# Paradigm-by-scaffold cell assignment for each agent, matching the
# scheme used in trajectory_clusters_extended.py.
AGENT_CELL: dict[str, str] = {
    "Claude-3":             "SWE-agent base",
    "Claude-3.5":           "SWE-agent base",
    "GPT-4":                "SWE-agent base",
    "GPT-4o":               "SWE-agent base",
    "Claude-3.7-thinking":  "SWE-agent extended-thinking",
    "Claude-4":             "SWE-agent extended-thinking",
    "Agentless+Claude-3.5": "Agentless",
    "DARS+R1":              "DARS",
    "Moatless+V3":          "Moatless",
}

CELL_ORDER: list[str] = [
    "SWE-agent base",
    "SWE-agent extended-thinking",
    "Agentless",
    "DARS",
    "Moatless",
]


@dataclass(frozen=True)
class Rule:
    name: str
    description: str
    pattern: str
    must_hold: bool
    prediction: str


def load_rules(path: Path) -> list[Rule]:
    """Load rules from YAML; pull the optional `prediction` annotation."""
    raw = yaml.safe_load(path.read_text())
    out: list[Rule] = []
    for entry in raw["rules"]:
        out.append(
            Rule(
                name=str(entry["name"]),
                description=str(entry.get("description", "")).strip(),
                pattern=str(entry["pattern"]),
                must_hold=bool(entry["must_hold"]),
                prediction=str(entry.get("prediction", "")).strip(),
            )
        )
    return out


def load_records(path: Path) -> list[dict[str, object]]:
    """Read the JSONL corpus into memory."""
    return [json.loads(line) for line in path.open() if line.strip()]


def encode_sequence(atoms: list[str]) -> str:
    """Render an atom list as a single space-joined string with a
    trailing space so right-anchored patterns like ``SUBMIT $`` can
    match the final atom."""
    return " ".join(atoms) + " "


def evaluate_rule(rule: Rule, encoded: str) -> bool:
    """Return True iff the trajectory satisfies the rule."""
    matched = re.search(rule.pattern, encoded) is not None
    if rule.must_hold:
        return matched
    return not matched


def evaluate_corpus(records: list[dict[str, object]], rules: list[Rule]) -> dict[str, object]:
    """Score every (rule, trajectory) pair and aggregate per cell and per agent."""
    by_cell_rule_pass: dict[tuple[str, str], int] = defaultdict(int)
    by_cell_rule_total: dict[tuple[str, str], int] = defaultdict(int)
    by_agent_rule_pass: dict[tuple[str, str], int] = defaultdict(int)
    by_agent_rule_total: dict[tuple[str, str], int] = defaultdict(int)

    for record in records:
        agent = str(record["agent"])
        cell = AGENT_CELL.get(agent, "UNKNOWN")
        atoms = list(record["canonical"])  # type: ignore[arg-type]
        encoded = encode_sequence(atoms)
        for rule in rules:
            satisfied = evaluate_rule(rule, encoded)
            by_cell_rule_total[(cell, rule.name)] += 1
            by_agent_rule_total[(agent, rule.name)] += 1
            if satisfied:
                by_cell_rule_pass[(cell, rule.name)] += 1
                by_agent_rule_pass[(agent, rule.name)] += 1

    per_cell = {
        rule.name: {
            cell: {
                "n": by_cell_rule_total[(cell, rule.name)],
                "pass": by_cell_rule_pass[(cell, rule.name)],
                "pass_rate": (
                    by_cell_rule_pass[(cell, rule.name)] / by_cell_rule_total[(cell, rule.name)]
                    if by_cell_rule_total[(cell, rule.name)] > 0
                    else 0.0
                ),
            }
            for cell in CELL_ORDER
        }
        for rule in rules
    }
    per_agent = {
        rule.name: {
            agent: {
                "n": by_agent_rule_total[(agent, rule.name)],
                "pass": by_agent_rule_pass[(agent, rule.name)],
                "pass_rate": (
                    by_agent_rule_pass[(agent, rule.name)] / by_agent_rule_total[(agent, rule.name)]
                    if by_agent_rule_total[(agent, rule.name)] > 0
                    else 0.0
                ),
            }
            for agent in AGENT_CELL
        }
        for rule in rules
    }
    rules_payload = [
        {
            "name": rule.name,
            "description": rule.description,
            "pattern": rule.pattern,
            "must_hold": rule.must_hold,
            "prediction": rule.prediction,
        }
        for rule in rules
    ]
    return {
        "n_trajectories": len(records),
        "rules": rules_payload,
        "per_cell": per_cell,
        "per_agent": per_agent,
    }


def plot_per_cell_pass_rates(per_cell: dict[str, dict[str, dict[str, float]]], out_path: Path) -> None:
    """Small-multiple bar chart: rules on rows, cells on x, pass rate on y."""
    rows: list[dict[str, object]] = []
    for rule_name, by_cell in per_cell.items():
        for cell, stats in by_cell.items():
            rows.append({
                "rule": rule_name,
                "cell": cell,
                "pass_rate": stats["pass_rate"],
                "n": stats["n"],
            })
    df = pd.DataFrame(rows)
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(
                "cell:N",
                sort=CELL_ORDER,
                axis=alt.Axis(title=None, labelAngle=-30),
            ),
            y=alt.Y(
                "pass_rate:Q",
                axis=alt.Axis(title="pass rate", format=".0%"),
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color("cell:N", sort=CELL_ORDER, legend=None),
            tooltip=["rule:N", "cell:N", alt.Tooltip("pass_rate:Q", format=".1%"), "n:Q"],
        )
        .properties(width=140, height=120)
        .facet(
            row=alt.Row("rule:N", header=alt.Header(labelAlign="left", labelAnchor="start")),
        )
        .resolve_scale(y="shared")
        .properties(
            title=alt.TitleParams(
                text="Pattern-matcher pass rates",
                fontSize=11,
                anchor="start",
            ),
        )
    )
    chart.save(str(out_path), scale_factor=2)


def main() -> int:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading corpus from {SEQ_PATH} ...")
    records = load_records(SEQ_PATH)
    print(f"  {len(records)} trajectories")

    print(f"Loading rules from {RULES_PATH} ...")
    rules = load_rules(RULES_PATH)
    print(f"  {len(rules)} rules")

    print("Evaluating ...")
    payload = evaluate_corpus(records, rules)

    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {OUT_JSON}")

    print("Plotting per-cell pass rates ...")
    plot_per_cell_pass_rates(payload["per_cell"], OUT_FIG)  # type: ignore[arg-type]
    print(f"Saved {OUT_FIG}")

    print()
    print("PER-CELL PASS RATES")
    print("-" * 80)
    for rule in rules:
        print(f"\n{rule.name}  ({'must hold' if rule.must_hold else 'must NOT match'})")
        print(f"  prediction: {rule.prediction}")
        for cell in CELL_ORDER:
            stats = payload["per_cell"][rule.name][cell]  # type: ignore[index]
            print(f"  {cell:32s} {stats['pass_rate']:6.1%}  (n={stats['n']}, pass={stats['pass']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
