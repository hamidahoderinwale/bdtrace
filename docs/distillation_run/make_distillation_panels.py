"""Altair generators for the distillation A/B/C panels (Tufte-minimal, themed).

Reads parent + child fingerprints (produced by the rollout + canonicalization
step) and emits the three panels the case study needs:

  A  fig_distillation_A_entropy.png       per-trajectory action entropy, parent vs child
  B  fig_distillation_B_conditional.png   conditional JSD by prefix action (freq-weighted)
  C  fig_distillation_C_outcome.png       native-vocabulary Jaccard with parent, by outcome

These are the three the reviewer flagged as needing the most work; they are
written here in Altair (per the all-Altair requirement) and will be
vision-checked once the child data exists. They CANNOT render until
`fingerprints_parent.jsonl` and `fingerprints_child.jsonl` are produced by the
rollout step — this script does not fabricate data.

Expected fingerprint row schema (one per trajectory):
  {"trace_id","role":"parent|child","outcome":"resolved|unresolved",
   "canonical":[...atoms...], "native":[...atoms...]}

Usage (after the rollout produces the fingerprints):
    python distillation_run/make_distillation_panels.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, BLUE, COPPER, GREEN, MAGENTA  # noqa: E402

register()

DATA = ROOT / "distillation_run"
OUT = ROOT / "docs" / "papers" / "figures"
PARENT = DATA / "fingerprints_parent.jsonl"
CHILD = DATA / "fingerprints_child.jsonl"


def _entropy(seq: list[str]) -> float:
    if not seq:
        return 0.0
    c = Counter(seq)
    p = np.array(list(c.values()), dtype=float)
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def _load(path: Path, role: str) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"missing {path.name}. Run the rollout + canonicalization first; "
            "this script renders real fingerprints only, it does not fabricate."
        )
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def panel_a_entropy(parent: list[dict], child: list[dict]) -> None:
    rows = []
    for role, data in (("parent", parent), ("child", child)):
        for layer in ("canonical", "native"):
            for r in data:
                rows.append({"role": role, "layer": layer, "entropy": _entropy(r.get(layer, []))})
    df = pd.DataFrame(rows)
    chart = (
        alt.Chart(df)
        .mark_boxplot(size=28)
        .encode(
            x=alt.X("role:N", title=None, axis=alt.Axis(domain=False, ticks=False, labelAngle=0)),
            y=alt.Y("entropy:Q", title="Action entropy (bits)", axis=alt.Axis(domain=False, ticks=False)),
            color=alt.Color("role:N", legend=None,
                            scale=alt.Scale(domain=["parent", "child"], range=[COPPER, BLUE])),
            column=alt.Column("layer:N", title=None, header=alt.Header(labelFontSize=11)),
        )
        .properties(width=130, height=240, title="Action entropy: parent vs child")
    )
    chart.save(str(OUT / "fig_distillation_A_entropy.png"), scale_factor=2)
    print("wrote fig_distillation_A_entropy.png")


def panel_b_conditional(parent: list[dict], child: list[dict], k_top: int = 8) -> None:
    """Conditional JSD by prefix action, frequency-weighted; bar per prefix atom."""
    def cond(data):
        nxt = {}
        freq = Counter()
        for r in data:
            s = r.get("canonical", [])
            for a, b in zip(s, s[1:]):
                nxt.setdefault(a, Counter())[b] += 1
                freq[a] += 1
        return nxt, freq

    pn, pf = cond(parent)
    cn, _ = cond(child)
    atoms = sorted((set(pn) & set(cn)) - {"submit"}, key=lambda a: -pf[a])[:k_top]  # submit terminal -> degenerate JSD
    vocab = sorted({b for a in atoms for b in (set(pn.get(a, {})) | set(cn.get(a, {})))})

    def dist(counter):
        v = np.array([counter.get(t, 0) for t in vocab], dtype=float)
        return v / v.sum() if v.sum() else v

    def jsd(p, q):
        m = 0.5 * (p + q)
        def kl(x, y):
            mask = x > 0
            return float((x[mask] * np.log2(x[mask] / y[mask])).sum())
        return 0.5 * kl(p, m) + 0.5 * kl(q, m)

    rows = [{"prefix": a, "jsd": jsd(dist(pn[a]), dist(cn[a])), "n": pf[a]} for a in atoms]
    df = pd.DataFrame(rows)
    chart = (
        alt.Chart(df)
        .mark_bar(size=18, color=MAGENTA)
        .encode(
            x=alt.X("jsd:Q", title="Conditional JSD", axis=alt.Axis(domain=False, ticks=False)),
            y=alt.Y("prefix:N", sort="-x", title=None, axis=alt.Axis(domain=False, ticks=False)),
        )
        .properties(width=300, height=220, title="Conditional JSD by prefix action")
    )
    chart.save(str(OUT / "fig_distillation_B_conditional.png"), scale_factor=2)
    print("wrote fig_distillation_B_conditional.png")


def panel_c_outcome(parent: list[dict], child: list[dict]) -> None:
    """Native-vocabulary Jaccard of each child trajectory with the parent vocab, by outcome."""
    parent_vocab = {a for r in parent for a in r.get("native", [])}

    def jac(s):
        v = set(s)
        u = v | parent_vocab
        return len(v & parent_vocab) / len(u) if u else 0.0

    rows = [{"outcome": r.get("outcome", "?"), "jaccard": jac(r.get("native", []))} for r in child]
    df = pd.DataFrame(rows)
    chart = (
        alt.Chart(df)
        .mark_boxplot(size=30)
        .encode(
            x=alt.X("outcome:N", title=None, axis=alt.Axis(domain=False, ticks=False, labelAngle=0)),
            y=alt.Y("jaccard:Q", title="Native-vocabulary Jaccard with parent",
                    axis=alt.Axis(domain=False, ticks=False)),
            color=alt.Color("outcome:N", legend=None,
                            scale=alt.Scale(domain=["resolved", "unresolved"], range=[GREEN, MAGENTA])),
        )
        .properties(width=240, height=240, title="Vocabulary overlap with parent by outcome")
    )
    chart.save(str(OUT / "fig_distillation_C_outcome.png"), scale_factor=2)
    print("wrote fig_distillation_C_outcome.png")


def main() -> None:
    parent = _load(PARENT, "parent")
    child = _load(CHILD, "child")
    print(f"parent: {len(parent)} traj, child: {len(child)} traj")
    panel_a_entropy(parent, child)
    panel_b_conditional(parent, child)
    panel_c_outcome(parent, child)


if __name__ == "__main__":
    main()
