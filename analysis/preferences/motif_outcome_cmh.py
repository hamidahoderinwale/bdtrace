"""Difficulty-adjusted motif-outcome association via Mantel-Haenszel.

For each BPE compound motif, tests whether its presence in a trajectory
predicts pass/fail after stratifying by instance difficulty (n_resolved
bucket out of 9 agents). Informative strata exclude 0/9 (all-fail) and
9/9 (all-pass); the in-between strata carry the signal.

Method: Mantel-Haenszel CMH test across difficulty strata.
For each stratum k: 2x2 table (motif present/absent x pass/fail).
CMH combines tables into a single difficulty-adjusted OR and chi-square.
FDR correction (Benjamini-Hochberg) over all tested motifs.

Reads:
    output/paper2_pilot/bpe_sequences_extended.jsonl
    output/paper2_pilot/extended_pass_fail.json
Writes:
    output/paper2_pilot/motif_outcome_cmh.json
    output/figures/fig_motif_cmh.png
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dist
import altair as alt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, AGENT_SHORT, BLUE, VERMILLION, GRAY
register()

OUT     = ROOT / "output" / "paper2_pilot"
FIG_OUT = ROOT / "output" / "figures"
N_AGENTS_FULL = 9
INFORMATIVE_STRATA = set(range(1, N_AGENTS_FULL))  # 1..8 (out of 9)
MIN_MOTIF_COUNT = 20   # minimum total appearances to test
TOP_N = 15             # motifs to show in figure

SUBMISSION_TO_AGENT = {
    "20240402_sweagent_claude3opus":                "Claude-3",
    "20240402_sweagent_gpt4":                       "GPT-4",
    "20240620_sweagent_claude3.5sonnet":            "Claude-3.5",
    "20240728_sweagent_gpt4o":                      "GPT-4o",
    "20250226_sweagent_claude-3-7-sonnet-20250219": "Claude-3.7-thinking",
    "20250526_sweagent_claude-4-sonnet-20250514":   "Claude-4",
    "20241202_agentless-1.5_claude-3.5-sonnet-20241022": "Agentless+Claude-3.5",
    "20250205_dars_agent_claude_3.5_sonnet_deepseek_r1": "DARS+R1",
    "20250111_moatless_deepseek_v3":                "Moatless+V3",
}


def mantel_haenszel(tables: list[tuple[int, int, int, int]]) -> tuple[float, float, float]:
    """
    Mantel-Haenszel combined OR and chi-square p-value.

    Each table is (a, b, c, d) where:
        a = motif present, pass
        b = motif present, fail
        c = motif absent,  pass
        d = motif absent,  fail

    Returns (OR_MH, chi2_stat, p_value).
    Returns (nan, nan, 1.0) when denominator is zero.
    """
    num = 0.0   # Σ a*d/n
    den = 0.0   # Σ b*c/n
    exp_a = 0.0 # Σ (a+b)(a+c)/n
    obs_a = 0.0 # Σ a
    var   = 0.0 # Σ variance term

    for a, b, c, d in tables:
        n = a + b + c + d
        if n < 2:
            continue
        num   += a * d / n
        den   += b * c / n
        exp_a += (a + b) * (a + c) / n
        obs_a += a
        denom_v = (a+b) * (a+c) * (b+d) * (c+d)
        if denom_v > 0:
            var += denom_v / (n * n * (n - 1))

    if den == 0 or var == 0:
        return float("nan"), float("nan"), 1.0

    or_mh  = num / den
    # Yates-corrected chi-square
    chi2_stat = (abs(obs_a - exp_a) - 0.5) ** 2 / var
    p_value   = float(chi2_dist.sf(chi2_stat, df=1))
    return or_mh, chi2_stat, p_value


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    # Load BPE sequences (9-agent extended corpus)
    records = []
    with open(ROOT / "output/paper2_pilot/bpe_sequences_extended.jsonl") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Load pass/fail from extended_pass_fail.json instead of the 4-agent
    # parquet. Build (agent, instance_id, passed) rows by checking each
    # submission's resolved-instance set.
    pf = json.loads((ROOT / "output/paper2_pilot/extended_pass_fail.json").read_text())
    resolved_by_agent: dict[str, set[str]] = {}
    for sub, agent in SUBMISSION_TO_AGENT.items():
        resolved_by_agent[agent] = set(pf.get(sub, {}).get("resolved", []))

    traj_rows = []
    for r in records:
        a, iid = r["agent"], r["instance_id"]
        passed = iid in resolved_by_agent.get(a, set())
        traj_rows.append({"agent": a, "instance_id": iid, "passed": bool(passed)})
    traj = pd.DataFrame(traj_rows)
    n_res = traj.groupby("instance_id")["passed"].sum().rename("n_resolved")
    traj = traj.merge(n_res, on="instance_id")

    bpe_df = pd.DataFrame([
        {"agent": r["agent"], "instance_id": r["instance_id"],
         "bpe": set(r["bpe"])}
        for r in records
    ])
    df = bpe_df.merge(
        traj[["agent", "instance_id", "passed", "n_resolved"]],
        on=["agent", "instance_id"],
    )

    # Keep only informative strata
    df_inf = df[df["n_resolved"].isin(INFORMATIVE_STRATA)].copy()
    print(f"Informative rows: {len(df_inf)}  "
          f"(strata {sorted(INFORMATIVE_STRATA)})")

    # Build motif vocabulary from informative strata
    from collections import Counter
    motif_counts: Counter = Counter()
    for bpe_set in df_inf["bpe"]:
        motif_counts.update(bpe_set)

    motifs = [m for m, c in motif_counts.items()
              if "+" in m and c >= MIN_MOTIF_COUNT]
    print(f"Testing {len(motifs)} compound motifs (count >= {MIN_MOTIF_COUNT})")

    results = []
    for motif in motifs:
        df_inf["present"] = df_inf["bpe"].apply(lambda s: motif in s)

        tables = []
        for stratum in sorted(INFORMATIVE_STRATA):
            sub = df_inf[df_inf["n_resolved"] == stratum]
            a = int(( sub["present"] &  sub["passed"]).sum())
            b = int(( sub["present"] & ~sub["passed"]).sum())
            c = int((~sub["present"] &  sub["passed"]).sum())
            d = int((~sub["present"] & ~sub["passed"]).sum())
            tables.append((a, b, c, d))

        or_mh, chi2_stat, p_val = mantel_haenszel(tables)
        results.append({
            "motif":   motif,
            "or_mh":   or_mh,
            "chi2":    chi2_stat,
            "p_raw":   p_val,
            "count":   motif_counts[motif],
        })

    res_df = pd.DataFrame(results).dropna(subset=["or_mh"])
    res_df = res_df.sort_values("p_raw")

    # FDR correction (Benjamini-Hochberg)
    n = len(res_df)
    ranks = np.arange(1, n + 1)
    res_df = res_df.reset_index(drop=True)
    res_df["rank"] = ranks
    res_df["p_adj"] = (res_df["p_raw"] * n / res_df["rank"]).clip(upper=1.0)
    # Monotone step-up
    res_df["p_adj"] = res_df["p_adj"][::-1].cummin()[::-1]

    res_df["log_or"] = np.log(res_df["or_mh"].clip(lower=1e-4, upper=1e4))
    res_df["direction"] = res_df["log_or"].apply(
        lambda x: "pass-enriched" if x > 0 else "fail-enriched"
    )
    res_df["significant"] = res_df["p_adj"] < 0.05

    sig = res_df[res_df["significant"]]
    print(f"\nSignificant motifs (FDR < 0.05): {len(sig)}")
    print(f"  Pass-enriched: {(sig['log_or'] > 0).sum()}")
    print(f"  Fail-enriched: {(sig['log_or'] < 0).sum()}")
    print("\nTop 10 by |log-OR|:")
    top = pd.concat([
        res_df.nlargest(10, "log_or"),
        res_df.nsmallest(10, "log_or"),
    ]).drop_duplicates("motif")
    for _, row in top.iterrows():
        sig_star = "*" if row["significant"] else " "
        print(f"  {sig_star} OR={row['or_mh']:.2f}  p_adj={row['p_adj']:.3f}  "
              f"n={row['count']:4d}  {row['motif'][:60]}")

    # Save JSON
    out_json = {
        "n_rows": len(df_inf),
        "n_motifs_tested": len(motifs),
        "n_significant_fdr05": int(len(sig)),
        "results": res_df[
            ["motif", "or_mh", "log_or", "p_raw", "p_adj",
             "direction", "significant", "count"]
        ].to_dict(orient="records"),
    }
    (OUT / "motif_outcome_cmh.json").write_text(
        json.dumps(out_json, indent=2, default=float)
    )

    # Figure: top motifs by |log_or|, colored by direction
    top_pass = res_df[res_df["log_or"] > 0].nlargest(TOP_N // 2, "log_or")
    top_fail = res_df[res_df["log_or"] < 0].nsmallest(TOP_N // 2, "log_or")
    fig_df = pd.concat([top_pass, top_fail]).reset_index(drop=True)

    # Shorten motif labels for display
    def short_label(m: str, max_len: int = 42) -> str:
        return m if len(m) <= max_len else m[:max_len - 2] + ".."

    fig_df["label"] = fig_df["motif"].apply(short_label)
    y_order = fig_df.sort_values("log_or")["label"].tolist()

    color_scale = alt.Scale(
        domain=["pass-enriched", "fail-enriched"],
        range=[BLUE, VERMILLION],
    )

    base = (
        alt.Chart(fig_df)
        .encode(
            y=alt.Y("label:N", sort=y_order,
                    axis=alt.Axis(title=None, labelFontSize=9)),
            color=alt.Color("direction:N", scale=color_scale,
                            legend=alt.Legend(title=None, orient="bottom")),
        )
    )

    dots = base.mark_point(filled=True, size=70, strokeWidth=0).encode(
        x=alt.X("log_or:Q",
                title="Log odds ratio (difficulty-adjusted)",
                axis=alt.Axis(format=".1f")),
    )

    sig_marks = (
        alt.Chart(fig_df[fig_df["significant"]])
        .mark_text(text="*", fontSize=14, dy=-2)
        .encode(
            y=alt.Y("label:N", sort=y_order),
            x=alt.X("log_or:Q"),
            color=alt.Color("direction:N", scale=color_scale, legend=None),
        )
    )

    chart = (
        (dots + sig_marks)
        .properties(
            width=340, height=300,
            title=alt.TitleParams(
                "Motif-outcome association (difficulty-adjusted)",
                fontSize=13, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
    )

    out_fig = FIG_OUT / "fig_motif_cmh.png"
    chart.save(str(out_fig), scale_factor=2)
    print(f"\nSaved {out_fig}")
    print(f"Saved {OUT / 'motif_outcome_cmh.json'}")


if __name__ == "__main__":
    main()
