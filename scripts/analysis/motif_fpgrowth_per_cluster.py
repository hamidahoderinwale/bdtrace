"""Per-FIM-form FP-growth action templates.

For each FIM canonical form (task class) with sufficient sample size,
run FP-growth on the canonical action sets of trajectories in that class.
Returns the action combinations most associated with success.

Atom-level (76 actions) gives denser co-occurrence than BPE motif level
(~170 motifs, mean 9 per trajectory) and produces interpretable templates
suitable for both recommender and methodology surfaces.

Per-pattern reporting:
    support       fraction of in-form trajectories that contain the pattern
    pass_rate     pass rate among trajectories containing the pattern
    base_rate     pass rate of the form overall
    lift          pass_rate / base_rate

Reads:
    output/paper2_pilot/bpe_sequences.jsonl
    output/canonical_forms/instance_assignments.parquet
    output/trajectories/lite_all_models.parquet
Writes:
    output/paper2_pilot/motif_fpgrowth_templates.json
    output/figures/fig_motif_fpgrowth_templates.png
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import pandas as pd
import altair as alt
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, BLUE, GREEN, OLIVE
register()

BPE_FILE   = ROOT / "output" / "paper2_pilot" / "bpe_sequences.jsonl"
FORMS_FILE = ROOT / "output" / "canonical_forms" / "instance_assignments.parquet"
LITE_FILE  = ROOT / "output" / "trajectories" / "lite_all_models.parquet"
OUT_JSON   = ROOT / "output" / "paper2_pilot" / "motif_fpgrowth_templates.json"
OUT_FIG    = ROOT / "output" / "figures" / "fig_motif_fpgrowth_templates.png"

MIN_SAMPLES_PER_FORM = 15
MIN_SUPPORT          = 0.30
MAX_PATTERN_LEN      = 4
TOP_K_PER_FORM       = 3
AGENT_SHORT          = {
    "20240402_sweagent_claude3opus":     "Claude-3",
    "20240620_sweagent_claude3.5sonnet": "Claude-3.5",
    "20240402_sweagent_gpt4":            "GPT-4",
    "20240728_sweagent_gpt4o":           "GPT-4o",
}


def load_bpe() -> pd.DataFrame:
    rows = []
    with BPE_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            canonical = rec.get("canonical", [])
            if not canonical:
                continue
            rows.append({
                "agent":       rec["agent"],
                "instance_id": rec["instance_id"],
                "motif_set":   list(set(canonical)),
            })
    return pd.DataFrame(rows)


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    bpe_df = load_bpe()

    forms_df = pd.read_parquet(FORMS_FILE)[["instance_id", "form_name"]].drop_duplicates("instance_id")

    lite_df = pd.read_parquet(LITE_FILE)
    lite_df["agent"] = lite_df["model_id"].map(AGENT_SHORT)
    lite_df = lite_df[lite_df["agent"].notna()][["agent", "instance_id", "passed"]].copy()

    df = bpe_df.merge(forms_df, on="instance_id", how="inner")
    df = df.merge(lite_df,  on=["agent", "instance_id"], how="inner")

    print(f"Joined corpus: {len(df)} trajectories across {df['form_name'].nunique()} forms")

    results: dict[str, dict] = {}

    for form_name, sub in df.groupby("form_name"):
        if len(sub) < MIN_SAMPLES_PER_FORM:
            continue
        n        = len(sub)
        base     = float(sub["passed"].mean())
        passed_set = set(zip(sub.loc[sub["passed"], "agent"], sub.loc[sub["passed"], "instance_id"]))

        encoder = TransactionEncoder()
        oh      = encoder.fit_transform(sub["motif_set"].tolist())
        oh_df   = pd.DataFrame(oh, columns=encoder.columns_)
        oh_df["__key__"] = list(zip(sub["agent"], sub["instance_id"]))

        try:
            fp = fpgrowth(
                oh_df.drop(columns=["__key__"]),
                min_support=MIN_SUPPORT,
                use_colnames=True,
                max_len=MAX_PATTERN_LEN,
            )
        except Exception as e:
            print(f"  {form_name}: fpgrowth failed: {e}")
            continue
        if fp.empty:
            continue

        fp = fp.copy()
        fp["pattern_size"] = fp["itemsets"].apply(len)
        fp = fp[fp["pattern_size"] >= 2]
        if fp.empty:
            continue

        fp["pattern_list"] = fp["itemsets"].apply(lambda s: sorted(list(s)))
        rows_for_form = []
        for _, r in fp.iterrows():
            pattern   = frozenset(r["pattern_list"])
            mask      = oh_df[list(pattern)].all(axis=1)
            keys_with = [k for k, m in zip(oh_df["__key__"], mask) if m]
            if not keys_with:
                continue
            n_with  = len(keys_with)
            n_pass_with = sum(1 for k in keys_with if k in passed_set)
            pass_rate = n_pass_with / n_with
            lift      = (pass_rate / base) if base > 0 else float("nan")
            rows_for_form.append({
                "pattern":      r["pattern_list"],
                "size":         int(r["pattern_size"]),
                "support":      round(float(r["support"]), 3),
                "n_with":       int(n_with),
                "pass_rate":    round(pass_rate, 3),
                "base_rate":    round(base, 3),
                "lift":         round(lift, 3),
                "n_pass_with":  int(n_pass_with),
            })
        if not rows_for_form:
            continue

        rows_for_form.sort(key=lambda r: (-r["lift"], -r["support"]))
        results[form_name] = {
            "n_trajectories": int(n),
            "base_pass_rate": round(base, 3),
            "top_patterns":   rows_for_form[:TOP_K_PER_FORM],
        }

        print(f"\n  {form_name} (n={n}, base pass={base:.2%}):")
        for r in rows_for_form[:TOP_K_PER_FORM]:
            pat_str = " + ".join(r["pattern"])
            print(f"    [{r['size']}-itemset] {pat_str[:90]:90s}  "
                  f"sup={r['support']:.2f}  pass={r['pass_rate']:.2f}  lift={r['lift']:.2f}")

    OUT_JSON.write_text(json.dumps({
        "min_samples_per_form": MIN_SAMPLES_PER_FORM,
        "min_support": MIN_SUPPORT,
        "max_pattern_len": MAX_PATTERN_LEN,
        "top_k_per_form": TOP_K_PER_FORM,
        "results": results,
    }, indent=2, default=str))
    print(f"\nSaved {OUT_JSON}")

    rows_plot = []
    for form, info in results.items():
        if not info["top_patterns"]:
            continue
        if pd.isna(info["top_patterns"][0]["lift"]):
            continue
        p = info["top_patterns"][0]
        rows_plot.append({
            "form":      form,
            "row_label": f"{form} (n={info['n_trajectories']})",
            "pattern":   " + ".join(p["pattern"]),
            "support":   p["support"],
            "lift":      p["lift"],
            "pass_rate": p["pass_rate"],
            "base_rate": p["base_rate"],
        })
    if not rows_plot:
        print("No plottable patterns.")
        return
    plot_df = pd.DataFrame(rows_plot).sort_values("lift", ascending=False).reset_index(drop=True)
    row_order = plot_df["row_label"].tolist()

    bars = (
        alt.Chart(plot_df)
        .mark_bar(size=14, color=BLUE)
        .encode(
            y=alt.Y("row_label:N", sort=row_order, axis=alt.Axis(title=None)),
            x=alt.X("lift:Q",
                    title="Lift (pass rate of trajectories with pattern / base rate)",
                    scale=alt.Scale(domain=[0, max(plot_df["lift"]) * 1.05])),
        )
    )
    pattern_text = (
        alt.Chart(plot_df)
        .mark_text(align="left", dx=6, fontSize=10, color="#444444")
        .encode(
            y=alt.Y("row_label:N", sort=row_order),
            x=alt.X("lift:Q"),
            text="pattern:N",
        )
    )

    chart = (
        alt.layer(bars, pattern_text)
        .properties(
            width=420, height=18 * len(plot_df) + 60,
            title=alt.TitleParams(
                "Top FP-growth pattern per FIM form",
                fontSize=12, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )
    chart.save(str(OUT_FIG), scale_factor=2)
    print(f"Saved {OUT_FIG}")


if __name__ == "__main__":
    main()
