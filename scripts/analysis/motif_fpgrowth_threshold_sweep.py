"""FP-growth threshold sweep on extended corpus.

For each FIM form, runs FP-growth at min_support in {0.10, 0.20, 0.30, 0.40, 0.50}
and reports which patterns are stable across thresholds.

A pattern is "stable" if it appears as one of the top-K pass-enriched
patterns at multiple thresholds. Stable patterns are the robust findings;
sensitivity-dependent patterns appear only at low thresholds.

Reads:
    output/paper2_pilot/bpe_sequences_extended.jsonl
    output/canonical_forms/instance_assignments.parquet
    output/trajectories/lite_all_models.parquet
    output/paper2_pilot/extended_pass_fail.json
Writes:
    output/paper2_pilot/motif_fpgrowth_threshold_sweep.json
    output/figures/fig_motif_fpgrowth_stability.png   (headline form)
    output/figures/fig_motif_fpgrowth_stability_summary.png  (per-form summary)
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
from scripts.theme import register, BLUE, OLIVE, GREEN
register()

BPE_FILE   = ROOT / "output" / "paper2_pilot" / "bpe_sequences_extended.jsonl"
FORMS_FILE = ROOT / "output" / "canonical_forms" / "instance_assignments.parquet"
LITE_FILE  = ROOT / "output" / "trajectories" / "lite_all_models.parquet"
EXT_PF     = ROOT / "output" / "paper2_pilot" / "extended_pass_fail.json"
OUT_JSON   = ROOT / "output" / "paper2_pilot" / "motif_fpgrowth_threshold_sweep.json"
OUT_FIG_HEAD    = ROOT / "output" / "figures" / "fig_motif_fpgrowth_stability.png"
OUT_FIG_SUMMARY = ROOT / "output" / "figures" / "fig_motif_fpgrowth_stability_summary.png"

THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50]
MIN_SAMPLES_PER_FORM = 30
MAX_PATTERN_LEN      = 4


def load_canonical_envelopes() -> pd.DataFrame:
    rows = []
    with BPE_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            seq = r.get("canonical", [])
            if not seq:
                continue
            rows.append({
                "submission":   r["submission"],
                "agent":        r["agent"],
                "instance_id":  r["instance_id"],
                "atom_set":     list(set(seq)),
            })
    return pd.DataFrame(rows)


def load_passfail() -> dict:
    out = {}
    if LITE_FILE.exists():
        lite = pd.read_parquet(LITE_FILE)[["instance_id", "model_id", "passed"]].copy()
        for _, row in lite.iterrows():
            out[(row["model_id"], row["instance_id"])] = bool(row["passed"])
    if EXT_PF.exists():
        ext = json.loads(EXT_PF.read_text())
        for sub, info in ext.items():
            resolved = set(info.get("resolved", []) or [])
            instances = set(resolved)
            for k in ("generated", "no_generation", "no_logs"):
                v = info.get(k)
                if isinstance(v, list):
                    instances |= set(v)
            for iid in instances:
                out[(sub, iid)] = iid in resolved
    return out


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    bpe = load_canonical_envelopes()
    forms = pd.read_parquet(FORMS_FILE)[["instance_id", "form_name"]].drop_duplicates("instance_id")
    df = bpe.merge(forms, on="instance_id", how="inner")
    pf = load_passfail()
    df["passed"] = df.apply(lambda r: pf.get((r["submission"], r["instance_id"]), None), axis=1)
    df = df.dropna(subset=["passed"]).copy()
    df["passed"] = df["passed"].astype(bool)
    print(f"Joined corpus: {len(df)} trajectories across {df['form_name'].nunique()} forms")

    out_results: dict[str, dict] = {}

    for form, sub in df.groupby("form_name"):
        if len(sub) < MIN_SAMPLES_PER_FORM:
            continue
        n = len(sub)
        base = float(sub["passed"].mean())
        passed_keys = set(zip(sub.loc[sub["passed"], "submission"], sub.loc[sub["passed"], "instance_id"]))

        enc = TransactionEncoder()
        oh = enc.fit_transform(sub["atom_set"].tolist())
        oh_df = pd.DataFrame(oh, columns=enc.columns_)
        oh_df["__key__"] = list(zip(sub["submission"], sub["instance_id"]))
        cols = [c for c in oh_df.columns if c != "__key__"]

        # Per-threshold patterns
        per_t: dict[str, dict] = {}
        for t in THRESHOLDS:
            try:
                fp = fpgrowth(oh_df[cols], min_support=t, use_colnames=True, max_len=MAX_PATTERN_LEN)
            except Exception:
                continue
            if fp.empty:
                per_t[f"{t:.2f}"] = []
                continue
            fp = fp.copy()
            fp = fp[fp["itemsets"].apply(len) >= 2]
            rows_for_t = []
            for _, r in fp.iterrows():
                pattern = sorted(list(r["itemsets"]))
                mask = oh_df[pattern].all(axis=1)
                keys_with = [k for k, m in zip(oh_df["__key__"], mask) if m]
                if not keys_with: continue
                n_with = len(keys_with)
                n_pass = sum(1 for k in keys_with if k in passed_keys)
                pass_rate = n_pass / n_with
                lift = (pass_rate / base) if base > 0 else float("nan")
                rows_for_t.append({
                    "pattern":   pattern,
                    "key":       " + ".join(pattern),
                    "size":      len(pattern),
                    "support":   round(float(r["support"]), 3),
                    "n_with":    int(n_with),
                    "pass_rate": round(pass_rate, 3),
                    "lift":      round(lift, 3) if lift == lift else None,
                })
            per_t[f"{t:.2f}"] = rows_for_t

        # Build stability map per pattern
        all_patterns: dict[str, dict] = {}
        for t_key, rows in per_t.items():
            for r in rows:
                k = r["key"]
                if k not in all_patterns:
                    all_patterns[k] = {"pattern": r["pattern"], "size": r["size"],
                                       "by_threshold": {}, "n_thresholds_present": 0}
                all_patterns[k]["by_threshold"][t_key] = {"support": r["support"], "lift": r["lift"], "pass_rate": r["pass_rate"]}
                all_patterns[k]["n_thresholds_present"] = len(all_patterns[k]["by_threshold"])

        for k, info in all_patterns.items():
            info["stability"] = info["n_thresholds_present"] / len(THRESHOLDS)
        all_patterns_sorted = sorted(all_patterns.values(),
                                     key=lambda p: (-p["n_thresholds_present"],
                                                    -max((b["lift"] or 0) for b in p["by_threshold"].values())))

        out_results[form] = {
            "n": int(n),
            "base_pass": round(base, 3),
            "thresholds": [f"{t:.2f}" for t in THRESHOLDS],
            "n_distinct_patterns_any_threshold": len(all_patterns),
            "patterns": all_patterns_sorted,
        }
        n_stable_all = sum(1 for p in all_patterns_sorted if p["n_thresholds_present"] == len(THRESHOLDS))
        n_stable_4plus = sum(1 for p in all_patterns_sorted if p["n_thresholds_present"] >= 4)
        print(f"  {form:35s} n={n:4d} base={base:.2f} all_thr={n_stable_all:3d} ≥4thr={n_stable_4plus:3d} any={len(all_patterns):3d}")

    OUT_JSON.write_text(json.dumps({
        "thresholds": THRESHOLDS,
        "min_samples_per_form": MIN_SAMPLES_PER_FORM,
        "max_pattern_len": MAX_PATTERN_LEN,
        "results": out_results,
    }, indent=2, default=str))
    print(f"\nSaved {OUT_JSON}")

    # Headline figure: stability heatmap for the most-trajectory-rich form
    headline_form = max(out_results.items(), key=lambda kv: kv[1]["n"])[0]
    info = out_results[headline_form]
    rows = []
    for p in info["patterns"]:
        for t in info["thresholds"]:
            cell = p["by_threshold"].get(t)
            rows.append({
                "pattern":   p["pattern"][:80],  # truncate label
                "key":       " + ".join(p["pattern"])[:60],
                "threshold": t,
                "lift":      cell["lift"] if cell else None,
                "stable":    p["n_thresholds_present"],
            })
    if rows:
        plot_df = pd.DataFrame(rows)
        # only keep patterns with ≥2 threshold appearances for readability
        keep = plot_df.groupby("key")["lift"].apply(lambda s: s.notna().sum() >= 2)
        keep_keys = keep[keep].index.tolist()
        plot_df = plot_df[plot_df["key"].isin(keep_keys)]
        # sort patterns by stability then by max lift
        order_keys = (plot_df.dropna(subset=["lift"]).groupby("key")
                              .agg(stab=("stable", "max"), max_lift=("lift", "max"))
                              .sort_values(["stab", "max_lift"], ascending=[False, False])
                              .index.tolist())

        chart = (
            alt.Chart(plot_df)
            .mark_circle(size=180)
            .encode(
                y=alt.Y("key:N", sort=order_keys, axis=alt.Axis(title=None, labelLimit=300)),
                x=alt.X("threshold:N", title="min_support threshold"),
                color=alt.Color("lift:Q", scale=alt.Scale(scheme="blues"),
                                legend=alt.Legend(title="lift")),
                tooltip=["key:N", "threshold:N", "lift:Q"],
            )
            .properties(
                width=320, height=20 * len(order_keys) + 60,
                title=alt.TitleParams(
                    f"Pattern stability across support thresholds: {headline_form} (n={info['n']})",
                    fontSize=12, color="#111111", anchor="start",
                ),
            )
            .configure_view(strokeWidth=0)
            .configure_axis(grid=False)
        )
        chart.save(str(OUT_FIG_HEAD), scale_factor=2)
        print(f"Saved {OUT_FIG_HEAD}")

    # Summary figure: per-form bar of n_stable
    summary_rows = []
    for form, info in out_results.items():
        n_stable_all  = sum(1 for p in info["patterns"] if p["n_thresholds_present"] == len(THRESHOLDS))
        n_4plus       = sum(1 for p in info["patterns"] if p["n_thresholds_present"] >= 4)
        n_any         = len(info["patterns"])
        summary_rows.append({"form": form,
                             "n_stable_all": n_stable_all,
                             "n_4plus_only": n_4plus - n_stable_all,
                             "n_other":      n_any - n_4plus,
                             "n_total":      n_any,
                             "n_trajectories": info["n"]})
    sdf = pd.DataFrame(summary_rows).sort_values("n_stable_all", ascending=False)
    long_rows = []
    for _, r in sdf.iterrows():
        for cat, v in [("stable across all 5", r["n_stable_all"]),
                       ("stable ≥4 only", r["n_4plus_only"]),
                       ("≤3 only", r["n_other"])]:
            long_rows.append({"form": r["form"], "category": cat, "count": v,
                              "row_label": f"{r['form']} (n={r['n_trajectories']})"})
    long_df = pd.DataFrame(long_rows)
    row_order = sdf.assign(row_label=lambda d: d.apply(lambda r: f"{r['form']} (n={r['n_trajectories']})", axis=1))["row_label"].tolist()
    color_scale = alt.Scale(
        domain=["stable across all 5", "stable ≥4 only", "≤3 only"],
        range=[GREEN, BLUE, OLIVE],
    )
    chart2 = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            y=alt.Y("row_label:N", sort=row_order, axis=alt.Axis(title=None)),
            x=alt.X("count:Q", title="number of patterns"),
            color=alt.Color("category:N", scale=color_scale,
                            legend=alt.Legend(title=None, orient="bottom")),
            order=alt.Order("category:N"),
        )
        .properties(
            width=340, height=18 * len(sdf) + 60,
            title=alt.TitleParams(
                "Pattern stability across support thresholds, per FIM form",
                fontSize=12, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )
    chart2.save(str(OUT_FIG_SUMMARY), scale_factor=2)
    print(f"Saved {OUT_FIG_SUMMARY}")


if __name__ == "__main__":
    main()
