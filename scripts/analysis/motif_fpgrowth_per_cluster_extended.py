"""Per-FIM-form FP-growth on extended corpus (8 submissions).

Same as scripts/analysis/motif_fpgrowth_per_cluster.py but reads
bpe_sequences_extended.jsonl (covers all 8 submissions including the
4 new scaffolds). Pass/fail comes from the union of the 4-agent
parquet and the extended_pass_fail.json published per submission.

Reads:
    output/paper2_pilot/bpe_sequences_extended.jsonl
    output/canonical_forms/instance_assignments.parquet
    output/trajectories/lite_all_models.parquet
    output/paper2_pilot/extended_pass_fail.json
Writes:
    output/paper2_pilot/motif_fpgrowth_templates_extended.json
    output/figures/fig_motif_fpgrowth_templates_extended.png
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
from scripts.theme import register, BLUE
register()

BPE_FILE   = ROOT / "output" / "paper2_pilot" / "bpe_sequences_extended.jsonl"
FORMS_FILE = ROOT / "output" / "canonical_forms" / "instance_assignments.parquet"
LITE_FILE  = ROOT / "output" / "trajectories" / "lite_all_models.parquet"
EXT_PF     = ROOT / "output" / "paper2_pilot" / "extended_pass_fail.json"
OUT_JSON   = ROOT / "output" / "paper2_pilot" / "motif_fpgrowth_templates_extended.json"
OUT_FIG    = ROOT / "output" / "figures"  / "fig_motif_fpgrowth_templates_extended.png"

MIN_SAMPLES_PER_FORM = 30
MIN_SUPPORT          = 0.30
MAX_PATTERN_LEN      = 4
TOP_K_PER_FORM       = 3


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
            generated = info.get("generated") or info.get("no_generation") or []
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
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    bpe = load_canonical_envelopes()
    forms = pd.read_parquet(FORMS_FILE)[["instance_id", "form_name"]].drop_duplicates("instance_id")
    df = bpe.merge(forms, on="instance_id", how="inner")

    pf = load_passfail()
    df["passed"] = df.apply(lambda r: pf.get((r["submission"], r["instance_id"]), None), axis=1)
    df = df.dropna(subset=["passed"]).copy()
    df["passed"] = df["passed"].astype(bool)

    print(f"Joined corpus: {len(df)} trajectories across {df['form_name'].nunique()} forms × {df['agent'].nunique()} agents")

    results: dict[str, dict] = {}

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

        try:
            fp = fpgrowth(
                oh_df.drop(columns=["__key__"]),
                min_support=MIN_SUPPORT,
                use_colnames=True,
                max_len=MAX_PATTERN_LEN,
            )
        except Exception as e:
            print(f"  {form}: fpgrowth failed: {e}")
            continue
        if fp.empty:
            continue
        fp = fp.copy()
        fp = fp[fp["itemsets"].apply(len) >= 2]
        if fp.empty:
            continue
        fp["pattern_list"] = fp["itemsets"].apply(lambda s: sorted(list(s)))

        rows_for_form = []
        for _, r in fp.iterrows():
            pattern = list(r["pattern_list"])
            mask = oh_df[pattern].all(axis=1)
            keys_with = [k for k, m in zip(oh_df["__key__"], mask) if m]
            if not keys_with:
                continue
            n_with = len(keys_with)
            n_pass_with = sum(1 for k in keys_with if k in passed_keys)
            pass_rate = n_pass_with / n_with
            lift = (pass_rate / base) if base > 0 else float("nan")
            rows_for_form.append({
                "pattern":     pattern,
                "size":        len(pattern),
                "support":     round(float(r["support"]), 3),
                "n_with":      int(n_with),
                "pass_rate":   round(pass_rate, 3),
                "base_rate":   round(base, 3),
                "lift":        round(lift, 3) if not (lift != lift) else None,
                "n_pass_with": int(n_pass_with),
            })
        if not rows_for_form:
            continue

        rows_for_form.sort(key=lambda r: -(r["lift"] or 0))
        results[form] = {
            "n_trajectories": int(n),
            "base_pass_rate": round(base, 3),
            "n_distinct_agents_in_form": int(sub["agent"].nunique()),
            "top_patterns": rows_for_form[:TOP_K_PER_FORM],
        }
        print(f"\n  {form} (n={n}, base pass={base:.2%}, agents={sub['agent'].nunique()}):")
        for r in rows_for_form[:TOP_K_PER_FORM]:
            pat_str = " + ".join(r["pattern"])[:80]
            print(f"    [{r['size']}-itemset] {pat_str:80s}  sup={r['support']:.2f}  pass={r['pass_rate']:.2f}  lift={r['lift']}")

    OUT_JSON.write_text(json.dumps({
        "min_samples_per_form": MIN_SAMPLES_PER_FORM,
        "min_support": MIN_SUPPORT,
        "max_pattern_len": MAX_PATTERN_LEN,
        "top_k_per_form": TOP_K_PER_FORM,
        "n_trajectories_total": int(len(df)),
        "n_forms_reported": len(results),
        "results": results,
    }, indent=2, default=str))
    print(f"\nSaved {OUT_JSON}")

    plot_rows = []
    for form, info in results.items():
        if not info["top_patterns"]:
            continue
        p = info["top_patterns"][0]
        if p["lift"] is None:
            continue
        plot_rows.append({
            "form": form,
            "row_label": f"{form} (n={info['n_trajectories']})",
            "pattern": " + ".join(p["pattern"]),
            "support": p["support"],
            "lift": p["lift"],
        })
    if not plot_rows:
        print("No plottable patterns.")
        return
    plot_df = pd.DataFrame(plot_rows).sort_values("lift", ascending=False).reset_index(drop=True)
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
    text = (
        alt.Chart(plot_df)
        .mark_text(align="left", dx=6, fontSize=10, color="#444444")
        .encode(
            y=alt.Y("row_label:N", sort=row_order),
            x=alt.X("lift:Q"),
            text="pattern:N",
        )
    )
    chart = (
        alt.layer(bars, text)
        .properties(
            width=440, height=18 * len(plot_df) + 60,
            title=alt.TitleParams(
                "Top FP-growth pattern per FIM form (extended corpus)",
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
