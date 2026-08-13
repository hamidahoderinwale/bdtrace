"""Enumerate SWE-bench submission pools (Lite / Verified / Multimodal).

Queries the github.com/swe-bench/experiments repo's metadata.yaml files
for every submission and builds a table showing:
  - which submissions have public trajectories (`assets.trajs`)
  - which backbone model (`tags.model`)
  - which org (`tags.org`)
  - open-source model / open-source system flags

Usage:
    python -m analysis.preferences.enumerate_submissions
"""

from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

import yaml

SPLITS = ["lite", "verified", "multimodal"]
OUT = Path("output/paper2_pilot/submission_enumeration.json")
OUT_MD = Path("docs/paper2_pilot/submission_enumeration.md")


def gh(path: str) -> dict | list:
    result = subprocess.run(
        ["gh", "api", path], capture_output=True, text=True, check=True
    )
    return json.loads(result.stdout)


def list_submissions(split: str) -> list[str]:
    items = gh(f"repos/swe-bench/experiments/contents/evaluation/{split}")
    return sorted([i["name"] for i in items if i["type"] == "dir"])


def fetch_metadata(split: str, submission: str) -> dict | None:
    try:
        r = gh(f"repos/swe-bench/experiments/contents/evaluation/{split}/{submission}/metadata.yaml")
        content = base64.b64decode(r["content"]).decode("utf-8")
    except Exception:
        return None
    try:
        parsed = yaml.safe_load(content) or {}
    except Exception:
        parsed = {}
    return {
        "assets": parsed.get("assets", {}) or {},
        "tags": parsed.get("tags", {}) or {},
        "info": parsed.get("info", {}) or {},
        "raw": content,
    }


def classify(meta: dict) -> dict:
    if not meta:
        return {"has_trajs": False, "model": "", "org": "", "os_model": "?", "os_system": "?"}
    assets = meta.get("assets", {}) or {}
    tags = meta.get("tags", {}) or {}
    model_field = tags.get("model", "")
    if isinstance(model_field, list):
        model = ", ".join(str(m) for m in model_field)
    else:
        model = str(model_field) if model_field else ""
    return {
        "has_trajs": bool(assets.get("trajs")),
        "model": model.strip(),
        "org": str(tags.get("org", "")),
        "os_model": tags.get("os_model", "?"),
        "os_system": tags.get("os_system", "?"),
    }


def family_of(model: str) -> str:
    m = model.lower()
    if "claude" in m:
        return "Claude"
    if "gpt" in m or "o1" in m or "o3" in m or "o4" in m:
        return "OpenAI"
    if "gemini" in m:
        return "Gemini"
    if "llama" in m:
        return "Llama"
    if "qwen" in m:
        return "Qwen"
    if "deepseek" in m:
        return "DeepSeek"
    if "mixtral" in m or "mistral" in m:
        return "Mistral"
    if "granite" in m:
        return "Granite"
    if "doubao" in m:
        return "Doubao"
    if "grok" in m:
        return "Grok"
    return "other/unknown"


def scaffold_of(submission: str, org: str) -> str:
    s = submission.lower()
    for tag, label in [
        ("sweagent", "SWE-agent"),
        ("swe-agent", "SWE-agent"),
        ("agentless", "Agentless"),
        ("opendevin", "OpenDevin"),
        ("openhands", "OpenHands"),
        ("moatless", "Moatless"),
        ("autocoderover", "AutoCodeRover"),
        ("swe-fixer", "SWE-Fixer"),
        ("sima", "SIMA"),
        ("aide", "Aide"),
        ("lingma", "Lingma"),
        ("hyperagent", "HyperAgent"),
        ("coder", "CodeR"),
        ("rag_", "RAG baseline"),
        ("mentat", "Mentat"),
        ("factory", "Factory"),
        ("marscode", "MarsCode"),
        ("abanteai", "AbanteAI"),
        ("infant", "Infant"),
        ("codestory", "CodeStory"),
        ("ibm", "IBM Research"),
        ("masai", "MASAI"),
    ]:
        if tag in s:
            return label
    if org:
        return org
    return "unknown"


def build_table(per_split: dict) -> list[dict]:
    rows = []
    for split, subs in per_split.items():
        for sub_id, meta in subs.items():
            c = classify(meta)
            rows.append({
                "split": split,
                "submission_id": sub_id,
                "has_trajs": c["has_trajs"],
                "model": c["model"],
                "family": family_of(c["model"]),
                "scaffold": scaffold_of(sub_id, c.get("org", "")),
                "org": c["org"],
                "os_model": c["os_model"],
                "os_system": c["os_system"],
            })
    return rows


def write_markdown(rows: list[dict], path: Path) -> None:
    with_trajs = [r for r in rows if r["has_trajs"]]
    lines = ["# SWE-bench submission enumeration",
             "",
             f"_Auto-generated. Total submissions: {len(rows)}; with public trajectories: {len(with_trajs)}._",
             ""]

    # Per-split summary
    lines.append("## Per-split totals")
    lines.append("")
    lines.append("| split | total | with trajs |")
    lines.append("|---|---|---|")
    for split in SPLITS:
        tot = sum(1 for r in rows if r["split"] == split)
        wt = sum(1 for r in rows if r["split"] == split and r["has_trajs"])
        lines.append(f"| {split} | {tot} | {wt} |")
    lines.append("")

    # Per-family × per-split with-trajs counts
    lines.append("## Family × split (submissions with public trajectories only)")
    lines.append("")
    families = sorted({r["family"] for r in with_trajs})
    lines.append("| family | " + " | ".join(SPLITS) + " | total |")
    lines.append("|---|" + "|".join(["---"] * len(SPLITS)) + "|---|")
    for fam in families:
        row = [fam]
        total = 0
        for split in SPLITS:
            n = sum(1 for r in with_trajs if r["family"] == fam and r["split"] == split)
            row.append(str(n))
            total += n
        row.append(str(total))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Per-scaffold
    lines.append("## Scaffold × split (submissions with public trajectories only)")
    lines.append("")
    scaffolds = sorted({r["scaffold"] for r in with_trajs})
    lines.append("| scaffold | " + " | ".join(SPLITS) + " | total |")
    lines.append("|---|" + "|".join(["---"] * len(SPLITS)) + "|---|")
    for sc in scaffolds:
        row = [sc]
        total = 0
        for split in SPLITS:
            n = sum(1 for r in with_trajs if r["scaffold"] == sc and r["split"] == split)
            row.append(str(n))
            total += n
        row.append(str(total))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Full with-trajs list
    lines.append("## Submissions with public trajectories (full list)")
    lines.append("")
    lines.append("| split | submission_id | family | scaffold | model |")
    lines.append("|---|---|---|---|---|")
    for r in sorted(with_trajs, key=lambda x: (x["split"], x["family"], x["submission_id"])):
        lines.append(
            f"| {r['split']} | `{r['submission_id']}` | {r['family']} | {r['scaffold']} | {r['model']} |"
        )

    path.write_text("\n".join(lines))


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    per_split: dict[str, dict] = {}
    for split in SPLITS:
        print(f"\n=== {split} ===")
        subs = list_submissions(split)
        print(f"  {len(subs)} submissions")
        per_split[split] = {}
        for i, sub in enumerate(subs):
            print(f"    [{i+1}/{len(subs)}] {sub}")
            meta = fetch_metadata(split, sub)
            per_split[split][sub] = meta

    rows = build_table(per_split)
    OUT.write_text(json.dumps({"rows": rows}, indent=2))
    write_markdown(rows, OUT_MD)

    with_trajs = [r for r in rows if r["has_trajs"]]
    print(f"\nTotal: {len(rows)} submissions; with public trajectories: {len(with_trajs)}")

    # Summary by family
    fam_counts: dict[str, int] = {}
    for r in with_trajs:
        fam_counts[r["family"]] = fam_counts.get(r["family"], 0) + 1
    print("\nBy family (with trajectories):")
    for fam, n in sorted(fam_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {fam:<20s}  {n}")

    # Summary by scaffold
    sc_counts: dict[str, int] = {}
    for r in with_trajs:
        sc_counts[r["scaffold"]] = sc_counts.get(r["scaffold"], 0) + 1
    print("\nBy scaffold (with trajectories):")
    for sc, n in sorted(sc_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {sc:<20s}  {n}")

    print(f"\nSaved:\n  {OUT}\n  {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
