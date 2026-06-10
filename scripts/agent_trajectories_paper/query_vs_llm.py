"""EXP runner — structural query vs LLM classification of trajectory properties.

(A) procgrep-style structural query over the canonical atom sequence
    (deterministic ground truth; timed over the FULL corpus; $0), vs
(B) LLM judges over the serialized trajectory text (latency, cost, label).

Structural predicates are judged on a BALANCED per-predicate sample (equal
positives/negatives, so accuracy is meaningful against chance = 0.5). A fuzzy
predicate (compositional divergence) has no ground truth and is scored only by
inter-judge agreement (Fleiss' kappa) — consolidating the paper's classifier box.

  python query_vs_llm.py --k 15 --out query_vs_llm_full.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import httpx

ROOT = Path("/Users/hamidaho/learning-from-dev/bidirect-align-dev-traces")
ENVF = ROOT / "distillation_run/intervention/.env"
CORPUS = ROOT / "output/paper2_pilot/local_rawtext.jsonl"
TEXT_CAP = 4000

PREDICATES: dict[str, tuple] = {
    "edit_streak_5": (
        lambda a: bool(re.search(r"(?:edit ){5,}", " ".join(a) + " ")),
        "the agent made 5 or more file edits in a row, with no reading, searching, or test runs in between",
    ),
    "submitted_without_test": (
        lambda a: ("submit" in a) and ("run_test" not in a),
        "the agent submitted/finished without ever running a test",
    ),
    "tested_before_first_edit": (
        lambda a: ("run_test" in a) and ("edit" not in a or a.index("run_test") < a.index("edit")),
        "the agent ran a test before making its first file edit",
    ),
    "read_streak_4": (
        lambda a: bool(re.search(r"(?:read_file ){4,}", " ".join(a) + " ")),
        "the agent read files 4 or more times in a row without any other kind of action",
    ),
    "never_searched": (
        lambda a: "search_repo" not in a,
        "the agent never searched the repository",
    ),
}
FUZZY = {
    "compositional_divergence": (
        None,
        "the agent's overall procedure is compositionally divergent — it reaches the right area of the "
        "code but performs the wrong operations, or the right operations in the wrong order",
    ),
}
DEFAULT_JUDGES = (
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-haiku",
    "deepseek/deepseek-chat",
)


def _load_key() -> str:
    for line in ENVF.read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit("OPENROUTER_API_KEY not found")


def fleiss_kappa(items: list[list[bool | None]]) -> float | None:
    """Fleiss' kappa over binary labels; items = [[judge labels] per trace]."""
    counts = [sum(1 for x in row if x is not None) for row in items]
    r = max(counts) if counts else 0  # modal rater count; keep only fully-rated items
    if r < 2:
        return None
    table = [[sum(1 for x in row if x is True), sum(1 for x in row if x is False)]
             for row, c in zip(items, counts, strict=False) if c == r]
    if len(table) < 2:
        return None
    n = len(table)
    p_cat = [sum(t[j] for t in table) / (n * r) for j in range(2)]
    p_item = [(t[0] ** 2 + t[1] ** 2 - r) / (r * (r - 1)) for t in table]
    p_bar = sum(p_item) / n
    p_e = sum(p * p for p in p_cat)
    return round((p_bar - p_e) / (1 - p_e), 3) if p_e != 1 else None


def judge(client, key, model, text, definition):
    prompt = (
        "Below is a serialized trace of an AI coding agent's actions on a software task.\n\n"
        f"TRACE:\n{text[:TEXT_CAP]}\n\n"
        f"Question: Is the following true — {definition}?\n"
        "Answer with exactly one word: yes or no."
    )
    t = time.time()
    try:
        r = client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "max_tokens": 4,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=60.0,
        )
        j = r.json()
        ans = j["choices"][0]["message"]["content"].strip().lower()
        tok = int(j.get("usage", {}).get("total_tokens", 0))
        return (True if ans.startswith("y") else False if ans.startswith("n") else None), time.time() - t, tok
    except Exception:  # noqa: BLE001
        return None, time.time() - t, 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=15, help="positives and negatives per predicate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="*", default=list(DEFAULT_JUDGES))
    ap.add_argument("--out", default="query_vs_llm_full.json")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    corpus = [json.loads(l) for l in open(CORPUS)]
    corpus = [r for r in corpus if r.get("atoms") and r.get("text")]

    # (A) procgrep timing over the FULL corpus.
    t0 = time.time()
    truth_full = {p: [fn(r["atoms"]) for r in corpus] for p, (fn, _) in PREDICATES.items()}
    procgrep_ms = (time.time() - t0) * 1000.0
    us_per = procgrep_ms / (len(PREDICATES) * len(corpus)) * 1000
    print(f"procgrep: {len(PREDICATES)} preds x {len(corpus)} traces in {procgrep_ms:.2f} ms "
          f"({us_per:.1f} µs/decision)", flush=True)

    n_judge = sum(min(args.k, sum(t)) + min(args.k, len(t) - sum(t)) for t in truth_full.values())
    n_judge += 2 * args.k  # fuzzy
    print(f"expected judge calls: ~{n_judge * len(args.models)} "
          f"({len(PREDICATES)+1} preds x balanced ~{2*args.k} x {len(args.models)} judges)", flush=True)

    key = _load_key()
    out = {"corpus": len(corpus), "models": args.models,
           "procgrep_ms_full": round(procgrep_ms, 3), "procgrep_us_per_decision": round(us_per, 1),
           "predicates": {}}
    calls = tok_tot = 0
    lat_all: list[float] = []

    with httpx.Client() as client:
        # structural predicates — balanced sample, accuracy + kappa
        for p, (fn, definition) in PREDICATES.items():
            pos = [r for r in corpus if truth_full[p][corpus.index(r)]] if False else \
                  [r for i, r in enumerate(corpus) if truth_full[p][i]]
            neg = [r for i, r in enumerate(corpus) if not truth_full[p][i]]
            rng.shuffle(pos); rng.shuffle(neg)
            sample = pos[: args.k] + neg[: args.k]
            gt = [True] * min(args.k, len(pos)) + [False] * min(args.k, len(neg))
            per_item_labels = []
            jr = {m: {"labels": [], "lat": [], "tok": 0} for m in args.models}
            for r in sample:
                row = []
                for m in args.models:
                    lab, lat, tk = judge(client, key, m, r["text"], definition)
                    jr[m]["labels"].append(lab); jr[m]["lat"].append(lat); jr[m]["tok"] += tk
                    lat_all.append(lat); calls += 1; tok_tot += tk; row.append(lab)
                per_item_labels.append(row)
            jres = {}
            for m in args.models:
                lab = jr[m]["labels"]
                ok = sum(1 for x, y in zip(lab, gt, strict=False) if x is not None and x == y)
                ans = sum(1 for x in lab if x is not None)
                jres[m] = {"accuracy": round(ok / ans, 3) if ans else None, "answered": ans}
            kappa = fleiss_kappa(per_item_labels)
            out["predicates"][p] = {"kind": "structural", "n_balanced": len(sample),
                                    "judges": jres, "kappa": kappa}
            accs = "  ".join(f"{m.split('/')[-1]}={jres[m]['accuracy']}" for m in args.models)
            print(f"  {p:26s} n={len(sample)} acc(chance .50): {accs}  κ={kappa}", flush=True)

        # fuzzy predicate — kappa only
        for p, (_, definition) in FUZZY.items():
            sample = rng.sample(corpus, min(2 * args.k, len(corpus)))
            per_item_labels = []
            for r in sample:
                row = []
                for m in args.models:
                    lab, lat, tk = judge(client, key, m, r["text"], definition)
                    lat_all.append(lat); calls += 1; tok_tot += tk; row.append(lab)
                per_item_labels.append(row)
            kappa = fleiss_kappa(per_item_labels)
            pos_rate = [sum(1 for x in row if x) for row in per_item_labels]
            out["predicates"][p] = {"kind": "fuzzy", "n": len(sample), "kappa": kappa,
                                    "no_ground_truth": True}
            print(f"  {p:26s} n={len(sample)} (no ground truth)  κ={kappa}", flush=True)

    out["totals"] = {"judge_calls": calls, "total_tokens": tok_tot,
                     "mean_llm_latency_s": round(sum(lat_all) / len(lat_all), 2) if lat_all else 0}
    (ROOT / "output/paper2_pilot" / args.out).write_text(json.dumps(out, indent=2))
    print(f"\nSPEED: procgrep {out['procgrep_us_per_decision']} µs vs LLM "
          f"{out['totals']['mean_llm_latency_s']}s per decision "
          f"(~{out['totals']['mean_llm_latency_s']*1e6/out['procgrep_us_per_decision']:.0f}x). "
          f"calls={calls} tokens={tok_tot}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
