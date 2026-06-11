"""EXP runner — structural query vs LLM classification of trajectory properties.

(A) procgrep structural query over the canonical atom sequence (deterministic
    ground truth; timed over the FULL corpus; $0) vs
(B) LLM judges over the serialized trajectory text (label + confidence score).

Structural predicates are judged on a BALANCED per-predicate sample (chance =
0.50). A fuzzy predicate (compositional divergence) has no ground truth and is
scored only by inter-judge agreement.

Metrics (all additive, nothing dropped):
  - accuracy (balanced) + Fleiss kappa  [original]
  - precision / recall / F1 / MCC vs the exact answer, with bootstrap 90% CIs
  - ROC-AUC + Brier (calibration) from elicited confidence scores
  - pairwise Cohen kappa (inter-judge; tolerates a missing judge)
  - latency + tokens per judge  -> quality-vs-cost Pareto

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
import numpy as np
from sklearn.metrics import (
    brier_score_loss,
    cohen_kappa_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

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
# 3 cheap judges + 1 stronger judge (does scale clear the structural predicates?).
DEFAULT_JUDGES = (
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-haiku",
    "deepseek/deepseek-chat",
    "anthropic/claude-sonnet-4-6",
)
RNG = np.random.RandomState(0)


def _load_key() -> str:
    for line in ENVF.read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit("OPENROUTER_API_KEY not found")


def fleiss_kappa(items: list[list[bool | None]]) -> float | None:
    counts = [sum(1 for x in row if x is not None) for row in items]
    r = max(counts) if counts else 0
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


def pairwise_cohen(per_item: list[list[bool | None]], n_judges: int) -> float | None:
    """Mean Cohen kappa over judge pairs, each on the items both answered."""
    ks = []
    for i in range(n_judges):
        for j in range(i + 1, n_judges):
            a, b = [], []
            for row in per_item:
                if row[i] is not None and row[j] is not None:
                    a.append(int(row[i]))
                    b.append(int(row[j]))
            if len(set(a)) > 1 or len(set(b)) > 1:
                if len(a) >= 2:
                    ks.append(cohen_kappa_score(a, b))
    return round(float(np.mean(ks)), 3) if ks else None


def _metrics(labels: list[bool | None], scores: list[float | None], truth: list[bool]) -> dict:
    """P/R/F1/MCC (+ bootstrap CI) and AUC/Brier where computable."""
    idx = [i for i, x in enumerate(labels) if x is not None]
    y = np.array([truth[i] for i in idx], dtype=int)
    yhat = np.array([int(labels[i]) for i in idx], dtype=int)
    n = len(idx)
    out: dict = {"answered": n}
    if n == 0 or len(set(y.tolist())) < 2:
        return out | {"note": "degenerate (no positives/negatives answered)"}
    out["accuracy"] = round(float((y == yhat).mean()), 3)
    p, r, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
    out |= {"precision": round(float(p), 3), "recall": round(float(r), 3), "f1": round(float(f1), 3)}
    out["mcc"] = round(float(matthews_corrcoef(y, yhat)), 3)
    # bootstrap 90% CI on F1
    boots = []
    for _ in range(1000):
        b = RNG.randint(0, n, n)
        if len(set(y[b].tolist())) < 2:
            continue
        boots.append(precision_recall_fscore_support(y[b], yhat[b], average="binary", zero_division=0)[2])
    if boots:
        out["f1_ci90"] = [round(float(np.percentile(boots, 5)), 3), round(float(np.percentile(boots, 95)), 3)]
    # AUC + Brier from confidence scores
    sc = [scores[i] for i in idx]
    if all(s is not None for s in sc):
        s = np.array(sc, dtype=float)
        try:
            out["auc"] = round(float(roc_auc_score(y, s)), 3)
            out["brier"] = round(float(brier_score_loss(y, s)), 3)
        except ValueError:
            pass
    return out


def judge(client, key, model, text, definition):
    """Return (label|None, score|None, latency_s, tokens). Score in [0,1]."""
    prompt = (
        "Below is a serialized trace of an AI coding agent's actions on a software task.\n\n"
        f"TRACE:\n{text[:TEXT_CAP]}\n\n"
        f"Question: Is the following true — {definition}?\n"
        "Answer with one word (yes or no), then a space and your confidence as a percentage 0-100.\n"
        "Example: yes 85"
    )
    t = time.time()
    try:
        r = client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "max_tokens": 8,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=60.0,
        )
        j = r.json()
        ans = j["choices"][0]["message"]["content"].strip().lower()
        tok = int(j.get("usage", {}).get("total_tokens", 0))
        label = True if ans.startswith("y") else False if ans.startswith("n") else None
        conf = re.search(r"(\d{1,3})", ans)
        c = min(100, int(conf.group(1))) / 100 if conf else 0.5
        score = c if label else (1 - c) if label is False else None
        return label, score, time.time() - t, tok
    except Exception:  # noqa: BLE001
        return None, None, time.time() - t, 0


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

    t0 = time.time()
    truth_full = {p: [fn(r["atoms"]) for r in corpus] for p, (fn, _) in PREDICATES.items()}
    procgrep_ms = (time.time() - t0) * 1000.0
    us_per = procgrep_ms / (len(PREDICATES) * len(corpus)) * 1000
    print(f"procgrep: {len(PREDICATES)} preds x {len(corpus)} traces in {procgrep_ms:.2f} ms "
          f"({us_per:.1f} µs/decision)", flush=True)

    key = _load_key()
    out = {"corpus": len(corpus), "models": args.models,
           "procgrep_ms_full": round(procgrep_ms, 3), "procgrep_us_per_decision": round(us_per, 1),
           "predicates": {}}
    calls = tok_tot = 0
    lat_by_model: dict[str, list[float]] = {m: [] for m in args.models}
    tok_by_model: dict[str, int] = {m: 0 for m in args.models}

    with httpx.Client() as client:
        for p, (fn, definition) in PREDICATES.items():
            pos = [r for i, r in enumerate(corpus) if truth_full[p][i]]
            neg = [r for i, r in enumerate(corpus) if not truth_full[p][i]]
            rng.shuffle(pos); rng.shuffle(neg)
            sample = pos[: args.k] + neg[: args.k]
            gt = [True] * min(args.k, len(pos)) + [False] * min(args.k, len(neg))
            per_item = []
            jr = {m: {"lab": [], "sc": []} for m in args.models}
            for r in sample:
                row = []
                for m in args.models:
                    lab, sc, lat, tk = judge(client, key, m, r["text"], definition)
                    jr[m]["lab"].append(lab); jr[m]["sc"].append(sc)
                    lat_by_model[m].append(lat); tok_by_model[m] += tk
                    calls += 1; tok_tot += tk; row.append(lab)
                per_item.append(row)
            judges = {m: _metrics(jr[m]["lab"], jr[m]["sc"], gt) for m in args.models}
            out["predicates"][p] = {
                "kind": "structural", "n_balanced": len(sample),
                "judges": judges,
                "fleiss_kappa": fleiss_kappa(per_item),
                "pairwise_cohen_kappa": pairwise_cohen(per_item, len(args.models)),
            }
            f1s = "  ".join(f"{m.split('/')[-1]}:F1={judges[m].get('f1','?')}/AUC={judges[m].get('auc','?')}"
                            for m in args.models)
            print(f"  {p:24s} {f1s}  κ={out['predicates'][p]['pairwise_cohen_kappa']}", flush=True)

        for p, (_, definition) in FUZZY.items():
            sample = rng.sample(corpus, min(2 * args.k, len(corpus)))
            per_item = []
            for r in sample:
                row = []
                for m in args.models:
                    lab, sc, lat, tk = judge(client, key, m, r["text"], definition)
                    lat_by_model[m].append(lat); tok_by_model[m] += tk
                    calls += 1; tok_tot += tk; row.append(lab)
                per_item.append(row)
            out["predicates"][p] = {"kind": "fuzzy", "n": len(sample), "no_ground_truth": True,
                                    "fleiss_kappa": fleiss_kappa(per_item),
                                    "pairwise_cohen_kappa": pairwise_cohen(per_item, len(args.models))}
            print(f"  {p:24s} (fuzzy)  κ={out['predicates'][p]['pairwise_cohen_kappa']}", flush=True)

    # Pareto-ready per-judge summary: mean F1 across structural predicates vs latency/cost.
    pareto = {}
    for m in args.models:
        f1s = [out["predicates"][p]["judges"][m].get("f1") for p in PREDICATES
               if out["predicates"][p]["judges"][m].get("f1") is not None]
        lat = lat_by_model[m]
        pareto[m] = {"mean_f1": round(float(np.mean(f1s)), 3) if f1s else None,
                     "mean_latency_s": round(float(np.mean(lat)), 2) if lat else 0,
                     "tokens": tok_by_model[m]}
    out["pareto"] = {"procgrep": {"mean_f1": 1.0, "us_per_decision": round(us_per, 1), "cost": 0}, **pareto}
    out["totals"] = {"judge_calls": calls, "total_tokens": tok_tot}
    (ROOT / "output/paper2_pilot" / args.out).write_text(json.dumps(out, indent=2))
    print(f"\nprocgrep {us_per:.1f} µs/decision (exact, $0) vs judges:")
    for m, v in pareto.items():
        print(f"  {m:28s} F1={v['mean_f1']}  {v['mean_latency_s']}s/decision")
    print(f"wrote {args.out}  ({calls} calls, {tok_tot} tokens)")


if __name__ == "__main__":
    main()
