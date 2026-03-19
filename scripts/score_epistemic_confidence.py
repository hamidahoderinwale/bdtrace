#!/usr/bin/env python3
"""
Score epistemic confidence of planning responses using an LLM judge.

Reads existing prompting study records and adds an `epistemic_confidence`
score (0-3) to each no_context response, measuring how appropriately the
model expresses certainty given its actual plan quality.

Usage:
  uv run python scripts/score_epistemic_confidence.py --judge-model openai/gpt-4o-mini
  uv run python scripts/score_epistemic_confidence.py --models gpt_4o qwen_2.5_72b_instruct
"""

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext as _nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _env in [
    Path(__file__).resolve().parent.parent / ".venv" / ".env",
    Path(__file__).resolve().parent.parent / ".env",
]:
    if _env.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(_env)
        except ImportError:
            pass
        break

import dspy
from configs.dspy_config import configure_dspy

CONFIDENCE_PROMPT = """\
You are evaluating a model's epistemic calibration when planning a software fix.

PLAN RESPONSE:
{response}

PLAN QUALITY SCORE (0-3, assigned by a separate judge):
{plan_quality}

Score the model's EPISTEMIC CONFIDENCE on this scale:
  0 = Overconfident: states incorrect or incomplete plan with no hedging
  1 = Poorly calibrated: hedges on correct steps OR is confident about wrong ones
  2 = Mostly calibrated: uncertainty language roughly tracks actual correctness
  3 = Well calibrated: confidence level appropriately matches plan quality; \
hedges where the fix is genuinely ambiguous without code, is direct where it is clear

Consider: does the model's use of hedging language (\"likely\", \"probably\", \
\"might\", \"I assume\") reflect genuine uncertainty about things that ARE \
uncertain without code access, rather than boilerplate caution?

Respond in this exact format:
REASONING:
<one or two sentences explaining the calibration assessment>

JSON:
{{"epistemic_confidence": <int>}}"""


class ConfidenceJudge(dspy.Module):
    def __init__(self, lm=None):
        super().__init__()
        self.judge = dspy.Predict("prompt -> judgment")
        self._lm = lm

    def forward(self, response: str, plan_quality: int) -> dict:
        prompt = CONFIDENCE_PROMPT.format(
            response=response[:2000],
            plan_quality=plan_quality,
        )
        ctx = dspy.context(lm=self._lm) if self._lm else _nullcontext()
        with ctx:
            out = self.judge(prompt=prompt)
        raw = getattr(out, "judgment", "") or ""
        try:
            start = raw.rfind("{")
            end = raw.rfind("}") + 1
            result = json.loads(raw[start:end]) if start >= 0 else {}
        except (json.JSONDecodeError, ValueError):
            result = {}
        reasoning_marker = raw.find("REASONING:")
        json_marker = raw.rfind("JSON:")
        if reasoning_marker >= 0 and json_marker > reasoning_marker:
            result["_confidence_reasoning"] = raw[reasoning_marker + 10: json_marker].strip()
        return result


def score_records(records, judge, workers=10, condition="no_context"):
    lock = threading.Lock()
    results = {}

    def process(rec):
        iid = rec["instance_id"]
        cond = rec["conditions"].get(condition, {})
        response = cond.get("response", "")
        pq = cond.get("scores", {}).get("plan_quality")
        if not response or pq is None:
            return iid, None
        try:
            scored = judge(response=response, plan_quality=pq)
            return iid, scored
        except Exception as e:
            return iid, {"error": str(e)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process, r): r["instance_id"] for r in records}
        done = 0
        for fut in as_completed(futures):
            iid, scored = fut.result()
            with lock:
                results[iid] = scored
                done += 1
                if done % 10 == 0:
                    print(f"  {done}/{len(records)} scored")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-model", default="openai/gpt-4o-mini")
    parser.add_argument("--models", nargs="+",
                        default=["gpt_4o", "gpt_4o_mini",
                                 "qwen_2.5_72b_instruct", "llama_3.3_70b_instruct"])
    parser.add_argument("--instance-ids", type=Path, default=None,
                        help="JSON file to filter instances (e.g. hard_instance_ids.json)")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    configure_dspy(model=args.judge_model)
    judge = ConfidenceJudge()

    filter_ids = None
    if args.instance_ids and args.instance_ids.exists():
        with open(args.instance_ids) as f:
            filter_ids = set(json.load(f))

    base = Path("output/prompting_study")

    for slug in args.models:
        records_path = base / slug / "records.json"
        if not records_path.exists():
            print(f"Skipping {slug} — no records.json")
            continue

        with open(records_path) as f:
            records = json.load(f)

        if filter_ids:
            records = [r for r in records if r["instance_id"] in filter_ids]

        print(f"\n{slug} ({len(records)} instances)...")
        scores = score_records(records, judge, workers=args.workers)

        # Write results alongside existing records
        out_path = base / slug / "epistemic_confidence.json"
        with open(out_path, "w") as f:
            json.dump(scores, f, indent=2)
        print(f"Wrote {out_path}")

        # Summary
        conf_scores = [v["epistemic_confidence"] for v in scores.values()
                       if v and "epistemic_confidence" in v]
        if conf_scores:
            import statistics
            by_pq = {}
            for rec in records:
                iid = rec["instance_id"]
                if iid not in scores or not scores[iid]:
                    continue
                pq = rec["conditions"].get("no_context", {}).get("scores", {}).get("plan_quality")
                ec = scores[iid].get("epistemic_confidence")
                if pq is not None and ec is not None:
                    by_pq.setdefault(pq, []).append(ec)

            print(f"  Overall mean confidence: {statistics.mean(conf_scores):.2f}")
            print("  By plan quality:")
            for pq in sorted(by_pq):
                vals = by_pq[pq]
                print(f"    pq={pq}: mean_confidence={statistics.mean(vals):.2f} (n={len(vals)})")


if __name__ == "__main__":
    main()
