#!/usr/bin/env python3
"""
Prompting study: procedural scaffolds vs raw logs vs no context.

Asks a model to complete planning and explainability tasks under three conditions:
  no_context   -- task description only
  raw_logs     -- task + raw patch (unified diff)
  procedural   -- task + structural representations (edits certificate + motifs)

A separate judge model scores each response. Structural methods then localize
failure -- asking where in the procedure the deviation occurred and whether it
recurs across instances.

Usage:
  python scripts/run_prompting_study.py --limit 20
  python scripts/run_prompting_study.py --limit 20 --judge-model openai/gpt-4o
  python scripts/run_prompting_study.py --input output/datasets/swe_bench_lite_resolved/test.parquet --limit 20
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext as _nullcontext
from pathlib import Path

# Wong colorblind-safe palette
# no_context = gray (null baseline), raw_logs = orange, procedural = blue
CONDITION_COLORS = {
    "no_context": "#999999",
    "raw_logs": "#E69F00",
    "procedural": "#0072B2",
}
CONDITION_ORDER = ["no_context", "raw_logs", "procedural"]
SCORE_DIMS = ["localization", "edit_type", "plan_quality", "explanation"]

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
from data.swe_bench import load_swe_bench_lite
from representations import semantic_edits_repr, motifs_repr, tokens_repr
from eval.analysis import divergence_from_baseline

CONDITIONS = ["no_context", "raw_logs", "procedural"]

TASK_PROMPT = """\
You are given a software bug report. Complete two tasks:

1. PLAN: Describe your step-by-step approach to fixing this bug. Be specific about \
which files or functions you would change and why.

2. EXPLAIN: After fixing, explain what your patch does and why it resolves the issue.

Bug report:
{problem_statement}
{context}"""

RAW_CONTEXT = """\

Patch (unified diff):
{patch}"""

PROCEDURAL_CONTEXT = """\

Structural representation of the fix:
Edits: {edits}
Workflow motifs: {motifs}"""

JUDGE_PROMPT = """\
You are evaluating a software engineer's response to a bug report.
You have access to the reference patch — the actual fix that resolves the bug.
Use it as ground truth when scoring.

Bug report:
{problem_statement}

Reference patch (ground truth):
{gold_patch}

Response being evaluated:
{response}

First, reason briefly about each dimension (1-2 sentences each):

LOCALIZATION: Does the response identify the same file(s) and function(s) as the reference patch?
EDIT_TYPE: Does the response correctly characterise what kind of change is needed \
(e.g. add validation, fix off-by-one, change default) relative to the reference?
PLAN_QUALITY: Would following this plan produce a patch that matches the reference? \
Is the sequence of steps coherent and complete?
EXPLANATION: Is the explanation grounded in what the reference patch actually does, \
or does it describe something different?

Then score each dimension 0-3:
  0 = wrong / irrelevant
  1 = partially correct but significant gaps
  2 = mostly correct, minor gaps
  3 = exact match to reference

Identify the first plan step (0-indexed) where the response diverges from the reference \
(-1 if fully aligned). Characterise the divergence:
  surface       -- wrong file/function name (token-level mismatch)
  compositional -- right location, wrong operation or order (syntactic-level mismatch)
  relational    -- right operations but wrong dependency or interaction between components (graph-level)
  none          -- no divergence

Respond in this exact format:
REASONING:
localization: <reasoning>
edit_type: <reasoning>
plan_quality: <reasoning>
explanation: <reasoning>

JSON:
{{"localization": <int>, "edit_type": <int>, "plan_quality": <int>, "explanation": <int>, \
"first_deviation_step": <int>, "divergence_level": "<surface|compositional|relational|none>"}}"""


class TaskModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict("context -> response")

    def forward(self, context: str) -> str:
        out = self.generate(context=context)
        return getattr(out, "response", "") or ""


class JudgeModule(dspy.Module):
    def __init__(self, lm=None):
        super().__init__()
        self.judge = dspy.Predict("prompt -> judgment")
        self._lm = lm

    def forward(self, prompt: str) -> dict:
        ctx = dspy.context(lm=self._lm) if self._lm else _nullcontext()
        with ctx:
            out = self.judge(prompt=prompt)
        raw = getattr(out, "judgment", "") or ""
        # Extract JSON block — appears after "JSON:" marker or as the last {...}
        try:
            marker = raw.rfind("JSON:")
            search_from = marker + 5 if marker >= 0 else 0
            start = raw.find("{", search_from)
            end = raw.rfind("}") + 1
            result = json.loads(raw[start:end]) if start >= 0 else {}
        except (json.JSONDecodeError, ValueError):
            result = {}
        # Preserve reasoning for inspection
        reasoning_marker = raw.find("REASONING:")
        json_marker = raw.rfind("JSON:")
        if reasoning_marker >= 0 and json_marker > reasoning_marker:
            result["_reasoning"] = raw[reasoning_marker + 10: json_marker].strip()
        return result


def _build_context(condition: str, instance: dict, edits: list, motifs_val: dict) -> str:
    problem = instance.get("problem_statement", "")
    if not problem:
        for event in instance.get("events", []):
            if event.get("type") == "prompt":
                problem = event.get("details", {}).get("text", "")
                break

    patch = _extract_patch(instance)

    if condition == "no_context":
        context = ""
    elif condition == "raw_logs":
        context = RAW_CONTEXT.format(patch=patch[:3000] if patch else "(no patch)")
    else:
        edits_str = json.dumps(edits[:5], indent=2) if edits else "(none)"
        if isinstance(motifs_val, dict):
            motifs_seq = motifs_val.get("sequence", [])[:10]
        elif isinstance(motifs_val, list):
            motifs_seq = motifs_val[:10]
        else:
            motifs_seq = []
        motifs_str = json.dumps(motifs_seq) if motifs_seq else "(none)"
        context = PROCEDURAL_CONTEXT.format(edits=edits_str, motifs=motifs_str)

    return TASK_PROMPT.format(problem_statement=problem, context=context)


def _extract_patch(instance: dict) -> str:
    lines = []
    for event in instance.get("events", []):
        if event.get("type") == "code_change":
            d = event.get("details", {})
            path = d.get("file_path", "")
            before = d.get("before_content", "")
            after = d.get("after_content", "")
            if before or after:
                lines.append(f"--- {path}\n+++ {path}")
                for l in before.splitlines():
                    lines.append(f"-{l}")
                for l in after.splitlines():
                    lines.append(f"+{l}")
    return "\n".join(lines)


def _extract_representations(instance: dict) -> tuple[list, dict]:
    edits = []
    motifs_val = {}
    for event in instance.get("events", []):
        if event.get("type") == "code_change":
            d = event.get("details", {})
            cert = semantic_edits_repr(
                d.get("before_content", ""),
                d.get("after_content", ""),
                d.get("file_path"),
            )
            if cert:
                edits.append(cert)
    motifs_val = motifs_repr(instance) or {}
    return edits, motifs_val


_print_lock = threading.Lock()


def _call_with_retry(fn, *args, max_retries: int = 5, **kwargs):
    """Call fn with exponential backoff on rate-limit (429) or transient errors."""
    delay = 2.0
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            msg = str(e).lower()
            is_rate_limit = "429" in msg or "rate limit" in msg or "too many requests" in msg
            is_transient  = "500" in msg or "502" in msg or "503" in msg or "timeout" in msg
            if (is_rate_limit or is_transient) and attempt < max_retries - 1:
                with _print_lock:
                    print(f"  [retry {attempt+1}/{max_retries}] {type(e).__name__}: sleeping {delay:.0f}s", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
            else:
                raise


def _process_one(
    instance: dict,
    task_module: TaskModule,
    judge_module: JudgeModule,
) -> dict:
    """Process a single instance across all conditions. Thread-safe."""
    instance_id = instance.get("instance_id", "unknown")
    problem = ""
    for event in instance.get("events", []):
        if event.get("type") == "prompt":
            problem = event.get("details", {}).get("text", "")
            break
    if not problem:
        problem = instance.get("problem_statement", "")

    edits, motifs_val = _extract_representations(instance)
    tokens = tokens_repr(instance)

    record: dict = {
        "instance_id": instance_id,
        "tokens": tokens,
        "edits": edits,
        "conditions": {},
        "task_model": instance.get("_task_model", "unknown"),
    }

    gold_patch = _extract_patch(instance)
    # Truncate patch to keep prompt manageable but preserve enough for the judge
    gold_patch_trunc = gold_patch[:4000] if gold_patch else "(not available)"

    for condition in CONDITIONS:
        ctx = _build_context(condition, instance, edits, motifs_val)
        response = _call_with_retry(task_module, context=ctx)
        judge_prompt = JUDGE_PROMPT.format(
            problem_statement=problem,
            gold_patch=gold_patch_trunc,
            response=response,
        )
        scores = _call_with_retry(judge_module, prompt=judge_prompt)
        record["conditions"][condition] = {
            "response": response,
            "scores": scores,
        }

    with _print_lock:
        print(f"  {instance_id}: done", flush=True)
    return record


def run_study(
    instances: list[dict],
    task_module: TaskModule,
    judge_module: JudgeModule,
    workers: int = 1,
    checkpoint_path: Path | None = None,
) -> list[dict]:
    # Resume from checkpoint if present
    done_ids: set[str] = set()
    records: list[dict] = []
    if checkpoint_path and checkpoint_path.exists():
        try:
            with open(checkpoint_path) as f:
                records = json.load(f)
            done_ids = {r["instance_id"] for r in records}
            print(f"  Resuming from checkpoint: {len(done_ids)} already done")
        except Exception:
            pass

    remaining = [i for i in instances if i.get("instance_id") not in done_ids]

    if workers <= 1:
        for instance in remaining:
            records.append(_process_one(instance, task_module, judge_module))
            if checkpoint_path:
                with open(checkpoint_path, "w") as f:
                    json.dump(records, f, default=str)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_process_one, inst, task_module, judge_module): inst
                for inst in remaining
            }
            for future in as_completed(futures):
                inst = futures[future]
                try:
                    records.append(future.result())
                except Exception as e:
                    with _print_lock:
                        print(f"  {inst.get('instance_id', '?')}: ERROR — {e}", flush=True)
                if checkpoint_path:
                    with _print_lock:
                        with open(checkpoint_path, "w") as f:
                            json.dump(records, f, default=str)

    return records


def _localize_failures(records: list[dict]) -> list[dict]:
    """
    For each instance, find conditions where first_deviation_step >= 0
    and check whether that step index recurs across instances (structural recurrence).
    """
    step_counts: dict[tuple, int] = {}
    for rec in records:
        for condition, result in rec["conditions"].items():
            step = result.get("scores", {}).get("first_deviation_step", -1)
            if isinstance(step, int) and step >= 0:
                key = (condition, step)
                step_counts[key] = step_counts.get(key, 0) + 1

    failures = []
    for rec in records:
        instance_failures = []
        for condition, result in rec["conditions"].items():
            step = result.get("scores", {}).get("first_deviation_step", -1)
            if isinstance(step, int) and step >= 0:
                recurrence = step_counts.get((condition, step), 1)
                instance_failures.append({
                    "condition": condition,
                    "first_deviation_step": step,
                    "recurrence_count": recurrence,
                    "is_structural": recurrence > 1,
                })
        failures.append({
            "instance_id": rec["instance_id"],
            "failures": instance_failures,
        })
    return failures


def _condition_records(records: list[dict], condition: str) -> list[dict]:
    """Flatten records for one condition into divergence_from_baseline format."""
    out = []
    for rec in records:
        cond_data = rec["conditions"].get(condition, {})
        scores = cond_data.get("scores", {})
        plan_score = scores.get("plan_quality", 0)
        out.append({
            "instance_id": rec["instance_id"],
            "tokens": rec.get("tokens", []),
            "condition": condition,
            "plan_quality": plan_score,
            "localization": scores.get("localization", 0),
            "edit_type": scores.get("edit_type", 0),
            "explanation": scores.get("explanation", 0),
        })
    return out


def _plot_results(records: list[dict], failures: list[dict], out_dir: Path) -> None:
    try:
        import altair as alt
        import pandas as pd
    except ImportError:
        print("Install with: uv sync --extra notebooks  (skipping plots)")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    color_scale = alt.Scale(
        domain=CONDITION_ORDER,
        range=[CONDITION_COLORS[c] for c in CONDITION_ORDER],
    )

    # 1. Grouped bar: mean score per condition × dimension
    rows = []
    for rec in records:
        for condition in CONDITION_ORDER:
            scores = rec["conditions"].get(condition, {}).get("scores", {})
            for dim in SCORE_DIMS:
                v = scores.get(dim)
                if isinstance(v, (int, float)):
                    rows.append({"condition": condition, "dimension": dim, "score": float(v)})

    if rows:
        df = pd.DataFrame(rows)
        mean_df = df.groupby(["condition", "dimension"])["score"].mean().reset_index()
        chart = (
            alt.Chart(mean_df)
            .mark_bar()
            .encode(
                alt.X("condition:N", title=None, sort=CONDITION_ORDER,
                      axis=alt.Axis(labelAngle=0)),
                alt.Y("score:Q", title="mean score (0–3)", scale=alt.Scale(domain=[0, 3])),
                alt.Color("condition:N", scale=color_scale, legend=None),
                alt.Column("dimension:N", title=None,
                           header=alt.Header(labelOrient="bottom", labelPadding=4)),
            )
            .properties(width=110, height=200,
                        title="Condition scores by evaluation dimension")
        )
        chart.save(out_dir / "condition_scores.png")
        print("Saved condition_scores.png")

    # 2. Score lift: procedural − no_context per instance per dimension
    lift_rows = []
    for rec in records:
        iid = rec["instance_id"]
        baseline = rec["conditions"].get("no_context", {}).get("scores", {})
        proc = rec["conditions"].get("procedural", {}).get("scores", {})
        for dim in SCORE_DIMS:
            b = baseline.get(dim)
            p = proc.get(dim)
            if isinstance(b, (int, float)) and isinstance(p, (int, float)):
                lift_rows.append({"instance_id": iid, "dimension": dim, "lift": float(p) - float(b)})

    if lift_rows:
        lift_df = pd.DataFrame(lift_rows)
        lift_chart = (
            alt.Chart(lift_df)
            .mark_boxplot(size=30)
            .encode(
                alt.X("dimension:N", title=None, axis=alt.Axis(labelAngle=0)),
                alt.Y("lift:Q", title="score lift (procedural − no context)",
                      scale=alt.Scale(domain=[-3, 3])),
                alt.Color("dimension:N",
                          scale=alt.Scale(scheme="tableau10"), legend=None),
            )
            .properties(width=350, height=240,
                        title="Per-instance lift: procedural over no-context")
        )
        zero = (
            alt.Chart(pd.DataFrame([{"y": 0}]))
            .mark_rule(color="#444444", strokeDash=[4, 2])
            .encode(alt.Y("y:Q"))
        )
        (lift_chart + zero).save(out_dir / "score_lift.png")
        print("Saved score_lift.png")

    # 3. Failure localization: heatmap of condition × first_deviation_step
    heat_rows = []
    for rec in failures:
        for f in rec["failures"]:
            heat_rows.append({
                "condition": f["condition"],
                "step": f["first_deviation_step"],
                "structural": f["is_structural"],
            })

    if heat_rows:
        heat_df = pd.DataFrame(heat_rows)
        counts = heat_df.groupby(["condition", "step"]).size().reset_index(name="count")
        heatmap = (
            alt.Chart(counts)
            .mark_rect()
            .encode(
                alt.X("step:O", title="first deviation step (plan index)"),
                alt.Y("condition:N", title=None, sort=CONDITION_ORDER),
                alt.Color("count:Q",
                          scale=alt.Scale(scheme="blues"), title="failures"),
            )
            .properties(width=360, height=160,
                        title="Failure localization: where plans deviate")
        )
        heatmap.save(out_dir / "failure_localization_heatmap.png")
        print("Saved failure_localization_heatmap.png")

    # 4. Structural vs model-specific failures per condition
    if heat_rows:
        heat_df["failure_type"] = heat_df["structural"].map(
            {True: "structural (recurring)", False: "model-specific"}
        )
        type_counts = (
            heat_df.groupby(["condition", "failure_type"])
            .size()
            .reset_index(name="count")
        )
        type_chart = (
            alt.Chart(type_counts)
            .mark_bar()
            .encode(
                alt.X("condition:N", title=None, sort=CONDITION_ORDER,
                      axis=alt.Axis(labelAngle=0)),
                alt.Y("count:Q", title="failure count"),
                alt.Color("failure_type:N",
                          scale=alt.Scale(
                              domain=["structural (recurring)", "model-specific"],
                              range=["#D55E00", "#56B4E9"],
                          ),
                          title="failure type"),
            )
            .properties(width=300, height=220,
                        title="Structural vs model-specific failures by condition")
        )
        type_chart.save(out_dir / "failure_types.png")
        print("Saved failure_types.png")

    # 5. Divergence level distribution per condition
    div_rows = []
    for rec in records:
        for condition in CONDITION_ORDER:
            scores = rec["conditions"].get(condition, {}).get("scores", {})
            level = scores.get("divergence_level", "none")
            if level and level != "none":
                div_rows.append({"condition": condition, "divergence_level": level})

    if div_rows:
        LEVEL_ORDER = ["surface", "compositional", "relational"]
        LEVEL_COLORS = ["#CC79A7", "#F0E442", "#009E73"]
        div_df = pd.DataFrame(div_rows)
        div_counts = div_df.groupby(["condition", "divergence_level"]).size().reset_index(name="count")
        div_chart = (
            alt.Chart(div_counts)
            .mark_bar()
            .encode(
                alt.X("condition:N", title=None, sort=CONDITION_ORDER,
                      axis=alt.Axis(labelAngle=0)),
                alt.Y("count:Q", title="failure count"),
                alt.Color("divergence_level:N",
                          sort=LEVEL_ORDER,
                          scale=alt.Scale(domain=LEVEL_ORDER, range=LEVEL_COLORS),
                          title="divergence level"),
                alt.Order("divergence_level:N", sort="ascending"),
            )
            .properties(width=300, height=220,
                        title="Divergence level by condition (surface → compositional → relational)")
        )
        div_chart.save(out_dir / "divergence_levels.png")
        print("Saved divergence_levels.png")


def main():
    parser = argparse.ArgumentParser(description="Run prompting study")
    parser.add_argument("--limit", type=int, default=None, help="Instances to run (default: all)")
    parser.add_argument("--input", type=Path, default=None, help="Parquet of pre-extracted instances")
    parser.add_argument("--split", type=str, default="test", help="SWE-bench split to use (default: test)")
    parser.add_argument("--task-model", type=str, default=None, help="Model for task generation (default: DSPY_MODEL env or gpt-4o-mini)")
    parser.add_argument("--judge-model", type=str, default=None, help="Model for judging (default: same as task model)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Task model temperature (default: 0.0)")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens for task model (default: 1024)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: output/prompting_study/<task-model>)")
    parser.add_argument("--workers", type=int, default=20, help="Parallel threads for API calls (default: 20)")
    parser.add_argument("--conditions", nargs="+", default=None,
                        choices=CONDITIONS, help="Conditions to run (default: all)")
    parser.add_argument("--instance-ids", type=Path, default=None,
                        help="JSON file with list of instance_ids to filter to")
    args = parser.parse_args()

    task_model = args.task_model
    if not configure_dspy(model=task_model, temperature=args.temperature, max_tokens=args.max_tokens):
        print("Set OPENROUTER_API_KEY or OPENAI_API_KEY")
        sys.exit(1)

    # Resolve the actual model name used (for output dir naming)
    import os
    _raw = task_model or os.environ.get("DSPY_MODEL", "openai/gpt-4o-mini")
    task_model_slug = _raw.split("/")[-1].replace("-", "_")

    if args.output_dir is None:
        args.output_dir = Path(f"output/prompting_study/{task_model_slug}")

    if args.input and args.input.exists():
        import pandas as pd
        df = pd.read_parquet(args.input)
        instances = df.head(args.limit).to_dict("records")
        # Wrap parquet rows back into trace format if needed
        instances = [
            r if "events" in r else {"instance_id": r.get("instance_id"), "events": [], **r}
            for r in instances
        ]
    else:
        instances = list(load_swe_bench_lite(split=args.split, limit=args.limit))
    # Filter to specific instance IDs if requested
    if args.instance_ids and args.instance_ids.exists():
        with open(args.instance_ids) as f:
            keep = set(json.load(f))
        instances = [i for i in instances if i.get("instance_id") in keep]
        print(f"Filtered to {len(instances)} instances from {args.instance_ids.name}")

    for inst in instances:
        inst["_task_model"] = task_model_slug

    # Override conditions if requested
    if args.conditions:
        CONDITIONS[:] = args.conditions

    print(f"Loaded {len(instances)} instances, conditions: {CONDITIONS}")

    task_module = TaskModule()

    judge_model_name = args.judge_model
    if judge_model_name:
        _api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        _use_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))
        if _use_openrouter and not judge_model_name.startswith("openrouter/"):
            judge_model_name = f"openrouter/{judge_model_name}" if "/" in judge_model_name else f"openrouter/openai/{judge_model_name}"
        judge_lm = dspy.LM(
            model=judge_model_name,
            api_key=_api_key,
            temperature=0.0,
            max_tokens=1024,
            cache=True,
        )
    else:
        judge_lm = None

    judge_module = JudgeModule(lm=judge_lm)

    checkpoint_path = args.output_dir / "records.checkpoint.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running study ({args.workers} workers)...")
    records = run_study(
        instances, task_module, judge_module,
        workers=args.workers,
        checkpoint_path=checkpoint_path,
    )

    failures = _localize_failures(records)

    # Condition-level score summary
    summary: dict = {"conditions": {}, "n_instances": len(records)}
    for condition in CONDITIONS:
        flat = _condition_records(records, condition)
        if flat:
            dims = ["plan_quality", "localization", "edit_type", "explanation"]
            summary["conditions"][condition] = {
                dim: sum(r[dim] for r in flat) / len(flat) for dim in dims
            }

    # Divergence: how much do token-level representations agree across conditions?
    all_flat = []
    for condition in CONDITIONS:
        all_flat.extend(_condition_records(records, condition))
    div = divergence_from_baseline(
        records,
        baseline_key="tokens",
        structured_keys=["edits"],
    )
    summary["token_edit_divergence"] = div.get("per_procedure", {})

    records_path = args.output_dir / "records.json"
    with open(records_path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"Wrote {records_path}")

    failures_path = args.output_dir / "failure_localization.json"
    with open(failures_path, "w") as f:
        json.dump(failures, f, indent=2)
    print(f"Wrote {failures_path}")

    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")

    print("\nCondition scores:")
    for condition, scores in summary["conditions"].items():
        score_str = "  ".join(f"{k}={v:.2f}" for k, v in scores.items())
        print(f"  {condition}: {score_str}")

    _plot_results(records, failures, args.output_dir)


if __name__ == "__main__":
    main()
