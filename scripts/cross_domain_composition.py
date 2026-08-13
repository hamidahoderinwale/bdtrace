#!/usr/bin/env python3
"""
Cross-domain compositional generalization pilot.

Tests whether composition failure correlates across domains for the same model.
Three domains with prescribed action vocabularies, chain compositions of
increasing length. If a model's accuracy-vs-length curve is similar across
domains, compositional capacity is a general property of the model.
"""

import json
import random
import re
import time
from pathlib import Path

import httpx
import numpy as np
from scipy import stats

# Config

ENV_PATH = Path("/Users/hamidaho/ai-ecosystem-v2/.venv/.env")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = [
    "anthropic/claude-sonnet-4-20250514",
    "openai/gpt-4o-mini",
    "meta-llama/llama-3.3-70b-instruct",
]

CHAIN_LENGTHS = [2, 4, 6, 8, 10]
TASKS_PER_CELL = 5
SEED = 42

OUT_DIR = Path("/Users/hamidaho/learning-from-dev/bidirect-align-dev-traces/output/cross_domain_composition")


def load_api_key() -> str:
    for line in ENV_PATH.read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1]
    raise RuntimeError("No OPENROUTER_API_KEY found")


# Domain 1: arithmetic

def gen_arithmetic(length: int, rng: random.Random) -> dict:
    start = rng.randint(3, 15)
    value = start
    ops = []
    for i in range(length):
        # Ensure diversity: cycle through op types
        op_type = ["add", "subtract", "multiply"][i % 3]
        if op_type == "add":
            arg = rng.randint(2, 9)
            value += arg
        elif op_type == "subtract":
            arg = rng.randint(1, min(5, max(1, value - 1)))
            value -= arg
        else:
            arg = rng.randint(2, 3)
            value *= arg
        ops.append((op_type, arg))

    lines = [f"{i+1}. {n}({a})" for i, (n, a) in enumerate(ops)]
    prompt = (
        "You have a calculator with exactly three operations:\n"
        "- add(n): adds n to the current value\n"
        "- subtract(n): subtracts n from the current value\n"
        "- multiply(n): multiplies the current value by n\n\n"
        f"Starting value: {start}\n\n"
        "Apply these operations in order:\n"
        + "\n".join(lines)
        + "\n\nWhat is the final value? Reply with ONLY the number."
    )
    return {"domain": "arithmetic", "length": length, "ground_truth": value,
            "prompt": prompt, "ops": ops, "start": start}


def check_arithmetic(response: str, gt) -> bool:
    nums = re.findall(r"-?\d+\.?\d*", response.strip())
    if not nums:
        return False
    try:
        return abs(float(nums[-1]) - gt) < 0.01
    except ValueError:
        return False


# Domain 2: string transformation

WORDS = ["hello", "world", "python", "model", "agent", "train", "delta", "craft"]

def gen_string(length: int, rng: random.Random) -> dict:
    start = rng.choice(WORDS)
    value = start
    ops = []
    op_pool = ["uppercase", "lowercase", "reverse", "replace", "append"]

    for i in range(length):
        op = op_pool[i % len(op_pool)]

        if op == "uppercase":
            ops.append(("uppercase", None))
            value = value.upper()
        elif op == "lowercase":
            ops.append(("lowercase", None))
            value = value.lower()
        elif op == "reverse":
            ops.append(("reverse", None))
            value = value[::-1]
        elif op == "replace":
            chars = [c for c in set(value) if c.isalpha()]
            if not chars:
                ops.append(("reverse", None))
                value = value[::-1]
            else:
                old = rng.choice(chars)
                new = rng.choice([c for c in "XYZ*#" if c != old])
                ops.append(("replace", (old, new)))
                value = value.replace(old, new)
        elif op == "append":
            suf = rng.choice(["!", "?", "_ok", "++"])
            ops.append(("append", suf))
            value = value + suf

    lines = []
    for i, (n, a) in enumerate(ops, 1):
        if n == "replace":
            lines.append(f'{i}. replace("{a[0]}", "{a[1]}")')
        elif n == "append":
            lines.append(f'{i}. append("{a}")')
        else:
            lines.append(f"{i}. {n}")

    prompt = (
        "You have a string processor with exactly these operations:\n"
        "- uppercase: convert entire string to uppercase\n"
        "- lowercase: convert entire string to lowercase\n"
        "- reverse: reverse the string\n"
        '- replace("a", "b"): replace ALL occurrences of "a" with "b"\n'
        '- append("s"): append s to the end\n\n'
        f'Starting string: "{start}"\n\n'
        "Apply these operations in order:\n"
        + "\n".join(lines)
        + '\n\nWhat is the final string? Reply with ONLY the string in double quotes.'
    )
    return {"domain": "string", "length": length, "ground_truth": value,
            "prompt": prompt, "ops": [(n, a) for n, a in ops], "start": start}


def check_string(response: str, gt) -> bool:
    m = re.search(r'"([^"]*)"', response)
    if m:
        return m.group(1) == gt
    m = re.search(r"'([^']*)'", response)
    if m:
        return m.group(1) == gt
    return response.strip().strip("\"'") == gt


# Domain 3: list manipulation

def gen_list(length: int, rng: random.Random) -> dict:
    start = rng.sample(range(1, 20), rng.randint(5, 7))
    value = list(start)
    ops = []
    op_pool = ["sort", "reverse", "append", "remove_first", "take_first"]

    for i in range(length):
        op = op_pool[i % len(op_pool)]

        if op == "sort":
            ops.append(("sort", None))
            value = sorted(value)
        elif op == "reverse":
            ops.append(("reverse", None))
            value = list(reversed(value))
        elif op == "append":
            x = rng.randint(1, 15)
            ops.append(("append", x))
            value.append(x)
        elif op == "remove_first":
            if len(value) > 2:
                ops.append(("remove_first", None))
                value = value[1:]
            else:
                ops.append(("sort", None))
                value = sorted(value)
        elif op == "take_first":
            n = min(rng.randint(3, 4), len(value))
            if n < len(value):
                ops.append(("take_first", n))
                value = value[:n]
            else:
                ops.append(("reverse", None))
                value = list(reversed(value))

    lines = []
    for i, (n, a) in enumerate(ops, 1):
        if a is not None:
            lines.append(f"{i}. {n}({a})")
        else:
            lines.append(f"{i}. {n}")

    prompt = (
        "You have a list processor with exactly these operations:\n"
        "- sort: sort the list in ascending order\n"
        "- reverse: reverse the list\n"
        "- append(x): add x to the end\n"
        "- remove_first: remove the first element\n"
        "- take_first(n): keep only the first n elements\n\n"
        f"Starting list: {start}\n\n"
        "Apply these operations in order:\n"
        + "\n".join(lines)
        + "\n\nWhat is the final list? Reply with ONLY the list like [1, 2, 3]."
    )
    return {"domain": "list", "length": length, "ground_truth": value,
            "prompt": prompt, "ops": [(n, a) for n, a in ops], "start": start}


def check_list(response: str, gt) -> bool:
    m = re.search(r"\[([^\]]*)\]", response)
    if not m:
        return False
    try:
        parsed = json.loads(f"[{m.group(1)}]")
        return parsed == gt
    except (json.JSONDecodeError, ValueError):
        return False


# API

def call_model(model: str, prompt: str, api_key: str, retries: int = 2) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 150,
    }
    for attempt in range(retries + 1):
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.post(OPENROUTER_URL, json=data, headers=headers)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == retries:
                return f"ERROR: {e}"
            time.sleep(2)


# Analysis

def analyze(results: list[dict]):
    models = sorted(set(r["model"] for r in results))
    domains = sorted(set(r["domain"] for r in results))
    lengths = sorted(set(r["length"] for r in results))

    # Accuracy table
    acc = {}
    print("\n" + "=" * 70)
    print("Accuracy by model × domain × chain length")
    print("=" * 70)

    for model in models:
        short = model.split("/")[-1]
        print(f"\n  {short}")
        for domain in domains:
            cells = []
            for length in lengths:
                hits = [r for r in results
                        if r["model"] == model and r["domain"] == domain
                        and r["length"] == length]
                a = sum(r["correct"] for r in hits) / max(len(hits), 1)
                acc[(model, domain, length)] = a
                cells.append(f"{a:5.0%}")
            print(f"    {domain:12s}  " + "  ".join(
                f"L{l}={v}" for l, v in zip(lengths, cells)))

    # Cross-domain correlation per model
    print("\n" + "=" * 70)
    print("Within-model cross-domain correlation")
    print("(does the same model's accuracy curve match across domains?)")
    print("=" * 70)

    within_model_rs = []
    for model in models:
        short = model.split("/")[-1]
        print(f"\n  {short}")
        curves = {d: [acc[(model, d, l)] for l in lengths] for d in domains}
        for i, d1 in enumerate(domains):
            for d2 in domains[i + 1:]:
                c1, c2 = curves[d1], curves[d2]
                if len(set(c1)) > 1 and len(set(c2)) > 1:
                    r, p = stats.spearmanr(c1, c2)
                    within_model_rs.append(r)
                    print(f"    {d1} vs {d2}: r={r:+.3f}  p={p:.3f}")
                else:
                    print(f"    {d1} vs {d2}: no variance (all perfect or all fail)")

    # Cross-model correlation per domain
    print("\n" + "=" * 70)
    print("Within-domain cross-model correlation")
    print("(do different models agree on which lengths are hard?)")
    print("=" * 70)

    within_domain_rs = []
    for domain in domains:
        print(f"\n  {domain}")
        curves = {m: [acc[(m, domain, l)] for l in lengths] for m in models}
        for i, m1 in enumerate(models):
            for m2 in models[i + 1:]:
                s1, s2 = m1.split("/")[-1], m2.split("/")[-1]
                c1, c2 = curves[m1], curves[m2]
                if len(set(c1)) > 1 and len(set(c2)) > 1:
                    r, p = stats.spearmanr(c1, c2)
                    within_domain_rs.append(r)
                    print(f"    {s1} vs {s2}: r={r:+.3f}  p={p:.3f}")
                else:
                    print(f"    {s1} vs {s2}: no variance")

    # Key comparison
    print("\n" + "=" * 70)
    print("Key result")
    print("=" * 70)
    if within_model_rs:
        print(f"  Mean within-model cross-domain r: {np.mean(within_model_rs):+.3f}  (n={len(within_model_rs)})")
    if within_domain_rs:
        print(f"  Mean within-domain cross-model r:  {np.mean(within_domain_rs):+.3f}  (n={len(within_domain_rs)})")
    print()
    if within_model_rs and within_domain_rs:
        wm = np.mean(within_model_rs)
        wd = np.mean(within_domain_rs)
        if wm > wd:
            print("  → composition capacity looks model-specific (same model, similar curve across domains)")
        else:
            print("  → composition capacity looks domain-specific (same domain, similar curve across models)")

    # Per-model composition ceiling (length where accuracy first drops below 80%)
    print("\n" + "=" * 70)
    print("Composition ceiling (first length where accuracy < 80%)")
    print("=" * 70)
    for model in models:
        short = model.split("/")[-1]
        ceilings = {}
        for domain in domains:
            ceiling = max(lengths)
            for l in lengths:
                if acc[(model, domain, l)] < 0.8:
                    ceiling = l
                    break
            ceilings[domain] = ceiling
        print(f"  {short:30s}  " + "  ".join(
            f"{d}={ceilings[d]}" for d in domains))

    return acc


# Main

def main():
    api_key = load_api_key()
    rng = random.Random(SEED)

    generators = {
        "arithmetic": (gen_arithmetic, check_arithmetic),
        "string": (gen_string, check_string),
        "list": (gen_list, check_list),
    }

    # Generate tasks
    tasks = []
    for domain in sorted(generators):
        gen_fn = generators[domain][0]
        for length in CHAIN_LENGTHS:
            for idx in range(TASKS_PER_CELL):
                t = gen_fn(length, rng)
                t["task_idx"] = idx
                tasks.append(t)

    n_calls = len(tasks) * len(MODELS)
    print(f"Tasks: {len(tasks)}  |  Models: {len(MODELS)}  |  API calls: {n_calls}")

    # Run
    results = []
    for model in MODELS:
        short = model.split("/")[-1]
        print(f"\n{'='*50}")
        print(f"  {short}")
        print(f"{'='*50}")

        for task in tasks:
            checker = generators[task["domain"]][1]
            response = call_model(model, task["prompt"], api_key)
            correct = checker(response, task["ground_truth"])

            results.append({
                "model": model,
                "domain": task["domain"],
                "length": task["length"],
                "task_idx": task["task_idx"],
                "ground_truth": task["ground_truth"],
                "response": response,
                "correct": correct,
            })

            mark = "+" if correct else "X"
            gt_str = str(task["ground_truth"])[:30]
            resp_str = response[:40].replace("\n", " ")
            print(f"  {mark} {task['domain']:10s} L={task['length']:2d}  "
                  f"gt={gt_str:>30s}  resp={resp_str}")

            time.sleep(0.5)

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {len(results)} results to {OUT_DIR / 'results.json'}")

    # Analyze
    analyze(results)


if __name__ == "__main__":
    main()
