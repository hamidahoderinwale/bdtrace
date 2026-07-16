"""
Fix-type labeling: AST stage extraction + LLM summary → discrete fix_type label.

Two-layer pipeline:
1. AST stage: extract hunk-local features from the diff (added/removed lines only).
   Produces a structural certificate without the full-file noise problem.
2. LLM summary: given the stage features + diff text, produce a fix_type label
   from a closed vocabulary (guard_clause, api_change, refactor, etc.)

This provides the fix-type stratification needed for:
- Conditional procedural analysis (within fix type, does procedure predict pass?)
- Solution novelty scoring (motif rarity within fix type)
- Canonical template mining per fix type

Reuses:
- configs/dspy_config.py: LM configuration
- representations/inferred pattern: DSPy Signature + Module
- data/swebench_trajectories.py: file-type classifier pattern
"""

import ast
import re
from pathlib import Path
from typing import Any

import dspy
import yaml

## Load vocabulary from configs/benchmarks.yaml at import time.
## This means the list of fix types lives in one place — the config file.

def _load_fix_types() -> list[str]:
    config_path = Path(__file__).resolve().parents[3] / "configs" / "benchmarks.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    vocab: dict = cfg.get("fix_type_vocabulary", {})
    if not vocab:
        raise ValueError(f"fix_type_vocabulary missing from {config_path}")
    return list(vocab.keys())


def _load_fix_type_descriptions() -> dict[str, str]:
    config_path = Path(__file__).resolve().parents[3] / "configs" / "benchmarks.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("fix_type_vocabulary", {})


FIX_TYPES: list[str] = _load_fix_types()
FIX_TYPE_DESCRIPTIONS: dict[str, str] = _load_fix_type_descriptions()


## Stage 1: Hunk-local AST feature extraction

def _hunk_lines(patch: str) -> tuple[list[str], list[str]]:
    """Extract added (+) and removed (-) lines from a diff patch."""
    added, removed = [], []
    for line in patch.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            removed.append(line[1:])
    return added, removed


def _safe_parse(src: str) -> ast.AST | None:
    try:
        return ast.parse(src)
    except SyntaxError:
        return None


def _node_types(tree: ast.AST) -> list[str]:
    return [type(n).__name__ for n in ast.walk(tree)]


def _op_signature(lines: list[str]) -> dict[str, Any]:
    """
    Extract structural features from a set of code lines (hunk-local).

    Returns:
        node_types: AST node types present (if parseable)
        has_return: contains return statement
        has_if: contains conditional
        has_raise: contains raise
        has_try: contains try/except
        has_import: contains import
        has_assert: contains assert
        api_calls: external function/method calls (name.method pattern)
        n_lines: line count
        raw: joined lines for LLM input
    """
    joined = "\n".join(lines)
    tree = _safe_parse(joined)
    node_types: list[str] = _node_types(tree) if tree else []

    # API call detection: name.method(...) patterns
    api_calls = list(dict.fromkeys(re.findall(r'\b([A-Za-z_]\w*\.[A-Za-z_]\w*)\s*\(', joined)))

    return {
        "node_types": node_types[:30],
        "has_return": "Return" in node_types or "return" in joined,
        "has_if": "If" in node_types or "if " in joined,
        "has_raise": "Raise" in node_types or "raise " in joined,
        "has_try": "Try" in node_types or "try:" in joined,
        "has_import": "Import" in node_types or "import " in joined,
        "has_assert": "Assert" in node_types or "assert " in joined,
        "api_calls": api_calls[:10],
        "n_lines": len(lines),
        "raw": joined[:500],
    }


def extract_ast_stage(patch: str, before_content: str = "", after_content: str = "") -> dict[str, Any]:
    """
    Extract AST-level stage features from a patch.

    Uses hunk lines only (not full file) to avoid full-file noise.
    Returns a structured certificate ready for the LLM.
    """
    added, removed = _hunk_lines(patch)
    added_sig = _op_signature(added)
    removed_sig = _op_signature(removed)

    # File-level context: which layer is this in?
    file_match = re.search(r'diff --git a/([\w/._-]+)', patch)
    file_path = file_match.group(1) if file_match else ""
    is_test = bool(re.search(r'test_|_test\.|/tests?/', file_path))
    is_config = bool(re.search(r'settings|config|setup\.py', file_path))

    # Number of files changed
    n_files = len(re.findall(r'^diff --git', patch, re.MULTILINE))

    return {
        "file_path": file_path,
        "is_test_file": is_test,
        "is_config_file": is_config,
        "n_files_changed": n_files,
        "added": added_sig,
        "removed": removed_sig,
        "net_lines": added_sig["n_lines"] - removed_sig["n_lines"],
        "patch_preview": patch[:600],
    }


def format_stage_for_llm(stage: dict[str, Any]) -> str:
    """Serialize AST stage to a compact string for LLM input."""
    lines = [
        f"File: {stage['file_path']} (test={stage['is_test_file']}, n_files={stage['n_files_changed']})",
        f"Added ({stage['added']['n_lines']} lines): {stage['added']['raw'][:300]}",
        f"Removed ({stage['removed']['n_lines']} lines): {stage['removed']['raw'][:300]}",
        f"Added node types: {stage['added']['node_types'][:12]}",
        f"Added API calls: {stage['added']['api_calls']}",
        f"Signals: return={stage['added']['has_return']} if={stage['added']['has_if']} "
        f"raise={stage['added']['has_raise']} try={stage['added']['has_try']}",
    ]
    return "\n".join(lines)


## Stage 2: DSPy LLM fix-type labeler

class FixTypeSignature(dspy.Signature):
    """
    Classify a code patch into a fix type from the given vocabulary.
    Ground the classification in the AST stage features.
    """
    stage_features = dspy.InputField(desc="AST-level patch stage: file, added/removed lines, node types, API calls, signals")
    problem_statement = dspy.InputField(desc="Issue description (1-3 sentences)")
    fix_type_vocabulary = dspy.InputField(desc="Allowed fix_type values")

    fix_type = dspy.OutputField(desc="One value from fix_type_vocabulary. Must be exact.")
    summary = dspy.OutputField(desc="One sentence: what changed and why, grounded in the stage features")
    library_pattern = dspy.OutputField(desc="Library/framework API pattern used if any (e.g. 'Flask abort(404)', 'Django Q objects'), else 'none'")
    confidence = dspy.OutputField(desc="high / medium / low")


class FixTypeModule(dspy.Module):
    """DSPy module for fix-type classification."""

    def __init__(self, predictor: dspy.Predict | None = None):
        super().__init__()
        self.predictor = predictor or dspy.Predict(FixTypeSignature)

    def forward(
        self,
        stage: dict[str, Any],
        problem_statement: str = "",
    ) -> dict[str, Any]:
        stage_str = format_stage_for_llm(stage)
        # Pass label + description pairs so the LLM has full context from the config
        vocab_str = "\n".join(f"  {k}: {v}" for k, v in FIX_TYPE_DESCRIPTIONS.items())
        out = self.predictor(
            stage_features=stage_str,
            problem_statement=problem_statement[:400] or "(not provided)",
            fix_type_vocabulary=vocab_str,
        )
        # Normalize fix_type to vocabulary
        ft = (out.fix_type or "other").strip().lower().replace(" ", "_")
        if ft not in FIX_TYPES:
            # Fuzzy match
            for candidate in FIX_TYPES:
                if candidate in ft or ft in candidate:
                    ft = candidate
                    break
            else:
                ft = "other"
        return {
            "fix_type": ft,
            "summary": out.summary or "",
            "library_pattern": out.library_pattern or "none",
            "confidence": out.confidence or "medium",
            "stage": stage,
        }
