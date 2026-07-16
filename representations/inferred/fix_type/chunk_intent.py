"""
Chunk-level fix mechanism classification.

Classifies each EditChunk into one mechanism label from fix_mechanism_vocabulary
(configs/benchmarks.yaml). A patch then becomes a sequence of mechanism tokens,
one per hunk — analogous to how motif sequences tokenize behavioral events.

Example:
    patch with 3 hunks → ["add_import", "add_iteration", "add_guard"]

This sequence representation is richer than either:
- A single whole-patch label (loses per-hunk resolution)
- A bag of AST op types (loses semantic intent)

Follows the ChunkDescriber / FixTypeModule DSPy pattern exactly:
vocabulary loaded from YAML config at import time, not hard-coded.
"""

from pathlib import Path

import dspy
import yaml

from analysis.procedures.ast_edit_sequences import EditChunk, patch_to_chunks


def _load_mechanism_vocab() -> dict[str, str]:
    config_path = Path(__file__).resolve().parents[3] / "configs" / "benchmarks.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    vocab = cfg.get("fix_mechanism_vocabulary", {})
    if not vocab:
        raise ValueError(f"fix_mechanism_vocabulary missing from {config_path}")
    return vocab


MECHANISM_VOCAB: dict[str, str] = _load_mechanism_vocab()
MECHANISM_LABELS: list[str] = list(MECHANISM_VOCAB.keys())


class ChunkIntentSignature(dspy.Signature):
    """
    Classify one diff hunk into the fix mechanism it implements.

    The AST sequence describes the structural shape of the change.
    The added/removed lines show the actual code.
    The context shows surrounding unchanged code.
    Ground the label in the AST sequence — not the variable names or comments.
    """
    ast_sequence = dspy.InputField(
        desc="Ordered AST tokens for this hunk: DEL_X tokens then ADD_X tokens. "
             "E.g. 'DEL_For DEL_comprehension ADD_For ADD_Assign ADD_If'"
    )
    removed_lines = dspy.InputField(
        desc="Lines removed in this hunk (up to 6 lines)"
    )
    added_lines = dspy.InputField(
        desc="Lines added in this hunk (up to 6 lines)"
    )
    context_lines = dspy.InputField(
        desc="Surrounding unchanged lines for code context (up to 4 lines)"
    )
    mechanism_vocabulary = dspy.InputField(
        desc="Allowed mechanism labels with descriptions"
    )
    mechanism = dspy.OutputField(
        desc="Exactly one label from mechanism_vocabulary. "
             "Choose the label that best describes what this hunk structurally accomplishes."
    )


class ChunkIntentModule(dspy.Module):
    """Classifies each EditChunk into a fix mechanism label from the YAML vocabulary."""

    def __init__(self, predictor: dspy.Predict | None = None):
        super().__init__()
        self.predictor = predictor or dspy.Predict(ChunkIntentSignature)
        self._vocab_str = "\n".join(
            f"  {label}: {desc}" for label, desc in MECHANISM_VOCAB.items()
        )

    def forward(self, chunk: EditChunk) -> str:
        """Return one mechanism label for a single chunk."""
        out = self.predictor(
            ast_sequence=" ".join(chunk.sequence[:20]) or "(no parseable AST)",
            removed_lines="\n".join(chunk.removed_lines[:6]) or "(none)",
            added_lines="\n".join(chunk.added_lines[:6]) or "(none)",
            context_lines="\n".join(chunk.context_lines[:4]) or "(none)",
            mechanism_vocabulary=self._vocab_str,
        )
        label = (out.mechanism or "other").strip().lower().replace(" ", "_")
        # Normalize to vocabulary
        if label not in MECHANISM_LABELS:
            for candidate in MECHANISM_LABELS:
                if candidate in label or label in candidate:
                    label = candidate
                    break
            else:
                label = "other"
        return label

    def label_patch(self, patch: str) -> list[str]:
        """
        Parse a unified diff patch into chunks and return one mechanism
        label per non-empty chunk. Result is the mechanism sequence for
        this patch.
        """
        chunks = patch_to_chunks(patch)
        return [self(chunk) for chunk in chunks if not chunk.is_empty]

    def label_trace(self, trace: dict) -> list[str]:
        """
        Extract mechanism sequence from a resolved trace record.
        Iterates code_change events, builds per-file diffs, classifies each chunk.
        """
        import difflib
        labels = []
        for ev in trace["events"]:
            if ev["type"] != "code_change":
                continue
            d = ev["details"]
            if not d["file_path"].endswith(".py"):
                continue
            before = (d["before_content"] or "").splitlines(keepends=True)
            after = (d["after_content"] or "").splitlines(keepends=True)
            if before == after:
                continue
            raw = "".join(difflib.unified_diff(
                before, after,
                fromfile=d["file_path"], tofile=d["file_path"],
            ))
            if not raw:
                continue
            patch = f"diff --git a/{d['file_path']} b/{d['file_path']}\n" + raw
            labels.extend(self.label_patch(patch))
        return labels
