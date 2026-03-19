"""
Stage-wise edit describer: one grounded sentence per hunk chunk.

Takes a list of EditChunk objects (from ast_edit_sequences.patch_to_chunks)
and produces a staged edit narrative — one sentence per chunk describing
what that chunk structurally accomplished, grounded in its AST sequence.

Example output for a two-chunk patch:
    [
      "Replaced equality-with-None check with identity comparison (is None).",
      "Added guard clause that raises ValueError when value is empty."
    ]

This is strictly more informative than a single fix_type label:
- Each chunk is described independently, preserving the edit order
- The AST sequence grounds the description structurally
- The context lines anchor it to the actual code

Integrates with fix_type.py: the staged descriptions become the
structural certificate passed to FixTypeModule, replacing the flat
hunk-lines summary.
"""

import dspy

from analysis.procedures.ast_edit_sequences import EditChunk


class ChunkDescribeSignature(dspy.Signature):
    """
    Describe what a single code edit chunk accomplished in one sentence.
    Ground the description in the AST sequence — the sequence tells you
    the structural shape of the change; the context tells you what code
    surrounds it.
    """
    ast_sequence = dspy.InputField(
        desc="Ordered AST node tokens for this chunk: DEL_X tokens (removed) "
             "followed by ADD_X tokens (added). E.g. 'DEL_If DEL_Compare ADD_If ADD_Compare ADD_Return'"
    )
    removed_lines = dspy.InputField(
        desc="Raw lines removed in this chunk (up to 5)"
    )
    added_lines = dspy.InputField(
        desc="Raw lines added in this chunk (up to 5)"
    )
    context = dspy.InputField(
        desc="Surrounding unchanged lines for code context (up to 4 lines)"
    )
    description = dspy.OutputField(
        desc="One sentence describing what this chunk accomplished structurally. "
             "Start with a verb. Be specific about the structural operation "
             "(e.g. 'Added guard clause', 'Replaced equality check', 'Extracted helper function'). "
             "Do not describe variable names or values unless essential."
    )


class ChunkDescriber(dspy.Module):
    """Describes each edit chunk as one grounded sentence."""

    def __init__(self, predictor: dspy.Predict | None = None):
        super().__init__()
        self.predictor = predictor or dspy.Predict(ChunkDescribeSignature)

    def forward(self, chunk: EditChunk) -> str:
        """Return one sentence describing what this chunk accomplished."""
        out = self.predictor(
            ast_sequence=" ".join(chunk.sequence[:20]) or "(no parseable AST)",
            removed_lines="\n".join(chunk.removed_lines[:5]) or "(none)",
            added_lines="\n".join(chunk.added_lines[:5]) or "(none)",
            context="\n".join(chunk.context_lines[:4]) or "(none)",
        )
        return (out.description or "").strip()

    def describe_patch(self, chunks: list[EditChunk]) -> list[str]:
        """
        Describe all chunks of a patch, returning one sentence per chunk.
        Skips empty chunks.
        """
        return [self(chunk) for chunk in chunks if not chunk.is_empty]
