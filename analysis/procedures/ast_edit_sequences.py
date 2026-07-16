"""
Hunk-local AST node sequence extractor — per-chunk, staged.

Each `@@` hunk in a unified diff is a coherent, self-contained edit.
This module parses hunks individually so downstream consumers get:

    patch → [Chunk, Chunk, ...]

where each Chunk carries:
    - header:      the raw @@ line (file position context)
    - removed_seq: DEL_X tokens from deleted lines
    - added_seq:   ADD_X tokens from added lines
    - sequence:    combined [DEL_X, ..., ADD_X, ...] for PrefixSpan
    - context:     surrounding unchanged lines (for LLM grounding)

This feeds two consumers:
1. PrefixSpan / TF-IDF distance matrix — use `chunk.sequence` per chunk,
   or flatten to `patch_to_ast_sequence()` for one sequence per patch.
2. DSPy stage-wise describer — use `ChunkDescriber.forward(chunk)` to get
   one grounded sentence per chunk, building a staged edit narrative.

The chunk boundary is important: each chunk is 3-15 lines on average,
so ADD_Name appears 1-2 times rather than 4+. Patterns are denser and
more distinctive than in the flat whole-patch representation.
"""

import ast
import re
import textwrap
from collections import Counter
from dataclasses import dataclass, field


## Node types that are purely syntactic scaffolding — no semantic content
_SKIP_NODES = frozenset({
    "Module", "Interactive", "Expression", "FunctionType",
    "Load", "Store", "Del", "AugLoad", "AugStore", "Param",
    "And", "Or",
    "Add", "Sub", "Mult", "MatMult", "Div", "Mod", "Pow",
    "LShift", "RShift", "BitOr", "BitXor", "BitAnd", "FloorDiv",
    "Invert", "Not", "UAdd", "USub",
    "Eq", "NotEq", "Lt", "LtE", "Gt", "GtE", "Is", "IsNot", "In", "NotIn",
    "expr_context", "boolop", "operator", "unaryop", "cmpop",
})

## Fallback keyword list when dedented lines still won't parse
## (mid-expression fragments, decorator lines, continuations).
## Ordered most-specific first to avoid partial matches.
_FALLBACK_KEYWORDS = (
    "elif ", "else:", "except ", "assert ", "lambda ",
    "yield ", "return ", "raise ", "import ", "while ",
    "class ", "with ", "async ", "for ", "def ", "try:", "if ",
)

_HUNK_HEADER = re.compile(r'^@@[^@@]*@@')


@dataclass
class EditChunk:
    """One @@ hunk from a unified diff, with per-chunk AST sequences."""
    header: str               # raw @@ line
    file_path: str            # file this hunk belongs to
    removed_lines: list[str]  # raw deleted lines (without leading -)
    added_lines: list[str]    # raw added lines (without leading +)
    context_lines: list[str]  # surrounding unchanged lines (for LLM)
    removed_seq: list[str] = field(default_factory=list)  # DEL_X tokens
    added_seq: list[str] = field(default_factory=list)    # ADD_X tokens

    @property
    def sequence(self) -> list[str]:
        """Combined DEL then ADD token sequence for PrefixSpan."""
        return self.removed_seq + self.added_seq

    @property
    def is_empty(self) -> bool:
        return not self.removed_seq and not self.added_seq

    def summary_str(self) -> str:
        """Compact representation for display / LLM input."""
        lines = [f"Hunk {self.header} in {self.file_path}"]
        if self.removed_lines:
            lines.append("  removed: " + " | ".join(self.removed_lines[:4]))
        if self.added_lines:
            lines.append("  added:   " + " | ".join(self.added_lines[:4]))
        lines.append(f"  AST seq: {' → '.join(self.sequence[:10])}"
                     + (" ..." if len(self.sequence) > 10 else ""))
        return "\n".join(lines)


def _lines_to_ast_nodes(lines: list[str]) -> list[str]:
    """
    Parse a list of code lines and return AST node type names in walk order.

    Dedents before parsing. Falls back to leading-keyword scanning for
    fragments that remain unparseable after dedent.
    """
    if not lines:
        return []
    src = textwrap.dedent("\n".join(lines))
    try:
        tree = ast.parse(src)
        return [
            type(n).__name__
            for n in ast.walk(tree)
            if type(n).__name__ not in _SKIP_NODES
        ]
    except SyntaxError:
        tokens: list[str] = []
        for line in lines:
            stripped = line.strip()
            for kw in _FALLBACK_KEYWORDS:
                if stripped.startswith(kw) or stripped == kw.rstrip():
                    tokens.append(kw.strip().rstrip(":"))
                    break
        return tokens


def patch_to_chunks(patch: str) -> list[EditChunk]:
    """
    Parse a unified diff into a list of EditChunk objects, one per @@ hunk.

    Preserves chunk boundaries so each chunk can be described independently.
    Filters out non-Python files (no .py in the diff --git line).
    """
    chunks: list[EditChunk] = []
    current_file = ""
    current_header = ""
    removed: list[str] = []
    added: list[str] = []
    context: list[str] = []

    def _flush() -> None:
        if not current_header:
            return
        chunk = EditChunk(
            header=current_header,
            file_path=current_file,
            removed_lines=removed[:],
            added_lines=added[:],
            context_lines=context[:6],  # up to 6 context lines for LLM
        )
        chunk.removed_seq = [f"DEL_{n}" for n in _lines_to_ast_nodes(removed)]
        chunk.added_seq = [f"ADD_{n}" for n in _lines_to_ast_nodes(added)]
        if not chunk.is_empty:
            chunks.append(chunk)

    for line in patch.splitlines():
        # New file in the diff
        if line.startswith("diff --git"):
            _flush()
            current_header = ""
            removed, added, context = [], [], []
            m = re.search(r'b/(.+\.py)$', line)
            current_file = m.group(1) if m else ""
            continue

        # Skip file header lines
        if line.startswith("---") or line.startswith("+++"):
            continue

        # New hunk header
        if line.startswith("@@"):
            _flush()
            removed, added, context = [], [], []
            current_header = _HUNK_HEADER.match(line).group(0) if _HUNK_HEADER.match(line) else line[:40]
            continue

        if not current_file:  # non-Python file
            continue

        if line.startswith("+"):
            added.append(line[1:])
        elif line.startswith("-"):
            removed.append(line[1:])
        else:
            context.append(line[1:] if line.startswith(" ") else line)

    _flush()
    return chunks


def patch_to_ast_sequence(patch: str) -> list[str]:
    """
    Flat sequence for the whole patch (backwards-compatible).
    Concatenates chunk sequences in hunk order.
    """
    return [tok for chunk in patch_to_chunks(patch) for tok in chunk.sequence]


def corpus_ast_sequences(patches: dict[str, str]) -> dict[str, list[str]]:
    """Extract flat AST sequences for a corpus of patches."""
    return {iid: patch_to_ast_sequence(p) for iid, p in patches.items()}


def vocabulary_stats(sequences: list[list[str]]) -> list[tuple[str, int]]:
    """(token, count) sorted by frequency across all sequences."""
    counter: Counter = Counter()
    for seq in sequences:
        counter.update(seq)
    return counter.most_common()
