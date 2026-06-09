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


def _try_parse(src: str) -> list[str] | None:
    """Return AST node names if src parses cleanly, else None."""
    try:
        return [
            type(n).__name__
            for n in ast.walk(ast.parse(src))
            if type(n).__name__ not in _SKIP_NODES
        ]
    except SyntaxError:
        return None


def _lines_to_ast_nodes(lines: list[str]) -> list[str]:
    """
    Parse a list of code lines and return AST node type names in walk order.

    Uses a tiered strategy before falling back to keyword scanning:
      1. Direct parse after dedent.
      2. Append a dummy body when the fragment ends with ':' (handles if/for/
         while/def/class/with headers that appear without their body in the hunk).
      3. Prepend a dummy predecessor for elif/else (needs a preceding if-block).
      4. Prepend a dummy try-block for except/finally.
      5. Wrap in a function body for return/yield/raise/del at module scope.
      6. Keyword scan as last resort.
    """
    if not lines:
        return []

    src = textwrap.dedent("\n".join(lines))
    stripped = src.strip()

    # Tier 1: direct
    result = _try_parse(src)
    if result is not None:
        return result

    # Tier 2: compound-statement header missing its body
    if stripped.endswith(":"):
        result = _try_parse(stripped + "\n    pass")
        if result is not None:
            return result

    # Tier 3: elif / else — needs a preceding if-block (and possibly a body)
    if stripped.startswith(("elif ", "else:")):
        candidate = "if True:\n    pass\n" + stripped
        if stripped.endswith(":"):
            candidate += "\n    pass"
        result = _try_parse(candidate)
        if result is not None:
            return result

    # Tier 4: except / finally — needs a preceding try-block
    if stripped.startswith(("except", "finally")):
        candidate = "try:\n    pass\n" + stripped
        if stripped.endswith(":"):
            candidate += "\n    pass"
        result = _try_parse(candidate)
        if result is not None:
            return result

    # Tier 5: return / yield / raise / del outside a function scope
    if any(stripped.startswith(kw) for kw in ("return", "yield", "raise", "del ")):
        indented = "\n".join(f"    {l}" for l in textwrap.dedent("\n".join(lines)).splitlines())
        result = _try_parse(f"def _f():\n{indented}")
        if result is not None:
            # Drop the synthetic FunctionDef scaffolding from the result
            return [n for n in result if n not in {"FunctionDef", "arguments", "arg"}]

    # Tier 6: keyword scan (last resort)
    tokens: list[str] = []
    for line in lines:
        s = line.strip()
        for kw in _FALLBACK_KEYWORDS:
            if s.startswith(kw) or s == kw.rstrip():
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


_FALLBACK_KW_STEMS = frozenset({
    'elif', 'else', 'except', 'assert', 'lambda', 'yield', 'return',
    'raise', 'import', 'while', 'class', 'with', 'async', 'for',
    'def', 'try', 'if',
})


def _is_keyword_fallback(token: str) -> bool:
    """True when token was produced by keyword scanning, not AST parsing."""
    stem = token[4:]  # strip ADD_ / DEL_
    return stem.lower() in _FALLBACK_KW_STEMS and stem[0].islower()


def _fullfile_cert(before: str, after: str) -> list[str]:
    """
    Full-file Counter diff: parse complete before/after source, return
    ADD_X / DEL_X for node types whose count changed.  Zero fallback.
    """
    from collections import Counter as _Counter

    def _count(src: str) -> _Counter:
        try:
            return _Counter(
                type(n).__name__
                for n in ast.walk(ast.parse(src))
                if type(n).__name__ not in _SKIP_NODES
            )
        except SyntaxError:
            return _Counter()

    b, a = _count(before or ''), _count(after or '')
    tokens = (
        [f'ADD_{t}' for t in sorted(a - b)]
        + [f'DEL_{t}' for t in sorted(b - a)]
    )
    return tokens


def patch_to_ast_sequence(
    patch: str,
    before_content: str = '',
    after_content: str = '',
) -> list[str]:
    """
    Flat AST token sequence for a patch.

    Primary path: tiered fragment parser (hunk-local, captures replacements).
    Fallback path: full-file Counter diff (zero fallback tokens) used when
    the fragment parser produces only keyword-scanned tokens for every hunk.

    Pass ``before_content`` and ``after_content`` (full file source) to
    enable the full-file fallback; without them the function behaves as
    before (fragment-only).
    """
    tokens = [tok for chunk in patch_to_chunks(patch) for tok in chunk.sequence]

    # If the fragment parser produced anything beyond keyword tokens, use it.
    if any(not _is_keyword_fallback(t) for t in tokens):
        return tokens

    # All tokens are keyword fallbacks (or empty): try full-file diff.
    if before_content or after_content:
        ff = _fullfile_cert(before_content, after_content)
        if ff:
            return ff

    return tokens  # last resort: return whatever the fragment parser found


def corpus_ast_sequences(patches: dict[str, str]) -> dict[str, list[str]]:
    """Extract flat AST sequences for a corpus of patches."""
    return {iid: patch_to_ast_sequence(p) for iid, p in patches.items()}


def vocabulary_stats(sequences: list[list[str]]) -> list[tuple[str, int]]:
    """(token, count) sorted by frequency across all sequences."""
    counter: Counter = Counter()
    for seq in sequences:
        counter.update(seq)
    return counter.most_common()
