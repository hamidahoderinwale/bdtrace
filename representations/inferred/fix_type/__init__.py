"""Fix-type labeling module (DSPy-based)."""

from .chunk_describer import ChunkDescriber, ChunkDescribeSignature
from .fix_type import (
    FIX_TYPES,
    FIX_TYPE_DESCRIPTIONS,
    FixTypeModule,
    FixTypeSignature,
    extract_ast_stage,
    format_stage_for_llm,
)

__all__ = [
    "ChunkDescriber",
    "ChunkDescribeSignature",
    "FIX_TYPES",
    "FIX_TYPE_DESCRIPTIONS",
    "FixTypeModule",
    "FixTypeSignature",
    "extract_ast_stage",
    "format_stage_for_llm",
]
