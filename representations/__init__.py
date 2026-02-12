"""
Representation Engineering Framework

Provides six encoders spanning a privacy-expressiveness frontier:
- raw: Complete event logs with PII redaction
- tokens: Canonicalized token sequences  
- functions: Function-level changes
- edits: AST-based edit operations
- modules: File dependency graphs
- motifs: Abstract workflow patterns
"""

from .encoders.raw import raw_repr, raw_repr_str
from .encoders.tokens import tokens_repr, tokens_repr_str
from .encoders.edits import semantic_edits_repr, semantic_edits_repr_str
from .encoders.functions import functions_repr, functions_repr_str
from .encoders.modules import module_graph_repr, file_edit_graph_repr, file_edit_graph_repr_str
from .encoders.motifs import motifs_repr, motifs_repr_str

# Maintain backward compatibility aliases if needed
functions_repr_str = functions_repr_str
semantic_edits_repr = semantic_edits_repr

__all__ = [
    "raw_repr", "raw_repr_str",
    "tokens_repr", "tokens_repr_str",
    "semantic_edits_repr", "semantic_edits_repr_str",
    "functions_repr", "functions_repr_str",
    "module_graph_repr", "file_edit_graph_repr", "file_edit_graph_repr_str",
    "motifs_repr", "motifs_repr_str",
]

__version__ = "1.1.0"
