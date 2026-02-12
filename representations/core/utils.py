"""
Utility Functions

Helper functions for AST parsing, tokenization, function extraction,
import extraction, file operations, and PII redaction.
"""

import ast
import json
import re
from collections import defaultdict
from pathlib import Path


# Function extraction patterns
JS_FUNCTION_PATTERN = re.compile(r"\bfunction\s+([A-Za-z_][\w]*)\s*\(")
ARROW_FUNCTION_PATTERN = re.compile(r"\b([A-Za-z_][\w]*)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>")
EXPORT_FUNCTION_PATTERN = re.compile(r"\b(?:exports|module\.exports)\.([A-Za-z_][\w]*)\s*=")
CLASS_STATEMENT_PATTERN = re.compile(r"\bclass\s+([A-Za-z_][\w]*)\b")


def _python_function_names(code: str) -> list[str]:
    """Extract Python function/class names from code."""
    try:
        tree = ast.parse(code)
        return [node.name for node in ast.walk(tree) 
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    except SyntaxError:
        return []


def extract_function_names_from_code(code: str, filename: str | None = None) -> list[str]:
    """Extract function/class names from code snippet."""
    names: List[str] = []
    if not code:
        return names
    if filename and filename.endswith(".py"):
        names.extend(_python_function_names(code))
    if not names:
        names.extend(JS_FUNCTION_PATTERN.findall(code))
        names.extend(ARROW_FUNCTION_PATTERN.findall(code))
        names.extend(EXPORT_FUNCTION_PATTERN.findall(code))
        names.extend(CLASS_STATEMENT_PATTERN.findall(code))
    return list(dict.fromkeys(names))


def _extract_code_tokens(code: str, file_path: str | None = None) -> list[str]:
    """Extract token types from code content using language-aware AST parsing.
    
    This function extracts token types (IDENTIFIER, KEYWORD, OPERATOR, etc.)
    rather than exact values, preserving code structure while maintaining privacy.
    
    Uses AST parsers for all supported languages, falling back to generic tokenization
    if AST parsing fails or is unavailable.
    
    Args:
        code: Code content to tokenize
        file_path: Optional file path for language detection
    
    Returns:
        List of token type strings (e.g., ['CONST', 'IDENTIFIER', 'ASSIGN', 'STRING_LITERAL'])
    """
    if not code:
        return []
    
    language = None
    
    # Detect language from file extension if available
    if file_path:
        ext = Path(file_path).suffix.lower()
        language_map = {
            '.py': 'python',
            '.js': 'javascript', '.jsx': 'javascript',
            '.ts': 'typescript', '.tsx': 'typescript',
            '.java': 'java', '.cpp': 'cpp', '.c': 'c',
            '.go': 'go', '.rs': 'rust', '.rb': 'ruby',
            '.php': 'php', '.swift': 'swift', '.kt': 'kotlin',
        }
        language = language_map.get(ext, 'unknown')
    
    # Try AST-based tokenization for supported languages
    # AST parsing preserves structure better than regex-based approaches
    if language == 'python':
        try:
            return _tokenize_python_ast(code)
        except SyntaxError:
            pass  # Fall back to generic tokenization
    
    elif language in ('javascript', 'typescript'):
        try:
            return _tokenize_js_ast(code, language)
        except Exception:
            pass  # Fall back to generic tokenization
    
    elif language == 'java':
        try:
            return _tokenize_java_ast(code)
        except Exception:
            pass  # Fall back to generic tokenization
    
    # Generic tokenization: extract token types, not values
    # This preserves structure while maintaining privacy and works across all languages
    return _tokenize_generic(code)


def _tokenize_python_ast(code: str) -> list[str]:
    """Extract token types from Python code using AST."""
    try:
        tree = ast.parse(code)
        tokens = []
        
        for node in ast.walk(tree):
            # Map AST node types to token types
            node_type = type(node).__name__
            
            if isinstance(node, ast.FunctionDef):
                tokens.append('FUNCTION')
                tokens.append('IDENTIFIER')  # Function name (canonicalized)
            elif isinstance(node, ast.ClassDef):
                tokens.append('CLASS')
                tokens.append('IDENTIFIER')  # Class name (canonicalized)
            elif isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    tokens.append('IDENTIFIER')
                elif isinstance(node.ctx, ast.Load):
                    tokens.append('IDENTIFIER')
            elif isinstance(node, ast.Constant):
                if isinstance(node.value, str):
                    tokens.append('STRING_LITERAL')
                elif isinstance(node.value, (int, float)):
                    tokens.append('NUMBER')
                elif isinstance(node.value, bool):
                    tokens.append('BOOLEAN')
            elif isinstance(node, ast.Call):
                tokens.append('CALL')
            elif isinstance(node, ast.Attribute):
                tokens.append('ATTRIBUTE')
            elif isinstance(node, ast.Assign):
                tokens.append('ASSIGN')
            elif isinstance(node, ast.If):
                tokens.append('IF')
            elif isinstance(node, ast.For):
                tokens.append('FOR')
            elif isinstance(node, ast.While):
                tokens.append('WHILE')
            elif isinstance(node, ast.Return):
                tokens.append('RETURN')
            elif isinstance(node, ast.Import):
                tokens.append('IMPORT')
            elif isinstance(node, ast.ImportFrom):
                tokens.append('IMPORT')
                tokens.append('FROM')
        
        return tokens
    except SyntaxError:
        return _tokenize_generic(code)


def _tokenize_js_ast(code: str, language: str = 'javascript') -> list[str]:
    """Extract token types from JavaScript/TypeScript code using AST parser.
    
    Uses esprima or similar parser library if available, falls back to generic.
    """
    try:
        # Try to import esprima (common JS parser library)
        # If not available, will fall back to generic tokenization
        import esprima
        
        tokens = []
        tree = esprima.parseScript(code, {'tokens': True, 'tolerant': True})
        
        # Extract token types from parsed tokens
        for token in tree.tokens:
            token_type = token.type
            
            # Map esprima token types to our canonical types
            if token_type in ('Identifier', 'Keyword'):
                if token.value in ('function', 'class', 'const', 'let', 'var', 'async', 'await'):
                    tokens.append(token.value.upper())
                else:
                    tokens.append('IDENTIFIER')
            elif token_type == 'String':
                tokens.append('STRING_LITERAL')
            elif token_type == 'Numeric':
                tokens.append('NUMBER')
            elif token_type == 'Boolean':
                tokens.append('BOOLEAN')
            elif token_type == 'Punctuator':
                # Map operators
                if token.value == '=':
                    tokens.append('ASSIGN')
                elif token.value in ('+', '-', '*', '/', '%'):
                    tokens.append('OPERATOR')
                elif token.value in ('(', ')'):
                    tokens.append('PAREN')
                elif token.value in ('[', ']'):
                    tokens.append('BRACKET')
                elif token.value in ('{', '}'):
                    tokens.append('BRACE')
                else:
                    tokens.append('PUNCTUATOR')
            elif token_type == 'Keyword':
                tokens.append(token.value.upper())
        
        return tokens
    except ImportError:
        # esprima not available, fall back to generic
        return _tokenize_generic(code)
    except Exception:
        # Parsing failed, fall back to generic
        return _tokenize_generic(code)


def _tokenize_java_ast(code: str) -> list[str]:
    """Extract token types from Java code using AST parser.
    
    Uses javalang or similar parser library if available, falls back to generic.
    """
    try:
        # Try to import javalang (common Java parser library)
        import javalang
        
        tokens = []
        tree = javalang.parse.parse(code)
        
        # Walk AST and extract token types
        for path, node in tree:
            node_type = type(node).__name__
            
            if node_type == 'MethodDeclaration':
                tokens.append('METHOD')
                tokens.append('IDENTIFIER')
            elif node_type == 'ClassDeclaration':
                tokens.append('CLASS')
                tokens.append('IDENTIFIER')
            elif node_type == 'VariableDeclarator':
                tokens.append('VARIABLE')
                tokens.append('IDENTIFIER')
            elif node_type == 'MethodInvocation':
                tokens.append('CALL')
            elif node_type == 'Assignment':
                tokens.append('ASSIGN')
            elif node_type == 'IfStatement':
                tokens.append('IF')
            elif node_type == 'ForStatement':
                tokens.append('FOR')
            elif node_type == 'WhileStatement':
                tokens.append('WHILE')
            elif node_type == 'ReturnStatement':
                tokens.append('RETURN')
            elif node_type == 'Import':
                tokens.append('IMPORT')
            elif node_type == 'StringLiteral':
                tokens.append('STRING_LITERAL')
            elif node_type == 'IntegerLiteral' or node_type == 'FloatingPointLiteral':
                tokens.append('NUMBER')
        
        return tokens
    except ImportError:
        # javalang not available, fall back to generic
        return _tokenize_generic(code)
    except Exception:
        # Parsing failed, fall back to generic
        return _tokenize_generic(code)


def _tokenize_generic(code: str) -> list[str]:
    """Generic tokenization that extracts token types without language-specific parsing.
    
    This is a fallback that works across languages by recognizing common patterns.
    """
    tokens = []
    lines = code.split('\n')
    
    # Common keywords across languages
    keywords = {
        'function', 'def', 'class', 'const', 'let', 'var', 'if', 'else',
        'for', 'while', 'return', 'import', 'export', 'from', 'async', 'await',
        'try', 'catch', 'throw', 'new', 'this', 'super', 'extends', 'implements'
    }
    
    # Common operators
    operators = {
        '=', '==', '===', '!=', '!==', '<', '>', '<=', '>=',
        '+', '-', '*', '/', '%', '&&', '||', '!', '++', '--',
        '+=', '-=', '*=', '/=', '?', '??', '?.', '=>'
    }
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('//') or line.startswith('#'):
            continue
        
        # Simple pattern matching
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', line)
        for word in words:
            if word.lower() in keywords:
                tokens.append(word.upper())
            else:
                tokens.append('IDENTIFIER')
        
        # Operators and punctuation
        ops = re.findall(r'[{}()[\].,;:+\-*/=<>!&|?]+', line)
        for op in ops:
            if op in operators:
                tokens.append(op.upper().replace('=', 'ASSIGN').replace('+', 'PLUS'))
            elif op in '()':
                tokens.append('PAREN')
            elif op in '[]':
                tokens.append('BRACKET')
            elif op in '{}':
                tokens.append('BRACE')
            else:
                tokens.append('OPERATOR')
        
        # String literals
        if '"' in line or "'" in line or '`' in line:
            tokens.append('STRING_LITERAL')
        
        # Numbers
        if re.search(r'\d', line):
            tokens.append('NUMBER')
    
    return tokens


def count_ops(code: str) -> int:
    """Count operations in Python code (for complexity estimation)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0
    return sum(
        isinstance(node, (ast.If, ast.For, ast.While, ast.FunctionDef, ast.AsyncFunctionDef, ast.Assign))
        for node in ast.walk(tree)
    )


def extract_imports_from_code(file_path: str, code: str) -> list[str]:
    """Extract import statements from code."""
    if not code:
        return []

    file_ext = file_path.split(".")[-1].lower() if "." in file_path else ""
    patterns = {
        "js": [
            r"import\s+.*?\s+from\s+['\"](.+?)['\"]",
            r"require\(['\"](.+?)['\"]\)",
            r"import\(['\"](.+?)['\"]\)",
        ],
        "ts": [
            r"import\s+.*?\s+from\s+['\"](.+?)['\"]",
            r"import\s+['\"](.+?)['\"]",
        ],
        "tsx": [
            r"import\s+.*?\s+from\s+['\"](.+?)['\"]",
            r"import\s+['\"](.+?)['\"]",
        ],
        "jsx": [
            r"import\s+.*?\s+from\s+['\"](.+?)['\"]",
            r"require\(['\"](.+?)['\"]\)",
        ],
        "py": [
            r"^import\s+(\S+)",
            r"^from\s+(\S+)\s+import",
        ],
    }

    lang_patterns = patterns.get(file_ext, patterns.get("js", []))
    imports: List[str] = []
    for pattern in lang_patterns:
        matches = re.findall(pattern, code, re.MULTILINE)
        imports.extend(matches)

    unique_imports = []
    seen = set()
    for imp in imports:
        if imp.startswith(".") or imp.startswith("/") or "/" in imp:
            if imp not in seen:
                seen.add(imp)
                unique_imports.append(imp)
    return unique_imports


def files_repr(trace: dict) -> str:
    """Extract file-level representation with action counts."""
    file_actions: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {
            "edits": 0,
            "navigations": 0,
            "ai_context": 0,
            "prompts": 0,
            "terminal_refs": 0,
        }
    )

    for event in trace.get("events", []):
        details = event.get("details") or {}
        event_type = event.get("type", "").lower()

        file_path = (
            details.get("file_path")
            or details.get("file")
            or details.get("target")
            or details.get("uri", {}).get("fsPath")
            or details.get("uri", {}).get("path")
        )

        if not file_path and isinstance(details, dict):
            for key in ["file_path", "file", "target", "path"]:
                if key in details:
                    file_path = details[key]
                    break

        if not file_path:
            continue

        file_path = str(file_path)
        if event_type in ("file_change", "code_change", "entry_created"):
            file_actions[file_path]["edits"] += 1
        elif event_type in ("ide_state", "navigate"):
            file_actions[file_path]["navigations"] += 1
        elif event_type in ("prompt", "model_context"):
            file_actions[file_path]["ai_context"] += 1
            file_actions[file_path]["prompts"] += 1
        elif event_type in ("terminal_command", "tool_interaction"):
            command = details.get("command") or details.get("text") or ""
            if file_path in command or any(part in command for part in file_path.split("/")):
                file_actions[file_path]["terminal_refs"] += 1

    file_reprs: List[str] = []
    for file_path, counts in sorted(file_actions.items()):
        total_actions = sum(counts.values())
        if total_actions > 0:
            file_reprs.append(
                f"{file_path}:"
                f"e{counts['edits']}:"
                f"n{counts['navigations']}:"
                f"a{counts['ai_context']}:"
                f"p{counts['prompts']}:"
                f"t{counts['terminal_refs']}"
            )
    return " | ".join(file_reprs) or trace.get("workspace_path", "unknown")


def dependencies_repr(trace: dict) -> str:
    """Extract dependency relationships from code imports."""
    dependencies: List[str] = []
    seen_deps = set()

    for event in trace.get("events", []):
        details = event.get("details") or {}
        event_type = event.get("type", "").lower()
        if event_type not in ("file_change", "code_change", "entry_created"):
            continue

        file_path = (
            details.get("file_path")
            or details.get("file")
            or details.get("target")
        )
        if not file_path:
            continue

        file_path = str(file_path)
        code_content = details.get("after_content") or details.get("before_content") or ""
        if not code_content:
            continue

        imports = extract_imports_from_code(file_path, code_content)
        for imported_path in imports:
            dep_key = f"{file_path}→{imported_path}"
            if dep_key not in seen_deps:
                seen_deps.add(dep_key)
                dependencies.append(dep_key)

    if not dependencies:
        return files_repr(trace)[:512]
    return " | ".join(dependencies)


def get_file_action_stats(traces: list[dict]) -> dict[str, dict[str, int]]:
    """Get file action statistics across multiple traces."""
    stats: Dict[str, Dict[str, int | set]] = defaultdict(
        lambda: {
            "total_edits": 0,
            "total_navigations": 0,
            "total_ai_context": 0,
            "total_prompts": 0,
            "total_terminal_refs": 0,
            "sessions": set(),
            "operation_count": 0,
        }
    )

    def extract_file_paths_from_command(command: str) -> List[str]:
        paths: List[str] = []
        patterns = [
            r'["\']([^"\']+\.(py|js|ts|tsx|jsx|json|md|txt|yaml|yml|sh|bash|zsh))["\']',
            r'([a-zA-Z0-9_\-/]+\\.(py|js|ts|tsx|jsx|json|md|txt|yaml|yml|sh|bash|zsh))',
            r'\./([a-zA-Z0-9_\-/]+\\.(py|js|ts|tsx|jsx|json|md|txt|yaml|yml|sh|bash|zsh))',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, command)
            for match in matches:
                if isinstance(match, tuple):
                    paths.append(match[0] if match[0] else match[1])
                else:
                    paths.append(match)
        return list(dict.fromkeys(paths))

    for trace in traces:
        session_id = trace.get("session_id", "unknown")

        for event in trace.get("events", []):
            details = event.get("details") or {}
            event_type = event.get("type", "").lower()

            if event_type == "prompt":
                context_files = details.get("context_files", [])
                if isinstance(context_files, str):
                    try:
                        context_files = json.loads(context_files)
                    except (json.JSONDecodeError, TypeError):
                        context_files = []
                if isinstance(context_files, list):
                    for file_info in context_files:
                        if isinstance(file_info, dict):
                            file_path = (
                                file_info.get("path")
                                or file_info.get("file_path")
                                or file_info.get("file")
                            )
                        elif isinstance(file_info, str):
                            file_path = file_info
                        else:
                            continue
                        if file_path:
                            fp = str(file_path)
                            stats[fp]["sessions"].add(session_id)
                            stats[fp]["total_prompts"] += 1
                            stats[fp]["total_ai_context"] += 1
            elif event_type == "terminal_command":
                command = details.get("command") or ""
                if command:
                    for file_path in extract_file_paths_from_command(command):
                        stats[file_path]["sessions"].add(session_id)
                        stats[file_path]["total_terminal_refs"] += 1
            else:
                file_path = (
                    details.get("file_path")
                    or details.get("file")
                    or details.get("target")
                    or details.get("uri", {}).get("fsPath")
                    or details.get("uri", {}).get("path")
                )
                if file_path:
                    fp = str(file_path)
                    stats[fp]["sessions"].add(session_id)

                    after_content = details.get("after_content") or ""
                    before_content = details.get("before_content") or ""
                    code_snippet = after_content if len(after_content) > len(before_content) else before_content
                    if not code_snippet:
                        code_snippet = details.get("code") or details.get("content") or ""
                    if code_snippet:
                        stats[fp]["operation_count"] += count_ops(code_snippet)

                    if event_type in ("file_change", "code_change", "entry_created"):
                        stats[fp]["total_edits"] += 1
                    elif event_type in ("ide_state", "navigate"):
                        stats[fp]["total_navigations"] += 1

    result: Dict[str, Dict[str, int]] = {}
    for file_path, value in stats.items():
        result[file_path] = {
            **{k: v for k, v in value.items() if k != "sessions"},
            "unique_sessions": len(value["sessions"]),
        }
    return result


def redact_pii(text: str, redact_all_strings: bool = True) -> str:
    """Redact PII from text content.
    
    Args:
        text: Text content to redact
        redact_all_strings: If True, redact all string literals. If False, only redact detected PII.
    
    Returns:
        Text with PII redacted
    """
    if not text:
        return text
    
    # Email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    
    # Names (capitalized words, likely names)
    text = re.sub(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', '[NAME]', text)
    
    # URLs
    text = re.sub(r'https?://[^\s<>"{}|\\^`\[\]]+', '[URL]', text)
    
    # IP addresses
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[IP]', text)
    
    # JWT tokens
    text = re.sub(r'\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]{20,}\b', '[JWT_TOKEN]', text)
    
    # High entropy strings (potential secrets)
    text = re.sub(r'\b[A-Za-z0-9+/]{32,}={0,2}\b', '[SECRET]', text)
    
    # Phone numbers
    text = re.sub(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b', '[PHONE]', text)
    
    # SSN
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
    
    # Credit card numbers
    text = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CREDIT_CARD]', text)
    
    return text


def redact_code_pii(code: str, redact_all_strings: bool = True) -> str:
    """Redact PII from code content, preserving code structure.
    
    Args:
        code: Code content to redact
        redact_all_strings: If True, replace all string literals with [STR]. If False, only redact PII-containing strings.
    
    Returns:
        Code with PII redacted
    """
    if not code:
        return code
    
    # First, redact PII patterns
    code = redact_pii(code, redact_all_strings)
    
    # Optionally redact all string literals (for higher privacy)
    if redact_all_strings:
        # Replace string literals (single, double, template literals)
        code = re.sub(r'"[^"]*"', '"[STR]"', code)
        code = re.sub(r"'[^']*'", "'[STR]'", code)
        code = re.sub(r'`[^`]*`', '`[STR]`', code)
    
    return code

