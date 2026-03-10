"""Import extraction from Python source."""

import ast
from pathlib import Path


def _resolve_module_to_path(
    module: str,
    repo_root: Path,
    current_file: Path,
) -> Path | None:
    if not module:
        return None
    parts = module.split(".")
    if parts[0].startswith("."):
        base = current_file.parent
        parts[0] = parts[0].lstrip(".")
    else:
        base = repo_root
    for part in parts[:-1]:
        base = base / part
    name = parts[-1] if parts else ""
    for c in [base / f"{name}.py", base / name / "__init__.py"]:
        if c.exists():
            return c
    return None


def extract_imports_from_file(
    file_path: Path,
    repo_root: Path | None = None,
) -> list[tuple[str, str]]:
    """Extract import edges from a Python file."""
    repo_root = repo_root or file_path.parent
    try:
        tree = ast.parse(file_path.read_text())
    except (SyntaxError, OSError):
        return []

    edges = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    target = _resolve_module_to_path(alias.name, repo_root, file_path)
                    edges.append((str(file_path), str(target) if target else alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if not module and node.level:
                continue
            if node.level and node.level > 0:
                parent = file_path.parent
                for _ in range(node.level - 1):
                    parent = parent.parent
                target = _resolve_module_to_path(module, parent, file_path) if module else None
                fallback = str(parent / module.replace(".", "/")) if module else str(parent)
                edges.append((str(file_path), str(target) if target else fallback))
            elif module:
                target = _resolve_module_to_path(module, repo_root, file_path)
                edges.append((str(file_path), str(target) if target else module))
    return edges


def _normalize_path(p: Path | str, repo_root: Path) -> str:
    p = Path(p) if isinstance(p, str) else p
    try:
        return str(p.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(p)


def neighbors_of(
    touched_files: list[str],
    repo_root: Path,
    all_py_files: list[Path] | None = None,
) -> set[Path]:
    """Expand touched_files to include import neighborhood."""
    repo_root = Path(repo_root).resolve()
    _skip = {".venv", "venv", "node_modules", "__pycache__", ".git"}
    if all_py_files is None:
        all_py_files = [p for p in repo_root.rglob("*.py") if not any(s in p.parts for s in _skip)]

    touched_norm = {_normalize_path(f, repo_root) for f in touched_files}
    path_by_norm = {_normalize_path(p, repo_root): p for p in all_py_files}

    import_edges = []
    for fp in all_py_files:
        import_edges.extend(extract_imports_from_file(fp, repo_root))

    result = {path_by_norm[t] for t in touched_norm if t in path_by_norm}

    def tgt_to_norm(tgt: str) -> str | None:
        p = Path(tgt)
        if p.is_absolute() and p.exists():
            n = _normalize_path(p, repo_root)
            return n if n in path_by_norm else None
        cand = repo_root / tgt
        if cand.exists():
            n = _normalize_path(cand, repo_root)
            return n if n in path_by_norm else None
        for norm in path_by_norm:
            if norm.endswith(f"/{tgt}") or norm == f"{tgt}.py":
                return norm
        return None

    for src, tgt in import_edges:
        src_norm = _normalize_path(src, repo_root)
        tgt_norm = tgt_to_norm(tgt) if isinstance(tgt, str) else None
        if src_norm in touched_norm and tgt_norm and tgt_norm in path_by_norm:
            result.add(path_by_norm[tgt_norm])
        if tgt_norm in touched_norm and src_norm in path_by_norm:
            result.add(path_by_norm[src_norm])
    return result
