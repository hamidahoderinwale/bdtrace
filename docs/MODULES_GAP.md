# Modules Representation: Design vs Implementation

## Intended design

**Modules** is the graph-tier representation above structural (edits). It combines:

- **Import graph** — Python imports between files (dependency structure)
- **Co-edit graph** — files modified together (from git history or trace events)

Distance: graph edit distance (symmetric edge diff via networkx).

## Implementation gap

| Context | What runs | Result |
|--------|-----------|--------|
| Repo on disk | `module_graph_repr(repo_path, commit, touched_files)` | Full graph: import + co-edit |
| Trace only (no repo) | `file_edit_graph_repr(trace)` | Co-edit only, from `events` with `file_path` |
| Single-file trace | Co-edit requires ≥2 different files | Empty (no cross-file edges) |

The extraction pipeline uses `file_edit_graph_repr` (trace fallback). When traces lack `repo_path`, the import graph is never built. For single-file instances (e.g. SWE-bench Lite), co-edit is empty by definition.

## When modules is informative

- Multi-file edits (agent trajectories, multi-file patches)
- Repo available at extraction (diff_resolution clones repo → can use `module_graph_repr`)
- Single-file with repo: import graph still has edges (touched file → its imports)

## Mitigations

1. **Use full `module_graph_repr` when repo is available** — diff_resolution computes modules from the cloned repo and attaches to trace. Requires `pydriller` and `networkx` (see pyproject.toml).
2. **Annotate or skip modules in plots** when degenerate (all instances have modules_count=0).
3. **Future**: Intra-file graph fallback from AST (function/call graph) when single-file and no repo.
