import json
from pathlib import Path
from datetime import datetime
from ..core.utils import extract_function_names_from_code

def module_graph_repr(trace: dict) -> list[str]:
    """Extract module/file-level representation (DEPRECATED)."""
    modules = []
    seen_files = set()
    for event in trace.get('events', []):
        details = event.get('details', {})
        if isinstance(details, str):
            try:
                details = json.loads(details)
            except:
                details = {}
        
        file_path = details.get('file_path') or details.get('file')
        if file_path and file_path not in seen_files:
            module_name = Path(file_path).stem
            if module_name:
                modules.append(module_name)
                seen_files.add(file_path)
        
        op = event.get('operation') or event.get('type')
        if op and any(k in str(op).lower() for k in ['import', 'export', 'module', 'file']):
            modules.append(str(op))
    return modules

def _extract_file_edits(trace: dict) -> list[dict]:
    edits = []
    for event in trace.get('events', []):
        details = event.get('details', {})
        if isinstance(details, str):
            try: details = json.loads(details)
            except: details = {}
        
        file_path = details.get('file_path') or details.get('file')
        timestamp = event.get('timestamp', 0)
        
        if isinstance(timestamp, str):
            try: timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).timestamp()
            except: timestamp = 0
        
        if file_path:
            edits.append({'time': timestamp or 0, 'path': file_path})
    return edits

def file_edit_graph_repr(trace: dict, time_window_sec: int = 300, canonicalize: bool = False) -> list[str]:
    """Extract file-level edit graph representation."""
    edits = _extract_file_edits(trace)
    if len(edits) < 2: return []
    
    all_files = list(set(e['path'] for e in edits))
    if canonicalize:
        name_map = {f: f"F{i:03d}" for i, f in enumerate(sorted(all_files))}
    else:
        name_map = {f: Path(f).stem for f in all_files}
    
    edges = set()
    for i, e1 in enumerate(edits):
        for e2 in edits[i+1:]:
            if e1['path'] == e2['path']: continue
            if 0 <= e2['time'] - e1['time'] <= time_window_sec:
                edges.add((name_map[e1['path']], name_map[e2['path']]))
                break
    
    repr_tokens = [f"E_{s}_{d}" for s, d in sorted(edges)]
    
    edit_counts = {}
    for e in edits:
        f = name_map[e['path']]
        edit_counts[f] = edit_counts.get(f, 0) + 1
    for f, count in sorted(edit_counts.items()):
        repr_tokens.append(f"EDITS_{f}_{count}")
        
    repr_tokens.append(f"NODES_{len(all_files)}")
    repr_tokens.append(f"EDGES_{len(edges)}")
    return repr_tokens

def file_edit_graph_repr_str(trace: dict, time_window_sec: int = 300, canonicalize: bool = False, limit: int = 100) -> str:
    tokens = file_edit_graph_repr(trace, time_window_sec, canonicalize)
    if not tokens: return "EMPTY_GRAPH"
    res = " ".join(tokens[:limit])
    if len(tokens) > limit: res += f" ... [truncated]"
    return res

