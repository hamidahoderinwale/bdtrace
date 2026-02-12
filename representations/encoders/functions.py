import json
from ..core.utils import extract_function_names_from_code

def functions_repr(trace: dict, include_prompts: bool = True) -> list[str]:
    """Extract function-level representation."""
    if not trace or not isinstance(trace, dict):
        return []
    
    funcs = []
    events = trace.get('events', [])
    if not events:
        return []
    
    for event in events:
        if not isinstance(event, dict):
            continue
            
        try:
            details = event.get('details', {})
            if isinstance(details, str):
                try:
                    details = json.loads(details)
                except (json.JSONDecodeError, TypeError):
                    details = {}
            
            if not isinstance(details, dict):
                details = {}
            
            code = details.get('after_content') or details.get('before_content') or details.get('code', '')
            file_path = details.get('file_path') or details.get('file', '')
            
            if code and isinstance(code, str):
                try:
                    func_names = extract_function_names_from_code(code, file_path)
                    if func_names:
                        funcs.extend(func_names)
                except Exception:
                    pass
            
            op = event.get('operation') or event.get('type')
            if op and isinstance(op, str) and 'function' in op.lower():
                funcs.append(op)
        except Exception:
            continue
    
    return funcs

def functions_repr_str(trace: Dict, limit: int = 50) -> str:
    """Extract functions as a string representation."""
    functions = functions_repr(trace, include_prompts=True)
    if not functions:
        return "EMPTY_TRACE"
    func_str = " → ".join(functions[:limit])
    if len(functions) > limit:
        func_str += f" ... [truncated from {len(functions)} functions]"
    return func_str

