import json
from ..core.utils import _extract_code_tokens

def tokens_repr(trace: dict, include_prompts: bool = True) -> list[str]:
    """Extract token-level representation: sequence of token types from code."""
    if not trace or not isinstance(trace, dict):
        return []
    
    tokens = []
    events = trace.get('events', [])
    if not events:
        return []
    
    for event in events:
        if not isinstance(event, dict):
            continue
            
        try:
            event_type = (event.get('type') or '').lower()
            details = event.get('details', {})
            
            if isinstance(details, str):
                try:
                    details = json.loads(details)
                except (json.JSONDecodeError, TypeError):
                    details = {}
            
            if isinstance(details, dict):
                code_content = details.get('after_content') or details.get('before_content') or details.get('code', '')
                file_path = details.get('file_path') or details.get('file')
                
                if code_content and isinstance(code_content, str):
                    try:
                        code_tokens = _extract_code_tokens(code_content, file_path)
                        canonicalized = []
                        id_counter = 1
                        for token in code_tokens:
                            if token == 'IDENTIFIER':
                                canonicalized.append(f'ID_{id_counter:03d}')
                                id_counter += 1
                            else:
                                canonicalized.append(token)
                        
                        tokens.extend(canonicalized[:200])
                        continue
                    except Exception:
                        pass
            
            kind = event.get('type') or event.get('annotation') or event.get('intent')
            if kind:
                tokens.append(str(kind))
        except Exception:
            continue
    
    return tokens

def tokens_repr_str(trace: dict, limit: int = 200) -> str:
    """Extract tokens as a string representation."""
    tokens = tokens_repr(trace, include_prompts=True)
    if not tokens:
        return "EMPTY_TRACE"
    token_str = " ".join(tokens[:limit])
    if len(tokens) > limit:
        token_str += f" ... [truncated from {len(tokens)} tokens]"
    return token_str

