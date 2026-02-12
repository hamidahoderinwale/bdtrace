import json
from ..core.utils import redact_code_pii, redact_pii

def raw_repr(trace: dict, include_metadata: bool = True, redact_pii_enabled: bool = True) -> dict:
    """Extract raw representation: triple of (code_change, prompt, metadata) with PII redaction."""
    result = {
        'code_changes': [],
        'prompts': [],
    }
    
    if include_metadata:
        result['metadata'] = {
            'session_id': trace.get('session_id'),
            'workspace_path': trace.get('workspace_path'),
            'event_count': len(trace.get('events', [])),
            'code_change_count': 0,
            'prompt_count': 0,
        }
    
    for event in trace.get('events', []):
        event_type = (event.get('type') or '').lower()
        details = event.get('details', {})
        
        if not isinstance(details, dict):
            continue
        
        if event_type in ('code_change', 'file_change', 'entry_created'):
            before_content = details.get('before_content', '')
            after_content = details.get('after_content', '')
            file_path = details.get('file_path') or details.get('file', '')
            
            if redact_pii_enabled:
                before_content = redact_code_pii(before_content)
                after_content = redact_code_pii(after_content)
            
            code_change = {
                'file_path': file_path,
                'before_content': before_content,
                'after_content': after_content,
                'diff_summary': details.get('diff_summary', ''),
                'timestamp': event.get('timestamp'),
            }
            
            if include_metadata:
                code_change['metadata'] = {
                    'event_id': event.get('id'),
                    'event_type': event_type,
                    'lines_added': details.get('lines_added'),
                    'lines_removed': details.get('lines_removed'),
                    'chars_added': details.get('chars_added'),
                    'chars_deleted': details.get('chars_deleted'),
                    'ai_generated': event.get('ai_generated', False),
                    'annotation': event.get('annotation'),
                    'intent': event.get('intent'),
                }
            
            result['code_changes'].append(code_change)
            if include_metadata:
                result['metadata']['code_change_count'] += 1
        
        elif event_type in ('prompt', 'prompt_sent', 'conversation'):
            prompt_content = details.get('text') or details.get('content') or event.get('text') or ''
            
            if redact_pii_enabled:
                prompt_content = redact_pii(prompt_content)
            
            prompt = {
                'content': prompt_content,
                'timestamp': event.get('timestamp'),
            }
            
            if include_metadata:
                prompt['metadata'] = {
                    'event_id': event.get('id'),
                    'event_type': event_type,
                    'conversation_id': details.get('conversation_id'),
                    'model': details.get('model'),
                    'context_files': details.get('context_files'),
                }
            
            result['prompts'].append(prompt)
            if include_metadata:
                result['metadata']['prompt_count'] += 1
    
    if 'prompts' in trace:
        for prompt_data in trace.get('prompts', []):
            prompt_content = prompt_data.get('text') or prompt_data.get('content', '')
            if redact_pii_enabled:
                prompt_content = redact_pii(prompt_content)
            
            prompt = {
                'content': prompt_content,
                'timestamp': prompt_data.get('timestamp'),
            }
            
            if include_metadata:
                prompt['metadata'] = {
                    'prompt_id': prompt_data.get('id'),
                    'conversation_id': prompt_data.get('conversation_id'),
                    'model': prompt_data.get('model'),
                    'context_files': prompt_data.get('context_files'),
                }
            
            result['prompts'].append(prompt)
            if include_metadata:
                result['metadata']['prompt_count'] += 1
    
    return result

def raw_repr_str(trace: dict, include_metadata: bool = True, redact_pii_enabled: bool = True) -> str:
    """Get raw representation as a formatted string."""
    raw = raw_repr(trace, include_metadata, redact_pii_enabled)
    parts = []
    
    if raw['code_changes']:
        parts.append("CODE_CHANGES:")
        for i, change in enumerate(raw['code_changes'], 1):
            parts.append(f"\n  Change {i}:")
            parts.append(f"    File: {change.get('file_path', 'unknown')}")
            if change.get('before_content'):
                parts.append(f"    Before: {change['before_content'][:200]}...")
            if change.get('after_content'):
                parts.append(f"    After: {change['after_content'][:200]}...")
            parts.append(f"    Summary: {change.get('diff_summary', '')}")
    
    if raw['prompts']:
        parts.append("\nPROMPTS:")
        for i, prompt in enumerate(raw['prompts'], 1):
            parts.append(f"\n  Prompt {i}:")
            parts.append(f"    Content: {prompt['content'][:300]}...")
    
    if include_metadata and raw.get('metadata'):
        parts.append(f"\nMETADATA: {raw['metadata']}")
    
    return "\n".join(parts) if parts else "EMPTY_RAW_REPRESENTATION"

