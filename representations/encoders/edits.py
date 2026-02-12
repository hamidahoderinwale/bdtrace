import json
import hashlib
from pathlib import Path
from collections import Counter
from ..core.intent import intent_tokens_for_event, extract_event_intent, extract_emergent_intent
from ..core.utils import extract_function_names_from_code

def semantic_edits_repr(
    trace: dict, 
    include_prompts: bool = True, 
    include_intent: bool = True,
    canonicalize: bool = False,
    use_emergent: bool = True,
) -> List[str]:
    """Extract semantic edit representation: operation->target pairs with intent encoding."""
    if not trace or not isinstance(trace, dict):
        return []
    
    edits = []
    events = trace.get('events', [])
    if not events:
        return []
    
    max_segment_size = 10
    
    for i, event in enumerate(events):
        if not isinstance(event, dict):
            continue
            
        try:
            event_type = event.get('type', '').lower()
            op = event.get('operation') or event.get('verb') or event_type
            target = event.get('target') or event.get('file') or event.get('symbol')
            details = event.get('details', {})
            
            if isinstance(details, str):
                try:
                    details = json.loads(details)
                except (json.JSONDecodeError, TypeError):
                    details = {}
            
            if not isinstance(details, dict):
                details = {}
            
            file_path = details.get('file_path') or details.get('file')
            lines_added = details.get('lines_added', 0) or 0
            lines_removed = details.get('lines_removed', 0) or 0
            diff_summary = details.get('diff_summary', '')
            
            if event_type in ('file_change', 'code_change', 'entry_modified'):
                if lines_added > 0 and lines_removed == 0:
                    op = 'ADD'
                elif lines_removed > 0 and lines_added == 0:
                    op = 'REMOVE'
                else:
                    op = 'MODIFY'
            elif event_type in ('entry_created', 'file_created'):
                op = 'CREATE'
            elif event_type in ('entry_deleted', 'file_deleted'):
                op = 'DELETE'
            
            if file_path:
                if canonicalize:
                    path_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
                    target_base = f"F_{path_hash}"
                else:
                    path_obj = Path(file_path)
                    target_base = path_obj.stem
                
                code_content = details.get('after_content') or details.get('before_content', '')
                if code_content:
                    funcs = extract_function_names_from_code(code_content, file_path)
                    if funcs:
                        if canonicalize:
                            func_hash = hashlib.md5(str(funcs[0]).encode()).hexdigest()[:6]
                            target = f"{target_base}::FN_{func_hash}"
                        else:
                            target = f"{target_base}::{funcs[0]}"
                    else:
                        target = target_base
                else:
                    target = target_base
                
                size_indicator = 'SMALL'
                if lines_added > 50 or lines_removed > 50:
                    size_indicator = 'LARGE'
                elif lines_added > 10 or lines_removed > 10:
                    size_indicator = 'MEDIUM'
                
                edit_str = f"{op}->{target}"
                if size_indicator != 'SMALL':
                    edit_str += f"->{size_indicator}"
                
                if diff_summary:
                    summary_words = diff_summary.split()[:3]
                    if summary_words:
                        edit_str += f"->{'_'.join(summary_words[:2])}"
                
                if include_intent:
                    event_intents = intent_tokens_for_event(
                        event, 
                        include_llm=False,
                        use_canonicalized_paths=canonicalize,
                        use_emergent=use_emergent,
                    )
                    if event_intents:
                        edit_str += f"->{event_intents[0]}"
                        edits.append(edit_str)
                        for additional_intent in event_intents[1:]:
                            edits.append(f"INTENT->{additional_intent}")
                    else:
                        edits.append(edit_str)
                else:
                    edits.append(edit_str)
            elif op and target:
                edit_str = f"{op}->{target}"
                if include_intent:
                    event_intents = intent_tokens_for_event(
                        event, 
                        include_llm=False,
                        use_canonicalized_paths=canonicalize,
                        use_emergent=use_emergent,
                    )
                    if event_intents:
                        edit_str += f"->{event_intents[0]}"
                        edits.append(edit_str)
                        for additional_intent in event_intents[1:]:
                            edits.append(f"INTENT->{additional_intent}")
                    else:
                        edits.append(edit_str)
                else:
                    edits.append(edit_str)
            elif op:
                if include_intent:
                    event_intents = intent_tokens_for_event(
                        event, 
                        include_llm=False,
                        use_canonicalized_paths=canonicalize,
                        use_emergent=use_emergent,
                    )
                    if event_intents:
                        edits.append(f"{op}->{event_intents[0]}")
                        for additional_intent in event_intents[1:]:
                            edits.append(f"INTENT->{additional_intent}")
                    else:
                        edits.append(str(op))
                else:
                    edits.append(str(op))
        
            if include_intent and (i + 1) % max_segment_size == 0:
                segment_events = events[max(0, i - max_segment_size + 1):i + 1]
                segment_intent_counts = Counter()
                for seg_event in segment_events:
                    if use_emergent:
                        seg_intents = extract_emergent_intent(seg_event, use_llm=False)
                    else:
                        seg_intents = extract_event_intent(seg_event, use_canonicalized_paths=canonicalize)
                    segment_intent_counts.update(seg_intents)
                
                top_segment_intents = [intent for intent, _ in segment_intent_counts.most_common(3)]
                for seg_intent in top_segment_intents:
                    edits.append(f"SEGMENT_INTENT->{seg_intent}")
                
        except Exception:
            continue
    
    return edits

def semantic_edits_repr_str(
    trace: Dict, 
    limit: int = 50, 
    include_intent: bool = True, 
    canonicalize: bool = False,
    use_emergent: bool = True,
) -> str:
    """Extract semantic edits as a string representation."""
    edits = semantic_edits_repr(
        trace, 
        include_prompts=True, 
        include_intent=include_intent, 
        canonicalize=canonicalize,
        use_emergent=use_emergent
    )
    if not edits:
        return "EMPTY_TRACE"
    edit_str = " → ".join(edits[:limit])
    if len(edits) > limit:
        edit_str += f" ... [truncated from {len(edits)} edits]"
    return edit_str

