"""
Event Canonicalization

Rule-free canonical event encoding for sequence mining.
Uses hashing to collapse patterns and ensure privacy + stability.
"""

import hashlib
import json
import re
from typing import Dict, List, Optional


def canonicalize_event(event: Dict) -> str:
    """Rule-free canonical event encoder.
    
    This produces a stable finite alphabet for sequence mining without
    hard-coded semantic rules. Uses hashing to collapse patterns and
    ensure privacy + stability across IDEs, agents, and languages.
    
    Args:
        event: Event dictionary
    
    Returns:
        Canonical event symbol like "EV_a13f92"
    """
    # Robust input handling: ensure event is a dict
    if not isinstance(event, dict):
        return "EV_OTHER"
    
    # Extract raw event type (handle None and non-string types)
    event_type = event.get("type") or event.get("operation") or event.get("verb")
    if event_type is None:
        raw = "other"
    else:
        raw = str(event_type).lower()
    
    # Normalize delimiters (handle various separators)
    parts = re.split(r"[._/\\\-]+", raw)
    if not parts:
        return "EV_OTHER"
    
    # Use head (first meaningful part) for stability
    head = parts[0].strip()
    if not head:
        return "EV_OTHER"
    
    # Hash to collapse long tails (privacy + stability)
    # Use first 6 hex chars for readable but stable symbols
    hash_digest = hashlib.sha1(head.encode()).hexdigest()[:6]
    return f"EV_{hash_digest}"


def extract_prompt_from_event(event: Dict) -> Optional[str]:
    """Extract prompt text from an event.
    
    Handles various event structures where prompts might be stored.
    
    Args:
        event: Event dictionary
    
    Returns:
        Prompt text if found, None otherwise
    """
    # Robust input handling
    if event is None or not isinstance(event, dict):
        return None
    
    # Check event type (handle None and non-string types)
    event_type_raw = event.get('type')
    if event_type_raw is None:
        event_type = ''
    else:
        event_type = str(event_type_raw).lower()
    
    # Direct prompt events
    if event_type in ('prompt', 'prompt_sent', 'conversation', 'ai_prompt'):
        # Check various fields where prompt text might be stored
        text = (
            event.get('text') or 
            event.get('content') or 
            event.get('prompt') or
            event.get('message')
        )
        if text:
            return str(text)
        
        # Check details field
        details = event.get('details', {})
        if isinstance(details, dict):
            text = (
                details.get('text') or 
                details.get('content') or 
                details.get('prompt') or
                details.get('message')
            )
            if text:
                return str(text)
        elif isinstance(details, str):
            # Try to parse JSON details
            try:
                details_dict = json.loads(details)
                text = (
                    details_dict.get('text') or 
                    details_dict.get('content') or 
                    details_dict.get('prompt')
                )
                if text:
                    return str(text)
            except (json.JSONDecodeError, TypeError):
                pass
    
    return None


def event_sequence(trace: Dict, include_prompts: bool = True, include_llm_intents: bool = False) -> List[str]:
    """Build canonical event sequence for a trace, including prompt-derived intent markers.
    
    This is the core sequence used for motif mining. Now includes INTENT tokens
    derived from prompts, making motifs intent-aware.
    
    Args:
        trace: Trace dictionary with events
        include_prompts: If True, insert INTENT tokens for prompts; if False, skip prompts
    
    Returns:
        List of canonical event symbols, including INTENT markers
    """
    from .intent import canonicalize_prompt
    
    sequence = []
    
    # Robust input handling
    if not isinstance(trace, dict):
        return []
    
    events = trace.get('events', [])
    if not isinstance(events, list):
        return []
    
    for event in events:
        # Skip None or non-dict events
        if event is None or not isinstance(event, dict):
            continue
        
        # Check if this is a prompt event
        if include_prompts:
            prompt_text = extract_prompt_from_event(event)
            if prompt_text:
                from .intent import intent_tokens_for_prompt
                sequence.extend(intent_tokens_for_prompt(prompt_text, include_llm=include_llm_intents))
        
        # Canonicalize the event itself
        canonical = canonicalize_event(event)
        sequence.append(canonical)
    
    # Also check for prompts stored separately in trace
    if include_prompts and 'prompts' in trace:
        for prompt_data in trace.get('prompts', []):
            if isinstance(prompt_data, dict):
                prompt_text = (
                    prompt_data.get('text') or 
                    prompt_data.get('content') or 
                    prompt_data.get('prompt')
                )
                if prompt_text:
                    from .intent import intent_tokens_for_prompt
                    sequence.extend(intent_tokens_for_prompt(str(prompt_text), include_llm=include_llm_intents))
    
    return sequence


# ============================================================================
# Legacy Canonical Event Vocabulary (Deprecated - kept for compatibility)
# ============================================================================

# NOTE: The new system uses hash-based canonicalization (canonicalize_event above)
# This map is kept for backward compatibility but is no longer used by default.

CANONICAL_EVENT_MAP = {
    # File operations
    'file.create': 'CREATE',
    'file.new': 'CREATE',
    'file.add': 'CREATE',
    'file.rename': 'MODIFY',
    'file.move': 'MODIFY',
    'file.save': 'MODIFY',
    'file.delete': 'DELETE',
    'file.remove': 'DELETE',
    
    # Code editing operations
    'code_change': 'MODIFY',
    'edit': 'MODIFY',
    'modify': 'MODIFY',
    'update': 'MODIFY',
    'change': 'MODIFY',
    'cursor.insertText': 'MODIFY',
    'cursor.deleteText': 'DELETE',
    'cursor.replaceText': 'MODIFY',
    
    # AI/LLM operations
    'ai.applyEdit': 'AI_EDIT',
    'ai.suggestionAccepted': 'AI_ACCEPT',
    'ai.suggestionRejected': 'AI_REJECT',
    'prompt': 'AI_PROMPT',
    'ai': 'AI_INTERACTION',
    
    # Testing and debugging
    'test': 'TEST',
    'run.test': 'TEST',
    'debug': 'DEBUG',
    'run.debug': 'DEBUG',
    'run': 'RUN',
    'execute': 'RUN',
    
    # Navigation and exploration
    'navigate': 'NAVIGATE',
    'search': 'SEARCH',
    'browse': 'NAVIGATE',
    
    # Build/compile operations
    'build': 'BUILD',
    'compile': 'BUILD',
    'deploy': 'DEPLOY',
    
    # Version control
    'git.commit': 'COMMIT',
    'git.push': 'PUSH',
    'git.pull': 'PULL',
    'git.merge': 'MERGE',
}


def canonicalize_event_legacy(event: Dict) -> str:
    """Legacy canonicalization using hard-coded semantic rules.
    
    DEPRECATED: Use canonicalize_event() instead for rule-free hashing.
    Kept for backward compatibility.
    """
    # Try multiple fields for event type
    event_type = (
        event.get('type') or 
        event.get('operation') or 
        event.get('verb') or 
        event.get('action') or 
        ''
    ).lower()
    
    # Check canonical map
    for raw_pattern, canonical in CANONICAL_EVENT_MAP.items():
        if raw_pattern.lower() in event_type:
            return canonical
    
    # Check intent/annotation for semantic hints
    intent = (event.get('intent') or event.get('annotation') or '').lower()
    if intent:
        if any(kw in intent for kw in ['create', 'add', 'new', 'generate']):
            return 'CREATE'
        elif any(kw in intent for kw in ['modify', 'edit', 'update', 'change', 'refactor']):
            return 'MODIFY'
        elif any(kw in intent for kw in ['delete', 'remove', 'drop']):
            return 'DELETE'
        elif any(kw in intent for kw in ['test', 'verify', 'check']):
            return 'TEST'
        elif any(kw in intent for kw in ['debug', 'fix', 'error']):
            return 'DEBUG'
        elif any(kw in intent for kw in ['ai', 'prompt', 'suggest']):
            return 'AI_INTERACTION'
    
    # Default: use event type if available, otherwise OTHER
    if event_type:
        words = event_type.split('.')
        if len(words) > 1:
            return words[-1].upper()
        return event_type.upper()[:10]
    
    return 'OTHER'






