#!/usr/bin/env python3
"""
Parse Raw Cursor Database Exports to Structured Traces
Converts raw SQLite output to standardized trace format

Usage:
    python parse_to_traces.py --input cursor_exports/ --output traces.jsonl
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import re

def parse_raw_json_value(raw_text: str) -> Any:
    """Parse raw JSON from SQLite output (may be malformed)"""
    raw_text = raw_text.strip()
    if not raw_text:
        return None
    
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # Try to fix common issues
        # Sometimes SQLite outputs multiple JSON objects concatenated
        try:
            # Try parsing each line separately
            lines = raw_text.split('\n')
            results = []
            for line in lines:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
            return results if len(results) > 1 else results[0] if results else None
        except:
            return None

def extract_prompts(prompts_file: Path) -> List[Dict]:
    """Extract prompts from raw file"""
    if not prompts_file.exists():
        return []
    
    with open(prompts_file, 'r') as f:
        content = f.read()
    
    data = parse_raw_json_value(content)
    if not data:
        return []
    
    # Handle both single object and array
    if isinstance(data, dict):
        return [data]
    elif isinstance(data, list):
        return data
    return []

def extract_conversations(conversations_file: Path) -> List[Dict]:
    """Extract conversations from raw file"""
    if not conversations_file.exists():
        return []
    
    with open(conversations_file, 'r') as f:
        content = f.read()
    
    data = parse_raw_json_value(content)
    if not data:
        return []
    
    conversations = []
    if isinstance(data, list):
        conversations = data
    elif isinstance(data, dict):
        if 'conversations' in data:
            conversations = data['conversations']
        else:
            conversations = [data]
    
    return conversations

def build_trace_from_data(prompts: List[Dict], conversations: List[Dict], workspace_id: str = None) -> Dict:
    """Build a structured trace from extracted data"""
    # Create events from prompts and conversations
    events = []
    
    # Add prompt events
    for prompt in prompts:
        if not isinstance(prompt, dict):
            continue
        
        event = {
            'type': 'prompt',
            'timestamp': prompt.get('timestamp') or prompt.get('createdAt', ''),
            'details': {
                'text': prompt.get('text') or prompt.get('prompt', ''),
                'model': prompt.get('model', 'unknown'),
            }
        }
        
        # Add context if available
        if 'context' in prompt:
            event['details']['context'] = prompt['context']
        
        events.append(event)
    
    # Add conversation events
    for conv in conversations:
        if not isinstance(conv, dict):
            continue
        
        messages = conv.get('messages', [])
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            
            event = {
                'type': 'conversation_message',
                'timestamp': msg.get('timestamp') or msg.get('createdAt', ''),
                'details': {
                    'role': msg.get('role', 'unknown'),
                    'content': msg.get('content', ''),
                    'conversation_id': conv.get('id', ''),
                }
            }
            events.append(event)
    
    # Sort events by timestamp
    events.sort(key=lambda e: e.get('timestamp', ''))
    
    # Build trace
    trace = {
        'session_id': workspace_id or 'unknown',
        'workspace': workspace_id,
        'events': events,
        'metadata': {
            'total_prompts': len(prompts),
            'total_conversations': len(conversations),
            'total_events': len(events),
        }
    }
    
    return trace

def main():
    parser = argparse.ArgumentParser(
        description='Parse raw Cursor database exports to structured traces'
    )
    parser.add_argument('--input', '-i', required=True, help='Input directory with raw exports')
    parser.add_argument('--output', '-o', required=True, help='Output file (JSON or JSONL)')
    parser.add_argument('--format', choices=['json', 'jsonl'], default='jsonl', 
                       help='Output format (default: jsonl)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1
    
    print(f"Parsing raw exports from: {input_dir}")
    
    # Read raw files
    prompts_file = input_dir / 'prompts_raw.txt'
    conversations_file = input_dir / 'conversations_raw.txt'
    
    prompts = extract_prompts(prompts_file)
    conversations = extract_conversations(conversations_file)
    
    print(f"Extracted {len(prompts)} prompts and {len(conversations)} conversations")
    
    # Find workspace-specific files
    workspace_files = list(input_dir.glob('workspace_*.txt'))
    traces = []
    
    # Build main trace
    if prompts or conversations:
        main_trace = build_trace_from_data(prompts, conversations, 'global')
        if main_trace['events']:
            traces.append(main_trace)
    
    # Build workspace-specific traces
    for workspace_file in workspace_files:
        workspace_id = workspace_file.stem.replace('workspace_', '')
        
        with open(workspace_file, 'r') as f:
            content = f.read()
        
        data = parse_raw_json_value(content)
        if data:
            # Try to extract prompts/conversations from workspace data
            workspace_prompts = []
            workspace_conversations = []
            
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        if 'prompt' in item or 'text' in item:
                            workspace_prompts.append(item)
                        elif 'messages' in item:
                            workspace_conversations.append(item)
            
            trace = build_trace_from_data(workspace_prompts, workspace_conversations, workspace_id)
            if trace['events']:
                traces.append(trace)
    
    print(f"Built {len(traces)} traces")
    
    # Write output
    output_path = Path(args.output)
    
    if args.format == 'jsonl' or output_path.suffix == '.jsonl':
        # Write JSONL (one trace per line)
        with open(output_path, 'w') as f:
            for trace in traces:
                f.write(json.dumps(trace) + '\n')
        print(f"Wrote {len(traces)} traces to {output_path} (JSONL)")
    else:
        # Write JSON array
        with open(output_path, 'w') as f:
            json.dump(traces, f, indent=2)
        print(f"Wrote {len(traces)} traces to {output_path} (JSON)")
    
    print("Parse complete!")

if __name__ == '__main__':
    main()

