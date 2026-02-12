#!/usr/bin/env python3
"""
Context Precision Calculator
Calculates CP = |Context_documents ∩ Diff_documents| / |Context_documents|
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database_connector import DatabaseConnector
import config


def extract_context_files(prompt: Dict) -> List[str]:
    """Extract context file paths from prompt"""
    context_files_json = prompt.get('context_files_json')
    if not context_files_json:
        return []
    
    try:
        if isinstance(context_files_json, str):
            context = json.loads(context_files_json)
        else:
            context = context_files_json
        
        files = []
        
        # Handle different context file formats
        if isinstance(context, list):
            files = [f if isinstance(f, str) else f.get('path') or f.get('fileName') for f in context]
        elif isinstance(context, dict):
            if 'attachedFiles' in context:
                files = [f.get('path') or f.get('fileName') for f in context['attachedFiles']]
            elif 'codebaseFiles' in context:
                files = [f.get('path') or f.get('fileName') for f in context['codebaseFiles']]
            elif 'files' in context:
                files = [f if isinstance(f, str) else f.get('path') or f.get('fileName') for f in context['files']]
        
        # Filter out None/empty
        return [f for f in files if f]
    except Exception as e:
        print(f"[CP] Error extracting context files: {e}", file=sys.stderr)
        return []


def calculate_cp(prompt: Dict, diff_files: List[str]) -> Dict:
    """Calculate Context Precision for a prompt"""
    context_files = extract_context_files(prompt)
    
    if not context_files:
        return {
            'prompt_id': prompt.get('id'),
            'cp': 0.0,
            'context_file_count': 0,
            'diff_file_count': len(diff_files),
            'intersection_count': 0,
            'context_files': [],
            'diff_files': diff_files,
            'unused_context_files': []
        }
    
    # Calculate intersection
    context_set = set(context_files)
    diff_set = set(diff_files)
    intersection = [f for f in context_files if f in diff_set]
    unused = [f for f in context_files if f not in diff_set]
    
    # Calculate CP
    cp = len(intersection) / len(context_files) if context_files else 0.0
    
    return {
        'prompt_id': prompt.get('id'),
        'timestamp': prompt.get('timestamp'),
        'cp': cp,
        'context_file_count': len(context_files),
        'diff_file_count': len(diff_files),
        'intersection_count': len(intersection),
        'context_files': context_files,
        'diff_files': diff_files,
        'intersection': intersection,
        'unused_context_files': unused
    }


def calculate_baseline_cp(db: DatabaseConnector, workspace_path: str = None, limit: int = 1000) -> Dict:
    """Calculate baseline CP for all prompts"""
    prompts = db.get_prompts(workspace_path=workspace_path, limit=limit)
    
    cp_scores = []
    cp_records = []
    
    for prompt in prompts:
        # Get diff files for this prompt
        prompt_id = prompt.get('id')
        diff_files_data = db.get_entries_for_prompt(
            prompt_id, 
            time_window_seconds=config.CP_TIME_WINDOW_SECONDS
        )
        diff_files = [e.get('file_path') for e in diff_files_data if e.get('file_path')]
        
        # Calculate CP
        cp_record = calculate_cp(prompt, diff_files)
        cp_scores.append(cp_record['cp'])
        cp_records.append(cp_record)
    
    if not cp_scores:
        return {
            'average': 0.0,
            'median': 0.0,
            'min': 0.0,
            'max': 0.0,
            'count': 0,
            'distribution': {}
        }
    
    # Calculate statistics
    cp_scores_sorted = sorted(cp_scores)
    n = len(cp_scores_sorted)
    
    median = cp_scores_sorted[n // 2] if n > 0 else 0.0
    if n % 2 == 0 and n > 1:
        median = (cp_scores_sorted[n // 2 - 1] + cp_scores_sorted[n // 2]) / 2
    
    # Distribution buckets
    distribution = {
        '0.0-0.2': sum(1 for cp in cp_scores if 0.0 <= cp < 0.2),
        '0.2-0.4': sum(1 for cp in cp_scores if 0.2 <= cp < 0.4),
        '0.4-0.6': sum(1 for cp in cp_scores if 0.4 <= cp < 0.6),
        '0.6-0.8': sum(1 for cp in cp_scores if 0.6 <= cp < 0.8),
        '0.8-1.0': sum(1 for cp in cp_scores if 0.8 <= cp <= 1.0)
    }
    
    return {
        'average': sum(cp_scores) / len(cp_scores),
        'median': median,
        'min': min(cp_scores),
        'max': max(cp_scores),
        'count': len(cp_scores),
        'distribution': distribution,
        'records': cp_records
    }


def main():
    parser = argparse.ArgumentParser(description='Calculate Context Precision')
    parser.add_argument('--workspace', type=str, default=None,
                       help='Workspace path to analyze')
    parser.add_argument('--prompt-id', type=int, default=None,
                       help='Calculate CP for specific prompt ID')
    parser.add_argument('--baseline', action='store_true',
                       help='Calculate baseline CP for all prompts')
    parser.add_argument('--limit', type=int, default=1000,
                       help='Limit number of prompts (for baseline)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file (default: stdout)')
    
    args = parser.parse_args()
    
    # Connect to database
    try:
        db = DatabaseConnector()
    except Exception as e:
        print(f"[ERROR] Failed to connect to database: {e}", file=sys.stderr)
        sys.exit(1)
    
    try:
        if args.prompt_id:
            # Calculate CP for single prompt
            prompts = db.get_prompts(limit=1)
            prompt = next((p for p in prompts if p.get('id') == args.prompt_id), None)
            
            if not prompt:
                print(f"[ERROR] Prompt {args.prompt_id} not found", file=sys.stderr)
                sys.exit(1)
            
            diff_files_data = db.get_entries_for_prompt(
                args.prompt_id,
                time_window_seconds=config.CP_TIME_WINDOW_SECONDS
            )
            diff_files = [e.get('file_path') for e in diff_files_data if e.get('file_path')]
            
            result = calculate_cp(prompt, diff_files)
        elif args.baseline:
            # Calculate baseline
            result = calculate_baseline_cp(db, workspace_path=args.workspace, limit=args.limit)
        else:
            print("[ERROR] Specify --prompt-id or --baseline", file=sys.stderr)
            sys.exit(1)
        
        # Output
        output_json = json.dumps(result, indent=2)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output_json)
        else:
            print(output_json)
    
    finally:
        db.close()


if __name__ == '__main__':
    main()

