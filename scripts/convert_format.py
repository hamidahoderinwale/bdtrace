#!/usr/bin/env python3
"""
Format Conversion Utility for Developer Traces
Converts between JSON, JSONL, and Parquet formats

Usage:
    # JSON to JSONL (streaming, large datasets)
    python convert_format.py --input traces.json --output traces.jsonl
    
    # JSONL to Parquet (HuggingFace datasets)
    python convert_format.py --input traces.jsonl --output traces.parquet
    
    # JSON to Parquet (direct)
    python convert_format.py --input traces.json --output traces.parquet
    
    # With filters
    python convert_format.py --input traces.jsonl --output filtered.jsonl --min-events 10
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Iterator, Dict, Any

def detect_format(file_path: str) -> str:
    """Auto-detect format from file extension"""
    ext = Path(file_path).suffix.lower()
    if ext == '.jsonl':
        return 'jsonl'
    elif ext == '.parquet':
        return 'parquet'
    elif ext == '.json':
        return 'json'
    else:
        raise ValueError(f"Unknown format: {ext}. Use .json, .jsonl, or .parquet")

def read_json(file_path: str) -> Iterator[Dict[str, Any]]:
    """Read standard JSON (array of objects or single object)"""
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    # Handle both single object and array
    if isinstance(data, dict):
        # Single session
        yield data
    elif isinstance(data, list):
        # Array of sessions
        for item in data:
            yield item
    else:
        raise ValueError("JSON must be object or array of objects")

def read_jsonl(file_path: str) -> Iterator[Dict[str, Any]]:
    """Read JSON Lines (one object per line) - streaming"""
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}", file=sys.stderr)

def read_parquet(file_path: str) -> Iterator[Dict[str, Any]]:
    """Read Parquet file"""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("Error: pyarrow required for Parquet support", file=sys.stderr)
        print("Install: pip install pyarrow", file=sys.stderr)
        sys.exit(1)
    
    table = pq.read_table(file_path)
    for batch in table.to_batches():
        for row in batch.to_pylist():
            yield row

def write_json(data: Iterator[Dict[str, Any]], output_path: str):
    """Write standard JSON (array format)"""
    items = list(data)
    with open(output_path, 'w') as f:
        json.dump(items, f, indent=2)
    print(f"Wrote {len(items)} items to {output_path}")

def write_jsonl(data: Iterator[Dict[str, Any]], output_path: str):
    """Write JSON Lines (one object per line) - streaming friendly"""
    count = 0
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
            count += 1
    print(f"Wrote {count} items to {output_path}")

def write_parquet(data: Iterator[Dict[str, Any]], output_path: str, schema=None):
    """Write Parquet file with optional schema"""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print("Error: pyarrow required for Parquet support", file=sys.stderr)
        print("Install: pip install pyarrow", file=sys.stderr)
        sys.exit(1)
    
    # Convert iterator to list (Parquet needs schema inference)
    items = list(data)
    
    if not items:
        print("Warning: No data to write", file=sys.stderr)
        return
    
    # Infer schema from data if not provided
    if schema is None:
        table = pa.Table.from_pylist(items)
    else:
        table = pa.Table.from_pylist(items, schema=schema)
    
    # Write with compression
    pq.write_table(table, output_path, compression='snappy')
    print(f"Wrote {len(items)} items to {output_path} (Parquet with Snappy compression)")

def filter_data(data: Iterator[Dict[str, Any]], args) -> Iterator[Dict[str, Any]]:
    """Apply filters to data stream"""
    for item in data:
        # Filter by minimum events
        if args.min_events:
            events = item.get('events', [])
            if len(events) < args.min_events:
                continue
        
        # Filter by workspace
        if args.workspace:
            if item.get('workspace') != args.workspace:
                continue
        
        # Filter by date range
        if args.since:
            timestamp = item.get('timestamp') or item.get('events', [{}])[0].get('timestamp', '')
            if timestamp < args.since:
                continue
        
        if args.until:
            timestamp = item.get('timestamp') or item.get('events', [{}])[0].get('timestamp', '')
            if timestamp > args.until:
                continue
        
        yield item

def main():
    parser = argparse.ArgumentParser(
        description='Convert developer traces between JSON, JSONL, and Parquet formats',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert JSON to JSONL (for streaming/large datasets)
  %(prog)s --input traces.json --output traces.jsonl
  
  # Convert JSONL to Parquet (for HuggingFace)
  %(prog)s --input traces.jsonl --output dataset.parquet
  
  # Filter while converting
  %(prog)s --input all.jsonl --output filtered.jsonl --min-events 10 --workspace /my/project
  
  # Check file stats
  %(prog)s --input traces.jsonl --stats
        """
    )
    
    parser.add_argument('--input', '-i', required=True, help='Input file path')
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--stats', action='store_true', help='Show statistics only, no conversion')
    
    # Filters
    parser.add_argument('--min-events', type=int, help='Minimum number of events per session')
    parser.add_argument('--workspace', help='Filter by workspace path')
    parser.add_argument('--since', help='Filter by start date (ISO format)')
    parser.add_argument('--until', help='Filter by end date (ISO format)')
    
    args = parser.parse_args()
    
    # Detect input format
    input_format = detect_format(args.input)
    print(f"Reading {input_format.upper()} from: {args.input}")
    
    # Read data
    if input_format == 'json':
        data = read_json(args.input)
    elif input_format == 'jsonl':
        data = read_jsonl(args.input)
    elif input_format == 'parquet':
        data = read_parquet(args.input)
    
    # Apply filters
    data = filter_data(data, args)
    
    # Stats mode
    if args.stats:
        count = 0
        total_events = 0
        workspaces = set()
        
        for item in data:
            count += 1
            events = item.get('events', [])
            total_events += len(events)
            if 'workspace' in item:
                workspaces.add(item['workspace'])
        
        print(f"\n=== Statistics ===")
        print(f"Sessions: {count}")
        print(f"Total events: {total_events}")
        print(f"Avg events/session: {total_events/count if count > 0 else 0:.1f}")
        print(f"Unique workspaces: {len(workspaces)}")
        return
    
    # Conversion mode
    if not args.output:
        parser.error("--output required for conversion")
    
    output_format = detect_format(args.output)
    print(f"Writing {output_format.upper()} to: {args.output}")
    
    # Write data
    if output_format == 'json':
        write_json(data, args.output)
    elif output_format == 'jsonl':
        write_jsonl(data, args.output)
    elif output_format == 'parquet':
        write_parquet(data, args.output)
    
    print("Conversion complete!")

if __name__ == '__main__':
    main()

