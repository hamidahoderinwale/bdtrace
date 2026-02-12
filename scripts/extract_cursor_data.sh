#!/bin/bash
# Standalone Cursor Database Extraction Script
# Extracts raw data from Cursor databases without running the companion service
# Usage: ./extract_cursor_data.sh [output_dir]

set -e

OUTPUT_DIR="${1:-./cursor_exports}"
CURSOR_BASE="$HOME/Library/Application Support/Cursor"
GLOBAL_DB="$CURSOR_BASE/User/globalStorage/state.vscdb"
WORKSPACE_STORAGE="$CURSOR_BASE/User/workspaceStorage"

echo "=== Cursor Database Extraction ==="
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if Cursor database exists
if [ ! -f "$GLOBAL_DB" ]; then
    echo "Error: Cursor global database not found at: $GLOBAL_DB"
    exit 1
fi

echo "[1/4] Extracting prompts from global database..."
sqlite3 "$GLOBAL_DB" "SELECT value FROM ItemTable WHERE key = 'aiService.prompts'" > "$OUTPUT_DIR/prompts_raw.txt" 2>/dev/null || echo "[]" > "$OUTPUT_DIR/prompts_raw.txt"

echo "[2/4] Extracting conversations from global database..."
sqlite3 "$GLOBAL_DB" "SELECT value FROM ItemTable WHERE key LIKE 'aiService.conversations%'" > "$OUTPUT_DIR/conversations_raw.txt" 2>/dev/null || echo "[]" > "$OUTPUT_DIR/conversations_raw.txt"

echo "[3/4] Extracting AI generations from global database..."
sqlite3 "$GLOBAL_DB" "SELECT value FROM ItemTable WHERE key = 'aiService.generations'" > "$OUTPUT_DIR/generations_raw.txt" 2>/dev/null || echo "[]" > "$OUTPUT_DIR/generations_raw.txt"

echo "[4/4] Scanning workspace databases..."
WORKSPACE_COUNT=0
if [ -d "$WORKSPACE_STORAGE" ]; then
    for workspace_dir in "$WORKSPACE_STORAGE"/*; do
        if [ -d "$workspace_dir" ]; then
            workspace_id=$(basename "$workspace_dir")
            workspace_db="$workspace_dir/state.vscdb"
            
            if [ -f "$workspace_db" ]; then
                echo "  - Extracting from workspace: $workspace_id"
                
                # Extract workspace-specific data
                sqlite3 "$workspace_db" "SELECT value FROM ItemTable WHERE key LIKE 'aiService%'" > "$OUTPUT_DIR/workspace_${workspace_id}.txt" 2>/dev/null || true
                
                # Try to get workspace path from workspace.json
                if [ -f "$workspace_dir/workspace.json" ]; then
                    cp "$workspace_dir/workspace.json" "$OUTPUT_DIR/workspace_${workspace_id}_meta.json"
                fi
                
                WORKSPACE_COUNT=$((WORKSPACE_COUNT + 1))
            fi
        fi
    done
fi

echo ""
echo "=== Extraction Complete ==="
echo "Extracted data from $WORKSPACE_COUNT workspaces"
echo "Raw files saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Parse raw files: python scripts/parse_to_traces.py --input $OUTPUT_DIR"
echo "  2. Convert to JSONL: python scripts/convert_format.py --input traces.json --output traces.jsonl"
echo "  3. Convert to Parquet: python scripts/convert_format.py --input traces.jsonl --output traces.parquet"

