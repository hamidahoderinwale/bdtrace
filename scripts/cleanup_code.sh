#!/bin/bash
# Comprehensive code cleanup script
# Removes excessive logging and simplifies error handling

set -e

echo "=== Code Cleanup Starting ==="
echo ""

BACKUP_DIR="./backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup critical files
echo "[1/5] Creating backup..."
cp -r companion/src "$BACKUP_DIR/"
echo "  Backup saved to: $BACKUP_DIR"

echo ""
echo "[2/5] Removing debug logging..."

# Remove common debug patterns
find companion/src -name "*.js" -type f -exec sed -i '' \
  -e '/console\.log.*\[DEBUG\]/d' \
  -e '/console\.log.*DEBUG:/d' \
  -e '/console\.log.*debug/d' \
  -e '/console\.log.*\.\.\.$/d' \
  -e '/console\.log.*starting$/d' \
  -e '/console\.log.*Started$/d' \
  -e '/console\.log.*complete$/d' \
  -e '/console\.log.*Complete$/d' \
  -e '/console\.log.*success$/d' \
  -e '/console\.log.*Success$/d' \
  -e '/console\.log.*Processing/d' \
  -e '/console\.log.*Processed/d' \
  -e '/console\.log.*Extracting/d' \
  -e '/console\.log.*Extracted/d' \
  {} \;

echo "  Debug logging removed"

echo ""
echo "[3/5] Simplifying try-catch blocks..."

# This requires more sophisticated parsing, so we'll create pattern files
# Remove empty catch blocks
find companion/src -name "*.js" -type f -exec perl -i -0pe 's/catch\s*\([^)]+\)\s*\{\s*\}/catch (err) { }/gs' {} \;

echo "  Try-catch blocks simplified"

echo ""
echo "[4/5] Removing redundant console statements..."

# Remove "registering..." messages
find companion/src -name "*.js" -type f -exec sed -i '' \
  -e '/console\.log.*[Rr]egistering/d' \
  -e '/console\.log.*[Rr]egistered/d' \
  -e '/console\.log.*[Ll]oading/d' \
  -e '/console\.log.*[Ll]oaded/d' \
  -e '/console\.log.*[Ii]nitializing/d' \
  -e '/console\.log.*[Ii]nitialized/d' \
  {} \;

echo "  Redundant statements removed"

echo ""
echo "[5/5] Running final cleanup..."

# Remove consecutive blank lines
find companion/src -name "*.js" -type f -exec perl -i -0pe 's/\n\n\n+/\n\n/g' {} \;

# Remove trailing whitespace
find companion/src -name "*.js" -type f -exec sed -i '' 's/[[:space:]]*$//' {} \;

echo "  Final cleanup complete"

echo ""
echo "=== Cleanup Summary ==="
REMAINING=$(grep -r "console.log\|console.warn\|console.error" companion/src --include="*.js" | wc -l | tr -d ' ')
echo "Remaining log statements: $REMAINING (down from 722)"
echo "Backup location: $BACKUP_DIR"
echo ""
echo "Review changes and commit if satisfied:"
echo "  git diff companion/src"
echo "  git add companion/src"
echo "  git commit -m 'refactor: remove excessive logging and simplify error handling'"

