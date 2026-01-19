#!/bin/bash
# Reset script - clears all data, models, and registry

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Hotswap Reset Script ==="
echo "Project directory: $PROJECT_DIR"
echo ""

# Check if server is running
if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
    echo "WARNING: Server is running on port 8000!"
    echo "Stop the server first, then run reset."
    echo ""
    read -p "Stop anyway and reset? [y/N]: " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Remove training data
echo "Removing training data..."
rm -rf "$PROJECT_DIR/data/"*.pt "$PROJECT_DIR/data/"*.csv 2>/dev/null || true

# Remove model checkpoints
echo "Removing model checkpoints..."
rm -rf "$PROJECT_DIR/models/checkpoints/"*.pt 2>/dev/null || true

# Remove registry database
echo "Removing registry database..."
rm -f "$PROJECT_DIR/models/registry.db" 2>/dev/null || true

# Recreate directories
mkdir -p "$PROJECT_DIR/data"
mkdir -p "$PROJECT_DIR/models/checkpoints"

echo ""
echo "Reset complete! Ready for fresh start."
