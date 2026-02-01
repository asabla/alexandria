#!/usr/bin/env bash
#
# archive-plan.sh - Archive the current .plan/ to docs/plans/
#
# Usage: ./scripts/plan/archive-plan.sh [archive-number]
#
# This script moves the current .plan/ directory to the docs/plans/ archive
# with a numbered prefix and updates metadata.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PLAN_DIR="$REPO_ROOT/.plan"
ARCHIVE_ROOT="$REPO_ROOT/docs/plans"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 [archive-number]"
    echo ""
    echo "Archives the current .plan/ directory to docs/plans/"
    echo ""
    echo "Arguments:"
    echo "  archive-number    Optional. The number prefix for the archive (e.g., 02)"
    echo "                    If not provided, auto-increments from existing archives."
    echo ""
    echo "Examples:"
    echo "  $0              # Auto-number based on existing archives"
    echo "  $0 02           # Creates docs/plans/02_<plan-slug>/"
    exit 1
}

error() {
    echo -e "${RED}Error:${NC} $1" >&2
    exit 1
}

success() {
    echo -e "${GREEN}Success:${NC} $1"
}

info() {
    echo -e "${BLUE}Info:${NC} $1"
}

# Check if .plan/ exists
if [[ ! -d "$PLAN_DIR" ]]; then
    error ".plan/ directory does not exist. Nothing to archive."
fi

# Check if metadata.yaml exists
if [[ ! -f "$PLAN_DIR/metadata.yaml" ]]; then
    error "metadata.yaml not found in .plan/. Is this a valid plan?"
fi

# Extract plan slug from metadata
PLAN_SLUG=$(grep '^slug:' "$PLAN_DIR/metadata.yaml" | sed 's/slug: *//' | tr -d '"' | tr -d "'")
if [[ -z "$PLAN_SLUG" ]]; then
    error "Could not extract plan slug from metadata.yaml"
fi

# Determine archive number
if [[ $# -ge 1 ]]; then
    ARCHIVE_NUM="$1"
else
    # Auto-increment: find highest existing number and add 1
    HIGHEST=$(find "$ARCHIVE_ROOT" -maxdepth 1 -type d -name '[0-9][0-9]_*' 2>/dev/null | \
              sed 's/.*\/\([0-9][0-9]\)_.*/\1/' | sort -n | tail -1)
    if [[ -z "$HIGHEST" ]]; then
        ARCHIVE_NUM="01"
    else
        NEXT=$((10#$HIGHEST + 1))
        ARCHIVE_NUM=$(printf "%02d" $NEXT)
    fi
fi

# Format archive number with leading zero if needed
ARCHIVE_NUM=$(printf "%02d" $((10#$ARCHIVE_NUM)))

ARCHIVE_DIR="$ARCHIVE_ROOT/${ARCHIVE_NUM}_${PLAN_SLUG}"

# Check if archive already exists
if [[ -d "$ARCHIVE_DIR" ]]; then
    error "Archive directory already exists: $ARCHIVE_DIR"
fi

info "Archiving plan to: $ARCHIVE_DIR"

# Create archive directory
mkdir -p "$ARCHIVE_DIR"

# Update metadata with archive info
ARCHIVED_DATE=$(date -Iseconds)
{
    cat "$PLAN_DIR/metadata.yaml"
    echo ""
    echo "# Archive information"
    echo "archived_at: \"$ARCHIVED_DATE\""
    echo "archive_path: \"docs/plans/${ARCHIVE_NUM}_${PLAN_SLUG}\""
} > "$PLAN_DIR/metadata.yaml.tmp"
mv "$PLAN_DIR/metadata.yaml.tmp" "$PLAN_DIR/metadata.yaml"

# Also update status to archived
sed -i 's/^status:.*/status: "archived"/' "$PLAN_DIR/metadata.yaml"

# Move all files to archive
mv "$PLAN_DIR"/* "$ARCHIVE_DIR/"

# Remove .plan/ directory
rmdir "$PLAN_DIR"

success "Plan archived to: $ARCHIVE_DIR"
echo ""
echo "The .plan/ directory has been removed."
echo "To start a new plan: ./scripts/plan/new-plan.sh \"Plan Name\""
