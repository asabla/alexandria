#!/usr/bin/env bash
#
# new-plan.sh - Create a new planning directory from templates
#
# Usage: ./scripts/plan/new-plan.sh "Plan Name"
#
# This script creates a new .plan/ directory at the repository root,
# populated with templates for structured planning.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEMPLATE_DIR="$SCRIPT_DIR/templates"
PLAN_DIR="$REPO_ROOT/.plan"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 \"Plan Name\""
    echo ""
    echo "Creates a new .plan/ directory from templates."
    echo ""
    echo "Arguments:"
    echo "  Plan Name    The name of the plan (e.g., \"Initial Implementation\")"
    echo ""
    echo "Examples:"
    echo "  $0 \"Initial Implementation\""
    echo "  $0 \"Feature: User Authentication\""
    exit 1
}

error() {
    echo -e "${RED}Error:${NC} $1" >&2
    exit 1
}

warn() {
    echo -e "${YELLOW}Warning:${NC} $1"
}

success() {
    echo -e "${GREEN}Success:${NC} $1"
}

# Check arguments
if [[ $# -lt 1 ]]; then
    usage
fi

PLAN_NAME="$1"

# Check if .plan/ already exists
if [[ -d "$PLAN_DIR" ]]; then
    error ".plan/ directory already exists. Archive it first with: ./scripts/plan/archive-plan.sh"
fi

# Check templates exist
if [[ ! -d "$TEMPLATE_DIR" ]]; then
    error "Template directory not found: $TEMPLATE_DIR"
fi

# Create plan directory
echo "Creating new plan: $PLAN_NAME"
mkdir -p "$PLAN_DIR"

# Get current date
CREATED_DATE=$(date -Iseconds)
CREATED_DATE_SHORT=$(date +%Y-%m-%d)

# Generate slug from plan name (lowercase, spaces to hyphens)
PLAN_SLUG=$(echo "$PLAN_NAME" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | sed 's/--*/-/g' | sed 's/^-//' | sed 's/-$//')

# Copy and process templates
for template in "$TEMPLATE_DIR"/*.tmpl; do
    if [[ -f "$template" ]]; then
        filename=$(basename "$template" .tmpl)
        target="$PLAN_DIR/$filename"
        
        # Replace placeholders
        sed -e "s/{{PLAN_NAME}}/$PLAN_NAME/g" \
            -e "s/{{PLAN_SLUG}}/$PLAN_SLUG/g" \
            -e "s/{{CREATED_DATE}}/$CREATED_DATE/g" \
            -e "s/{{CREATED_DATE_SHORT}}/$CREATED_DATE_SHORT/g" \
            "$template" > "$target"
        
        echo "  Created: $filename"
    fi
done

success "Plan created at .plan/"
echo ""
echo "Next steps:"
echo "  1. Edit .plan/00_overview.md with your high-level plan"
echo "  2. Break down into stories in .plan/01_stories.md"
echo "  3. Define tasks in .plan/02_tasks.md"
echo ""
echo "When complete, archive with: ./scripts/plan/archive-plan.sh <number>"
