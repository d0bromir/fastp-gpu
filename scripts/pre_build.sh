#!/bin/bash

################################################################################
# Pre-Build Version Check & Auto-Increment
#
# This script is called before the build to:
# 1. Check if source files have changed
# 2. Auto-increment patch version if changed
# 3. Update the version checksum
#
# To use with CMake, add to CMakeLists.txt:
#   add_custom_command(TARGET fastp PRE_BUILD
#       COMMAND bash ${PROJECT_SOURCE_DIR}/scripts/pre_build.sh
#   )
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Run version manager check
echo "=== Version Check ==="
bash scripts/version_manager.sh check

# If source files changed, increment version
if bash scripts/version_manager.sh check 2>&1 | grep -q "changed"; then
    echo ""
    echo "=== Auto-Incrementing Patch Version ==="
    bash scripts/version_manager.sh increment
    echo ""
    echo "New version: $(bash scripts/version_manager.sh get)"
fi
