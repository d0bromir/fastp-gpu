#!/bin/bash

################################################################################
# fastp Version Manager - Auto-increment patch version on source code changes
#
# This script manages the fastp version number and automatically increments
# the patch version when source files are modified.
#
# Usage:
#   ./scripts/version_manager.sh check              # Check current version
#   ./scripts/version_manager.sh increment          # Increment patch version
#   ./scripts/version_manager.sh set <version>      # Set specific version
#   ./scripts/version_manager.sh get                # Print current version
################################################################################

set -e

# Configuration
VERSION_FILE="src/common.h"
VERSION_CHECKSUM_FILE=".version_checksum"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

################################################################################
# Helper Functions
################################################################################

# Get current version from common.h
get_current_version() {
    grep '#define FASTP_VER' "$VERSION_FILE" | sed 's/#define FASTP_VER "\(.*\)"/\1/'
}

# Update version in common.h
set_version() {
    local new_version="$1"
    sed -i "s/#define FASTP_VER \"[^\"]*\"/#define FASTP_VER \"$new_version\"/" "$VERSION_FILE"
    echo "Version updated to: $new_version"
}

# Increment patch version
increment_patch() {
    local current=$(get_current_version)
    local major=$(echo "$current" | cut -d. -f1)
    local minor=$(echo "$current" | cut -d. -f2)
    local patch=$(echo "$current" | cut -d. -f3)
    
    patch=$((patch + 1))
    local new_version="${major}.${minor}.${patch}"
    
    set_version "$new_version"
    echo "$new_version"
}

# Calculate checksum of source files
calculate_source_checksum() {
    find src -name "*.cpp" -o -name "*.h" -o -name "*.cu" | \
        xargs ls -la | \
        md5sum | \
        cut -d' ' -f1
}

# Check if source files have changed
source_files_changed() {
    local current_checksum=$(calculate_source_checksum)
    
    if [[ ! -f "$VERSION_CHECKSUM_FILE" ]]; then
        return 0  # No checksum file, consider as changed
    fi
    
    local stored_checksum=$(cat "$VERSION_CHECKSUM_FILE")
    [[ "$current_checksum" != "$stored_checksum" ]]
}

# Save checksum of current source files
save_source_checksum() {
    calculate_source_checksum > "$VERSION_CHECKSUM_FILE"
}

################################################################################
# Commands
################################################################################

case "${1:-check}" in
    check)
        if source_files_changed; then
            echo "⚠️  Source files have changed since last build"
            echo "Current version: $(get_current_version)"
            echo "Run 'increment' to update patch version"
        else
            echo "✓ Source files unchanged"
            echo "Current version: $(get_current_version)"
        fi
        ;;
    
    increment)
        echo "Incrementing patch version..."
        new_version=$(increment_patch)
        save_source_checksum
        echo "✓ Version incremented to: $new_version"
        ;;
    
    set)
        if [[ -z "$2" ]]; then
            echo "Error: version string required"
            echo "Usage: $0 set <version>"
            exit 1
        fi
        set_version "$2"
        save_source_checksum
        ;;
    
    get)
        get_current_version
        ;;
    
    save-checksum)
        save_source_checksum
        echo "✓ Source checksum saved"
        ;;

    release-tag-name)
        # Emit the canonical fork release tag name for the current FASTP_VER.
        # Scheme: v<upstream-baseline>-fastp-d0bromir.<N>
        # Strips any "-d0bromir" suffix from FASTP_VER to recover the upstream
        # baseline portion. <N> defaults to 1 and may be overridden via $2.
        iteration="${2:-1}"
        ver=$(get_current_version)
        baseline="${ver%-d0bromir}"
        echo "v${baseline}-fastp-d0bromir.${iteration}"
        ;;

    *)
        echo "Usage: $0 {check|increment|set <version>|get|save-checksum|release-tag-name [N]}"
        echo ""
        echo "Commands:"
        echo "  check                  - Check if source files changed (shows current version)"
        echo "  increment              - Increment patch version and update checksum"
        echo "  set <version>          - Set specific version (format: X.Y.Z)"
        echo "  get                    - Print current version only"
        echo "  save-checksum          - Save current source file checksum"
        echo "  release-tag-name [N]   - Emit canonical fork release tag for FASTP_VER"
        exit 1
        ;;
esac
