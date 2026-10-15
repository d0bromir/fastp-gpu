# Automatic Version Management for fastp

## Overview

The fastp project now includes an automatic version management system that increments the patch version whenever source code changes. This ensures version numbers stay in sync with code changes.

## How It Works

### Version File
- **Location:** `src/common.h`
- **Current Format:** `#define FASTP_VER "X.Y.Z"`

### Checksum Tracking
- **File:** `.version_checksum`
- **Contains:** MD5 checksum of all source files
- **Tracked:** All `.cpp`, `.h`, and `.cu` files in `src/`

### Automatic Increment
When you build the project:
1. CMake runs the version manager check
2. If source files changed since last checksum
3. Patch version is automatically incremented
4. New checksum is saved

## Usage

### Manual Commands

```bash
# Check current version and if files changed
make version-check

# Get just the version number
make version-get

# Manually increment patch version
make version-increment

# Set specific version
scripts/version_manager.sh set 1.3.0
```

### Automatic (On Build)

```bash
# Using native Makefile (auto-increments if needed)
make
make WITH_CUDA=1
```

## Examples

### Example 1: Current State
```bash
$ make version-get
1.2.2

$ make
=== Version Check ===
✓ Source files unchanged
Current version: 1.2.2
[Compiling fastp...]

$ ./fastp --version
fastp 1.2.2
```

### Example 2: After Source Changes
```bash
# Edit a source file
$ echo "// comment" >> src/filter.cpp

# Next build auto-increments
$ make
=== Version Check ===
⚠️  Source files have changed since last build
Current version: 1.2.2

=== Auto-Incrementing Patch Version ===
Version updated to: 1.2.3
✓ Version incremented to: 1.2.3

[Compiling fastp...]

$ ./fastp --version
fastp 1.2.3
```

### Example 3: Reset After Commit
After committing code changes to git, update the checksum without incrementing:

```bash
# Update checksum after git commit
scripts/version_manager.sh save-checksum

# Next build won't increment
$ make version-check
✓ Source files unchanged
Current version: 1.2.3
```

## Version Control Workflow

### Development Branch
1. Make code changes
2. Build (version auto-increments)
3. Test
4. Commit with bumped version

### Release Workflow
1. Create release branch
2. Code freeze - no more changes
3. Run: `scripts/version_manager.sh save-checksum`
4. Build and test
5. Tag version: `git tag v1.2.1`
6. Merge to main

## Integration with Git Hooks

Optional: Add to `.git/hooks/pre-commit` to auto-increment before commits:

```bash
#!/bin/bash
# Auto-increment version before commit if source changed
bash scripts/version_manager.sh check > /dev/null 2>&1
if grep -q "changed" <(bash scripts/version_manager.sh check 2>&1); then
    bash scripts/version_manager.sh increment
    git add src/common.h .version_checksum
fi
```

## Files Involved

```
/path/to/fastp_d0bromir/
├── scripts/
│   ├── version_manager.sh      # Main version management script
│   └── pre_build.sh            # Pre-build hook (invoked by Makefile)
├── src/
│   └── common.h                # Contains FASTP_VER definition
├── Makefile                    # Canonical build system + version targets
└── .version_checksum           # Auto-generated checksum (gitignored)
```

## Important Notes

1. **Gitignore:** `.version_checksum` should be in `.gitignore` (only tracks local builds)
2. **Deterministic:** Version only increments when actual source changes occur
3. **Manual Override:** Use `scripts/version_manager.sh set X.Y.Z` to override
4. **Build System:** Native `Makefile` is the canonical build entry point

## Troubleshooting

### Version not incrementing
```bash
# Force rescan
rm .version_checksum
make  # Will recalculate and potentially increment
```

### Version stuck on old number
```bash
# Check what's in the checksum file
cat .version_checksum

# Manually update after cleanup
rm .version_checksum
scripts/version_manager.sh save-checksum
```

### Disable auto-increment
Edit `Makefile` and comment out the `pre_build.sh` invocation if needed.

---

**Implementation Date:** January 29, 2026  
**Version System:** Semantic Versioning (X.Y.Z)  
**Patch Increment Trigger:** Any source code change in `src/` directory

---

## Release Tagging Policy (added 2026-05-03)

### Why the scheme was changed

Tags `1.0` … `1.16` (and `1.16-rebased-on-1.3.3`) were created with bare
integer minor numbers (`1.10`, `1.11`, …) while upstream OpenGene/fastp uses
SemVer triplets with a `v` prefix (`v1.3.0`, `v1.3.3`). The two schemes
collide numerically — `git tag --sort=v:refname` sorts `v1.3.3` *before*
`1.16` (treating it as `1.3.3 < 1.16`), and a reader could mistake fork tag
`1.16` for "newer than upstream v1.3.3", when in fact `1.16` was based on
upstream `v1.2.2`.

### New scheme

All fork release tags from `v1.3.3-fastp-d0bromir.1` onward follow:

```
v<UPSTREAM_BASELINE>-fastp-d0bromir.<N>
```

- `<UPSTREAM_BASELINE>` — the OpenGene/fastp tag this release was rebased
  onto (e.g. `1.3.3`, `1.3.4`).
- `<N>` — fork iteration on top of that baseline, starting at `1`.
- Format is valid SemVer pre-release syntax, sorts correctly, and never
  overlaps with upstream `vX.Y.Z` tags.

Examples:
- `v1.3.3-fastp-d0bromir.1` — first fork release on top of upstream v1.3.3.
- `v1.3.3-fastp-d0bromir.2` — second fork-only fix without re-rebasing.
- `v1.3.4-fastp-d0bromir.1` — first release after rebasing onto upstream v1.3.4.

`FASTP_VER` in `src/common.h` follows the same scheme but without the `v`
prefix, e.g. `1.3.3-d0bromir`.

### Legacy tags

Tags `1.0` … `1.16` and `1.16-rebased-on-1.3.3` are kept **as-is** for
historical compatibility. They are immutable. Do not delete or move them.
`v1.3.3-fastp-d0bromir.1` is a bridge tag pointing at the same commit as
`1.16-rebased-on-1.3.3`.

### Helper

```bash
scripts/version_manager.sh release-tag-name 1
# → v1.3.3-fastp-d0bromir.1
```

