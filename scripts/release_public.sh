#!/usr/bin/env bash
# release_public.sh — Cut paired GitHub releases on the public repo
# (d0bromir/fastp-gpu, Zenodo-friendly) and the work repo (d0bromir/fastp,
# private rebase mirror) so the two stay in sync. Reusable: re-run any
# time a new tag is needed.
#
# What it does:
#   1. Reads the fastp version from scripts/version_manager.sh.
#   2. Computes the next release tag as
#        v<baseline>-fastp-d0bromir.<N>
#      where <baseline> is FASTP_VER with any trailing '-d0bromir'
#      stripped (e.g. 1.3.3) and <N> is the next free integer across the
#      work repo's local tags AND the public repo's existing releases.
#      Both repos receive the same tag, so the upstream baseline stays
#      pinned (e.g. v1.3.3) and only the d0bromir.<N> iteration bumps.
#      Override via --tag (public) and/or --work-tag (work repo).
#   3. Builds clean CPU and GPU binaries via scripts/build_all.sh
#      (each build runs its own unit tests; failure aborts).
#   4. Refreshes the scratch clone of the public repo, tags its current
#      HEAD with <tag>, pushes the tag.
#   5. Tags the work repo's current HEAD with <work-tag>, pushes the tag.
#   6. Computes SHA256SUMS for the binaries.
#   7. Creates the GitHub release on EACH repo via `gh release create`,
#      uploading the same asset set:
#         - fastp        (GPU build)
#         - fastp-cpu    (CPU build)
#         - SHA256SUMS
#         - LICENSE
#      Release notes are generic and reusable. They contain no
#      paper-specific text. Override with --notes-file or --notes.
#
# Usage:
#   scripts/release_public.sh                       # both repos, auto tag
#                                                   #   (next d0bromir.<N>)
#   scripts/release_public.sh --no-work-repo        # public repo only
#   scripts/release_public.sh --no-public-repo      # work repo only
#   scripts/release_public.sh --tag v1.3.4-fastp-d0bromir.1 \
#       --work-tag v1.3.4-fastp-d0bromir.1          # custom tags
#   scripts/release_public.sh --title "1.3.4 hotfix"
#   scripts/release_public.sh --notes-file NOTES.md
#   scripts/release_public.sh --notes "One-line note"
#   scripts/release_public.sh --draft               # create as draft
#   scripts/release_public.sh --prerelease          # mark as prerelease
#   scripts/release_public.sh --dry-run             # build + stage, no push
#   scripts/release_public.sh --skip-build          # use existing
#                                                   #   ./fastp + ./fastp-cpu
#
# Env overrides:
#   PUBLIC_REMOTE_URL    default: https://github.com/d0bromir/fastp-gpu.git
#   PUBLIC_REPO          default: d0bromir/fastp-gpu  (passed to `gh -R`)
#   PUBLIC_BRANCH        default: master
#   WORK_REPO            default: d0bromir/fastp      (passed to `gh -R`)
#   WORK_REMOTE          default: origin              (in the work repo)
#   SCRATCH_DIR          default: <repo>/.publish-scratch
#   STAGING_DIR          default: <repo>/.release-staging
#
# Requirements: gh (authenticated), git, sha256sum, scripts/build_all.sh.
# The release tag is created on the *public* repo's HEAD, so make sure
# the desired snapshot has already been published via
# scripts/publish_public.sh and validated via scripts/verify_public_publish.sh
# before running this script.

set -euo pipefail

REMOTE_URL="${PUBLIC_REMOTE_URL:-https://github.com/d0bromir/fastp-gpu.git}"
PUBLIC_REPO="${PUBLIC_REPO:-d0bromir/fastp-gpu}"
BRANCH="${PUBLIC_BRANCH:-master}"
WORK_REPO="${WORK_REPO:-d0bromir/fastp}"
WORK_REMOTE="${WORK_REMOTE:-origin}"

TAG=""
WORK_TAG=""
TITLE=""
NOTES=""
NOTES_FILE=""
DRAFT=0
PRERELEASE=0
DRY_RUN=0
SKIP_BUILD=0
DO_PUBLIC=1
DO_WORK=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)         TAG="${2:?--tag needs an argument}"; shift 2 ;;
        --tag=*)       TAG="${1#--tag=}"; shift ;;
        --work-tag)    WORK_TAG="${2:?--work-tag needs an argument}"; shift 2 ;;
        --work-tag=*)  WORK_TAG="${1#--work-tag=}"; shift ;;
        --title)       TITLE="${2:?--title needs an argument}"; shift 2 ;;
        --title=*)     TITLE="${1#--title=}"; shift ;;
        --notes)       NOTES="${2:?--notes needs an argument}"; shift 2 ;;
        --notes=*)     NOTES="${1#--notes=}"; shift ;;
        --notes-file)  NOTES_FILE="${2:?--notes-file needs an argument}"; shift 2 ;;
        --notes-file=*) NOTES_FILE="${1#--notes-file=}"; shift ;;
        --draft)       DRAFT=1; shift ;;
        --prerelease)  PRERELEASE=1; shift ;;
        --dry-run)     DRY_RUN=1; shift ;;
        --skip-build)  SKIP_BUILD=1; shift ;;
        --no-work-repo)   DO_WORK=0; shift ;;
        --no-public-repo) DO_PUBLIC=0; shift ;;
        -h|--help)
            sed -n '2,50p' "$0"
            exit 0
            ;;
        *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

SCRATCH_DIR="${SCRATCH_DIR:-$REPO_ROOT/.publish-scratch}"
PUBLIC_DIR="$SCRATCH_DIR/fastp-gpu"
STAGING_DIR="${STAGING_DIR:-$REPO_ROOT/.release-staging}"

step()  { echo ""; echo ">>> $*"; }
ok()    { echo "    [OK] $*"; }
fail()  { echo "    [FAIL] $*" >&2; exit 1; }

command -v gh >/dev/null         || fail "gh CLI not found"
command -v sha256sum >/dev/null  || fail "sha256sum not found"
gh auth status >/dev/null 2>&1   || fail "gh is not authenticated (run: gh auth login)"

# ---------- version + tag ------------------------------------------------
VERSION="$(bash scripts/version_manager.sh get 2>/dev/null || echo unknown)"
[[ "$VERSION" == "unknown" ]] && fail "could not read version from src/common.h"
BASELINE="${VERSION%-d0bromir}"
TAG_PREFIX="v${BASELINE}-fastp-d0bromir."

# Auto-compute the next iteration N so the upstream baseline stays pinned
# (e.g. v1.3.3) and only the d0bromir.<N> suffix increments. We consult
# both the work repo's tags and the public repo's tags so the next N is
# the global next on either side. Honours --tag / --work-tag overrides.
compute_next_iteration() {
    local prefix="$1" next=1 t n
    # Local work-repo tags.
    while IFS= read -r t; do
        [[ -z "$t" ]] && continue
        n="${t#$prefix}"
        [[ "$n" =~ ^[0-9]+$ ]] || continue
        (( n + 1 > next )) && next=$((n + 1))
    done < <(git tag --list "${prefix}*" 2>/dev/null)
    # Public-repo tags via gh (does not need the scratch clone yet).
    if (( DO_PUBLIC == 1 )) && command -v gh >/dev/null; then
        while IFS= read -r t; do
            [[ -z "$t" ]] && continue
            n="${t#$prefix}"
            [[ "$n" =~ ^[0-9]+$ ]] || continue
            (( n + 1 > next )) && next=$((n + 1))
        done < <(gh release list -R "$PUBLIC_REPO" --limit 100 \
                    --json tagName -q '.[].tagName' 2>/dev/null)
    fi
    echo "$next"
}

if [[ -z "$TAG" || -z "$WORK_TAG" ]]; then
    git fetch --tags "$WORK_REMOTE" >/dev/null 2>&1 || true
    NEXT_N="$(compute_next_iteration "$TAG_PREFIX")"
    DEFAULT_TAG="${TAG_PREFIX}${NEXT_N}"
    [[ -z "$TAG" ]]      && TAG="$DEFAULT_TAG"
    [[ -z "$WORK_TAG" ]] && WORK_TAG="$DEFAULT_TAG"
fi
# Note: TITLE intentionally left empty when not supplied, so each repo's
# release defaults to its own tag (public TAG vs work WORK_TAG).

# ---------- build binaries ----------------------------------------------
if (( SKIP_BUILD == 0 )); then
    step "Building CPU and GPU binaries via scripts/build_all.sh"
    bash scripts/build_all.sh cpu
    bash scripts/build_all.sh gpu
else
    echo "    [SKIP] --skip-build: using existing ./fastp-cpu and ./fastp"
fi

CPU_BIN="$REPO_ROOT/fastp-cpu"
GPU_BIN="$REPO_ROOT/fastp"
[[ -x "$CPU_BIN" ]] || fail "CPU binary missing: $CPU_BIN"
[[ -x "$GPU_BIN" ]] || fail "GPU binary missing: $GPU_BIN"

# ---------- stage assets -------------------------------------------------
step "Staging release assets in $STAGING_DIR"
rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"
cp -p "$GPU_BIN" "$STAGING_DIR/fastp"
cp -p "$CPU_BIN" "$STAGING_DIR/fastp-cpu"
cp -p "$REPO_ROOT/LICENSE" "$STAGING_DIR/LICENSE"
( cd "$STAGING_DIR" && sha256sum fastp fastp-cpu > SHA256SUMS )
ok "staged: $(ls -la "$STAGING_DIR" | tail -n +2)"

# ---------- refresh public clone (needed to know its HEAD for notes) ----
PUBLIC_SHA=""
PUBLIC_SHA_SHORT=""
if (( DO_PUBLIC == 1 )); then
    step "Refreshing public clone $PUBLIC_DIR"
    mkdir -p "$SCRATCH_DIR"
    if [[ -d "$PUBLIC_DIR/.git" ]]; then
        git -C "$PUBLIC_DIR" remote set-url origin "$REMOTE_URL"
        git -C "$PUBLIC_DIR" fetch --tags --prune origin
    else
        git clone "$REMOTE_URL" "$PUBLIC_DIR"
    fi
    git -C "$PUBLIC_DIR" checkout -q "$BRANCH"
    git -C "$PUBLIC_DIR" reset -q --hard "origin/$BRANCH"
    PUBLIC_SHA="$(git -C "$PUBLIC_DIR" rev-parse HEAD)"
    PUBLIC_SHA_SHORT="$(git -C "$PUBLIC_DIR" rev-parse --short HEAD)"
    ok "public HEAD: $PUBLIC_SHA_SHORT"
fi

# ---------- release notes -----------------------------------------------
HOST_INFO="$(uname -srm 2>/dev/null || uname -a)"
GPU_VER_LINE="$($GPU_BIN --version 2>&1 | head -1 || true)"
NOTES_RENDERED="$STAGING_DIR/RELEASE_NOTES.md"
if [[ -n "$NOTES_FILE" ]]; then
    [[ -f "$NOTES_FILE" ]] || fail "--notes-file does not exist: $NOTES_FILE"
    cp "$NOTES_FILE" "$NOTES_RENDERED"
elif [[ -n "$NOTES" ]]; then
    printf '%s\n' "$NOTES" > "$NOTES_RENDERED"
else
    # Provenance line points at whichever repo is the published source.
    if (( DO_PUBLIC == 1 )); then
        SRC_LINE="- Source: https://github.com/$PUBLIC_REPO at \`$PUBLIC_SHA_SHORT\`"
    else
        SRC_LINE="- Source: https://github.com/$WORK_REPO at \`$(git rev-parse --short HEAD)\`"
    fi
    cat > "$NOTES_RENDERED" <<EOF
## fastp-gpu ${TAG:-$WORK_TAG}

GPU-accelerated fork of [OpenGene/fastp](https://github.com/OpenGene/fastp).
Drop-in replacement for upstream \`fastp\` with optional CUDA acceleration
of the per-read trimming, filtering, and statistics kernels.

### Binaries (Linux x86_64 / aarch64, dynamically linked)
- \`fastp\` — GPU build, CUDA + libisal + libdeflate.
- \`fastp-cpu\` — CPU-only build (no CUDA dependency).

Both binaries report \`$VERSION\` for \`--version\`.

### Verification
SHA256 checksums of the binaries are in \`SHA256SUMS\`. To verify after
download:

\`\`\`
sha256sum -c SHA256SUMS
\`\`\`

### Build provenance
$SRC_LINE
- Build host: \`$HOST_INFO\`
- Built: $(date -u +%Y-%m-%dT%H:%M:%SZ)

### Citing this release
This release is archived on Zenodo. Use the DOI shown in the Zenodo
sidebar of this repository to cite a specific version.
EOF
fi

# ---------- shared assets list ------------------------------------------
ASSETS=(
    "$STAGING_DIR/fastp"
    "$STAGING_DIR/fastp-cpu"
    "$STAGING_DIR/SHA256SUMS"
    "$STAGING_DIR/LICENSE"
)

# release_to_repo <repo> <tag> <title> <git_dir> <remote> <target_sha>
#   - <git_dir>     local git checkout to tag (work tree or scratch clone)
#   - <remote>      remote name to push the tag to (within <git_dir>)
#   - <target_sha>  full SHA in <git_dir> to attach the tag to
release_to_repo() {
    local repo="$1" tag="$2" title="$3" gdir="$4" remote="$5" target="$6"
    local target_short
    target_short="$(git -C "$gdir" rev-parse --short "$target")"

    step "Release plan ($repo)"
    echo "    repo        : $repo"
    echo "    tag         : $tag (on $target_short)"
    echo "    title       : $title"
    echo "    draft       : $([[ $DRAFT -eq 1 ]] && echo yes || echo no)"
    echo "    prerelease  : $([[ $PRERELEASE -eq 1 ]] && echo yes || echo no)"
    echo "    binary ver  : $GPU_VER_LINE"
    echo "    assets:"
    printf '      %s\n' "${ASSETS[@]}"

    # Tag-existence guard.
    if git -C "$gdir" rev-parse "refs/tags/$tag" >/dev/null 2>&1; then
        local existing
        existing="$(git -C "$gdir" rev-list -n1 "$tag")"
        if [[ "$existing" != "$target" ]]; then
            fail "tag $tag already exists in $gdir at $existing (target is $target). Refusing to move it."
        fi
        ok "tag $tag already at target — reusing"
    else
        if (( DRY_RUN == 0 )); then
            git -C "$gdir" \
                -c user.name='d0bromir-publish-bot' \
                -c user.email='publish-bot@d0bromir.invalid' \
                tag -a "$tag" -m "Release $tag" "$target"
            git -C "$gdir" push "$remote" "refs/tags/$tag"
            ok "tagged $tag on $target_short and pushed to $remote"
        else
            echo "    [DRY-RUN] would tag $target_short as $tag and push to $remote"
        fi
    fi

    if (( DRY_RUN == 1 )); then
        echo "    [DRY-RUN] would create release $tag on $repo"
        return 0
    fi

    # Release-existence guard.
    if gh release view "$tag" -R "$repo" >/dev/null 2>&1; then
        fail "release $tag already exists on $repo. Delete it first or pick a new tag."
    fi

    local gh_flags=(-R "$repo" --title "$title" --notes-file "$NOTES_RENDERED")
    (( DRAFT == 1 ))      && gh_flags+=(--draft)
    (( PRERELEASE == 1 )) && gh_flags+=(--prerelease)

    step "Creating GitHub release $tag on $repo"
    gh release create "$tag" "${ASSETS[@]}" "${gh_flags[@]}"
    local rel_url
    rel_url="$(gh release view "$tag" -R "$repo" --json url -q .url)"
    ok "$rel_url"
}

echo
echo "    notes:"
sed 's/^/      /' "$NOTES_RENDERED"

if (( DO_PUBLIC == 1 )); then
    release_to_repo "$PUBLIC_REPO" "$TAG" "${TITLE:-$TAG}" \
        "$PUBLIC_DIR" origin "$PUBLIC_SHA"
fi
if (( DO_WORK == 1 )); then
    WORK_SHA="$(git rev-parse HEAD)"
    release_to_repo "$WORK_REPO" "$WORK_TAG" "${TITLE:-$WORK_TAG}" \
        "$REPO_ROOT" "$WORK_REMOTE" "$WORK_SHA"
fi

if (( DRY_RUN == 1 )); then
    echo
    echo "Dry-run complete. Re-run without --dry-run to create the release(s)."
    exit 0
fi

echo
echo "=========================================================="
echo "  Release(s) published"
(( DO_PUBLIC == 1 )) && echo "    public : $PUBLIC_REPO @ $TAG"
(( DO_WORK == 1 ))   && echo "    work   : $WORK_REPO @ $WORK_TAG"
echo "=========================================================="
echo
if (( DO_PUBLIC == 1 )); then
    echo "If Zenodo integration is enabled on $PUBLIC_REPO, the new tag will"
    echo "be archived automatically and a DOI will appear on the Zenodo entry."
fi
