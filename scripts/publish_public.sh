#!/usr/bin/env bash
# publish_public.sh — Publish a curated snapshot of this work repo to the
# public d0bromir/fastp-gpu GitHub repository.
#
# This script does NOT mirror the work repo's commit history. Instead it
# builds a clean snapshot of the source / build / test tree (plus the
# publication tables and figures) and commits it to the public repo as a
# single "Sync" commit. The public repo therefore contains only the
# information we want to release, while the work repo's private commit
# log stays private.
#
# What is included (whitelist, see PUBLISH_PATHS below):
#   - Top-level build files: Makefile, README.md, LICENSE, .gitignore
#   - .github/workflows/                 (CI definitions only)
#   - src/                               (full source)
#   - scripts/                           (build / test / benchmark scripts)
#   - testdata/                          (test fixtures)
#   - docs/*.md                          (top-level user docs:
#                                         BUILD_WITH_CUDA, ARCHITECTURE, etc.)
#   - docs/publication/figures/          (paper figures + their CSVs +
#                                         plot scripts)
#   - docs/publication/tables/           (paper tables, CSVs)
#   - benchmark_results/                 (raw + canonical benchmark CSVs
#                                         and per-run logs)
#
# What is NOT published:
#   - .git history of the work repo (private rebase mirror)
#   - .github/copilot-instructions.md    (internal AI tooling config)
#   - docs/publication/*.tex, *.pdf      (paper sources / built PDFs)
#   - docs/publication/paper_source/     (manuscript drafts)
#   - docs/publication/required_changes/ (internal review notes)
#   - scripts/build_paper.sh,            (paper / manuscript build
#     scripts/build_application_notes.sh, scripts — internal only)
#     scripts/build_st_sim_concept.sh,
#     scripts/build_tus_paper.sh
#   - obj/, release-artifact/, built binaries
#
# Usage:
#   scripts/publish_public.sh                 # build snapshot, commit, push
#   scripts/publish_public.sh --dry-run       # build snapshot, show diff,
#                                             #   skip the push
#   scripts/publish_public.sh --no-push       # build + commit, skip the push
#                                             #   (leaves scratch repo for
#                                             #    inspection)
#   scripts/publish_public.sh --force         # force-with-lease push
#                                             #   (needed once at first
#                                             #    publish to overwrite
#                                             #    the existing remote tree)
#   scripts/publish_public.sh --rewrite-history
#                                             # wipe public commit history
#                                             #   entirely and replace it
#                                             #   with a single orphan
#                                             #   commit (implies --force).
#                                             #   Use to scrub previously
#                                             #   published files (e.g.
#                                             #   leaked paper scripts)
#                                             #   without a trace.
#   scripts/publish_public.sh --branch <name> # publish to a non-default
#                                             #   public branch
#   scripts/publish_public.sh --message "..." # custom commit message
#
# Fork-relationship behaviour:
#   d0bromir/fastp-gpu is a GitHub fork of OpenGene/fastp. To keep the
#   fork from showing as "N commits behind", every publish rebases the
#   public branch onto OpenGene/fastp:master and applies the d0bromir
#   snapshot as a single sync commit on top. The public history is
#   therefore: upstream/master + 1 sync commit. This requires a force
#   push (the public branch is rebuilt each time), which is safe because
#   the public repo's commit log carries no information beyond the
#   latest sync commit. Disable the rebase with --no-rebase-upstream.
#
# Env overrides (rarely needed):
#   PUBLIC_REMOTE_URL    default: https://github.com/d0bromir/fastp-gpu.git
#   PUBLIC_BRANCH        default: master
#   UPSTREAM_REMOTE_URL  default: https://github.com/OpenGene/fastp.git
#   UPSTREAM_BRANCH      default: master
#   SCRATCH_DIR          default: <repo>/.publish-scratch
#   ALLOW_DIRTY=1        skip the work-repo clean-tree check (NOT recommended)

set -euo pipefail

REMOTE_URL="${PUBLIC_REMOTE_URL:-https://github.com/d0bromir/fastp-gpu.git}"
BRANCH="${PUBLIC_BRANCH:-master}"
UPSTREAM_URL="${UPSTREAM_REMOTE_URL:-https://github.com/OpenGene/fastp.git}"
UPSTREAM_BRANCH="${UPSTREAM_BRANCH:-master}"
DRY_RUN=0
DO_PUSH=1
FORCE=0
REWRITE_HISTORY=0
REBASE_UPSTREAM=1
COMMIT_MSG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY_RUN=1; DO_PUSH=0; shift ;;
        --no-push)    DO_PUSH=0; shift ;;
        --force)      FORCE=1; shift ;;
        --rewrite-history) REWRITE_HISTORY=1; FORCE=1; shift ;;
        --no-rebase-upstream) REBASE_UPSTREAM=0; shift ;;
        --branch)     BRANCH="${2:?--branch needs an argument}"; shift 2 ;;
        --branch=*)   BRANCH="${1#--branch=}"; shift ;;
        --message)    COMMIT_MSG="${2:?--message needs an argument}"; shift 2 ;;
        --message=*)  COMMIT_MSG="${1#--message=}"; shift ;;
        -h|--help)
            sed -n '2,55p' "$0"
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

SCRATCH_DIR="${SCRATCH_DIR:-$REPO_ROOT/.publish-scratch}"
PUBLIC_DIR="$SCRATCH_DIR/fastp-gpu"

# ---------- whitelist ----------------------------------------------------
# Paths are interpreted relative to REPO_ROOT. Directories are copied
# recursively (without their own .git, if any). Files are copied as-is.
PUBLISH_PATHS=(
    Makefile
    README.md
    LICENSE
    .gitignore
    .github/workflows
    src
    scripts
    testdata
    docs/ARCHITECTURE.md
    docs/BUILD_WITH_CUDA.md
    docs/CUDA_ACCELERATION.md
    docs/INDEX.md
    docs/PERFORMANCE_SUMMARY.md
    docs/QUICK_REFERENCE.md
    docs/UPSTREAM_REJECTED.md
    docs/VERSION_MANAGEMENT.md
    docs/publication/figures
    docs/publication/tables
    benchmark_results
)

# Patterns excluded even when they live under a whitelisted directory.
RSYNC_EXCLUDES=(
    --exclude='__pycache__'
    --exclude='*.pyc'
    --exclude='.DS_Store'
    --exclude='*.aux'
    --exclude='*.out'
    # Strip large intermediate artifacts that may end up under
    # benchmark_results/ during a run; canonical CSVs / JSONs / .log
    # text reports are kept.
    --exclude='*.fq.gz'
    --exclude='*.fastq.gz'
    --exclude='*.bam'
    --exclude='*.sam'
    # Paper / manuscript build scripts are NOT part of the public release.
    --exclude='build_paper.sh'
    --exclude='build_application_notes.sh'
    --exclude='build_st_sim_concept.sh'
    --exclude='build_tus_paper.sh'
)
# -------------------------------------------------------------------------

# Sanity: every whitelisted path must exist in the work repo.
missing=0
for p in "${PUBLISH_PATHS[@]}"; do
    if [[ ! -e "$REPO_ROOT/$p" ]]; then
        echo "ERROR: whitelisted path does not exist: $p" >&2
        missing=1
    fi
done
[[ $missing -eq 1 ]] && exit 4

# Working-tree clean check.
if [[ "${ALLOW_DIRTY:-0}" != "1" ]]; then
    if ! git diff --quiet --ignore-submodules HEAD -- 2>/dev/null; then
        echo "ERROR: work-repo working tree has uncommitted changes." >&2
        echo "       Commit, stash, or set ALLOW_DIRTY=1 to override." >&2
        git status --short >&2
        exit 3
    fi
fi

WORK_SHA="$(git rev-parse --short=12 HEAD)"
WORK_SHA_FULL="$(git rev-parse HEAD)"
VERSION="$(bash scripts/version_manager.sh get 2>/dev/null || echo unknown)"

if [[ -z "$COMMIT_MSG" ]]; then
    COMMIT_MSG="Sync from d0bromir/fastp @ ${WORK_SHA} (v${VERSION})"
fi

# ---------- prepare scratch clone ---------------------------------------
mkdir -p "$SCRATCH_DIR"

if [[ $REWRITE_HISTORY -eq 1 ]]; then
    # Wipe scratch entirely and re-init as a fresh, history-less repo.
    # Combined with --force, this overwrites the public repo with a
    # single orphan commit, eliminating prior commit history "without
    # a trace".
    echo "REWRITE_HISTORY: discarding any existing scratch clone at $PUBLIC_DIR ..."
    rm -rf "$PUBLIC_DIR"
    git init -q -b "$BRANCH" "$PUBLIC_DIR"
    git -C "$PUBLIC_DIR" remote add origin "$REMOTE_URL"
elif [[ -d "$PUBLIC_DIR/.git" ]]; then
    echo "Refreshing existing scratch clone at $PUBLIC_DIR ..."
    git -C "$PUBLIC_DIR" remote set-url origin "$REMOTE_URL"
    git -C "$PUBLIC_DIR" fetch --prune origin || {
        echo "NOTE: fetch failed (remote may be empty); continuing." >&2
    }
else
    echo "Cloning $REMOTE_URL into $PUBLIC_DIR ..."
    if ! git clone "$REMOTE_URL" "$PUBLIC_DIR" 2>/dev/null; then
        echo "Remote appears empty or unreachable; initializing a new local repo."
        rm -rf "$PUBLIC_DIR"
        git init -q -b "$BRANCH" "$PUBLIC_DIR"
        git -C "$PUBLIC_DIR" remote add origin "$REMOTE_URL"
    fi
fi

# Ensure we are on the requested branch in the scratch clone.
(
    cd "$PUBLIC_DIR"
    if [[ $REWRITE_HISTORY -eq 1 ]]; then
        # Fresh repo — branch was set by `git init -b`, nothing to do.
        :
    elif git show-ref --verify --quiet "refs/heads/$BRANCH"; then
        git checkout -q "$BRANCH"
        if git show-ref --verify --quiet "refs/remotes/origin/$BRANCH"; then
            git reset -q --hard "origin/$BRANCH"
        fi
    elif git show-ref --verify --quiet "refs/remotes/origin/$BRANCH"; then
        git checkout -q -b "$BRANCH" "origin/$BRANCH"
    else
        git checkout -q -b "$BRANCH"
    fi
)

# ---------- rebase onto upstream OpenGene/fastp -------------------------
# d0bromir/fastp-gpu is a GitHub fork of OpenGene/fastp. To prevent the
# "N commits behind" banner, reset the public branch to the tip of
# OpenGene/fastp:master before laying down our snapshot. Each publish
# therefore produces history = upstream/master + 1 sync commit, which
# requires force-push but keeps the fork in lockstep with upstream.
if [[ $REBASE_UPSTREAM -eq 1 ]]; then
    (
        cd "$PUBLIC_DIR"
        if git remote get-url upstream >/dev/null 2>&1; then
            git remote set-url upstream "$UPSTREAM_URL"
        else
            git remote add upstream "$UPSTREAM_URL"
        fi
        echo "Fetching upstream $UPSTREAM_URL ($UPSTREAM_BRANCH) ..."
        if ! git fetch --quiet --no-tags upstream "$UPSTREAM_BRANCH"; then
            echo "ERROR: could not fetch upstream/$UPSTREAM_BRANCH from $UPSTREAM_URL" >&2
            exit 5
        fi
        UPSTREAM_SHA="$(git rev-parse "upstream/$UPSTREAM_BRANCH")"
        echo "Resetting $BRANCH to upstream/$UPSTREAM_BRANCH @ ${UPSTREAM_SHA:0:12}"
        git reset -q --hard "upstream/$UPSTREAM_BRANCH"
    )
    # Rebasing onto upstream rewrites the public branch every time, so
    # the push must be a force push.
    FORCE=1
fi

# ---------- wipe working tree (preserve .git) ---------------------------
# Guarantees deletions in the work repo propagate to the public repo.
echo "Wiping working tree of scratch clone (preserving .git) ..."
find "$PUBLIC_DIR" -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +

# ---------- copy whitelisted paths --------------------------------------
echo "Copying whitelisted paths into scratch clone ..."
for p in "${PUBLISH_PATHS[@]}"; do
    src="$REPO_ROOT/$p"
    dst="$PUBLIC_DIR/$p"
    mkdir -p "$(dirname "$dst")"
    if [[ -d "$src" ]]; then
        rsync -a --delete "${RSYNC_EXCLUDES[@]}" "$src/" "$dst/"
    else
        cp -p "$src" "$dst"
    fi
done

# Sync metadata file recording provenance.
cat > "$PUBLIC_DIR/.publish-meta" <<EOF
# Auto-generated by scripts/publish_public.sh — do not edit by hand.
work_repo_sha   = $WORK_SHA_FULL
fastp_version   = $VERSION
generated_utc   = $(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

# ---------- stage and commit --------------------------------------------
cd "$PUBLIC_DIR"
git add -A

if git diff --cached --quiet; then
    echo
    echo "No changes vs current public/$BRANCH — nothing to publish."
    cd "$REPO_ROOT"
    exit 0
fi

echo
echo "=========================================================="
echo "  Publish plan"
echo "    work repo HEAD : $WORK_SHA_FULL"
echo "    fastp version  : $VERSION"
echo "    public remote  : $REMOTE_URL"
echo "    public branch  : $BRANCH"
if [[ $REBASE_UPSTREAM -eq 1 ]]; then
    echo "    rebased onto   : $UPSTREAM_URL ($UPSTREAM_BRANCH)"
else
    echo "    rebased onto   : (disabled via --no-rebase-upstream)"
fi
echo "    commit message : $COMMIT_MSG"
echo "    push           : $([[ $DO_PUSH -eq 1 ]] && echo yes || echo no)"
echo "    force          : $([[ $FORCE -eq 1 ]] && echo with-lease || echo no)"
echo "    scratch dir    : $PUBLIC_DIR"
echo "=========================================================="
echo
echo "File-level diff vs current public/$BRANCH:"
git --no-pager diff --cached --stat | tail -n 60
echo

# Use a bot identity for the snapshot commit so it doesn't pick up
# the user's personal git identity.
git -c user.name='d0bromir-publish-bot' \
    -c user.email='publish-bot@d0bromir.invalid' \
    commit -q -m "$COMMIT_MSG"

NEW_SHA="$(git rev-parse --short HEAD)"
echo "Created snapshot commit $NEW_SHA on branch $BRANCH."

# ---------- push --------------------------------------------------------
if [[ $DO_PUSH -eq 1 ]]; then
    PUSH_ARGS=(origin "$BRANCH:$BRANCH")
    if [[ $REWRITE_HISTORY -eq 1 || $REBASE_UPSTREAM -eq 1 ]]; then
        # In both cases the local branch was rebuilt from a base the
        # remote tracking ref does not know about, so --force-with-lease
        # would reject with "stale info". Use plain --force; both modes
        # are already gated by an explicit opt-in (or the documented
        # default that publish == upstream/master + sync commit).
        PUSH_ARGS=(--force "${PUSH_ARGS[@]}")
    elif [[ $FORCE -eq 1 ]]; then
        PUSH_ARGS=(--force-with-lease "${PUSH_ARGS[@]}")
    fi
    echo "+ git push ${PUSH_ARGS[*]}"
    git push "${PUSH_ARGS[@]}"
    echo
    echo "Published snapshot $NEW_SHA to $REMOTE_URL ($BRANCH)."
else
    echo
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "Dry-run: snapshot built and committed locally only."
    else
        echo "Skipped push (--no-push). Snapshot committed locally."
    fi
    echo "Inspect with:    git -C $PUBLIC_DIR log -1 --stat"
    echo "Push later with: git -C $PUBLIC_DIR push origin $BRANCH"
fi

cd "$REPO_ROOT"
