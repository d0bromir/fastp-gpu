#!/usr/bin/env bash
# =============================================================================
# run_tests.sh — Canonical test runner for fastp_d0bromir
#
# Two test tiers, both mandatory (hard-fail on mismatch):
#
#   1. Built-in unit tests (inherited from upstream OpenGene), invoked via
#      `<binary> test`.  Run for every binary present:
#        - ../fastp_opengene/fastp   (if present; --skip-opengene to skip)
#        - ./fastp-cpu              (if present)
#        - ./fastp                   (if present)
#
#   2. CPU↔GPU regression on testdata/R{1,2}.fq:
#      runs both d0bromir CPU and GPU binaries on the bundled testdata and
#      verifies that the decompressed output FASTQ md5 plus a fixed list of
#      JSON biology fields are byte-equal between the two builds.  Skipped
#      automatically if either ./fastp-cpu or ./fastp is missing.
#
# Exit codes:
#    0 — all tests passed (or were legitimately skipped)
#    1 — a binary's built-in unit tests failed
#    2 — CPU↔GPU regression mismatch
#    3 — usage / environment error
#
# Usage:
#   scripts/run_tests.sh                  # all tiers, all available binaries
#   scripts/run_tests.sh --unit-only      # tier 1 only
#   scripts/run_tests.sh --regression-only# tier 2 only
#   scripts/run_tests.sh --skip-opengene  # don't try ../fastp_opengene/fastp
#   scripts/run_tests.sh -q               # less verbose
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OPENGENE_BIN="${OPENGENE_BIN:-$(cd "$PROJ_DIR/.." && pwd)/fastp_opengene/fastp}"
CPU_BIN="${CPU_BIN:-$PROJ_DIR/fastp-cpu}"
GPU_BIN="${GPU_BIN:-$PROJ_DIR/fastp}"

TESTDATA_R1="$PROJ_DIR/testdata/R1.fq"
TESTDATA_R2="$PROJ_DIR/testdata/R2.fq"

DO_UNIT=1
DO_REGRESSION=1
SKIP_OPENGENE=0
QUIET=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --unit-only)        DO_REGRESSION=0; shift ;;
        --regression-only)  DO_UNIT=0;       shift ;;
        --skip-opengene)    SKIP_OPENGENE=1; shift ;;
        -q|--quiet)         QUIET=1;         shift ;;
        -h|--help)
            sed -n '2,32p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 3 ;;
    esac
done

# Same JSON biology fields validated by run_benchmark.sh deep-validation.
VALIDATE_JSON_FIELDS=(
    summary.before_filtering.total_reads
    summary.before_filtering.total_bases
    summary.after_filtering.total_reads
    summary.after_filtering.total_bases
    summary.after_filtering.q20_rate
    summary.after_filtering.q30_rate
    filtering_result.passed_filter_reads
    filtering_result.low_quality_reads
    filtering_result.too_many_N_reads
    filtering_result.too_short_reads
    adapter_cutting.adapter_trimmed_reads
    adapter_cutting.adapter_trimmed_bases
)

step()  { echo ""; echo ">>> $*"; }
ok()    { echo "    [OK] $*"; }
info()  { (( QUIET )) || echo "    $*"; }
fail()  { echo "    [FAIL] $*" >&2; }

# ---------------------------------------------------------------------------
# Tier 1 — built-in unit tests
# ---------------------------------------------------------------------------
unit_run_one() {
    local label="$1" bin="$2"
    if [[ ! -x "$bin" ]]; then
        info "[skip] $label — binary not present ($bin)"
        return 0
    fi
    info "[unit] $label  ($bin)"
    local out rc=0
    out=$("$bin" test 2>&1) || rc=$?
    if (( rc != 0 )); then
        echo "$out"
        fail "$label: built-in unit tests FAILED (exit $rc)"
        return 1
    fi
    if echo "$out" | grep -qiE '^FAILED|: failed'; then
        echo "$out"
        fail "$label: built-in unit tests reported failures"
        return 1
    fi
    ok "$label: built-in unit tests passed"
    return 0
}

run_unit_tier() {
    step "Tier 1 — built-in unit tests"
    local rc=0
    if (( ! SKIP_OPENGENE )); then
        unit_run_one "opengene"     "$OPENGENE_BIN" || rc=1
    fi
    unit_run_one "d0bromir_cpu" "$CPU_BIN"      || rc=1
    unit_run_one "d0bromir_gpu" "$GPU_BIN"      || rc=1
    return $rc
}

# ---------------------------------------------------------------------------
# Tier 2 — CPU↔GPU regression on testdata/
# ---------------------------------------------------------------------------
fq_md5() {
    [[ -f "$1" ]] || { echo ""; return; }
    gunzip -c "$1" 2>/dev/null | md5sum | awk '{print $1}'
}

json_field_get() {
    python3 - "$1" "$2" <<'PY' 2>/dev/null || true
import json, sys
f, p = sys.argv[1], sys.argv[2]
d = json.load(open(f))
for k in p.split('.'):
    if isinstance(d, dict) and k in d:
        d = d[k]
    else:
        sys.exit(0)
if isinstance(d, float):
    print(f"{d:.6f}")
else:
    print(d)
PY
}

# Run a binary on testdata, write output to caller-supplied paths.
# Args: bin, json_out, fq1_out, fq2_out
run_on_testdata() {
    local bin="$1" json="$2" fq1="$3" fq2="$4"
    "$bin" \
        -i "$TESTDATA_R1" -I "$TESTDATA_R2" \
        -o "$fq1"        -O "$fq2" \
        -j "$json"       -h /dev/null \
        --thread 2 \
        >/dev/null 2>&1
}

run_regression_tier() {
    step "Tier 2 — CPU↔GPU regression on testdata/"
    if [[ ! -x "$CPU_BIN" || ! -x "$GPU_BIN" ]]; then
        info "[skip] need both fastp-cpu and fastp; skipping regression"
        return 0
    fi
    if [[ ! -f "$TESTDATA_R1" || ! -f "$TESTDATA_R2" ]]; then
        fail "testdata missing: $TESTDATA_R1 / $TESTDATA_R2"
        return 2
    fi

    local tmp; tmp=$(mktemp -d /tmp/fastp_test_XXXXXX)
    trap "rm -rf '$tmp'" RETURN

    local cpu_json="$tmp/cpu.json" cpu_r1="$tmp/cpu_R1.fq.gz" cpu_r2="$tmp/cpu_R2.fq.gz"
    local gpu_json="$tmp/gpu.json" gpu_r1="$tmp/gpu_R1.fq.gz" gpu_r2="$tmp/gpu_R2.fq.gz"

    info "[regression] running CPU build on testdata..."
    run_on_testdata "$CPU_BIN" "$cpu_json" "$cpu_r1" "$cpu_r2" \
        || { fail "CPU build failed on testdata"; return 2; }

    info "[regression] running GPU build on testdata..."
    run_on_testdata "$GPU_BIN" "$gpu_json" "$gpu_r1" "$gpu_r2" \
        || { fail "GPU build failed on testdata"; return 2; }

    # FASTQ md5 (decompressed)
    local cpu_md5_1 gpu_md5_1 cpu_md5_2 gpu_md5_2
    cpu_md5_1=$(fq_md5 "$cpu_r1"); gpu_md5_1=$(fq_md5 "$gpu_r1")
    cpu_md5_2=$(fq_md5 "$cpu_r2"); gpu_md5_2=$(fq_md5 "$gpu_r2")

    if [[ -z "$cpu_md5_1" || "$cpu_md5_1" != "$gpu_md5_1" ]]; then
        fail "R1 FASTQ md5 mismatch (cpu=$cpu_md5_1 gpu=$gpu_md5_1)"
        echo "    diagnostics retained at: $tmp"
        trap - RETURN
        return 2
    fi
    if [[ "$cpu_md5_2" != "$gpu_md5_2" ]]; then
        fail "R2 FASTQ md5 mismatch (cpu=$cpu_md5_2 gpu=$gpu_md5_2)"
        echo "    diagnostics retained at: $tmp"
        trap - RETURN
        return 2
    fi

    # JSON biology fields
    local mismatches=""
    for field in "${VALIDATE_JSON_FIELDS[@]}"; do
        local cpu_v gpu_v
        cpu_v=$(json_field_get "$cpu_json" "$field")
        gpu_v=$(json_field_get "$gpu_json" "$field")
        [[ -z "$cpu_v" && -z "$gpu_v" ]] && continue
        if [[ "$cpu_v" != "$gpu_v" ]]; then
            mismatches+=$'\n'"      $field: cpu=$cpu_v  gpu=$gpu_v"
        fi
    done
    if [[ -n "$mismatches" ]]; then
        fail "JSON biology fields differ:$mismatches"
        echo "    diagnostics retained at: $tmp"
        trap - RETURN
        return 2
    fi

    ok "CPU↔GPU regression: md5(R1)=$cpu_md5_1  md5(R2)=$cpu_md5_2  + ${#VALIDATE_JSON_FIELDS[@]} JSON fields equal"
    return 0
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo "============================================================"
echo " fastp_d0bromir test suite"
echo " Project: $PROJ_DIR"
echo "============================================================"

rc=0
if (( DO_UNIT )); then
    run_unit_tier || rc=1
fi
if (( DO_REGRESSION )); then
    if ! run_regression_tier; then
        # propagate exact code: 2 from regression tier
        rc=2
    fi
fi

echo ""
echo "============================================================"
if (( rc == 0 )); then
    echo " ALL TESTS PASSED"
else
    echo " TEST FAILURE (exit $rc)"
fi
echo "============================================================"
exit $rc
