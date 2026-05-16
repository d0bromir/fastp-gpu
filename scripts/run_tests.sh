#!/usr/bin/env bash
# =============================================================================
# run_tests.sh — Canonical test runner for fastp_d0bromir
#
# Three test tiers, all mandatory by default (hard-fail on mismatch):
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
#   3. Clinical validation on testdata/R{1,2}.fq (requires fastp-cpu):
#      3a. Determinism       — two runs at T=2 produce identical decompressed
#                              FASTQ md5 and identical JSON biology fields.
#      3b. Cross-thread      — T=2 and T=4 produce identical JSON biology
#                              fields (filtering decisions must not depend on
#                              thread count).
#      3c. Version audit     — the binary's -v output contains the version
#                              string defined in src/common.h (FASTP_VER).
#      3d. Read conservation — total_reads_in == sum of all filtering_result
#                              bucket counts (no reads silently lost or
#                              duplicated).  Required by IEC 62304 §5.5 and
#                              IVDR 2017/746 Annex I §17 (data integrity).
#      3e. PE synchronization— R1 and R2 output files must contain exactly
#                              the same read count after paired-end filtering
#                              (broken pairs must not be emitted).
#      3f. Graceful failure  — supplying a non-FASTQ input file must produce
#                              a non-zero exit code (IEC 62304 §5.5.2
#                              defensive / error-path validation).
#      3g. PE count mismatch — paired-end input with R1 and R2 read counts
#                              that differ must produce a non-zero exit code
#                              (fixes H-03; IEC 62304 §5.5.2).
#      3h. Partial corruption— a FASTQ file with valid records followed by a
#                              malformed record must produce a non-zero exit
#                              code, not silently truncate (fixes H-11;
#                              IEC 62304 §5.5.2).
#      3i. Invalid quality   — a FASTQ record with a quality score outside the
#                              valid Phred+33 ASCII range (33–126) must produce
#                              a non-zero exit code (fixes H-09; H-09 code-
#                              level defence before SOUP library compression).
#
# Exit codes:
#    0 — all tests passed (or were legitimately skipped)
#    1 — a binary's built-in unit tests failed
#    2 — CPU↔GPU regression mismatch or clinical validation failure
#    3 — usage / environment error
#
# Usage:
#   scripts/run_tests.sh                  # all tiers, all available binaries
#   scripts/run_tests.sh --unit-only      # tier 1 only
#   scripts/run_tests.sh --regression-only# tier 2 only
#   scripts/run_tests.sh --clinical-only  # tier 3 only
#   scripts/run_tests.sh --skip-clinical  # tiers 1+2 only (fast CI)
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
DO_CLINICAL=1
SKIP_OPENGENE=0
QUIET=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --unit-only)        DO_REGRESSION=0; DO_CLINICAL=0; shift ;;
        --regression-only)  DO_UNIT=0;       DO_CLINICAL=0; shift ;;
        --clinical-only)    DO_UNIT=0;       DO_REGRESSION=0; shift ;;
        --skip-clinical)    DO_CLINICAL=0;                    shift ;;
        --skip-opengene)    SKIP_OPENGENE=1;                  shift ;;
        -q|--quiet)         QUIET=1;                          shift ;;
        -h|--help)
            sed -n '2,42p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 3 ;;
    esac
done

# Same JSON biology fields validated by run_benchmark.sh deep-validation,
# plus adapter_dimer_reads and too_long_reads which are present in the
# filtering_result object and relevant to the read-conservation equation.
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
    filtering_result.adapter_dimer_reads
    filtering_result.too_short_reads
    filtering_result.too_long_reads
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
    # Run from PROJ_DIR so that the binary's relative testdata/ paths resolve.
    out=$(cd "$PROJ_DIR" && "$bin" test 2>&1) || rc=$?
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

# Run a binary on testdata with an explicit thread count.
# Args: bin, threads, json_out, fq1_out, fq2_out
run_on_testdata_t() {
    local bin="$1" threads="$2" json="$3" fq1="$4" fq2="$5"
    "$bin" \
        -i "$TESTDATA_R1" -I "$TESTDATA_R2" \
        -o "$fq1"        -O "$fq2" \
        -j "$json"       -h /dev/null \
        --thread "$threads" \
        >/dev/null 2>&1
}

# Backwards-compat wrapper used by tier 2 (always 2 threads).
run_on_testdata() {
    run_on_testdata_t "$1" 2 "$2" "$3" "$4"
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
# Tier 3 — Clinical validation
#   3a. Determinism       — two runs at T=2 give identical md5 + JSON fields
#   3b. Cross-thread      — T=2 vs T=4 give identical JSON biology fields
#   3c. Version audit     — binary -v reports the FASTP_VER from src/common.h
# ---------------------------------------------------------------------------
json_fields_compare() {
    # Compare a named set of JSON fields between two files.
    # Args: label, file_a, label_a, file_b, label_b
    # Echoes mismatch lines; returns 1 if any mismatch found.
    local label="$1" fa="$2" la="$3" fb="$4" lb="$5"
    local mismatches="" av bv
    for field in "${VALIDATE_JSON_FIELDS[@]}"; do
        av=$(json_field_get "$fa" "$field")
        bv=$(json_field_get "$fb" "$field")
        [[ -z "$av" && -z "$bv" ]] && continue
        if [[ "$av" != "$bv" ]]; then
            mismatches+=$'\n'"      $field: $la=$av  $lb=$bv"
        fi
    done
    if [[ -n "$mismatches" ]]; then
        fail "$label: JSON fields differ:$mismatches"
        return 1
    fi
    return 0
}

run_clinical_tier() {
    step "Tier 3 — Clinical validation (determinism + cross-thread + version audit)"

    if [[ ! -x "$CPU_BIN" ]]; then
        info "[skip] need fastp-cpu; skipping clinical validation"
        return 0
    fi
    if [[ ! -f "$TESTDATA_R1" || ! -f "$TESTDATA_R2" ]]; then
        fail "testdata missing: $TESTDATA_R1 / $TESTDATA_R2"
        return 2
    fi

    local tmp; tmp=$(mktemp -d /tmp/fastp_clinical_XXXXXX)
    trap "rm -rf '$tmp'" RETURN
    local rc=0

    # ---- 3a: Determinism -----------------------------------------------
    info "[clinical-3a] Determinism: two runs at T=2, verify md5 + JSON identical..."
    run_on_testdata_t "$CPU_BIN" 2 "$tmp/da.json" "$tmp/da_R1.fq.gz" "$tmp/da_R2.fq.gz" \
        || { fail "clinical-3a: run A failed on testdata"; return 2; }
    run_on_testdata_t "$CPU_BIN" 2 "$tmp/db.json" "$tmp/db_R1.fq.gz" "$tmp/db_R2.fq.gz" \
        || { fail "clinical-3a: run B failed on testdata"; return 2; }

    local da_md5_1 db_md5_1 da_md5_2 db_md5_2
    da_md5_1=$(fq_md5 "$tmp/da_R1.fq.gz"); db_md5_1=$(fq_md5 "$tmp/db_R1.fq.gz")
    da_md5_2=$(fq_md5 "$tmp/da_R2.fq.gz"); db_md5_2=$(fq_md5 "$tmp/db_R2.fq.gz")

    if [[ -z "$da_md5_1" || "$da_md5_1" != "$db_md5_1" || "$da_md5_2" != "$db_md5_2" ]]; then
        fail "clinical-3a: FASTQ output is non-deterministic"
        fail "  R1: run_a=$da_md5_1  run_b=$db_md5_1"
        fail "  R2: run_a=$da_md5_2  run_b=$db_md5_2"
        echo "    diagnostics retained at: $tmp"; trap - RETURN
        rc=2
    else
        json_fields_compare "clinical-3a" "$tmp/da.json" "run_a" "$tmp/db.json" "run_b" \
            || { echo "    diagnostics retained at: $tmp"; trap - RETURN; rc=2; }
        (( rc == 0 )) && ok "clinical-3a: determinism — md5(R1)=$da_md5_1  md5(R2)=$da_md5_2  + ${#VALIDATE_JSON_FIELDS[@]} JSON fields stable"
    fi

    # ---- 3b: Cross-thread biological consistency -----------------------
    info "[clinical-3b] Cross-thread: T=2 vs T=4, verify JSON biology fields identical..."
    run_on_testdata_t "$CPU_BIN" 2 "$tmp/t2.json" "$tmp/t2_R1.fq.gz" "$tmp/t2_R2.fq.gz" \
        || { fail "clinical-3b: T=2 run failed"; return 2; }
    run_on_testdata_t "$CPU_BIN" 4 "$tmp/t4.json" "$tmp/t4_R1.fq.gz" "$tmp/t4_R2.fq.gz" \
        || { fail "clinical-3b: T=4 run failed"; return 2; }

    json_fields_compare "clinical-3b" "$tmp/t2.json" "T=2" "$tmp/t4.json" "T=4" \
        && ok "clinical-3b: cross-thread consistency — ${#VALIDATE_JSON_FIELDS[@]} JSON fields identical at T=2 and T=4" \
        || { echo "    diagnostics retained at: $tmp"; trap - RETURN; rc=2; }

    # ---- 3d: Read conservation (IEC 62304 §5.5 / IVDR Annex I §17) ----
    info "[clinical-3d] Read conservation: total_reads == sum of all filtering_result buckets..."
    # Reuse the T=2 run already in $tmp/t2.json (from 3b).
    local conservation_ok=0
    python3 - "$tmp/t2.json" <<'PY' && conservation_ok=1 || true
import json, sys
d = json.load(open(sys.argv[1]))
total = d['summary']['before_filtering']['total_reads']
fr = d['filtering_result']
accounted = sum(v for v in fr.values() if isinstance(v, int))
if total != accounted:
    print(f"    CONSERVATION FAIL: total_reads={total}, sum(filtering_result)={accounted}, diff={total - accounted}")
    sys.exit(1)
PY
    if (( conservation_ok )); then
        ok "clinical-3d: read conservation — filtering_result buckets account for all input reads"
    else
        fail "clinical-3d: read conservation equation FAILED (see output above)"
        echo "    diagnostics retained at: $tmp"; trap - RETURN
        rc=2
    fi

    # ---- 3e: PE synchronization ----------------------------------------
    info "[clinical-3e] PE synchronization: R1 and R2 output must have equal read count..."
    # Reuse the T=2 FASTQ outputs from 3b.
    local r1_lines r2_lines
    r1_lines=$(gunzip -c "$tmp/t2_R1.fq.gz" 2>/dev/null | wc -l)
    r2_lines=$(gunzip -c "$tmp/t2_R2.fq.gz" 2>/dev/null | wc -l)
    if [[ "$r1_lines" != "$r2_lines" || $(( r1_lines % 4 )) -ne 0 ]]; then
        fail "clinical-3e: PE synchronization failed — R1=${r1_lines} lines, R2=${r2_lines} lines"
        echo "    diagnostics retained at: $tmp"; trap - RETURN
        rc=2
    else
        ok "clinical-3e: PE synchronization — $(( r1_lines / 4 )) paired reads in output (R1==R2)"
    fi

    # ---- 3f: Graceful failure on invalid input (IEC 62304 §5.5.2) -----
    info "[clinical-3f] Graceful failure: invalid FASTQ must produce non-zero exit..."
    local bad_fq="$tmp/bad_input.fq"
    printf 'THIS_IS_NOT_A_FASTQ\nblah blah\n+\n!!!!!\n' > "$bad_fq"
    local fail_rc=0
    "$CPU_BIN" \
        -i "$bad_fq" -I "$TESTDATA_R2" \
        -o /dev/null -O /dev/null \
        -j /dev/null -h /dev/null \
        --thread 2 >/dev/null 2>&1 || fail_rc=$?
    if (( fail_rc == 0 )); then
        fail "clinical-3f: tool exited 0 on non-FASTQ input (expected non-zero)"
        rc=2
    else
        ok "clinical-3f: graceful failure — non-FASTQ input correctly rejected (exit $fail_rc)"
    fi

    # ---- 3g: PE input count mismatch must produce non-zero exit (H-03 fix) ---
    info "[clinical-3g] PE count mismatch: R1 with extra read must produce non-zero exit..."
    local pem_r1="$tmp/pem_r1.fq"
    local pem_r2="$tmp/pem_r2.fq"
    # R1: 2 reads, R2: 1 read — mismatch must be detected and rejected
    printf '@pem_read1\nACGTACGT\n+\nIIIIIIII\n@pem_read2\nACGTACGT\n+\nIIIIIIII\n' > "$pem_r1"
    printf '@pem_read1\nACGTACGT\n+\nIIIIIIII\n' > "$pem_r2"
    local pem_rc=0
    "$CPU_BIN" \
        -i "$pem_r1" -I "$pem_r2" \
        -o /dev/null -O /dev/null \
        -j /dev/null -h /dev/null \
        --thread 2 >/dev/null 2>&1 || pem_rc=$?
    if (( pem_rc == 0 )); then
        fail "clinical-3g: tool exited 0 on PE count mismatch (expected non-zero)"
        rc=2
    else
        ok "clinical-3g: PE count mismatch correctly rejected (exit $pem_rc)"
    fi

    # ---- 3h: Partially corrupt FASTQ must produce non-zero exit (H-11 fix) ---
    info "[clinical-3h] Partial corruption: valid reads followed by malformed record must produce non-zero exit..."
    local partial_fq="$tmp/partial_bad.fq"
    # First record is valid; second has quality longer than sequence
    printf '@good_read\nACGTACGT\n+\nIIIIIIII\n@bad_read\nACGT\n+\nIIIIIIII\n' > "$partial_fq"
    local partial_rc=0
    "$CPU_BIN" \
        -i "$partial_fq" \
        -o /dev/null \
        -j /dev/null -h /dev/null \
        --thread 1 >/dev/null 2>&1 || partial_rc=$?
    if (( partial_rc == 0 )); then
        fail "clinical-3h: tool exited 0 on partially corrupt FASTQ (expected non-zero)"
        rc=2
    else
        ok "clinical-3h: partially corrupt FASTQ correctly rejected (exit $partial_rc)"
    fi

    # ---- 3i: Invalid quality score rejected (H-09 code-level fix) -----------
    info "[clinical-3i] Invalid quality score: out-of-range quality byte must produce non-zero exit..."
    local badqual_fq="$tmp/bad_quality.fq"
    # Quality character \x01 (ASCII 1) is below valid Phred+33 floor (ASCII 33);
    # use printf with escaped byte so the shell doesn't interpret it as special.
    printf '@valid_read\nACGTACGT\n+\nIIIIIIII\n@bad_qual\nACGTACGT\n+\n' > "$badqual_fq"
    printf 'IIII\x01III\n' >> "$badqual_fq"
    local badqual_rc=0
    "$CPU_BIN" \
        -i "$badqual_fq" \
        -o /dev/null \
        -j /dev/null -h /dev/null \
        --thread 1 >/dev/null 2>&1 || badqual_rc=$?
    if (( badqual_rc == 0 )); then
        fail "clinical-3i: tool exited 0 on out-of-range quality score (expected non-zero)"
        rc=2
    else
        ok "clinical-3i: out-of-range quality score correctly rejected (exit $badqual_rc)"
    fi

    # ---- 3c: Version audit ---------------------------------------------
    info "[clinical-3c] Version audit: binary -v output matches src/common.h FASTP_VER..."
    local common_h="$PROJ_DIR/src/common.h"
    if [[ ! -f "$common_h" ]]; then
        info "[skip] clinical-3c: $common_h not found"
    else
        local expected_ver actual_ver
        expected_ver=$(grep '#define FASTP_VER' "$common_h" | head -1 \
            | sed 's/.*"\(.*\)".*/\1/')
        # binary -v prints "fastp <VER>" to stdout; discard stderr startup line
        actual_ver=$("$CPU_BIN" -v 2>/dev/null \
            | grep -oP '\d+\.\d+\.\d+[a-zA-Z0-9._-]*' | head -1 || true)
        if [[ -z "$expected_ver" ]]; then
            fail "clinical-3c: could not extract FASTP_VER from $common_h"
            rc=2
        elif [[ "$actual_ver" == "$expected_ver" ]]; then
            ok "clinical-3c: version audit — binary reports v$actual_ver, matches src/common.h"
        else
            fail "clinical-3c: version mismatch — binary='$actual_ver', common.h='$expected_ver'"
            rc=2
        fi
    fi

    return $rc
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
if (( DO_CLINICAL )); then
    if ! run_clinical_tier; then
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
