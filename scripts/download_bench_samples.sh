#!/usr/bin/env bash
# =============================================================================
# download_bench_samples.sh — Download all benchmark FASTQ datasets from ENA
#
# DATASET TABLE
# ──────────────────────────────────────────────────────────────────────────────
# Key  Accession    Study        Source          Instrument        Read len   Size
# ──────────────────────────────────────────────────────────────────────────────
# WGS datasets (used in run_benchmark.sh for WGS_SE / WGS_PE tests):
#
#  ERR1044780  ERP001517  1000 Genomes, 1KGP  HiSeq 2000 PE 100bp  6.3G+6.5G
#              Sample: HG00513 (Han Chinese South, CHS)
#              ENA: https://www.ebi.ac.uk/ena/browser/view/ERR1044780
#
#  ERR1044319  ERP001517  1000 Genomes, 1KGP  HiSeq 2000 PE 100bp  9.2G+9.0G
#              Sample: HG00443 (Han Chinese South, CHS)
#              ENA: https://www.ebi.ac.uk/ena/browser/view/ERR1044319
#
#  DRR216653   DRP007155  Kagawa Univ (DDBJ)   NovaSeq 6000 PE 150bp  20G+20G
#              Sample: SAMD00210915 (Homo sapiens WGS)
#              ENA: https://www.ebi.ac.uk/ena/browser/view/DRR216653
#
#  ERR1305370  PRJEB6463  Personalis 1KGP HiSeq X Ten PE 151bp 2.9G+2.9G
#              Diverse-platform extension of 1KGP for benchmarking; clinical-grade
#              ENA: https://www.ebi.ac.uk/ena/browser/view/ERR1305370
#
#  ERR1305374  PRJEB6463  Personalis 1KGP HiSeq X Ten PE 151bp 2.7G+2.7G
#              Same study as ERR1305370 — different sample
#              ENA: https://www.ebi.ac.uk/ena/browser/view/ERR1305374
#
#  SRR5589994  PRJNA387623  HiSeq 2000 BS-Seq SE 52bp 26.8G
#              Bisulfite-seq, very large single-end methylation file
#              ENA: https://www.ebi.ac.uk/ena/browser/view/SRR5589994
#
# ──────────────────────────────────────────────────────────────────────────────
# RNA-Seq datasets (extra read-length / platform diversity):
#
#  SRR13074117  PRJNA678979  HiSeq X Ten RNA-Seq PE 152bp 3.0G+3.1G
#               ENA: https://www.ebi.ac.uk/ena/browser/view/SRR13074117
#
#  DRR346892    PRJDB13049   NextSeq 500 RNA-Seq PE 71bp 0.6G+0.6G
#               ENA: https://www.ebi.ac.uk/ena/browser/view/DRR346892
#
#  SRR5186485   PRJNA265627  HiSeq 2500 RNA-Seq PE 150bp 1.5G+1.5G
#               ENA: https://www.ebi.ac.uk/ena/browser/view/SRR5186485
#
# ──────────────────────────────────────────────────────────────────────────────
# Public panel datasets (substitute for private TST-15 panel):
#
#  DRR262998   PRJDB10859  Kobe Univ (DDBJ)   MiSeq PE 156bp  83MB+83MB
#              Study: PDAC 4-gene panel (KRAS/TP53/CDKN2A/SMAD4)
#              Sample: Pancreatic ductal carcinoma 1, 1.04M reads
#              ENA: https://www.ebi.ac.uk/ena/browser/view/DRR262998
#
#  DRR263018   PRJDB10859  Kobe Univ (DDBJ)   MiSeq PE 156bp  119MB+130MB
#              Study: PDAC 4-gene panel (KRAS/TP53/CDKN2A/SMAD4)
#              Sample: Pancreatic ductal carcinoma 25, 1.54M reads
#              ENA: https://www.ebi.ac.uk/ena/browser/view/DRR263018
#
# NOTE: The private TST-15 panel (S1A_S1_L001_*.fastq.gz) is clinical data
#       and cannot be downloaded from a public source.  The PDAC panel datasets
#       above are the closest publicly available equivalents.
# ──────────────────────────────────────────────────────────────────────────────
#
# MD5 checksums (from ENA):
#   ERR1044780_1: f0f10736b0c09a461fad4d9c0f7e05ca  ERR1044780_2: 7f970afd61ed9edb3622a8bb03052b7d
#   ERR1044319_1: dd1360ada44e4c5113ea3c6930c79869  ERR1044319_2: a4db3f6103f74c824f7502e6ea439bcb
#   DRR216653_1:  ca773d8819f8537d70a2949b8d1d0961  DRR216653_2:  21446c4abcdf1531b9e50f4ff2b8d78a
#   DRR262998_1:  a06ab68f3e8b82cb1ead17bfac100605  DRR262998_2:  adea6d31cf2392f9a7626e4b49d57fd7
#   DRR263018_1:  c654e3360be91b7ed2c5b7713e187bc0  DRR263018_2:  e785da598d933c9d87460d8cc47fe5b6
#
# Usage:
#   ./scripts/download_bench_samples.sh            # all missing files
#   ./scripts/download_bench_samples.sh wgs        # WGS datasets only
#   ./scripts/download_bench_samples.sh rna        # RNA-Seq datasets only
#   ./scripts/download_bench_samples.sh panel      # public panel datasets only
#   ./scripts/download_bench_samples.sh --verify   # verify checksums only (no download)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WGS_DIR="${WGS_DIR:-/home/mpiuser/FASTQ/WGS}"
PANEL_DIR="${PANEL_DIR:-/home/mpiuser/FASTQ/Panel_public}"
RNA_DIR="${RNA_DIR:-/home/mpiuser/FASTQ/RNA}"

ENA_FTP="https://ftp.sra.ebi.ac.uk"

# ---------------------------------------------------------------------------
# Dataset definitions: name  url_r1  url_r2  md5_r1  md5_r2  dest_r1  dest_r2
# Each entry: "accession;url_r1;url_r2;md5_r1;md5_r2;dest_r1;dest_r2"
# ---------------------------------------------------------------------------

WGS_DATASETS=(
  "ERR1044780;\
${ENA_FTP}/vol1/fastq/ERR104/000/ERR1044780/ERR1044780_1.fastq.gz;\
${ENA_FTP}/vol1/fastq/ERR104/000/ERR1044780/ERR1044780_2.fastq.gz;\
f0f10736b0c09a461fad4d9c0f7e05ca;\
7f970afd61ed9edb3622a8bb03052b7d;\
${WGS_DIR}/ERR1044780_1.fastq.gz;\
${WGS_DIR}/ERR1044780_2.fastq.gz"

  "ERR1044319;\
${ENA_FTP}/vol1/fastq/ERR104/009/ERR1044319/ERR1044319_1.fastq.gz;\
${ENA_FTP}/vol1/fastq/ERR104/009/ERR1044319/ERR1044319_2.fastq.gz;\
dd1360ada44e4c5113ea3c6930c79869;\
a4db3f6103f74c824f7502e6ea439bcb;\
${WGS_DIR}/ERR1044319_1.fastq.gz;\
${WGS_DIR}/ERR1044319_2.fastq.gz"

  "DRR216653;\
${ENA_FTP}/vol1/fastq/DRR216/DRR216653/DRR216653_1.fastq.gz;\
${ENA_FTP}/vol1/fastq/DRR216/DRR216653/DRR216653_2.fastq.gz;\
ca773d8819f8537d70a2949b8d1d0961;\
21446c4abcdf1531b9e50f4ff2b8d78a;\
${WGS_DIR}/DRR216653_1.fastq.gz;\
${WGS_DIR}/DRR216653_2.fastq.gz"

  "ERR1305370;\
${ENA_FTP}/vol1/fastq/ERR130/000/ERR1305370/ERR1305370_1.fastq.gz;\
${ENA_FTP}/vol1/fastq/ERR130/000/ERR1305370/ERR1305370_2.fastq.gz;\
b9b48f30eccfc1c226db58427740291e;\
7e1f4f830ff76f71c12121e7a741ce64;\
${WGS_DIR}/ERR1305370_1.fastq.gz;\
${WGS_DIR}/ERR1305370_2.fastq.gz"

  "ERR1305374;\
${ENA_FTP}/vol1/fastq/ERR130/004/ERR1305374/ERR1305374_1.fastq.gz;\
${ENA_FTP}/vol1/fastq/ERR130/004/ERR1305374/ERR1305374_2.fastq.gz;\
20717ea177faca2f799f0fc71827ae53;\
5c65e0e7eef243ad1c273fced4b13bc2;\
${WGS_DIR}/ERR1305374_1.fastq.gz;\
${WGS_DIR}/ERR1305374_2.fastq.gz"

  "SRR5589994;\
${ENA_FTP}/vol1/fastq/SRR558/004/SRR5589994/SRR5589994.fastq.gz;\
;\
a18eaf55bd7c3a7ea235195df54fa119;\
;\
${WGS_DIR}/SRR5589994.fastq.gz;\
"
)

# RNA-Seq datasets (read-length / platform diversity beyond WGS)
RNA_DATASETS=(
  "SRR13074117;\
${ENA_FTP}/vol1/fastq/SRR130/017/SRR13074117/SRR13074117_1.fastq.gz;\
${ENA_FTP}/vol1/fastq/SRR130/017/SRR13074117/SRR13074117_2.fastq.gz;\
8e6096b810dd297bceea61d9b0fd7968;\
fdbc60c7b04f8cbbf503848eeba6128f;\
${RNA_DIR}/SRR13074117_1.fastq.gz;\
${RNA_DIR}/SRR13074117_2.fastq.gz"

  "DRR346892;\
${ENA_FTP}/vol1/fastq/DRR346/DRR346892/DRR346892_1.fastq.gz;\
${ENA_FTP}/vol1/fastq/DRR346/DRR346892/DRR346892_2.fastq.gz;\
bc6ff4728abd6c02d6717e10708e4b0c;\
ae5e223d17e89ddf710483e58aadb749;\
${RNA_DIR}/DRR346892_1.fastq.gz;\
${RNA_DIR}/DRR346892_2.fastq.gz"

  "SRR5186485;\
${ENA_FTP}/vol1/fastq/SRR518/005/SRR5186485/SRR5186485_1.fastq.gz;\
${ENA_FTP}/vol1/fastq/SRR518/005/SRR5186485/SRR5186485_2.fastq.gz;\
2a2eacb4a2f6511533e0b1c4478dd289;\
63679f0920754ce6b35b13802a6a49e5;\
${RNA_DIR}/SRR5186485_1.fastq.gz;\
${RNA_DIR}/SRR5186485_2.fastq.gz"
)

PANEL_DATASETS=(
  "DRR262998;\
${ENA_FTP}/vol1/fastq/DRR262/DRR262998/DRR262998_1.fastq.gz;\
${ENA_FTP}/vol1/fastq/DRR262/DRR262998/DRR262998_2.fastq.gz;\
a06ab68f3e8b82cb1ead17bfac100605;\
adea6d31cf2392f9a7626e4b49d57fd7;\
${PANEL_DIR}/DRR262998_1.fastq.gz;\
${PANEL_DIR}/DRR262998_2.fastq.gz"

  "DRR263018;\
${ENA_FTP}/vol1/fastq/DRR263/DRR263018/DRR263018_1.fastq.gz;\
${ENA_FTP}/vol1/fastq/DRR263/DRR263018/DRR263018_2.fastq.gz;\
c654e3360be91b7ed2c5b7713e187bc0;\
e785da598d933c9d87460d8cc47fe5b6;\
${PANEL_DIR}/DRR263018_1.fastq.gz;\
${PANEL_DIR}/DRR263018_2.fastq.gz"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
step()    { echo ""; echo ">>> $*"; }
ok()      { echo "    [OK]   $*"; }
skip()    { echo "    [SKIP] $*"; }
fail()    { echo "    [FAIL] $*" >&2; FAILURES=$((FAILURES + 1)); }

FAILURES=0

verify_md5() {
    local file="$1" expected="$2"
    local actual
    actual=$(md5sum "$file" | awk '{print $1}')
    if [[ "$actual" == "$expected" ]]; then
        ok "MD5 OK  $(basename "$file")"
        return 0
    else
        fail "MD5 MISMATCH  $(basename "$file")  expected=$expected  got=$actual"
        return 1
    fi
}

download_file() {
    local url="$1" dest="$2" expected_md5="$3"

    if [[ -f "$dest" ]]; then
        local actual
        actual=$(md5sum "$dest" | awk '{print $1}')
        if [[ "$actual" == "$expected_md5" ]]; then
            skip "$(basename "$dest")  (already present, MD5 OK)"
            return 0
        else
            echo "    [WARN] $(basename "$dest") present but MD5 mismatch — re-downloading"
            rm -f "$dest"
        fi
    fi

    echo "    Downloading $(basename "$dest") ..."
    wget --quiet --show-progress --continue -O "$dest" "$url" || {
        fail "wget failed for $url"
        return 1
    }
    verify_md5 "$dest" "$expected_md5" || return 1
}

download_dataset() {
    local entry="$1"
    IFS=';' read -r acc url_r1 url_r2 md5_r1 md5_r2 dest_r1 dest_r2 <<< "$entry"

    step "$acc"
    mkdir -p "$(dirname "$dest_r1")"
    download_file "$url_r1" "$dest_r1" "$md5_r1"
    [[ -n "$url_r2" ]] && download_file "$url_r2" "$dest_r2" "$md5_r2"
}

verify_dataset() {
    local entry="$1"
    IFS=';' read -r acc url_r1 url_r2 md5_r1 md5_r2 dest_r1 dest_r2 <<< "$entry"

    step "$acc"
    local pairs="$dest_r1:$md5_r1"
    [[ -n "$dest_r2" ]] && pairs="$pairs $dest_r2:$md5_r2"
    for f_and_md5 in $pairs; do
        local f="${f_and_md5%%:*}"
        local m="${f_and_md5##*:}"
        if [[ -f "$f" ]]; then
            verify_md5 "$f" "$m"
        else
            fail "Missing: $f"
        fi
    done
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
TARGET="${1:-all}"

print_table() {
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════"
    echo " Benchmark dataset provenance"
    echo "════════════════════════════════════════════════════════════════════════════"
    printf " %-16s  %-12s  %-30s  %-18s  %s\n" "Accession" "Study" "Description" "Instrument" "Size (R1+R2)"
    echo "────────────────────────────────────────────────────────────────────────────"
    printf " %-16s  %-12s  %-30s  %-18s  %s\n" "ERR1044780" "ERP001517" "1KGP HG00513 WGS (CHS)" "HiSeq 2000 PE 100bp" "6.3G + 6.5G"
    printf " %-16s  %-12s  %-30s  %-18s  %s\n" "ERR1044319" "ERP001517" "1KGP HG00443 WGS (CHS)" "HiSeq 2000 PE 100bp" "9.2G + 9.0G"
    printf " %-16s  %-12s  %-30s  %-18s  %s\n" "DRR216653"  "DRP007155" "Kagawa Univ WGS"        "NovaSeq 6000 PE 150bp" "20G + 20G"
    printf " %-16s  %-12s  %-30s  %-18s  %s\n" "ERR1305370" "PRJEB6463"  "Personalis 1KGP WGS"   "HiSeq X Ten PE 151bp" "2.9G + 2.9G"
    printf " %-16s  %-12s  %-30s  %-18s  %s\n" "ERR1305374" "PRJEB6463"  "Personalis 1KGP WGS"   "HiSeq X Ten PE 151bp" "2.7G + 2.7G"
    printf " %-16s  %-12s  %-30s  %-18s  %s\n" "SRR5589994" "PRJNA387623" "BS-Seq SE 52bp"        "HiSeq 2000 SE 52bp"   "26.8G"
    echo "────────────────────────────────────────────────────────────────────────────"
    printf " %-16s  %-12s  %-30s  %-18s  %s\n" "SRR13074117" "PRJNA678979" "RNA-Seq HiSeq X"     "HiSeq X Ten PE 152bp" "3.0G + 3.1G"
    printf " %-16s  %-12s  %-30s  %-18s  %s\n" "DRR346892"   "PRJDB13049"  "RNA-Seq NextSeq"     "NextSeq 500 PE 71bp"  "0.6G + 0.6G"
    printf " %-16s  %-12s  %-30s  %-18s  %s\n" "SRR5186485"  "PRJNA265627" "RNA-Seq HiSeq 2500"  "HiSeq 2500 PE 150bp"  "1.5G + 1.5G"
    echo "────────────────────────────────────────────────────────────────────────────"
    printf " %-16s  %-12s  %-30s  %-18s  %s\n" "DRR262998"  "PRJDB10859" "PDAC 4-gene panel (1M reads)" "MiSeq PE 156bp" "83MB + 83MB"
    printf " %-16s  %-12s  %-30s  %-18s  %s\n" "DRR263018"  "PRJDB10859" "PDAC 4-gene panel (1.5M reads)" "MiSeq PE 156bp" "119MB + 130MB"
    echo "════════════════════════════════════════════════════════════════════════════"
    echo " NOTE: TST-15 panel (S1A_S1_L001) is private clinical data — not downloadable."
    echo "       DRR262998/DRR263018 are the closest public equivalents (same technology,"
    echo "       similar read count and length, same category: cancer hotspot panel)."
    echo "════════════════════════════════════════════════════════════════════════════"
    echo ""
}

print_table

case "$TARGET" in
    all)
        for ds in "${WGS_DATASETS[@]}";   do download_dataset "$ds"; done
        for ds in "${RNA_DATASETS[@]}";   do download_dataset "$ds"; done
        for ds in "${PANEL_DATASETS[@]}"; do download_dataset "$ds"; done
        ;;
    wgs)
        for ds in "${WGS_DATASETS[@]}"; do download_dataset "$ds"; done
        ;;
    rna)
        for ds in "${RNA_DATASETS[@]}"; do download_dataset "$ds"; done
        ;;
    panel)
        for ds in "${PANEL_DATASETS[@]}"; do download_dataset "$ds"; done
        ;;
    --verify)
        for ds in "${WGS_DATASETS[@]}";   do verify_dataset "$ds"; done
        for ds in "${RNA_DATASETS[@]}";   do verify_dataset "$ds"; done
        for ds in "${PANEL_DATASETS[@]}"; do verify_dataset "$ds"; done
        ;;
    *)
        echo "Usage: $0 [all|wgs|rna|panel|--verify]"
        exit 1
        ;;
esac

echo ""
if [[ $FAILURES -eq 0 ]]; then
    echo "=== All done. No failures. ==="
else
    echo "=== Done with $FAILURES failure(s). ==="
    exit 1
fi
