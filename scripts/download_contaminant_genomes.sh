#!/usr/bin/env bash
# scripts/download_contaminant_genomes.sh
#
# Download additional biological contaminant reference genomes from NCBI and
# append them to data/contaminants.fa in the correct '# source: NAME' format.
#
# The genomes included here are too large to ship with the binary distribution
# but are among the most common sources of biological contamination in NGS
# experiments.  After running this script, rebuild fastp so the larger hash
# table is pre-sized at startup:
#
#   scripts/build_all.sh all
#
# Usage:
#   scripts/download_contaminant_genomes.sh [OPTIONS] [GENOME ...]
#
# Options:
#   -o, --out FILE    Append to FILE instead of data/contaminants.fa
#   -l, --list        Print available genome names and exit
#   --all             Download all available genomes
#   -h, --help        Show this help
#
# Genome names (pass one or more, or --all):
#   ecoli           Escherichia coli K-12 MG1655        (4.6 Mbp, NC_000913)
#   mycoplasma_pneu Mycoplasma pneumoniae M129           (0.8 Mbp, NC_000912)
#   staphylococcus  Staphylococcus aureus MRSA252        (2.9 Mbp, NC_002952)
#   staph_epi       Staphylococcus epidermidis RP62A     (2.5 Mbp, NC_004461)
#   strep_pyo       Streptococcus pyogenes M1 GAS        (1.9 Mbp, NC_002737)
#   klebsiella      Klebsiella pneumoniae HS11286        (5.6 Mbp, NC_016845)
#   acinetobacter   Acinetobacter baumannii ACICU        (3.9 Mbp, NC_010611)
#   bacillus        Bacillus subtilis 168                (4.2 Mbp, NC_000964)
#   helicobacter    Helicobacter pylori 26695            (1.7 Mbp, NC_000915)
#   ralstonia       Ralstonia pickettii 12J              (5.5 Mbp, NC_010529)
#   pseudomonas     Pseudomonas aeruginosa PAO1          (6.3 Mbp, NC_002516)
#   cutibacterium   Cutibacterium acnes KPA171202        (2.6 Mbp, NC_006085)
#   sc_mtdna        Saccharomyces cerevisiae mtDNA       ( 86 kbp, NC_001224)
#   herpes          Human herpesvirus 1 (HSV-1)          (152 kbp, NC_001806)
#   adenovirus5     Human adenovirus 5                   ( 36 kbp, NC_001405)
#   ebv             Epstein-Barr virus (EBV)             (172 kbp, NC_007605)
#   hcmv            Human cytomegalovirus AD169          (236 kbp, NC_006273)
#   rrna_human      Homo sapiens 45S rRNA repeat         ( 44 kbp, U13369)
#   rrna_ecoli      Escherichia coli 16S+23S rRNA operon (  7 kbp, J01695)
#   influenza_h1n1  Influenza A/H1N1 (8 segments)       (  ~13 kbp, various)
#
# Examples:
#   # Download E. coli and Mycoplasma pneumoniae only:
#   scripts/download_contaminant_genomes.sh ecoli mycoplasma_pneu
#
#   # Download everything into a custom file:
#   scripts/download_contaminant_genomes.sh --all -o /data/my_contaminants.fa

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_FILE="${REPO_ROOT}/data/contaminants.fa"
NCBI_BASE="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&rettype=fasta&retmode=text"

# ---------------------------------------------------------------------------
# Genome catalogue: (source_name accession description)
# ---------------------------------------------------------------------------
declare -A GENOME_ACCESSION GENOME_DESC GENOME_COMMENT

GENOME_ACCESSION[ecoli]="NC_000913"
GENOME_DESC[ecoli]="Escherichia coli K-12 MG1655 complete genome (4 641 652 bp)"
GENOME_COMMENT[ecoli]="Most common lab bacterium; found in gut microbiome samples,\n# environmental controls, and reagent contamination."

GENOME_ACCESSION[mycoplasma_pneu]="NC_000912"
GENOME_DESC[mycoplasma_pneu]="Mycoplasma pneumoniae M129 complete genome (816 394 bp)"
GENOME_COMMENT[mycoplasma_pneu]="Second most prevalent Mycoplasma cell-culture contaminant;\n# also a common respiratory pathogen in clinical samples."

GENOME_ACCESSION[staphylococcus]="NC_002952"
GENOME_DESC[staphylococcus]="Staphylococcus aureus MRSA252 complete genome (2 902 619 bp)"
GENOME_COMMENT[staphylococcus]="Common skin microbiome member; appears in low-input and\n# single-cell preparations from human donors."

GENOME_ACCESSION[staph_epi]="NC_004461"
GENOME_DESC[staph_epi]="Staphylococcus epidermidis RP62A complete genome (2 499 279 bp)"
GENOME_COMMENT[staph_epi]="Most abundant bacterium on human skin; dominant contaminant\n# in low-biomass dermatology, blood, and implant sequencing studies."

GENOME_ACCESSION[strep_pyo]="NC_002737"
GENOME_DESC[strep_pyo]="Streptococcus pyogenes M1 GAS complete genome (1 852 441 bp)"
GENOME_COMMENT[strep_pyo]="Group A Streptococcus; common respiratory and wound pathogen.\n# Appears in throat swab, FFPE, and oropharyngeal sequencing datasets."

GENOME_ACCESSION[klebsiella]="NC_016845"
GENOME_DESC[klebsiella]="Klebsiella pneumoniae HS11286 complete genome (5 593 591 bp)"
GENOME_COMMENT[klebsiella]="Nosocomial ESKAPE pathogen; common in clinical metagenomics,\n# hospital environment studies, and gut microbiome datasets."

GENOME_ACCESSION[acinetobacter]="NC_010611"
GENOME_DESC[acinetobacter]="Acinetobacter baumannii ACICU complete genome (3 904 116 bp)"
GENOME_COMMENT[acinetobacter]="ESKAPE nosocomial pathogen; a frequent contaminant in hospital\n# metagenomic surveillance and environmental samples."

GENOME_ACCESSION[bacillus]="NC_000964"
GENOME_DESC[bacillus]="Bacillus subtilis 168 complete genome (4 215 606 bp)"
GENOME_COMMENT[bacillus]="Common lab organism; spores survive standard decontamination\n# and appear in reagent blanks and environmental controls."

GENOME_ACCESSION[helicobacter]="NC_000915"
GENOME_DESC[helicobacter]="Helicobacter pylori 26695 complete genome (1 667 867 bp)"
GENOME_COMMENT[helicobacter]="Infects ~50% of the global population; present in any gastric\n# biopsy or upper GI clinical sequencing dataset."

GENOME_ACCESSION[ralstonia]="NC_010529"
GENOME_DESC[ralstonia]="Ralstonia pickettii 12J complete genome (5 496 450 bp)"
GENOME_COMMENT[ralstonia]="The single most frequently reported bacterium in sequencing kit\n# negative controls (Salter et al. 2014 BMC Biol; Weyrich et al. 2019).\n# Originates from ultra-pure water systems used in reagent manufacture."

GENOME_ACCESSION[pseudomonas]="NC_002516"
GENOME_DESC[pseudomonas]="Pseudomonas aeruginosa PAO1 complete genome (6 264 404 bp)"
GENOME_COMMENT[pseudomonas]="Environmental and opportunistic pathogen; common in water\n# sources used in lab workflows."

GENOME_ACCESSION[cutibacterium]="NC_006085"
GENOME_DESC[cutibacterium]="Cutibacterium acnes KPA171202 complete genome (2 560 265 bp)"
GENOME_COMMENT[cutibacterium]="Dominant skin microbiome bacterium; contaminates low-input\n# and FFPE library preparations."

GENOME_ACCESSION[sc_mtdna]="NC_001224"
GENOME_DESC[sc_mtdna]="Saccharomyces cerevisiae mitochondrial genome (85 779 bp)"
GENOME_COMMENT[sc_mtdna]="Yeast mtDNA; present in any sample processed with yeast-derived\n# enzymes or in fermentation/biotechnology workflows."

GENOME_ACCESSION[herpes]="NC_001806"
GENOME_DESC[herpes]="Human herpesvirus 1 (HSV-1) genome (152 261 bp)"
GENOME_COMMENT[herpes]="Appears in patient-derived samples and latent cell-line\n# infections. Reference: NCBI NC_001806"

GENOME_ACCESSION[adenovirus5]="NC_001405"
GENOME_DESC[adenovirus5]="Human adenovirus 5 genome (35 938 bp)"
GENOME_COMMENT[adenovirus5]="Common research viral vector; contaminates cell lines\n# that have been transduced with adenovirus vectors."

GENOME_ACCESSION[ebv]="NC_007605"
GENOME_DESC[ebv]="Epstein-Barr virus (EBV) genome (172 764 bp)"
GENOME_COMMENT[ebv]="Infects >95% of adults latently; appears in any B-cell\n# or lymphoblastoid cell line (LCL) preparation."

GENOME_ACCESSION[hcmv]="NC_006273"
GENOME_DESC[hcmv]="Human cytomegalovirus AD169 genome (236 261 bp)"
GENOME_COMMENT[hcmv]="Common latent herpesvirus; detected in primary human cell\n# cultures and clinical specimens."

GENOME_ACCESSION[rrna_human]="U13369"
GENOME_DESC[rrna_human]="Homo sapiens 45S ribosomal RNA repeat unit (44 838 bp)"
GENOME_COMMENT[rrna_human]="rRNA contamination from incomplete depletion; also appears\n# in any total-RNA or poly-A capture library."

GENOME_ACCESSION[rrna_ecoli]="J01695"
GENOME_DESC[rrna_ecoli]="Escherichia coli 16S+23S rRNA operon (7 459 bp)"
GENOME_COMMENT[rrna_ecoli]="Bacterial rRNA operon; appears when rRNA depletion is incomplete\n# in bacterial RNA-seq or metatranscriptomic library preparations."

# Influenza A H1N1 is 8 segments; handled specially below in the fetch loop.
GENOME_ACCESSION[influenza_h1n1]="CY121687 CY121689 CY121691 CY121693 CY121695 CY121697 CY121699 CY121701"
GENOME_DESC[influenza_h1n1]="Influenza A/H1N1 A/California/07/2009 pandemic strain (8 segments, ~13.6 kbp)"
GENOME_COMMENT[influenza_h1n1]="Pandemic H1N1 reference strain. All 8 genomic segments are\n# fetched and concatenated into one source entry."

ALL_GENOMES=(ecoli mycoplasma_pneu staphylococcus staph_epi strep_pyo klebsiella
             acinetobacter bacillus helicobacter ralstonia pseudomonas cutibacterium
             sc_mtdna herpes adenovirus5 ebv hcmv rrna_human rrna_ecoli influenza_h1n1)

# ---------------------------------------------------------------------------
usage() {
    sed -n '/^# Usage:/,/^[^#]/p' "$0" | head -n -1 | sed 's/^# \?//'
    exit 0
}

list_genomes() {
    printf "%-20s %-12s %s\n" "NAME" "ACCESSION" "DESCRIPTION"
    printf "%-20s %-12s %s\n" "----" "---------" "-----------"
    for g in "${ALL_GENOMES[@]}"; do
        printf "%-20s %-12s %s\n" "$g" "${GENOME_ACCESSION[$g]}" "${GENOME_DESC[$g]}"
    done
    exit 0
}

SELECTED=()
DO_ALL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--out)   OUT_FILE="$2"; shift 2 ;;
        -l|--list)  list_genomes ;;
        --all)      DO_ALL=true; shift ;;
        -h|--help)  usage ;;
        -*)         echo "Unknown option: $1" >&2; exit 1 ;;
        *)          SELECTED+=("$1"); shift ;;
    esac
done

if $DO_ALL; then
    SELECTED=("${ALL_GENOMES[@]}")
fi

if [[ ${#SELECTED[@]} -eq 0 ]]; then
    echo "No genomes specified. Use --list to see available genomes or --all to download all." >&2
    echo "Example: $0 ecoli mycoplasma_pneu" >&2
    exit 1
fi

# Validate names
for g in "${SELECTED[@]}"; do
    if [[ -z "${GENOME_ACCESSION[$g]+x}" ]]; then
        echo "Unknown genome name: '$g'. Run '$0 --list' to see valid names." >&2
        exit 1
    fi
done

mkdir -p "$(dirname "${OUT_FILE}")"

echo "Appending to: ${OUT_FILE}"
echo ""

for g in "${SELECTED[@]}"; do
    acc="${GENOME_ACCESSION[$g]}"
    desc="${GENOME_DESC[$g]}"
    comment="${GENOME_COMMENT[$g]}"

    # Count space-separated accessions (multi-segment genomes like influenza).
    num_acc=$(echo "$acc" | wc -w)

    if [[ "$num_acc" -gt 1 ]]; then
        echo -n "  Downloading ${g} (${num_acc} segments)... "
        {
            echo ""
            echo "# source: ${g}"
            echo "# ${desc}"
            echo -e "# ${comment}"
            echo "# References: NCBI ${acc}"
            for seg in $acc; do
                curl -sf "${NCBI_BASE}&id=${seg}"
            done
        } >> "${OUT_FILE}"
    else
        echo -n "  Downloading ${g} (${acc})... "
        {
            echo ""
            echo "# source: ${g}"
            echo "# ${desc}"
            echo -e "# ${comment}"
            echo "# Reference: NCBI ${acc}"
            curl -sf "${NCBI_BASE}&id=${acc}"
        } >> "${OUT_FILE}"
    fi

    local_kb=$(( $(wc -c < "${OUT_FILE}") / 1024 ))
    echo "done  (total file: ${local_kb} KB)"
done

echo ""
echo "Sources now in ${OUT_FILE}:"
grep "^# source:" "${OUT_FILE}" | sed 's/^# source: /  /'
echo ""
echo "Rebuild fastp to incorporate the updated database:"
echo "  scripts/build_all.sh all"
