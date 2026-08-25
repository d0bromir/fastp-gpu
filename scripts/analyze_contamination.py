#!/usr/bin/env python3
"""
analyze_contamination.py — AI-powered lab QC diagnostics from fastp-gpu JSON reports.

Reads one or more fastp.json files, applies rule-based pattern matching to classify
what went wrong in the library preparation or sequencing workflow, and optionally
calls an LLM to translate the findings into plain-language corrective actions for
the lab team.

Usage:
    analyze_contamination.py [OPTIONS] report.json [report2.json ...]

Options:
    --ai                 Enable LLM phase (requires API key or Ollama)
    --model MODEL        OpenAI model name or "ollama:<name>" (default: gpt-4o-mini)
    --api-key KEY        OpenAI API key; overrides OPENAI_API_KEY env var
    --api-url URL        OpenAI-compatible base URL (default: https://api.openai.com/v1)
    --ollama-url URL     Ollama base URL (default: http://localhost:11434)
    --format FORMAT      text | json | markdown  (default: text)
    --lab-context TEXT   Optional free-text sample/experiment description
    --out FILE           Write report to FILE instead of stdout
"""

import argparse
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

SEVERITY_CRITICAL = "CRITICAL"
SEVERITY_WARNING  = "WARNING"
SEVERITY_INFO     = "INFO"


@dataclass
class Finding:
    category:  str            # e.g. "HIGH_DUPLICATION"
    severity:  str            # CRITICAL / WARNING / INFO
    title:     str            # one-line headline
    evidence:  str            # measured value(s) that triggered the rule
    cause:     str            # most probable lab cause
    action:    str            # corrective action for next run
    details:   str = ""       # optional extra context


@dataclass
class Sample:
    label:     str
    report:    Dict[str, Any]
    findings:  List[Finding] = field(default_factory=list)
    ai_text:   str = ""


# ---------------------------------------------------------------------------
# Thresholds (tune here, not in the rule functions)
# ---------------------------------------------------------------------------

THRESH = dict(
    adapter_rate_warn    = 0.05,   # adapter_trimmed_reads / total_reads
    adapter_rate_crit    = 0.25,
    adapter_dimer_rate   = 0.05,   # adapter_dimer_reads / total_reads
    dup_rate_warn        = 0.30,
    dup_rate_crit        = 0.60,
    q30_warn             = 0.80,
    q30_crit             = 0.60,
    low_qual_rate_warn   = 0.05,   # low_quality_reads / total_reads
    n_rate_warn          = 0.01,   # too_many_N_reads / total_reads
    contam_rate_warn     = 0.01,   # contaminated_reads / total_reads
    contam_rate_crit     = 0.05,
    insert_short_warn    = 100,    # bp
    insert_short_crit    = 50,
    gc_low_warn          = 0.38,
    gc_high_warn         = 0.60,
    too_short_rate_warn  = 0.05,
    yield_loss_warn      = 0.20,   # 1 - passed / total
)


# ---------------------------------------------------------------------------
# Known contamination source → (probable cause, corrective action)
# ---------------------------------------------------------------------------

SOURCE_INTEL: Dict[str, Tuple[str, str]] = {
    "truseq":         ("Illumina TruSeq adapter carry-over into reads. "
                       "Insert size is shorter than the read length.",
                       "Increase fragment size during library prep, or verify "
                       "size-selection step removed short fragments. "
                       "Confirm adapter trimming settings in fastp."),
    "nextera":        ("Illumina Nextera (Tn5 transposome) adapter read-through. "
                       "Library was over-tagmented or not properly size-selected.",
                       "Reduce tagmentation time or DNA input concentration. "
                       "Tighten AMPure bead size selection (0.6× / 0.8× ratio)."),
    "phix174":        ("PhiX174 bacteriophage control spike-in detected. "
                       "This is expected at low levels; high levels suggest the "
                       "PhiX library was incorrectly proportioned.",
                       "If <1% this is normal. If >5%, reduce PhiX spike-in "
                       "fraction to 1% and recheck cluster density calibration."),
    "illumina_pe_primers": ("Illumina PE primer read-through detected. "
                            "Library insert size may be smaller than the primer length.",
                            "Perform a stricter size selection (e.g. 0.8× AMPure) "
                            "to remove primer dimers before sequencing."),
    "nextera_primers":     ("Nextera primer sequences detected in reads. "
                            "Likely primer dimer carry-over.",
                            "Clean up library with AMPure beads before loading. "
                            "Consider increasing bead ratio or using two-sided selection."),
    "bgi_dnbseq":     ("BGI/DNBSEQ adapter sequence detected on an Illumina run. "
                       "Possible sample cross-contamination from a BGI library or "
                       "reagent lot contamination.",
                       "Audit reagent lots and swap reagents. Verify sample identity "
                       "against LIMS."),
    "iontorrent":     ("Ion Torrent adapter sequences detected on an Illumina run. "
                       "Samples may have been processed on the wrong platform or "
                       "cross-contaminated with an Ion Torrent library.",
                       "Inspect the sample manifest for platform mislabelling. "
                       "Clean the bench and check pipettes used for library dilution."),
    "pacbio":         ("PacBio adapter sequences detected on an Illumina run. "
                       "Likely cross-contamination during library pooling or dilution.",
                       "Inspect sample provenance. Dedicated tips and tubes for "
                       "each platform reduce this risk."),
    "polyruns":       ("Homo-polymer runs (poly-A/T/G/C) detected. "
                       "Common causes: (a) incomplete library, cDNA poly-A tailing, "
                       "(b) NovaSeq G-quadruplex artefacts at low cluster density, "
                       "or (c) failed cluster quality.",
                       "For RNA-seq: verify rRNA depletion and poly-A selection did "
                       "not co-purify free poly-A tails. "
                       "For DNA: check cluster density and re-run if density was low."),
    "lambdaphage":    ("Lambda phage DNA detected. "
                       "Lambda is used in some QC spike-ins and as a carrier DNA; "
                       "bench contamination is also possible.",
                       "Identify source: if not an intentional spike-in, clean the "
                       "bench, autoclave tubes, and replace pipette tips."),
    "ms2_phage":      ("MS2 bacteriophage RNA detected. "
                       "MS2 is an RNA spike-in control; unexpected presence may mean "
                       "wrong sample prep protocol.",
                       "Verify whether MS2 was intentionally added. If not, "
                       "check RNA extraction reagents for contamination."),
    "human_mtdna":    ("Human mitochondrial DNA at elevated levels. "
                       "May indicate degraded or low-input DNA where mtDNA is "
                       "disproportionately amplified.",
                       "Increase input DNA mass. Ensure storage conditions prevent "
                       "DNA degradation (−80 °C, avoid freeze-thaw cycles)."),
    "mouse_mtdna":    ("Mouse mitochondrial DNA detected. "
                       "Possible animal-derived sample contamination or cross-contamination.",
                       "Inspect sample origin. If human-only samples were expected, "
                       "check for bench contamination from mouse cell line work."),
    "rat_mtdna":      ("Rat mitochondrial DNA detected. "
                       "Same as above but with rat tissue origin.",
                       "Same corrective action as Mouse_mtDNA."),
    "rrna_human":     ("Human ribosomal RNA sequences detected. "
                       "rRNA depletion or poly-A selection was incomplete.",
                       "Repeat rRNA depletion with a fresh depletion kit. "
                       "Verify DNase treatment step was performed before RNA-seq prep."),
    "rrna_ecoli":     ("E. coli ribosomal RNA detected. "
                       "Bacterial contamination during RNA extraction or library prep.",
                       "Decontaminate workspace with RNaseZap. "
                       "Replace RNA extraction reagents."),
    "ecoli":          ("Escherichia coli genomic DNA detected. "
                       "Common bench contaminant; may indicate reagent or tip contamination.",
                       "Clean all bench surfaces with bleach followed by 70% ethanol. "
                       "Replace any reagents prepared in non-sterile conditions."),
    "mycoplasma_genitalium": ("Mycoplasma genitalium detected. "
                              "Mycoplasma contamination of cell cultures is extremely common.",
                              "Test all cell lines with a mycoplasma PCR kit. "
                              "Discard contaminated cultures and decontaminate incubators."),
    "mycoplasma_pneu": ("Mycoplasma pneumoniae detected. "
                        "Possible clinical sample or culture contamination.",
                        "Treat cell cultures with anti-mycoplasma reagent (e.g. BM-Cyclin). "
                        "Test all other lines in the lab."),
    "staphylococcus": ("Staphylococcus aureus genomic DNA detected. "
                       "Bench or skin contamination from the technician or environment.",
                       "Wear gloves throughout library prep. "
                       "Clean pipettes, laminar flow hood, and bench with 70% ethanol."),
    "staph_epi":      ("Staphylococcus epidermidis (skin commensal) detected. "
                       "Handling contamination; very common in low-input libraries.",
                       "Use low-bind tubes. Wear gloves and minimize sample handling time."),
    "pseudomonas":    ("Pseudomonas aeruginosa detected. "
                       "Environmental contaminant; found in water sources and reagents.",
                       "Check water quality (HPLC-grade only). "
                       "Inspect reagent lots for expiry and storage conditions."),
    "cutibacterium":  ("Cutibacterium acnes (skin microbiome) detected. "
                       "Skin handling contamination, especially in FFPE or biopsy libraries.",
                       "Increase glove change frequency. "
                       "Pre-treat extraction buffers with UV for 15 min."),
    "strep_pyo":      ("Streptococcus pyogenes detected. "
                       "Respiratory or contact contamination from lab personnel.",
                       "Ensure all staff wear masks during library prep. "
                       "Clean extraction hood with UV."),
    "klebsiella":     ("Klebsiella pneumoniae detected. "
                       "Hospital or environmental contaminant. "
                       "Check reagent water source and autoclave cycles.",
                       "Verify autoclave sterilisation logs. "
                       "Switch to commercially certified sterile water."),
    "acinetobacter":  ("Acinetobacter baumannii detected. "
                       "Multi-drug-resistant hospital pathogen; indicates clinical sample "
                       "contamination or inadequate biosafety procedures.",
                       "Review biosafety level compliance. "
                       "Confirm inactivation protocol for clinical samples before library prep."),
    "bacillus":       ("Bacillus subtilis (or related spore-forming) DNA detected. "
                       "Spores are extremely resistant; reagent or lab contamination.",
                       "Autoclave all buffers. "
                       "Replace tip boxes and discard any reagents stored in open tubes."),
    "helicobacter":   ("Helicobacter pylori DNA detected. "
                       "Likely clinical or biopsy sample origin, or cross-contamination.",
                       "Verify sample manifest. "
                       "Use dedicated extraction areas for clinical GI samples."),
    "ralstonia":      ("Ralstonia species detected. "
                       "Known reagent and water contaminant in molecular biology. "
                       "Frequently found in ultrapure water systems.",
                       "Flush ultrapure water system and replace filters. "
                       "Switch to commercially supplied molecular-biology-grade water."),
    "herpes":         ("Herpes Simplex Virus (HSV) DNA detected. "
                       "Clinical sample derived or contamination from viral stock.",
                       "Verify expected virus in sample. "
                       "Enforce BSL-2 protocols for all HSV samples."),
    "adenovirus5":    ("Human Adenovirus 5 detected. "
                       "May be an intentional viral vector, spike-in control, or contamination.",
                       "Check whether AdV5 was expected in this sample. "
                       "If not, identify the source vector stock."),
    "ebv":            ("Epstein-Barr Virus (EBV) detected. "
                       "Common in immortalised B-cell lines (LCLs) and clinical samples.",
                       "If using LCLs this is expected. "
                       "Otherwise check cell line provenance and contamination history."),
    "hcmv":           ("Human Cytomegalovirus (HCMV) detected. "
                       "Found in clinical samples and certain cell lines (e.g. HFF). "
                       "Verify sample origin.",
                       "If unexpected, quarantine cell lines and test by PCR."),
    "sars_cov_2":     ("SARS-CoV-2 sequences detected. "
                       "Clinical sample with active or residual viral RNA/DNA, or "
                       "contamination from COVID-19 diagnostic lab cross-over.",
                       "Enforce BSL-2+ handling. "
                       "Verify inactivation was performed before nucleic acid extraction."),
    "hiv_1":          ("HIV-1 sequences detected. "
                       "Clinical sample (expected) or cross-contamination. "
                       "Requires strict BSL-2 procedures.",
                       "Confirm sample identity against clinical LIMS. "
                       "Verify inactivation step was completed."),
    "hepatitis_b":    ("Hepatitis B Virus detected. "
                       "Clinical sample origin or inadequate inactivation. BSL-2 handling required.",
                       "Confirm sample status in LIMS. "
                       "Review decontamination procedure for extraction robot."),
    "hepatitis_c":    ("Hepatitis C Virus RNA detected. "
                       "Clinical sample or cross-contamination from HCV diagnostic work.",
                       "Verify sample identity. "
                       "Use dedicated HCV-processing pipettes and tips."),
    "hpv_16":         ("HPV-16 sequences detected. "
                       "Clinical sample (cervical, oropharyngeal biopsy) or cell line origin.",
                       "Confirm sample type in LIMS. "
                       "HeLa cells carry HPV-18; cross-contamination is possible."),
    "sc_mtdna":       ("Saccharomyces cerevisiae mitochondrial DNA detected. "
                       "Yeast contamination of a non-yeast sample, possibly from media.",
                       "Inspect media preparation area for yeast contamination. "
                       "Confirm sample is not yeast-derived."),
    "influenza_h1n1": ("Influenza A H1N1 RNA/DNA detected. "
                       "Clinical sample or cross-contamination from diagnostic lab.",
                       "Verify sample manifest. "
                       "Confirm influenza inactivation protocol was followed."),
}


def _lookup_source(name: str) -> Tuple[str, str]:
    """Return (cause, action) for a contamination source name (case-insensitive prefix match)."""
    key = name.lower().replace(" ", "_").replace("-", "_")
    # Exact match first
    if key in SOURCE_INTEL:
        return SOURCE_INTEL[key]
    # Prefix / substring match
    for k, v in SOURCE_INTEL.items():
        if key.startswith(k) or k.startswith(key):
            return v
    return (
        f"Unknown contaminant source '{name}' detected in reads.",
        "Inspect the contamination source and cross-reference with sample metadata. "
        "Check reagent lots and bench decontamination procedures.",
    )


# ---------------------------------------------------------------------------
# Rule-based classifier
# ---------------------------------------------------------------------------

def classify_findings(report: Dict[str, Any]) -> List[Finding]:
    findings: List[Finding] = []

    summary      = report.get("summary", {})
    bf           = summary.get("before_filtering", {})
    af           = summary.get("after_filtering", {})
    filt         = report.get("filtering_result", {})
    dup          = report.get("duplication", {})
    ins          = report.get("insert_size", {})
    adapter      = report.get("adapter_cutting", {})
    contam       = report.get("contamination", {})
    r1_before    = report.get("read1_before_filtering", {})

    total_reads  = bf.get("total_reads", 0) or 1  # avoid div-by-zero

    # --- 1. Adapter contamination ---
    adapter_trimmed = adapter.get("adapter_trimmed_reads", 0)
    adapter_rate    = adapter_trimmed / total_reads
    if adapter_rate >= THRESH["adapter_rate_crit"]:
        findings.append(Finding(
            category="ADAPTER_CONTAMINATION",
            severity=SEVERITY_CRITICAL,
            title="Severe adapter read-through",
            evidence=f"{adapter_rate:.1%} of reads contained adapter sequence "
                     f"({adapter_trimmed:,} / {total_reads:,})",
            cause="Library insert size is much shorter than the read length. "
                  "The sequencer read through the insert and into the adapter on the other end. "
                  "This typically results from over-fragmentation, degraded DNA, or a very "
                  "small-insert library being loaded without size selection.",
            action="Increase input DNA fragment size. Perform SPRI/AMPure bead size selection "
                   "to remove fragments shorter than the read length. Confirm fragment distribution "
                   "with Bioanalyzer or TapeStation before loading.",
        ))
    elif adapter_rate >= THRESH["adapter_rate_warn"]:
        findings.append(Finding(
            category="ADAPTER_CONTAMINATION",
            severity=SEVERITY_WARNING,
            title="Elevated adapter read-through",
            evidence=f"{adapter_rate:.1%} of reads contained adapter sequence",
            cause="A fraction of the library has inserts shorter than the read length. "
                  "May indicate a bimodal size distribution with a short-fragment tail.",
            action="Tighten size-selection step. Check Bioanalyzer trace for a short-fragment "
                   "shoulder below 150 bp. Increase lower-cutoff AMPure bead ratio.",
        ))

    # --- 2. Adapter dimers → too-short reads ---
    ad_dimer     = filt.get("adapter_dimer_reads", 0)
    ad_dimer_rate = ad_dimer / total_reads
    if ad_dimer_rate >= THRESH["adapter_dimer_rate"]:
        findings.append(Finding(
            category="ADAPTER_DIMER",
            severity=SEVERITY_CRITICAL,
            title="High adapter dimer content",
            evidence=f"{ad_dimer_rate:.1%} reads were adapter dimers ({ad_dimer:,})",
            cause="Adapter molecules ligated directly to each other without an insert. "
                  "Causes: insufficient DNA input (ligation reaction competed with itself), "
                  "adapters in excess, or ligation performed at wrong temperature/time.",
            action="Increase DNA input to the recommended range (typically 10–200 ng for "
                   "TruSeq). Use size selection (0.8× AMPure) after ligation to remove dimers. "
                   "Verify adapter concentration was not above the recommended molar ratio.",
        ))

    # --- 3. Duplication ---
    dup_rate = dup.get("rate", 0.0)
    if dup_rate >= THRESH["dup_rate_crit"]:
        findings.append(Finding(
            category="HIGH_DUPLICATION",
            severity=SEVERITY_CRITICAL,
            title="Very high PCR duplication",
            evidence=f"Duplication rate: {dup_rate:.1%}",
            cause="Extremely low DNA input or excessive PCR amplification cycles. "
                  "The same template molecule was sequenced many times. "
                  "Possible causes: sample degradation, incorrect quantification, "
                  "wrong number of PCR cycles for the input mass.",
            action="Re-quantify the input DNA with Qubit (not NanoDrop). "
                   "Reduce PCR cycle number (use the manufacturer's cycle chart). "
                   "Consider PCR-free library prep if input mass allows (≥1 µg). "
                   "Assess sample integrity on TapeStation before re-extraction.",
        ))
    elif dup_rate >= THRESH["dup_rate_warn"]:
        findings.append(Finding(
            category="HIGH_DUPLICATION",
            severity=SEVERITY_WARNING,
            title="Elevated PCR duplication",
            evidence=f"Duplication rate: {dup_rate:.1%}",
            cause="Moderate over-amplification. Input mass was borderline or "
                  "one to two extra PCR cycles were added.",
            action="Reduce PCR cycles by 2. Verify input quantification. "
                   "For low-input protocols ensure the correct enrichment cycle number "
                   "was selected from the kit table.",
        ))

    # --- 4. Q30 rate ---
    q30 = bf.get("q30_rate", 1.0)
    if q30 <= THRESH["q30_crit"]:
        findings.append(Finding(
            category="LOW_QUALITY",
            severity=SEVERITY_CRITICAL,
            title="Critically low Q30 base quality",
            evidence=f"Q30 rate: {q30:.1%} (target ≥80%)",
            cause="Flow cell chemistry failure, severely degraded DNA/RNA, "
                  "incorrect sequencing cycle number, reagent storage failure, "
                  "or cluster density too high (overlapping clusters).",
            action="Check sequencer run metrics in BaseSpace/LIMS for cluster density "
                   "and error rate per cycle. If cluster density is high, reload at lower "
                   "concentration. If DNA was degraded, re-extract. "
                   "Contact Illumina support if run metrics indicate instrument issue.",
        ))
    elif q30 <= THRESH["q30_warn"]:
        findings.append(Finding(
            category="LOW_QUALITY",
            severity=SEVERITY_WARNING,
            title="Below-target Q30 base quality",
            evidence=f"Q30 rate: {q30:.1%} (target ≥80%)",
            cause="Suboptimal sequencing run. Possible mild cluster density issue, "
                  "partially degraded nucleic acid, or end-of-run quality drop.",
            action="Review run metrics per cycle in BaseSpace. "
                   "If quality drops sharply in the last 20 cycles, reduce read length. "
                   "Otherwise check sample quality on Bioanalyzer.",
        ))

    # --- 5. High N rate ---
    n_reads     = filt.get("too_many_N_reads", 0)
    n_rate      = n_reads / total_reads
    if n_rate >= THRESH["n_rate_warn"]:
        findings.append(Finding(
            category="TECHNICAL_FAILURE",
            severity=SEVERITY_WARNING,
            title="Elevated uncalled-base (N) reads",
            evidence=f"{n_rate:.1%} of reads had too many N bases ({n_reads:,})",
            cause="Flow cell surface defect, low cluster density, reagent failure, "
                  "or camera/optics issue. N calls indicate the sequencer could not "
                  "confidently call a base.",
            action="Check sequencer run report for per-tile failure patterns. "
                   "If localised tiles show N clusters, it may be a surface defect or "
                   "bubble during flow cell loading. Re-run the sample on a new flow cell.",
        ))

    # --- 6. Insert size ---
    insert_peak = ins.get("peak", 200)
    if insert_peak > 0 and insert_peak <= THRESH["insert_short_crit"]:
        findings.append(Finding(
            category="INSERT_SIZE_ANOMALY",
            severity=SEVERITY_CRITICAL,
            title="Critically short insert size",
            evidence=f"Insert size peak: {insert_peak} bp",
            cause="Library is predominantly adapter dimers or very short fragments. "
                  "Input DNA may be severely degraded (DIN <3), or size selection failed.",
            action="Assess DNA integrity with TapeStation (DIN score). "
                   "If DIN <4, re-extract from a higher-quality source. "
                   "Perform two-sided AMPure selection (0.6× lower + 0.8× upper) to "
                   "enrich for 200–500 bp fragments.",
        ))
    elif insert_peak > 0 and insert_peak <= THRESH["insert_short_warn"]:
        findings.append(Finding(
            category="INSERT_SIZE_ANOMALY",
            severity=SEVERITY_WARNING,
            title="Short insert size",
            evidence=f"Insert size peak: {insert_peak} bp",
            cause="DNA fragmentation was too aggressive, or input material is partially "
                  "degraded. FFPE and cell-free DNA samples routinely have short inserts.",
            action="Adjust sonication/enzymatic fragmentation parameters. "
                   "For FFPE samples, this may be unavoidable; use a short-insert library "
                   "kit designed for FFPE material.",
        ))

    # --- 7. GC content ---
    gc = bf.get("gc_content", 0.5)
    if gc < THRESH["gc_low_warn"]:
        findings.append(Finding(
            category="GC_BIAS",
            severity=SEVERITY_WARNING,
            title="Low GC content",
            evidence=f"Mean GC: {gc:.1%} (expected ~40–60%)",
            cause="Low-GC organism (e.g. A. thaliana, many bacteria), AT-rich "
                  "contamination, or PCR bias against high-GC regions.",
            action="Verify the expected GC for this organism. "
                   "If sample was human/mouse, investigate AT-rich contaminants or "
                   "PCR bias (use a hot-start polymerase with GC enhancer).",
        ))
    elif gc > THRESH["gc_high_warn"]:
        findings.append(Finding(
            category="GC_BIAS",
            severity=SEVERITY_WARNING,
            title="High GC content",
            evidence=f"Mean GC: {gc:.1%} (expected ~40–60%)",
            cause="High-GC organism (e.g. Mycobacterium, Pseudomonas), "
                  "rRNA contamination, or PCR bias against low-GC regions.",
            action="Verify the expected GC for this organism. "
                   "If unexpectedly high, check for rRNA contamination (see rRNA findings). "
                   "For high-GC organisms, use a specialised high-GC PCR polymerase.",
        ))

    # --- 8. Low quality reads fraction ---
    lq_reads = filt.get("low_quality_reads", 0)
    lq_rate  = lq_reads / total_reads
    if lq_rate >= THRESH["low_qual_rate_warn"]:
        findings.append(Finding(
            category="LOW_QUALITY",
            severity=SEVERITY_WARNING,
            title="High low-quality read fraction",
            evidence=f"{lq_rate:.1%} of reads failed quality filter ({lq_reads:,})",
            cause="Degraded nucleic acid, sequencing chemistry issues, or "
                  "off-spec library quality.",
            action="Reassess input nucleic acid quality. Check RIN (RNA) or DIN (DNA). "
                   "Review sequencing run metrics for per-cycle quality degradation.",
        ))

    # --- 9. Too-short reads after trimming ---
    ts_reads = filt.get("too_short_reads", 0)
    ts_rate  = ts_reads / total_reads
    if ts_rate >= THRESH["too_short_rate_warn"]:
        findings.append(Finding(
            category="OVER_TRIMMING",
            severity=SEVERITY_WARNING,
            title="High fraction of reads too short after trimming",
            evidence=f"{ts_rate:.1%} of reads were discarded as too short ({ts_reads:,})",
            cause="Library contains many short inserts that were trimmed to below the "
                  "minimum length. Consistent with adapter dimer carry-over or "
                  "over-fragmented library.",
            action="Improve size selection to remove sub-100 bp fragments before library "
                   "quantification and loading. Ensure quality trimming thresholds are "
                   "appropriate for the library type.",
        ))

    # --- 10. Overall yield loss ---
    passed = af.get("total_reads", 0)
    yield_loss = 1 - (passed / total_reads)
    if yield_loss >= THRESH["yield_loss_warn"]:
        findings.append(Finding(
            category="LOW_YIELD",
            severity=SEVERITY_WARNING,
            title="High read loss after QC filtering",
            evidence=f"{yield_loss:.1%} of reads were discarded "
                     f"({total_reads - passed:,} discarded / {total_reads:,} total)",
            cause="A combination of quality, adapter, and length failures caused "
                  "substantial read loss. This may indicate systemic library prep issues "
                  "rather than a single failure mode.",
            action="Address each contributing failure category in order of severity. "
                   "Consider submitting a repeat library before proceeding with analysis.",
        ))

    # --- 11. Biological contamination ---
    total_contaminated = contam.get("total_contaminated_reads", 0)
    contam_rate = total_contaminated / total_reads
    sources = contam.get("sources", [])

    for source in sources:
        name       = source.get("name", f"Source {source.get('id', '?')}")
        src_reads  = source.get("reads", 0)
        src_rate   = src_reads / total_reads
        cause, action = _lookup_source(name)

        sev = SEVERITY_INFO
        if src_rate >= THRESH["contam_rate_crit"]:
            sev = SEVERITY_CRITICAL
        elif src_rate >= THRESH["contam_rate_warn"]:
            sev = SEVERITY_WARNING

        findings.append(Finding(
            category="BIOLOGICAL_CONTAMINATION",
            severity=sev,
            title=f"Contamination: {name}",
            evidence=f"{src_rate:.2%} of reads match {name} ({src_reads:,} / {total_reads:,})",
            cause=cause,
            action=action,
        ))

    # --- 12. Yield OK with no issues: report as INFO ---
    if not findings:
        passed_reads = filt.get("passed_filter_reads", 0)
        findings.append(Finding(
            category="PASS",
            severity=SEVERITY_INFO,
            title="Library passed all QC checks",
            evidence=(f"Q30={q30:.1%}, dup={dup_rate:.1%}, "
                      f"adapter={adapter_rate:.1%}, "
                      f"yield={passed_reads/total_reads:.1%}"),
            cause="No significant QC anomalies detected.",
            action="No corrective action required. Proceed with downstream analysis.",
        ))

    return findings


# ---------------------------------------------------------------------------
# LLM interface (OpenAI-compatible REST + Ollama)
# ---------------------------------------------------------------------------

def _build_prompt(sample: Sample, lab_context: str) -> str:
    """Build the LLM prompt from structured findings."""
    findings_text = "\n".join(
        f"- [{f.severity}] {f.title}: {f.evidence}\n"
        f"  Probable cause: {f.cause}\n"
        f"  Suggested action: {f.action}"
        for f in sample.findings
    )

    context_line = f"Lab context: {lab_context}" if lab_context else ""

    return textwrap.dedent(f"""\
        You are a senior NGS laboratory quality control specialist reviewing an \
Illumina sequencing QC report for a new sequencing lab. Your audience is the \
bench-level laboratory technician who performed the library preparation, not a \
bioinformatician.
        {context_line}
        Sample: {sample.label}

        Automated QC findings:
        {findings_text}

        Write a plain-language diagnostic report that:
        1. States what most likely went wrong in the lab, at which specific workflow \
step (DNA extraction, quantification, fragmentation, end-repair, adapter ligation, \
PCR amplification, size selection, or loading), and why you draw that conclusion.
        2. Ranks the issues by urgency (most critical first).
        3. Provides exactly what the technician should do differently in the next run, \
step by step.
        4. Notes which findings are expected or acceptable (e.g. low PhiX at <1%).
        5. If multiple findings point to the same root cause, group them.

        Be specific and actionable. Do not repeat the numbers already listed above; \
refer to them only when needed to justify a conclusion. Keep the report under 400 words.
    """)


def call_llm(
    prompt: str,
    model: str,
    api_key: Optional[str],
    api_url: str,
    ollama_url: str,
) -> str:
    """Send prompt to OpenAI-compatible or Ollama endpoint. Returns response text."""
    try:
        import requests  # guaranteed present; checked at startup
    except ImportError:
        return "(LLM call failed: 'requests' library not available)"

    headers = {"Content-Type": "application/json"}

    if model.startswith("ollama:"):
        # Ollama API
        ollama_model = model[len("ollama:"):]
        url = ollama_url.rstrip("/") + "/api/generate"
        payload = {
            "model": ollama_model,
            "prompt": prompt,
            "stream": False,
        }
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
        except Exception as exc:
            return f"(Ollama LLM call failed: {exc})"
    else:
        # OpenAI-compatible API
        if not api_key:
            return (
                "(LLM call skipped: no API key provided. "
                "Pass --api-key KEY or set OPENAI_API_KEY env var, "
                "or use --no-ai for rule-based output only.)"
            )
        headers["Authorization"] = f"Bearer {api_key}"
        url = api_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 800,
            "temperature": 0.3,
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            return f"(OpenAI LLM call failed: {exc})"


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

SEV_SYMBOL = {SEVERITY_CRITICAL: "✗", SEVERITY_WARNING: "!", SEVERITY_INFO: "✓"}
SEV_ORDER  = {SEVERITY_CRITICAL: 0, SEVERITY_WARNING: 1, SEVERITY_INFO: 2}


def _sorted_findings(findings: List[Finding]) -> List[Finding]:
    return sorted(findings, key=lambda f: (SEV_ORDER[f.severity], f.title))


def format_text(sample: Sample) -> str:
    lines = [f"=== QC DIAGNOSIS: {sample.label} ===\n"]
    for f in _sorted_findings(sample.findings):
        sym = SEV_SYMBOL.get(f.severity, "?")
        lines.append(f"[{sym} {f.severity}] {f.title}")
        lines.append(f"  Evidence : {f.evidence}")
        lines.append(f"  Cause    : {f.cause}")
        lines.append(f"  Action   : {f.action}")
        if f.details:
            lines.append(f"  Details  : {f.details}")
        lines.append("")
    if sample.ai_text:
        lines.append("--- AI LAB REPORT ---")
        lines.append(sample.ai_text)
        lines.append("")
    return "\n".join(lines)


def format_markdown(sample: Sample) -> str:
    lines = [f"## QC Diagnosis: {sample.label}\n"]
    for f in _sorted_findings(sample.findings):
        sym = SEV_SYMBOL.get(f.severity, "?")
        lines.append(f"### {sym} {f.title}")
        lines.append(f"**Severity:** {f.severity}  ")
        lines.append(f"**Evidence:** {f.evidence}  ")
        lines.append(f"**Probable cause:** {f.cause}  ")
        lines.append(f"**Corrective action:** {f.action}  ")
        if f.details:
            lines.append(f"**Details:** {f.details}  ")
        lines.append("")
    if sample.ai_text:
        lines.append("---\n\n### AI Lab Report\n")
        lines.append(sample.ai_text)
        lines.append("")
    return "\n".join(lines)


def format_json_report(sample: Sample) -> dict:
    return {
        "sample": sample.label,
        "findings": [
            {
                "category": f.category,
                "severity": f.severity,
                "title": f.title,
                "evidence": f.evidence,
                "cause": f.cause,
                "action": f.action,
            }
            for f in _sorted_findings(sample.findings)
        ],
        "ai_report": sample.ai_text or None,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "AI-powered contamination and QC diagnostics from fastp-gpu JSON reports."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Rule-based only (default, no API key needed):
              analyze_contamination.py fastp.json

              # Enable AI with OpenAI GPT-4o-mini (requires OPENAI_API_KEY):
              analyze_contamination.py --ai fastp.json

              # Enable AI with a local Ollama model:
              analyze_contamination.py --ai --model ollama:llama3 fastp.json

              # Markdown output to file, multiple samples:
              analyze_contamination.py --format markdown --out report.md s1.json s2.json
        """),
    )
    p.add_argument("reports", nargs="+", metavar="REPORT.json",
                   help="One or more fastp JSON report files")
    p.add_argument("--model", default="gpt-4o-mini",
                   help='OpenAI model name or "ollama:<name>" (default: gpt-4o-mini)')
    p.add_argument("--api-key", default=None,
                   help="OpenAI API key (default: $OPENAI_API_KEY)")
    p.add_argument("--api-url", default="https://api.openai.com/v1",
                   help="OpenAI-compatible base URL")
    p.add_argument("--ollama-url", default="http://localhost:11434",
                   help="Ollama base URL (default: http://localhost:11434)")
    p.add_argument("--ai", action="store_true",
                   help="Enable LLM phase (requires --api-key / $OPENAI_API_KEY or --model ollama:...)")
    p.add_argument("--format", choices=["text", "json", "markdown"], default="text",
                   help="Output format (default: text)")
    p.add_argument("--lab-context", default="",
                   help="Optional sample/experiment description for LLM context")
    p.add_argument("--out", default=None, metavar="FILE",
                   help="Write report to FILE (default: stdout)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")

    samples: List[Sample] = []
    for path in args.reports:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                report = json.load(fh)
        except FileNotFoundError:
            print(f"ERROR: File not found: {path}", file=sys.stderr)
            return 1
        except json.JSONDecodeError as exc:
            print(f"ERROR: Could not parse JSON from {path}: {exc}", file=sys.stderr)
            return 1

        label = os.path.basename(path)
        sample = Sample(label=label, report=report)
        sample.findings = classify_findings(report)

        if args.ai:
            prompt = _build_prompt(sample, args.lab_context)
            sample.ai_text = call_llm(
                prompt=prompt,
                model=args.model,
                api_key=api_key,
                api_url=args.api_url,
                ollama_url=args.ollama_url,
            )

        samples.append(sample)

    # Format output
    if args.format == "json":
        output = json.dumps(
            [format_json_report(s) for s in samples],
            indent=2,
            ensure_ascii=False,
        )
    elif args.format == "markdown":
        output = "\n".join(format_markdown(s) for s in samples)
    else:
        output = "\n".join(format_text(s) for s in samples)

    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(output)
        except OSError as exc:
            print(f"ERROR: Cannot write to {args.out}: {exc}", file=sys.stderr)
            return 1
    else:
        print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
