# GPU-Accelerated FASTQ QC/Preprocessing: Competitor Landscape (Verified)

**Purpose.** An AI-assisted literature review (external chat transcript, not
authoritative) proposed a set of GPU-accelerated FASTQ QC/preprocessing tools as
potential competitors for `fastp-gpu`. This note independently verifies each
claim against primary sources (papers, package registries, source repositories)
before any tool is cited in the manuscript or included in a benchmark. Several
claims in the original transcript did not hold up and are corrected below.

**Date verified:** 2026-08-16. **Extended:** 2026-08-22 (second independent
search pass, see §1.6 and §5). **Corrected:** 2026-08-24 (§1.1 — an
unsupported "sometimes marketed as GPU-accelerated" claim about
RabbitQCPlus, present in this note's own text and in the manuscript,
withdrawn after direct verification against RabbitQCPlus's own README and
peer-reviewed description found no such marketing anywhere).

---

## 1. Summary of corrections to the proposed table

1. **RabbitQCPlus is not GPU-accelerated. Corrected 2026-08-24: we also no
   longer say it was "sometimes marketed" that way** — checked after a
   reviewer/author challenge and unsupported. RabbitQCPlus's README and
   peer-reviewed description (Yan et al. 2023, *Methods* 216:39–50, PMID
   37330158) make no GPU/CUDA claim anywhere; the closest thing, "Integrated
   and optimized the CARE error correction engine," doesn't mention GPU
   either. The "marketed as GPU" framing was our own inference from that
   CARE integration (CARE has a separate GPU build) rather than anything
   RabbitQCPlus's authors said. Independently verified and unchanged:
   **zero `.cu` files**, `Makefile` hard-codes `THRUST_DEVICE_SYSTEM_OMP`
   for every build (GitHub's "10% Cuda" stat is vendored headers, not
   compiled device code), and its own benchmarks use "CARE (v2.0.0, CPU
   version)." RabbitQCPlus is a legitimate CPU/SIMD baseline competitor, but
   must not be described as GPU-accelerated or as ever having claimed to be.
2. **G-CNV (Manconi et al., CNR Italy) is a real, peer-reviewed, published GPU
   tool, but its source is not currently obtainable.** Confirmed directly via
   Frontiers in Bioengineering and Biotechnology (10.3389/fbioe.2015.00028,
   full author list, volume, and article number verified 2026-08-22); its
   official download host (`itb.cnr.it`) returns HTTP 404, and no mirror was
   found on GitHub or Software Heritage. It can be **cited in Related Work as
   historical prior art** but cannot be benchmarked.
   **A second, related Manconi-group item — informally referred to in this
   note's earlier draft as "G-FastQC," title "Pre-processing of high-throughput
   sequencing data" (Manconi, Moscatelli, Gnocchi, Milanesi, 2017) — is
   weaker evidence than originally stated here.** A 2026-08-22 re-verification
   found Google Scholar lists its venue only as `publications.cnr.it` (the CNR
   institutional repository itself, not a journal or conference), no DOI
   exists, and the `iris.cnr.it` handle it redirects to returns HTTP 403 like
   G-CNV's. "G-FastQC" does not appear as a name for this item anywhere
   independent of this note. **This item should not be cited as a
   peer-reviewed publication** (no `\bibitem` in the manuscript); it may be
   described in text with a plain URL to the CNR repository page, similar to
   the NGS-GPU pipeline treatment in §1.6.
3. **G-CNV targets an obsolete CUDA compute capability even if source were
   recovered.** The paper states it was developed for "NVIDIA GPU cards based
   on the most recent Kepler architecture" with "CUDA (release ≥6.0)."
   Kepler (compute capability 3.5) compilation support was removed starting
   with CUDA 12.0; our host runs CUDA 13.2. The code would need non-trivial
   porting even if recovered.
4. **Parabricks provides no FASTQ QC/trim tool** (see
   [parabricks-compatibility-evidence.md](parabricks-compatibility-evidence.md)
   for the full compatibility investigation) and is excluded here as already
   covered.
5. **One genuine, currently buildable GPU FASTQ tool was found: CARE / CARE-GPU**
   (Kallenborn et al.), but it performs **read error correction**, not
   FastQC/fastp-style QC, filtering, or adapter trimming. It is a different
   functional category and not a like-for-like competitor.
6. **A second, independent academic GPU-QC claim was found and also ruled
   out: the "NGS-GPU pipeline" (CIPF, Valencia — Dopazo/Medina/Blanquer
   group, ~2011–2012).** Like G-FastQC/G-CNV, this is a different research
   group, different country, and different tool, but the same failure mode:
   its host (`docs.bioinfo.cipf.es`, a Redmine instance) now shows
   "Under maintenance"; the Wayback Machine's CDX index shows the project
   page only ever existed 2012–2016 (oldest capture 2012-03-26, no
   substantive capture after 2016-04-13); and even while live, file
   downloads required a Redmine account. Notably, its own downloadable
   artefact was named `fastq-hpc-tools`, not `fastq-gpu-tools` — the
   filename itself hedges the GPU-acceleration claim made in the project's
   prose. This is now the **second** independently-verified case of an
   academic GPU-FASTQ-QC tool going unobtainable within roughly a decade,
   which strengthens rather than weakens the "no viable competitor exists"
   conclusion — see §6.

**Bottom line:** after verification, there is no other GPU-accelerated tool
that both (a) performs FastQC/fastp-equivalent raw-FASTQ QC or preprocessing
and (b) is currently obtainable and buildable on modern hardware. `fastp-gpu`
appears to fill a genuine gap; the appropriate benchmark set is the CPU tools
below plus `fastp-gpu` itself, not a fabricated GPU-vs-GPU comparison.

---

## 2. Verified tool table

| Tool | Category | Genuinely GPU-accelerated? | FASTQ QC / filter / trim? | Source obtainable today? | Buildable on our A100 (CUDA 13.2)? | Actively maintained? |
|---|---|---|---|---|---|---|
| FastQC | QC report | No | Yes (report only, no filtering) | Yes | N/A (CPU/Java) | Yes |
| [Falco](https://github.com/smithlabcode/falco) | QC report | No | Yes (FastQC-compatible, ~3× faster) | Yes, GitHub | N/A (CPU) | Yes |
| fastp | QC + trim/filter | No | Yes (full) | Yes, GitHub | N/A (CPU) | Yes |
| [RabbitQCPlus](https://github.com/RabbitBio/RabbitQCPlus) | QC + trim/filter | **No** (corrected — see §1.1) | Yes (comprehensive, SIMD-optimized) | Yes, GitHub, Bioconda | N/A (CPU) | Yes |
| NVIDIA Parabricks (`fq2bam` etc.) | Alignment + variant calling | Yes | No raw-FASTQ QC (only a BAM SAM-flag filter) | Yes (proprietary container) | No — confirmed SIGILL on our Neoverse-N1 (see Parabricks note) | Yes (NVIDIA) |
| "Pre-processing of high-throughput sequencing data" (Manconi et al. 2017; informally "G-FastQC" in this note) | QC + filter + trim | Claimed (item prose), **unverifiable** — no peer-reviewed venue or DOI found (see §3) | Claimed, per item description | **Not found** (CNR institutional-repository page redirects to `iris.cnr.it`, 403; no mirror) | Unknown — could not obtain | No |
| G-CNV (Manconi et al. 2015) | CNV data-prep (filter/mask/dedup on GPU; adapter removal parallelized on CPU via cutadapt) | Partial (3 of 4 QC ops on GPU) | Partial (no FastQC-style report) | **Not found** (`itb.cnr.it` download link is 404) | No — targets Kepler/CUDA ≥6.0; compute capability 3.5 unsupported by CUDA ≥12 | No (2015, unmaintained) |
| [CARE / CARE-GPU](https://github.com/fkallen/CARE) | Read error correction (not QC/trim) | **Yes** (52% CUDA codebase; requires CUDA 11+, Pascal (cc 6.0)+) | No — different task | Yes, GitHub, actively maintained (2022+) | Yes — A100 is Ampere (cc 8.0), well above the Pascal minimum | Yes |
| NGS-GPU pipeline (CIPF, ~2011–2012) | QC report + read filtering | Claimed (project prose), unverifiable — never obtained | Yes (QC + filter module); no adapter trimming found | **Not found** (`docs.bioinfo.cipf.es` "Under maintenance"; Wayback CDX shows dead since ~2016; downloads required Redmine login even when live) | Unknown — source never obtained | No (no activity since ~2012–2014) |

---

## 3. Verification method

- **RabbitQCPlus GPU-code check** (not a claim RabbitQCPlus itself made —
  see the correction in §1.1): searched the GitHub repository directly for
  `extension:cu` (0 results), `THRUST_DEVICE_SYSTEM_CUDA` (0 results), and
  inspected `Makefile` — every build path forces
  `-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP` and compiles with
  `g++`/`gcc` only; no `nvcc` invocation exists anywhere in the build.
- **"G-FastQC" existence and citability (2026-08-22 re-verification):**
  Google Scholar confirms the title "Pre-processing of high-throughput
  sequencing data," authors A. Manconi, M. Moscatelli, M. Gnocchi,
  L. Milanesi, year 2017 — but lists its *venue* only as
  `publications.cnr.it`, i.e. the hosting repository itself, not a journal
  or conference. `publications.cnr.it/doc/449170` redirects to
  `iris.cnr.it/handle/20.500.14243/420927`, which returns HTTP 403 (as does
  a direct fetch attempt). No DOI was found via Scholar, Semantic Scholar,
  DBLP (which lists no Manconi publications for 2015–2018 at all under this
  title), or targeted search. **Conclusion: this item cannot be confirmed as
  peer-reviewed and must not be cited via `\citep{}` to a fabricated venue**
  — see the manuscript-changes draft, §5, for the recommended informal
  `\url{}` treatment instead.
- **G-CNV existence and hardware requirement:** full text retrieved from
  Frontiers in Bioengineering and Biotechnology (open access,
  10.3389/fbioe.2015.00028). §2.5 "Hardware and software requirements"
  states explicitly: *"G-CNV has been designed to work with NVIDIA GPU cards
  based on the most recent Kepler architecture. G-CNV works on Linux-based
  systems equipped with CUDA (release ≥6.0)."* Download link given in the
  paper (`http://www.itb.cnr.it/web/bioinformatics/gcnv`) returns HTTP 404
  today.
- **Software Heritage / Wayback Machine:** both blocked automated retrieval
  (bot-challenge wall / HTTP 503 respectively); not pursued further per the
  policy of not brute-forcing blocked paths. If a human with a browser can
  retrieve either archive, that would change the "not found" conclusion.
- **CARE GPU requirement:** confirmed directly from the repository's
  `Readme.md` — *"Additional prerequisites for GPU version: CUDA Toolkit 11 or
  newer; A CUDA-capable graphics card with Pascal architecture (e.g. Nvidia
  GTX 1080) or newer."* Build target `make gpu` produces `care-gpu`.
- **NGS-GPU pipeline (2026-08-22 pass):** project page
  `http://docs.bioinfo.cipf.es/projects/ngs-gpu-pipeline` returns a Redmine
  "Under maintenance" page today. The Wayback Machine CDX index for this URL
  shows captures only between 2012-03-26 and 2016-04-13 (the tool's entire
  observed lifespan); no later capture exists. The downloadable release
  artefact referenced in archived pages is named `fastq-hpc-tools` (~469 KB,
  dated 2012-03-07) — an HPC/CPU-cluster naming, not a GPU one — and file
  downloads on the live Redmine required an authenticated account even when
  the site was up, so the source was never independently obtained or
  inspected for real device code, unlike CARE and RabbitQCPlus where the
  repository could be searched directly for `.cu` files.

---

## 4. Recommendation

For the manuscript's related-work / competitor discussion:

1. Cite **G-CNV** (2015, peer-reviewed, `\citep{manconi2015}`) as historical
   GPU FASTQ-preprocessing prior art. Describe (do not `\citep`) the related
   2017 Manconi-group item and the CIPF **NGS-GPU pipeline** (~2011–2012) as
   two further, weaker-evidence claims from two independent academic
   groups — noting all are no longer available/verifiable and could not be
   benchmarked — see §6 for why this pattern itself is informative rather
   than incidental.
2. Benchmark against the **CPU** competitor set that is actually obtainable
   and relevant: **FastQC, Falco, fastp, RabbitQCPlus** — all real, maintained,
   buildable tools that operate on the same FASTQ→QC/trim task as `fastp-gpu`.
3. Do **not** present a "GPU vs GPU" comparison table — no other currently
   obtainable, buildable tool performs the same function on GPU. Framing
   `fastp-gpu` as the first *currently available* GPU-accelerated
   FastQC/fastp-equivalent (rather than "the first ever," given G-CNV's and
   the 2017 item's prior claims) is the defensible claim.
4. **CARE-GPU is a candidate for a separate, clearly-labelled comparison** if
   error correction is ever added to `fastp-gpu`'s scope — but including it in
   the main QC/trim benchmark today would conflate two different tasks and
   should be avoided.

No benchmark has been run yet for any of the CPU tools above. See the
companion request in the session for next steps before committing compute time
on the production `galaxy` host.

---

## 5. Empirical benchmark (galaxy, 2026-08-16)

FastQC v0.12.1 and Falco v2.0.1 were built and run against the **same public
datasets** already defined as canonical entries in `run_benchmark.sh`
(`Panel_PDAC_PE_1M` / `Panel_PDAC_PE_1.5M`, DRR262998 / DRR263018 — the closest
public equivalent to the private clinical TST-15 panel), on the same host
(`galaxy`, ARM Neoverse-N1, 2× A100) as our own benchmark suite, via
[scripts/run_competitor_benchmark.sh](../../../scripts/run_competitor_benchmark.sh).

**RabbitQCPlus could not be built**: 8 core source files (`adapter.cpp`,
`cpu_alignment.cpp`, `duplicate.cpp`, `FastxStream.cpp`, `state.cpp`,
`lib/deflate_decompress.hpp`, and others) unconditionally include x86-only SIMD
intrinsics headers (`pmmintrin.h`, `immintrin.h`) with no ARM/NEON fallback —
confirmed via `grep -rlE "immintrin.h|pmmintrin.h|__m128i|__m256i"` across the
repository. This is a hard architectural incompatibility, not a portability bug
fixable with a small patch (unlike the separate, unrelated missing
`#include <cstdint>` in its vendored `cxxopts.hpp`, which was fixed locally to
get this far). RabbitQCPlus is therefore **x86_64-only** and excluded from this
benchmark.

| Dataset | Tool | Threads | Wall time (s) | Peak RSS |
|---|---|---|---|---|
| Panel_PDAC_PE_1M (DRR262998, ~1M reads) | `fastp` (CPU) | 8 | 1.95 | 1.57 GB |
| | `fastp-gpu` | 8 | 2.38 | 2.31 GB |
| | FastQC | 2 (per-file) | 10.33 | 0.88 GB |
| | Falco (R1 + R2) | 8 | 1.15 + 1.14 = 2.29 | 0.11 GB |
| Panel_PDAC_PE_1.5M (DRR263018, ~1.5M reads) | `fastp` (CPU) | 8 | 2.54 | 1.74 GB |
| | `fastp-gpu` | 8 | 2.95 | 2.46 GB |
| | FastQC | 2 (per-file) | 14.28 | 0.87 GB |
| | Falco (R1 + R2) | 8 | 1.60 + 1.71 = 3.31 | 0.13–0.14 GB |

Observations (reported honestly, including results unfavourable to `fastp-gpu`):

- **`fastp-gpu` is slower than `fastp` (CPU) on both of these small files**
  (2.38 s vs 1.95 s; 2.95 s vs 2.54 s). This matches the fork's own documented
  behaviour: GPU dispatch overhead (CUDA context init, host↔device transfer)
  is not amortised at this input size (~85–130 MB per file). See
  `optimization_attempts/2026-05-15_gpu-init-size-gate.md` for the underlying
  investigation. These panel datasets are too small to demonstrate GPU
  advantage and are not representative of the WGS-scale datasets used
  elsewhere in the paper's main results.
- **Falco is the fastest tool here**, and comparable to `fastp` (CPU) per
  file — expected, since Falco only produces a QC report and does not filter,
  trim, or write processed FASTQ output, so it does less work per read.
- **FastQC (the Java reference implementation) is markedly slower** than
  every native tool, consistent with its reputation and with prior figures
  already in this repository (`docs/PERFORMANCE_SUMMARY.md`).
- These numbers are **exploratory and single-rep**, not the validated,
  multi-repetition benchmark methodology of `scripts/run_benchmark.sh`. They
  should not be quoted as final paper figures without repeating at the WGS
  dataset sizes used elsewhere and following the same statistical rigor
  (repetitions, resource sampling) as the canonical benchmark.

### Environment notes for reproducing on `galaxy`

- Non-interactive SSH sessions do not source the CUDA `PATH`/`LD_LIBRARY_PATH`
  additions from `~/.bashrc` (its early-return guard skips non-interactive
  shells). Building/running the GPU binary requires exporting
  `PATH=/usr/local/cuda/bin:$PATH` and
  `LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:$LD_LIBRARY_PATH` explicitly in
  the same command.
- The pre-built `fastp_d0bromir_gpu` binary in `~/tools/bin/` predates the
  CUDA 13.1.2 / nvCOMP 5.3.0.16 package upgrade (see repository memory) and
  fails with `libnvcomp.so.5: cannot open shared object file`. Rebuild via
  `scripts/build_all.sh all` before benchmarking; do not reuse the stale
  prebuilt binaries.

---

## 6. Why no GPU competitor survives: an arithmetic-intensity explanation

The 2026-08-22 search pass (NGS-GPU pipeline, RiboDetector, NVIDIA RAPIDS /
GenomeWorks / nvcomp) found nothing that changes the conclusion of §1: no
GPU-accelerated FASTQ QC/trim/filter tool functionally comparable to fastp is
both currently obtainable and buildable. Two independent academic groups
(Manconi et al., CNR Italy, with two related claims — G-CNV, peer-reviewed
in 2015, and a second 2017 item whose citability could not be confirmed on
2026-08-22 re-verification; and Dopazo/Medina/Blanquer, CIPF Spain, ~2011)
have each made a GPU-accelerated FASTQ-QC claim, and none of the resulting
tools is obtainable today. Meanwhile CPU tools performing the *same task*
(FastQC, fastp, RabbitQCPlus) remain alive and maintained on GitHub. This
pattern is not random attrition — it has a straightforward algorithmic
explanation, and it is directly relevant to how this manuscript should frame
`fastp-gpu`'s own contribution.

**The task has low arithmetic intensity.** FASTQ QC/trim/filter performs a
handful of comparisons and increments per base (quality-score thresholding,
adapter k-mer matching, base counting) — a streaming pass dominated by I/O
and decompression, not compute. This is already stated in the manuscript's
own related-work discussion (`GPU_ACCELERATED_FASTP_PAPER_TCBB.tex`):
*"FASTQ preprocessing is dominated by I/O and decompression rather than
compute, which offers a less favourable arithmetic-to-memory ratio for GPU
offload."* The GPU tools that *did* survive and thrive in genomics — aligners
(CUSHAW2-GPU, BarraCUDA), deep-learning variant callers (DeepVariant/Parabricks),
error correctors (CARE-GPU) — all have substantially higher compute-per-byte
(dynamic-programming alignment matrices, neural-network inference, k-mer graph
traversal), which amortises PCIe transfer and kernel-launch overhead. QC/trim
does not have that headroom, which is presumably why RabbitQCPlus's authors
chose AVX-512/AVX2 SIMD over GPU: it captures most of the available
parallelism without paying host↔device transfer cost.

**Our own empirical data corroborates this, not just the literature.** In
`benchmark_results/fastp-gpu_v1.3.3-d0bromir/vs_opengene_v1.3.3/galaxy_arm_a100/20260510_172614/full_benchmark_20260510_172614.csv`
(Panel_SE_148M, single-end WGS-scale), `fastp-gpu` vs `fastp` (CPU) wall time
is essentially at parity across thread counts, and GPU is measurably *slower*
at low/medium thread counts where the CPU implementation is not yet
contended:

| Threads | fastp (CPU) wall (s) | fastp-gpu wall (s) | GPU vs CPU |
|---|---|---|---|
| 1 | 7.771 | 8.293 | GPU 6.7% slower |
| 2 | 4.741 | 4.713 | ~tied |
| 4 | 3.964 | 4.007 | ~tied |
| 8 | 3.817 | 3.996 | GPU 4.7% slower |
| 16 | 3.883 | 3.950 | ~tied |
| 32 | 4.376 | 3.949 | GPU 9.8% faster |

The one regime where GPU pulls ahead is at 32 threads, where CPU-side thread
contention on the 128-core Neoverse-N1 host starts to hurt the CPU path more
than GPU dispatch overhead hurts the GPU path — consistent with the
arithmetic-intensity explanation, not a general GPU speed advantage.

**Implication for the manuscript's framing (relevant to Reviewer 2, Weakness
4).** The honest claim is not "GPU beats CPU at this task" but "GPU parity or
modest advantage is achievable in a *genuinely GPU-unfavourable* workload
class, in specific regimes (thread contention, or when downstream
GPU-resident pipeline stages can consume the output without a host round
trip), at zero cost to correctness." The absence of any surviving third-party
GPU competitor — verified against primary sources across two independent
search passes and three academic tool claims from two independent
research groups — is itself
evidence for this framing: the community has not sustained a GPU QC/trim tool
because the arithmetic-to-memory ratio does not reward it, and `fastp-gpu`'s
contribution should be positioned accordingly rather than as an unqualified
throughput win. See `docs/publication/required_changes/JCB/reply-to-reviewer-2.md`
for how this is used in the JCB revision response.

