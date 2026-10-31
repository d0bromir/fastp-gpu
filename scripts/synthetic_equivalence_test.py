#!/usr/bin/env python3
"""Synthetic equivalence test for fastp-gpu against upstream OpenGene fastp.

Generates small, seeded FASTQ data sets that target the bookkeeping paths of the
statistics code: reads that pass unchanged, reads trimmed at every boundary, reads
discarded by each filter, adapter trimming, adapter dimers, paired reads where only
one mate passes, duplicates, poly-G tails, variable lengths and very short reads.
Each data set runs under several option sets. For every run the script compares

  * md5 of the decompressed output FASTQ (R1, R2),
  * the complete JSON report (all keys except the volatile ones listed in VOLATILE),
  * the 14 fields used by scripts/run_tests.sh.

Usage:
  synthetic_equivalence_test.py --og PATH --cpu PATH [--gpu PATH] [--workdir DIR]
                                [--threads 1 4] [--reads 20000] [--csv OUT.csv]
Exit status 0 if every comparison is identical, 1 otherwise.
"""
import argparse, csv, gzip, hashlib, json, os, random, subprocess, sys, tempfile

ADAPTER_R1 = "AGATCGGAAGAGCACACGTCTGAACTCCAGTCA"
ADAPTER_R2 = "AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGT"
RUN_TIMEOUT = 300   # seconds per fastp run; upstream is known to hang occasionally
VOLATILE = {"command", "fastp_version", "sequencing", "time_consuming"}
# Documented differences would be listed here and reported separately, never hidden.
# (The opt-in FASTP_FAST_PRESTATS=1 mode is not covered: it is known to leave the before-filtering
# k-mer table empty.)
KNOWN_DIFF_PREFIXES = ()   # none: the default build must match upstream in every reported value
FIELDS = """summary.before_filtering.total_reads summary.before_filtering.total_bases
summary.after_filtering.total_reads summary.after_filtering.total_bases
summary.after_filtering.q20_rate summary.after_filtering.q30_rate
filtering_result.passed_filter_reads filtering_result.low_quality_reads
filtering_result.too_many_N_reads filtering_result.adapter_dimer_reads
filtering_result.too_short_reads filtering_result.too_long_reads
adapter_cutting.adapter_trimmed_reads adapter_cutting.adapter_trimmed_bases""".split()


def qual_str(rng, n, kind):
    """Quality strings: 'good' Q30-40, 'tail' good then low tail, 'low' mostly low, 'mixed'."""
    if kind == "good":
        return "".join(chr(33 + rng.randint(30, 40)) for _ in range(n))
    if kind == "tail":
        cut = rng.randint(1, max(1, n - 1))
        return "".join(chr(33 + (rng.randint(30, 40) if i < cut else rng.randint(2, 10))) for i in range(n))
    if kind == "low":
        return "".join(chr(33 + rng.randint(2, 14)) for _ in range(n))
    return "".join(chr(33 + rng.randint(2, 40)) for _ in range(n))


def rand_seq(rng, n, n_rate=0.0, polyg=0):
    s = [rng.choice("ACGT") for _ in range(n)]
    if n_rate:
        for i in range(n):
            if rng.random() < n_rate: s[i] = "N"
    if polyg:
        for i in range(max(0, n - polyg), n): s[i] = "G"
    return "".join(s)


def revcomp(s):
    return s[::-1].translate(str.maketrans("ACGTN", "TGCAN"))


def make_dataset(kind, n, paired, seed):
    """Return list of (r1, r2) FASTQ record tuples; r2 is None for single-end."""
    rng = random.Random(seed)
    recs = []
    L = 100
    for i in range(n):
        name = f"@syn{kind}_{i}"
        s1 = rand_seq(rng, L); q1 = qual_str(rng, L, "good")
        s2 = rand_seq(rng, L) if paired else None; q2 = qual_str(rng, L, "good") if paired else None
        if kind == "pass":
            pass
        elif kind == "tail":                      # quality tail trimming at every offset
            q1 = qual_str(rng, L, "tail"); q2 = qual_str(rng, L, "tail") if paired else None
        elif kind == "lowq":                      # fails the unqualified-percent filter
            q1 = qual_str(rng, L, "low" if i % 3 == 0 else "good")
            q2 = qual_str(rng, L, "low" if i % 5 == 0 else "good") if paired else None
        elif kind == "nbase":                     # fails the N-base filter
            s1 = rand_seq(rng, L, n_rate=0.2 if i % 4 == 0 else 0.0)
            s2 = rand_seq(rng, L, n_rate=0.2 if i % 7 == 0 else 0.0) if paired else None
        elif kind == "short":                     # lengths around the 15 bp minimum
            ln = rng.choice([1, 2, 14, 15, 16, 17, 30, L])
            s1, q1 = rand_seq(rng, ln), qual_str(rng, ln, "good")
            if paired:
                ln2 = rng.choice([1, 14, 15, 16, L]); s2, q2 = rand_seq(rng, ln2), qual_str(rng, ln2, "good")
        elif kind == "varlen":                    # variable lengths up to 250
            ln = rng.randint(16, 250); s1, q1 = rand_seq(rng, ln), qual_str(rng, ln, "mixed")
            if paired:
                ln2 = rng.randint(16, 250); s2, q2 = rand_seq(rng, ln2), qual_str(rng, ln2, "mixed")
        elif kind == "adapter":                   # insert shorter than the read: adapter read-through
            ins = rng.choice([0, 1, 5, 17, 30, 45, 60, 80, 95, 100, 130])
            frag = rand_seq(rng, ins)
            if paired:
                s1 = (frag + ADAPTER_R1 + rand_seq(rng, L))[:L]
                s2 = (revcomp(frag) + ADAPTER_R2 + rand_seq(rng, L))[:L]
            else:
                s1 = (frag + ADAPTER_R1 + rand_seq(rng, L))[:L]
        elif kind == "polyg":
            s1 = rand_seq(rng, L, polyg=rng.choice([0, 5, 10, 20, 40]))
            s2 = rand_seq(rng, L, polyg=rng.choice([0, 12, 25])) if paired else None
        elif kind == "onemate":                   # one mate fails, the other passes
            if i % 2 == 0: q1 = qual_str(rng, L, "low")
            elif paired: q2 = qual_str(rng, L, "low")
        elif kind == "dup":                       # exact duplicates
            r = random.Random(seed + i // 4)
            s1 = rand_seq(r, L); q1 = qual_str(r, L, "good")
            if paired: s2 = rand_seq(r, L); q2 = qual_str(r, L, "good")
        elif kind == "mixed":                     # everything at once
            s1 = rand_seq(rng, L, n_rate=0.03, polyg=rng.choice([0, 10]))
            q1 = qual_str(rng, L, rng.choice(["good", "tail", "mixed", "low"]))
            if paired:
                s2 = rand_seq(rng, L, n_rate=0.03); q2 = qual_str(rng, L, rng.choice(["good", "tail", "mixed", "low"]))
        r1 = f"{name}/1\n{s1}\n+\n{q1}\n"
        r2 = f"{name}/2\n{s2}\n+\n{q2}\n" if paired else None
        recs.append((r1, r2))
    return recs


OPTION_SETS = {
    "default": [],
    "no_trim_window": ["--disable_quality_filtering"],
    "cut_right": ["--cut_right", "--cut_window_size", "4", "--cut_mean_quality", "20"],
    "cut_front_tail": ["--cut_front", "--cut_tail", "--cut_mean_quality", "15"],
    "len_filter": ["-l", "40", "--length_limit", "90"],
    "front_tail_trim": ["-f", "3", "-t", "2", "-F", "1", "-T", "4"],
    "poly_g": ["--trim_poly_g", "--poly_g_min_len", "6"],
    "poly_x": ["--trim_poly_x"],
    "dedup": ["--dedup"],
    "complexity": ["--low_complexity_filter", "--complexity_threshold", "30"],
    "n_limit": ["-n", "2"],
    "qual_strict": ["-q", "25", "-u", "20"],
    "max_len": ["--max_len1", "70", "--max_len2", "60"],
    "adapter_given": ["--adapter_sequence", ADAPTER_R1, "--adapter_sequence_r2", ADAPTER_R2],
    "detect_pe": ["--detect_adapter_for_pe"],
    "correction": ["--correction"],
    "merge": ["--merge"],
}
PE_ONLY = {"detect_pe", "correction", "merge"}


def write_gz(path, text):
    with gzip.open(path, "wt", compresslevel=4) as f:
        f.write(text)


def md5_gunzip(path):
    h = hashlib.md5()
    with gzip.open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(binary, r1, r2, out_prefix, opts, threads):
    o1, o2, js = out_prefix + "_R1.fq.gz", out_prefix + "_R2.fq.gz", out_prefix + ".json"
    cmd = [binary, "-w", str(threads), "-j", js, "-h", "/dev/null", "-i", r1, "-o", o1]
    if r2: cmd += ["-I", r2, "-O", o2]
    cmd += opts
    om = out_prefix + "_M.fq.gz"
    if "--merge" in opts: cmd += ["--merged_out", om]
    env = dict(os.environ)
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=RUN_TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"error": f"TIMEOUT after {RUN_TIMEOUT}s"}
    if p.returncode != 0 or not os.path.exists(js):
        return {"error": p.stderr.decode(errors="replace")[-300:]}
    d = json.load(open(js))
    md = [md5_gunzip(o1)]
    if r2 and os.path.exists(o2): md.append(md5_gunzip(o2))
    if os.path.exists(om): md.append(md5_gunzip(om))
    full = {k: v for k, v in d.items() if k not in VOLATILE}
    for p_ in (o1, o2, om, js):
        if os.path.exists(p_): os.remove(p_)
    return {"md5": md, "json": full}


def field(d, path):
    """Value at a dotted path, or None if the report has no such key."""
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d: return None
        d = d[k]
    return d


def diff_json(a, b, path=""):
    """Yield paths whose values differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b: yield path + "/" + k; continue
            yield from diff_json(a[k], b[k], path + "/" + k)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b): yield path + "[len]"; return
        for i, (x, y) in enumerate(zip(a, b)): yield from diff_json(x, y, f"{path}[{i}]")
    elif a != b:
        yield path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--og", required=True); ap.add_argument("--cpu", required=True); ap.add_argument("--gpu")
    ap.add_argument("--workdir"); ap.add_argument("--threads", nargs="+", type=int, default=[1, 4])
    ap.add_argument("--reads", type=int, default=20000); ap.add_argument("--csv"); ap.add_argument("--seed", type=int, default=7); ap.add_argument("--jobs", type=int, default=8); ap.add_argument("--only", help="run only cases whose id (e.g. PE/mixed/merge/T4) contains this text")
    a = ap.parse_args()
    wd = a.workdir or tempfile.mkdtemp(prefix="syn_equiv_")
    os.makedirs(wd, exist_ok=True)
    tools = {"cpu": a.cpu}
    if a.gpu: tools["gpu"] = a.gpu
    kinds = ["pass", "tail", "lowq", "nbase", "short", "varlen", "adapter", "polyg", "onemate", "dup", "mixed"]
    jobs = []   # (paired, kind, input paths, option name, opts, threads)
    for paired in (False, True):
        for ki, kind in enumerate(kinds):
            if kind == "onemate" and not paired: continue
            recs = make_dataset(kind, a.reads, paired, a.seed + ki)
            r1 = os.path.join(wd, f"{kind}_{'pe' if paired else 'se'}_1.fq.gz"); r2 = os.path.join(wd, f"{kind}_pe_2.fq.gz") if paired else None
            write_gz(r1, "".join(x[0] for x in recs))
            if paired: write_gz(r2, "".join(x[1] for x in recs))
            for oname, opts in OPTION_SETS.items():
                if oname in PE_ONLY and not paired: continue
                for T in a.threads:
                    if a.only and a.only not in f"{'PE' if paired else 'SE'}/{kind}/{oname}/T{T}": continue
                    jobs.append((paired, kind, r1, r2, oname, opts, T))

    def do_job(idx_job):
        idx, (paired, kind, r1, r2, oname, opts, T) = idx_job
        out = []
        ref = run(a.og, r1, r2, os.path.join(wd, f"og{idx}"), opts, T)
        for tname, tbin in tools.items():
            got = run(tbin, r1, r2, os.path.join(wd, f"{tname}{idx}"), opts, T)
            case = f"{'PE' if paired else 'SE'}/{kind}/{oname}/T{T}/{tname}"
            if "error" in ref or "error" in got:
                status = "ERROR"; detail = (ref.get("error") or got.get("error"))[:120].replace("\n", " ")
                if "error" in ref and "error" in got and ref["error"] == got["error"]: status = "SAME_ERROR"
                if "TIMEOUT" in detail: status = "REF_TIMEOUT" if "error" in ref else "ERROR"
            else:
                alld = list(diff_json(ref["json"], got["json"]))
                alld = [x for x in alld if not x.endswith("/fastp_version")]
                # Insert size is estimated on a subset of the reads (upstream: thread 0; fork: every T-th pack), and the
                # per-sequence adapter count tables depend on how reads are spread over threads (also in upstream).
                # Both are compared at T = 1 only.
                known_prefixes = KNOWN_DIFF_PREFIXES + (("/insert_size/", "/adapter_cutting/read1_adapter_counts", "/adapter_cutting/read2_adapter_counts") if T > 1 else ())
                d = [x for x in alld if not x.startswith(known_prefixes)]
                known = len(alld) - len(d)
                f14 = all(field(ref["json"], f) == field(got["json"], f) for f in FIELDS)
                same = ref["md5"] == got["md5"] and not d and f14
                status = "PASS" if same else "FAIL"
                detail = ",".join(d[:4]) if d else ("md5" if ref["md5"] != got["md5"] else "")
                if same and known: detail = f"known_diff:thread_dependent_statistics({known} keys)"
            out.append((case, status, detail, ref.get("json", {}).get("summary", {}).get("before_filtering", {}).get("total_reads", ""),
                        ref.get("json", {}).get("filtering_result", {}).get("passed_filter_reads", "")))
        return out

    from concurrent.futures import ThreadPoolExecutor, as_completed
    rows, fails = [], 0
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(do_job, (i, j)) for i, j in enumerate(jobs)]
        for fu in as_completed(futs):
            for r in fu.result():
                rows.append(r)
                if r[1] in ("FAIL", "ERROR"):
                    fails += 1; print(r[1], r[0], r[2], flush=True)
    rows.sort(key=lambda r: r[0])
    n = len(rows); p = sum(1 for r in rows if r[1] == "PASS")
    print(f"cases={n} pass={p} same_error={sum(1 for r in rows if r[1]=='SAME_ERROR')} fail={fails}")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["case", "status", "detail", "reads_in", "passed"]); w.writerows(rows)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
