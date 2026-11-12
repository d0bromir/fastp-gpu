#!/usr/bin/env python3
"""Summarise the equal-core experiment (raw/equal_core.csv, raw/equal_core_ablation.csv).
Mean +- SD wall time, mean CPU seconds (user+sys), peak RSS, and Welch's t-test for
GPU vs CPU wall time per cell. Pure Python (no scipy)."""
import csv, collections, math, statistics as st, sys, os

def betacf(a, b, x):
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > 1e-300 else 1e-300); h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1 + aa / c if abs(1 + aa / c) > 1e-300 else 1e-300; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1 + aa / c if abs(1 + aa / c) > 1e-300 else 1e-300
        de = d * c; h *= de
        if abs(de - 1) < 1e-12: break
    return h

def ibeta(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2): return math.exp(lb) * betacf(a, b, x) / a
    return 1 - math.exp(lb) * betacf(b, a, 1 - x) / b

def welch(x, y):
    vx, vy = st.variance(x) / len(x), st.variance(y) / len(y)
    t = (st.mean(x) - st.mean(y)) / math.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    return t, ibeta(df / 2, 0.5, df / (df + t * t))

here = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(here, "raw/equal_core.csv"))))
g = collections.defaultdict(list)
for r in rows: g[(r["dataset"], r["layout"], int(r["cores"]), r["mode"])].append(r)
LAB = {("ERR1044780", "SE"): "WGS SE 6.7 GB", ("ERR1044780", "PE"): "WGS PE 13.6 GB",
       ("ERR1044319", "PE"): "WGS PE 19.4 GB", ("DRR216653", "PE"): "WGS PE 41.3 GB"}
print("dataset,cores,mode,finished,runs,wall_mean,wall_sd,cpu_s_mean,rss_gib_max")
stats = {}
for (d, l) in LAB:
    for C in (8, 16, 32):
        for m in ("upstream", "cpu", "gpu"):
            ok = [r for r in g[(d, l, C, m)] if r["exit"] == "0"]
            w = [float(r["wall_s"]) for r in ok]; c = [float(r["user_s"]) + float(r["sys_s"]) for r in ok]
            stats[(d, l, C, m)] = w
            if ok:
                print(f"{LAB[(d,l)]},{C},{m},{len(ok)},{len(g[(d,l,C,m)])},{st.mean(w):.1f},"
                      f"{st.stdev(w) if len(w) > 1 else 0:.1f},{st.mean(c):.0f},"
                      f"{max(int(r['rss_kb']) for r in ok) / 2**20:.1f}")
            else:
                print(f"{LAB[(d,l)]},{C},{m},0,{len(g[(d,l,C,m)])},,,,")
print("\nGPU vs CPU wall (Welch)")
for (d, l) in LAB:
    for C in (8, 16, 32):
        p, c = stats[(d, l, C, "gpu")], stats[(d, l, C, "cpu")]
        t, pv = welch(p, c)
        print(f"{LAB[(d,l)]},{C},{100*(st.mean(p)/st.mean(c)-1):+.1f}%,t={t:.1f},p={pv:.2g}")
ab = list(csv.DictReader(open(os.path.join(here, "raw/equal_core_ablation.csv"))))
print("\nAblation on equal cores (41.3 GB PE)")
for C in ("8", "32"):
    for cfg in ("one_compressor", "no_async"):
        v = [r for r in ab if r["cores"] == C and r["config"] == cfg and r["exit"] == "0"]
        w = [float(r["wall_s"]) for r in v]; c = [float(r["user_s"]) + float(r["sys_s"]) for r in v]
        print(f"{C},{cfg},{len(v)},{st.mean(w):.1f},{st.stdev(w):.1f},{st.mean(c):.0f},{max(int(r['rss_kb']) for r in v)/2**20:.1f}")
xe_path = os.path.join(here, "raw/equal_core_xeon.csv")
if os.path.exists(xe_path):
    xe = list(csv.DictReader(open(xe_path)))
    print("\nXeon host (a2), 19.4 GB PE, cores 0..C-1")
    print("cores,mode,finished,runs,wall_mean,wall_sd,cpu_s_mean,mean_cores")
    xm = {}
    for C in ("8", "16", "32"):
        for m in ("upstream", "cpu"):
            allr = [r for r in xe if r["cores"] == C and r["mode"] == m]
            v = [r for r in allr if r["exit"] == "0" and r["reads_in"] == "234238034"]
            w = [float(r["wall_s"]) for r in v]
            c = [float(r["user_s"]) + float(r["sys_s"]) for r in v]
            sd = st.stdev(w) if len(w) > 1 else float("nan")
            xm[(C, m)] = (w, c)
            print(f"{C},{m},{len(v)},{len(allr)},{st.mean(w):.1f},{sd:.1f},{st.mean(c):.0f},{st.mean(c)/st.mean(w):.1f}")
    print("CPU build vs upstream (cells with >= 2 finished upstream runs)")
    for C in ("8", "16", "32"):
        (wu, cu), (wc, cc) = xm[(C, "upstream")], xm[(C, "cpu")]
        if len(wu) < 2:
            print(f"{C},only {len(wu)} finished upstream run(s)")
            continue
        t, pv = welch(wc, wu)
        print(f"{C},wall {100*(st.mean(wc)/st.mean(wu)-1):+.1f}%,ratio {st.mean(wu)/st.mean(wc):.2f},"
              f"cpu_s {100*(st.mean(cc)/st.mean(cu)-1):+.1f}%,t={t:.1f},p={pv:.2g}")
