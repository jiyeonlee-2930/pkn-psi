"""Revision experiments (all with real cryptographic operations).

Protocols (same crypto layer, same OKVS payload; they differ only in binning
and in which party is the receiver):

  Baseline   cuckoo binning, largest-set receiver      (KKRT/KMPRT-style)
  PKN-std    cuckoo binning, smallest-set receiver     (standard mode)
  PKN-MPHF   MPHF binning,   smallest-set receiver     (proposed)
  MPHF-max   MPHF binning,   largest-set receiver      (2x2 ablation only)

Experiments
  E1   representative profile n=5, delta=10                    -> Table 3
  F1   delta sweep n=3, delta in {1,2,4,8,16,32}                -> Fig. 1 / Table 4
  A1   binning isolated (receiver = smallest for both)          -> Table 5
  A2   2x2: binning x receiver choice                           -> Table 6
  A3   party count n in {3,5,10,20}                             -> Table 7
  E2   set-size scalability m_min in {2^12, 2^14, 2^16}         -> new table
  LK   empirical membership-inference against the public h*     -> Discussion

Counters are deterministic for a given profile; wall-clock time and peak memory
are repeated R times and reported as mean, std and 95% CI (Student t).

Network model (star topology, all traffic through the receiver's link):
  T_net = 8 * bytes_total / bandwidth + ROUNDS * RTT
  LAN: 1 Gbit/s, RTT 0.5 ms     WAN: 100 Mbit/s, RTT 80 ms

    python experiments_rev.py            # everything
    python experiments_rev.py --only E1 A2 --reps 3
"""

import argparse
import csv
import math
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

from pknpsi.datagen import make_datasets, imbalanced_sizes
from pknpsi.protocols import run

PROTOS = {
    "Baseline": ("cuckoo", "max"),
    "PKN-std": ("cuckoo", "min"),
    "PKN-MPHF": ("pkn", "min"),
    "MPHF-max": ("pkn", "max"),
}
ROUNDS = 6           # sizes, h*/base-OT msg 1, base-OT msg 2+3, U, OKVS, I
NETS = {"LAN": (1e9, 0.5e-3), "WAN": (1e8, 80e-3)}
OVERLAP = 0.3
M_MIN = 2000

OUT = Path("results_rev")
OUT.mkdir(exist_ok=True)
RAW = []


def t95(n):
    # two-sided 95% Student t quantiles for small n
    table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
             8: 2.365, 9: 2.306, 10: 2.262, 15: 2.145, 20: 2.093}
    return table.get(n, 1.96)


MEM_REPS = 3        # peak memory is traced in a separate run for the first reps


def one(exp, label, sizes, rep, extra=None):
    proto, recv = PROTOS[label]
    ds = make_datasets(sizes, overlap=OVERLAP, seed=1000 + rep)
    # timing run: no memory tracing (tracemalloc slows pure-Python loops)
    r = run(ds, protocol=proto, receiver=recv, seed=1000 + rep, trace_memory=False)
    if not r.correct:
        raise RuntimeError(f"incorrect output: {exp} {label} {sizes}")
    peak = ""
    if rep < MEM_REPS:
        peak = run(ds, protocol=proto, receiver=recv, seed=1000 + rep,
                   trace_memory=True).peak_kib
    c = r.counters
    row = dict(exp=exp, protocol=label, n=len(sizes), m_min=min(sizes),
               m_max=max(sizes), delta=round(max(sizes) / min(sizes), 3),
               rep=rep, correct=r.correct, recv_size=r.receiver_size,
               ot_instances=c.ot_instances, sym_ops=c.sym_ops,
               bins=c.bins, stash=c.stash, bytes_total=c.bytes_total,
               bytes_r2s=c.bytes_r2s, bytes_s2r=c.bytes_s2r,
               bytes_index=c.bytes_index, bytes_baseot=c.bytes_baseot,
               index_bits=c.index_bits, okvs_retries=c.okvs_retries,
               sec=r.seconds, t_mphf=c.t_mphf, t_baseot=c.t_baseot,
               t_okvs=c.t_okvs, peak_KiB=peak)
    for net, (bw, rtt) in NETS.items():
        row[f"t_net_{net}"] = 8 * c.bytes_total / bw + ROUNDS * rtt
        row[f"t_total_{net}"] = row["sec"] + row[f"t_net_{net}"]
    if extra:
        row.update(extra)
    RAW.append(row)
    return row


def sweep(exp, configs, labels, reps):
    for sizes in configs:
        for label in labels:
            for rep in range(reps):
                row = one(exp, label, sizes, rep)
                print(f"[{exp}] {label:9s} n={row['n']:2d} d={row['delta']:5.1f} "
                      f"rep={rep} bytes={row['bytes_total']:>10,} "
                      f"sec={row['sec']:.2f}", flush=True)
        save()


def save():
    if not RAW:
        return
    keys = list(dict.fromkeys(k for r in RAW for k in r))
    with open(OUT / "raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(RAW)
    summarize()


METRICS = ["ot_instances", "sym_ops", "bytes_total", "peak_KiB", "sec",
           "t_mphf", "t_baseot", "t_okvs", "t_total_LAN", "t_total_WAN"]


def summarize():
    groups = {}
    for r in RAW:
        k = (r["exp"], r["protocol"], r["n"], r["m_min"], r["m_max"])
        groups.setdefault(k, []).append(r)
    rows = []
    for (exp, proto, n, mmin, mmax), rs in groups.items():
        out = dict(exp=exp, protocol=proto, n=n, m_min=mmin, m_max=mmax,
                   delta=round(mmax / mmin, 3), reps=len(rs))
        for m in METRICS:
            v = [float(r[m]) for r in rs if r[m] != ""]
            mu = st.mean(v)
            sd = st.stdev(v) if len(v) > 1 else 0.0
            out[f"{m}_mean"] = mu
            out[f"{m}_sd"] = sd
            out[f"{m}_ci95"] = t95(len(v)) * sd / math.sqrt(len(v)) if len(v) > 1 else 0.0
        rows.append(out)
    keys = list(rows[0].keys())
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def leakage(reps):
    """Membership inference by a sender against the public h*."""
    from scipy.stats import rankdata
    from pknpsi.mphf import MPHF, U64
    rows = []
    for delta in (1, 2, 5, 10):
        for frac in (0.3, 1.0):
            if delta == 1 and frac == 1.0:
                continue          # D_j = D*: no non-members, AUC undefined
            m_star, m_j = M_MIN, M_MIN * delta
            k = int(frac * m_star)
            aucs, real, sim = [], [], []
            for s in range(reps):
                rng = np.random.default_rng(s)
                u = np.unique(rng.integers(0, 2 ** 63, size=m_star + m_j + 10,
                                           dtype=np.int64).astype(U64))
                rng.shuffle(u)
                common = u[:k]
                D_j = np.concatenate([common, u[k:m_j]])
                D_s = np.concatenate([common, u[m_j:m_j + m_star - k]])
                h = MPHF(D_s, alpha=0.94, seed=s)
                pos = h(D_j)
                _, inv, cnt = np.unique(pos, return_inverse=True, return_counts=True)
                score = -(cnt[inv] - 1).astype(float)
                lab = np.zeros(m_j, bool)
                lab[:k] = True
                if 0 < k < m_j:
                    r_ = rankdata(score)
                    n1, n0 = lab.sum(), (~lab).sum()
                    aucs.append((r_[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
            rows.append(dict(delta=delta, overlap=frac, m_star=m_star, m_j=m_j,
                             reps=reps, auc_mean=st.mean(aucs) if aucs else float("nan"),
                             auc_sd=st.stdev(aucs) if len(aucs) > 1 else 0.0))
            print(f"[LK] delta={delta} overlap={frac} AUC={rows[-1]['auc_mean']:.3f}", flush=True)
    with open(OUT / "leakage.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    want = lambda e: a.only is None or e in a.only
    R = a.reps
    R_small = max(2, R // 2)
    t0 = time.time()

    if want("E1"):
        sweep("E1", [imbalanced_sizes(5, M_MIN, 10)],
              ["Baseline", "PKN-std", "PKN-MPHF"], R)
    if want("A1"):
        cfg = [[M_MIN] + [mx] * (n - 1) for n in (3, 5) for mx in (2000, 8000)]
        sweep("A1", cfg, ["PKN-std", "PKN-MPHF"], R_small)
    if want("A2"):
        sweep("A2", [imbalanced_sizes(3, M_MIN, d) for d in (1, 2, 10, 50)],
              ["Baseline", "PKN-std", "MPHF-max", "PKN-MPHF"], R_small)
    if want("A3"):
        sweep("A3", [imbalanced_sizes(n, M_MIN, 5) for n in (3, 5, 10, 20)],
              ["Baseline", "PKN-std", "PKN-MPHF"], R_small)
    if want("F1"):
        sweep("F1", [imbalanced_sizes(3, M_MIN, d) for d in (1, 2, 4, 8, 16, 32)],
              ["Baseline", "PKN-std", "PKN-MPHF"], R)
    if want("E2"):
        sweep("E2", [imbalanced_sizes(3, m, 4) for m in (2 ** 12, 2 ** 14, 2 ** 16)],
              ["Baseline", "PKN-MPHF"], max(2, R // 3))
    if want("LK"):
        leakage(R)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
