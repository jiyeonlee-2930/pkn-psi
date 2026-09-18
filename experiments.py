"""Experiments A1-A4 and the scalability study E2.

Every figure printed here is implementation-independent (counts and bytes)
except the columns explicitly labelled as seconds and peak memory.

All ablations share the same receiver set size M_MIN so that A1-A4 each
vary exactly one controlled factor.

    python3 experiments.py            # A1, A2, A3, A4
    python3 experiments.py --scale    # add E2 up to 2^18
    python3 experiments.py --verify   # check the closed-form cost model
"""

import argparse
import csv
import math
import sys

import numpy as np

from pknpsi.datagen import make_datasets, imbalanced_sizes
from pknpsi.protocols import run, out_bits

ROWS = []

# Receiver set size shared by every ablation.
M_MIN = 2000

# Codeword width of the OT extension matrix (w in Section IV-A).
CODEWORD_BITS = 448

# Imbalance ratios swept by A4. The cuckoo baseline places every element
# three times, so its running time grows steeply with m_max; delta = 25
# already puts 50,000 elements in each sender set and is the largest ratio
# that completes in reasonable time. The extreme regime is covered by A2,
# which reaches delta = 50 while measuring OT instances only.
A4_DELTAS = (1, 2, 5, 10, 25)

VERIFY = False


def record(tag, r):
    d = r.row()
    d["experiment"] = tag
    ROWS.append(d)
    return d


def pct(new, old):
    return 0.0 if old == 0 else 100.0 * (old - new) / old


def hdr(title):
    print(f"\n{'=' * 92}\n{title}\n{'=' * 92}")


def bits_per_key(sizes, row):
    """Measured size of the broadcast MPHF index, in bits per key."""
    return row["index_bits"] / sum(sizes)


def verify_cost_model(sizes, row):
    """Compare the measured byte counters against the closed-form model.

    model = (n-1)*m_min*w  +  M*l  +  sum_j |h_j|
    where M = sum_j m_j - m_min and l = min(64, sigma + ceil(log2(n*m_max))).
    """
    n = len(sizes)
    m_min, m_max = min(sizes), max(sizes)
    M = sum(sizes) - m_min
    ell = out_bits(n, m_max)

    model_r2s = (n - 1) * m_min * CODEWORD_BITS // 8
    model_s2r = M * ell // 8
    model_idx = row["index_bits"] // 8
    model_tot = model_r2s + model_s2r + model_idx

    print(f"      cost model   r2s {model_r2s:>10} | s2r {model_s2r:>10} | "
          f"idx {model_idx:>8} | total {model_tot:>11}")
    print(f"      measured     r2s {row['bytes_r2s']:>10} | "
          f"s2r {row['bytes_s2r']:>10} | idx {row['bytes_index']:>8} | "
          f"total {row['bytes_total']:>11}")
    delta = row["bytes_total"] - model_tot
    verdict = "exact" if delta == 0 else f"MISMATCH {delta:+d} bytes"
    print(f"      l = {ell}, M = {M}  ->  {verdict}")


# --------------------------------------------------------------- A1
def a1(seed=1):
    hdr("A1  binning structure: cuckoo -> MPHF   (Theorem 2)")
    print(f"{'n':>3} {'m_min':>7} {'m_max':>8} "
          f"{'OT cuckoo':>10} {'OT mphf':>9} {'red%':>6} "
          f"{'plc cuckoo':>11} {'plc mphf':>9} {'red%':>6} "
          f"{'bytes cuckoo':>13} {'bytes mphf':>11} {'red%':>6} "
          f"{'bits/key':>9}")
    for n, delta in [(3, 1), (3, 4), (5, 1), (5, 4)]:
        sizes = imbalanced_sizes(n, M_MIN, delta)
        ds = make_datasets(sizes, 0.3, seed)
        a = record("A1", run(ds, "cuckoo", "min", seed))
        b = record("A1", run(ds, "pkn", "min", seed))
        assert a["correct"] and b["correct"]
        print(f"{n:>3} {a['m_min']:>7} {a['m_max']:>8} "
              f"{a['ot_instances']:>10} {b['ot_instances']:>9} "
              f"{pct(b['ot_instances'], a['ot_instances']):>6.1f} "
              f"{a['placements']:>11} {b['placements']:>9} "
              f"{pct(b['placements'], a['placements']):>6.1f} "
              f"{a['bytes_total']:>13} {b['bytes_total']:>11} "
              f"{pct(b['bytes_total'], a['bytes_total']):>6.1f} "
              f"{bits_per_key(sizes, b):>9.2f}")
        if VERIFY:
            verify_cost_model(sizes, b)


# --------------------------------------------------------------- A2
def a2(seed=2):
    hdr("A2  receiver designation: largest set -> smallest set   (Theorem 3)")
    print(f"{'delta':>6} {'m_min':>7} {'m_max':>8} "
          f"{'OT recv=max':>12} {'OT recv=min':>12} {'ratio':>7} "
          f"{'1.27*delta':>11}")
    for delta in (1, 2, 10, 50):
        sizes = imbalanced_sizes(3, M_MIN, delta)
        ds = make_datasets(sizes, 0.3, seed)
        a = record("A2", run(ds, "cuckoo", "max", seed))
        b = record("A2", run(ds, "pkn", "min", seed))
        assert a["correct"] and b["correct"]
        print(f"{delta:>6} {b['m_min']:>7} {b['m_max']:>8} "
              f"{a['ot_instances']:>12} {b['ot_instances']:>12} "
              f"{a['ot_instances'] / b['ot_instances']:>7.2f} "
              f"{1.27 * delta:>11.2f}")


# --------------------------------------------------------------- A3
def a3(seed=4):
    hdr("A3  scaling in the number of parties")
    print(f"{'n':>3} {'m_min':>7} {'OT cuckoo':>10} {'OT mphf':>9} {'red%':>6} "
          f"{'sym cuckoo':>12} {'sym mphf':>11} {'red%':>6} "
          f"{'bytes cuckoo':>13} {'bytes mphf':>11} {'red%':>6} {'|I|':>6}")
    for n in (3, 5, 10, 20):
        sizes = imbalanced_sizes(n, M_MIN, 5)
        ds = make_datasets(sizes, 0.4, seed)
        a = record("A3", run(ds, "cuckoo", "min", seed))
        b = record("A3", run(ds, "pkn", "min", seed))
        assert a["correct"] and b["correct"]
        print(f"{n:>3} {b['m_min']:>7} "
              f"{a['ot_instances']:>10} {b['ot_instances']:>9} "
              f"{pct(b['ot_instances'], a['ot_instances']):>6.1f} "
              f"{a['sym_ops']:>12} {b['sym_ops']:>11} "
              f"{pct(b['sym_ops'], a['sym_ops']):>6.1f} "
              f"{a['bytes_total']:>13} {b['bytes_total']:>11} "
              f"{pct(b['bytes_total'], a['bytes_total']):>6.1f} {b['|I|']:>6}")
        if VERIFY:
            verify_cost_model(sizes, b)


# --------------------------------------------------------------- A4
def a4(seed=5):
    hdr("A4  size-imbalance regimes")
    print(f"{'delta':>6} {'m_min':>7} {'m_max':>8} "
          f"{'OT cuckoo':>10} {'OT mphf':>9} {'red%':>6} "
          f"{'bytes cuckoo':>13} {'bytes mphf':>11} {'red%':>6} "
          f"{'peak red%':>10} {'bits/key':>9}")
    for delta in A4_DELTAS:
        sizes = imbalanced_sizes(3, M_MIN, delta)
        ds = make_datasets(sizes, 0.3, seed)
        a = record("A4", run(ds, "cuckoo", "min", seed))
        b = record("A4", run(ds, "pkn", "min", seed))
        assert a["correct"] and b["correct"]
        print(f"{delta:>6} {b['m_min']:>7} {b['m_max']:>8} "
              f"{a['ot_instances']:>10} {b['ot_instances']:>9} "
              f"{pct(b['ot_instances'], a['ot_instances']):>6.1f} "
              f"{a['bytes_total']:>13} {b['bytes_total']:>11} "
              f"{pct(b['bytes_total'], a['bytes_total']):>6.1f} "
              f"{pct(b['peak_KiB'], a['peak_KiB']):>10.1f} "
              f"{bits_per_key(sizes, b):>9.2f}", flush=True)
        if VERIFY:
            verify_cost_model(sizes, b)


# --------------------------------------------------------------- E2
def e2(seed=6):
    hdr("E2  scalability")
    print(f"{'m_min':>8} {'m_max':>9} {'OT cuckoo':>10} {'OT mphf':>9} "
          f"{'red%':>6} {'bytes red%':>10} {'sec cuckoo':>11} {'sec mphf':>9} "
          f"{'bits/key':>9}")
    for e in (12, 14, 16, 18):
        m = 2 ** e
        sizes = imbalanced_sizes(3, m, 4)
        ds = make_datasets(sizes, 0.2, seed)
        a = record("E2", run(ds, "cuckoo", "min", seed))
        b = record("E2", run(ds, "pkn", "min", seed))
        assert a["correct"] and b["correct"]
        print(f"{a['m_min']:>8} {a['m_max']:>9} "
              f"{a['ot_instances']:>10} {b['ot_instances']:>9} "
              f"{pct(b['ot_instances'], a['ot_instances']):>6.1f} "
              f"{pct(b['bytes_total'], a['bytes_total']):>10.1f} "
              f"{a['sec']:>11.2f} {b['sec']:>9.2f} "
              f"{bits_per_key(sizes, b):>9.2f}")


def main():
    global VERIFY
    p = argparse.ArgumentParser()
    p.add_argument("--scale", action="store_true")
    p.add_argument("--verify", action="store_true",
                   help="print a term-by-term check of the closed-form model")
    p.add_argument("--csv", default="results.csv")
    args = p.parse_args()
    VERIFY = args.verify

    try:
        a1(); a2(); a3(); a4()
        if args.scale:
            e2()
    except KeyboardInterrupt:
        print("\ninterrupted - writing the rows collected so far", flush=True)

    if ROWS:
        # epsilon was removed from the manuscript together with Lemma 1.
        drop = {"epsilon"}
        cols = sorted({k for r in ROWS for k in r} - drop)
        cols = ["experiment", "protocol"] + [c for c in cols
                                             if c not in ("experiment", "protocol")]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(ROWS)
        print(f"\n{len(ROWS)} rows written to {args.csv}")


if __name__ == "__main__":
    sys.exit(main())
