"""Ablation bar chart for PKN-PSI, drawn from results.csv.

Reads the file experiments.py writes and draws one panel per ablation:

    A1  binning structure        OT instances
    A2  receiver designation     OT instances (log)
    A3  party-count scaling      symmetric-key operations
    A4  gain versus imbalance    transmitted volume (log)

Panel captions sit below each axis. Bars carry hatching as well as colour,
so the figure survives greyscale printing.

    python3 plot_ablation.py
    python3 plot_ablation.py --csv results.csv --out fig_ablation
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito, colorblind-safe.
C_BASE = "#D55E00"       # vermillion
C_PKN = "#0072B2"        # blue
H_BASE = "///"
H_PKN = ""

LABEL_BASE = "Cuckoo baseline"
LABEL_PKN = "PKN-PSI"

CAPTIONS = {
    "A1": "(a) A1. Binning structure (Theorem 2).\n"
          "$m_{\\min}=2000$ throughout, so the count\n"
          "does not depend on $m_{\\max}$.",
    "A2": "(b) A2. Receiver designation (Theorem 3).\n"
          "At $\\delta=1$ the sets coincide, so the\n"
          "residual gain comes from binning.",
    "A3": "(c) A3. Party-count scaling.\n"
          "The shared zero-sharing term grows\n"
          "as $(n+1)M$, narrowing the margin.",
    "A4": "(d) A4. Gain versus imbalance.\n"
          "Both protocols designate the smallest\n"
          "set as receiver.",
}


def load(path):
    """Group rows of results.csv by experiment tag, preserving file order."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{path} is empty")

    groups = defaultdict(list)
    for r in rows:
        groups[r["experiment"]].append(r)
    print(f"read {len(rows)} rows from {path}: "
          + ", ".join(f"{k} ({len(v)})" for k, v in groups.items()))
    return groups


def pair(rows, metric):
    """Split a group into (labels-source, baseline values, PKN values)."""
    base = [r for r in rows if r["protocol"] == "cuckoo"]
    pkn = [r for r in rows if r["protocol"] == "pkn"]
    if len(base) != len(pkn):
        sys.exit("baseline and PKN rows are not paired in results.csv")
    return base, [float(r[metric]) for r in base], [float(r[metric]) for r in pkn]


def draw(axis, labels, base, pkn, ylabel, log=False):
    x = range(len(labels))
    w = 0.38
    axis.bar([i - w / 2 for i in x], base, w, label=LABEL_BASE,
             color=C_BASE, hatch=H_BASE, edgecolor="white", linewidth=0.5)
    axis.bar([i + w / 2 for i in x], pkn, w, label=LABEL_PKN,
             color=C_PKN, hatch=H_PKN, edgecolor="white", linewidth=0.5)

    if log:
        axis.set_yscale("log")
    axis.set_xticks(list(x))
    axis.set_xticklabels(labels)
    axis.set_ylabel(ylabel)
    axis.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    axis.set_axisbelow(True)
    axis.legend(loc="upper left", framealpha=0.9)

    # reduction annotation above the taller bar of each pair
    for i, (b, p) in enumerate(zip(base, pkn)):
        if b <= 0:
            continue
        red = 100.0 * (b - p) / b
        top = max(b, p)
        axis.annotate(f"\u2212{red:.0f}%", xy=(i, top),
                      xytext=(0, 3), textcoords="offset points",
                      ha="center", va="bottom", fontsize=7)

    # headroom so the annotations are not clipped
    if log:
        axis.set_ylim(top=max(max(base), max(pkn)) * 3.0)
    else:
        axis.set_ylim(top=max(max(base), max(pkn)) * 1.18)


def make_figure(groups, out):
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "hatch.linewidth": 0.6,
    })

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 7.2))
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.135, top=0.98,
                        wspace=0.28, hspace=0.72)
    (a1, a2), (a3, a4) = axes

    # --- A1: OT instances, labelled by m_max (m_min is fixed at 2000)
    rows, base, pkn = pair(groups["A1"], "ot_instances")
    labels = [f"$n$={r['n']}\n$\\delta$={int(float(r['delta']))}" for r in rows]
    draw(a1, labels, base, pkn, "OT instances")

    # --- A2: OT instances on a log axis, labelled by delta
    rows, base, pkn = pair(groups["A2"], "ot_instances")
    labels = [f"$\\delta$={int(float(r['delta']))}" for r in rows]
    draw(a2, labels, base, pkn, "OT instances", log=True)

    # --- A3: symmetric-key operations, labelled by party count
    rows, base, pkn = pair(groups["A3"], "sym_ops")
    labels = [f"$n$={r['n']}" for r in rows]
    draw(a3, labels, base, pkn, "Symmetric-key operations")

    # --- A4: transmitted volume on a log axis, labelled by delta
    rows, base, pkn = pair(groups["A4"], "bytes_total")
    labels = [f"$\\delta$={int(float(r['delta']))}" for r in rows]
    draw(a4, labels, base, pkn, "Transmitted volume (bytes)", log=True)

    # captions below each panel
    for axis, tag in ((a1, "A1"), (a2, "A2"), (a3, "A3"), (a4, "A4")):
        pos = axis.get_position()
        fig.text(pos.x0 + pos.width / 2, pos.y0 - 0.055, CAPTIONS[tag],
                 ha="center", va="top", fontsize=7.5, linespacing=1.45)

    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=600)
    plt.close(fig)
    print(f"saved {out}.png and {out}.pdf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results.csv")
    p.add_argument("--out", default="fig_ablation")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"{args.csv} not found; run experiments.py first")

    groups = load(args.csv)
    missing = [t for t in ("A1", "A2", "A3", "A4") if t not in groups]
    if missing:
        sys.exit(f"missing experiment tags in {args.csv}: {missing}")

    make_figure(groups, args.out)


if __name__ == "__main__":
    sys.exit(main())
