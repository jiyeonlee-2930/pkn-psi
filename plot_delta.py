"""Size-imbalance sweep figure for PKN-PSI.

Panel captions are placed below each axis.

Both protocols run on the same datasets. PKN-PSI designates the holder of
the smallest set as receiver; the baseline designates the holder of the
largest set, as in published cuckoo constructions. The baseline setting
therefore corresponds to ablation A2, not A4 -- say so in the caption.

    python3 plot_delta.py                      # run the sweep, then plot
    python3 plot_delta.py --from-csv delta.csv # replot only, no re-run
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Same receiver set size as ablations A1-A4.
M_MIN = 2000
N_PARTIES = 3
DELTAS = (1, 2, 5, 10, 25)
OVERLAP = 0.3
SEED = 3

EPS = 1.27          # bins per element in the cuckoo baseline

# Okabe-Ito, colorblind-safe. Markers differ too, so hue is never load-bearing.
C_BYTES = "#0072B2"      # blue
C_SYM = "#009E73"        # green
C_MEM = "#E69F00"        # orange
C_REF = "0.35"           # grey

CAPTION_A = "(a) Reduction over the baseline widens with size imbalance"
CAPTION_B = r"(b) Measured OT-instance ratio against the closed form $1.27\,\delta$"


def pct(new, old):
    return 0.0 if old == 0 else 100.0 * (old - new) / old


# ------------------------------------------------------------------ sweep

def sweep(deltas, n_parties, m_min, overlap, seed):
    from pknpsi.datagen import make_datasets, imbalanced_sizes
    from pknpsi.protocols import run

    rows = []
    for delta in deltas:
        sizes = imbalanced_sizes(n_parties, m_min, delta)
        ds = make_datasets(sizes, overlap, seed)
        base = run(ds, "cuckoo", "max", seed).row()
        pkn = run(ds, "pkn", "min", seed).row()
        assert base["correct"] and pkn["correct"]
        rows.append({
            "n": n_parties,
            "seed": seed,
            "overlap": overlap,
            "delta": delta,
            "m_min": pkn["m_min"],
            "m_max": pkn["m_max"],
            "bytes_base": base["bytes_total"],
            "bytes_pkn": pkn["bytes_total"],
            "bytes_red": pct(pkn["bytes_total"], base["bytes_total"]),
            "mem_base": base["peak_KiB"],
            "mem_pkn": pkn["peak_KiB"],
            "mem_red": pct(pkn["peak_KiB"], base["peak_KiB"]),
            "sym_base": base["sym_ops"],
            "sym_pkn": pkn["sym_ops"],
            "sym_red": pct(pkn["sym_ops"], base["sym_ops"]),
            "ot_base": base["ot_instances"],
            "ot_pkn": pkn["ot_instances"],
            "ot_ratio": base["ot_instances"] / pkn["ot_instances"],
            "ot_predicted": EPS * delta,
            "bits_per_key": pkn["index_bits"] / sum(sizes),
        })
        print(f"  delta={delta:<4} done", flush=True)
    return rows


def read_rows(path):
    """Load a sweep produced earlier, so the figure can be restyled cheaply."""
    with open(path, newline="") as f:
        raw = list(csv.DictReader(f))
    if not raw:
        sys.exit(f"{path} is empty")

    rows = []
    for r in raw:
        d = int(float(r["delta"]))
        out = {k: float(v) for k, v in r.items()
               if k not in ("n", "seed", "delta", "m_min", "m_max")}
        out["delta"] = d
        out["m_min"] = int(float(r["m_min"]))
        out["m_max"] = int(float(r["m_max"]))
        out["n"] = int(float(r.get("n", N_PARTIES)))
        out["seed"] = int(float(r.get("seed", SEED)))
        # older sweeps did not store the closed-form column
        out.setdefault("ot_predicted", EPS * d)
        rows.append(out)

    rows.sort(key=lambda r: r["delta"])
    print(f"read {len(rows)} rows from {path}")
    return rows


# ----------------------------------------------------------------- report

def print_table(rows):
    n, m_min = rows[0]["n"], rows[0]["m_min"]
    print(f"\nSweep table.  n = {n}, m_min = {m_min}, "
          f"baseline receiver = largest set\n")
    print(f"{'delta':>6} {'m_max':>8} {'bytes base':>12} {'bytes pkn':>11} "
          f"{'transmitted':>12} {'peak mem':>10} {'sym ops':>9} "
          f"{'OT ratio':>9} {'1.27*delta':>11} {'bits/key':>9}")
    for r in rows:
        print(f"{r['delta']:>6} {r['m_max']:>8} "
              f"{int(r['bytes_base']):>12} {int(r['bytes_pkn']):>11} "
              f"{r['bytes_red']:>11.1f}% {r['mem_red']:>9.1f}% "
              f"{r['sym_red']:>8.1f}% {r['ot_ratio']:>9.2f} "
              f"{r['ot_predicted']:>11.2f} {r['bits_per_key']:>9.2f}")

    bk = [r["bits_per_key"] for r in rows]
    print(f"\nbits/key over this sweep: {min(bk):.2f} - {max(bk):.2f}")
    print("Quote this range in the manuscript, not a narrower one.")


# ----------------------------------------------------------------- figure

def make_figure(rows, out):
    deltas = [r["delta"] for r in rows]

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    # No constrained_layout: the captions are figure-level text and need
    # reserved space that automatic layout would not account for.
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 3.1))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.30, top=0.965,
                        wspace=0.30)

    # (a) reductions --------------------------------------------------
    series = [
        ("Transmitted volume", "bytes_red", "o", "-", C_BYTES),
        ("Symmetric-key operations", "sym_red", "^", "--", C_SYM),
        ("Peak memory (implementation-dependent)", "mem_red", "s", ":", C_MEM),
    ]
    for label, key, marker, style, color in series:
        ax.plot(deltas, [r[key] for r in rows], marker=marker, linestyle=style,
                color=color, markersize=4, linewidth=1.2, label=label)

    ax.set_ylabel("Reduction over the baseline (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", framealpha=0.9)

    # (b) ratio against the closed form -------------------------------
    bx.plot(deltas, [r["ot_predicted"] for r in rows], linestyle="-",
            color=C_REF, linewidth=1.0, label=r"Closed form $1.27\,\delta$")
    bx.plot(deltas, [r["ot_ratio"] for r in rows], marker="o", linestyle="none",
            color=C_BYTES, markersize=5, markerfacecolor="none",
            markeredgewidth=1.2, label="Measured")

    bx.set_yscale("log")
    bx.set_ylabel("OT-instance ratio (cuckoo / PKN-PSI)")
    bx.legend(loc="upper left", framealpha=0.9)

    for axis in (ax, bx):
        axis.set_xscale("log")
        axis.set_xticks(deltas)
        axis.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        axis.set_xticklabels([str(d) for d in deltas])
        axis.set_xlabel(r"Size-imbalance index $\delta = m_{\max}/m_{\min}$")
        axis.grid(True, which="major", linewidth=0.4, alpha=0.5)

    # captions below each panel
    for axis, caption in ((ax, CAPTION_A), (bx, CAPTION_B)):
        pos = axis.get_position()
        fig.text(pos.x0 + pos.width / 2, 0.055, caption,
                 ha="center", va="center", fontsize=8, wrap=True)

    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=600)
    plt.close(fig)
    print(f"\nsaved {out}.png and {out}.pdf")


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows written to {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="fig_delta",
                   help="output basename for the figure")
    p.add_argument("--csv", default="delta.csv",
                   help="where the sweep is written")
    p.add_argument("--from-csv", default=None,
                   help="replot from an existing sweep instead of re-running")
    p.add_argument("--deltas", default=",".join(str(d) for d in DELTAS))
    p.add_argument("--n", type=int, default=N_PARTIES)
    p.add_argument("--m-min", type=int, default=M_MIN)
    p.add_argument("--overlap", type=float, default=OVERLAP)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    if args.from_csv:
        if not os.path.exists(args.from_csv):
            sys.exit(f"{args.from_csv} not found")
        rows = read_rows(args.from_csv)
    else:
        deltas = tuple(int(d) for d in args.deltas.split(","))
        print(f"sweeping delta in {deltas} at n={args.n}, "
              f"m_min={args.m_min}, overlap={args.overlap}, seed={args.seed}",
              flush=True)
        rows = sweep(deltas, args.n, args.m_min, args.overlap, args.seed)
        write_csv(rows, args.csv)

    print_table(rows)
    make_figure(rows, args.out)


if __name__ == "__main__":
    sys.exit(main())
