"""Figures and manuscript tables from results_rev/ (output of experiments_rev.py).

    python plot_rev.py

Writes
  results_rev/fig1_delta.{png,pdf}     Fig. 1  delta sweep (F1)
  results_rev/fig2_ablation.{png,pdf}  Fig. 2  A1, A2 (2x2), A3, E1 runtime split
  results_rev/tables.md                Tables 3-7, E2 and the leakage table,
                                       ready to paste into the manuscript
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results_rev")

# Okabe-Ito based, validated for CVD; every series also has its own marker/line.
STYLE = {
    "Baseline": dict(color="#6B6B6B", marker="s", ls="--", label="Baseline (cuckoo, largest receiver)"),
    "PKN-std":  dict(color="#D55E00", marker="^", ls="-.", label="PKN-PSI standard mode (cuckoo)"),
    "PKN-MPHF": dict(color="#0072B2", marker="o", ls="-",  label="PKN-PSI (MPHF)"),
    "MPHF-max": dict(color="#009E73", marker="D", ls=":",  label="MPHF, largest receiver"),
}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
                     "legend.frameon": False})


def load():
    rows = list(csv.DictReader(open(R / "summary.csv")))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except ValueError:
                pass
    return rows


def pick(rows, exp, proto):
    return sorted([r for r in rows if r["exp"] == exp and r["protocol"] == proto],
                  key=lambda r: (r["n"], r["delta"], r["m_max"]))


def red(new, old):
    return 100.0 * (old - new) / old if old else 0.0


def fig1(rows):
    base, pkn = pick(rows, "F1", "Baseline"), pick(rows, "F1", "PKN-MPHF")
    if not base:
        return
    d = [r["delta"] for r in pkn]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.4))
    series = [("bytes_total_mean", "Transmitted volume", "o", "-", "#0072B2"),
              ("sym_ops_mean", "Symmetric-key operations", "^", "--", "#D55E00"),
              ("peak_KiB_mean", "Peak memory", "s", ":", "#009E73"),
              ("t_total_WAN_mean", "End-to-end time (WAN model)", "D", "-.", "#CC79A7")]
    for key, lab, mk, ls, col in series:
        a.plot(d, [red(p[key], q[key]) for p, q in zip(pkn, base)],
               marker=mk, ls=ls, color=col, lw=1.6, ms=5, label=lab)
    a.set_xscale("log", base=2)
    a.set_xticks(d, [f"{x:g}" for x in d])
    a.set_xlabel("Size-imbalance index δ = m_max / m_min")
    a.set_ylabel("Reduction over baseline (%)")
    a.set_ylim(0, 100)
    a.set_title("(a) Reduction widens with imbalance", fontsize=9, loc="left")
    a.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)

    ratio = [q["ot_instances_mean"] / p["ot_instances_mean"] for p, q in zip(pkn, base)]
    b.plot(d, [1.27 * x for x in d], color="#6B6B6B", lw=1.2, label="Closed form 1.27·δ")
    b.plot(d, ratio, "o", color="#0072B2", ms=6, mfc="none", mew=1.5, label="Measured")
    b.set_xscale("log", base=2)
    b.set_yscale("log")
    b.set_xticks(d, [f"{x:g}" for x in d])
    b.set_xlabel("Size-imbalance index δ")
    b.set_ylabel("OT-instance ratio (baseline / PKN-PSI)")
    b.set_title("(b) OT instances: measured vs. 1.27·δ", fontsize=9, loc="left")
    b.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(R / f"fig1_delta.{ext}", dpi=300)


def bars(ax, groups, protos, key, err=None, log=False):
    import numpy as np
    x = np.arange(len(groups))
    w = 0.8 / len(protos)
    for i, p in enumerate(protos):
        vals = [g.get(p, {}).get(key, 0) for g in groups]
        e = [g.get(p, {}).get(err, 0) for g in groups] if err else None
        st = STYLE[p]
        ax.bar(x + (i - (len(protos) - 1) / 2) * w, vals, w * 0.92, yerr=e,
               color=st["color"], label=st["label"], capsize=2,
               edgecolor="white", linewidth=0.8)
    if log:
        ax.set_yscale("log")
    return x


def fig2(rows):
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6))

    # A1: binning isolated
    ax = axs[0, 0]
    keys = sorted({(r["n"], r["m_max"]) for r in rows if r["exp"] == "A1"})
    groups = [{p: r for p in ("PKN-std", "PKN-MPHF") for r in pick(rows, "A1", p)
               if (r["n"], r["m_max"]) == k} for k in keys]
    x = bars(ax, groups, ["PKN-std", "PKN-MPHF"], "ot_instances_mean")
    ax.set_xticks(x, [f"n={int(n)}\nm_max={int(m) // 1000}k" for n, m in keys])
    ax.set_ylabel("OT instances")
    ax.set_title("A1. Binning only (receiver = smallest)", fontsize=9, loc="left")

    # A2: 2x2
    ax = axs[0, 1]
    protos = ["Baseline", "PKN-std", "MPHF-max", "PKN-MPHF"]
    ds = sorted({r["delta"] for r in rows if r["exp"] == "A2"})
    groups = [{p: r for p in protos for r in pick(rows, "A2", p) if r["delta"] == d} for d in ds]
    x = bars(ax, groups, protos, "ot_instances_mean", log=True)
    ax.set_xticks(x, [f"δ={d:g}" for d in ds])
    ax.set_ylabel("OT instances (log)")
    ax.set_title("A2. Binning × receiver choice", fontsize=9, loc="left")

    # A3: party count
    ax = axs[1, 0]
    for p in ("Baseline", "PKN-std", "PKN-MPHF"):
        rs = pick(rows, "A3", p)
        st = STYLE[p]
        ax.plot([r["n"] for r in rs], [r["sym_ops_mean"] for r in rs],
                marker=st["marker"], ls=st["ls"], color=st["color"], lw=1.6, ms=5)
    ax.set_xlabel("Number of parties n (δ = 5)")
    ax.set_ylabel("Symmetric-key operations")
    ax.set_title("A3. Scaling with the number of parties", fontsize=9, loc="left")

    # E1: runtime split, WAN model
    ax = axs[1, 1]
    protos = ["Baseline", "PKN-std", "PKN-MPHF"]
    e1 = {p: pick(rows, "E1", p)[0] for p in protos if pick(rows, "E1", p)}
    comp = [e1[p]["sec_mean"] for p in protos]
    net = [e1[p]["t_total_WAN_mean"] - e1[p]["sec_mean"] for p in protos]
    ci = [e1[p]["t_total_WAN_ci95"] for p in protos]
    cols = [STYLE[p]["color"] for p in protos]
    ax.bar(range(3), comp, 0.6, color=cols, edgecolor="white", linewidth=0.8)
    ax.bar(range(3), net, 0.6, bottom=comp, color=cols, alpha=0.45,
           hatch="///", edgecolor="white", linewidth=0.8, yerr=ci, capsize=3)
    ax.set_xticks(range(3), ["Baseline", "PKN std.", "PKN MPHF"])
    ax.set_ylabel("Time (s)")
    ax.set_title("E1. Compute (solid) + WAN (hatched)", fontsize=9, loc="left")

    h, l = axs[0, 1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(R / f"fig2_ablation.{ext}", dpi=300)


def fmt(v, d=0):
    return f"{v:,.{d}f}"


def tables(rows):
    out = []

    def t(title, header, body):
        out.append(f"\n### {title}\n")
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "---|" * len(header))
        out.extend("| " + " | ".join(r) + " |" for r in body)

    # Table 3
    e1 = {p: pick(rows, "E1", p)[0] for p in ("Baseline", "PKN-std", "PKN-MPHF") if pick(rows, "E1", p)}
    if e1:
        b = e1["Baseline"]
        body = []
        for p, r in e1.items():
            body.append([p, fmt(r["ot_instances_mean"]), fmt(r["sym_ops_mean"]),
                         fmt(r["bytes_total_mean"]),
                         f"{fmt(r['peak_KiB_mean'])} ± {fmt(r['peak_KiB_ci95'])}",
                         f"{r['sec_mean']:.2f} ± {r['sec_ci95']:.2f}",
                         f"{r['t_total_LAN_mean']:.2f}", f"{r['t_total_WAN_mean']:.2f}"])
        p = e1["PKN-MPHF"]
        body.append(["Reduction (MPHF vs Baseline)",
                     f"{red(p['ot_instances_mean'], b['ot_instances_mean']):.1f}%",
                     f"{red(p['sym_ops_mean'], b['sym_ops_mean']):.1f}%",
                     f"{red(p['bytes_total_mean'], b['bytes_total_mean']):.1f}%",
                     f"{red(p['peak_KiB_mean'], b['peak_KiB_mean']):.1f}%",
                     f"{red(p['sec_mean'], b['sec_mean']):.1f}%",
                     f"{red(p['t_total_LAN_mean'], b['t_total_LAN_mean']):.1f}%",
                     f"{red(p['t_total_WAN_mean'], b['t_total_WAN_mean']):.1f}%"])
        t(f"Table 3. E1 (n = 5, δ = 10, reps = {int(b['reps'])}; mean ± 95% CI)",
          ["Protocol", "OT inst.", "Sym. ops", "Bytes", "Peak KiB", "Compute s",
           "Total LAN s", "Total WAN s"], body)
        out.append(f"\nMPHF build {p['t_mphf_mean']:.3f} s, base OTs {p['t_baseot_mean']:.3f} s, "
                   f"OKVS encode {p['t_okvs_mean']:.3f} s (PKN-MPHF, mean).")

    # Table 4
    base, pkn = pick(rows, "F1", "Baseline"), pick(rows, "F1", "PKN-MPHF")
    if base:
        body = [[f"{p['delta']:g}",
                 f"{red(p['bytes_total_mean'], q['bytes_total_mean']):.1f}",
                 f"{red(p['peak_KiB_mean'], q['peak_KiB_mean']):.1f}",
                 f"{red(p['sym_ops_mean'], q['sym_ops_mean']):.1f}",
                 f"{red(p['t_total_WAN_mean'], q['t_total_WAN_mean']):.1f}",
                 f"{q['ot_instances_mean'] / p['ot_instances_mean']:.2f}"]
                for p, q in zip(pkn, base)]
        t("Table 4. Reduction of PKN-PSI (MPHF) over the baseline vs δ (n = 3, m_min = 2000)",
          ["δ", "Bytes (%)", "Peak mem. (%)", "Sym. ops (%)", "WAN time (%)", "OT ratio"], body)

    # Table 5
    a1s, a1m = pick(rows, "A1", "PKN-std"), pick(rows, "A1", "PKN-MPHF")
    if a1s:
        body = [[f"n = {int(s['n'])}, m_max = {int(s['m_max'])}", fmt(s["ot_instances_mean"]),
                 fmt(m["ot_instances_mean"]), f"{red(m['ot_instances_mean'], s['ot_instances_mean']):.1f}",
                 fmt(s["bytes_total_mean"]), fmt(m["bytes_total_mean"]),
                 f"{red(m['bytes_total_mean'], s['bytes_total_mean']):.1f}"]
                for s, m in zip(a1s, a1m)]
        t("Table 5. A1: binning isolated (receiver = smallest set for both)",
          ["Setting", "OT cuckoo", "OT MPHF", "OT red. (%)", "Bytes cuckoo", "Bytes MPHF", "Bytes red. (%)"], body)

    # Table 6
    protos = ["Baseline", "PKN-std", "MPHF-max", "PKN-MPHF"]
    ds = sorted({r["delta"] for r in rows if r["exp"] == "A2"})
    if ds:
        body = []
        for d in ds:
            g = {p: r for p in protos for r in pick(rows, "A2", p) if r["delta"] == d}
            body.append([f"{d:g}"] + [fmt(g[p]["ot_instances_mean"]) for p in protos]
                        + [f"{g['Baseline']['ot_instances_mean'] / g['PKN-std']['ot_instances_mean']:.2f}",
                           f"{g['MPHF-max']['ot_instances_mean'] / g['PKN-MPHF']['ot_instances_mean']:.2f}",
                           f"{g['PKN-std']['ot_instances_mean'] / g['PKN-MPHF']['ot_instances_mean']:.2f}"])
        t("Table 6. A2: 2×2 factorial, OT instances (n = 3, m_min = 2000)",
          ["δ", "Cuckoo/largest", "Cuckoo/smallest", "MPHF/largest", "MPHF/smallest",
           "Receiver effect (cuckoo)", "Receiver effect (MPHF)", "Binning effect"], body)

    # Table 7
    b3, s3, m3 = pick(rows, "A3", "Baseline"), pick(rows, "A3", "PKN-std"), pick(rows, "A3", "PKN-MPHF")
    if b3:
        body = [[f"{int(b['n'])}", fmt(b["sym_ops_mean"]), fmt(s["sym_ops_mean"]), fmt(m["sym_ops_mean"]),
                 f"{red(m['sym_ops_mean'], b['sym_ops_mean']):.1f}",
                 f"{red(m['bytes_total_mean'], b['bytes_total_mean']):.1f}"]
                for b, s, m in zip(b3, s3, m3)]
        t("Table 7. A3: number of parties (δ = 5, m_min = 2000)",
          ["n", "Sym. ops baseline", "Sym. ops std.", "Sym. ops MPHF", "Sym. red. (%)", "Bytes red. (%)"], body)

    # E2
    b2, m2 = pick(rows, "E2", "Baseline"), pick(rows, "E2", "PKN-MPHF")
    if b2:
        body = [[fmt(m["m_min"]), fmt(m["m_max"]), f"{b['sec_mean']:.2f}", f"{m['sec_mean']:.2f}",
                 f"{m['t_mphf_mean']:.2f}", fmt(b["bytes_total_mean"]), fmt(m["bytes_total_mean"]),
                 f"{red(m['bytes_total_mean'], b['bytes_total_mean']):.1f}"]
                for b, m in zip(b2, m2)]
        t("Table E2. Set-size scalability (n = 3, δ = 4)",
          ["m_min", "m_max", "Baseline s", "PKN s", "of which MPHF build s", "Bytes baseline", "Bytes PKN", "Bytes red. (%)"], body)

    # leakage
    lk = R / "leakage.csv"
    if lk.exists():
        lr = list(csv.DictReader(open(lk)))
        body = [[r["delta"], r["overlap"], f"{float(r['auc_mean']):.3f} ± {float(r['auc_sd']):.3f}"] for r in lr]
        t("Leakage. Membership-inference AUC of a simple attack on the public h* (0.5 = no advantage)",
          ["δ", "Overlap (D_j ∩ D*) / m*", "AUC (mean ± sd)"], body)

    (R / "tables.md").write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    rows = load()
    fig1(rows)
    fig2(rows)
    tables(rows)
    print("wrote", ", ".join(sorted(p.name for p in R.iterdir() if p.suffix in (".png", ".pdf", ".md"))))
