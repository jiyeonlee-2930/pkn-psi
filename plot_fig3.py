"""Figure 3 (ablation summary) redrawn from the revision results (2026-10-20).

Only change from the submitted figure: numbers with five or more digits
use a thousands comma (10000 -> 10,000).
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

GRAY, ORANGE, GREEN, BLUE = "#6b6b6b", "#d55e00", "#009e73", "#0072b2"


def comma5(x, pos):
    return f"{x:,.0f}" if abs(x) >= 10000 else f"{x:.0f}"


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False})
fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(10.6, 8.2))

# A1: binning only, receiver = smallest (Table 6)
labels = ["n=3\nm_max=2k", "n=3\nm_max=8k", "n=5\nm_max=2k", "n=5\nm_max=8k"]
cuckoo = [5080, 5080, 10160, 10160]
mphf = [4000, 4000, 8000, 8000]
x = range(4)
w = 0.38
a1.bar([i - w / 2 for i in x], cuckoo, w, color=ORANGE, zorder=3)
a1.bar([i + w / 2 for i in x], mphf, w, color=BLUE, zorder=3)
a1.set_xticks(list(x))
a1.set_xticklabels(labels)
a1.set_ylabel("OT instances")
a1.set_title("A1. Binning only (receiver = smallest)", loc="left", fontsize=11)
a1.yaxis.set_major_formatter(FuncFormatter(comma5))
a1.grid(True, color="#e0e0e0", zorder=0)

# A2: 2x2 factorial, log scale (Table 7)
deltas = ["δ=1", "δ=2", "δ=10", "δ=50"]
a2_data = [
    ([5080, 10160, 50800, 254000], GRAY),     # cuckoo / largest
    ([5080, 5080, 5080, 5080], ORANGE),        # cuckoo / smallest
    ([4000, 8000, 40000, 200000], GREEN),      # MPHF / largest
    ([4000, 4000, 4000, 4000], BLUE),          # MPHF / smallest
]
w2 = 0.2
for k, (vals, col) in enumerate(a2_data):
    a2.bar([i + (k - 1.5) * w2 for i in range(4)], vals, w2 * 0.9,
           color=col, zorder=3)
a2.set_yscale("log")
a2.set_ylim(3500, 3.2e5)
a2.set_xticks(range(4))
a2.set_xticklabels(deltas)
a2.set_ylabel("OT instances (log)")
a2.set_title("A2. Binning × receiver choice", loc="left", fontsize=11)
a2.grid(True, color="#e0e0e0", zorder=0)

# A3: number of parties, delta = 5 (Table 8)
ns = [3, 5, 10, 20]
a3.plot(ns, [166800, 461600, 1548600, 5222600], "--s", color=GRAY, lw=2, ms=7)
a3.plot(ns, [174160, 428320, 1413720, 4884520], "-.^", color=ORANGE, lw=2, ms=7)
a3.plot(ns, [92000, 264000, 1044000, 4104000], "-o", color=BLUE, lw=2, ms=7)
a3.set_xticks([5, 10, 15, 20])
a3.set_xlabel("Number of parties n (δ = 5)")
a3.set_ylabel("Symmetric-key operations")
a3.set_title("A3. Scaling with the number of parties", loc="left", fontsize=11)
a3.grid(True, color="#e0e0e0", zorder=0)

# E1: compute (solid) + modeled WAN transfer (hatched), Table 4
names = ["Baseline", "PKN std.", "PKN MPHF"]
comp = [1.37, 0.80, 0.46]
ci = [0.02, 0.01, 0.00]
wan_total = [2.45, 1.51, 1.05]
cols = [GRAY, ORANGE, BLUE]
for i in range(3):
    a4.bar(i, comp[i], 0.55, color=cols[i], zorder=3)
    a4.bar(i, wan_total[i] - comp[i], 0.55, bottom=comp[i], color=cols[i],
           alpha=0.45, hatch="//", edgecolor="white", zorder=3)
    a4.errorbar(i, wan_total[i], yerr=max(ci[i], 0.005), color="black",
                capsize=5, lw=1.5, zorder=4)
a4.set_xticks(range(3))
a4.set_xticklabels(names)
a4.set_ylabel("Time (s)")
a4.set_title("E1. Compute (solid) + WAN (hatched)", loc="left", fontsize=11)
a4.grid(True, color="#e0e0e0", zorder=0)

handles = [Patch(color=GRAY, label="Baseline (cuckoo, largest receiver)"),
           Patch(color=ORANGE, label="PKN-PSI standard mode (cuckoo)"),
           Patch(color=GREEN, label="MPHF, largest receiver"),
           Patch(color=BLUE, label="PKN-PSI (MPHF)")]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
           fontsize=10)
fig.tight_layout(rect=(0, 0.07, 1, 1), h_pad=2.5)

for ext in ("png", "pdf"):
    fig.savefig(f"fig3_ablation_summary.{ext}", dpi=600)
print("saved")
