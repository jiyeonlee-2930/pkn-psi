"""Headline comparison: PKN-PSI vs a cuckoo-hashing OT-extension baseline.

Reproduces Table R1 of the paper (n=5, delta=10):
    symmetric-key ops -44.2%,  peak memory -51.3%,  transmitted volume -84.9%

    python comparison.py
"""
from pknpsi.datagen import make_datasets, imbalanced_sizes
from pknpsi.protocols import run


def pct(new, old):
    return 0.0 if old == 0 else 100.0 * (old - new) / old


def main():
    configs = [
        ("n=3, delta=1  (balanced)", 3, 2000, 1),
        ("n=3, delta=10",            3, 2000, 10),
        ("n=5, delta=10  (headline)",5, 2000, 10),
    ]
    print(f"{'config':>26} {'sym base':>9} {'sym pkn':>8} {'sym%':>6} "
          f"{'mem base':>9} {'mem pkn':>8} {'mem%':>6} "
          f"{'byte base':>11} {'byte pkn':>10} {'byte%':>6}")
    for label, n, m_min, delta in configs:
        ds = make_datasets(imbalanced_sizes(n, m_min, delta), 0.3, seed=7)
        base = run(ds, "cuckoo", "max", 7)   # baseline: cuckoo + receiver = largest
        pkn = run(ds, "pkn", "min", 7)       # proposed: MPHF + receiver = smallest
        assert base.correct and pkn.correct
        bs, ps = base.counters.sym_ops, pkn.counters.sym_ops
        bm, pm = base.peak_kib, pkn.peak_kib
        bb, pb = base.counters.bytes_total, pkn.counters.bytes_total
        print(f"{label:>26} {bs:>9} {ps:>8} {pct(ps,bs):>6.1f} "
              f"{bm:>9.0f} {pm:>8.0f} {pct(pm,bm):>6.1f} "
              f"{bb:>11} {pb:>10} {pct(pb,bb):>6.1f}")


if __name__ == "__main__":
    main()
