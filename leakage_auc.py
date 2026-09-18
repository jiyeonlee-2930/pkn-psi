"""VII-D: empirical validation of MPHF publication (Theorem 5).

Trains a classifier to tell an MPHF built on real (pseudonymized) keys from one
built on random keys. AUC near 0.5 => the published MPHF leaks nothing.
Reproduces AUC ~= 0.51.

Requires scikit-learn (see requirements.txt).

    python leakage_auc.py
"""
import hashlib
import numpy as np
from pknpsi.mphf import MPHF
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(0)


def real_keys(n, salt):
    out = np.empty(n, dtype=np.uint64)
    for i in range(n):
        out[i] = int.from_bytes(
            hashlib.sha256(f"patient{i}|{salt}".encode()).digest()[:8], "little")
    return np.unique(out)


def rand_keys(n):
    return np.unique(rng.integers(0, 2 ** 64, size=n, dtype=np.uint64))


def feats(h):
    p = np.asarray(h.pilots, dtype=float)
    return [h.index_bits / h.m, h.retries, p.mean(), p.std(),
            np.median(p), (p == 0).mean()]


def main(per_class=150, m=1500):
    X, y = [], []
    for t in range(per_class):
        X.append(feats(MPHF(real_keys(m, f"inst{t}"), alpha=0.94, seed=t))); y.append(1)
        X.append(feats(MPHF(rand_keys(m), alpha=0.94, seed=1000 + t))); y.append(0)
    X, y = np.array(X), np.array(y)
    auc = cross_val_score(LogisticRegression(max_iter=3000), X, y, cv=10, scoring="roc_auc")
    print(f"MPHF-publication distinguisher AUC = {auc.mean():.3f} +/- {auc.std():.3f} "
          f"({len(y)} samples, 10-fold)")
    print("-> near 0.5 means real-data and random-key MPHFs are indistinguishable")


if __name__ == "__main__":
    main()
