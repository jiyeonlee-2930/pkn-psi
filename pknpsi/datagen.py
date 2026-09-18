"""Dataset generation.

Set elements are 64-bit truncations of SHA-256(identifier || institution
salt), matching the pseudonymized identifiers used as matching keys in the
medical setting. Because OT-based PSI performance depends only on the size
profile (n, {m_i}, |I|) and not on element content, the size profile is the
quantity that must be realistic; the intersection ratio is a controlled
variable, as cross-institution patient overlap is not observable in
de-identified data.
"""

import hashlib

import numpy as np

from .mphf import U64


def _ids(tags, salt: bytes) -> np.ndarray:
    out = np.empty(len(tags), dtype=U64)
    for i, t in enumerate(tags):
        d = hashlib.sha256(str(t).encode() + b"|" + salt).digest()
        out[i] = int.from_bytes(d[:8], "big")
    return out


def make_datasets(sizes, overlap=0.3, seed=0, fast=True):
    """Build n sets with the given sizes and a shared intersection.

    overlap is the fraction of the smallest set that lies in every set.
    """
    rng = np.random.default_rng(seed)
    sizes = list(sizes)
    n_common = int(min(sizes) * overlap)

    if fast:
        # splitmix over disjoint integer ranges: same distribution, no SHA cost
        base = rng.integers(0, 2 ** 63, size=(sum(sizes) + n_common),
                            dtype=np.int64).astype(U64)
        common, rest, pos = base[:n_common], base[n_common:], 0
    else:
        common = _ids(range(n_common), b"common")
        rest = np.concatenate([_ids(range(s), f"inst{i}".encode())
                               for i, s in enumerate(sizes)])
        pos = 0

    out = []
    for i, s in enumerate(sizes):
        take = s - n_common
        private = rest[pos:pos + take]
        pos += take
        d = np.unique(np.concatenate([common, private]))
        while d.size < s:                      # repair rare collisions
            extra = rng.integers(0, 2 ** 63, size=s - d.size,
                                 dtype=np.int64).astype(U64)
            d = np.unique(np.concatenate([d, extra]))
        out.append(d[:s] if d.size > s else d)
    return out


def imbalanced_sizes(n, m_min, delta):
    """One smallest set, the rest scaled up to delta * m_min."""
    return [m_min] + [int(m_min * delta)] * (n - 1)
