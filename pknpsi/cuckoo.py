"""Cuckoo hashing with a stash: the binning structure of KKRT16 and
Pinkas et al. 2018.

The receiver allocates b = ceil(EPS * m) bins with K hash functions and
evicts on collision. Elements that exceed the eviction bound land in the
stash. Two consequences matter for PSI and are the target of ablation A1:

  1. the bin count exceeds the element count by the factor EPS (1.27);
  2. the sender does not know which of the K candidate bins holds an
     element, so it must place every element into all K of them.
"""

import numpy as np

from .mphf import splitmix, U64

EPS = 1.27          # bins per element, as used by Pinkas et al. 2018
K_HASH = 3          # number of cuckoo hash functions
MAX_EVICT = 500

_SEEDS = (0xA1, 0xB2, 0xC3, 0xD4, 0xE5)


def candidate_bins(keys, b, k=K_HASH):
    """The k candidate bin indices of each key. Defined outside D."""
    keys = np.asarray(keys, dtype=U64)
    return np.stack(
        [(splitmix(keys, _SEEDS[i]) % U64(b)).astype(np.int64) for i in range(k)]
    )


class CuckooTable:
    def __init__(self, keys, eps=EPS, k=K_HASH, rng=None):
        keys = np.asarray(keys, dtype=U64)
        self.m = int(keys.size)
        self.k = k
        self.b = max(k, int(np.ceil(eps * self.m)))
        self.rng = rng or np.random.default_rng(0)
        self._insert(keys)

    def _insert(self, keys):
        cand = candidate_bins(keys, self.b, self.k)
        table = np.full(self.b, -1, dtype=np.int64)   # bin -> key position
        stash = []
        for idx in range(self.m):
            cur, which = idx, 0
            for _ in range(MAX_EVICT):
                slot = int(cand[which, cur])
                prev = table[slot]
                table[slot] = cur
                if prev == -1:
                    cur = -1
                    break
                cur = int(prev)
                # evicted element must move to a different hash function
                opts = [j for j in range(self.k) if int(cand[j, cur]) != slot]
                which = int(self.rng.choice(opts)) if opts else (which + 1) % self.k
            if cur != -1:
                stash.append(cur)
        self.table = table
        self.stash = np.array(stash, dtype=np.int64)
        self.keys = keys

    @property
    def bins(self) -> int:
        return self.b

    @property
    def stash_size(self) -> int:
        return int(self.stash.size)

    def bin_of(self, position: int) -> int:
        """Bin holding the key at the given position, or -1 if stashed."""
        hit = np.flatnonzero(self.table == position)
        return int(hit[0]) if hit.size else -1
