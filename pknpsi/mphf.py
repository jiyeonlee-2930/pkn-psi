"""PTHash-style minimal perfect hash function.

Keys are 64-bit unsigned integers (pseudonymized identifiers truncated to
64 bits). Construction follows PTHash: keys are distributed into partitions,
partitions are processed largest-first, and a pilot value is searched per
partition so that its keys occupy distinct free slots. Slots at or above m
are remapped onto the free slots below m, which makes the function minimal
(output domain exactly [0, m)).

The decisive property for PKN-PSI: h is a bijection D -> [0, m), so the bin
count equals the element count with no spare bins and no stash, and h(y) is
defined for arbitrary y outside D.
"""

import numpy as np

U64 = np.uint64

_C1 = U64(0x9E3779B97F4A7C15)
_C2 = U64(0xBF58476D1CE4E5B9)
_C3 = U64(0x94D049BB133111EB)


_MASK = (1 << 64) - 1


def _scalar_mix(x: int, seed: int) -> int:
    """Scalar twin of splitmix, used inside the pilot search."""
    z = (x + seed + 0x9E3779B97F4A7C15) & _MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    return z ^ (z >> 31)


def splitmix(x, seed):
    """Vectorized 64-bit mixing function."""
    with np.errstate(over="ignore"):
        z = (np.asarray(x, dtype=U64) + U64(seed) + _C1).astype(U64)
        z = ((z ^ (z >> U64(30))) * _C2).astype(U64)
        z = ((z ^ (z >> U64(27))) * _C3).astype(U64)
        return (z ^ (z >> U64(31))).astype(U64)


class MPHF:
    """Minimal perfect hash function over a set of uint64 keys."""

    SEED_PART = 0x1111
    SEED_POS = 0x2222
    SEED_PILOT = 0x3333

    def __init__(self, keys, alpha=1.0, lam=4.0, max_pilot=2 ** 16, seed=0):
        keys = np.asarray(keys, dtype=U64)
        self.m = int(keys.size)
        self.alpha = float(alpha)
        self.seed = int(seed)
        self.N = max(self.m, int(np.ceil(self.m / self.alpha)))
        self.n_parts = max(1, int(np.ceil(self.m / lam)))
        self.retries = 0
        # A given position seed can be unsatisfiable on degenerate key sets
        # (for example two keys agreeing on the low bits when N is small).
        # PTHash handles this by reseeding and rebuilding.
        for attempt in range(64):
            try:
                self._build(keys, max_pilot)
                self.attempt = attempt
                return
            except RuntimeError:
                self.seed = (self.seed + 0x9E37) & 0xFFFFFFFF
        raise RuntimeError("MPHF construction failed after 64 reseeds")

    # -- construction -------------------------------------------------
    def _build(self, keys, max_pilot):
        parts = (splitmix(keys, self.SEED_PART + self.seed)
                 % U64(self.n_parts)).astype(np.int64)
        hk = splitmix(keys, self.SEED_POS + self.seed)

        order = np.argsort(parts, kind="stable")
        parts_sorted = parts[order]
        hk_sorted = hk[order].tolist()          # scalar arithmetic is faster
        starts = np.searchsorted(parts_sorted, np.arange(self.n_parts), "left")
        ends = np.searchsorted(parts_sorted, np.arange(self.n_parts), "right")
        sizes = (ends - starts)

        # largest partitions first: they are the hardest to place
        part_order = np.argsort(-sizes, kind="stable").tolist()
        starts, ends = starts.tolist(), ends.tolist()

        pilots = np.zeros(self.n_parts, dtype=np.uint32)
        occupied = bytearray(self.N)
        N, seed, retries = self.N, self.SEED_PILOT + self.seed, 0
        pilot_hash = []

        for p in part_order:
            s, e = starts[p], ends[p]
            if s == e:
                continue
            block = hk_sorted[s:e]
            for pilot in range(max_pilot):
                if pilot >= len(pilot_hash):
                    pilot_hash.append(_scalar_mix(pilot, seed))
                ph = pilot_hash[pilot]
                pos = [(k ^ ph) % N for k in block]
                if any(occupied[q] for q in pos) or len(set(pos)) != len(pos):
                    retries += 1
                    continue
                for q in pos:
                    occupied[q] = 1
                pilots[p] = pilot
                break
            else:
                raise RuntimeError("pilot search exhausted; lower alpha")

        self.retries = retries
        occupied = np.frombuffer(bytes(occupied), dtype=np.uint8).astype(bool)

        self.pilots = pilots
        self.max_pilot = int(pilots.max()) if self.n_parts else 0

        # free-slot remapping makes the function minimal
        if self.N > self.m:
            free = np.flatnonzero(~occupied[:self.m])
            high = np.flatnonzero(occupied[self.m:])
            remap = np.zeros(self.N - self.m, dtype=np.int64)
            remap[high] = free[:high.size]
            self.remap = remap
        else:
            self.remap = None

    # -- evaluation ---------------------------------------------------
    def __call__(self, keys):
        """Evaluate h on an array of keys. Defined for keys outside D."""
        keys = np.asarray(keys, dtype=U64)
        parts = (splitmix(keys, self.SEED_PART + self.seed)
                 % U64(self.n_parts)).astype(np.int64)
        hk = splitmix(keys, self.SEED_POS + self.seed)
        ph = splitmix(self.pilots[parts].astype(U64),
                      self.SEED_PILOT + self.seed)
        pos = ((hk ^ ph) % U64(self.N)).astype(np.int64)
        if self.remap is not None:
            high = pos >= self.m
            if high.any():
                pos[high] = self.remap[pos[high] - self.m]
        return pos

    # -- accounting ---------------------------------------------------
    @property
    def index_bits(self) -> int:
        """Serialized size of the published structure.

        Pilots are stored at the width actually required. A production
        PTHash applies compressed encodings and reaches 2.1-3.2 bits/key;
        this prototype is uncompressed, so the figure is an upper bound.
        """
        w = max(1, int(np.ceil(np.log2(self.max_pilot + 1))))
        bits = self.n_parts * w
        if self.remap is not None:
            bits += self.remap.size * int(np.ceil(np.log2(max(2, self.m))))
        return bits

    @property
    def bits_per_key(self) -> float:
        return self.index_bits / max(1, self.m)

    def verify(self, keys) -> bool:
        pos = self(keys)
        return pos.size == np.unique(pos).size and pos.min() >= 0 and pos.max() < self.m
