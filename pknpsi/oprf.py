"""Batched oblivious PRF over OT extension (KKRT16 / KK13 style).

Both the proposed protocol and the cuckoo baseline are built on this single
engine, so any measured difference between them comes from the binning
structure and the receiver designation, never from the cryptographic layer.

Construction. The receiver holds a random matrix T of shape (rows, w) and
sends U = T XOR C, where row j of C is the codeword of the receiver's item
in bin j. The sender, holding a random w-bit string s obtained through base
OTs, recovers Q with row q_j = t_j XOR (c_j AND s). For a candidate y the
sender evaluates H(q_j XOR (C(y) AND s)), which equals the receiver's H(t_j)
exactly when y is the item in bin j.

Base OTs are accounted for but not executed: they number w in both
protocols and therefore cancel in every comparison reported here.
"""

import numpy as np

from .mphf import splitmix, U64

W_BITS = 448          # pseudo-random codeword width
W_BYTES = W_BITS // 8
W_WORDS = W_BYTES // 8
_CW_SEEDS = tuple(0x5000 + 17 * i for i in range(W_WORDS))
_OUT_SEED = 0x7777


def codeword(keys) -> np.ndarray:
    """Pseudo-random code C: {0,1}^64 -> {0,1}^W_BITS, vectorized."""
    keys = np.asarray(keys, dtype=U64)
    words = np.stack([splitmix(keys, s) for s in _CW_SEEDS], axis=1)
    return words.view(np.uint8).reshape(keys.size, W_BYTES)


def _fold(rows: np.ndarray, out_bits: int) -> np.ndarray:
    """Correlation-robust output function H, truncated to out_bits."""
    words = rows.reshape(rows.shape[0], W_WORDS, 8).view(U64).reshape(-1, W_WORDS)
    acc = np.zeros(rows.shape[0], dtype=U64)
    for i in range(W_WORDS):
        acc = splitmix(acc ^ words[:, i], _OUT_SEED + i)
    if out_bits >= 64:
        return acc
    return acc & U64((1 << out_bits) - 1)


class OPRFEngine:
    """One OPRF instance between a receiver and one sender."""

    def __init__(self, rows: int, out_bits: int, counters, rng):
        self.rows = int(rows)
        self.out_bits = int(out_bits)
        self.c = counters
        self.rng = rng
        self.s = rng.integers(0, 256, size=W_BYTES, dtype=np.uint8)
        self.c.add(base_ots=W_BITS, ot_instances=self.rows)

    def receiver_encode(self, items) -> np.ndarray:
        """Receiver commits its per-bin items and returns its OPRF values."""
        items = np.asarray(items, dtype=U64)
        assert items.size == self.rows
        C = codeword(items)
        T = self.rng.integers(0, 256, size=(self.rows, W_BYTES), dtype=np.uint8)
        self._Q = T ^ (C & self.s)              # what the sender reconstructs
        self.c.add(sym_ops=self.rows * 2,
                   bytes_r2s=self.rows * W_BYTES)   # U = T XOR C is transmitted
        return _fold(T, self.out_bits)

    def sender_eval(self, items, bins) -> np.ndarray:
        """Sender evaluates the OPRF on its own items at the given bins."""
        items = np.asarray(items, dtype=U64)
        bins = np.asarray(bins, dtype=np.int64)
        C = codeword(items)
        rows = self._Q[bins] ^ (C & self.s)
        self.c.add(sym_ops=items.size * 2, placements=items.size)
        return _fold(rows, self.out_bits)


def zero_shares(items, party: int, n_parties: int, pair_keys, out_bits: int,
                counters) -> np.ndarray:
    """s_j(x) = XOR over l != j of F(k_{jl}, x), so that XOR over all j is 0."""
    items = np.asarray(items, dtype=U64)
    acc = np.zeros(items.size, dtype=U64)
    for other in range(n_parties):
        if other == party:
            continue
        key = pair_keys[(min(party, other), max(party, other))]
        acc ^= splitmix(items, key)
        counters.add(sym_ops=items.size)
    if out_bits < 64:
        acc &= U64((1 << out_bits) - 1)
    return acc
