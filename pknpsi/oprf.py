"""Batched oblivious PRF over OT extension (KKRT16 / IKNP03), executed for real.

Roles. The PSI receiver P* is the OPRF receiver; it plays the *sender* of the
w = 448 base OTs, with seed pairs (k_i^0, k_i^1). The PSI sender P_j plays the
base-OT receiver with choice bits s in {0,1}^w and learns k_i^{s_i} only.

Extension (column-wise, IKNP). For a receiver table of r rows with items R_b,
let C be the r x w matrix whose row b is the codeword C(R_b), and c^i its i-th
column. P* computes t^i = G(k_i^0) and sends u^i = t^i xor G(k_i^1) xor c^i.
P_j computes q^i = G(k_i^{s_i}) xor s_i * u^i = t^i xor s_i * c^i, so that row-wise
Q_b = T_b xor (C(R_b) AND s).  P_j never sees R_b.

Outputs. Receiver: M[b] = H(b, T_b). Sender, for candidate y at row b:
Eval(y, b) = H(b, Q_b xor (C(y) AND s)), which equals M[b] iff C(y) = C(R_b).

Both protocols call this single engine; the cryptographic layer is identical.
"""

import secrets

import numpy as np

from .crypto import (U64, W_BITS, W_BYTES, base_ots, codeword, corr_hash, prg,
                     prf_u64)


def _columns(mat_bytes: np.ndarray) -> np.ndarray:
    """(r, W_BYTES) row-major bit matrix -> (W_BITS, ceil(r/8)) packed columns."""
    bits = np.unpackbits(mat_bytes, axis=1)            # (r, W_BITS)
    return np.packbits(bits.T, axis=1)                 # (W_BITS, ceil(r/8))


def _rows(col_bytes: np.ndarray, r: int) -> np.ndarray:
    """Inverse of _columns."""
    bits = np.unpackbits(col_bytes, axis=1)[:, :r]     # (W_BITS, r)
    return np.packbits(bits.T, axis=1)                 # (r, W_BYTES)


class OPRFEngine:
    """One OPRF instance between the receiver P* and one sender P_j."""

    def __init__(self, rows: int, out_bits: int, counters, rng=None):
        self.rows = int(rows)
        self.out_bits = int(out_bits)
        self.c = counters
        # --- base OTs: P* sends seed pairs, P_j chooses with s ---------------
        self._k0 = [secrets.token_bytes(16) for _ in range(W_BITS)]   # P* only
        self._k1 = [secrets.token_bytes(16) for _ in range(W_BITS)]   # P* only
        s_bits = [secrets.randbits(1) for _ in range(W_BITS)]          # P_j only
        self._ks = base_ots(list(zip(self._k0, self._k1)), s_bits, counters)
        self._s_bits = np.array(s_bits, dtype=np.uint8)
        self.s = np.packbits(self._s_bits)                             # P_j only
        self.c.add(ot_instances=self.rows)

    # ------------------------------------------------------------ P* side
    def receiver_encode(self, items) -> np.ndarray:
        """P* commits its per-row items; returns M[b] = H(b, T_b) and sends U."""
        items = np.asarray(items, dtype=U64)
        assert items.size == self.rows
        r, nb = self.rows, (self.rows + 7) // 8
        Ccol = _columns(codeword(items))
        T = np.stack([prg(k, nb) for k in self._k0])
        U = T ^ np.stack([prg(k, nb) for k in self._k1]) ^ Ccol
        self.c.add(sym_ops=2 * r, bytes_r2s=U.nbytes)
        self._U = U                                   # the transmitted message
        return corr_hash(np.arange(r), _rows(T, r), self.out_bits)

    # ------------------------------------------------------------ P_j side
    def _sender_Q(self) -> np.ndarray:
        if not hasattr(self, "_Q"):
            nb = (self.rows + 7) // 8
            G = np.stack([prg(k, nb) for k in self._ks])
            mask = self._s_bits[:, None] * np.uint8(0xFF)
            self._Q = _rows(G ^ (self._U & mask), self.rows)
        return self._Q

    def sender_eval(self, items, bins) -> np.ndarray:
        """P_j evaluates the OPRF on its own items at the given rows."""
        items = np.asarray(items, dtype=U64)
        bins = np.asarray(bins, dtype=np.int64)
        Q = self._sender_Q()
        rows = Q[bins] ^ (codeword(items) & self.s)
        self.c.add(sym_ops=2 * items.size, placements=items.size)
        return corr_hash(bins, rows, self.out_bits)


def zero_shares(items, party: int, n_parties: int, pair_keys, out_bits: int,
                counters, sid: int = 0) -> np.ndarray:
    """s_j(x) = XOR_{i != j} F(k_{ji}, x || sid), so that XOR over all j is 0."""
    items = np.asarray(items, dtype=U64)
    acc = np.zeros(items.size, dtype=U64)
    for other in range(n_parties):
        if other == party:
            continue
        key = pair_keys[(min(party, other), max(party, other))]
        acc ^= prf_u64(key, items, tweak=sid)
        counters.add(sym_ops=items.size)
    if out_bits < 64:
        acc &= U64((1 << out_bits) - 1)
    return acc
