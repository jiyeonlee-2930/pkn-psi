"""Oblivious key-value store (3-hash XOR table solved by peeling).

Encode({k_i -> v_i}) returns a table P of L = ceil(EXPANSION * m) ell-bit cells
such that Decode(P, k_i) = P[h1(k_i)] xor P[h2(k_i)] xor P[h3(k_i)] = v_i.
The three positions come from a public, data-independent hash into disjoint
thirds of the table. Cells left free by the peeling are filled with uniform
randomness, so when the values are uniform the table is uniform and independent
of the keys (obliviousness, cf. Garimella et al., CRYPTO 2021).

This replaces the per-sender MPHF h_j of the earlier design: a published MPHF
is injective on D_j and therefore reveals membership information, whereas an
OKVS with pseudorandom values reveals nothing about its keys.
"""

import secrets

import numpy as np

from .crypto import U64, prf_u64

EXPANSION = 1.3          # 3-hypergraph peeling threshold is 1/0.818 = 1.222
_POS_KEY = bytes.fromhex("6f6b76732d706f736974696f6e6b6579")


class OKVSFailure(RuntimeError):
    pass


def positions(keys, L: int, seed: int) -> np.ndarray:
    """(m, 3) cell indices, one per third of the table."""
    third = L // 3
    h = np.stack([prf_u64(_POS_KEY, keys, tweak=(seed << 2) | t)
                  for t in range(3)], axis=1)
    return (h % U64(third)).astype(np.int64) + np.arange(3) * third


def _peel(pos: np.ndarray, L: int):
    """Vectorized peeling. Returns a list of rounds (key idx, pivot cell)."""
    m = pos.shape[0]
    alive = np.ones(m, dtype=bool)
    count = np.bincount(pos.ravel(), minlength=L)
    rounds = []
    while alive.any():
        deg1 = count == 1
        hit = deg1[pos] & alive[:, None]                  # (m, 3)
        keys = np.flatnonzero(hit.any(axis=1))
        if keys.size == 0:
            raise OKVSFailure("2-core is non-empty")
        piv = pos[keys, hit[keys].argmax(axis=1)]
        rounds.append((keys, piv))
        alive[keys] = False
        np.subtract.at(count, pos[keys].ravel(), 1)
    return rounds


def encode(keys, values, out_bits: int, counters=None, max_tries: int = 32):
    """Returns (table, seed)."""
    keys = np.asarray(keys, dtype=U64)
    values = np.asarray(values, dtype=U64)
    if np.unique(keys).size != keys.size:
        raise ValueError("OKVS keys must be distinct")
    m = keys.size
    L = 3 * max(1, int(np.ceil(EXPANSION * m / 3)))
    mask = U64((1 << out_bits) - 1) if out_bits < 64 else U64(0xFFFFFFFFFFFFFFFF)
    for seed in range(max_tries):
        pos = positions(keys, L, seed)
        try:
            rounds = _peel(pos, L)
        except OKVSFailure:
            if counters is not None:
                counters.add(okvs_retries=1)
            continue
        P = np.frombuffer(secrets.token_bytes(8 * L), dtype=U64).copy() & mask
        for idx, piv in reversed(rounds):
            p = pos[idx]
            other = P[p[:, 0]] ^ P[p[:, 1]] ^ P[p[:, 2]] ^ P[piv]
            P[piv] = values[idx] ^ other
        return P, seed
    raise OKVSFailure("encoding failed after retries")


def decode(table: np.ndarray, seed: int, keys) -> np.ndarray:
    pos = positions(keys, table.size, seed)
    return table[pos[:, 0]] ^ table[pos[:, 1]] ^ table[pos[:, 2]]
