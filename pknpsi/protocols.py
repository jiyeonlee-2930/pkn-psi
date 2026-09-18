"""PKN-PSI and the cuckoo baseline.

Both are multi-party, both use zero-sharing so that no partial intersection
is revealed, and both call the same OPRF engine. They differ in exactly two
places, which are the two claims of the paper:

  binning structure   MPHF (m bins, 1 placement) vs cuckoo (1.27m + stash
                      bins, 3 placements)                      -> A1
  receiver choice     smallest set vs largest set              -> A2

Payload indexing. Every party publishes an MPHF over its own set. The
receiver's MPHF indexes the OPRF rows, so a sender can compute the row of
any element it holds. Each sender's own MPHF indexes its payload array, so
the receiver reads exactly one slot per sender per element and no candidate
matching is required. The baseline is given this same payload indexing even
though published cuckoo constructions do not have it, which makes every
measured gain reported here a lower bound.
"""

import time
import tracemalloc

import numpy as np

from .cuckoo import CuckooTable, candidate_bins, EPS, K_HASH
from .metrics import Counters, Result
from .mphf import MPHF, U64
from .oprf import OPRFEngine, zero_shares

SIGMA = 40
ALPHA0 = 0.94


def out_bits(n_parties: int, m_max: int) -> int:
    return min(64, SIGMA + int(np.ceil(np.log2(max(2, n_parties * m_max)))))


def _pair_keys(n, rng):
    return {(i, j): int(rng.integers(1, 2 ** 62))
            for i in range(n) for j in range(i + 1, n)}


def _run(fn, *args, **kw):
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    sec = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, peak / 1024.0, sec


# ---------------------------------------------------------------- PKN-PSI

def _pkn_core(datasets, recv, c, rng):
    n = len(datasets)
    sizes = [d.size for d in datasets]
    m_max = max(sizes)
    ell = out_bits(n, m_max)
    keys = _pair_keys(n, rng)

    # Phase 2: every party builds and publishes an MPHF
    hs = []
    for j, d in enumerate(datasets):
        h = MPHF(d, alpha=ALPHA0, seed=j)
        hs.append(h)
        c.add(build_retries=h.retries, index_bits=h.index_bits,
              bytes_index=h.index_bits // 8)
    h_star = hs[recv]
    D_star = datasets[recv]
    m_star = D_star.size
    c.add(bins=m_star)

    # receiver's row layout: row b holds the element with h_*(x) = b
    rows = np.empty(m_star, dtype=U64)
    rows[h_star(D_star)] = D_star

    s_star = zero_shares(D_star, recv, n, keys, ell, c)
    row_of_x = h_star(D_star)

    acc = s_star.copy()
    for j in range(n):
        if j == recv:
            continue
        # Phase 3: OPRF with m_min rows
        eng = OPRFEngine(m_star, ell, c, rng)
        M = eng.receiver_encode(rows)

        # Phase 4: sender places each element once, at row h_*(y)
        D_j = datasets[j]
        F = eng.sender_eval(D_j, h_star(D_j))
        A = np.zeros(D_j.size, dtype=U64)
        A[hs[j](D_j)] = F ^ zero_shares(D_j, j, n, keys, ell, c)
        c.add(bytes_s2r=D_j.size * ell // 8)

        # Phase 5: receiver reads exactly one slot of A per element
        acc ^= A[hs[j](D_star)] ^ M[row_of_x]

    return D_star[acc == 0]


# --------------------------------------------------------------- baseline

def _cuckoo_core(datasets, recv, c, rng):
    n = len(datasets)
    sizes = [d.size for d in datasets]
    m_max = max(sizes)
    ell = out_bits(n, m_max)
    keys = _pair_keys(n, rng)

    hs = [MPHF(d, alpha=1.0, seed=j) for j, d in enumerate(datasets)]
    for h in hs:
        c.add(build_retries=h.retries)

    D_star = datasets[recv]
    m_star = D_star.size
    tbl = CuckooTable(D_star, rng=rng)
    n_rows = tbl.bins + tbl.stash_size
    c.add(bins=tbl.bins, stash=tbl.stash_size)

    rows = np.zeros(n_rows, dtype=U64)
    occ = tbl.table >= 0
    rows[np.flatnonzero(occ)] = D_star[tbl.table[occ]]
    if tbl.stash_size:
        rows[tbl.bins:] = D_star[tbl.stash]

    # which row holds each of the receiver's elements
    row_of = np.empty(m_star, dtype=np.int64)
    row_of[tbl.table[occ]] = np.flatnonzero(occ)
    if tbl.stash_size:
        row_of[tbl.stash] = np.arange(tbl.bins, n_rows)

    s_star = zero_shares(D_star, recv, n, keys, ell, c)
    acc = s_star.copy()

    for j in range(n):
        if j == recv:
            continue
        eng = OPRFEngine(n_rows, ell, c, rng)
        M = eng.receiver_encode(rows)

        D_j = datasets[j]
        cand = candidate_bins(D_j, tbl.bins)          # K rows per element
        slot = hs[j](D_j)
        A = np.zeros((D_j.size, K_HASH), dtype=U64)
        sj = zero_shares(D_j, j, n, keys, ell, c)
        for t in range(K_HASH):
            A[slot, t] = eng.sender_eval(D_j, cand[t]) ^ sj
        c.add(bytes_s2r=D_j.size * K_HASH * ell // 8)

        # receiver knows which of its candidate bins actually holds x
        my_slot = hs[j](D_star)
        my_cand = candidate_bins(D_star, tbl.bins)
        which = np.zeros(m_star, dtype=np.int64)
        for t in range(K_HASH):
            which = np.where(my_cand[t] == row_of, t, which)
        acc ^= A[my_slot, which] ^ M[row_of]

    return D_star[acc == 0]


# ------------------------------------------------------------------- API

def run(datasets, protocol="pkn", receiver="min", seed=0) -> Result:
    """Execute one protocol on a list of uint64 arrays."""
    datasets = [np.asarray(d, dtype=U64) for d in datasets]
    sizes = [int(d.size) for d in datasets]
    n = len(datasets)
    rng = np.random.default_rng(seed)

    if receiver == "min":
        recv = int(np.argmin(sizes))
    elif receiver == "max":
        recv = int(np.argmax(sizes))
    else:
        recv = int(receiver)

    c = Counters()
    core = _pkn_core if protocol == "pkn" else _cuckoo_core
    out, peak, sec = _run(core, datasets, recv, c, rng)

    truth = set(datasets[0].tolist())
    for d in datasets[1:]:
        truth &= set(d.tolist())

    return Result(
        protocol=protocol,
        n_parties=n,
        sizes=sizes,
        delta=max(sizes) / min(sizes),
        epsilon=sum(sizes) / (n * max(sizes)),
        receiver_index=recv,
        receiver_size=sizes[recv],
        intersection_size=len(truth),
        correct=(set(out.tolist()) == truth),
        counters=c,
        peak_kib=peak,
        seconds=sec,
    )
