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
from .crypto import prf_u64
from .metrics import Counters, Result
from .mphf import MPHF, U64
from . import okvs
from .oprf import OPRFEngine, zero_shares

SIGMA = 40
ALPHA0 = 0.94


def out_bits(n_parties: int, m_max: int) -> int:
    return min(64, SIGMA + int(np.ceil(np.log2(max(2, n_parties * m_max)))))


def _pair_keys(n, rng):
    """Pairwise 128-bit PRF keys for zero-sharing (one-time setup)."""
    import secrets
    return {(i, j): secrets.token_bytes(16)
            for i in range(n) for j in range(i + 1, n)}


# Public key for deriving OKVS keys from (element, candidate index).
_KD_KEY = bytes.fromhex("6b65792d646572697661746976653031")


def _okvs_key(items, idx):
    """Distinct 64-bit OKVS key for element y at candidate index idx."""
    return prf_u64(_KD_KEY, items, tweak=int(idx))


TRACE_MEMORY = True     # set per call through run(..., trace_memory=...)


def _run(fn, *args, **kw):
    """Run fn; time it, and trace peak memory only when TRACE_MEMORY is set.

    tracemalloc slows pure-Python loops (MPHF pilot search, cuckoo insertion)
    considerably, so timing runs should use trace_memory=False and memory
    should be measured in a separate run.
    """
    if TRACE_MEMORY:
        tracemalloc.start()
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    sec = time.perf_counter() - t0
    peak = 0
    if TRACE_MEMORY:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return out, peak / 1024.0, sec


# ---------------------------------------------------------------- PKN-PSI

def _pkn_core(datasets, recv, c, rng, payload="okvs"):
    """PKN-PSI: the receiver publishes an MPHF h_* that fixes the OPRF rows.

    payload = "okvs"  : each sender encodes {y -> OPRF(y) xor s_j(y)} as an OKVS
                        (revised design; senders publish nothing data-dependent)
    payload = "mphf"  : each sender publishes its own MPHF h_j and an array
                        (original design, kept to reproduce the leakage analysis)
    """
    n = len(datasets)
    sizes = [d.size for d in datasets]
    m_max = max(sizes)
    ell = out_bits(n, m_max)
    keys = _pair_keys(n, rng)

    # the receiver builds and publishes h_*
    D_star = datasets[recv]
    m_star = D_star.size
    _t = time.perf_counter()
    h_star = MPHF(D_star, alpha=ALPHA0, seed=recv)
    c.add(t_mphf=time.perf_counter() - _t)
    c.add(build_retries=h_star.retries, index_bits=h_star.index_bits,
          bytes_index=h_star.index_bits // 8, bins=m_star)
    hs = {}
    if payload == "mphf":
        for j, d in enumerate(datasets):
            if j != recv:
                hs[j] = MPHF(d, alpha=ALPHA0, seed=j)
                c.add(build_retries=hs[j].retries, index_bits=hs[j].index_bits,
                      bytes_index=hs[j].index_bits // 8)

    # receiver's row layout: row b holds the element with h_*(x) = b
    row_of_x = h_star(D_star)
    rows = np.empty(m_star, dtype=U64)
    rows[row_of_x] = D_star

    acc = zero_shares(D_star, recv, n, keys, ell, c)
    for j in range(n):
        if j == recv:
            continue
        # OPRF with exactly m_* rows
        eng = OPRFEngine(m_star, ell, c, rng)
        M = eng.receiver_encode(rows)

        # sender evaluates each element once, at row h_*(y)
        D_j = datasets[j]
        vals = eng.sender_eval(D_j, h_star(D_j)) ^ zero_shares(D_j, j, n, keys, ell, c)

        if payload == "okvs":
            _t = time.perf_counter()
            table, seed = okvs.encode(D_j, vals, ell, c)
            c.add(t_okvs=time.perf_counter() - _t)
            c.add(bytes_s2r=table.size * ell // 8)
            got = okvs.decode(table, seed, D_star)
        else:
            A = np.zeros(D_j.size, dtype=U64)
            A[hs[j](D_j)] = vals
            c.add(bytes_s2r=D_j.size * ell // 8)
            got = A[hs[j](D_star)]

        acc ^= got ^ M[row_of_x]

    return D_star[acc == 0]


# --------------------------------------------------------------- baseline

def _cuckoo_core(datasets, recv, c, rng):
    """Cuckoo binning on the receiver, OKVS payload on each sender.

    Nothing data-dependent is published: the cuckoo hash functions are public
    and fixed in advance, and each sender's payload is an OKVS whose values are
    pseudorandom. The sender evaluates every element at its K candidate rows
    and at every stash row, because it cannot know which one holds a match.
    """
    n = len(datasets)
    sizes = [d.size for d in datasets]
    m_max = max(sizes)
    ell = out_bits(n, m_max)
    keys = _pair_keys(n, rng)

    D_star = datasets[recv]
    m_star = D_star.size
    tbl = CuckooTable(D_star, rng=rng)
    n_stash = tbl.stash_size
    n_rows = tbl.bins + n_stash
    c.add(bins=tbl.bins, stash=n_stash)

    rows = np.zeros(n_rows, dtype=U64)
    occ = tbl.table >= 0
    rows[np.flatnonzero(occ)] = D_star[tbl.table[occ]]
    if n_stash:
        rows[tbl.bins:] = D_star[tbl.stash]

    # row holding each receiver element, and its candidate index:
    # t in [0, K) for a cuckoo bin, K + i for stash slot i
    row_of = np.empty(m_star, dtype=np.int64)
    row_of[tbl.table[occ]] = np.flatnonzero(occ)
    my_cand = candidate_bins(D_star, tbl.bins)
    which = np.full(m_star, -1, dtype=np.int64)
    for t in range(K_HASH - 1, -1, -1):
        which = np.where(my_cand[t] == row_of, t, which)
    if n_stash:
        row_of[tbl.stash] = np.arange(tbl.bins, n_rows)
        which[tbl.stash] = K_HASH + np.arange(n_stash)
    assert (which >= 0).all()

    s_star = zero_shares(D_star, recv, n, keys, ell, c)
    acc = s_star.copy()

    for j in range(n):
        if j == recv:
            continue
        eng = OPRFEngine(n_rows, ell, c, rng)
        M = eng.receiver_encode(rows)

        # sender: one OKVS entry per (element, candidate row)
        D_j = datasets[j]
        sj = zero_shares(D_j, j, n, keys, ell, c)
        cand = candidate_bins(D_j, tbl.bins)
        kv_keys, kv_vals = [], []
        for t in range(K_HASH):
            kv_keys.append(_okvs_key(D_j, t))
            kv_vals.append(eng.sender_eval(D_j, cand[t]) ^ sj)
        for i in range(n_stash):
            kv_keys.append(_okvs_key(D_j, K_HASH + i))
            kv_vals.append(eng.sender_eval(D_j, np.full(D_j.size, tbl.bins + i)) ^ sj)
        _t = time.perf_counter()
        table, seed = okvs.encode(np.concatenate(kv_keys),
                                  np.concatenate(kv_vals), ell, c)
        c.add(t_okvs=time.perf_counter() - _t)
        c.add(bytes_s2r=table.size * ell // 8)

        # receiver: decode exactly one entry per element
        dec = np.empty(m_star, dtype=U64)
        for t in np.unique(which):
            sel = which == t
            dec[sel] = okvs.decode(table, seed, _okvs_key(D_star[sel], t))
        acc ^= dec ^ M[row_of]

    return D_star[acc == 0]


# ------------------------------------------------------------------- API

def run(datasets, protocol="pkn", receiver="min", seed=0,
        trace_memory=True) -> Result:
    """Execute one protocol on a list of uint64 arrays."""
    global TRACE_MEMORY
    TRACE_MEMORY = trace_memory
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
    if protocol == "pkn":
        out, peak, sec = _run(_pkn_core, datasets, recv, c, rng, payload="okvs")
    elif protocol == "pkn_mphf_payload":
        out, peak, sec = _run(_pkn_core, datasets, recv, c, rng, payload="mphf")
    else:
        out, peak, sec = _run(_cuckoo_core, datasets, recv, c, rng)

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
