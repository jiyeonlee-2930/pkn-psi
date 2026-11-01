"""Correctness and theorem checks.

    pytest -q
"""

import numpy as np
import pytest

from pknpsi.cuckoo import CuckooTable, EPS, K_HASH
from pknpsi.datagen import make_datasets, imbalanced_sizes
from pknpsi.mphf import MPHF
from pknpsi.protocols import run


# ------------------------------------------------------------ MPHF
@pytest.mark.parametrize("m", [1, 2, 17, 1000, 5000])
@pytest.mark.parametrize("alpha", [0.94, 1.0])
def test_mphf_is_bijective(m, alpha):
    keys = np.unique(np.random.default_rng(m).integers(
        0, 2 ** 62, size=m, dtype=np.int64).astype(np.uint64))
    h = MPHF(keys, alpha=alpha)
    pos = h(keys)
    assert pos.min() >= 0 and pos.max() < keys.size
    assert np.unique(pos).size == keys.size


def test_mphf_defined_outside_the_key_set():
    rng = np.random.default_rng(0)
    keys = np.unique(rng.integers(0, 2 ** 62, size=500,
                                  dtype=np.int64).astype(np.uint64))
    h = MPHF(keys)
    other = rng.integers(0, 2 ** 62, size=500, dtype=np.int64).astype(np.uint64)
    pos = h(other)
    assert pos.min() >= 0 and pos.max() < keys.size


def test_mphf_has_no_spare_bins_unlike_cuckoo():
    keys = np.unique(np.random.default_rng(1).integers(
        0, 2 ** 62, size=2000, dtype=np.int64).astype(np.uint64))
    m = keys.size
    assert MPHF(keys, alpha=1.0).m == m           # bins == elements
    tbl = CuckooTable(keys)
    assert tbl.bins >= int(EPS * m)               # cuckoo needs spare bins


# --------------------------------------------------------- protocols
@pytest.mark.parametrize("proto", ["pkn", "cuckoo"])
@pytest.mark.parametrize("sizes", [[300, 300, 300], [200, 800, 1500],
                                   [100, 100], [150, 600, 600, 600]])
def test_intersection_is_exact(proto, sizes):
    ds = make_datasets(sizes, overlap=0.3, seed=11)
    r = run(ds, protocol=proto, receiver="min", seed=3)
    assert r.correct


def test_empty_intersection():
    ds = make_datasets([200, 600, 600], overlap=0.0, seed=12)
    r = run(ds, protocol="pkn", receiver="min", seed=4)
    assert r.correct and r.intersection_size == 0


def test_full_intersection():
    base = make_datasets([300], overlap=0.0, seed=13)[0]
    r = run([base, base, base], protocol="pkn", receiver="min", seed=5)
    assert r.correct and r.intersection_size == base.size


# ---------------------------------------------------------- theorems
def test_theorem2_ot_instances_equal_receiver_set_size():
    """MPHF binning uses exactly m_min OT rows per sender."""
    ds = make_datasets(imbalanced_sizes(3, 500, 4), 0.3, 21)
    r = run(ds, protocol="pkn", receiver="min", seed=6)
    assert r.counters.ot_instances == 500 * 2
    assert r.counters.bins == 500


def test_theorem2_single_placement():
    """The sender places each element once, not K_HASH times."""
    ds = make_datasets(imbalanced_sizes(3, 500, 4), 0.3, 22)
    a = run(ds, protocol="cuckoo", receiver="min", seed=6)
    b = run(ds, protocol="pkn", receiver="min", seed=6)
    assert a.counters.placements == K_HASH * b.counters.placements


@pytest.mark.parametrize("delta", [2, 5, 10])
def test_theorem3_reduction_factor_is_eps_times_delta(delta):
    """Choosing the smallest set as receiver saves a factor of EPS * delta."""
    ds = make_datasets(imbalanced_sizes(3, 400, delta), 0.3, 23)
    worst = run(ds, protocol="cuckoo", receiver="max", seed=6)
    best = run(ds, protocol="pkn", receiver="min", seed=6)
    ratio = worst.counters.ot_instances / best.counters.ot_instances
    assert ratio == pytest.approx(EPS * delta, rel=0.02)


# ------------------------------------------- revision: stash, OKVS, OT
def test_cuckoo_correct_with_nonempty_stash(monkeypatch):
    """Receiver elements in the stash must still be found."""
    import pknpsi.cuckoo as ck
    monkeypatch.setattr(ck, "MAX_EVICT", 1)        # force stash occupancy
    ds = make_datasets([400, 900, 900], overlap=0.5, seed=31)
    r = run(ds, protocol="cuckoo", receiver="min", seed=7)
    assert r.counters.stash > 0
    assert r.correct


def test_okvs_roundtrip():
    from pknpsi import okvs
    rng = np.random.default_rng(5)
    k = np.unique(rng.integers(0, 2 ** 63, 5000, dtype=np.int64).astype(np.uint64))
    v = rng.integers(0, 2 ** 50, k.size, dtype=np.int64).astype(np.uint64)
    table, seed = okvs.encode(k, v, 50)
    assert (okvs.decode(table, seed, k) == v).all()
    assert table.size <= int(np.ceil(okvs.EXPANSION * k.size)) + 3


def test_base_ot_delivers_chosen_message_only():
    import secrets
    from pknpsi.crypto import base_ots
    pairs = [(secrets.token_bytes(16), secrets.token_bytes(16)) for _ in range(64)]
    choices = [secrets.randbits(1) for _ in range(64)]
    out = base_ots(pairs, choices)
    assert all(o == p[c] for o, p, c in zip(out, pairs, choices))
    assert all(o != p[1 - c] for o, p, c in zip(out, pairs, choices))


def test_oprf_matches_only_on_equal_items():
    from pknpsi.metrics import Counters
    from pknpsi.oprf import OPRFEngine
    rng = np.random.default_rng(9)
    R = rng.integers(0, 2 ** 63, 64, dtype=np.int64).astype(np.uint64)
    eng = OPRFEngine(64, 64, Counters())
    M = eng.receiver_encode(R)
    same = eng.sender_eval(R, np.arange(64))
    other = eng.sender_eval(R[::-1].copy(), np.arange(64))
    assert (same == M).all()
    assert (other != M).sum() >= 63          # R[::-1] agrees with R at most at one row


@pytest.mark.parametrize("delta", [2, 5, 10])
def test_receiver_designation_isolated_factor_is_delta(delta):
    """Same binning (cuckoo), only the receiver changes: OT ratio = delta."""
    ds = make_datasets(imbalanced_sizes(3, 400, delta), 0.3, 24)
    worst = run(ds, protocol="cuckoo", receiver="max", seed=6)
    best = run(ds, protocol="cuckoo", receiver="min", seed=6)
    assert worst.correct and best.correct
    ratio = worst.counters.ot_instances / best.counters.ot_instances
    assert ratio == pytest.approx(delta, rel=0.02)
