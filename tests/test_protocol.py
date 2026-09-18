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
