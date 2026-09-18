"""Instrumented counters.

Every quantity reported in the paper is accumulated here so that the two
protocols are measured by the same instrument. Counters are
implementation-independent: they do not depend on Python speed.
"""

from dataclasses import dataclass, field, asdict


@dataclass
class Counters:
    # --- computation -------------------------------------------------
    ot_instances: int = 0        # extension rows == bins of the receiver
    base_ots: int = 0            # public-key operations
    sym_ops: int = 0             # PRF / hash invocations
    placements: int = 0          # sender-side bin placements
    build_retries: int = 0       # MPHF pilot-search retries

    # --- structure ---------------------------------------------------
    bins: int = 0                # receiver table size
    stash: int = 0               # cuckoo stash occupancy
    index_bits: int = 0          # published index structure size, bits

    # --- communication (bytes) --------------------------------------
    bytes_r2s: int = 0           # receiver -> sender (OT extension matrix)
    bytes_s2r: int = 0           # sender -> receiver (OPRF outputs)
    bytes_index: int = 0         # broadcast of the published MPHF

    def add(self, **kw):
        for k, v in kw.items():
            setattr(self, k, getattr(self, k) + v)

    @property
    def bytes_total(self) -> int:
        return self.bytes_r2s + self.bytes_s2r + self.bytes_index

    def as_dict(self) -> dict:
        d = asdict(self)
        d["bytes_total"] = self.bytes_total
        return d


@dataclass
class Result:
    protocol: str
    n_parties: int
    sizes: list
    delta: float
    epsilon: float
    receiver_index: int
    receiver_size: int
    intersection_size: int
    correct: bool
    counters: Counters = field(default_factory=Counters)
    peak_kib: float = 0.0
    seconds: float = 0.0

    def row(self) -> dict:
        d = {
            "protocol": self.protocol,
            "n": self.n_parties,
            "m_min": min(self.sizes),
            "m_max": max(self.sizes),
            "delta": round(self.delta, 3),
            "epsilon": round(self.epsilon, 4),
            "recv_size": self.receiver_size,
            "|I|": self.intersection_size,
            "correct": self.correct,
            "peak_KiB": round(self.peak_kib, 1),
            "sec": round(self.seconds, 3),
        }
        d.update(self.counters.as_dict())
        return d
