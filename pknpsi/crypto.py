"""Cryptographic primitives (real, not simulated).

  PRF          AES-128 in ECB mode on one 16-byte block (x || tweak), truncated.
  PRG          AES-128 in CTR mode keyed by a 16-byte seed.
  codeword C   KKRT16 pseudo-random code: AES under a public code key,
               4 blocks per element, truncated to W_BITS = 448 bits.
  H            correlation-robust hash: SHA-256(b || row), truncated to ell bits.
  base OT      Chou-Orlandi "Simplest OT" over secp256k1 (semi-honest,
               CDH + random oracle).

All primitives operate on numpy arrays; the AES calls are batched so that one
call to the OpenSSL backend processes a whole array.
"""

import hashlib
import secrets

import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

U64 = np.uint64
W_BITS = 448
W_BYTES = W_BITS // 8

# Public key of the pseudo-random code (fixed per deployment; in KKRT16 it may
# be sampled by the sender after the base OTs).
CODE_KEY = bytes.fromhex("6b6b72743136636f64656b6579303031")


# ----------------------------------------------------------------- AES core

def _aes_ecb(key: bytes, data: bytes) -> bytes:
    enc = Cipher(algorithms.AES(key), modes.ECB()).encryptor()
    return enc.update(data) + enc.finalize()


def _blocks(x: np.ndarray, tweak: int) -> bytes:
    """Pack uint64 values with a 64-bit tweak into 16-byte AES blocks."""
    x = np.asarray(x, dtype=U64)
    blk = np.empty((x.size, 2), dtype="<u8")
    blk[:, 0] = x
    blk[:, 1] = np.uint64(tweak & 0xFFFFFFFFFFFFFFFF)
    return blk.tobytes()


def prf_u64(key: bytes, x, tweak: int = 0) -> np.ndarray:
    """F_k(x || tweak) truncated to 64 bits, for an array of uint64 inputs."""
    x = np.asarray(x, dtype=U64)
    if x.size == 0:
        return np.zeros(0, dtype=U64)
    out = np.frombuffer(_aes_ecb(key, _blocks(x, tweak)), dtype="<u8")
    return out.reshape(-1, 2)[:, 0].astype(U64)


def prg(seed: bytes, nbytes: int) -> np.ndarray:
    """Expand a 16-byte seed into nbytes pseudo-random bytes (AES-CTR)."""
    enc = Cipher(algorithms.AES(seed), modes.CTR(b"\x00" * 16)).encryptor()
    return np.frombuffer(enc.update(b"\x00" * nbytes) + enc.finalize(),
                         dtype=np.uint8)


def codeword(items) -> np.ndarray:
    """C: {0,1}^64 -> {0,1}^448, rows of W_BYTES bytes."""
    items = np.asarray(items, dtype=U64)
    parts = [np.frombuffer(_aes_ecb(CODE_KEY, _blocks(items, i)), dtype=np.uint8)
             .reshape(items.size, 16) for i in range(4)]
    return np.concatenate(parts, axis=1)[:, :W_BYTES].copy()


def corr_hash(index, rows: np.ndarray, out_bits: int) -> np.ndarray:
    """H(b, row) = SHA-256(b || row) truncated to out_bits, one value per row."""
    index = np.asarray(index, dtype="<u8")
    rows = np.ascontiguousarray(rows, dtype=np.uint8)
    out = np.empty(rows.shape[0], dtype=U64)
    sha = hashlib.sha256
    ib = index.tobytes()
    rb = rows.tobytes()
    w = rows.shape[1]
    for i in range(rows.shape[0]):
        d = sha(ib[8 * i:8 * i + 8] + rb[w * i:w * i + w]).digest()
        out[i] = int.from_bytes(d[:8], "little")
    if out_bits < 64:
        out &= U64((1 << out_bits) - 1)
    return out


# ------------------------------------------------------------------ base OT
#
# Chou-Orlandi "Simplest OT" (LATINCRYPT 2015) over secp256k1, semi-honest.
#   sender:   a,  A = aG                                   (sends A)
#   receiver: b,  B = bG if c = 0 else A + bG              (sends B)
#   sender:   k0 = KDF(aB), k1 = KDF(a(B - A)); sends E_i = k_i xor m_i
#   receiver: k_c = KDF(bA), outputs m_c

from coincurve import PrivateKey, PublicKey

_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
POINT_BYTES = 33


def _scalar() -> bytes:
    while True:
        k = secrets.randbelow(_N)
        if k:
            return k.to_bytes(32, "big")


def _kdf(point: PublicKey, i: int) -> bytes:
    return hashlib.sha256(i.to_bytes(4, "little") +
                          point.format(compressed=True)).digest()[:16]


def _xor(a: bytes, b: bytes) -> bytes:
    return bytes(x ^ y for x, y in zip(a, b))


def base_ots(pairs, choices, counters=None):
    """Run len(choices) 1-out-of-2 OTs on 16-byte messages.

    pairs   list of (m0, m1) held by the OT sender
    choices list of bits held by the OT receiver
    returns the receiver's outputs [m_{c_i}]
    Each role uses only its own secrets and the transmitted messages.
    """
    import time as _time
    _t0 = _time.perf_counter()
    k = len(choices)
    # sender
    a = _scalar()
    A = PublicKey.from_secret(a)
    negaA = A.multiply(a).multiply((_N - 1).to_bytes(32, "big"))
    # receiver
    bs = [_scalar() for _ in range(k)]
    Bs = [PublicKey.from_secret(b) if c == 0
          else PublicKey.combine_keys([A, PublicKey.from_secret(b)])
          for b, c in zip(bs, choices)]
    # sender
    cts = []
    for i, (B, (m0, m1)) in enumerate(zip(Bs, pairs)):
        aB = B.multiply(a)
        k0 = _kdf(aB, i)
        k1 = _kdf(PublicKey.combine_keys([aB, negaA]), i)
        cts.append((_xor(k0, m0), _xor(k1, m1)))
    # receiver
    out = [_xor(_kdf(A.multiply(b), i), cts[i][c])
           for i, (b, c) in enumerate(zip(bs, choices))]
    if counters is not None:
        counters.add(base_ots=k,
                     bytes_baseot=POINT_BYTES * (1 + k) + 2 * 16 * k,
                     t_baseot=_time.perf_counter() - _t0)
    return out
