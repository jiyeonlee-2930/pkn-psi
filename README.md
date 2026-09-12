# pkn-psi
# PKN-PSI

Reference implementation of the protocol described in:

> **A Minimal Perfect Hashing-Based Multi-Party PSI Protocol for Size-Imbalanced Datasets**
> Jiyeon Lee, Department of AI Information Security, Halla University

## Overview

PKN-PSI is a multi-party private set intersection (PSI) protocol for settings in which
participant set sizes are highly imbalanced. It replaces cuckoo hashing with a minimal
perfect hash function (MPHF) and designates the holder of the smallest set as the receiver,
so that the number of oblivious transfer instances scales with the smallest set rather than
the largest. Partial-intersection leakage is prevented by zero-sharing.

Security is analyzed in the semi-honest model.

## Status

Code is being migrated to this repository and will be published here in full.

## Requirements

- Python 3.10 or later

Dependencies are listed in `requirements.txt`.

## Installation

```bash
git clone https://github.com/<username>/pkn-psi.git
cd pkn-psi
pip install -r requirements.txt
```

## Repository layout

```
pkn-psi/
├── pknpsi/                  # Protocol implementation
│   ├── __init__.py
│   ├── mphf.py              # Minimal perfect hash function (PTHash)
│   ├── cuckoo.py            # Cuckoo hashing baseline
│   ├── oprf.py              # OPRF over OT extension
│   ├── protocols.py         # PKN-PSI and baseline protocols
│   ├── datagen.py           # Synthetic dataset generation
│   └── metrics.py           # Symmetric-key ops, memory, transmitted bytes
├── experiments/
│   ├── run_main.py          # Main comparison
│   ├── run_ablation.py      # Ablation studies A1-A4
│   ├── plot_delta.py        # Figure: cost vs. delta
│   ├── plot_example1.py     # Figure: Example 1 illustration
│   └── find_seed.py         # MPHF seed search utility
├── results/                 # Generated tables and figures (git-ignored)
├── tests/
│   └── test_protocol.py
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── LICENSE
└── README.md
```

## Reproducing the results

Each script writes its output to `results/`.

| Paper item | Command |
|---|---|
| Main comparison tables | `python -m experiments.run_main` |
| Ablation tables (A1-A4) | `python -m experiments.run_ablation` |
| Figure: cost vs. delta | `python -m experiments.plot_delta` |
| Figure: Example 1 | `python -m experiments.plot_example1` |

To run the test suite:

```bash
pytest
```

## Data

No datasets are distributed. All experiments use synthetic sets of randomly generated
128-bit integers, with size profiles modeled on published institutional statistics.
Generation is handled by `pknpsi/datagen.py`, so all reported results are reproducible
from source.

## Citation

To be added upon publication.

## Funding

This work was supported by the Korea Internet & Security Agency (KISA, RS-2026-25527707)
and the ANCHOR program (2026-ANCHOR-10-008).

## License

To be added.
