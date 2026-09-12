# pkn-psi
PKN-PSI

Reference implementation of the protocol described in:

A Minimal Perfect Hashing-Based Multi-Party PSI Protocol for Size-Imbalanced Datasets Jiyeon Lee, Department of AI Information Security, Halla University

Overview

PKN-PSI is a multi-party private set intersection (PSI) protocol for settings in which participant set sizes are highly imbalanced. It replaces cuckoo hashing with a minimal perfect hash function (MPHF) and designates the holder of the smallest set as the receiver, so that the number of oblivious transfer instances scales with the smallest set rather than the largest. Partial-intersection leakage is prevented by zero-sharing.

Security is analyzed in the semi-honest model.

Status

Code is being migrated to this repository and will be published here in full.

Data

No datasets are distributed. All experiments use synthetic sets of randomly generated 128-bit integers, with size profiles modeled on published institutional statistics. Generation scripts are included with the code, so all reported results are reproducible from source.

Requirements
Python 3.10 or later

Dependencies are listed in requirements.txt.
