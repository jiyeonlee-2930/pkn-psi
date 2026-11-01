
### Table 3. E1 (n = 5, δ = 10, reps = 10; mean ± 95% CI)

| Protocol | OT inst. | Sym. ops | Bytes | Peak KiB | Compute s | Total LAN s | Total WAN s |
|---|---|---|---|---|---|---|---|
| Baseline | 101,600 | 903,200 | 7,529,037 | 21,015 ± 0 | 1.37 ± 0.02 | 1.44 | 2.45 |
| PKN-std | 10,160 | 828,320 | 2,909,468 | 7,942 ± 0 | 0.80 ± 0.01 | 0.83 | 1.51 |
| PKN-MPHF | 8,000 | 504,000 | 1,306,466 | 5,585 ± 0 | 0.46 ± 0.00 | 0.48 | 1.05 |
| Reduction (MPHF vs Baseline) | 92.1% | 44.2% | 82.6% | 73.4% | 66.1% | 66.7% | 57.2% |

MPHF build 0.023 s, base OTs 0.217 s, OKVS encode 0.047 s (PKN-MPHF, mean).

### Table 4. Reduction of PKN-PSI (MPHF) over the baseline vs δ (n = 3, m_min = 2000)

| δ | Bytes (%) | Peak mem. (%) | Sym. ops (%) | WAN time (%) | OT ratio |
|---|---|---|---|---|---|
| 1 | 28.9 | 29.8 | 39.3 | 6.1 | 1.27 |
| 2 | 55.0 | 61.0 | 42.3 | 14.9 | 2.54 |
| 4 | 70.9 | 68.8 | 44.4 | 28.3 | 5.08 |
| 8 | 79.7 | 70.9 | 45.6 | 44.6 | 10.16 |
| 16 | 84.2 | 72.0 | 46.2 | 59.9 | 20.32 |
| 32 | 86.5 | 72.5 | 46.6 | 71.8 | 40.64 |

### Table 5. A1: binning isolated (receiver = smallest set for both)

| Setting | OT cuckoo | OT MPHF | OT red. (%) | Bytes cuckoo | Bytes MPHF | Bytes red. (%) |
|---|---|---|---|---|---|---|
| n = 3, m_max = 2000 | 5,080 | 4,000 | 21.3 | 446,584 | 317,594 | 28.9 |
| n = 3, m_max = 8000 | 5,080 | 4,000 | 21.3 | 772,234 | 426,144 | 44.8 |
| n = 5, m_max = 2000 | 10,160 | 8,000 | 21.3 | 897,068 | 635,662 | 29.1 |
| n = 5, m_max = 8000 | 10,160 | 8,000 | 21.3 | 1,560,068 | 856,666 | 45.1 |

### Table 6. A2: 2×2 factorial, OT instances (n = 3, m_min = 2000)

| δ | Cuckoo/largest | Cuckoo/smallest | MPHF/largest | MPHF/smallest | Receiver effect (cuckoo) | Receiver effect (MPHF) | Binning effect |
|---|---|---|---|---|---|---|---|
| 1 | 5,080 | 5,080 | 4,000 | 4,000 | 1.00 | 1.00 | 1.27 |
| 2 | 10,160 | 5,080 | 8,000 | 4,000 | 2.00 | 2.00 | 1.27 |
| 10 | 50,800 | 5,080 | 40,000 | 4,000 | 10.00 | 10.00 | 1.27 |
| 50 | 254,000 | 5,080 | 200,000 | 4,000 | 50.00 | 50.00 | 1.27 |

### Table 7. A3: number of parties (δ = 5, m_min = 2000)

| n | Sym. ops baseline | Sym. ops std. | Sym. ops MPHF | Sym. red. (%) | Bytes red. (%) |
|---|---|---|---|---|---|
| 3 | 166,800 | 174,160 | 92,000 | 44.8 | 74.4 |
| 5 | 461,600 | 428,320 | 264,000 | 42.8 | 75.8 |
| 10 | 1,548,600 | 1,413,720 | 1,044,000 | 32.6 | 76.5 |
| 20 | 5,222,600 | 4,884,520 | 4,104,000 | 21.4 | 76.7 |

### Table E2. Set-size scalability (n = 3, δ = 4)

| m_min | m_max | Baseline s | PKN s | of which MPHF build s | Bytes baseline | Bytes PKN | Bytes red. (%) |
|---|---|---|---|---|---|---|---|
| 4,096 | 16,384 | 0.68 | 0.28 | 0.04 | 2,947,927 | 817,016 | 72.3 |
| 16,384 | 65,536 | 2.33 | 0.74 | 0.18 | 11,696,599 | 3,136,132 | 73.2 |
| 65,536 | 262,144 | 10.67 | 3.35 | 0.74 | 46,930,904 | 12,541,054 | 73.3 |

### Leakage. Membership-inference AUC of a simple attack on the public h* (0.5 = no advantage)

| δ | Overlap (D_j ∩ D*) / m* | AUC (mean ± sd) |
|---|---|---|
| 1 | 0.3 | 0.595 ± 0.012 |
| 2 | 0.3 | 0.573 ± 0.009 |
| 2 | 1.0 | 0.769 ± 0.004 |
| 5 | 0.3 | 0.555 ± 0.006 |
| 5 | 1.0 | 0.650 ± 0.002 |
| 10 | 0.3 | 0.545 ± 0.008 |
| 10 | 1.0 | 0.609 ± 0.001 |
