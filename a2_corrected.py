"""Controlled 2x2 A2 experiment for PKN-PSI.

Run from the project root:
    .\.venv\Scripts\python.exe a2_corrected.py
    .\.venv\Scripts\python.exe a2_corrected.py --m-min 1000 --repeats 3
"""
import argparse
import csv
from pathlib import Path
import statistics
from pknpsi.datagen import make_datasets, imbalanced_sizes
from pknpsi.protocols import run

FIELDS = ['delta','repeat','n','m_min','m_max','protocol','receiver','recv_size','correct','ot_instances','sym_ops','placements','bytes_total','peak_KiB','sec','build_retries','index_bits']

def main():
    ap = argparse.ArgumentParser(description='A2 controlled 2x2 factorial comparison')
    ap.add_argument('--m-min', type=int, default=300)
    ap.add_argument('--repeats', type=int, default=1)
    ap.add_argument('--n', type=int, default=3)
    ap.add_argument('--deltas', type=int, nargs='+', default=[1,2,5,10])
    ap.add_argument('--output', default='a2_corrected_results.csv')
    args = ap.parse_args()
    if args.m_min < 2 or args.repeats < 1 or args.n < 2 or any(d < 1 for d in args.deltas):
        ap.error('m-min >= 2, repeats >= 1, n >= 2, deltas >= 1 required')
    rows=[]
    for delta in args.deltas:
        sizes = imbalanced_sizes(args.n,args.m_min,delta)
        for rep in range(args.repeats):
            seed = 20261017 + rep
            ds = make_datasets(sizes, overlap=0.3, seed=seed)
            for protocol in ('cuckoo','pkn'):
                for receiver in ('min','max'):
                    print(f'Running delta={delta}, repeat={rep+1}/{args.repeats}, {protocol}-{receiver} ...', flush=True)
                    r = run(ds,protocol=protocol,receiver=receiver,seed=seed)
                    c = r.counters
                    row = dict(delta=delta,repeat=rep+1,n=args.n,m_min=min(sizes),m_max=max(sizes),
                        protocol=protocol,receiver=receiver,recv_size=r.receiver_size,correct=r.correct,
                        ot_instances=c.ot_instances,sym_ops=c.sym_ops,placements=c.placements,
                        bytes_total=c.bytes_total,peak_KiB=round(r.peak_kib,2),sec=round(r.seconds,3),
                        build_retries=c.build_retries,index_bits=c.index_bits)
                    rows.append(row)
                    print(f"  correct={r.correct}, OT={c.ot_instances}, bytes={c.bytes_total}, time={r.seconds:.3f}s",flush=True)
                    if not r.correct:
                        raise RuntimeError(f'Incorrect intersection for {protocol}-{receiver}, delta={delta}')
            # save progress even if a later condition takes a long time
            with open(args.output,'w',newline='',encoding='utf-8-sig') as f:
                w=csv.DictWriter(f,fieldnames=FIELDS)
                w.writeheader(); w.writerows(rows)
    print(f'\nSaved {len(rows)} records to {Path(args.output).resolve()}')
    print('NOTE: OT and bytes are model counters, not real network/cryptographic benchmark measurements.')
    print('NOTE: At delta=1, min/max may select the same receiver due to ties.')

if __name__ == '__main__':
    main()
