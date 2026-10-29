"""Explicit resource-model sensitivity and payload accounting using real sizes.

Network, power and client requirements are simulated. None of these estimates
are physical IoMT energy measurements. The adaptive policy is the minimum
permitted configuration, an explicit minimum-required-level policy.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROOT, save_json

PROFILES=[
    {'name':'constrained','bandwidth_mbps':.25,'tx_w':.08,'rx_w':.04,'cpu_w':.04,'compute_scale':8.},
    {'name':'intermediate','bandwidth_mbps':1.,'tx_w':.3,'rx_w':.15,'cpu_w':.2,'compute_scale':3.},
    {'name':'capable','bandwidth_mbps':5.,'tx_w':.8,'rx_w':.4,'cpu_w':1.,'compute_scale':1.}]


def select_level(required, costs, weights=(1,1,1)):
    feasible=np.arange(required-1,3)
    c=np.asarray(costs,dtype=float)
    # Normalize each dimension before weighted scalarization.
    normalized=c/np.maximum(c.max(axis=0),1e-12)
    return int(feasible[np.argmin(normalized[feasible]@np.asarray(weights))]+1)


def simulate(seed, payload, n_clients=17, rounds=30, bandwidth_scale=1.):
    bench=pd.read_csv(ROOT/'results/crypto/raw.csv')
    nearest=int(min(bench.payload_bytes.unique(),key=lambda n:abs(n-payload)))
    stats=bench[bench.payload_bytes==nearest].groupby('level').agg({'crypto_ms':'median','extra_bytes':'first'})
    rng=np.random.default_rng(seed)
    device_class=rng.integers(0,3,n_clients)
    # Declared synthetic required-level distribution for this benchmark.
    requirements=rng.choice([1,2,3],size=(rounds,n_clients),p=[.244,.712,.044])
    rows=[]
    for t in range(rounds):
        for policy in ['none','fixed_L2','fixed_L3','adaptive','minimum_feasible']:
            energies=[];latencies=[];extras=[];totals=[];satisfied=[];levels=[]
            for i in range(n_clients):
                profile=PROFILES[int(device_class[i])];required=int(requirements[t,i])
                costs=[]
                for level in [1,2,3]:
                    cr=float(stats.loc[level,'crypto_ms'])/1000*profile['compute_scale']
                    extra=int(stats.loc[level,'extra_bytes'])
                    comm=8*(payload+extra)/(profile['bandwidth_mbps']*1e6*bandwidth_scale)
                    energy=(profile['tx_w']+profile['rx_w'])*comm+2*profile['cpu_w']*cr
                    costs.append([2*(cr+comm),energy,extra])
                level=0 if policy=='none' else (required if policy=='minimum_feasible' else
                       select_level(required,costs) if policy=='adaptive' else int(policy[-1]))
                if level:
                    lat,en,extra=costs[level-1]
                else:
                    comm=8*payload/(profile['bandwidth_mbps']*1e6*bandwidth_scale)
                    lat=2*comm;en=(profile['tx_w']+profile['rx_w'])*comm;extra=0
                # Synchronous communication critical path; energy sums clients.
                latencies.append(lat*1000);energies.append(en*1000)
                extras.append(2*extra);totals.append(2*(payload+extra));levels.append(level)
                satisfied.append(level>=required)
            rows.append({'seed':seed,'round':t+1,'policy':policy,'payload_bytes':payload,'bandwidth_scale':bandwidth_scale,
                         'latency_ms':max(latencies),'energy_mj':sum(energies),'extra_bytes':sum(extras),'total_bytes':sum(totals),
                         'security_satisfaction':float(np.mean(satisfied)),
                         'overprovision':float(np.maximum(np.array(levels)-requirements[t],0).mean()),
                         'benchmark_payload_bytes':nearest})
    return rows


def main():
    rows=[]
    for seed in [11,22,33,44,55]:
        for size in [4096,65536,164000,1048576]:
            for scale in [.25,1.,4.]:rows.extend(simulate(seed,size,bandwidth_scale=scale))
    out=ROOT/'results/resources';out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'simulations.csv',index=False)
    save_json(out/'assumptions.json',{'profiles':PROFILES,
        'client_requirements':'Independent per round categorical [0.244,0.712,0.044], inferred from Table 3; not recovered original client traces',
        'n_clients':17,'rounds':30,'two_directions':True,'fresh_keys_per_direction':True,
        'latency':'max-client upload+download+crypto; no training or server aggregation in this communication-only table',
        'energy':'sum-client simulated radio plus CPU energy; no training energy in this table',
        'compute_scaling':'Assumed multiplicative factors 8/3/1 applied to measured host crypto time',
        'adaptive':'Enumerate feasible levels; minimize normalized latency+energy+extra bytes; security requirement fixed externally',
        'interpretation':'100% satisfaction is enforcement of an input policy constraint, not evidence of cryptographic security',
        'minimum_feasible_baseline':True,'timings':'Median measured host time at nearest benchmark payload size, not device measurements',
        'sensitivity':'4 payload sizes x 3 bandwidth multipliers x 5 paired workload seeds'})
    # Validation of the system-level claims against independent controls.
    controls={}
    for p in (ROOT/'results/runs').glob('control_v2_*/metrics.json'):
        m=json.loads(p.read_text());controls[p.parent.name]={'initial':m['initial_state_hash'],'final':m['final_state_hash']}
    if len(controls)==4:
        assert len({m['initial'] for m in controls.values()})==1
        assert len({m['final'] for m in controls.values()})==1
    save_json(out/'transport_controls.json',{'runs':controls,'identical_final_models':len(controls)==4 and len({m['final'] for m in controls.values()})==1})


if __name__=='__main__':main()
