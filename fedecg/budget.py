"""Replay frozen model trajectories under explicit modeled system costs.

No clients are dropped and protection cannot change model accuracy at an equal
round. This evaluates model utility vs budget using post-selection checkpoint
replay, not a new real-time FL deployment or additional training rounds.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .common import ROOT,save_json
from .data import load_split
from .models import make_model
from .resources import PROFILES,select_level
from .train import metrics,predict


def main():
    selected=json.loads((ROOT/'results/selection.json').read_text())
    bench=pd.read_csv(ROOT/'results/crypto/raw.csv')
    output=[]
    d,tr,va,te=load_split('strict')
    for model,loss in [('mlp','ce'),(selected['model'],selected['loss'])]:
        for seed in [44,55]:
            folder=ROOT/f'results/runs/main_{model}_{loss}_s{seed}'
            cfg=json.loads((folder/'config.json').read_text());m=json.loads((folder/'metrics.json').read_text())
            device=cfg['device'];torch.set_num_threads(1)
            net=make_model(model).to(device)
            x=torch.from_numpy(d['x'][:,None]).to(device);rr=torch.from_numpy(d['rr']).to(device)
            trace=pd.read_csv(folder/'client_trace.csv');n_clients=int(trace.client.max()+1)
            rng=np.random.default_rng(seed+20000)
            requirements=rng.choice([1,2,3],size=(30,n_clients),p=[.244,.712,.044])
            profiles=np.random.default_rng(seed+40000).integers(0,3,n_clients)
            nearest=min(bench.payload_bytes.unique(),key=lambda n:abs(n-m['raw_parameter_bytes']))
            stats=bench[bench.payload_bytes==nearest].groupby('level').agg({'crypto_ms':'median','extra_bytes':'first'})
            totals={p:[0.,0.,0] for p in ['none','fixed_L2','fixed_L3','adaptive']}
            for round_idx in range(1,31):
                state=torch.load(folder/f'checkpoint_round{round_idx:03d}.pt',weights_only=True,map_location='cpu')
                net.load_state_dict(state)
                score=metrics(d['y'][te],predict(net,x,rr,te,device))['macro_f1']
                for policy in totals:
                    lat=[];energy=[];byte_count=0
                    for row in trace[trace['round']==round_idx].itertuples(index=False):
                        profile=PROFILES[int(profiles[row.client])];payload=int(row.payload_bytes)
                        costs=[]
                        for level in [1,2,3]:
                            cr=float(stats.loc[level,'crypto_ms'])/1000*profile['compute_scale']
                            extra=int(stats.loc[level,'extra_bytes'])
                            comm=8*(payload+extra)/(profile['bandwidth_mbps']*1e6)
                            costs.append([2*(cr+comm),(profile['tx_w']+profile['rx_w'])*comm+2*profile['cpu_w']*cr,extra])
                        level=0 if policy=='none' else select_level(int(requirements[round_idx-1,row.client]),costs) if policy=='adaptive' else int(policy[-1])
                        if level:
                            latency,en,extra=costs[level-1]
                        else:
                            comm=8*payload/(profile['bandwidth_mbps']*1e6)
                            latency=2*comm;en=(profile['tx_w']+profile['rx_w'])*comm;extra=0
                        training=row.training_seconds*profile['compute_scale']
                        lat.append(training+latency);energy.append(en+training*profile['cpu_w'])
                        byte_count+=2*(payload+extra)
                    totals[policy][0]+=max(lat);totals[policy][1]+=sum(energy);totals[policy][2]+=byte_count
                    output.append({'model':model,'loss':loss,'seed':seed,'round':round_idx,'policy':policy,
                                   'test_macro_f1':score,'cumulative_seconds':totals[policy][0],
                                   'cumulative_joules':totals[policy][1],'cumulative_bytes':totals[policy][2]})
            print('Budget replay complete',model,seed,flush=True)
    out=ROOT/'results/resources';pd.DataFrame(output).to_csv(out/'budget_replay.csv',index=False)
    save_json(out/'budget_assumptions.json',{
        'seeds':[44,55],'models':'Baseline and enhancement selected using validation only before this replay',
        'rounds':30,'checkpoint_selection':'All checkpoints replayed after model family selection; no test curve used for tuning',
        'training_cost':'Actual per-client host training wall time multiplied by assumed device compute scale',
        'system_latency':'max parallel client training plus protected upload and download, then zero modeled server aggregation time',
        'energy':'Sum assumed CPU and radio power multiplied by modeled training/communication durations',
        'claim_limit':'Illustrative budget sensitivity, not physical device energy or empirical asynchronous FL; no new rounds beyond 30 or dropped updates'})


if __name__=='__main__':main()
