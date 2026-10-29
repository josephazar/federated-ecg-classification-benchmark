"""Run serial, resumable study phases using the active virtual environment."""
import argparse
import json
import subprocess
import sys
import os
import torch
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fedecg.common import save_json
from fedecg.train import run

SCREEN_CANDIDATES=[('mlp','ce'),('mlp','weighted'),('mlp','focal'),('cnn','ce'),('cnn','weighted'),('cnn','focal'),('cnn_rr','weighted'),('cnn_rr','focal')]
FINAL_CANDIDATES=SCREEN_CANDIDATES+[(m,l) for m in ['mlp_rr','cnn_spatial_rr'] for l in ['ce','weighted','focal']]+[('mlp','balanced'),('mlp_rr','balanced')]


def config(model='mlp',loss='ce',seed=11,**kw):
    requested=os.getenv('FEDECG_DEVICE','auto')
    device=('cpu' if model.startswith('mlp') else 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu') if requested=='auto' else requested
    c=dict(model=model,loss=loss,seed=seed,protocol='strict',partition='patient',rounds=30,
           local_epochs=1,batch_size=128,lr=.001,participation=1.,device=device,threads=1,
           n_clients=10,alpha=.5,train_size=0,transport='none',save_rounds=False)
    c.update(kw);return c


def launch(label,cfg):
    save_json(ROOT/f'configs/runs/{label}.json',cfg)
    run(cfg,ROOT/f'results/runs/{label}')


def select(candidates=FINAL_CANDIDATES):
    # Do not load test endpoints here. Selection is exclusively validation based.
    groups={};seen=set()
    for p in (ROOT/'results/runs').glob('main_*/metrics.json'):
        c=json.loads((p.parent/'config.json').read_text());m=json.loads(p.read_text())
        if c['seed'] not in [11,22,33]:
            continue
        if (c['model'],c['loss']) not in candidates:
            continue
        key=(c['model'],c['loss'],c['seed'])
        if key in seen:
            raise ValueError(f'Duplicate screening condition {key}')
        seen.add(key)
        groups.setdefault((c['model'],c['loss']),[]).append(m['best_validation_macro_f1'])
    expected={(m,l,s) for m,l in candidates for s in [11,22,33]}
    if seen!=expected:
        raise ValueError(f'Incomplete screening grid; missing {sorted(expected-seen)}')
    scores={f'{k[0]}_{k[1]}':sum(v)/len(v) for k,v in groups.items()}
    winner=max(scores,key=scores.get)
    model,loss=winner.rsplit('_',1)
    selected={'model':model,'loss':loss,'selection_metric':'mean best validation macro F1 over screening seeds',
              'validation_scores':scores,'screening_seeds':[11,22,33],
              'candidate_grid':candidates,'complete_screening_grid':True,
              'algorithmic_selection_reads_test_metrics':False,
              'exploratory_test_reuse':True,
              'test_metrics_used':False,
              'test_metrics_used_scope':'The mechanical selection function does not read test scores; DS2 was inspected during exploratory study development.'}
    save_json(ROOT/'results/selection.json',selected)
    return selected


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['pilot','subset_reference','screen','rr_extension','balanced_extension','spatial_extension','confirm','robustness','controls','central','data_volume'])
    p.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    a=p.parse_args();os.environ['FEDECG_DEVICE']=a.device
    if a.phase=='pilot':
        launch('pilot_cnn',config('cnn','weighted',11,rounds=2))
    elif a.phase=='subset_reference':
        for seed in [11,22,33]:
            launch(f'subset_reference_mlp_ce_s{seed}',config(seed=seed,protocol='canonical',partition='dirichlet',train_size=6396))
    elif a.phase=='screen':
        for model,loss in [('mlp','ce'),('mlp','weighted'),('mlp','focal'),('cnn','ce'),('cnn','weighted'),('cnn','focal'),('cnn_rr','weighted'),('cnn_rr','focal')]:
            for seed in [11,22,33]:launch(f'main_{model}_{loss}_s{seed}',config(model,loss,seed))
        select(SCREEN_CANDIDATES)
    elif a.phase=='rr_extension':
        for loss in ['ce','weighted','focal']:
            for seed in [11,22,33]:launch(f'main_mlp_rr_{loss}_s{seed}',config('mlp_rr',loss,seed))
    elif a.phase=='spatial_extension':
        for loss in ['ce','weighted','focal']:
            for seed in [11,22,33]:launch(f'main_cnn_spatial_rr_{loss}_s{seed}',config('cnn_spatial_rr',loss,seed))
        select()
    elif a.phase=='balanced_extension':
        for model in ['mlp','mlp_rr']:
            for seed in [11,22,33]:launch(f'main_{model}_balanced_s{seed}',config(model,'balanced',seed))
    elif a.phase=='data_volume':
        for n in [6396,20000]:
            for seed in [11,22,33,44,55]:launch(f'volume_mlp_ce_{n}_s{seed}',config('mlp','ce',seed,train_size=n,test_size=0))
    elif a.phase=='confirm':
        s=json.loads((ROOT/'results/selection.json').read_text())
        for model,loss in [('mlp','ce'),(s['model'],s['loss'])]:
            for seed in [44,55]:launch(f'main_{model}_{loss}_s{seed}',config(model,loss,seed,save_rounds=True))
    elif a.phase=='robustness':
        s=json.loads((ROOT/'results/selection.json').read_text())
        scenarios=[('iid',dict(partition='iid')),('dir01',dict(partition='dirichlet',alpha=.1)),
                   ('dir05',dict(partition='dirichlet',alpha=.5)),('half',dict(participation=.5))]
        for tag,kw in scenarios:
            for model,loss in [('mlp','ce'),(s['model'],s['loss'])]:
                for seed in [11,22,33]:launch(f'robust_{tag}_{model}_{loss}_s{seed}',config(model,loss,seed,**kw))
    elif a.phase=='controls':
        for transport in ['none','fixed_L2','fixed_L3','adaptive']:
            launch(f'control_v2_{transport}',config(protocol='canonical',partition='dirichlet',train_size=6396,transport=transport))
    elif a.phase=='central':
        s=json.loads((ROOT/'results/selection.json').read_text())
        for seed in [11,22,33]:launch(f'central_{s["model"]}_{s["loss"]}_s{seed}',config(s['model'],s['loss'],seed,partition='central'))
