"""Independently recompute stored endpoints, uncertainty and transport preservation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fedecg.train import state_hash,predict
from fedecg.models import make_model


def f1(cm):
    tp=np.diagonal(cm,axis1=-2,axis2=-1);den=cm.sum(-1)+cm.sum(-2)
    return np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den>0).mean(-1)


def endpoint(y,prob,expected):
    pred=prob.argmax(1);cm=np.bincount(y*5+pred,minlength=25).reshape(5,5)
    assert cm.tolist()==expected['confusion_matrix']
    assert abs(float(f1(cm))-expected['macro_f1'])<1e-12
    assert abs(float((pred==y).mean())-expected['accuracy'])<1e-12


def patient_cms(path):
    with np.load(path) as a:
        return np.stack([np.bincount(a['y'][a['record']==r]*5+a['probs_best'][a['record']==r].argmax(1),minlength=25).reshape(5,5)
                         for r in sorted(set(a['record']))])


def final_cms(path):
    with np.load(path) as a:
        return np.stack([np.bincount(a['y'][a['record']==r]*5+a['probs_final'][a['record']==r].argmax(1),minlength=25).reshape(5,5)
                         for r in sorted(set(a['record']))])


def paired_interval(arr,seed,repeats=5000):
    rng=np.random.default_rng(seed);effects=[];nseed=arr.shape[1];npatient=arr.shape[2]
    for _ in range(repeats):
        si=rng.integers(0,nseed,nseed);pi=rng.integers(0,npatient,npatient)
        scores=f1(arr[:,si][:,:,pi].sum(2)).mean(1);effects.append(scores[1]-scores[0])
    scores=f1(arr.sum(2)).mean(1)
    return float(scores[1]-scores[0]),np.quantile(effects,[.025,.975])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-data',action='store_true');args=parser.parse_args()
    base=ROOT/'reference/artifacts';manifest=json.loads((base/'MANIFEST.json').read_text())
    for name,entry in manifest['files'].items():
        p=base/name
        assert p.stat().st_size==entry['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256'],name
    runs=base/'results/runs';validation=base/'results/validation_controls';integration=base/'results/benchmark/integration'
    torch.set_num_threads(1);data=None;forward=0
    if args.check_data:
        data=dict(np.load(ROOT/'data/processed/beats.npz'));x=torch.from_numpy(data['x'][:,None]);rr=torch.from_numpy(data['rr'])
    initial=list(runs.glob('*/metrics.json'));assert len(initial)==101
    for p in initial:
        cfg=json.loads((p.parent/'config.json').read_text());m=json.loads(p.read_text())
        with np.load(p.parent/'predictions.npz') as pred:
            for e in ['best','final']:
                endpoint(pred['y'],pred['probs_'+e],m[e])
                state=torch.load(p.parent/f'checkpoint_{e}.pt',map_location='cpu',weights_only=True)
                assert state_hash(state)==m[e+'_state_hash']
            if data is not None:
                assert np.array_equal(pred['y'],data['y'][pred['indices']])
                net=make_model(cfg['model']);net.load_state_dict(torch.load(p.parent/'checkpoint_best.pt',map_location='cpu',weights_only=True))
                sample=np.linspace(0,len(pred['y'])-1,min(128,len(pred['y'])),dtype=int)
                actual=predict(net,x,rr,pred['indices'][sample],'cpu')
                assert np.allclose(actual,pred['probs_best'][sample],atol=2e-5,rtol=2e-4),p.parent.name
                forward+=1
    followups=[p for stage in ['grouped_cv','equal_steps','subset_stability'] for p in (validation/stage).glob('*/metrics.json')]
    assert len(followups)==43
    for p in followups:
        m=json.loads(p.read_text());trace=pd.read_csv(p.parent/'client_trace.csv')
        assert int(trace.steps.sum())==m['total_local_steps']
        assert int(trace.sample_exposures.sum())==m['total_sample_exposures']
        assert state_hash(torch.load(p.parent/'checkpoint_final.pt',map_location='cpu',weights_only=True))==m['final_state_hash']
        with np.load(p.parent/'predictions.npz') as a:
            endpoint(a['val_y'],a['val_probs_final'],m['validation_final'])
            for e in ['best','final']:
                if 'test_'+e in m:endpoint(a['y'],a['probs_'+e],m['test_'+e])
    conditions=list(integration.glob('*/metrics.json'));assert len(conditions)==54
    hashes={};arrays={};records=[]
    for p in conditions:
        m=json.loads(p.read_text());messages=pd.read_csv(p.parent/'messages.csv');rounds=pd.read_csv(p.parent/'rounds.csv')
        assert len(messages)==1020 and len(rounds)==30 and messages[['round','client','direction']].drop_duplicates().shape[0]==1020
        assert int(messages.wire_component_bytes.sum())==m['wire_component_bytes']
        assert int(messages.raw_bytes.sum())==m['raw_bytes'] and int(messages.protection_extra_bytes.sum())==m['protection_extra_bytes']
        expected_extra=messages.level.map({0:0,1:5341,2:7574,3:10396})
        assert np.array_equal(messages.protection_extra_bytes,expected_extra)
        assert np.array_equal(messages.wire_component_bytes,messages.raw_bytes+expected_extra)
        assert int(rounds.local_steps.sum())==9630
        state=torch.load(p.parent/'checkpoint_final.pt',map_location='cpu',weights_only=True)
        original=torch.load(runs/f'main_{m["model"]}_{m["loss"]}_s{m["seed"]}/checkpoint_final.pt',map_location='cpu',weights_only=True)
        assert all(torch.equal(state[k],original[k]) for k in state)
        key=(m['model'],m['seed']);current=rounds.parameter_hash.tolist()
        if key in hashes:assert current==hashes[key]
        hashes[key]=current
        if m['repetition']==0:
            values=np.load(p.parent/'round_states.npz')['values']
            if key in arrays:assert np.array_equal(values,arrays[key])
            arrays[key]=values
            for row,want in zip(values,[m['initial_state_hash'],*current]):
                pos=0;reconstructed={}
                for k in sorted(state):
                    n=state[k].numel();reconstructed[k]=torch.from_numpy(row[pos:pos+n].reshape(state[k].shape).copy());pos+=n
                assert pos==len(row) and state_hash(reconstructed)==want
        records.append(m)
    table=pd.DataFrame(records).groupby(['model','loss','seed','policy'])[['elapsed_rounds_seconds','wire_component_bytes','protection_extra_bytes','peak_worker_rss_bytes']].mean()
    observed=table.groupby(['model','loss','policy']).agg(['mean','std']);saved=pd.read_csv(base/'results/benchmark/integration_summary.csv').set_index(['model','loss','policy'])
    for column in observed:
        assert np.allclose(observed[column].sort_index(),saved['_'.join(column)].sort_index(),atol=1e-7,rtol=1e-12)
    arr=np.stack([np.stack([patient_cms(runs/f'{stem}_s{s}/predictions.npz') for s in [11,22,33,44,55]]) for stem in ['main_mlp_ce','main_mlp_rr_balanced']])
    difference,ci=paired_interval(arr,20261028);expected=json.loads((base/'results/primary_uncertainty.json').read_text())
    assert abs(difference-expected['observed_macro_f1_difference'])<1e-12 and np.allclose(ci,expected['hierarchical_bootstrap_95ci'],atol=1e-12,rtol=0)
    eq=np.stack([np.stack([final_cms(validation/f'equal_steps/n{n}_s{s}/predictions.npz') for s in [11,22,33,44,55]]) for n in [6396,40142]])
    eq_difference,eq_ci=paired_interval(eq,20261029);expected=json.loads((validation/'equal_steps_uncertainty.json').read_text())
    assert abs(eq_difference-expected['difference_full_minus_small'])<1e-12 and np.allclose(eq_ci,expected['patient_seed_bootstrap_95ci'],atol=1e-12,rtol=0)
    pooled=pd.read_csv(validation/'grouped_cv_summary.csv')
    for row in pooled.to_dict('records'):
        cms=[json.loads((validation/f'grouped_cv/{row["model"]}_{row["loss"]}_s{row["seed"]}_f{f}/metrics.json').read_text())['validation_final']['confusion_matrix'] for f in range(3)]
        assert abs(float(f1(np.array(cms).sum(0)))-row['oof_macro_f1'])<1e-12
    crossed=[]
    for n in [6396,40142]:
        draws=[]
        for subset in [20261028,20261029,20261030]:
            paths=[validation/f'equal_steps/n{n}_s{s}/predictions.npz' if n==40142 or subset==20261028 else validation/f'subset_stability/subset{subset}_s{s}/predictions.npz' for s in [11,22,33]]
            draws.append(np.stack([final_cms(p) for p in paths]))
        crossed.append(np.stack(draws))
    crossed=np.stack(crossed);rng=np.random.default_rng(20261030);effects=[]
    for _ in range(5000):
        ss=rng.integers(0,3,3);ts=rng.integers(0,3,3);ps=rng.integers(0,22,22)
        scores=f1(crossed[:,ss][:,:,ts][:,:,:,ps].sum(3)).mean((1,2));effects.append(float(scores[1]-scores[0]))
    crossed_ci=np.quantile(effects,[.025,.975]);expected=json.loads((validation/'subset_stability_uncertainty.json').read_text())
    assert np.allclose(crossed_ci,expected['subset_training_patient_bootstrap_95ci'],atol=1e-12,rtol=0)
    summary={'manifest_files':len(manifest['files']),'initial_runs':101,'validation_runs':43,'integration_executions':54,
      'round_state_arrays_checked':18,'all_round_parameters_preserved':True,'checkpoint_forward_checks':forward,
      'main_effect':difference,'main_patient_seed_interval':ci.tolist(),'matched_steps_effect':eq_difference,'matched_steps_interval':eq_ci.tolist(),
      'grouped_pooled_endpoints_checked':len(pooled),'crossed_subset_interval':crossed_ci.tolist()}
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
