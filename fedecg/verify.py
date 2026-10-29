"""Recompute saved endpoints and verify completed experiment artifacts."""
import json
import numpy as np
import torch
from pathlib import Path

from .common import ROOT,save_json,sha256
from .data import load_split,patient_id
from .train import metrics,state_hash,predict
from .models import make_model


def main():
    verified=[];forward_checks=[]
    d=dict(np.load(ROOT/'data/processed/beats.npz'))
    x=torch.from_numpy(d['x'][:,None]);rr=torch.from_numpy(d['rr']);torch.set_num_threads(1)
    for folder in sorted((ROOT/'results/runs').iterdir()):
        if not (folder/'metrics.json').exists():continue
        cfg=json.loads((folder/'config.json').read_text());m=json.loads((folder/'metrics.json').read_text())
        pred=np.load(folder/'predictions.npz')
        assert np.array_equal(pred['y'],d['y'][pred['indices']]),folder.name
        assert np.array_equal(pred['record'],d['record'][pred['indices']]),folder.name
        assert np.array_equal(pred['peak'],d['peak'][pred['indices']]),folder.name
        for endpoint in ['best','final']:
            actual=metrics(pred['y'],pred['probs_'+endpoint])
            assert abs(actual['macro_f1']-m[endpoint]['macro_f1'])<1e-12,folder.name
            assert actual['confusion_matrix']==m[endpoint]['confusion_matrix'],folder.name
            state=torch.load(folder/f'checkpoint_{endpoint}.pt',weights_only=True,map_location='cpu')
            assert state_hash(state)==m[endpoint+'_state_hash'],folder.name
        manifest=json.loads((folder/'split_manifest.json').read_text())
        assert np.array_equal(pred['indices'],np.array(manifest['test'])),folder.name
        assert np.array_equal(pred['val_y'],d['y'][manifest['validation']]),folder.name
        assert np.array_equal(pred['val_record'],d['record'][manifest['validation']]),folder.name
        for split in ['train','validation','test']:
            assert np.bincount(d['y'][manifest[split]],minlength=5).tolist()==manifest['class_counts'][split],folder.name
        net=make_model(cfg['model']);net.load_state_dict(torch.load(folder/'checkpoint_best.pt',weights_only=True,map_location='cpu'))
        sample=np.unique(np.concatenate([np.linspace(0,len(pred['y'])-1,min(128,len(pred['y'])),dtype=int),
                   *[np.flatnonzero(pred['y']==k)[:4] for k in range(5)]]))
        regenerated=predict(net,x,rr,pred['indices'][sample],'cpu')
        difference=float(np.abs(regenerated-pred['probs_best'][sample]).max())
        assert np.allclose(regenerated,pred['probs_best'][sample],atol=2e-5,rtol=2e-4),(folder.name,difference)
        forward_checks.append({'run':folder.name,'sampled_predictions':len(sample),'max_absolute_probability_difference':difference,'device':'CPU replay; cross-device tolerance stated'})
        sets=[set(manifest[k]) for k in ['train','validation','test']]
        assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]),folder.name
        ids=[i for c in manifest['clients'] for i in c['indices']]
        assert len(set(ids))==len(ids) and set(ids)==sets[0],folder.name
        if cfg['protocol']=='strict':
            patients=[{patient_id(r) for r in manifest['records'][k]} for k in ['train','validation','test']]
            assert not (patients[0]&patients[1] or patients[0]&patients[2] or patients[1]&patients[2]),folder.name
        verified.append(folder.name)
    controls=[p for p in verified if p.startswith('control_v2_')]
    assert len(controls)==4
    assert len({json.loads((ROOT/'results/runs'/p/'metrics.json').read_text())['final_state_hash'] for p in controls})==1
    save_json(ROOT/'results/artifact_verification.json',{'verified_runs':verified,'n_verified':len(verified),
        'forward_checks':forward_checks,
        'checks':['saved predictions reproduce reported F1 and confusion matrices','checkpoint hashes match metrics',
                  'saved indices/labels/records/peaks bind to processed data and manifests',
                  'checkpoint forward passes reproduce sampled predictions within atol 2e-5, rtol 2e-4',
                  'sample sets disjoint','clients cover training samples exactly once','strict protocol has no patient leakage',
                  'real protection controls have identical final model parameters']})
    print('Verified',len(verified),'completed runs')


if __name__=='__main__':main()
