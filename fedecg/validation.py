"""ML follow-ups specified after independent review, with original artifacts kept.

Grouped validation uses DS1 subjects only and a fixed final-round endpoint.
The matched-compute experiment varies unique samples with equal local steps,
batch sizes and reference client weights. These are exploratory follow-ups;
previous DS2 examination is explicitly not erased by their execution.
"""
import argparse
import csv
import json
import math
import time
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .common import ROOT,save_json,sha256
from .identity import current_identity,ensure_identity
from .data import DS1,DS2,build_clients,load_split,patient_id,stratified_subset
from .models import make_model,class_weights,loss_fn
from .train import metrics,predict,state_hash

OUT=ROOT/'results/validation_controls'
SPEC=json.loads((ROOT/'configs/validation_controls.json').read_text())
FOLDS=SPEC['grouped_validation']['folds']


def _train(d,tr,va,cfg,out,reference=None,evaluate_test=None):
    out=Path(out)
    ensure_identity(out,current_identity(cfg,extra_sources=('fedecg/validation.py','configs/validation_controls.json')),allow_legacy=os.getenv('FEDECG_ALLOW_LEGACY_REUSE')=='1')
    if (out/'metrics.json').exists():
        assert json.loads((out/'config.json').read_text())==cfg
        return
    out.mkdir(parents=True,exist_ok=True);save_json(out/'config.json',cfg)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(cfg['seed'])
    clients=build_clients(d,tr,'patient',cfg['seed'])
    if reference is None:reference=clients
    assert len(reference)==len(clients)
    x=torch.from_numpy(d['x'][:,None]);rr=torch.from_numpy(d['rr']);y=torch.from_numpy(d['y'])
    weights=class_weights(y[tr],cfg['loss'])
    model=make_model(cfg['model']);local=make_model(cfg['model']);initial_hash=state_hash(model.state_dict())
    reference_sizes=[len(ix) for ix in reference]
    total=sum(reference_sizes);best=-1.;best_state=None;best_round=0;round_rows=[];traces=[]
    start_all=time.perf_counter()
    for round_idx in range(1,31):
        original={k:v.detach().clone() for k,v in model.state_dict().items()}
        accum={k:torch.zeros_like(v) for k,v in original.items()}
        for ci,ix in enumerate(clients):
            local.load_state_dict(original);local.train()
            opt=torch.optim.AdamW(local.parameters(),lr=.001,weight_decay=.0001)
            rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],round_idx,ci,0]))
            if cfg['sampling']=='matched_replacement':
                steps=math.ceil(reference_sizes[ci]/128)
                batches=[rng.choice(ix,128,replace=True) for _ in range(steps)]
            else:
                order=rng.permutation(ix);batches=[order[j:j+128] for j in range(0,len(order),128)]
            exposures=0
            for batch in batches:
                ids=torch.as_tensor(batch,dtype=torch.long)
                opt.zero_grad(set_to_none=True);loss=loss_fn(local(x[ids],rr[ids]),y[ids],cfg['loss'],weights)
                loss.backward();torch.nn.utils.clip_grad_norm_(local.parameters(),5.);opt.step();exposures+=len(batch)
            for k,v in local.state_dict().items():accum[k].add_(v,alpha=reference_sizes[ci]/total)
            traces.append({'round':round_idx,'client':ci,'record':int(d['record'][ix[0]]),'available_unique_beats':len(ix),
                           'reference_weight_beats':reference_sizes[ci],'steps':len(batches),'sample_exposures':exposures})
        model.load_state_dict(accum)
        # Validation observation does not adapt any training setting or round count.
        vp=predict(model,x,rr,va,'cpu');vm=metrics(d['y'][va],vp)
        round_rows.append({'round':round_idx,'val_macro_f1':vm['macro_f1'],'val_accuracy':vm['accuracy']})
        if vm['macro_f1']>best:
            best=vm['macro_f1'];best_round=round_idx;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        if round_idx in [1,10,20,30]:print(out.name,'round',round_idx,'val-F1',round(vm['macro_f1'],4),flush=True)
    final_state={k:v.detach().clone() for k,v in model.state_dict().items()}
    final_val=predict(model,x,rr,va,'cpu')
    preds={'val_y':d['y'][va],'val_record':d['record'][va],'val_indices':va,'val_probs_final':final_val}
    result={'validation_final':metrics(d['y'][va],final_val),'best_validation_macro_f1':best,'best_round':best_round,
            'n_train':len(tr),'n_validation':len(va),'n_clients':len(clients),'initial_state_hash':initial_hash,
            'final_state_hash':state_hash(final_state),'total_local_steps':sum(r['steps'] for r in traces),
            'total_sample_exposures':sum(r['sample_exposures'] for r in traces),'wall_seconds':time.perf_counter()-start_all,
            'selection_scope':'Exploratory, review-driven; earlier DS2 reuse disclosed. Fixed round 30 is primary.'}
    if evaluate_test is not None:
        te=evaluate_test
        p=predict(model,x,rr,te,'cpu');result['test_final']=metrics(d['y'][te],p)
        preds.update(y=d['y'][te],record=d['record'][te],indices=te,probs_final=p)
        model.load_state_dict(best_state)
        pb=predict(model,x,rr,te,'cpu');result['test_best']=metrics(d['y'][te],pb);preds['probs_best']=pb
        result['test_used_for_selection']=False
    else:
        result['DS2_evaluated']=False
    np.savez_compressed(out/'predictions.npz',**preds)
    torch.save(final_state,out/'checkpoint_final.pt')
    torch.save(best_state,out/'checkpoint_best.pt')
    save_json(out/'split_manifest.json',{'train_indices':tr.tolist(),'validation_indices':va.tolist(),
        'index_space':'strict DS1 filtered array' if 'fold' in cfg else 'global data/processed/beats.npz array',
        'global_index_lookup':'results/validation_controls/grouped_cv_global_indices.npy' if 'fold' in cfg else None,
        'train_records':sorted(map(int,set(d['record'][tr]))),'validation_records':sorted(map(int,set(d['record'][va]))),
        'class_counts':{'train':np.bincount(d['y'][tr],minlength=5).tolist(),'validation':np.bincount(d['y'][va],minlength=5).tolist()}})
    pd.DataFrame(round_rows).to_csv(out/'rounds.csv',index=False);pd.DataFrame(traces).to_csv(out/'client_trace.csv',index=False)
    save_json(out/'metrics.json',result)
    print('DONE',out.name,flush=True)


def grouped():
    raw=dict(np.load(ROOT/'data/processed/beats.npz'))
    records=sorted(set(DS1)-{201});mask=np.isin(raw['record'],records)
    # The arrays passed to the learner contain DS1 only, so no DS2 prediction is possible here.
    d={k:v[mask].copy() for k,v in raw.items()}
    np.save(OUT/'grouped_cv_global_indices.npy',np.flatnonzero(mask))
    assert not (set(d['record'])&set(DS2))
    assert sorted(r for fold in FOLDS for r in fold)==records
    for model,loss in SPEC['grouped_validation']['models']:
        for seed in [11,22,33]:
            for fold,held in enumerate(FOLDS):
                va=np.flatnonzero(np.isin(d['record'],held));tr=np.flatnonzero(~np.isin(d['record'],held))
                cfg={'model':model,'loss':loss,'seed':seed,'fold':fold,'sampling':'one_epoch','rounds':30,
                     'endpoint':'fixed final-round internal validation; no DS2 evaluation'}
                _train(d,tr,va,cfg,OUT/f'grouped_cv/{model}_{loss}_s{seed}_f{fold}')
    rows=[];fold_rows=[]
    for model,loss in SPEC['grouped_validation']['models']:
        for seed in [11,22,33]:
            chunks=[]
            for fold in range(3):
                folder=OUT/f'grouped_cv/{model}_{loss}_s{seed}_f{fold}'
                p=np.load(folder/'predictions.npz');chunks.append(p)
                m=json.loads((folder/'metrics.json').read_text())
                fold_rows.append({'model':model,'loss':loss,'seed':seed,'fold':fold,
                                  'n_train':m['n_train'],'n_validation':m['n_validation'],
                                  'macro_f1':m['validation_final']['macro_f1'],
                                  'accuracy':m['validation_final']['accuracy']})
            y=np.concatenate([p['val_y'] for p in chunks]);probs=np.concatenate([p['val_probs_final'] for p in chunks])
            indices=np.concatenate([p['val_indices'] for p in chunks]);assert len(set(indices))==len(d['y'])
            m=metrics(y,probs)
            rows.append({'model':model,'loss':loss,'seed':seed,'oof_macro_f1':m['macro_f1'],
                         'oof_accuracy':m['accuracy'],'oof_balanced_accuracy':m['balanced_accuracy'],
                         **{'recall_'+c:v['recall'] for c,v in m['per_class'].items()}})
    pd.DataFrame(rows).to_csv(OUT/'grouped_cv_summary.csv',index=False)
    pd.DataFrame(fold_rows).to_csv(OUT/'grouped_cv_folds.csv',index=False)
    g=pd.DataFrame(rows).groupby(['model','loss']).oof_macro_f1.mean();winner=g.idxmax()
    save_json(OUT/'grouped_cv_selection.json',{'model':winner[0],'loss':winner[1],'scores':{f'{a}_{b}':float(v) for (a,b),v in g.items()},
        'endpoint':'Mean fixed-round pooled DS1 out-of-fold macro F1 over three seeds',
        'DS2_evaluated':False,'claim_limit':'Internal validation diagnostic, not an unbiased external-test or nested model-selection estimate'})


def equal_steps():
    d,full,va,te=load_split('strict')
    _,small,v2,t2=load_split('strict',6396,test_size=0)
    assert np.array_equal(va,v2) and np.array_equal(te,t2)
    reference=build_clients(d,full,'patient',11)
    for size,tr in [(6396,small),(40142,full)]:
        for seed in [11,22,33,44,55]:
            cfg={'model':'mlp','loss':'ce','seed':seed,'train_size':size,'sampling':'matched_replacement','rounds':30,
                 'aggregation_weights':'Fixed full-training reference counts per patient in both arms',
                 'primary_endpoint':'Fixed round 30 test macro F1'}
            _train(d,tr,va,cfg,OUT/f'equal_steps/n{size}_s{seed}',reference=reference,evaluate_test=te)
    rows=[]
    for p in sorted((OUT/'equal_steps').glob('*/metrics.json')):
        c=json.loads((p.parent/'config.json').read_text());m=json.loads(p.read_text())
        rows.append({'train_size':c['train_size'],'seed':c['seed'],'test_final_macro_f1':m['test_final']['macro_f1'],
                     'test_best_macro_f1':m['test_best']['macro_f1'],'test_final_accuracy':m['test_final']['accuracy'],
                     'local_steps':m['total_local_steps'],'sample_exposures':m['total_sample_exposures']})
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'equal_steps_summary.csv',index=False)
    assert frame.local_steps.nunique()==1 and frame.sample_exposures.nunique()==1
    _bootstrap_equal()


def _bootstrap_equal():
    from .analyze import f1_from_cm
    cms=[]
    for n in [6396,40142]:
        seeds=[]
        for seed in [11,22,33,44,55]:
            p=np.load(OUT/f'equal_steps/n{n}_s{seed}/predictions.npz');y=p['y'];pred=p['probs_final'].argmax(1)
            seeds.append(np.stack([np.bincount(y[p['record']==r]*5+pred[p['record']==r],minlength=25).reshape(5,5)
                                   for r in sorted(set(p['record']))]))
        cms.append(np.stack(seeds))
    arr=np.stack(cms);rng=np.random.default_rng(20261029);effects=[]
    for _ in range(5000):
        si=rng.integers(0,5,5);pi=rng.integers(0,22,22)
        score=f1_from_cm(arr[:,si][:,:,pi].sum(2)).mean(1);effects.append(float(score[1]-score[0]))
    full=f1_from_cm(arr.sum(2));diff=full[1]-full[0]
    save_json(OUT/'equal_steps_uncertainty.json',{'difference_full_minus_small':float(diff.mean()),'paired_seed_differences':diff.tolist(),
        'patient_seed_bootstrap_95ci':np.quantile(effects,[.025,.975]).tolist(),'n_bootstrap':5000,
        'endpoint':'Fixed round 30 macro F1; 22 patients and five seeds paired across arms',
        'claim_limit':'Exploratory follow-up after earlier DS2 examination; no independent test confirmation'})


def influence():
    rows=[]
    for seed in [11,22,33,44,55]:
        a=np.load(ROOT/f'results/runs/main_mlp_ce_s{seed}/predictions.npz')
        b=np.load(ROOT/f'results/runs/main_mlp_rr_balanced_s{seed}/predictions.npz')
        assert np.array_equal(a['indices'],b['indices'])
        for record in sorted(set(a['record'])):
            mask=a['record']!=record
            am=metrics(a['y'][mask],a['probs_best'][mask]);bm=metrics(b['y'][mask],b['probs_best'][mask])
            rows.append({'seed':seed,'omitted_record':int(record),'baseline_macro_f1':am['macro_f1'],
                         'candidate_macro_f1':bm['macro_f1'],'difference':bm['macro_f1']-am['macro_f1']})
    pd.DataFrame(rows).to_csv(OUT/'leave_one_patient_out_influence.csv',index=False)
    save_json(OUT/'influence_note.json',{'purpose':'Sensitivity diagnostic of all 22 test patients; no patient excluded from headline evaluation',
        'selection':'Does not change model selection, training or reported full-test endpoints'})


def subset_stability():
    d,full,va,te=load_split('strict');reference=build_clients(d,full,'patient',11)
    for subset_seed in [20261029,20261030]:
        small=stratified_subset(full,d['y'],6396,subset_seed)
        for seed in [11,22,33]:
            cfg={'model':'mlp','loss':'ce','seed':seed,'train_size':6396,'subset_seed':subset_seed,
                 'sampling':'matched_replacement','rounds':30,
                 'aggregation_weights':'Fixed full-training reference counts per patient in both arms',
                 'primary_endpoint':'Fixed round 30 test macro F1'}
            _train(d,small,va,cfg,OUT/f'subset_stability/subset{subset_seed}_s{seed}',reference=reference,evaluate_test=te)
    from .analyze import f1_from_cm
    rows=[];all_cms=[]
    for n in [6396,40142]:
        subsets=[]
        for subset_seed in [20261028,20261029,20261030]:
            seeds=[]
            for seed in [11,22,33]:
                folder=OUT/f'equal_steps/n{n}_s{seed}' if n==40142 or subset_seed==20261028 else OUT/f'subset_stability/subset{subset_seed}_s{seed}'
                p=np.load(folder/'predictions.npz');y=p['y'];pred=p['probs_final'].argmax(1)
                cm=np.stack([np.bincount(y[p['record']==r]*5+pred[p['record']==r],minlength=25).reshape(5,5) for r in sorted(set(p['record']))])
                seeds.append(cm)
                if n==6396:
                    m=json.loads((folder/'metrics.json').read_text());full_m=json.loads((OUT/f'equal_steps/n40142_s{seed}/metrics.json').read_text())
                    rows.append({'subset_seed':subset_seed,'training_seed':seed,'small_macro_f1':m['test_final']['macro_f1'],
                                 'full_macro_f1':full_m['test_final']['macro_f1'],
                                 'difference':full_m['test_final']['macro_f1']-m['test_final']['macro_f1']})
            subsets.append(np.stack(seeds))
        all_cms.append(np.stack(subsets))
    pd.DataFrame(rows).to_csv(OUT/'subset_stability_summary.csv',index=False)
    arr=np.stack(all_cms);rng=np.random.default_rng(20261030);effects=[]
    for _ in range(5000):
        ss=rng.integers(0,3,3);ts=rng.integers(0,3,3);ps=rng.integers(0,22,22)
        scores=f1_from_cm(arr[:,ss][:,:,ts][:,:,:,ps].sum(3)).mean(axis=(1,2))
        effects.append(float(scores[1]-scores[0]))
    save_json(OUT/'subset_stability_uncertainty.json',{'mean_difference_full_minus_small':float(np.mean([r['difference'] for r in rows])),
        'subset_training_patient_bootstrap_95ci':np.quantile(effects,[.025,.975]).tolist(),
        'subset_seeds':[20261028,20261029,20261030],'training_seeds':[11,22,33],
        'test_patients':22,'n_bootstrap':5000,'full_models_reused_across_subsets':True,
        'claim_limit':'Three crossed subset draws and training seeds, not nine independent training seeds; exploratory repeated DS2 use remains'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['grouped','equal_steps','influence','subset_stability']);a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    {'grouped':grouped,'equal_steps':equal_steps,'influence':influence,'subset_stability':subset_stability}[a.phase]()
