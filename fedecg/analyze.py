"""Aggregate actual run artifacts and calculate paired patient-cluster uncertainty."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .common import ROOT, save_json
from .data import CLASSES, load_split
from .train import metrics


def collect():
    rows=[]
    for p in sorted((ROOT/'results/runs').glob('*/metrics.json')):
        c=json.loads((p.parent/'config.json').read_text());m=json.loads(p.read_text())
        row={'run':p.parent.name,**c,'best_round':m['best_round'],'validation_f1':m['best_validation_macro_f1'],
             'n_train':m['n_train'],'n_test':m['n_test'],'n_clients_actual':m['n_clients'],
             'parameters':m['parameter_count'],'payload_bytes':m['raw_parameter_bytes'],'wall_seconds':m['training_and_eval_wall_seconds']}
        for endpoint in ['best','final']:
            for metric in ['accuracy','macro_f1','balanced_accuracy','auc_ovr_macro','average_precision_macro']:
                row[endpoint+'_'+metric]=m[endpoint][metric]
            for cls in CLASSES:
                row[endpoint+'_recall_'+cls]=m[endpoint]['per_class'][cls]['recall']
        rows.append(row)
    frame=pd.DataFrame(rows);frame.to_csv(ROOT/'results/run_summary.csv',index=False)
    return frame


def f1_from_cm(cm):
    tp=np.diagonal(cm,axis1=-2,axis2=-1)
    denom=cm.sum(axis=-2)+cm.sum(axis=-1)
    return np.divide(2*tp,denom,out=np.zeros_like(tp,dtype=float),where=denom>0).mean(axis=-1)


def paired_bootstrap(baseline,enhanced,repeats=5000,output='primary_uncertainty.json'):
    seeds=[11,22,33,44,55]
    all_cms=[]
    for stem in [baseline,enhanced]:
        cms=[]
        for seed in seeds:
            d=np.load(ROOT/f'results/runs/{stem}_s{seed}/predictions.npz')
            y=d['y'];pred=d['probs_best'].argmax(1)
            cms.append(np.stack([np.bincount(y[d['record']==r]*5+pred[d['record']==r],minlength=25).reshape(5,5)
                                 for r in sorted(set(d['record']))]))
        all_cms.append(np.stack(cms))
    arr=np.stack(all_cms) # methods x seeds x patients x 5 x 5
    rng=np.random.default_rng(20261028);effects=[]
    n_patients=arr.shape[2]
    for _ in range(repeats):
        si=rng.integers(0,len(seeds),len(seeds));pi=rng.integers(0,n_patients,n_patients)
        sampled=arr[:,si][:,:,pi].sum(axis=2)
        score=f1_from_cm(sampled).mean(axis=1)
        effects.append(float(score[1]-score[0]))
    full=f1_from_cm(arr.sum(axis=2))
    differences=full[1]-full[0]
    t=stats.ttest_rel(full[1],full[0])
    save_json(ROOT/'results'/output,{
        'baseline':baseline,'enhancement':enhanced,'seeds':seeds,'test_patients':n_patients,
        'observed_macro_f1_difference':float(differences.mean()),
        'paired_seed_differences':differences.tolist(),'hierarchical_bootstrap_95ci':np.quantile(effects,[.025,.975]).tolist(),
        'bootstrap_repeats':repeats,'paired_t_p':float(t.pvalue),
        'interpretation':'Seeds and test patients resampled with replacement, paired across methods; F1 averaged over fixed five classes. Rare Q class has only seven test beats; selection uncertainty is not included.'})


def baseline_metrics():
    d,tr,va,te=load_split('strict')
    majority=int(np.bincount(d['y'][tr],minlength=5).argmax())
    probs=np.zeros((len(te),5));probs[:,majority]=1
    save_json(ROOT/'results/majority_baseline.json',metrics(d['y'][te],probs))
    concentration=[]
    for split,ix in [('train',tr),('validation',va),('test',te)]:
        for k,c in enumerate(CLASSES):
            chosen=ix[d['y'][ix]==k]
            ids,counts=np.unique(d['record'][chosen],return_counts=True)
            concentration.append({'split':split,'class':c,'support':len(chosen),'records_with_class':len(ids),
                'largest_record':int(ids[counts.argmax()]) if len(ids) else None,
                'largest_record_count':int(counts.max()) if len(counts) else 0,
                'largest_record_fraction':float(counts.max()/counts.sum()) if len(counts) else 0.})
    pd.DataFrame(concentration).to_csv(ROOT/'results/class_patient_concentration.csv',index=False)


def patient_diagnostics(stems):
    rows=[]
    for stem in stems:
        for seed in [11,22,33,44,55]:
            p=ROOT/f'results/runs/{stem}_s{seed}/predictions.npz'
            if not p.exists():continue
            d=np.load(p)
            for record in sorted(set(d['record'])):
                mask=d['record']==record;m=metrics(d['y'][mask],d['probs_best'][mask])
                present=[x['f1'] for x in m['per_class'].values() if x['support']>0]
                row={'model':stem,'seed':seed,'record':int(record),'beats':int(mask.sum()),
                     'accuracy':m['accuracy'],'macro_f1_present_classes':float(np.mean(present)),
                     'balanced_accuracy_present_classes':m['balanced_accuracy']}
                for c in CLASSES:
                    row[c+'_support']=m['per_class'][c]['support'];row[c+'_recall']=m['per_class'][c]['recall']
                rows.append(row)
    pd.DataFrame(rows).to_csv(ROOT/'results/patient_diagnostics.csv',index=False)


def main():
    frame=collect();baseline_metrics()
    s_path=ROOT/'results/selection.json'
    if s_path.exists():
        s=json.loads(s_path.read_text());stem=f'main_{s["model"]}_{s["loss"]}'
        if all((ROOT/f'results/runs/{stem}_s{i}/metrics.json').exists() for i in [11,22,33,44,55]):
            paired_bootstrap('main_mlp_ce',stem)
            patient_diagnostics(['main_mlp_ce',stem])
    if all((ROOT/f'results/runs/volume_mlp_ce_6396_s{i}/metrics.json').exists() for i in [11,22,33,44,55]):
        paired_bootstrap('volume_mlp_ce_6396','main_mlp_ce',output='data_volume_uncertainty.json')
    print(frame[['run','validation_f1','best_macro_f1','final_macro_f1']].to_string(index=False))


if __name__=='__main__':main()
