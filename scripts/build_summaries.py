"""Derive reporting tables from existing runs without fitting or selecting models."""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fedecg.data import DS1, load_split, stratified_subset

OUT = ROOT / 'results/benchmark'


def main():
    OUT.mkdir(exist_ok=True)
    summary = pd.read_csv(ROOT / 'results/run_summary.csv')
    screen = summary[summary.run.str.startswith('main_') & summary.seed.isin([11,22,33])].copy()
    assert len(screen) == 48
    screen['selected_checkpoint_steps'] = [
        int(pd.read_csv(ROOT/'results/runs'/r.run/'client_trace.csv').query('round <= @r.best_round').steps.sum())
        for r in screen.itertuples()]
    screen.to_csv(OUT/'screen_common_seeds_runs.csv', index=False)
    grid = screen.groupby(['model','loss']).agg(
        val_mean=('validation_f1','mean'), val_sd=('validation_f1','std'),
        test_mean=('best_macro_f1','mean'), test_sd=('best_macro_f1','std'),
        round_min=('best_round','min'), round_max=('best_round','max'),
        selected_steps_min=('selected_checkpoint_steps','min'), selected_steps_max=('selected_checkpoint_steps','max'))
    grid = grid.sort_values('val_mean', ascending=False).reset_index()
    assert len(grid) == 16 and not grid.val_mean.duplicated().any()
    grid.insert(0,'validation_rank',np.arange(1,17))
    grid.to_csv(OUT/'complete_screen_common_seeds.csv', index=False)
    baseline = screen[(screen.model=='mlp') & (screen.loss=='ce')].set_index('seed')
    paired=[]
    for (model,loss), g in screen.groupby(['model','loss']):
        delta = g.set_index('seed').best_macro_f1 - baseline.best_macro_f1
        paired.append({'model':model,'loss':loss,'n_common_seeds':3,
                       'mean_difference_from_mlp_ce':delta.mean(),'seed_sd_difference':delta.std(ddof=1),
                       **{f'difference_seed{s}':delta.loc[s] for s in [11,22,33]}})
    pd.DataFrame(paired).to_csv(OUT/'common_seed_paired_differences.csv',index=False)
    data = dict(np.load(ROOT/'data/processed/beats.npz'))
    spec = json.loads((ROOT/'configs/validation_controls.json').read_text())
    strict_ds1 = [r for r in DS1 if r!=201]
    supports=[]
    for fold, held in enumerate(spec['grouped_validation']['folds']):
        training = sorted(set(strict_ds1)-set(held))
        for k,name in enumerate('NSVFQ'):
            row={'fold':fold,'class':name,'training_records':';'.join(map(str,training)),
                 'held_records':';'.join(map(str,held))}
            for label, records in [('training',training),('held',held)]:
                keep=np.isin(data['record'],records) & (data['y']==k)
                row[f'{label}_beats']=int(keep.sum())
                row[f'{label}_patients_with_class']=int(len(set(data['record'][keep])))
            supports.append(row)
    pd.DataFrame(supports).to_csv(OUT/'grouped_fold_class_support.csv',index=False)
    _, tr, va, te = load_split('strict')
    main_support=[]
    for label, ix in [('training',tr),('validation',va),('test',te)]:
        for k,name in enumerate('NSVFQ'):
            for record in sorted(set(data['record'][ix])):
                main_support.append({'split':label,'record':int(record),'class':name,
                    'beats':int(((data['record'][ix]==record)&(data['y'][ix]==k)).sum())})
    pd.DataFrame(main_support).to_csv(OUT/'main_patient_class_support.csv',index=False)
    subset_rows=[]
    for n in [6396,20000,40142]:
        ix=tr if n==40142 else stratified_subset(tr,data['y'],n,20261028)
        for record in sorted(set(data['record'][tr])):
            ids=ix[data['record'][ix]==record]
            subset_rows.append({'train_size':n,'subset_seed':20261028,'record':int(record),'beats':len(ids),
                **{name:int((data['y'][ids]==k).sum()) for k,name in enumerate('NSVFQ')}})
    pd.DataFrame(subset_rows).to_csv(OUT/'training_subset_patient_allocation.csv',index=False)
    budget=[]
    for r in summary.itertuples():
        path=ROOT/'results/runs'/r.run
        trace=pd.read_csv(path/'client_trace.csv')
        budget.append({'run':r.run,'stage':'initial','model':r.model,'loss':r.loss,'seed':int(r.seed),
                       'n_train':int(r.n_train),'rounds':int(r.rounds),'best_round':int(r.best_round),
                       'full_run_steps':int(trace.steps.sum()),
                       'full_run_sample_exposures':int(trace.n.sum()*r.local_epochs),
                       'selected_checkpoint_steps':int(trace[trace['round']<=r.best_round].steps.sum()),
                       'metrics_path':str((path/'metrics.json').relative_to(ROOT))})
    for stage in ['grouped_cv','equal_steps','subset_stability']:
        for p in sorted((ROOT/'results/validation_controls'/stage).glob('*/metrics.json')):
            m=json.loads(p.read_text());cfg=json.loads((p.parent/'config.json').read_text())
            trace=pd.read_csv(p.parent/'client_trace.csv')
            full_steps=int(trace.steps.sum());exposures=int(trace.sample_exposures.sum())
            assert full_steps==m['total_local_steps'] and exposures==m['total_sample_exposures']
            budget.append({'run':p.parent.name,'stage':stage,'model':cfg.get('model','mlp'),
                           'loss':cfg.get('loss','ce'),'seed':cfg.get('seed'),
                           'n_train':m.get('n_train',cfg.get('train_size')),'rounds':30,
                           'best_round':m.get('best_round'),
                           'full_run_steps':full_steps,'full_run_sample_exposures':exposures,
                           'selected_checkpoint_steps':int(trace[trace['round']<=m['best_round']].steps.sum()),
                           'metrics_path':str(p.relative_to(ROOT))})
    for p in sorted((OUT/'integration').glob('*/metrics.json')):
        m=json.loads(p.read_text())
        budget.append({'run':p.parent.name,'stage':'integration_timing','model':m['model'],'loss':m['loss'],
                       'seed':m['seed'],'n_train':40142,'rounds':30,'full_run_steps':m['local_steps'],
                       'full_run_sample_exposures':40142*30,
                       'transport':m['policy'],'timing_repetition':m['repetition'],
                       'metrics_path':str(p.relative_to(ROOT))})
    pd.DataFrame(budget).to_csv(OUT/'complete_run_ledger.csv',index=False)
    (OUT/'supplement_manifest.json').write_text(json.dumps({
        'run_counts':dict(Counter(r['stage'] for r in budget)),
        'recipe_tie_occurred':False,'common_seeds':[11,22,33],
        'class_order':list('NSVFQ'),'source':'Saved predictions and metrics; no new predictive training or selection',
        'interpretation':'The common-seed test comparison is descriptive after test inspection; model selection used mean peak validation macro F1.'},indent=2))
    print('Reporting supplement:',len(budget),'executions;',len(grid),'recipes')


if __name__=='__main__': main()
