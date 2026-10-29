"""Replay six frozen CPU learners against the versioned numerical references."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fedecg.train import run


def main():
    checks=[]
    for model,loss in [('mlp','ce'),('mlp_rr','balanced')]:
        for seed in [11,22,33]:
            label=f'main_{model}_{loss}_s{seed}'
            ref=ROOT/'reference/artifacts/results/runs'/label
            cfg=json.loads((ref/'config.json').read_text());expected=json.loads((ref/'metrics.json').read_text())
            assert cfg['device']=='cpu'
            out=ROOT/'results/portability'/label;run(cfg,out);actual=json.loads((out/'metrics.json').read_text())
            assert actual['final_state_hash']==expected['final_state_hash'],label
            assert actual['best_state_hash']==expected['best_state_hash'],label
            assert actual['best']['macro_f1']==expected['best']['macro_f1'],label
            checks.append({'run':label,'best_state_hash':actual['best_state_hash'],'final_state_hash':actual['final_state_hash'],
                           'best_macro_f1':actual['best']['macro_f1'],'bitwise_checkpoint_reproduction':True})
    target=ROOT/'reference/portability_checks.json';target.write_text(json.dumps({'checks':checks,'scope':'Six full 30-round CPU learners replayed using the portable namespace and pinned environment.'},indent=2)+'\n')
    print('Six frozen learners reproduced exactly')


if __name__=='__main__':main()
