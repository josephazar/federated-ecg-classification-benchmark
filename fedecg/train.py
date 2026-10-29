"""Paired FedAvg training with validation-only checkpoint selection."""
import argparse
import csv
import json
import math
import platform
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from torch.nn import functional as F

from .common import ROOT, save_json, sha256
from .data import CLASSES, build_clients, load_split, patient_id
from .models import class_weights, loss_fn, make_model
from .identity import current_identity,ensure_identity
import os


def metrics(y,probs):
    pred=probs.argmax(1)
    precision,recall,f1,support=precision_recall_fscore_support(y,pred,labels=range(5),zero_division=0)
    aucs=[]; aps=[]
    for k in range(5):
        target=y==k
        if target.any() and (~target).any():
            aucs.append(roc_auc_score(target,probs[:,k]))
            aps.append(average_precision_score(target,probs[:,k]))
    return {'accuracy':float(accuracy_score(y,pred)), 'macro_f1':float(f1.mean()),
            'balanced_accuracy':float(recall[support>0].mean()), 'macro_precision':float(precision.mean()),
            'macro_recall':float(recall.mean()), 'auc_ovr_macro':float(np.mean(aucs)) if aucs else None,
            'average_precision_macro':float(np.mean(aps)) if aps else None,
            'per_class':{c:{'precision':float(p),'recall':float(r),'f1':float(f),'support':int(s)}
                         for c,p,r,f,s in zip(CLASSES,precision,recall,f1,support)},
            'confusion_matrix':confusion_matrix(y,pred,labels=range(5)).tolist()}


@torch.no_grad()
def predict(model, x, rr, indices, device, batch=2048):
    model.eval();outputs=[]
    for start in range(0,len(indices),batch):
        ix=torch.as_tensor(indices[start:start+batch],dtype=torch.long,device=device)
        outputs.append(model(x[ix],rr[ix]).softmax(1).cpu().numpy())
    return np.concatenate(outputs)


def state_hash(state):
    import hashlib
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run(cfg, out):
    out=Path(out)
    ensure_identity(out,current_identity(cfg),allow_legacy=os.getenv('FEDECG_ALLOW_LEGACY_REUSE')=='1')
    if (out/'metrics.json').exists():
        previous=json.loads((out/'config.json').read_text())
        if previous!=cfg: raise ValueError(f'Configuration mismatch at {out}')
        print('Already complete:',out,flush=True);return
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=cfg:
        raise ValueError(f'Refusing to mix different configurations at {out}')
    out.mkdir(parents=True,exist_ok=True)
    save_json(out/'config.json',cfg)
    seed=cfg['seed'];device=cfg.get('device','cpu')
    torch.set_num_threads(cfg.get('threads',1))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    d,tr,va,te=load_split(cfg.get('protocol','strict'),cfg.get('train_size',0),cfg.get('test_size'))
    clients=build_clients(d,tr,cfg.get('partition','patient'),seed,cfg.get('n_clients',10),cfg.get('alpha',.5))
    save_json(out/'split_manifest.json',{'protocol':cfg.get('protocol','strict'),
        'train':tr.tolist(),'validation':va.tolist(),'test':te.tolist(),
        'class_counts':{s:np.bincount(d['y'][ix],minlength=5).tolist() for s,ix in [('train',tr),('validation',va),('test',te)]},
        'records':{s:sorted(map(int,set(d['record'][ix]))) for s,ix in [('train',tr),('validation',va),('test',te)]},
        'clients':[{'id':i,'indices':ix.tolist(),'records':sorted(map(int,set(d['record'][ix]))),
                    'class_counts':np.bincount(d['y'][ix],minlength=5).tolist()} for i,ix in enumerate(clients)]})
    x=torch.from_numpy(d['x'][:,None]).to(device)
    rr=torch.from_numpy(d['rr']).to(device)
    y=torch.from_numpy(d['y']).to(device)
    weights=class_weights(torch.from_numpy(d['y'][tr]),cfg['loss']).to(device)
    global_model=make_model(cfg['model']).to(device)
    local_model=make_model(cfg['model']).to(device)
    initial_hash=state_hash(global_model.state_dict())
    scheduler=np.random.default_rng(seed+10000)
    context_rng=np.random.default_rng(seed+20000)
    schedules=[sorted(scheduler.choice(len(clients),max(1,math.ceil(len(clients)*cfg.get('participation',1.))),replace=False).tolist())
               for _ in range(cfg['rounds'])]
    save_json(out/'schedule.json',schedules)
    best_score=-1.;best_round=-1;best_state=None
    trace=[];timing=[]
    start_all=time.perf_counter()
    for round_idx,selected in enumerate(schedules,1):
        start=time.perf_counter()
        original={k:v.detach().clone() for k,v in global_model.state_dict().items()}
        total=sum(len(clients[i]) for i in selected)
        accum={k:torch.zeros_like(v) for k,v in original.items()}
        local_losses=[]
        for ci in selected:
            local_start=time.perf_counter()
            local_model.load_state_dict(original);local_model.train()
            opt=torch.optim.AdamW(local_model.parameters(),lr=cfg['lr'],weight_decay=1e-4)
            ix=clients[ci]
            loss_sum=0.;steps=0
            # Scheduling and data-order RNG never depend on protection or loss.
            for epoch in range(cfg.get('local_epochs',1)):
                order=np.random.default_rng(np.random.SeedSequence([seed,round_idx,ci,epoch])).permutation(ix)
                for b in range(0,len(order),cfg['batch_size']):
                    ids=torch.as_tensor(order[b:b+cfg['batch_size']],device=device,dtype=torch.long)
                    opt.zero_grad(set_to_none=True)
                    loss=loss_fn(local_model(x[ids],rr[ids]),y[ids],cfg['loss'],weights)
                    if cfg.get('prox_mu',0):
                        loss=loss+cfg['prox_mu']/2*sum((p-original[n]).square().sum() for n,p in local_model.named_parameters())
                    loss.backward();torch.nn.utils.clip_grad_norm_(local_model.parameters(),5.);opt.step()
                    loss_sum+=float(loss.detach().cpu());steps+=1
            if device=='mps':torch.mps.synchronize()
            elapsed=time.perf_counter()-local_start
            local_state=local_model.state_dict()
            payload=b''.join(v.detach().cpu().numpy().tobytes() for v in local_state.values())
            required=int(context_rng.choice([1,2,3],p=[.244,.712,.044]))
            transport=cfg.get('transport','none')
            extra=0;crypto_ms=0.;level=0
            if transport!='none':
                from .crypto import roundtrip
                level=required if transport=='adaptive' else int(transport[-1])
                restored,cm=roundtrip(payload,level)
                # Equality is required before accepting the update into FedAvg.
                assert restored==payload
                extra=cm['extra_bytes'];crypto_ms=cm['crypto_ms']
                offset=0;decoded={}
                for k,v in local_state.items():
                    size=v.numel()*v.element_size()
                    arr=np.frombuffer(restored[offset:offset+size],dtype=np.float32).copy().reshape(tuple(v.shape))
                    decoded[k]=torch.from_numpy(arr).to(device)
                    offset+=size
                local_state=decoded
            timing.append({'round':round_idx,'client':ci,'n':len(ix),'steps':steps,'training_seconds':elapsed,
                           'payload_bytes':len(payload),'extra_bytes':extra,'crypto_ms':crypto_ms,
                           'level':level,'required_level':required,'transport':transport})
            for k,v in local_state.items():
                accum[k].add_(v,alpha=len(ix)/total)
            local_losses.append(loss_sum/steps)
        global_model.load_state_dict(accum)
        probs=predict(global_model,x,rr,va,device)
        vm=metrics(d['y'][va],probs)
        row={'round':round_idx,'val_macro_f1':vm['macro_f1'],'val_accuracy':vm['accuracy'],
             'val_balanced_accuracy':vm['balanced_accuracy'],'train_loss_mean':float(np.mean(local_losses)),
             'round_wall_seconds':time.perf_counter()-start,'n_clients':len(selected)}
        trace.append(row)
        if vm['macro_f1']>best_score:
            best_score=vm['macro_f1'];best_round=round_idx
            best_state={k:v.detach().cpu().clone() for k,v in accum.items()}
        if cfg.get('save_rounds',False):
            torch.save({k:v.detach().cpu() for k,v in accum.items()},out/f'checkpoint_round{round_idx:03d}.pt')
        save_json(out/'progress.json',row)
        print(f'{out.name} round {round_idx}/{cfg["rounds"]} val-F1={vm["macro_f1"]:.4f} val-acc={vm["accuracy"]:.4f} {row["round_wall_seconds"]:.1f}s',flush=True)
    final_state={k:v.detach().cpu().clone() for k,v in global_model.state_dict().items()}
    final_probs=predict(global_model,x,rr,te,device)
    final_val=predict(global_model,x,rr,va,device)
    global_model.load_state_dict(best_state)
    best_probs=predict(global_model,x,rr,te,device)
    best_val=predict(global_model,x,rr,va,device)
    torch.save(best_state,out/'checkpoint_best.pt');torch.save(final_state,out/'checkpoint_final.pt')
    np.savez_compressed(out/'predictions.npz',y=d['y'][te],record=d['record'][te],peak=d['peak'][te],
                        indices=te,probs_best=best_probs,probs_final=final_probs,
                        val_y=d['y'][va],val_record=d['record'][va],val_probs=best_val)
    total_seconds=time.perf_counter()-start_all
    with open(out/'rounds.csv','w') as f:
        w=csv.DictWriter(f,fieldnames=list(trace[0]));w.writeheader();w.writerows(trace)
    with open(out/'client_trace.csv','w') as f:
        w=csv.DictWriter(f,fieldnames=list(timing[0]));w.writeheader();w.writerows(timing)
    result={'best_round':best_round,'best_validation_macro_f1':best_score,
            'best':metrics(d['y'][te],best_probs),'final':metrics(d['y'][te],final_probs),
            'validation_best':metrics(d['y'][va],best_val),'validation_final':metrics(d['y'][va],final_val),
            'n_train':len(tr),'n_val':len(va),'n_test':len(te),'n_clients':len(clients),
            'class_weights':weights.cpu().tolist(),'parameter_count':sum(p.numel() for p in global_model.parameters()),
            'raw_parameter_bytes':sum(p.numel()*p.element_size() for p in global_model.parameters()),
            'checkpoint_bytes':(out/'checkpoint_best.pt').stat().st_size,
            'initial_state_hash':initial_hash,'best_state_hash':state_hash(best_state),'final_state_hash':state_hash(final_state),
            'training_and_eval_wall_seconds':total_seconds,'host':{'machine':platform.machine(),'python':platform.python_version(),
            'torch':torch.__version__,'device':device,'threads':torch.get_num_threads()},
            'completed_utc':datetime.now(timezone.utc).isoformat(),
            'test_selection':'Checkpoint selection reads validation metrics only. Study development was exploratory and DS2 was inspected during architecture/weight exploration; this is not an untouched confirmation set.',
            'reproduction_status':'Standalone benchmark with explicit preprocessing, architectures and experimental configurations'}
    save_json(out/'metrics.json',result)
    print('DONE',out.name,'test',result['best']['macro_f1'],'final',result['final']['macro_f1'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.output)
