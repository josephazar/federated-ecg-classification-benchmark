"""MIT-BIH acquisition, annotation-assisted beat extraction, and split manifests.

The database's expert R-peak annotations are inputs: this is beat classification,
not an end-to-end R-peak detector. No test labels are used to fit transformations.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import requests
import wfdb

from .common import ROOT, save_json, sha256

BASE = 'https://physionet-open.s3.amazonaws.com/mitdb/1.0.0/'
DS1 = [101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,223,230]
DS2 = [100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234]
VALIDATION = [106,114,119,209]
CLASSES = ['N', 'S', 'V', 'F', 'Q']
GROUPS = ['NLRej', 'AaJS', 'VE', 'F', '/fQ']
LABELS = {s: k for k, group in enumerate(GROUPS) for s in group}


def patient_id(record):
    return 201 if int(record) == 202 else int(record)


def download():
    raw = ROOT / 'data/raw/mitdb'
    raw.mkdir(parents=True, exist_ok=True)
    rec = requests.get(BASE + 'RECORDS', timeout=60)
    rec.raise_for_status()
    (raw / 'RECORDS').write_bytes(rec.content)
    records = rec.text.split()
    assert len(records) == 48
    names = [f'{r}.{ext}' for r in records for ext in ['hea','dat','atr']]

    def fetch(name):
        p = raw / name
        if not p.exists():
            for attempt in range(4):
                try:
                    response = requests.get(BASE + name, timeout=(20,90))
                    response.raise_for_status()
                    tmp = p.with_suffix(p.suffix + '.part')
                    tmp.write_bytes(response.content)
                    tmp.replace(p)
                    break
                except requests.RequestException:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
        return {'file': name, 'bytes': p.stat().st_size, 'sha256': sha256(p), 'url': BASE + name}

    entries = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, future in enumerate(as_completed([pool.submit(fetch,n) for n in names]), 1):
            entries.append(future.result())
            if i % 24 == 0:
                print(f'Downloaded/verified {i}/{len(names)} files', flush=True)
    save_json(ROOT/'data/raw/download_manifest.json', {
        'source': BASE, 'version': '1.0.0', 'retrieved_utc': datetime.now(timezone.utc).isoformat(),
        'checksum_note': 'Local SHA256 provenance hashes, not a claim of comparison with server checksums.',
        'files': sorted(entries, key=lambda e:e['file'])})


def preprocess():
    raw = ROOT/'data/raw/mitdb'
    signals, rrs, labels, records, peaks = [], [], [], [], []
    descriptions = []
    before, after = 90, 162
    for rid in sorted(DS1+DS2):
        rec = wfdb.rdrecord(str(raw/str(rid)))
        ann = wfdb.rdann(str(raw/str(rid)), 'atr')
        channel = rec.sig_name.index('MLII') if 'MLII' in rec.sig_name else 0
        signal = rec.p_signal[:, channel]
        beats = [(int(t), LABELS[s]) for t,s in zip(ann.sample,ann.symbol) if s in LABELS]
        positions = np.array([t for t,_ in beats])
        intervals = np.diff(positions)/rec.fs
        counts = Counter()
        dropped = 0
        for j,(peak,y) in enumerate(beats):
            if peak < before or peak+after > len(signal):
                dropped += 1
                continue
            x = signal[peak-before:peak+after].astype(np.float32)
            if not np.isfinite(x).all():
                dropped += 1
                continue
            x = (x-x.mean())/max(float(x.std()), 1e-6)
            prev = float(intervals[j-1]) if j else 1.0
            prev2 = float(intervals[j-2]) if j>1 else prev
            local = float(np.median(intervals[max(0,j-10):j])) if j else 1.0
            rr = np.log(np.clip([prev, prev2, local, prev/max(local,1e-3)], .1, 5)).astype(np.float32)
            signals.append(x); rrs.append(rr); labels.append(y); records.append(rid); peaks.append(peak)
            counts[CLASSES[y]] += 1
        descriptions.append({'record':rid, 'patient':patient_id(rid), 'lead':rec.sig_name[channel],
                             'fs':float(rec.fs), 'counts':dict(counts), 'boundary_or_nonfinite_drops':dropped})
        print(f'Processed {rid}: {dict(counts)}', flush=True)
    out = ROOT/'data/processed'
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/'beats.npz', x=np.stack(signals), rr=np.stack(rrs),
                        y=np.array(labels,dtype=np.int64), record=np.array(records,dtype=np.int32),
                        peak=np.array(peaks,dtype=np.int64))
    save_json(out/'preprocessing.json', {
        'dataset':'MIT-BIH 1.0.0', 'classes':CLASSES, 'annotation_mapping':LABELS,
        'excluded_paced_records':[102,104,107,217], 'window_samples':252, 'before_R':90, 'after_R':162,
        'normalization':'per-beat zero mean / population SD; floor 1e-6; no learned test statistics',
        'filter':'none', 'rr_features':'log clipped previous RR, previous-previous RR, median past 10 RR, previous/median; no future RR',
        'lead':'MLII by header name (handles reversed channel order in record 114)',
        'requires_expert_R_peaks':True, 'records':descriptions, 'array_sha256':sha256(out/'beats.npz')})
    print('Saved', len(labels), 'beats', flush=True)


def stratified_subset(indices, y, n, seed):
    if n >= len(indices):
        return indices.copy()
    rng = np.random.default_rng(seed)
    counts = np.bincount(y[indices], minlength=5)
    quota = np.floor(counts*n/len(indices)).astype(int)
    quota = np.minimum(counts, np.maximum(quota, np.minimum(counts,8)))
    while quota.sum() > n:
        quota[np.argmax(quota)] -= 1
    while quota.sum() < n:
        available = counts-quota
        ideal = counts*n/len(indices)-quota
        ideal[available==0] = -np.inf
        quota[np.argmax(ideal)] += 1
    return np.sort(np.concatenate([rng.choice(indices[y[indices]==k],size=int(q),replace=False)
                                   for k,q in enumerate(quota) if q]))


def load_split(protocol='strict', train_size=0, test_size=None):
    d = dict(np.load(ROOT/'data/processed/beats.npz'))
    train_records = sorted(set(DS1)-set(VALIDATION)-({201} if protocol=='strict' else set()))
    train = np.flatnonzero(np.isin(d['record'],train_records))
    val = np.flatnonzero(np.isin(d['record'],VALIDATION))
    test = np.flatnonzero(np.isin(d['record'],DS2))
    if train_size:
        train = stratified_subset(train,d['y'],train_size,20261028)
        if test_size is None:
            test = stratified_subset(test,d['y'],6374,20261029)
    if test_size:
        test = stratified_subset(test,d['y'],test_size,20261029)
    if protocol=='strict':
        a,b,c = [{patient_id(r) for r in d['record'][idx]} for idx in [train,val,test]]
        assert not (a&b or a&c or b&c), 'Patient leakage'
    return d, train, val, test


def build_clients(d, train, partition, seed, n_clients=10, alpha=.5):
    if partition=='patient':
        return [train[d['record'][train]==r] for r in sorted(set(d['record'][train]))]
    if partition=='central':
        return [train]
    if partition not in ['iid','dirichlet']:
        raise ValueError(partition)
    rng=np.random.default_rng(seed+30000)
    for attempt in range(100):
        groups=[[] for _ in range(n_clients)]
        for k in range(5):
            ix=train[d['y'][train]==k].copy(); rng.shuffle(ix)
            probs=np.full(n_clients,1/n_clients) if partition=='iid' else rng.dirichlet(np.full(n_clients,alpha))
            cuts=np.round(np.cumsum(probs)[:-1]*len(ix)).astype(int)
            for g,part in zip(groups,np.split(ix,cuts)):
                g.extend(part.tolist())
        if min(map(len,groups))>=32:
            return [np.array(g,dtype=np.int64) for g in groups]
    raise ValueError('Could not construct clients with >=32 observations')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['download','preprocess'])
    globals()[p.parse_args().action]()
