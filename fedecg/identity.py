"""Bind future training runs to their implementation and processed data."""
import json
import os
from pathlib import Path
import importlib.metadata

from .common import ROOT,save_json,sha256


def current_identity(cfg, extra_sources=()):
    paths=['fedecg/common.py','fedecg/data.py','fedecg/models.py','fedecg/train.py','fedecg/identity.py',*extra_sources]
    return {'config':cfg,'source_sha256':{p:sha256(ROOT/p) for p in paths},
            'processed_array_sha256':sha256(ROOT/'data/processed/beats.npz'),
            'packages':{p:importlib.metadata.version(p) for p in ['torch','numpy','scikit-learn']}}


def ensure_identity(out,identity,allow_legacy=False):
    out=Path(out);path=out/'run_identity.json'
    if path.exists():
        if json.loads(path.read_text())!=identity:
            raise ValueError(f'Source/data/config identity changed at {out}; use a new output directory.')
    elif (out/'metrics.json').exists():
        if not allow_legacy:
            raise ValueError(f'Legacy run lacks a creation-time identity at {out}. Preserve it and use a new output directory, or explicitly set FEDECG_ALLOW_LEGACY_REUSE=1 for inspected legacy artifacts.')
        print('Explicit legacy reuse; creation-time source/data binding unavailable:',out,flush=True)
    else:
        out.mkdir(parents=True,exist_ok=True);save_json(path,identity)
