import json
from pathlib import Path

import pytest

from fedecg.identity import ensure_identity
from fedecg.validation import FOLDS
from fedecg.data import DS1,DS2,patient_id


def test_grouped_folds_cover_strict_ds1_once_and_exclude_ds2():
    records=[r for fold in FOLDS for r in fold]
    assert len(records)==len(set(records))==21
    assert set(records)==set(DS1)-{201}
    assert not ({patient_id(r) for r in records}&{patient_id(r) for r in DS2})


def test_source_identity_rejects_changed_inputs_and_unbound_legacy(tmp_path):
    ensure_identity(tmp_path,{'source':'a','data':'b'})
    ensure_identity(tmp_path,{'source':'a','data':'b'})
    with pytest.raises(ValueError):ensure_identity(tmp_path,{'source':'changed','data':'b'})
    legacy=tmp_path/'legacy';legacy.mkdir();(legacy/'metrics.json').write_text('{}')
    with pytest.raises(ValueError):ensure_identity(legacy,{'source':'a'})
    ensure_identity(legacy,{'source':'a'},allow_legacy=True)
    assert not (legacy/'run_identity.json').exists(), 'Never retrofit creation-time identity to legacy runs'


def test_selection_requires_a_complete_grid(tmp_path,monkeypatch):
    from scripts import run_study
    monkeypatch.setattr(run_study,'ROOT',tmp_path)
    candidate=[('mlp','ce')]
    for seed in [11,22]:
        p=tmp_path/f'results/runs/main_mlp_ce_s{seed}';p.mkdir(parents=True)
        (p/'config.json').write_text(json.dumps({'model':'mlp','loss':'ce','seed':seed}))
        (p/'metrics.json').write_text(json.dumps({'best_validation_macro_f1':.2}))
    with pytest.raises(ValueError):run_study.select(candidate)
    p=tmp_path/'results/runs/main_mlp_ce_s33';p.mkdir()
    (p/'config.json').write_text(json.dumps({'model':'mlp','loss':'ce','seed':33}))
    (p/'metrics.json').write_text(json.dumps({'best_validation_macro_f1':.2}))
    assert run_study.select(candidate)['complete_screening_grid']
