"""Extract small teaching inputs from the unchanged v1.0.0 reference artifacts."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT / 'reference/artifacts/results/runs'
    target = ROOT / 'notebooks/data'
    target.mkdir(parents=True, exist_ok=True)
    seeds = [11, 22, 33, 44, 55]
    stems = ['main_mlp_ce', 'main_mlp_rr_balanced']
    hashes, curves, methods = {}, [], []
    canonical = None
    for stem in stems:
        per_seed = []
        for seed in seeds:
            run = f'{stem}_s{seed}'
            folder = source / run
            for name in ['predictions.npz', 'rounds.csv', 'metrics.json']:
                p = folder / name
                hashes[p.relative_to(ROOT).as_posix()] = digest(p)
            with np.load(folder / 'predictions.npz', allow_pickle=False) as pred:
                binding = np.stack([pred['indices'], pred['record'], pred['y']])
                if canonical is None:
                    canonical = binding
                assert np.array_equal(binding, canonical), run
                patients = np.unique(pred['record'])
                guessed = pred['probs_best'].argmax(axis=1)
                matrices = np.stack([
                    np.bincount(pred['y'][pred['record'] == r] * 5 +
                                guessed[pred['record'] == r], minlength=25).reshape(5, 5)
                    for r in patients])
                expected = json.loads((folder / 'metrics.json').read_text())
                assert np.array_equal(matrices.sum(axis=0), expected['best']['confusion_matrix'])
                per_seed.append(matrices)
            curve = pd.read_csv(folder / 'rounds.csv')[['round', 'val_macro_f1']]
            curves.append(curve.assign(run=run, method=stem, seed=seed))
        methods.append(np.stack(per_seed))
    np.savez_compressed(target / 'primary_patient_confusions.npz',
                        confusions=np.stack(methods), seeds=seeds, patients=patients,
                        methods=np.array(stems), classes=np.array(list('NSVFQ')))
    pd.concat(curves, ignore_index=True).to_csv(target / 'validation_curves.csv', index=False)
    waveform_path = ROOT / 'data/processed/beats.npz'
    preprocessing = json.loads((ROOT / 'data/processed/preprocessing.json').read_text())
    assert digest(waveform_path) == preprocessing['array_sha256']
    support = pd.read_csv(ROOT / 'reference/summaries/main_patient_class_support.csv')
    training_records = support.loc[support['split'].eq('training'), 'record'].unique()
    wave_rows = []
    with np.load(waveform_path, allow_pickle=False) as data:
        for k, label in enumerate('NSVFQ'):
            keep = np.isin(data['record'], training_records) & (data['y'] == k)
            waveforms = data['x'][keep]
            mean = waveforms.mean(axis=0, dtype=np.float64)
            sd = waveforms.std(axis=0, dtype=np.float64)
            wave_rows.extend({'class': label, 'sample': i, 'time_ms': (i - 90) / 360 * 1000,
                              'mean': mean[i], 'sd': sd[i], 'training_beats': len(waveforms)}
                             for i in range(252))
    pd.DataFrame(wave_rows).to_csv(target / 'training_waveform_summary.csv', index=False)
    hashes[waveform_path.relative_to(ROOT).as_posix()] = preprocessing['array_sha256']
    inputs = list((ROOT / 'reference/summaries').glob('*'))
    inputs += [ROOT / 'data/processed/preprocessing.json', ROOT / 'reference/artifact_release.json']
    inputs += [target / 'primary_patient_confusions.npz', target / 'validation_curves.csv']
    inputs += [target / 'training_waveform_summary.csv']
    manifest = {
        'reference_version': 'v1.0.0',
        'reference_archive_sha256': json.loads((ROOT / 'reference/artifact_release.json').read_text())['sha256'],
        'description': 'Patient confusion counts and validation curves from saved runs, and training-only waveform means/SDs. No models were fitted or selected.',
        'axes': ['method', 'learning seed', 'test patient', 'true class', 'predicted class'],
        'files': {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(inputs)},
        'derivation_sources': hashes,
    }
    (target / 'provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Prepared ten saved learners: confusion counts and validation curves.')


if __name__ == '__main__':
    main()
