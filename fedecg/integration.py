"""Paired host integration benchmark using the unchanged protection wrapper."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

from .common import ROOT, save_json, sha256
from .data import load_split, build_clients
from .identity import current_identity, ensure_identity
from .models import make_model, class_weights, loss_fn
from .train import state_hash

SPEC = ROOT / 'configs/integration.json'
OUT = ROOT / 'results/benchmark/integration'


def encode(state):
    return b''.join(v.detach().cpu().contiguous().numpy().tobytes() for v in state.values())


def decode(payload, template):
    result = {}; offset = 0
    for name, value in template.items():
        dtype = value.detach().cpu().numpy().dtype
        size = value.numel() * value.element_size()
        a = np.frombuffer(payload[offset:offset + size], dtype=dtype).copy()
        result[name] = torch.from_numpy(a.reshape(tuple(value.shape)))
        offset += size
    if offset != len(payload):
        raise ValueError('Unexpected serialized model length')
    return result


def transmit(payload, policy, required):
    from .crypto import roundtrip
    if policy == 'none':
        return payload, len(payload), 0, 0
    level = required if policy == 'adaptive' else 3
    recovered, cm = roundtrip(payload, level)
    if recovered != payload:
        raise ValueError('Protection altered a model payload')
    # The unchanged wrapper reports all components except its existing AAD.
    aad_bytes = len(b'apqfl-level-' + str(level).encode('ascii'))
    extra = int(cm['extra_bytes']) + aad_bytes
    return recovered, len(payload) + extra, extra, level


def flat(state):
    return np.concatenate([state[k].detach().cpu().numpy().ravel() for k in sorted(state)])


def worker(cfg):
    spec = json.loads(SPEC.read_text())
    label = f'{cfg["model"]}_{cfg["loss"]}_s{cfg["seed"]}_{cfg["policy"]}_r{cfg["repetition"]}'
    out = OUT / label
    identity = current_identity(cfg, extra_sources=['fedecg/integration.py', 'fedecg/crypto.py', 'configs/integration.json'])
    identity['native_libraries'] = {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT/'vendor/oqs-install/lib').glob('liboqs*')) if p.is_file()}
    identity['extra_packages'] = {p: importlib.metadata.version(p) for p in ['cryptography', 'liboqs-python']}
    ensure_identity(out, identity)
    if (out / 'metrics.json').exists():
        return
    save_json(out / 'config.json', cfg)
    torch.set_num_threads(1); torch.manual_seed(cfg['seed'])
    torch.use_deterministic_algorithms(True)
    data, train, _, _ = load_split('strict')
    global_clients = build_clients(data, train, 'patient', cfg['seed'])
    clients = [np.searchsorted(train, c) for c in global_clients]
    x = torch.from_numpy(data['x'][train, None].copy())
    rr = torch.from_numpy(data['rr'][train].copy())
    y = torch.from_numpy(data['y'][train].copy())
    record = data['record'][train].copy()
    save_json(out / 'split_manifest.json', {
        'training_global_indices': train.tolist(), 'records': sorted(map(int, set(record))),
        'n_training_beats': len(train), 'n_clients': len(clients),
        'class_counts': np.bincount(y.numpy(), minlength=5).tolist(),
        'client_records': [sorted(map(int, set(record[ix]))) for ix in clients],
        'validation_or_test_evaluated': False,
    })
    del data
    weights = class_weights(y, cfg['loss'])
    global_model = make_model(cfg['model']); local_model = make_model(cfg['model'])
    schema = [{'name': k, 'shape': list(v.shape), 'dtype': str(v.dtype),
               'bytes': v.numel() * v.element_size()} for k, v in global_model.state_dict().items()]
    save_json(out / 'serialization_schema.json', schema)
    # Common warm-up outside measured rounds, including library loading in all modes.
    from .crypto import roundtrip
    for level in [1, 2, 3]:
        roundtrip(b'integration warmup', level)
    loss_fn(local_model(x[:128], rr[:128]), y[:128], cfg['loss'], weights).backward()
    local_model.zero_grad(set_to_none=True)
    context_rng = np.random.default_rng(cfg['seed'] + 20000)
    requirements = context_rng.choice([1, 2, 3], size=(spec['rounds'], len(clients)), p=[.244, .712, .044])
    save_json(out / 'schedule.json', {'clients': list(range(len(clients))), 'required_levels': requirements.tolist()})
    initial_hash = state_hash(global_model.state_dict())
    vectors = [flat(global_model.state_dict())]
    rounds = []; messages = []
    for r in range(1, spec['rounds'] + 1):
        begin = time.perf_counter()
        original = {k: v.detach().clone() for k, v in global_model.state_dict().items()}
        accum = {k: torch.zeros_like(v) for k, v in original.items()}
        counts = {'training_seconds': 0., 'serialization_decode_seconds': 0.,
                  'transport_seconds': 0., 'aggregation_seconds': 0.,
                  'raw_bytes': 0, 'wire_component_bytes': 0, 'protection_extra_bytes': 0,
                  'local_steps': 0}
        t = time.perf_counter(); download = encode(original)
        counts['serialization_decode_seconds'] += time.perf_counter() - t
        for ci, ix in enumerate(clients):
            required = int(requirements[r-1, ci])
            t = time.perf_counter(); recovered, wire, extra, level = transmit(download, cfg['policy'], required)
            counts['transport_seconds'] += time.perf_counter() - t
            assert recovered == download
            t = time.perf_counter(); local_model.load_state_dict(decode(recovered, original))
            counts['serialization_decode_seconds'] += time.perf_counter() - t
            for direction, raw, total, added, used in [('download', len(download), wire, extra, level)]:
                messages.append({'round': r, 'client': ci, 'direction': direction, 'required_level': required,
                                 'level': used, 'raw_bytes': raw, 'wire_component_bytes': total,
                                 'protection_extra_bytes': added})
            counts['raw_bytes'] += len(download); counts['wire_component_bytes'] += wire; counts['protection_extra_bytes'] += extra
            t = time.perf_counter(); local_model.train()
            opt = torch.optim.AdamW(local_model.parameters(), lr=.001, weight_decay=.0001)
            order = np.random.default_rng(np.random.SeedSequence([cfg['seed'], r, ci, 0])).permutation(ix)
            for b in range(0, len(order), 128):
                ids = torch.as_tensor(order[b:b+128], dtype=torch.long)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(local_model(x[ids], rr[ids]), y[ids], cfg['loss'], weights)
                loss.backward(); torch.nn.utils.clip_grad_norm_(local_model.parameters(), 5.)
                opt.step(); counts['local_steps'] += 1
            counts['training_seconds'] += time.perf_counter() - t
            t = time.perf_counter(); upload = encode(local_model.state_dict())
            counts['serialization_decode_seconds'] += time.perf_counter() - t
            t = time.perf_counter(); recovered, wire, extra, level = transmit(upload, cfg['policy'], required)
            counts['transport_seconds'] += time.perf_counter() - t
            assert recovered == upload
            t = time.perf_counter(); restored_state = decode(recovered, original)
            counts['serialization_decode_seconds'] += time.perf_counter() - t
            messages.append({'round': r, 'client': ci, 'direction': 'upload', 'required_level': required,
                             'level': level, 'raw_bytes': len(upload), 'wire_component_bytes': wire,
                             'protection_extra_bytes': extra})
            counts['raw_bytes'] += len(upload); counts['wire_component_bytes'] += wire; counts['protection_extra_bytes'] += extra
            t = time.perf_counter()
            for k, value in restored_state.items():
                accum[k].add_(value, alpha=len(ix) / len(train))
            counts['aggregation_seconds'] += time.perf_counter() - t
        t = time.perf_counter(); global_model.load_state_dict(accum)
        counts['aggregation_seconds'] += time.perf_counter() - t
        vectors.append(flat(accum))
        row = {'round': r, **counts, 'parameter_hash': state_hash(accum), 'round_wall_seconds': time.perf_counter() - begin}
        rounds.append(row)
    old_path = ROOT / f'results/runs/main_{cfg["model"]}_{cfg["loss"]}_s{cfg["seed"]}/metrics.json'
    expected = json.loads(old_path.read_text())['final_state_hash']
    final_hash = state_hash(global_model.state_dict())
    if final_hash != expected:
        raise ValueError(f'Frozen learner no longer matches the original main run: {label}')
    torch.save(global_model.state_dict(), out / 'checkpoint_final.pt')
    if cfg['repetition'] == 0:
        np.savez_compressed(out / 'round_states.npz', values=np.stack(vectors), rounds=np.arange(31))
    pd.DataFrame(rounds).to_csv(out / 'rounds.csv', index=False)
    pd.DataFrame(messages).to_csv(out / 'messages.csv', index=False)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    totals = {k: sum(row[k] for row in rounds) for k in counts}
    result = {**cfg, **totals, 'elapsed_rounds_seconds': sum(row['round_wall_seconds'] for row in rounds),
              'peak_worker_rss_bytes': int(peak if sys.platform == 'darwin' else peak * 1024),
              'initial_state_hash': initial_hash, 'final_state_hash': final_hash,
              'matches_existing_main_final_state': True, 'n_clients': len(clients), 'rounds': 30,
              'parameter_count': sum(v.numel() for v in global_model.parameters()),
              'payload_bytes': len(download), 'messages_verified': len(messages),
              'validation_or_test_evaluated': False, 'platform': platform.platform()}
    save_json(out / 'metrics.json', result)


def summarize():
    spec = json.loads(SPEC.read_text())
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob('*/metrics.json'))]
    assert len(rows) == 54
    a = pd.DataFrame(rows); a.to_csv(OUT.parent / 'integration_runs.csv', index=False)
    checks = []
    for model, loss in spec['recipes']:
        for seed in spec['learning_seeds']:
            b = a[(a.model == model) & (a.loss == loss) & (a.seed == seed)]
            assert len(b) == 9 and b.final_state_hash.nunique() == 1
            reference = None
            hashes = None
            for row in b.to_dict('records'):
                label = f'{model}_{loss}_s{seed}_{row["policy"]}_r{row["repetition"]}'
                trace = pd.read_csv(OUT / label / 'rounds.csv')
                current = trace.parameter_hash.tolist()
                if hashes is None: hashes = current
                assert hashes == current
                if row['repetition'] == 0:
                    v = np.load(OUT / label / 'round_states.npz')['values']
                    if reference is None: reference = v
                    assert np.array_equal(reference, v)
            checks.append({'model': model, 'loss': loss, 'seed': seed,
                           'matched_rounds': 30, 'matched_executions': 9,
                           'maximum_parameter_difference': 0., 'all_round_hashes_identical': True})
    save_json(OUT.parent / 'integration_preservation.json', {'comparisons': checks,
        'runs': 54, 'unique_learning_seed_model_pairs': 6,
        'test_predictions_created': False, 'all_existing_main_final_states_reproduced': True})
    values = ['elapsed_rounds_seconds', 'training_seconds', 'serialization_decode_seconds',
              'transport_seconds', 'aggregation_seconds', 'peak_worker_rss_bytes',
              'raw_bytes', 'wire_component_bytes', 'protection_extra_bytes']
    means = a.groupby(['model', 'loss', 'seed', 'policy'])[values].mean().reset_index()
    means.to_csv(OUT.parent / 'integration_seed_means.csv', index=False)
    summary = means.groupby(['model', 'loss', 'policy'])[values].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(OUT.parent / 'integration_summary.csv', index=False)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--worker', type=Path)
    p.add_argument('--summarize', action='store_true'); args = p.parse_args()
    if args.worker: worker(json.loads(args.worker.read_text())); return
    if args.summarize: summarize(); return
    spec = json.loads(SPEC.read_text()); OUT.mkdir(parents=True, exist_ok=True)
    tasks = [{'model': m, 'loss': loss, 'seed': s, 'policy': policy, 'repetition': rep}
             for m, loss in spec['recipes'] for s in spec['learning_seeds']
             for policy in spec['policies'] for rep in range(spec['timing_repetitions'])]
    order = np.random.default_rng(spec['execution_order_seed']).permutation(len(tasks))
    save_json(OUT.parent / 'integration_execution_order.json', [tasks[i] for i in order])
    for done, i in enumerate(order, 1):
        cfg = tasks[i]; label = f'{cfg["model"]}_{cfg["loss"]}_s{cfg["seed"]}_{cfg["policy"]}_r{cfg["repetition"]}'
        path = OUT.parent / 'configs' / f'{label}.json'; path.parent.mkdir(exist_ok=True)
        save_json(path, cfg)
        subprocess.run([sys.executable, '-m', 'fedecg.integration', '--worker', str(path)], check=True, cwd=ROOT)
        print(f'Integration {done}/54: {label}', flush=True)
    summarize()


if __name__ == '__main__': main()
