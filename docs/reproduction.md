# Reproduction

## Environment

Use a fresh checkout with Python 3.12. `scripts/setup.sh` creates `.venv` and
installs `requirements.txt`, then builds only the required native algorithms.
The native revision is `5a1a854b0dc9f2141bdc771c555ee60c37950183` and the Python
binding revision is `c6378cd5c8db74c0adf34ddcfbb96ee9c99f8061`.
All benchmark commands use `.venv/bin/python`.

Reference learners used Python 3.12.1, PyTorch 2.15.0, NumPy 2.5.4, SciPy 1.18.1,
scikit-learn 1.9.1 and wfdb 4.3.1. MLP workloads and later controls used one CPU
thread on arm64; the initial CNN screen used the MPS backend. The requirements
pin the complete collected environment. Floating-point results can vary across
backends, hardware and library builds; cross-device inference checks use
`atol=2e-5, rtol=2e-4`. Timing and peak memory are host-specific.

Raw signals, processed waveforms, environments, native builds and newly collected
outputs are ignored by Git. They are recreated locally. Saved numerical
references are an optional versioned release download with per-file checksums.

## Inspect without training

For a guided visual tour, begin with the [executed notebooks](../notebooks/README.md).
Their committed outputs are readable on GitHub. The notebook guide includes a
separate installation path that needs no raw signals or native transport build.

```sh
.venv/bin/python scripts/fetch_reference.py
.venv/bin/python scripts/verify_reference.py
```

For label bindings and checkpoint inference checks, first prepare the data and
add `--check-data`. The default verification checks archive/file integrity,
saved endpoints, checkpoint hashes, integration totals and every round's model
preservation, pooled grouped-validation endpoints, and the main, matched-exposure
and crossed-subset bootstrap intervals. It fits no model and
does not overwrite the reference artifacts.

The CSVs committed under `reference/summaries/` are small enough to inspect
directly. Full predictions, checkpoints, schedules, client traces and manifests
are under `reference/artifacts/` after extraction. Source-export metadata keeps
execution-source hashes distinct from portable-source hashes: namespace,
path and descriptive metadata changes are not represented as byte-identical
execution sources. Numerical implementation and the wire tag are retained.

## Check the portable learners

```sh
.venv/bin/python scripts/check_portability.py
```

This repeats 30 rounds for MLP/CE and MLP+RR/balanced CE with seeds 11,22,33 and
compares their best/final state hashes and selected macro-F1 to the frozen
reference. Its result is written to `reference/portability_checks.json`.
The expected exact-match claim is bounded to the pinned CPU environment.

## Full current workflow

After data preparation, use `scripts/reproduce.py`. It runs these stages in order:

| Stage | Conditions | Outputs |
|---|---:|---|
| Initial learning | 97 | `results/runs/`, `configs/runs/` |
| Analysis | Derived from fitted learners | `results/run_summary.csv`, uncertainty/diagnostic files |
| Validation and training controls | 43 | `results/validation_controls/` |
| Frozen transport integration | 54 timing executions | `results/benchmark/integration/` |
| Reporting | Derived tables and plots | `results/benchmark/`, `figures/` |

The initial stage contains one two-round pilot, three canonical small-subset
references, 48 screening runs, four additional-seed runs, 24 heterogeneity/
participation runs, four current transport controls, three centralized runs
and ten data-volume runs. Four retired transport controls are archived for
transparency but are not re-executed by the current workflow. Hence 101 initial
reference records correspond to 97 current conditions.

The validation stage comprises 27 grouped runs, ten matched-exposure runs and
six extra subset-draw runs. The integration stage is two models × three learning
seeds × three policies × three timing repeats. Those repeats share the same
six fitted-learning trajectories, so execution counts are not sample sizes
for predictive inference.

```sh
.venv/bin/python scripts/reproduce.py --device auto
```

For a CPU-only run, use `--device cpu`. To inspect or resume a stage, use
`--phase learning`, `analysis`, `validation`, `integration` or `reporting`.
Later stages require earlier outputs. Individual settings are in `configs/`.
The complete CNN grid takes longer than the small MLP replay; backend and
machine speed determine elapsed time.

Fresh outputs have creation-time source/data/configuration identities. Changing
those inputs is rejected when reusing a run folder. Keep reference artifacts
separate and use a fresh working copy for a changed experiment. Some historical
reference runs have retrospective audits instead of creation-time identities;
their recorded hashes are not retrofitted to imply stronger provenance.

The classifier screen is exploratory, with earlier DS2 inspection disclosed.
Broader patient validation does not convert repeated test use into independent
confirmation. The current negative findings and rare-class failures should be
retained when interpreting a reproduction.
