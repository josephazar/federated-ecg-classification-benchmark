# Federated ECG Classification Benchmark

[![Verify benchmark](https://github.com/josephazar/federated-ecg-classification-benchmark/actions/workflows/verify.yml/badge.svg)](https://github.com/josephazar/federated-ecg-classification-benchmark/actions/workflows/verify.yml)

A reproducible benchmark for **patient-disjoint ECG beat classification with
federated learning**. It studies how model architecture, class weighting,
patient-level validation and training exposure affect five-class performance on
MIT-BIH. A separate integration benchmark checks whether protecting actual
model downloads and uploads preserves the learner's parameters, and measures
the resulting host execution and serialized communication costs.

The repository provides explicit preprocessing, compact MLP/CNN baselines,
FedAvg, fixed experimental configurations, saved numerical references, and
verification scripts. It includes unsuccessful variants and uncertainty estimates.
Its purpose is to make the experiments inspectable and repeatable.

**Maintainer:** Joseph Azar — [joseph_azar@outlook.com](mailto:joseph_azar@outlook.com)

## Choose a reproduction path

| Goal | What to run | Retraining |
|---|---|---|
| Check the implementation | Install, prepare data, run the tests | None |
| Verify saved results | Fetch the versioned archive and run `verify_reference.py` | None |
| Check portable CPU reproduction | Run `check_portability.py` | Six 30-round MLP runs |
| Repeat the complete current workflow | Run `reproduce.py` | 194 executions, including timing repeats |

The reference archive contains **198 historical execution records**: 101 initial
runs, 43 validation/training-control runs and 54 integration timing executions.
Four initial transport controls are retained as retired records. The current
workflow runs the other 194 conditions. The 54 integration executions repeat
six model/learning-seed pairs; they are not 54 independent predictive trials.

## Install and prepare the data

Use Python 3.12 and a C compiler, Git and a POSIX shell. The setup script creates
`.venv`, installs the pinned requirements, and builds the pinned native
`liboqs` revision locally under `vendor/`. It does not install Python packages
into the system interpreter.

```sh
git clone https://github.com/josephazar/federated-ecg-classification-benchmark.git
cd federated-ecg-classification-benchmark
PYTHON_BOOTSTRAP=python3.12 sh scripts/setup.sh
.venv/bin/python -m fedecg.data download
.venv/bin/python -m fedecg.data preprocess
.venv/bin/python -m pytest -q
```

The raw signals are downloaded from the public MIT-BIH 1.0.0 distribution.
They are not included in Git or the reference archive. Dataset terms and source
links are in [Data](docs/data.md).

## Verify the numerical references

```sh
.venv/bin/python scripts/fetch_reference.py
.venv/bin/python scripts/verify_reference.py --check-data
```

The downloader checks the archive SHA256 and every extracted file against its
manifest. The verifier independently recomputes saved classification endpoints,
checkpoint hashes, bootstrap intervals, integration message counts and byte
totals. It compares every integration round across policies and all retained
round-state arrays. With `--check-data`, it also checks dataset label bindings
and CPU forward passes for the 101 initial checkpoints.

The portable release was checked in a fresh `.venv`: 24 tests passed, all 198
reference executions reconciled, and six complete CPU learner replays matched
their best/final checkpoints exactly. The recorded checks are in
[verification_checks.json](reference/verification_checks.json) and
[portability_checks.json](reference/portability_checks.json).

The [v1.0.0 reference release](https://github.com/josephazar/federated-ecg-classification-benchmark/releases/tag/v1.0.0)
contains approximately 417 MiB of compressed numerical artifacts. Its checksum
is pinned in [artifact_release.json](reference/artifact_release.json). Extraction
uses `reference/artifacts/`; new experiments use a separate `results/` tree.

## Repeat the experiments

```sh
# Exact frozen CPU replay check against the saved references
.venv/bin/python scripts/check_portability.py

# Full current workflow; auto selects the available compute backend
.venv/bin/python scripts/reproduce.py --device auto

# Or run stages separately
.venv/bin/python scripts/reproduce.py --phase learning --device cpu
.venv/bin/python scripts/reproduce.py --phase analysis
.venv/bin/python scripts/reproduce.py --phase validation
.venv/bin/python scripts/reproduce.py --phase integration
.venv/bin/python scripts/reproduce.py --phase reporting
```

Keep stage order for a fresh run: later stages depend on the fitted learners and
earlier summaries. Completed runs are reused only when their saved source,
configuration and data identities match. Preserve existing evidence and use a
fresh working copy when changing those inputs. Full instructions, phase counts,
hardware limits and expected outputs are in [Reproduction](docs/reproduction.md).

## What is implemented

- Beat extraction from named MLII leads and expert beat annotations; fixed
  N/S/V/F/Q mapping; per-beat normalization and four past-RR features.
- Patient-disjoint training, validation and testing, including the record
  201/202 shared-subject correction, and patient clients with natural skew.
- Five compact architectures and a 16-recipe exploratory screen using ordinary,
  weighted and balanced cross-entropy, and weighted focal loss.
- Three patient-group validation folds, matched optimizer-step experiments,
  crossed subset/training-seed sensitivity and patient influence diagnostics.
- Paired bootstrap uncertainty that pools patient confusion matrices before F1.
- Real library protection of full FP32 model messages, with none/fixed-L3/adaptive
  policies, paired learning schedules and complete byte/time accounting.

See [Methods](docs/methods.md) for exact architectures, loss formulas, checkpoint
selection, bootstrap pooling order and measurement boundaries.

## Main reference outcomes

| Comparison | Macro-F1, mean ± seed SD | Scope |
|---|---:|---|
| MLP / CE | 0.3545 ± 0.0075 | Five seeds, validation-selected checkpoint |
| MLP + RR / balanced CE | 0.3377 ± 0.0123 | Five seeds, initially selected recipe |
| Spatial CNN + RR / focal | 0.3781 ± 0.0068 | Three seeds, exploratory, not validation-selected |
| Matched steps / 6,396 beats | 0.3461 ± 0.0072 | Five seeds, fixed round 30 |
| Matched steps / 40,142 beats | 0.3511 ± 0.0038 | Five seeds, fixed round 30 |

The initially selected recipe's paired test difference is **−0.0168**, with a
patient/seed 95% interval of **[−0.0477, +0.0319]**. The matched-step full-minus-small
difference is **+0.0049**, with interval **[−0.0092, +0.0238]**. These results do not
establish a reliable predictive improvement. Earlier test inspection, few
validation patients and rare F/Q outcomes limit the interpretation.

All 54 integration executions preserve their paired model states at every round.
Adaptive protection reduces added protection bytes by 31.11% under the declared
trace, while total model traffic decreases by 1.86% for MLP and 1.64% for MLP+RR.
Host timing is descriptive for one machine. It does not measure network latency,
parallel client operation or physical device energy. Details and complete
machine-readable tables are in [Results](docs/results.md) and
[reference/summaries](reference/summaries).

![Patient group validation](figures/patient_group_validation.png)

## Layout

```text
fedecg/               Data, models, training, validation, statistics and transport
configs/              Experimental specifications and individual run settings
scripts/              Setup, reference download/verification and reproduction
tests/                Data separation, objective, identity and protection checks
reference/summaries/  Small result tables available directly in Git
reference/artifacts/  Optional downloaded numerical evidence; ignored by Git
figures/              Reproducible scientific figures
docs/                 Methods, data, results and reproduction guide
results/              New locally collected outputs; ignored by Git
```

## License

The implementation is released under the [MIT License](LICENSE). The dataset
and derived annotations remain subject to the terms stated by PhysioNet.
The native dependencies are fetched from their upstream projects with their
own licenses; their source and build directories are not vendored in this Git
history.
