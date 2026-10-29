# An illustrated tour of the benchmark

Start here if you prefer a visual explanation to reading the implementation.
All five notebooks are **executed and saved with their outputs**: GitHub shows
the diagrams, plots and tables directly. Open a notebook and choose **Preview**
if GitHub initially shows its source. No Jupyter installation is needed to read.

| Order | Notebook | What you will see |
|---|---|---|
| 1 | [Data, patients and the federated idea](01_data_and_federated_idea.ipynb) | The classification task, patient separation, unequal clients and the accuracy trap |
| 2 | [Models, objectives and FedAvg](02_models_and_fedavg.ipynb) | Actual architecture sizes, a two-branch diagram, tensor shapes and worked averaging |
| 3 | [Results and uncertainty](03_results_and_uncertainty.ipynb) | Every screened recipe, validation curves, paired seeds, class recalls and a recomputed bootstrap |
| 4 | [Experimental controls](04_experimental_controls.ipynb) | Grouped patient validation, training exposure and matched-budget comparisons |
| 5 | [Model-transfer measurements](05_model_transfer_measurements.ipynb) | Paired state preservation, message counts and the difference between overhead and total traffic |

Each notebook has a short learning goal, small code cells, interpreted outputs,
an exercise with an answer scaffold, and an optional extension. Read in order
or jump to a question of interest. The numerical comparisons retain the negative
findings and the limits of the exploratory evaluation.

## Run the tour locally

The tour works from a fresh checkout **without downloading ECG signals, fetching
the 417 MiB archive, training models or building the native transport library**.
It uses the small committed reference tables and three derived teaching inputs.

```sh
# From the repository root; Python 3.12
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-notebooks.txt
.venv/bin/python -m jupyterlab notebooks
```

If you already installed the full benchmark, reuse its `.venv` and install
`requirements-notebooks.txt` there. The numerical library versions match the
benchmark's pinned requirements. Run a notebook from its first cell onward.
The supplied execution script explicitly launches the current environment's
Python, so it does not depend on a system Python kernel registration.

To check all five in fresh kernels while retaining the committed outputs:

```sh
.venv/bin/python scripts/execute_notebooks.py --output-dir results/notebook-check
```

To deliberately refresh the outputs committed for GitHub:

```sh
.venv/bin/python scripts/execute_notebooks.py
```

Each run verifies the small input hashes, executes all cells, and writes an
`execution_checks.json` with code-cell and embedded-plot counts. Continuous
integration also executes the complete tour.

## Where the displayed numbers come from

| Input | Origin and purpose |
|---|---|
| `reference/summaries/` | Versioned benchmark tables and intervals already in Git |
| `data/processed/preprocessing.json` | Existing annotation mapping and preparation metadata |
| `data/validation_curves.csv` | Ten original 30-round validation curves for the five-seed comparison |
| `data/training_waveform_summary.csv` | Class-wise means and standard deviations of normalized training beats, with support counts; no individual waveform is redistributed |
| `data/primary_patient_confusions.npz` | Counts derived from the same ten saved selected-checkpoint predictions; axes are method × seed × patient × true class × predicted class |
| `data/provenance.json` | SHA256 values for notebook inputs, their prediction/curve sources, and the v1.0.0 reference archive |

The compact confusion counts retain everything needed for the primary fixed-
five-class F1 calculation and patient/seed bootstrap. Notebook 3 recomputes
the original 5,000-draw interval and asserts agreement. It does not infer new
predictions from counts. Toy examples in notebooks 2 and 5 are explicitly
labelled and kept separate from the experimental evidence.

To regenerate the compact inputs, first prepare the public data using the full
benchmark environment, then fetch the verified archive:

```sh
.venv/bin/python -m fedecg.data download
.venv/bin/python -m fedecg.data preprocess
.venv/bin/python scripts/fetch_reference.py
.venv/bin/python scripts/verify_reference.py
.venv/bin/python scripts/build_notebook_inputs.py
.venv/bin/python scripts/execute_notebooks.py
```

The notebooks are a guided view of the existing implementation and reference
results. Complete training and integration reproduction remain documented in
[the reproduction guide](../docs/reproduction.md). They do not launch training
as a side effect of opening or executing the tour.
