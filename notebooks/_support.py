"""Small presentation helpers; the numerical teaching steps stay in the notebooks."""
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLASSES = list('NSVFQ')
COLORS = ['#2563a6', '#d97724', '#27856c', '#aa4c75', '#7569ad']
NAMES = {'mlp': 'MLP', 'mlp_rr': 'MLP + RR', 'cnn': 'Compact CNN',
         'cnn_rr': 'Compact CNN + RR', 'cnn_spatial_rr': 'Spatial CNN + RR'}


def read_table(name):
    return pd.read_csv(ROOT / 'reference/summaries' / name)


def read_json(name):
    return json.loads((ROOT / 'reference/summaries' / name).read_text())


def setup():
    """Verify the small checked-in inputs without fetching data or building native code."""
    manifest = json.loads((ROOT / 'notebooks/data/provenance.json').read_text())
    for relative, expected in manifest['files'].items():
        path = ROOT / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Notebook input has changed: {relative}')
    plt.rcParams.update({
        'figure.figsize': (9, 4), 'figure.dpi': 125, 'savefig.dpi': 150,
        'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.titleweight': 'bold', 'axes.labelcolor': '#25334a',
        'axes.prop_cycle': plt.cycler(color=COLORS),
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
    })
    pd.set_option('display.max_columns', 12)
    pd.set_option('display.precision', 4)


def finish(fig):
    fig.tight_layout()
    plt.show()
