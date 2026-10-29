"""Execute each tutorial in a fresh kernel; keep outputs for GitHub's notebook viewer."""
import argparse
import json
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'notebooks',
                        help='Use results/notebook-check for a non-overwriting verification run.')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checked = []
    for path in sorted((ROOT / 'notebooks').glob('[0-9][0-9]_*.ipynb')):
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        client = NotebookClient(nb, timeout=180, kernel_name='python3',
                                resources={'metadata': {'path': str(ROOT / 'notebooks')}})
        # Use this isolated environment even if the user has another Python kernel registered.
        client.km = client.create_kernel_manager()
        client.km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher',
                                     '-f', '{connection_file}']
        client.execute()
        image_count = 0
        for cell in nb.cells:
            cell.metadata.pop('execution', None)
            for item in cell.get('outputs', []):
                if item.output_type == 'error':
                    raise RuntimeError(f'Failed cell in {path.name}')
                image_count += int('image/png' in item.get('data', {}))
        assert image_count > 0, f'No rendered plot in {path.name}'
        nb.metadata.pop('widgets', None)
        nbformat.write(nb, output / path.name)
        record = {'notebook': path.name, 'executed_cells': sum(c.cell_type == 'code' for c in nb.cells),
                  'embedded_plots': image_count, 'status': 'passed'}
        checked.append(record)
        print(json.dumps(record), flush=True)
    if len(checked) != 5:
        raise RuntimeError('Expected all five teaching notebooks.')
    (output / 'execution_checks.json').write_text(json.dumps({'checks': checked}, indent=2) + '\n')


if __name__ == '__main__':
    main()
