"""Download and verify the versioned numerical reference archive."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from zipfile import ZipFile

import requests

ROOT=Path(__file__).resolve().parents[1]


def extract_verified(archive,destination,expected):
    archive=Path(archive);destination=Path(destination)
    h=hashlib.sha256()
    with archive.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    if h.hexdigest()!=expected:raise ValueError('Reference archive SHA256 mismatch')
    with ZipFile(archive) as z:
        for name in z.namelist():
            p=Path(name)
            if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe archive path')
        manifest=json.loads(z.read('MANIFEST.json'))
        if set(z.namelist())!=set(manifest['files'])|{'MANIFEST.json'}:
            raise ValueError('Archive inventory mismatch')
        for name,entry in manifest['files'].items():
            payload=z.read(name)
            if len(payload)!=entry['bytes'] or hashlib.sha256(payload).hexdigest()!=entry['sha256']:
                raise ValueError(f'Reference file checksum mismatch: {name}')
        destination.mkdir(parents=True,exist_ok=True)
        z.extractall(destination)
    return manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,help='Use an already downloaded archive');a=p.parse_args()
    release=json.loads((ROOT/'reference/artifact_release.json').read_text())
    destination=ROOT/'reference/artifacts'
    with tempfile.TemporaryDirectory() as tmp:
        archive=a.archive
        if archive is None:
            archive=Path(tmp)/release['archive']
            with requests.get(release['url'],stream=True,timeout=(20,300)) as r:
                r.raise_for_status()
                with archive.open('wb') as f:
                    for chunk in r.iter_content(1024*1024):f.write(chunk)
        manifest=extract_verified(archive,destination,release['sha256'])
    print(f'Verified {len(manifest["files"])} files at {destination.relative_to(ROOT)}')


if __name__=='__main__':main()
