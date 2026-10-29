import hashlib
import json
from zipfile import ZipFile

import pytest

from scripts.fetch_reference import extract_verified


def test_reference_rejects_archive_digest_mismatch(tmp_path):
    p=tmp_path/'archive.zip';p.write_bytes(b'not the expected archive')
    with pytest.raises(ValueError,match='SHA256'):
        extract_verified(p,tmp_path/'out','0'*64)


def test_reference_rejects_parent_path_entry(tmp_path):
    p=tmp_path/'archive.zip'
    with ZipFile(p,'w') as z:z.writestr('../outside.txt','unsafe')
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='Unsafe'):
        extract_verified(p,tmp_path/'out',digest)
    assert not (tmp_path/'outside.txt').exists()


def test_reference_checks_each_file_before_extracting(tmp_path):
    p=tmp_path/'archive.zip';data=b'valid numerical evidence'
    with ZipFile(p,'w') as z:
        z.writestr('results/value.txt',data)
        z.writestr('MANIFEST.json',json.dumps({'files':{'results/value.txt':{'bytes':len(data),'sha256':'0'*64}}}))
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='file checksum'):
        extract_verified(p,tmp_path/'out',digest)
    assert not (tmp_path/'out').exists()
