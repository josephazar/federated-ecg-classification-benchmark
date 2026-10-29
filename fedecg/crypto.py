"""Real library round trips and host microbenchmarks, not a new crypto protocol.

Keys are freshly generated for each prototype update. Public-key authentication,
certificate distribution, adversarial protocols and device power are outside this
evaluation. The level-to-parameter mapping is an explicit subset_reference choice.
"""
import argparse
import csv
import os
import time

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from .common import ROOT, save_json

os.environ.setdefault('OQS_INSTALL_PATH',str(ROOT/'vendor/oqs-install'))

LEVELS={1:('ML-KEM-512','ML-DSA-44'),2:('ML-KEM-768','ML-DSA-65'),3:('ML-KEM-1024','ML-DSA-87')}


def roundtrip(payload,level,negative_tests=False):
    import oqs
    kem_name,sig_name=LEVELS[int(level)]
    begin=time.perf_counter_ns()
    with oqs.KeyEncapsulation(kem_name) as recipient, oqs.KeyEncapsulation(kem_name) as sender, oqs.Signature(sig_name) as signer:
        t=time.perf_counter_ns();kem_public=recipient.generate_keypair();kem_keygen=time.perf_counter_ns()-t
        t=time.perf_counter_ns();sig_public=signer.generate_keypair();sig_keygen=time.perf_counter_ns()-t
        t=time.perf_counter_ns();kem_cipher,shared=sender.encap_secret(kem_public);encap=time.perf_counter_ns()-t
        key=HKDF(algorithm=SHA256(),length=32,salt=None,info=b'fedecg-subset_reference-v1').derive(shared)
        nonce=os.urandom(12);aad=b'apqfl-level-'+str(level).encode()
        t=time.perf_counter_ns();cipher=AESGCM(key).encrypt(nonce,payload,aad);encrypt=time.perf_counter_ns()-t
        message=aad+kem_cipher+nonce+cipher
        t=time.perf_counter_ns();signature=signer.sign(message);sign=time.perf_counter_ns()-t
        t=time.perf_counter_ns();valid=signer.verify(message,signature,sig_public);verify=time.perf_counter_ns()-t
        if not valid:raise RuntimeError('Valid signature rejected')
        if negative_tests:
            altered=message[:-1]+bytes([message[-1]^1])
            assert not signer.verify(altered,signature,sig_public), 'Tampered message accepted'
        t=time.perf_counter_ns();recovered=recipient.decap_secret(kem_cipher);decap=time.perf_counter_ns()-t
        recovered_key=HKDF(algorithm=SHA256(),length=32,salt=None,info=b'fedecg-subset_reference-v1').derive(recovered)
        t=time.perf_counter_ns();plain=AESGCM(recovered_key).decrypt(nonce,cipher,aad);decrypt=time.perf_counter_ns()-t
        if plain!=payload:raise RuntimeError('Round-trip changed model payload')
        if negative_tests:
            from cryptography.exceptions import InvalidTag
            try:
                AESGCM(recovered_key).decrypt(nonce,cipher[:-1]+bytes([cipher[-1]^1]),aad)
            except InvalidTag:pass
            else:raise AssertionError('AEAD accepted tampered ciphertext')
    elapsed=(time.perf_counter_ns()-begin)/1e6
    raw={'level':int(level),'payload_bytes':len(payload),'kem_public_bytes':len(kem_public),
         'signature_public_bytes':len(sig_public),'kem_cipher_bytes':len(kem_cipher),'signature_bytes':len(signature),
         'nonce_bytes':len(nonce),'aead_expansion_bytes':len(cipher)-len(payload)}
    raw['extra_bytes']=sum(raw[k] for k in ['kem_public_bytes','signature_public_bytes','kem_cipher_bytes','signature_bytes','nonce_bytes','aead_expansion_bytes'])
    raw['wire_bytes']=len(payload)+raw['extra_bytes']
    raw['crypto_ms']=elapsed
    for n,v in [('kem_keygen',kem_keygen),('sig_keygen',sig_keygen),('encap',encap),('decap',decap),('sign',sign),('verify',verify),('encrypt',encrypt),('decrypt',decrypt)]:
        raw[n+'_ms']=v/1e6
    return plain,raw


def benchmark(repeats=100):
    import oqs
    rows=[]
    rng=np.random.default_rng(20261028)
    sizes=[4096,65536,164000,1048576]
    for level in LEVELS:
        roundtrip(b'warmup'*100,level,negative_tests=True)
    schedule=[(int(n),int(l),int(rep)) for n in sizes for l in LEVELS for rep in range(repeats)]
    rng.shuffle(schedule)
    for i,(size,level,rep) in enumerate(schedule):
        payload=bytes(size)
        _,row=roundtrip(payload,level)
        row['repeat']=rep;rows.append(row)
        if (i+1)%300==0:print('Crypto benchmarks',i+1,'/',len(schedule),flush=True)
    out=ROOT/'results/crypto';out.mkdir(parents=True,exist_ok=True)
    with open(out/'raw.csv','w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save_json(out/'metadata.json',{'liboqs':oqs.oqs_version(),'level_mapping':LEVELS,
        'repeats_per_level_payload':repeats,'randomized_order':True,'fresh_keys_per_update':True,
        'includes_both_public_keys_per_update':True,'aead':'AES-256-GCM, 12-byte nonce, 16-byte tag; HKDF-SHA256 derived key',
        'negative_tests':'Tampered signed message and tampered AEAD ciphertext rejected at all three levels',
        'limitations':'Host timings only; no microcontroller energy measurements or cryptographic security proof. Peer public-key trust is assumed.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repeats',type=int,default=100);a=p.parse_args();benchmark(a.repeats)
