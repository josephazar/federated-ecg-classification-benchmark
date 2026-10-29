import numpy as np
import pytest
import torch

from fedecg.data import load_split, build_clients, patient_id
from fedecg.models import class_weights, loss_fn
from fedecg.crypto import roundtrip


def test_actual_patient_and_sample_separation():
    d,tr,va,te=load_split('strict')
    groups=[{patient_id(x) for x in d['record'][ix]} for ix in [tr,va,te]]
    assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    assert 201 not in d['record'][tr] and 202 in d['record'][te]
    assert not (set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te))
    assert np.isfinite(d['x']).all() and np.isfinite(d['rr']).all()


def test_data_volume_control_keeps_identical_full_evaluation_sets():
    d,tr,va,te=load_split('strict')
    _,small,small_val,small_test=load_split('strict',6396,test_size=0)
    assert len(small)==6396 and set(small).issubset(set(tr))
    assert np.array_equal(va,small_val) and np.array_equal(te,small_test)


@pytest.mark.parametrize('partition',['iid','patient','dirichlet'])
def test_clients_cover_training_once_and_reproduce(partition):
    d,tr,_,_=load_split('strict')
    a=build_clients(d,tr,partition,11,10,.1)
    b=build_clients(d,tr,partition,11,10,.1)
    assert np.array_equal(np.sort(np.concatenate(a)),np.sort(tr))
    assert len(set(np.concatenate(a)))==len(tr)
    assert all(np.array_equal(x,y) for x,y in zip(a,b))


@pytest.mark.parametrize('kind',['ce','weighted','balanced','focal'])
def test_losses_remain_finite_with_absent_and_tiny_classes(kind):
    y=torch.tensor([0]*120+[1]*3+[4])
    weights=class_weights(y,kind)
    z=torch.zeros(len(y),5,requires_grad=True)
    loss=loss_fn(z,y,kind,weights)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
    assert weights[2]==0 and weights[3]==0


@pytest.mark.parametrize('level',[1,2,3])
def test_real_protection_roundtrip_and_tamper_rejection(level):
    payload=np.arange(1000,dtype=np.float32).tobytes()
    recovered,m=roundtrip(payload,level,negative_tests=True)
    assert recovered==payload
    assert m['wire_bytes']==len(payload)+m['extra_bytes'] and m['extra_bytes']>0
