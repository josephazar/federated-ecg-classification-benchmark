import numpy as np
import torch

from fedecg.integration import encode, decode


def test_model_serialization_preserves_shape_and_values():
    state = {'a': torch.tensor([[1., -3.], [0., 1e-12]]), 'b': torch.tensor([2., 4.])}
    payload = encode(state)
    restored = decode(payload, state)
    assert len(payload) == sum(v.numel() * v.element_size() for v in state.values())
    for k in state:
        assert np.array_equal(restored[k].numpy(), state[k].numpy())
        assert restored[k].shape == state[k].shape


def test_decoder_rejects_unaccounted_bytes():
    import pytest
    state = {'a': torch.tensor([1., 2.])}
    with pytest.raises(ValueError, match='Unexpected serialized'):
        decode(encode(state) + b'junk', state)


def test_actual_protection_accounting_includes_both_aad_and_payload():
    from fedecg.integration import transmit
    payload = b'check serialized model boundary' * 7
    for level, added in [(1,5341),(2,7574),(3,10396)]:
        recovered, total, extra, used = transmit(payload, 'adaptive', level)
        assert recovered == payload
        assert total == len(payload) + added and extra == added and used == level
    recovered, total, extra, used = transmit(payload, 'none', 3)
    assert recovered == payload and total == len(payload) and extra == used == 0


def test_transport_rejects_changed_payload(monkeypatch):
    import pytest
    import fedecg.crypto
    from fedecg.integration import transmit
    monkeypatch.setattr(fedecg.crypto,'roundtrip',lambda payload,level:(b'changed',{}))
    with pytest.raises(ValueError,match='altered a model payload'):
        transmit(b'original','fixed_L3',3)
