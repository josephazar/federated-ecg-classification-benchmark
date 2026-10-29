import numpy as np
import torch
from fedecg.analyze import f1_from_cm
from fedecg.models import loss_fn
from fedecg.resources import select_level


def test_macro_f1_includes_unpredicted_and_absent_classes():
    cm=np.zeros((5,5),dtype=int);cm[0,0]=90;cm[1,0]=10
    assert abs(f1_from_cm(cm)-(180/190)/5)<1e-12
    assert np.allclose(f1_from_cm(np.stack([cm,cm])),f1_from_cm(cm))


def test_minimum_feasible_policy_holds_requirements_fixed():
    costs=[[1,1,1],[2,2,2],[3,3,3]]
    for required in [1,2,3]:
        for weights in [(1,1,1),(1,0,0),(0,1,0),(0,0,1)]:
            assert select_level(required,costs,weights)==required
