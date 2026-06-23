import torch
import pytest
from tpv.tpv10 import TPV10Encoder


def _model(**kw):
    return TPV10Encoder(channels=8, tpv_h=8, tpv_w=8, tpv_z=4, num_classes=5, **kw)


def test_output_shapes():
    m = _model(use_checkpoint=False)
    pts = torch.rand(2, 16, 3) * 2 - 1
    out = m(pts)
    assert out["features"].shape == (2, 8, 16)
    assert out["aux_hw"].shape == (1, 5, 8, 8)
    assert out["aux_zh"].shape == (1, 5, 4, 8)
    assert out["aux_wz"].shape == (1, 5, 8, 4)


def test_backward():
    m = _model(use_checkpoint=False)
    pts = torch.rand(1, 8, 3) * 2 - 1
    out = m(pts)
    total = sum(v.mean() for v in out.values())
    total.backward()
    assert m.plane_hw.grad is not None


def test_dynamic_resolution():
    m = _model(use_checkpoint=False)
    pts = torch.rand(1, 8, 3) * 2 - 1
    out = m(pts, target_h=16, target_w=16)
    assert out["features"].shape == (1, 8, 8)
    assert out["aux_hw"].shape[2] == 16


@pytest.mark.parametrize("agg", ["adaptive", "sum", "mean"])
def test_aggregation_modes(agg):
    m = _model(use_checkpoint=False, aggregation=agg)
    pts = torch.rand(1, 4, 3) * 2 - 1
    out = m(pts)
    assert out["features"].shape == (1, 8, 4)
