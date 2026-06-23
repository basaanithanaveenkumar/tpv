import torch
from tpv.tpv4 import TPV4


def _model():
    return TPV4(channels=8, tpv_h=16, tpv_w=16, tpv_z=4)


def test_project_shape():
    m = _model()
    pts = torch.rand(2, 32, 3) * 2 - 1
    out = m.project(pts)
    assert out.shape == (2, 8, 32)


def test_forward_alias():
    m = _model()
    pts = torch.rand(1, 16, 3) * 2 - 1
    assert m(pts).shape == m.project(pts).shape


def test_integrate_does_not_break_forward():
    m = _model()
    pts = torch.rand(1, 8, 3) * 2 - 1
    feats = torch.randn(1, 8, 8)
    m.integrate_points(pts, feats, alpha=0.1)
    out = m(torch.rand(1, 16, 3) * 2 - 1)
    assert out.shape == (1, 8, 16)


def test_flops_scaling():
    m = _model()
    assert m.flops_per_query(1000) > m.flops_per_query(100)
