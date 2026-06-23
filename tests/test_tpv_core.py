import torch
import pytest
from tpv.tpv_core import TriPerspectiveView


def _tpv(agg="sum"):
    return TriPerspectiveView(channels=8, tpv_h=16, tpv_w=16, tpv_z=4, aggregation=agg)


def test_output_shape():
    tpv = _tpv()
    pts = torch.rand(2, 64, 3) * 2 - 1
    out = tpv(pts)
    assert out.shape == (2, 8, 64)


def test_sum_vs_mean():
    tpv_s = _tpv("sum")
    tpv_m = _tpv("mean")
    pts = torch.rand(1, 32, 3) * 2 - 1
    with torch.no_grad():
        tpv_m.plane_hw.data.copy_(tpv_s.plane_hw.data)
        tpv_m.plane_zh.data.copy_(tpv_s.plane_zh.data)
        tpv_m.plane_wz.data.copy_(tpv_s.plane_wz.data)
    torch.testing.assert_close(tpv_s(pts) / 3, tpv_m(pts))


def test_gradients_flow():
    tpv = _tpv()
    pts = torch.rand(1, 16, 3, requires_grad=False) * 2 - 1
    out = tpv(pts)
    out.mean().backward()
    for name, p in tpv.named_parameters():
        assert p.grad is not None, f"No grad for {name}"
        assert p.grad.norm() > 0


def test_batched():
    tpv = _tpv()
    for B in [1, 4]:
        out = tpv(torch.rand(B, 10, 3) * 2 - 1)
        assert out.shape[0] == B
