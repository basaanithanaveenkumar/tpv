import torch
from tpv.bev_to_tpv import BEVToTPV, TPVDetectionHead


def _converter():
    return BEVToTPV(in_channels=16, out_channels=8, tpv_h=16, tpv_w=16, tpv_z=4)


def test_plane_shapes():
    m = _converter()
    bev = torch.randn(2, 16, 20, 20)
    hw, zh, wz = m(bev)
    assert hw.shape == (2, 8, 16, 16)
    assert zh.shape == (2, 8, 4, 16)
    assert wz.shape == (2, 8, 16, 4)


def test_backward():
    m = _converter()
    bev = torch.randn(2, 16, 20, 20)
    hw, zh, wz = m(bev)
    (hw.mean() + zh.mean() + wz.mean()).backward()


def test_head_swap():
    m = _converter()
    head = TPVDetectionHead(8, 5, 16, 16, 4)
    bev = torch.randn(1, 16, 20, 20)
    hw, zh, wz = m(bev)
    logits = head(hw, zh, wz)
    assert logits.shape == (1, 5)
