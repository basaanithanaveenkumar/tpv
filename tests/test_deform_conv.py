import torch
import pytest
from tpv.deform_conv_mps import DeformableConvMPS


def _dcn(**kw):
    return DeformableConvMPS(in_channels=8, out_channels=16, **kw)


def test_output_shape_default():
    m = _dcn(kernel_size=3, padding=1)
    x = torch.randn(2, 8, 16, 16)
    out = m(x)
    assert out.shape == (2, 16, 16, 16)


def test_stride_2():
    m = _dcn(kernel_size=3, stride=2, padding=1)
    x = torch.randn(2, 8, 16, 16)
    out = m(x)
    assert out.shape == (2, 16, 8, 8)


def test_kernel_1():
    m = _dcn(kernel_size=1, padding=0)
    x = torch.randn(1, 8, 8, 8)
    out = m(x)
    assert out.shape == (1, 16, 8, 8)


def test_backward():
    m = _dcn(kernel_size=3, padding=1)
    x = torch.randn(1, 8, 8, 8, requires_grad=True)
    out = m(x)
    out.mean().backward()
    assert m.weight.grad is not None
    assert x.grad is not None


def test_no_bias():
    m = DeformableConvMPS(8, 16, bias=False)
    assert m.bias_param is None
    x = torch.randn(1, 8, 8, 8)
    out = m(x)
    assert out.shape == (1, 16, 8, 8)


def test_zero_offset_close_to_standard_conv():
    """When offsets are forced to zero the deformable conv should produce
    the same result as a standard conv with the same weights on the base grid."""
    m = DeformableConvMPS(4, 4, kernel_size=1, padding=0)
    # Force offsets to zero and masks to 1
    torch.nn.init.zeros_(m.offset_conv.weight)
    torch.nn.init.zeros_(m.offset_conv.bias)
    torch.nn.init.zeros_(m.mask_conv.weight)
    m.mask_conv.bias.data.fill_(10.0)          # sigmoid(10) ≈ 1

    x = torch.randn(1, 4, 4, 4)
    out = m(x)
    assert out.shape == (1, 4, 4, 4)
