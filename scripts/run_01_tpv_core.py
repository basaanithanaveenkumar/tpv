"""Prompt 1 — TriPerspectiveView demo."""
import torch
from tpv import get_device, TriPerspectiveView

device = get_device()
print(f"Device: {device}")

tpv = TriPerspectiveView(channels=64, tpv_h=100, tpv_w=100, tpv_z=8, aggregation="sum").to(device)
print("Plane shapes:", tpv.plane_shapes())

B, N = 2, 1024
points = torch.rand(B, N, 3, device=device) * 2 - 1   # uniform in [-1, 1]

feats = tpv.project(points)
print(f"Input points : {points.shape}")
print(f"Output feats : {feats.shape}")   # expect (2, 64, 1024)

loss = feats.mean()
loss.backward()
print(f"Backward OK  | plane_hw.grad norm = {tpv.plane_hw.grad.norm():.4f}")

tpv_mean = TriPerspectiveView(64, 100, 100, 8, aggregation="mean").to(device)
feats_m = tpv_mean(points)
print(f"Mean agg     : {feats_m.shape}")
print("Prompt 1 PASSED")
