"""Prompt 3 — TPV4 minimal demo (project + integrate_points)."""
import torch
from tpv import get_device, TPV4

device = get_device()
print(f"Device: {device}")

model = TPV4(channels=32, tpv_h=32, tpv_w=32, tpv_z=8, aggregation="sum").to(device)
print(f"Parameters   : {model.n_params():,}")
print(f"FLOPs/query  : {model.flops_per_query(1):,}")

B, N = 2, 256
points = torch.rand(B, N, 3, device=device) * 2 - 1

# ── Forward pass ─────────────────────────────────────────────────────────
feats = model.project(points)
print(f"project()    : {points.shape} → {feats.shape}")

# ── Backward ─────────────────────────────────────────────────────────────
feats.mean().backward()
print(f"Gradient OK  | hw.grad norm = {model.plane_hw.grad.norm():.4f}")

# ── integrate_points ─────────────────────────────────────────────────────
# New observations at a small subset of random points
new_pts = torch.rand(1, 8, 3, device=device) * 2 - 1
new_feats = torch.randn(1, 32, 8, device=device)

model.zero_grad()
model.integrate_points(new_pts, new_feats, alpha=0.2)
print("integrate_points(): planes updated in-place (no grad)")

# Verify the integrated planes still support forward+backward
feats2 = model(points)
feats2.mean().backward()
print(f"Post-integrate forward OK | feats2 {feats2.shape}")
print("Prompt 3 PASSED")
