"""Prompt 2 — BEVToTPV converter demo + head swap."""
import torch
from tpv import get_device, BEVToTPV
from tpv.bev_to_tpv import BEVDetectionHead, TPVDetectionHead

device = get_device()
print(f"Device: {device}")

B, C_in, H, W = 2, 128, 50, 50
bev = torch.randn(B, C_in, H, W, device=device)

converter = BEVToTPV(
    in_channels=C_in, out_channels=64,
    tpv_h=50, tpv_w=50, tpv_z=8,
).to(device)

plane_hw, plane_zh, plane_wz = converter(bev)
print(f"BEV input   : {bev.shape}")
print(f"plane_hw    : {plane_hw.shape}")   # (2, 64, 50, 50)
print(f"plane_zh    : {plane_zh.shape}")   # (2, 64, 8, 50)
print(f"plane_wz    : {plane_wz.shape}")   # (2, 64, 50, 8)

# ── Head swap demo ────────────────────────────────────────────────────────
num_classes = 10

bev_head = BEVDetectionHead(C_in, num_classes).to(device)
bev_logits = bev_head(bev)
print(f"\nBEV head logits : {bev_logits.shape}")

tpv_head = TPVDetectionHead(64, num_classes, 50, 50, 8).to(device)
tpv_logits = tpv_head(plane_hw, plane_zh, plane_wz)
print(f"TPV head logits : {tpv_logits.shape}")

# Verify backward through the entire BEV→TPV→head pipeline
loss = tpv_logits.mean()
loss.backward()
print("Backward through converter + head: OK")
print("Prompt 2 PASSED")
