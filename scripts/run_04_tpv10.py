"""Prompt 4 — TPV10 encoder demo with all 10 improvements."""
import torch
from tpv import get_device, TPV10Encoder

device = get_device()
print(f"Device: {device}")

model = TPV10Encoder(
    channels=64,
    tpv_h=32, tpv_w=32, tpv_z=8,
    num_classes=20,
    use_checkpoint=True,
    aggregation="adaptive",
).to(device)
print(f"TPV10 parameters: {model.n_params():,}")

B, N = 2, 512
points = torch.rand(B, N, 3, device=device) * 2 - 1

# ── Standard forward ──────────────────────────────────────────────────────
out = model(points)
print(f"features : {out['features'].shape}")   # (2, 64, 512)
print(f"aux_hw   : {out['aux_hw'].shape}")      # (1, 20, 32, 32)
print(f"aux_zh   : {out['aux_zh'].shape}")      # (1, 20, 8, 32)
print(f"aux_wz   : {out['aux_wz'].shape}")      # (1, 20, 32, 8)

# ── Backward (all three aux heads + main features) ───────────────────────
total_loss = (
    out["features"].mean()
    + out["aux_hw"].mean()
    + out["aux_zh"].mean()
    + out["aux_wz"].mean()
)
total_loss.backward()
print(f"Backward OK | plane_hw.grad norm = {model.plane_hw.grad.norm():.4f}")

# ── Improvement 4: dynamic resolution ────────────────────────────────────
model.zero_grad()
out_hires = model(points, target_h=64, target_w=64)
print(f"\nDynamic resolution (64×64):")
print(f"  features : {out_hires['features'].shape}")
print(f"  aux_hw   : {out_hires['aux_hw'].shape}")
out_hires["features"].mean().backward()
print("Dynamic resolution backward OK")
print("Prompt 4 PASSED")
