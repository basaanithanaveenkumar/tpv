"""Prompt 6 — DeformableConvMPS demo (CPU + MPS, no torchvision required)."""
import torch
import torch.nn as nn
from tpv import get_device, DeformableConvMPS

device = get_device()
print(f"Device: {device}")

# ── Basic forward / backward ─────────────────────────────────────────────
dcn = DeformableConvMPS(in_channels=32, out_channels=64, kernel_size=3,
                        stride=1, padding=1).to(device)
print(dcn)
print(f"Parameters: {sum(p.numel() for p in dcn.parameters()):,}")

B, C, H, W = 2, 32, 32, 32
x = torch.randn(B, C, H, W, device=device)
out = dcn(x)
print(f"Input  : {x.shape}")
print(f"Output : {out.shape}")   # (2, 64, 32, 32)
assert out.shape == (B, 64, H, W), f"Wrong output shape: {out.shape}"

loss = out.mean()
loss.backward()
print(f"Backward OK | weight.grad norm = {dcn.weight.grad.norm():.4f}")

# ── Stride and padding variants ──────────────────────────────────────────
dcn_s2 = DeformableConvMPS(32, 64, kernel_size=3, stride=2, padding=1).to(device)
out_s2 = dcn_s2(x)
print(f"Stride-2 output: {out_s2.shape}")   # (2, 64, 16, 16)
assert out_s2.shape == (B, 64, H // 2, W // 2)

# ── Compare with standard Conv2d on same input ───────────────────────────
std_conv = nn.Conv2d(32, 64, 3, padding=1).to(device)
out_std  = std_conv(x)
print(f"Standard conv2d output: {out_std.shape} (same spatial size as deformable)")

# ── Both CPU and MPS run ─────────────────────────────────────────────────
print("\nRunning on CPU:")
dcn_cpu = DeformableConvMPS(8, 16, kernel_size=3, padding=1)
x_cpu = torch.randn(1, 8, 16, 16)
print(f"  CPU output: {dcn_cpu(x_cpu).shape}")

if torch.backends.mps.is_available():
    print("Running on MPS:")
    dcn_mps = DeformableConvMPS(8, 16, kernel_size=3, padding=1).to("mps")
    x_mps = x_cpu.to("mps")
    print(f"  MPS output: {dcn_mps(x_mps).shape}")
else:
    print("MPS not available — skipping MPS check")

# ── Plug it into a small network ─────────────────────────────────────────
class SmallNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = DeformableConvMPS(3, 32, 3, padding=1)
        self.layer2 = DeformableConvMPS(32, 64, 3, stride=2, padding=1)
        self.head   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.fc(self.head(x).flatten(1))

net = SmallNet().to(device)
img = torch.randn(2, 3, 64, 64, device=device)
logits = net(img)
print(f"\nSmallNet logits: {logits.shape}")
logits.mean().backward()
print("SmallNet backward OK")
print("Prompt 6 PASSED")
