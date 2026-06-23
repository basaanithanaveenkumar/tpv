# TPV — Tri-Perspective View (pure PyTorch, macOS MPS-compatible)

Six progressively advanced implementations of Tri-Perspective View (TPV)
representations. No mmcv / mmseg / CUDA extensions required — runs on
**CPU, CUDA, and Apple Silicon MPS**.

## Setup (uv)

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Run each prompt

> On macOS, set the MPS fallback flag so PyTorch CPU-falls-back for ops
> not yet in the MPS kernel (e.g. `grid_sampler_2d_backward`).

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1

python scripts/run_01_tpv_core.py        # Prompt 1 — TriPerspectiveView class
python scripts/run_02_bev_to_tpv.py      # Prompt 2 — BEV → TPV converter + head swap
python scripts/run_03_tpv4.py            # Prompt 3 — TPV4 minimal (grid_sample + scatter)
python scripts/run_04_tpv10.py           # Prompt 4 — TPV10 encoder (10 improvements)
python scripts/run_05_compare.py         # Prompt 5 — TPV4 vs TPV10 technical comparison
python scripts/run_06_deform_conv_mps.py # Prompt 6 — Deformable conv (MPS-safe)
```

## Tests

```bash
pytest -q     # 23 tests, ~1.3 s
```

## Module map

| Prompt | File | Description |
|--------|------|-------------|
| 1 | `src/tpv/tpv_core.py` | `TriPerspectiveView` — 3 orthogonal learnable planes, bilinear `F.grid_sample` projection, sum/mean aggregation |
| 2 | `src/tpv/bev_to_tpv.py` | `BEVToTPV` — Conv-based BEV→TPV converter; `TPVDetectionHead` drop-in for a BEV head |
| 3 | `src/tpv/tpv4.py` | `TPV4` — minimal single-file TPV with `project()` and `integrate_points()` bilinear splatting |
| 4 | `src/tpv/tpv10.py` | `TPV10Encoder` — 10 improvements: multi-scale FPN, deformable attn, adaptive fusion, dynamic resolution, aux losses, cross-plane attn, sinusoidal PE, depth-aware agg, gradient checkpointing, LayerNorm |
| 5 | `scripts/run_05_compare.py` | Printed comparison: architecture, FLOPs, params, latency, use-cases, complexity |
| 6 | `src/tpv/deform_conv_mps.py` | `DeformableConvMPS` — deformable conv via `F.grid_sample` + learned offsets; no `torchvision.ops` required |

## MPS note

PyTorch 2.x does not yet implement `grid_sampler_2d_backward` in the MPS
kernel. Forward inference runs fully on MPS. For training, set:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

This makes only the backward of `grid_sample` fall back to CPU; all other
ops (convolutions, attention, layer norm) run on MPS as normal.
