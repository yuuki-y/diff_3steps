# diff_3steps

## DRR-to-CT Latent Encoder Design

See [`docs/latent_encoder_design.md`](docs/latent_encoder_design.md) for a proposed architecture and training strategy to encode two 256×256 DRRs into a geometry-aware latent space for CT reconstruction. It includes model details, loss design, memory-saving tactics for 3×RTX 6000 Ada GPUs, and a minimal training skeleton.

## Code

A lightweight PyTorch implementation of the proposed pipeline is provided in `drr_ct/`:

- `DrrToCtModel` (end-to-end) in [`drr_ct/pipeline.py`](drr_ct/pipeline.py)
- `DualViewEncoder` in [`drr_ct/encoder.py`](drr_ct/encoder.py)
- `EpipolarFusion` in [`drr_ct/fusion.py`](drr_ct/fusion.py)
- `LatentBottleneck` in [`drr_ct/latent.py`](drr_ct/latent.py)
- `CTDecoder` in [`drr_ct/decoder.py`](drr_ct/decoder.py)

Example forward pass:

```python
import torch
from drr_ct import DrrToCtModel, ProjectionGeometry, EpipolarMapping

model = DrrToCtModel()
batch = 2
drr_a = torch.randn(batch, 1, 256, 256)
drr_b = torch.randn(batch, 1, 256, 256)
geom_a = ProjectionGeometry(
    intrinsics=torch.eye(3).unsqueeze(0).repeat(batch, 1, 1),
    extrinsics=torch.eye(4).unsqueeze(0).repeat(batch, 1, 1),
    source_to_detector=torch.ones(batch),
    view_id="A",
)
geom_b = ProjectionGeometry(
    intrinsics=torch.eye(3).unsqueeze(0).repeat(batch, 1, 1),
    extrinsics=torch.eye(4).unsqueeze(0).repeat(batch, 1, 1),
    source_to_detector=torch.ones(batch),
    view_id="B",
)

mapping = EpipolarMapping(
    indices=torch.zeros(batch, 16 * 16, 4, dtype=torch.long),  # toy mapping at 16x16 token grid
    weights=None,
    shape_hw=(64, 64),
)

outputs = model(drr_a, drr_b, geom_a, geom_b, mapping)
loss = model.loss_fn(outputs, target_ct=torch.randn_like(outputs["ct"]))
loss.backward()
```
