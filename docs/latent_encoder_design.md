# DRR-Based Latent Encoder for CT Reconstruction

This document proposes an encoder optimized for extracting maximal information from two 256×256 DRRs to support CT reconstruction. Training assumes paired DRR–CT data; inference only receives the two DRRs. The design targets a 3×RTX 6000 Ada node and emphasizes memory safety and throughput.

## Goals and Constraints
- **Inputs (inference)**: Two DRRs at 256×256 resolution with associated projection geometry (intrinsics/extrinsics).
- **Outputs**: Compact 3D latent volume that preserves anatomy, texture, and view geometry, suitable for a downstream CT reconstructor (e.g., diffusion-based decoder or deterministic UNet).
- **Hardware**: 3×RTX 6000 Ada (48 GB each). Favor mixed precision, gradient checkpointing, activation offload, and sharded training to avoid OOM.
- **Performance target**: Latent sufficiently informative to enable <1 mm MAE and >35 dB PSNR on reconstructed CTs (dataset-dependent).

## High-Level Pipeline
1. **Dual-view encoder** extracts multi-scale 2D tokens per DRR (shared weights).
2. **Geometry-aware fusion** performs cross-view attention along epipolar lines and fuses into a canonical 3D latent grid (e.g., 16³ cells with 128–256 channels).
3. **Latent regularization** via variational bottleneck + contrastive priors to ensure compactness and view consistency.
4. **CT reconstructor** consumes the latent to predict a full CT volume (can be trained jointly or in a two-stage manner).

## Encoder Architecture
- **Backbone**: Hierarchical vision transformer (e.g., Swin/ConvNeXt hybrid) with patch size 4 and windowed attention; channels: 96→192→384→768. Shared weights for both DRRs to enforce symmetry.
- **Positional & geometry tokens**: Add sinusoidal 2D position encoding plus **camera-aware tokens** derived from projection matrices (ray direction, source-to-detector distance, view angle) injected at each stage.
- **Multi-scale feature pyramid**: Extract {1/4, 1/8, 1/16} resolutions. Each scale passes through lightweight neck (depthwise conv + SE block) to stabilize intensity variance across DRRs.

### Geometry-Aware Cross-View Fusion
- **Epipolar transformer**: For each pixel token in view A, attend to tokens in view B whose epipolar lines intersect; implemented via sparse attention using precomputed epipolar indices (limit to top-k=32 tokens for memory).
- **Cross-view gated attention**: Combine self-attention with cross-attention weights; gating scalar learned per head to balance inter-/intra-view cues.
- **Tri-planar volumization**: Project fused 2D tokens into a canonical 3D grid via learned ray-marching MLP that maps (u,v,ray_dir) to grid coordinates; aggregate with splatting + learned confidence weights.
- **Latent grid**: 16×16×16 grid with 192 channels; apply 3 blocks of 3D Swin-style attention + residual depthwise 3D convs. Optionally compress to 128 channels with a variational bottleneck (μ, σ).

## Latent Regularization and Losses
- **CT reconstruction loss**: L1 + SSIM on CT volume; add gradient loss on HU differences.
- **KL divergence**: Variational prior on latent grid to encourage compactness and enable sampling-based refinement.
- **Cross-view consistency**: InfoNCE between view-specific pooled tokens and latent grid projections to force geometry-consistent encoding.
- **DRR cycle loss (optional)**: Render DRRs from predicted CT and enforce L1/SSIM with inputs to stabilize geometry cues.
- **Adversarial patch loss**: Lightweight discriminator on CT patches to sharpen texture without large memory overhead.

## Training Strategy
1. **Stage 1 (supervised joint)**: Train encoder + CT reconstructor end-to-end with reconstruction + KL + consistency losses. Use mixed precision and gradient checkpointing on attention blocks.
2. **Stage 2 (latent prior tightening)**: Freeze reconstructor; continue training encoder + bottleneck with stronger KL/InfoNCE weighting to maximize information density.
3. **Stage 3 (distillation / self-supervision)**: Add masked DRR modeling on encoder outputs and distill from a heavier teacher (if available) to improve robustness to noise/occlusions.
4. **Fine-tuning**: Small LR, disable heavy augmentations, enable EMA weights for stability.

## Data and Augmentation
- Normalize DRRs per-scan (z-score); clip HU of CTs to clinical window before loss.
- Augmentations: random log-intensity scaling, slight detector shifts/rotations consistent with physical acquisition, cutout on DRRs to force reliance on both views, MixUp across slices sparingly.
- Geometry jitter: small perturbations to projection matrices during training; feed exact matrices during inference.

## Memory and Throughput Tactics (3×RTX 6000 Ada)
- **DDP with activation checkpointing** on transformer blocks; shard optimizer states with ZeRO-2/3 (DeepSpeed/FSDP).
- **Grad accumulation** to simulate larger batches (effective bs 16–32 pairs) while fitting in memory.
- **Sequence length control**: Limit tokens via 4×4 patching and token pruning after early layers to keep attention manageable (~4k tokens/view).
- **Mixed precision** (bf16 if supported) and fused kernels for attention/conv.
- **Profiling hooks** to auto-reduce epipolar top-k or latent channels if OOM is detected.

## CT Reconstructor Options
- **Deterministic UNet**: 3D UNet receiving latent grid (upsampled) + camera-aware embeddings; simpler and lighter.
- **Diffusion decoder**: Conditioned on latent grid; use 3D UNet backbone with classifier-free guidance; train with 100–250 steps and sample with 20–30 steps for inference.
- **Implicit neural representation head** (optional): MLP that queries continuous coordinates with latent grid context for super-resolution CT exports.

## Evaluation
- Metrics: MAE/PSNR/SSIM on HU-normalized CTs; 3D Dice/HD95 on organ masks if available.
- Ablations: remove epipolar attention, vary latent grid size, compare deterministic vs diffusion decoder.
- Robustness: evaluate under synthetic noise, partial occlusion, and mild geometric miscalibration.

## Minimal Training Skeleton (pseudo-code)
```python
enc = DualViewEncoder()
agg = EpipolarFusion()
latent_bottleneck = VAE3D()
recon = CTDecoder()

for drra, drrb, ct, geom in loader:
    fa = enc(drra, geom.a)
    fb = enc(drrb, geom.b)
    grid = agg(fa, fb, geom)
    mu, logvar, z = latent_bottleneck(grid)
    ct_pred = recon(z, geom)

    loss = l1(ct_pred, ct) + ssim(ct_pred, ct) \
         + kl(mu, logvar) * β \
         + info_nce(fa, fb, z) * λ
    if cycle:
         loss += drr_render_loss(ct_pred, [geom.a, geom.b]) * γ
    loss.backward()
    optim.step()
```

This architecture is optimized to extract maximum geometric and textural information from two DRRs, populate a compact latent space, and enable high-quality CT reconstruction under strict GPU memory constraints.
