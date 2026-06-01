"""
models/dc3swt.py — DC3SWT: Deformable C3 Swin Transformer Block
================================================================
Core attention module for DA-YOLO (Deformable Attention YOLO).
DC3SWT: deformable attention replaces W-MSA+SW-MSA, see arxiv:2010.04159

Motivation
----------
The standard Swin Transformer block used in C3SWT partitions feature maps into
fixed spatial windows (e.g. 7×7 or 8×8) and computes multi-head self-attention
only within each window. This assumption breaks down for small objects in remote
sensing imagery, which:

  • Rarely align with pre-defined window boundaries
  • Appear at arbitrary locations and scales
  • Form dense clusters that span multiple windows simultaneously

DC3SWT replaces the entire W-MSA + SW-MSA mechanism with Multi-Scale Deformable
Attention (MSDA, Zhu et al. 2020, arxiv:2010.04159). For each query location, the
network *learns* where to look by predicting M×K sampling offsets. This makes the
receptive field adaptive to object location and scale by design.

Implementation constraints (from design doc)
-------------------------------------------
1. No custom CUDA extensions — pure PyTorch only (F.grid_sample for sampling)
2. Identical I/O shapes to C3SWT — drop-in replacement with no YAML surgery
3. n_heads = max(1, c_ // 32) — guaranteed to divide d_model evenly
4. Runtime clamp: n_points = min(n_points, H*W) for very small feature maps
5. Every modified block carries: # DC3SWT: deformable attention replaces
   W-MSA+SW-MSA, see arxiv:2010.04159

Ablation position
-----------------
Baseline → +C3SWT (paper) → +DC3SWT (this) → full model (DC3SWT + BiFPN + CoordAtt)
Each row adds one component with a clear story.
"""

# DC3SWT: deformable attention replaces W-MSA+SW-MSA, see arxiv:2010.04159

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# timm >= 0.9 moved to timm.layers; fall back for older installs
try:
    from timm.layers import DropPath, trunc_normal_
except ImportError:
    from timm.models.layers import DropPath, trunc_normal_

from models.common import Conv, Bottleneck


# ── Utility ──────────────────────────────────────────────────────────────────

def _build_ref_points(H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Build normalized reference point grid for MSDA.

    Returns:
        Tensor of shape (1, H*W, 2) in [0, 1] range — (x, y) ordering.
    """
    ref_y = (torch.arange(H, device=device, dtype=dtype) + 0.5) / H
    ref_x = (torch.arange(W, device=device, dtype=dtype) + 0.5) / W
    ref_y, ref_x = torch.meshgrid(ref_y, ref_x, indexing="ij")
    ref = torch.stack([ref_x, ref_y], dim=-1)   # (H, W, 2) — x first for grid_sample
    return ref.reshape(1, H * W, 2)             # (1, N, 2)


# ── Core Deformable Attention Block ──────────────────────────────────────────

class MSDABlock(nn.Module):
    """
    Single-Level Multi-Scale Deformable Attention Block (pure PyTorch).

    # DC3SWT: deformable attention replaces W-MSA+SW-MSA, see arxiv:2010.04159

    Unlike the original MSDeformAttn (which uses a custom CUDA kernel), this
    implementation uses F.grid_sample for deformable feature sampling — fully
    compatible with any PyTorch-supported device (CPU / CUDA / MPS).

    For each query token at position p_q:
        1. Predict M × K sampling offsets  Δp_{mk}  (learned, bounded by tanh)
        2. Sample value features at  p_q + Δp_{mk}  via bilinear grid_sample
        3. Weight sampled features by attention scores (softmax over K per head)
        4. Aggregate and project to output

    Args:
        d_model   (int): Channel dimension.  Must be divisible by n_heads.
        n_heads   (int): Number of attention heads.  Use max(1, d_model // 32).
        n_points  (int): Sampling points per head (K).  Default 4.
        drop      (float): Dropout on internal projections.
        drop_path (float): Stochastic depth rate.
        offset_scale (float): Tanh-clamp scale for offsets (fraction of feature map).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_points: int = 4,
        drop: float = 0.0,
        drop_path: float = 0.0,
        offset_scale: float = 0.5,
    ):
        # DC3SWT: deformable attention replaces W-MSA+SW-MSA, see arxiv:2010.04159
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
        )
        self.d_model      = d_model
        self.n_heads      = n_heads
        self.n_points     = n_points          # max sampling points (clamped at runtime)
        self.head_dim     = d_model // n_heads
        self.offset_scale = offset_scale

        # ── Projections ──────────────────────────────────────────────────────
        self.query_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        # Offset predictor: for each head and point predict (Δx, Δy)
        self.offset_proj = nn.Linear(d_model, n_heads * n_points * 2)

        # Attention weight predictor: softmax over K points per head
        self.attn_proj   = nn.Linear(d_model, n_heads * n_points)

        # Output projection
        self.out_proj    = nn.Linear(d_model, d_model)

        # ── Normalization & Regularization ────────────────────────────────────
        self.norm      = nn.LayerNorm(d_model)
        self.drop      = nn.Dropout(drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        """
        Initialize so the network starts with uniform attention over a regular grid.
        Offsets start at zero → uniform sampling at reference points.
        Attention weights start at 1/n_points → uniform distribution.
        """
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.query_proj.bias, 0.0)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

        # Offsets start at zero → model starts from uniform reference sampling
        nn.init.constant_(self.offset_proj.weight, 0.0)
        nn.init.constant_(self.offset_proj.bias, 0.0)

        # Attention weights: uniform init → log(1/K) for each point pre-softmax
        nn.init.constant_(self.attn_proj.weight, 0.0)
        nn.init.constant_(self.attn_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        # DC3SWT: deformable attention replaces W-MSA+SW-MSA, see arxiv:2010.04159

        Args:
            x: (B, N, C)  where N = H × W
            H, W: spatial dimensions of the feature map

        Returns:
            (B, N, C) — same shape as input
        """
        B, N, C = x.shape
        assert N == H * W, f"Expected N={H*W}, got {N}"

        # ── Constraint 4: clamp n_points to avoid degenerate sampling ─────────
        # When H*W < n_points (very small feature maps at deep layers), we cannot
        # have more sampling points than available spatial locations.
        n_pts = min(self.n_points, N)

        shortcut = x

        # ── Pre-LayerNorm (Pre-LN is more stable for Transformer-style blocks) ─
        x = self.norm(x)

        # ── 1. Query and value projections ─────────────────────────────────────
        q = self.query_proj(x)   # (B, N, C)
        v = self.value_proj(x)   # (B, N, C)

        # ── 2. Predict sampling offsets ────────────────────────────────────────
        # Full offset tensor (B, N, n_heads * n_points * 2), then clip to n_pts
        offsets_full = self.offset_proj(q)
        offsets = offsets_full[..., : self.n_heads * n_pts * 2]   # (B, N, M*K*2)
        offsets = offsets.view(B, N, self.n_heads, n_pts, 2)

        # tanh bounds offsets to ±offset_scale of the feature map size
        offsets = torch.tanh(offsets) * self.offset_scale          # bounded in (-0.5, 0.5)

        # ── 3. Reference points (normalized [0,1] grid, x-first) ──────────────
        ref = _build_ref_points(H, W, x.device, x.dtype)           # (1, N, 2)
        ref = ref.view(1, N, 1, 1, 2).expand(B, -1, self.n_heads, n_pts, -1)

        # ── 4. Absolute sampling locations in [-1,1] for F.grid_sample ────────
        # ref in [0,1] → convert: loc = 2*(ref + offset) - 1
        sampling_locs = 2.0 * (ref + offsets) - 1.0   # (B, N, n_heads, n_pts, 2)

        # ── 5. Sample value features at deformable locations ──────────────────
        # Reshape v to (B, n_heads, head_dim, H, W) for per-head sampling
        v_spatial = v.reshape(B, H, W, C).permute(0, 3, 1, 2)           # (B, C, H, W)
        v_per_head = v_spatial.reshape(B, self.n_heads, self.head_dim, H, W)

        sampled_heads = []
        for m in range(self.n_heads):
            # locs for head m: (B, N, n_pts, 2) → (B, N*n_pts, 1, 2) for grid_sample
            locs_m = sampling_locs[:, :, m, :, :].reshape(B, N * n_pts, 1, 2)
            feat_m = v_per_head[:, m]                                    # (B, head_dim, H, W)

            # F.grid_sample: input (B, C_h, H, W), grid (B, H_out, W_out, 2)
            # Here H_out=N*n_pts, W_out=1
            smp = F.grid_sample(
                feat_m, locs_m,
                mode="bilinear", padding_mode="zeros", align_corners=False
            )   # (B, head_dim, N*n_pts, 1)

            smp = smp.squeeze(-1).permute(0, 2, 1)   # (B, N*n_pts, head_dim)
            smp = smp.reshape(B, N, n_pts, self.head_dim)
            sampled_heads.append(smp)

        # Stack: (B, N, n_heads, n_pts, head_dim)
        sampled = torch.stack(sampled_heads, dim=2)

        # ── 6. Attention weights (softmax over K points per head) ──────────────
        attn_full = self.attn_proj(q)
        attn = attn_full[..., : self.n_heads * n_pts]              # (B, N, M*K)
        attn = attn.view(B, N, self.n_heads, n_pts)
        attn = F.softmax(attn, dim=-1)                             # sum-to-1 over K

        # ── 7. Weighted aggregation ────────────────────────────────────────────
        # attn:    (B, N, n_heads, n_pts, 1)
        # sampled: (B, N, n_heads, n_pts, head_dim)
        out = (sampled * attn.unsqueeze(-1)).sum(dim=3)            # (B, N, n_heads, head_dim)
        out = out.reshape(B, N, C)                                 # concat heads

        # ── 8. Output projection + dropout ────────────────────────────────────
        out = self.drop(self.out_proj(out))

        # ── 9. Residual connection with stochastic depth ───────────────────────
        return shortcut + self.drop_path(out)


# ── DC3SWT Module ─────────────────────────────────────────────────────────────

class DC3SWT(nn.Module):
    """
    DC3SWT — Deformable C3 Swin Transformer Block.

    # DC3SWT: deformable attention replaces W-MSA+SW-MSA, see arxiv:2010.04159

    Drop-in replacement for C3SWT with identical input/output shape contract.
    Replaces the fixed-window W-MSA + SW-MSA attention with Multi-Scale
    Deformable Attention (MSDA, arxiv:2010.04159), implemented in pure PyTorch.

    Architecture (same outer structure as C3SWT):
        Input  (B, c1, H, W)
          │
          ├── cv1(1×1) ──→ Bottleneck×n ──→ MSDABlock ──→ spatial_drop ──┐
          │                                                                  ├── cv3(1×1) → Output
          └── cv2(1×1) ────────────────────────────────────────────────────┘

    Differences vs C3SWT:
        C3SWT  uses SwinTransformerBlock (W-MSA + SW-MSA, fixed 8×8 windows, mask)
        DC3SWT uses MSDABlock (pure-PyTorch MSDA, learned offsets, no windows)

    Advantages of DC3SWT over C3SWT:
        • No window padding needed for non-multiple spatial sizes
        • No attention mask recomputation per input shape
        • Adaptive receptive field — offsets attend across window boundaries
        • Graceful handling of small feature maps (n_points clamped to H*W)

    Args:
        c1      (int): Input channels.
        c2      (int): Output channels.
        n       (int): Number of Bottleneck repeats.
        shortcut(bool): Bottleneck residual connections.
        g       (int): Bottleneck groups.
        e       (float): Channel expansion ratio.
        n_points(int): Sampling points per attention head (K). Default 4.
        drop    (float): MLP dropout rate.
        drop_path(float): Stochastic depth rate.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
        n_points: int = 4,
        drop: float = 0.1,
        drop_path: float = 0.1,
    ):
        # DC3SWT: deformable attention replaces W-MSA+SW-MSA, see arxiv:2010.04159
        super().__init__()
        c_ = int(c2 * e)

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)

        # Bottleneck sequence (identical to C3SWT)
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n))
        )

        # ── Constraint 3: n_heads = max(1, c_ // 32) always divides c_ evenly ──
        # c_ = int(c2 * e). With e=0.5 and typical c2 in {64,128,256,512,1024},
        # c_ ∈ {32,64,128,256,512}. Each is divisible by max(1, c_//32):
        #   c_=32  → n_heads=1  ✓ (32//1=32)
        #   c_=64  → n_heads=2  ✓ (64//2=32)
        #   c_=128 → n_heads=4  ✓ (128//4=32)
        #   c_=256 → n_heads=8  ✓ (256//8=32)
        #   c_=512 → n_heads=16 ✓ (512//16=32)
        n_heads = max(1, c_ // 32)
        assert c_ % n_heads == 0, (
            f"DC3SWT: c_={c_} not divisible by n_heads={n_heads}. "
            f"Use c2 values that are multiples of 64 (after expansion e={e})."
        )

        self.msda = MSDABlock(
            d_model=c_,
            n_heads=n_heads,
            n_points=n_points,
            drop=drop,
            drop_path=drop_path,
        )

        # Light spatial channel dropout (same regularization as C3SWT)
        self.spatial_drop = nn.Dropout2d(0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        # DC3SWT: deformable attention replaces W-MSA+SW-MSA, see arxiv:2010.04159

        Args:
            x: (B, c1, H, W)

        Returns:
            (B, c2, H, W)  — same spatial dimensions preserved
        """
        y1 = self.m(self.cv1(x))        # (B, c_, H, W)
        B, C, H, W = y1.shape

        # Flatten spatial for attention: (B, C, H, W) → (B, H*W, C)
        y1_seq = y1.flatten(2).transpose(1, 2)   # (B, N, C)

        # DC3SWT: deformable attention replaces W-MSA+SW-MSA, see arxiv:2010.04159
        y1_seq = self.msda(y1_seq, H, W)         # (B, N, C)  — MSDA with residual

        # Restore spatial shape: (B, N, C) → (B, C, H, W)
        y1 = y1_seq.transpose(1, 2).reshape(B, C, H, W)

        # Spatial channel dropout (train-time regularization)
        y1 = self.spatial_drop(y1)

        return self.cv3(torch.cat((y1, self.cv2(x)), dim=1))


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("DC3SWT self-test (pure PyTorch — no custom CUDA):")
    print()

    # Test 1 — standard 40×40 feature map
    m = DC3SWT(256, 256, n=1)
    m.eval()
    x = torch.randn(1, 256, 40, 40)
    with torch.no_grad():
        out = m(x)
    assert out.shape == (1, 256, 40, 40), f"Test 1 failed: {out.shape}"
    print(f"  ✅ Standard (256→256, 40×40): {out.shape}")

    # Test 2 — non-square, non-multiple of any window size
    m2 = DC3SWT(128, 128, n=1)
    m2.eval()
    x2 = torch.randn(1, 128, 36, 37)   # asymmetric, not divisible by 8
    with torch.no_grad():
        out2 = m2(x2)
    assert out2.shape == (1, 128, 36, 37), f"Test 2 failed: {out2.shape}"
    print(f"  ✅ Non-square (128→128, 36×37): {out2.shape}")

    # Test 3 — tiny feature map (H*W < n_points) — constraint 4
    m3 = DC3SWT(64, 64, n=1, n_points=4)
    m3.eval()
    x3 = torch.randn(1, 64, 2, 2)   # H*W = 4 = n_points (boundary case)
    with torch.no_grad():
        out3 = m3(x3)
    assert out3.shape == (1, 64, 2, 2), f"Test 3 failed: {out3.shape}"
    print(f"  ✅ Tiny map (64→64, 2×2, n_pts clamped): {out3.shape}")

    # Test 4 — extreme tiny (H*W < n_points)
    m4 = DC3SWT(64, 64, n=1, n_points=8)
    m4.eval()
    x4 = torch.randn(1, 64, 2, 1)   # H*W = 2 < n_points=8
    with torch.no_grad():
        out4 = m4(x4)
    assert out4.shape == (1, 64, 2, 1), f"Test 4 failed: {out4.shape}"
    print(f"  ✅ Extreme tiny (H*W=2 < n_points=8, clamped): {out4.shape}")

    # Test 5 — channel reduction
    m5 = DC3SWT(256, 128, n=1)
    m5.eval()
    x5 = torch.randn(1, 256, 20, 20)
    with torch.no_grad():
        out5 = m5(x5)
    assert out5.shape == (1, 128, 20, 20), f"Test 5 failed: {out5.shape}"
    print(f"  ✅ Channel reduction (256→128, 20×20): {out5.shape}")

    # Test 6 — train vs eval (dropout active in train)
    m6 = DC3SWT(128, 128, n=1)
    x6 = torch.randn(1, 128, 16, 16)
    m6.train()
    out6_train = m6(x6)
    m6.eval()
    with torch.no_grad():
        out6_eval = m6(x6)
    assert out6_train.shape == out6_eval.shape == (1, 128, 16, 16)
    print(f"  ✅ Train vs eval mode (128→128, 16×16): {out6_eval.shape}")

    print()
    print("All DC3SWT self-tests passed. ✅")
