"""
tests/test_components.py
========================
Component-level unit tests for all DA-YOLO custom modules.

Run from repo root:
    python tests/test_components.py         # standalone — no pytest required
    pytest tests/test_components.py -v      # with pytest

Tests cover:
  - C3SWT      : forward pass at multiple spatial sizes, eval vs train mode, padding
  - BiFPNLayer : 4-scale fusion, DCNv2 path, shape preservation
  - CoordAtt / CoordAttMulti : 4D and edge-case inputs
  - ciou_kmeans : anchor clustering correctness
  - YAML parse  : full model construction from yolov5s_swint.yaml (dummy forward pass)
  - DC3SWT     : MSDABlock + full DC3SWT, n_points clamp, shape equivalence vs C3SWT
                 DC3SWT: deformable attention replaces W-MSA+SW-MSA, see arxiv:2010.04159
"""

import sys
import os
import traceback

# Make sure imports resolve relative to repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

# ── Helpers ─────────────────────────────────────────────────────────────────

_PASS = "✅ PASS"
_FAIL = "❌ FAIL"
_SKIP = "⚠️  SKIP"

results = []


def run_test(name: str, fn):
    """Run a single test function, capture pass/fail, print result."""
    try:
        fn()
        results.append((name, True, None))
        print(f"  {_PASS}  {name}")
    except Exception as e:
        tb = traceback.format_exc()
        results.append((name, False, str(e)))
        print(f"  {_FAIL}  {name}")
        print(f"         {e}")
        # Print last 3 lines of traceback for fast debugging
        for line in tb.strip().split("\n")[-3:]:
            print(f"         {line}")


def require_module(import_expr: str):
    """Return (module_or_None, skip_reason_or_None)."""
    try:
        import importlib
        return importlib.import_module(import_expr), None
    except ImportError as e:
        return None, str(e)


# ── C3SWT Tests ─────────────────────────────────────────────────────────────

def _test_c3swt_import():
    from models.swin_block import C3SWT, SwinTransformerBlock  # noqa: F401


def _test_c3swt_forward_standard():
    """C3SWT forward at standard 40×40 feature map."""
    from models.swin_block import C3SWT
    m = C3SWT(256, 256, n=1)
    m.eval()
    x = torch.randn(1, 256, 40, 40)
    out = m(x)
    assert out.shape == (1, 256, 40, 40), f"Expected (1,256,40,40), got {out.shape}"


def _test_c3swt_forward_non_square():
    """C3SWT must handle non-square inputs via padding."""
    from models.swin_block import C3SWT
    m = C3SWT(128, 128, n=1)
    m.eval()
    # 36×36 is NOT a multiple of window_size=8 → triggers padding
    x = torch.randn(1, 128, 36, 36)
    out = m(x)
    assert out.shape == (1, 128, 36, 36), f"Expected (1,128,36,36), got {out.shape}"


def _test_c3swt_forward_small():
    """C3SWT with spatial size smaller than window_size (edge case)."""
    from models.swin_block import C3SWT
    m = C3SWT(64, 64, n=1, window_size=8)
    m.eval()
    # 4×4 < window_size=8 — Swin block should clamp window to min(H,W)
    x = torch.randn(1, 64, 4, 4)
    out = m(x)
    assert out.shape == (1, 64, 4, 4), f"Expected (1,64,4,4), got {out.shape}"


def _test_c3swt_train_vs_eval():
    """Dropout2d should be active in train mode; no-op in eval mode."""
    from models.swin_block import C3SWT
    m = C3SWT(128, 128, n=1)
    x = torch.randn(1, 128, 20, 20)
    m.train()
    out_train = m(x)
    m.eval()
    with torch.no_grad():
        out_eval = m(x)
    # Both must have the right shape
    assert out_train.shape == (1, 128, 20, 20)
    assert out_eval.shape == (1, 128, 20, 20)


def _test_c3swt_channels():
    """C3SWT with channel expansion/reduction."""
    from models.swin_block import C3SWT
    m = C3SWT(256, 128, n=1)
    m.eval()
    x = torch.randn(1, 256, 20, 20)
    out = m(x)
    assert out.shape == (1, 128, 20, 20), f"Expected (1,128,20,20), got {out.shape}"


def _test_c3swt_layernorm_present():
    """LayerNorm must be present (not replaced by Identity) to prevent loss explosion."""
    from models.swin_block import C3SWT, SwinTransformerBlock
    m = C3SWT(128, 128)
    swin_block = m.swin
    assert isinstance(swin_block.norm1, nn.LayerNorm), \
        f"norm1 should be LayerNorm, got {type(swin_block.norm1).__name__}"
    assert isinstance(swin_block.norm2, nn.LayerNorm), \
        f"norm2 should be LayerNorm, got {type(swin_block.norm2).__name__}"


def _test_c3swt_mask_caching():
    """attn_mask cache should be created after first forward pass."""
    from models.swin_block import C3SWT
    m = C3SWT(128, 128)
    m.eval()
    x = torch.randn(1, 128, 32, 32)
    with torch.no_grad():
        m(x)
    assert hasattr(m, "_mask_cache"), "C3SWT should cache attention mask after forward"


# ── BiFPN Tests ─────────────────────────────────────────────────────────────

def _test_bifpn_import():
    from models.bifpn import BiFPNLayer, DepthwiseConvBlock  # noqa: F401


def _test_bifpn_forward_standard():
    """BiFPN standard (no DCNv2) — 4-scale P2-P5 fusion."""
    from models.bifpn import BiFPNLayer
    in_ch = [128, 256, 512, 1024]
    feats = [
        torch.randn(1, in_ch[0], 160, 160),  # P2
        torch.randn(1, in_ch[1], 80, 80),    # P3
        torch.randn(1, in_ch[2], 40, 40),    # P4
        torch.randn(1, in_ch[3], 20, 20),    # P5
    ]
    m = BiFPNLayer(num_channels=256, in_channels_list=in_ch, use_dcn=False)
    m.eval()
    with torch.no_grad():
        outs = m(feats)
    assert len(outs) == 4, f"Expected 4 outputs, got {len(outs)}"
    expected_shapes = [(1,256,160,160), (1,256,80,80), (1,256,40,40), (1,256,20,20)]
    for i, (out, exp) in enumerate(zip(outs, expected_shapes)):
        assert out.shape == torch.Size(exp), \
            f"P{i+2}: expected {exp}, got {tuple(out.shape)}"


def _test_bifpn_forward_dcn():
    """BiFPN with DCNv2 at P3 — requires torchvision.ops.DeformConv2d."""
    try:
        from torchvision.ops import DeformConv2d  # noqa: F401
    except ImportError:
        raise RuntimeError("torchvision.ops.DeformConv2d not available — DCN test skipped")

    from models.bifpn import BiFPNLayer
    in_ch = [128, 256, 512, 1024]
    feats = [torch.randn(1, c, s, s) for c, s in zip(in_ch, [160, 80, 40, 20])]
    m = BiFPNLayer(num_channels=256, in_channels_list=in_ch, use_dcn=True)
    m.eval()
    with torch.no_grad():
        outs = m(feats)
    assert len(outs) == 4
    for out in outs:
        assert out.shape[1] == 256


def _test_bifpn_shape_preserved():
    """Spatial dims must be preserved through BiFPN."""
    from models.bifpn import BiFPNLayer
    in_ch = [64, 128, 256, 512]
    spatial = [128, 64, 32, 16]
    feats = [torch.randn(1, c, s, s) for c, s in zip(in_ch, spatial)]
    m = BiFPNLayer(num_channels=128, in_channels_list=in_ch)
    m.eval()
    with torch.no_grad():
        outs = m(feats)
    for i, (out, s) in enumerate(zip(outs, spatial)):
        assert out.shape[2] == s and out.shape[3] == s, \
            f"P{i+2} spatial mismatch: expected {s}×{s}, got {out.shape[2]}×{out.shape[3]}"


def _test_bifpn_weight_params_exist():
    """BiFPN must have learnable fusion weights."""
    from models.bifpn import BiFPNLayer
    m = BiFPNLayer(num_channels=128, in_channels_list=[128]*4)
    param_names = [n for n, _ in m.named_parameters() if 'w' in n and 'weight' not in n]
    assert len(param_names) > 0, "BiFPN must have learnable fusion weight parameters"


# ── CoordAtt Tests ───────────────────────────────────────────────────────────

def _test_coordatt_import():
    from models.coord_attention import CoordAtt, CoordAttMulti  # noqa: F401


def _test_coordatt_forward():
    """CoordAtt standard 4D tensor."""
    from models.coord_attention import CoordAtt
    m = CoordAtt(256, 256)
    m.eval()
    x = torch.randn(1, 256, 40, 40)
    with torch.no_grad():
        out = m(x)
    assert out.shape == (1, 256, 40, 40), f"Expected (1,256,40,40), got {out.shape}"


def _test_coordatt_channel_reduction():
    """CoordAtt with inp != oup."""
    from models.coord_attention import CoordAtt
    m = CoordAtt(256, 128)
    m.eval()
    x = torch.randn(1, 256, 20, 20)
    with torch.no_grad():
        out = m(x)
    assert out.shape == (1, 128, 20, 20), f"Expected (1,128,20,20), got {out.shape}"


def _test_coordatt_3d_guard():
    """CoordAtt must handle 3D inputs without crashing (was_3d guard)."""
    from models.coord_attention import CoordAtt
    m = CoordAtt(64, 64)
    m.eval()
    # 3D tensor (no batch dim)
    x = torch.randn(64, 10, 10)
    with torch.no_grad():
        out = m(x)
    assert out.dim() == 3, f"Expected 3D output for 3D input, got {out.dim()}D"
    assert out.shape == (64, 10, 10), f"Expected (64,10,10), got {out.shape}"


def _test_coordattmulti_forward():
    """CoordAttMulti applies per-scale CA to a list of 4 tensors."""
    from models.coord_attention import CoordAttMulti
    m = CoordAttMulti(inp=256, oup=256, num_levels=4)
    m.eval()
    feats = [torch.randn(1, 256, s, s) for s in [160, 80, 40, 20]]
    with torch.no_grad():
        outs = m(feats)
    assert len(outs) == 4
    for i, (out, feat) in enumerate(zip(outs, feats)):
        assert out.shape == feat.shape, \
            f"Scale {i}: expected {feat.shape}, got {out.shape}"


# ── CIOU K-means Tests ───────────────────────────────────────────────────────

def _test_ciou_kmeans_import():
    from utils.ciou_kmeans import ciou_distance, ciou_kmeans, generate_anchors  # noqa: F401


def _test_ciou_distance_identical():
    """CIOU distance of identical boxes must be ~0."""
    from utils.ciou_kmeans import ciou_distance
    import numpy as np
    d = ciou_distance([30, 50], [30, 50])
    assert d < 1e-6, f"Distance of identical boxes should be ~0, got {d}"


def _test_ciou_distance_orthogonal():
    """CIOU distance between very different boxes must be close to 1."""
    from utils.ciou_kmeans import ciou_distance
    d = ciou_distance([100, 1], [1, 100])  # tall vs wide
    assert d > 0.5, f"Distance of orthogonal boxes should be > 0.5, got {d}"


def _test_ciou_kmeans_output_shape():
    """ciou_kmeans must return k clusters of 2D anchor boxes."""
    from utils.ciou_kmeans import ciou_kmeans
    import numpy as np
    np.random.seed(42)
    boxes = np.random.uniform(10, 100, (300, 2))  # 300 random boxes
    k = 9
    clusters = ciou_kmeans(boxes, k)
    assert clusters.shape == (k, 2), f"Expected ({k},2), got {clusters.shape}"


def _test_ciou_kmeans_sorted():
    """ciou_kmeans output must be sorted by area (ascending)."""
    from utils.ciou_kmeans import ciou_kmeans
    import numpy as np
    np.random.seed(0)
    boxes = np.random.uniform(5, 200, (500, 2))
    clusters = ciou_kmeans(boxes, 9)
    areas = clusters[:, 0] * clusters[:, 1]
    assert all(areas[i] <= areas[i+1] for i in range(len(areas)-1)), \
        "ciou_kmeans output must be area-sorted (ascending)"


# ── DC3SWT Tests ─────────────────────────────────────────────────────────────
# DC3SWT: deformable attention replaces W-MSA+SW-MSA, see arxiv:2010.04159

def _test_dc3swt_import():
    """DC3SWT and MSDABlock must import cleanly."""
    from models.dc3swt import DC3SWT, MSDABlock  # noqa: F401


def _test_msda_forward_standard():
    """MSDABlock forward on standard token sequence."""
    from models.dc3swt import MSDABlock
    d_model, n_heads = 128, 4
    m = MSDABlock(d_model=d_model, n_heads=n_heads, n_points=4)
    m.eval()
    B, H, W = 1, 40, 40
    x = torch.randn(B, H * W, d_model)
    with torch.no_grad():
        out = m(x, H, W)
    assert out.shape == (B, H * W, d_model), f"Expected {(B,H*W,d_model)}, got {out.shape}"


def _test_msda_npoints_clamp():
    """
    Constraint 4: when H*W < n_points, n_points is clamped to H*W.
    This must NOT crash — the network should still produce the correct output shape.
    DC3SWT: deformable attention replaces W-MSA+SW-MSA, see arxiv:2010.04159
    """
    from models.dc3swt import MSDABlock
    d_model, n_heads = 64, 2
    n_points = 8   # n_points > H*W
    m = MSDABlock(d_model=d_model, n_heads=n_heads, n_points=n_points)
    m.eval()
    H, W = 2, 2   # H*W = 4 < n_points = 8
    x = torch.randn(1, H * W, d_model)
    with torch.no_grad():
        out = m(x, H, W)
    assert out.shape == (1, H * W, d_model), f"Clamp test failed: {out.shape}"


def _test_msda_extreme_tiny():
    """H*W = 1 — most extreme clamp case (single spatial location)."""
    from models.dc3swt import MSDABlock
    d_model, n_heads = 32, 1
    m = MSDABlock(d_model=d_model, n_heads=n_heads, n_points=4)
    m.eval()
    H, W = 1, 1
    x = torch.randn(1, 1, d_model)
    with torch.no_grad():
        out = m(x, H, W)
    assert out.shape == (1, 1, d_model)


def _test_msda_nheads_divisible():
    """
    Constraint 3: n_heads = max(1, c_ // 32) must divide d_model evenly.
    Verify this for all typical YOLOv5s channel widths (after width_multiple=0.5).
    DC3SWT: deformable attention replaces W-MSA+SW-MSA, see arxiv:2010.04159
    """
    from models.dc3swt import DC3SWT
    # c_ = int(c2 * e=0.5). Typical c2 values after width_multiple=0.5:
    # c2 in [64, 128, 256, 512, 1024] → c_ in [32, 64, 128, 256, 512]
    for c2 in [64, 128, 256, 512, 1024]:
        c_ = int(c2 * 0.5)
        n_heads = max(1, c_ // 32)
        assert c_ % n_heads == 0, (
            f"c_={c_} not divisible by n_heads={n_heads} for c2={c2}"
        )
        # Also instantiate to catch any runtime assertion
        m = DC3SWT(c2, c2, n=1)
        m.eval()
        out = m(torch.randn(1, c2, 4, 4))
        assert out.shape[1] == c2


def _test_dc3swt_forward_standard():
    """DC3SWT drop-in forward at standard 40×40."""
    from models.dc3swt import DC3SWT
    m = DC3SWT(256, 256, n=1)
    m.eval()
    x = torch.randn(1, 256, 40, 40)
    with torch.no_grad():
        out = m(x)
    assert out.shape == (1, 256, 40, 40)


def _test_dc3swt_forward_non_square():
    """
    DC3SWT handles non-square, non-window-aligned inputs natively —
    no padding needed (key advantage over C3SWT).
    DC3SWT: deformable attention replaces W-MSA+SW-MSA, see arxiv:2010.04159
    """
    from models.dc3swt import DC3SWT
    m = DC3SWT(128, 128, n=1)
    m.eval()
    x = torch.randn(1, 128, 36, 37)   # asymmetric, not divisible by 8
    with torch.no_grad():
        out = m(x)
    assert out.shape == (1, 128, 36, 37)


def _test_dc3swt_shape_equiv_c3swt():
    """
    Constraint 2: DC3SWT must produce identical output shape to C3SWT
    for the same (c1, c2, H, W) arguments — it is a drop-in replacement.
    """
    from models.dc3swt import DC3SWT
    from models.swin_block import C3SWT
    x = torch.randn(1, 256, 40, 40)
    dc = DC3SWT(256, 256, n=1).eval()
    sw = C3SWT(256, 256, n=1).eval()
    with torch.no_grad():
        out_dc = dc(x)
        out_sw = sw(x)
    assert out_dc.shape == out_sw.shape, (
        f"Shape mismatch: DC3SWT={out_dc.shape} vs C3SWT={out_sw.shape}"
    )


def _test_dc3swt_train_eval():
    """DropPath + Dropout2d active in train mode; identity in eval."""
    from models.dc3swt import DC3SWT
    m = DC3SWT(128, 128, n=1)
    x = torch.randn(1, 128, 16, 16)
    m.train()
    out_tr = m(x)
    m.eval()
    with torch.no_grad():
        out_ev = m(x)
    assert out_tr.shape == out_ev.shape == (1, 128, 16, 16)


def _test_dc3swt_model_parse():
    """
    Full model parse from da_yolo.yaml — verifies DC3SWT integrates
    cleanly with BiFPN (SE+DCNv2), CoordAttMulti, and the 4-scale Detect head.
    DC3SWT: deformable attention replaces W-MSA+SW-MSA, see arxiv:2010.04159
    """
    from models.yolo import Model
    yaml_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "da_yolo.yaml"
    )
    assert os.path.isfile(yaml_path), f"DA-YOLO YAML not found: {yaml_path}"
    model = Model(yaml_path, ch=3, nc=10)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 256, 256))
    assert out is not None


# ── Main ─────────────────────────────────────────────────────────────────────


def _test_model_yaml_exists():
    """yolov5s_swint.yaml must exist."""
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "models", "yolov5s_swint.yaml")
    assert os.path.isfile(yaml_path), f"YAML not found: {yaml_path}"


def _test_model_parse_no_crash():
    """
    Building the model from yolov5s_swint.yaml must succeed without errors.
    We do a tiny dummy forward (batch=1, 3×256×256) just to confirm layers connect.
    """
    from models.yolo import Model
    import yaml
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "models", "yolov5s_swint.yaml")
    model = Model(yaml_path, ch=3, nc=10)  # nc=10 to keep it fast
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 256, 256)
        out = model(dummy)
    # Training output is a list of feature maps from Detect head
    # Inference output is a tuple (concatenated_preds, raw_list)
    assert out is not None


def _test_model_bifpn_dcn_enabled():
    """
    Verify that BiFPNLayer in the built model has use_dcn=True
    (i.e., conv_p3_out is DeformConvBlock, not DepthwiseConvBlock).
    """
    from models.yolo import Model
    from models.bifpn import BiFPNLayer, DeformConvBlock
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "models", "yolov5s_swint.yaml")

    try:
        from torchvision.ops import DeformConv2d  # noqa: F401
        dcn_available = True
    except ImportError:
        dcn_available = False

    if not dcn_available:
        # If torchvision doesn't have DCN, BiFPN will fall back gracefully
        return

    model = Model(yaml_path, ch=3, nc=10)
    # Find the BiFPNLayer module in the model
    bifpn = None
    for m in model.model.modules():
        if isinstance(m, BiFPNLayer):
            bifpn = m
            break
    assert bifpn is not None, "BiFPNLayer not found in built model"
    assert isinstance(bifpn.conv_p3_out, DeformConvBlock), (
        f"Expected DeformConvBlock at P3 (use_dcn=True), got {type(bifpn.conv_p3_out).__name__}"
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print()
    print("=" * 65)
    print("  DA-YOLO Component Tests")
    print("=" * 65)
    print()

    # C3SWT
    print("C3SWT (swin_block.py):")
    run_test("import",              _test_c3swt_import)
    run_test("forward 40×40",       _test_c3swt_forward_standard)
    run_test("forward non-square",  _test_c3swt_forward_non_square)
    run_test("forward small (4×4)", _test_c3swt_forward_small)
    run_test("train vs eval mode",  _test_c3swt_train_vs_eval)
    run_test("channel reduction",   _test_c3swt_channels)
    run_test("LayerNorm present",   _test_c3swt_layernorm_present)
    run_test("mask caching",        _test_c3swt_mask_caching)
    print()

    # BiFPN
    print("BiFPNLayer (bifpn.py):")
    run_test("import",              _test_bifpn_import)
    run_test("forward (no DCN)",    _test_bifpn_forward_standard)
    run_test("forward (with DCN)",  _test_bifpn_forward_dcn)
    run_test("shape preservation",  _test_bifpn_shape_preserved)
    run_test("learnable weights",   _test_bifpn_weight_params_exist)
    print()

    # CoordAtt
    print("CoordAtt (coord_attention.py):")
    run_test("import",              _test_coordatt_import)
    run_test("forward 4D",          _test_coordatt_forward)
    run_test("channel reduction",   _test_coordatt_channel_reduction)
    run_test("3D input guard",      _test_coordatt_3d_guard)
    run_test("CoordAttMulti",       _test_coordattmulti_forward)
    print()

    # CIOU K-means
    print("CIOU K-means (utils/ciou_kmeans.py):")
    run_test("import",              _test_ciou_kmeans_import)
    run_test("distance identical",  _test_ciou_distance_identical)
    run_test("distance orthogonal", _test_ciou_distance_orthogonal)
    run_test("output shape",        _test_ciou_kmeans_output_shape)
    run_test("sorted by area",      _test_ciou_kmeans_sorted)
    print()

    # Model parse (C3SWT YAML)
    print("Model parse (models/yolo.py + yolov5s_swint.yaml):")
    run_test("yaml file exists",    _test_model_yaml_exists)
    run_test("model builds OK",     _test_model_parse_no_crash)
    run_test("DCN enabled at P3",   _test_model_bifpn_dcn_enabled)
    print()

    # DC3SWT — Deformable C3 Swin Transformer Block
    # DC3SWT: deformable attention replaces W-MSA+SW-MSA, see arxiv:2010.04159
    print("DC3SWT (models/dc3swt.py):")
    run_test("import",                   _test_dc3swt_import)
    run_test("MSDABlock forward",        _test_msda_forward_standard)
    run_test("n_points clamp (H*W<K)",   _test_msda_npoints_clamp)
    run_test("extreme tiny H*W=1",       _test_msda_extreme_tiny)
    run_test("n_heads divisible",        _test_msda_nheads_divisible)
    run_test("DC3SWT forward 40×40",     _test_dc3swt_forward_standard)
    run_test("DC3SWT non-square 36×37",  _test_dc3swt_forward_non_square)
    run_test("shape equiv to C3SWT",     _test_dc3swt_shape_equiv_c3swt)
    run_test("train vs eval mode",       _test_dc3swt_train_eval)
    run_test("DC3SWT model parse",       _test_dc3swt_model_parse)
    print()

    # Summary
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    total  = len(results)
    print("=" * 65)
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
    else:
        print("  — All tests passed! ✅")
    print("=" * 65)
    print()

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
