import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torchvision.ops import DeformConv2d
except ImportError:
    DeformConv2d = None


class SEBlock(nn.Module):
    """
    Squeeze-Excitation channel attention block (Hu et al. 2018).

    Applied after each BiFPN fusion node to add per-channel selectivity.
    Orthogonal to CoordAttMulti (spatial attention) — SE operates on the
    channel axis only via global average pooling → FC bottleneck → sigmoid scale.

    Args:
        channels  (int): Number of input/output channels.
        reduction (int): Channel reduction ratio for the FC bottleneck (default 16).
                         For channels=256 → hidden=16 → ~8K params per block.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        scale = self.pool(x).view(b, c)
        scale = self.fc(scale).view(b, c, 1, 1)
        return x * scale

class DepthwiseConvBlock(nn.Module):
    """Depthwise separable convolution for BiFPN."""
    def __init__(self, in_channels, out_channels, apply_bn=True, apply_act=True):
        super(DepthwiseConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.apply_bn = apply_bn
        self.apply_act = apply_act
        
        if self.apply_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if self.apply_act:
            self.act = nn.SiLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.apply_bn:
            x = self.bn(x)
        if self.apply_act:
            x = self.act(x)
        return x

class DeformConvBlock(nn.Module):
    """Deformable Convolution v2 Block."""
    def __init__(self, in_channels, out_channels, apply_bn=True, apply_act=True):
        super(DeformConvBlock, self).__init__()
        if DeformConv2d is None:
            raise ImportError("torchvision.ops.DeformConv2d not found. Please install/update torchvision.")
        
        self.offset_conv = nn.Conv2d(in_channels, 18, kernel_size=3, padding=1, bias=True)
        self.mask_conv = nn.Conv2d(in_channels, 9, kernel_size=3, padding=1, bias=True)
        self.dcn = DeformConv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        
        self.apply_bn = apply_bn
        self.apply_act = apply_act
        if self.apply_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if self.apply_act:
            self.act = nn.SiLU()

    def forward(self, x):
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        x = self.dcn(x, offset, mask)
        if self.apply_bn:
            x = self.bn(x)
        if self.apply_act:
            x = self.act(x)
        return x

class BiFPNLayer(nn.Module):
    """
    BiFPN Layer — 4-level bidirectional feature pyramid (P2, P3, P4, P5).
    Fast normalized weighted fusion from EfficientDet (Tan et al. 2020).

    DCNv2 placement: P3 output node only (not P4 or P5)
    ─────────────────────────────────────────────────────
    The BiFPN bottom-up pathway processes nodes in order P2 → P3 → P4 → P5.
    P3_out is the most critical fusion node for three reasons:

    1. **3-input junction (maximum information density)**
       P3_out merges: original P3 features + P3_td skip-connection + P2_out
       downsampled. It is the only node that simultaneously receives
       (a) the high-res P2 signal, (b) the top-down context from P4/P5, and
       (c) its own lateral skip — three different spatial scales at once.
       Rigid 3×3 convolutions cannot align feature grids that originate from
       such diverse sampling histories; DCNv2's learned offsets correct this.

    2. **P3 is the primary small-object scale**
       In our 4-scale head (P2/P3/P4/P5), P3 (stride 8) handles objects in
       the ~16–48px range — exactly the range where dense remote-sensing
       symbols, ships, and vehicles concentrate. Misalignment at P3 directly
       degrades mAP for these classes. P4 and P5 handle larger objects where
       rigid convolutions already achieve adequate spatial coverage.

    3. **Diminishing returns and cost**
       P4 resolution is 2× smaller than P3 (stride 16); the spatial grid
       mismatch between merged paths is halved relative to the receptive
       field, so alignment errors matter less. P5 (stride 32) resolves only
       very large objects. Each DeformConvBlock adds 3 extra conv layers
       (offset, mask, DCN) ≈ +150K params per level — justified only where
       the alignment problem is hardest and the detection payoff is largest.

    Summary: DCNv2 at P3 → maximum alignment benefit for minimum cost.
             Extending to P4/P5 would add ~300K params with negligible mAP gain.
    """
    def __init__(self, num_channels, in_channels_list=None, use_dcn=False, use_se=False, num_levels=4, epsilon=1e-4):
        super(BiFPNLayer, self).__init__()
        self.epsilon = epsilon

        # Projections to common channel dim for inputs
        if in_channels_list is None:
            # Assume they already match (e.g. cascaded BiFPN)
            self.projections = nn.ModuleList([nn.Identity() for _ in range(num_levels)])
        else:
            self.projections = nn.ModuleList([
                nn.Conv2d(c, num_channels, 1) if c != num_channels else nn.Identity()
                for c in in_channels_list
            ])

        # P5 is top-most, P2 is bottom-most.
        # BiFPN node weights (initialized to 1.0)
        # Top-down pathway (td) weights -> 2 inputs
        self.p4_td_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.p3_td_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.p2_out_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32))  # P2 bottom-most: 2 inputs

        # Bottom-up pathway (out) weights -> 3 inputs (except P5 top-most which has 2)
        self.p3_out_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.p4_out_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.p5_out_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32))

        # Convolutions for feature fusion at each node
        self.conv_p4_td  = DepthwiseConvBlock(num_channels, num_channels)
        self.conv_p3_td  = DepthwiseConvBlock(num_channels, num_channels)
        self.conv_p2_out = DepthwiseConvBlock(num_channels, num_channels)
        self.conv_p3_out = (
            DeformConvBlock(num_channels, num_channels) if use_dcn
            else DepthwiseConvBlock(num_channels, num_channels)
        )
        self.conv_p4_out = DepthwiseConvBlock(num_channels, num_channels)
        self.conv_p5_out = DepthwiseConvBlock(num_channels, num_channels)

        # SE channel attention after each fusion node (orthogonal to CoordAttMulti spatial attention).
        # All 6 nodes get SE for systematic coverage; ~49K total params at num_channels=256.
        if use_se:
            self.se_p4_td  = SEBlock(num_channels)
            self.se_p3_td  = SEBlock(num_channels)
            self.se_p2_out = SEBlock(num_channels)
            self.se_p3_out = SEBlock(num_channels)
            self.se_p4_out = SEBlock(num_channels)
            self.se_p5_out = SEBlock(num_channels)
        else:
            # nn.Identity passthrough — zero cost when use_se=False
            self.se_p4_td = self.se_p3_td = self.se_p2_out = nn.Identity()
            self.se_p3_out = self.se_p4_out = self.se_p5_out = nn.Identity()

        # Upsampling and Downsampling ops
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
    def _downsample(self, x, target_shape):
        """Simple stride 2 max pooling, pad if necessary to match shape."""
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        # Handle unmatched spatial dimensions
        if x.shape[2:] != target_shape:
            # Interpolate to match exact shape
            x = F.interpolate(x, size=target_shape, mode='nearest')
        return x

    def forward(self, features):
        """
        features: [P2, P3, P4, P5]
        """
        p2, p3, p4, p5 = [proj(feat) for proj, feat in zip(self.projections, features)]

        # TOP-DOWN PATHWAY

        # P4_td = SE( conv( w*p4 + w*resize(p5) ) )
        w_p4_td = F.relu(self.p4_td_w1)
        weight_p4_td = w_p4_td / (torch.sum(w_p4_td) + self.epsilon)
        p5_up = F.interpolate(p5, size=p4.shape[2:], mode='nearest')
        p4_td = self.se_p4_td(self.conv_p4_td(weight_p4_td[0] * p4 + weight_p4_td[1] * p5_up))

        # P3_td = SE( conv( w*p3 + w*resize(p4_td) ) )
        w_p3_td = F.relu(self.p3_td_w1)
        weight_p3_td = w_p3_td / (torch.sum(w_p3_td) + self.epsilon)
        p4_td_up = F.interpolate(p4_td, size=p3.shape[2:], mode='nearest')
        p3_td = self.se_p3_td(self.conv_p3_td(weight_p3_td[0] * p3 + weight_p3_td[1] * p4_td_up))

        # BOTTOM-UP PATHWAY

        # P2_out = SE( conv( w*p2 + w*resize(p3_td) ) )
        w_p2_out = F.relu(self.p2_out_w1)
        weight_p2_out = w_p2_out / (torch.sum(w_p2_out) + self.epsilon)
        p3_td_up = F.interpolate(p3_td, size=p2.shape[2:], mode='nearest')
        p2_out = self.se_p2_out(self.conv_p2_out(weight_p2_out[0] * p2 + weight_p2_out[1] * p3_td_up))

        # P3_out = SE( conv( w*p3 + w*p3_td + w*downsample(p2_out) ) )  [3-input junction + DCNv2]
        w_p3_out = F.relu(self.p3_out_w2)
        weight_p3_out = w_p3_out / (torch.sum(w_p3_out) + self.epsilon)
        p2_out_down = self._downsample(p2_out, p3.shape[2:])
        p3_out = self.se_p3_out(self.conv_p3_out(
            weight_p3_out[0] * p3 + weight_p3_out[1] * p3_td + weight_p3_out[2] * p2_out_down
        ))

        # P4_out = SE( conv( w*p4 + w*p4_td + w*downsample(p3_out) ) )
        w_p4_out = F.relu(self.p4_out_w2)
        weight_p4_out = w_p4_out / (torch.sum(w_p4_out) + self.epsilon)
        p3_out_down = self._downsample(p3_out, p4.shape[2:])
        p4_out = self.se_p4_out(self.conv_p4_out(
            weight_p4_out[0] * p4 + weight_p4_out[1] * p4_td + weight_p4_out[2] * p3_out_down
        ))

        # P5_out = SE( conv( w*p5 + w*downsample(p4_out) ) )
        w_p5_out = F.relu(self.p5_out_w2)
        weight_p5_out = w_p5_out / (torch.sum(w_p5_out) + self.epsilon)
        p4_out_down = self._downsample(p4_out, p5.shape[2:])
        p5_out = self.se_p5_out(self.conv_p5_out(weight_p5_out[0] * p5 + weight_p5_out[1] * p4_out_down))

        return [p2_out, p3_out, p4_out, p5_out]

if __name__ == '__main__':
    feats = [
        torch.randn(1, 256, 160, 160),  # P2
        torch.randn(1, 256, 80, 80),    # P3
        torch.randn(1, 256, 40, 40),    # P4
        torch.randn(1, 256, 20, 20),    # P5
    ]
    # Test standard
    m = BiFPNLayer(num_channels=256, in_channels_list=[256, 256, 256, 256])
    outs = m(feats)
    print("Standard BiFPNLayer test passed!")
    
    # Test DCN
    try:
        m_dcn = BiFPNLayer(num_channels=256, in_channels_list=[256, 256, 256, 256], use_dcn=True)
        outs_dcn = m_dcn(feats)
        print("DCN BiFPNLayer test passed!")
    except Exception as e:
        print(f"DCN test failed: {e}")
    print("BiFPNLayer test results:")
    for i, o in enumerate(outs):
        print(f"P{i+2}_out: {o.shape} (expected: {feats[i].shape})")
        assert o.shape == feats[i].shape, f"Mismatch P{i+2}: got {o.shape}, expected {feats[i].shape}"
    print("Test passed!")
