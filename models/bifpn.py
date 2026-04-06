import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseConvBlock(nn.Module):
    """Depthwise separable convolution for BiFPN."""

    def __init__(self, in_channels, out_channels, apply_bn=True, apply_act=True):
        super().__init__()
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


class BiFPNLayer(nn.Module):
    """BiFPN Layer supporting 4 levels (P2, P3, P4, P5) with fast normalized fusion."""

    def __init__(self, num_channels, in_channels_list=None, num_levels=4, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon

        # Projections to common channel dim for inputs
        if in_channels_list is None:
            # Assume they already match (e.g. cascaded BiFPN)
            self.projections = nn.ModuleList([nn.Identity() for _ in range(num_levels)])
        else:
            self.projections = nn.ModuleList(
                [nn.Conv2d(c, num_channels, 1) if c != num_channels else nn.Identity() for c in in_channels_list]
            )

        # P5 is top-most, P2 is bottom-most.
        # BiFPN node weights (initialized to 1.0)
        # Top-down pathway (td) weights -> 2 inputs
        self.p4_td_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.p3_td_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.p2_out_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32))  # P2 bottom-most is just 2 inputs

        # Bottom-up pathway (out) weights -> 3 inputs (except P5 top-most which has 2)
        self.p3_out_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.p4_out_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.p5_out_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32))

        # Convolutions for the feature fusion
        self.conv_p4_td = DepthwiseConvBlock(num_channels, num_channels)
        self.conv_p3_td = DepthwiseConvBlock(num_channels, num_channels)
        self.conv_p2_out = DepthwiseConvBlock(num_channels, num_channels)

        self.conv_p3_out = DepthwiseConvBlock(num_channels, num_channels)
        self.conv_p4_out = DepthwiseConvBlock(num_channels, num_channels)
        self.conv_p5_out = DepthwiseConvBlock(num_channels, num_channels)

        # Upsampling and Downsampling ops
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def _downsample(self, x, target_shape):
        """Simple stride 2 max pooling, pad if necessary to match shape."""
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        # Handle unmatched spatial dimensions
        if x.shape[2:] != target_shape:
            # Interpolate to match exact shape
            x = F.interpolate(x, size=target_shape, mode="nearest")
        return x

    def forward(self, features):
        """Features: [P2, P3, P4, P5]."""
        p2, p3, p4, p5 = [proj(feat) for proj, feat in zip(self.projections, features)]

        # TOP-DOWN PATHWAY

        # P4_td = conv( (p4_td_w1[0]*p4 + p4_td_w1[1]*Resize(p5)) / (p4_td_w1[0] + p4_td_w1[1] + eps) )
        w_p4_td = F.relu(self.p4_td_w1)
        weight_p4_td = w_p4_td / (torch.sum(w_p4_td) + self.epsilon)
        p5_up = F.interpolate(p5, size=p4.shape[2:], mode="nearest")
        p4_td = self.conv_p4_td(weight_p4_td[0] * p4 + weight_p4_td[1] * p5_up)

        # P3_td = conv( w*p3 + w*Resize(p4_td) )
        w_p3_td = F.relu(self.p3_td_w1)
        weight_p3_td = w_p3_td / (torch.sum(w_p3_td) + self.epsilon)
        p4_td_up = F.interpolate(p4_td, size=p3.shape[2:], mode="nearest")
        p3_td = self.conv_p3_td(weight_p3_td[0] * p3 + weight_p3_td[1] * p4_td_up)

        # BOTTOM-UP PATHWAY

        # P2_out = conv( w*p2 + w*Resize(p3_td) )  <- P2 forms the bottom output
        w_p2_out = F.relu(self.p2_out_w1)
        weight_p2_out = w_p2_out / (torch.sum(w_p2_out) + self.epsilon)
        p3_td_up = F.interpolate(p3_td, size=p2.shape[2:], mode="nearest")
        p2_out = self.conv_p2_out(weight_p2_out[0] * p2 + weight_p2_out[1] * p3_td_up)

        # P3_out = conv( w*p3 + w*Resize(p2_out) + w*p3_td )
        w_p3_out = F.relu(self.p3_out_w2)
        weight_p3_out = w_p3_out / (torch.sum(w_p3_out) + self.epsilon)
        p2_out_down = self._downsample(p2_out, p3.shape[2:])
        p3_out = self.conv_p3_out(weight_p3_out[0] * p3 + weight_p3_out[1] * p3_td + weight_p3_out[2] * p2_out_down)

        # P4_out = conv( w*p4 + w*Resize(p3_out) + w*p4_td )
        w_p4_out = F.relu(self.p4_out_w2)
        weight_p4_out = w_p4_out / (torch.sum(w_p4_out) + self.epsilon)
        p3_out_down = self._downsample(p3_out, p4.shape[2:])
        p4_out = self.conv_p4_out(weight_p4_out[0] * p4 + weight_p4_out[1] * p4_td + weight_p4_out[2] * p3_out_down)

        # P5_out = conv( w*p5 + w*Resize(p4_out) )  <- Note P5_out has only 2 inputs!
        w_p5_out = F.relu(self.p5_out_w2)
        weight_p5_out = w_p5_out / (torch.sum(w_p5_out) + self.epsilon)
        p4_out_down = self._downsample(p4_out, p5.shape[2:])
        p5_out = self.conv_p5_out(weight_p5_out[0] * p5 + weight_p5_out[1] * p4_out_down)

        return [p2_out, p3_out, p4_out, p5_out]


if __name__ == "__main__":
    feats = [
        torch.randn(1, 256, 160, 160),  # P2
        torch.randn(1, 256, 80, 80),  # P3
        torch.randn(1, 256, 40, 40),  # P4
        torch.randn(1, 256, 20, 20),  # P5
    ]
    m = BiFPNLayer(num_channels=256, num_levels=4)
    outs = m(feats)
    print("BiFPNLayer test results:")
    for i, o in enumerate(outs):
        print(f"P{i + 2}_out: {o.shape} (expected: {feats[i].shape})")
        assert o.shape == feats[i].shape, f"Mismatch P{i + 2}: got {o.shape}, expected {feats[i].shape}"
    print("Test passed!")
