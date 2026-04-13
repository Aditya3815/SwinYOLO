import torch
import torch.nn as nn


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return x * self.relu(x + 3) / 6


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        was_3d = x.dim() == 3
        if was_3d:
            x = x.unsqueeze(0)

        identity = x

        _n, _c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        if was_3d:
            out = out.squeeze(0)

        return out


class CoordAttMulti(nn.Module):
    """Applies Coordinate Attention separately to a list of feature maps."""

    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.ca = CoordAtt(inp, oup, reduction)

    def forward(self, x):
        print("CoordAttMulti inputs:", [xi.shape for xi in x])
        return [self.ca(xi) for xi in x]


if __name__ == "__main__":
    x = torch.randn(1, 256, 40, 40)
    m = CoordAtt(256, 256)
    out = m(x)
    assert out.shape == (1, 256, 40, 40), f"Expected (1,256,40,40), got {out.shape}"
    print(f"CoordAtt test passed. Output shape: {out.shape}")
