"""The Blind2Unblind backbone network.

This is a small U-Net -- an ordinary convolutional encoder-decoder, with NO
architectural blind-spot trick (no dilated/shifted convolutions like earlier
blind-spot denoisers such as Noise2Void). That's the point of Blind2Unblind:
the blind-spot property comes entirely from masking the *input* (see
`denoiser_masker.py`), so the network itself can be a completely normal,
full-receptive-field U-Net. Architecture (depth=5, base width=48, nearest-
neighbor upsampling via pixel-shuffle-style repeat) matches the network used
in Wang et al., "Blind2Unblind: Self-Supervised Image Denoising with Visible
Blind Spots", CVPR 2022 -- verified against the authors' released code
(github.com/zejinwang/Blind2Unblind) for correctness, reimplemented here
rather than copied (the repo ships no LICENSE file).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    """Conv -> LeakyReLU, the network's only building block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, slope: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=True),
            nn.LeakyReLU(slope, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Nearest-neighbor 2x upsample (via repeat, not transposed conv -- avoids
    checkerboard artifacts) + skip concat + two conv blocks.
    """

    def __init__(self, in_channels: int, out_channels: int, slope: float = 0.1):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, slope=slope)
        self.conv2 = ConvBlock(out_channels, out_channels, slope=slope)

    @staticmethod
    def _upsample2x(x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        x = x.reshape(n, c, h, 1, w, 1)
        x = x.repeat(1, 1, 1, 2, 1, 2)
        return x.reshape(n, c, h * 2, w * 2)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self._upsample2x(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        return self.conv2(x)


class DenoiserUNet(nn.Module):
    """Blind2Unblind's backbone. Single-channel in/out for grayscale SSS
    (the original ships in_channels=out_channels=3 for RGB; sonar intensity
    is single-channel, so both default to 1 here).
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, depth: int = 5, base_width: int = 48, slope: float = 0.1):
        super().__init__()
        self.depth = depth
        wf = base_width

        self.head = nn.Sequential(ConvBlock(in_channels, wf, 3, slope), ConvBlock(wf, wf, 3, slope))
        self.down_path = nn.ModuleList([ConvBlock(wf, wf, 3, slope) for _ in range(depth)])

        self.up_path = nn.ModuleList()
        for i in range(depth):
            if i != depth - 1:
                in_ch = wf * 2 if i == 0 else wf * 3
            else:
                in_ch = wf * 2 + in_channels
            self.up_path.append(UpBlock(in_ch, wf * 2, slope))

        self.tail = nn.Sequential(
            ConvBlock(2 * wf, 2 * wf, 1, slope),
            ConvBlock(2 * wf, 2 * wf, 1, slope),
            nn.Conv2d(2 * wf, out_channels, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [x]
        h = self.head(x)
        for i, down in enumerate(self.down_path):
            h = F.max_pool2d(h, 2)
            if i != len(self.down_path) - 1:
                skips.append(h)
            h = down(h)

        for i, up in enumerate(self.up_path):
            h = up(h, skips[-i - 1])

        return self.tail(h)
