"""Blind2Unblind's "global-aware mask mapper".

Core idea: rather than restricting the network's receptive field to create a
blind spot (as earlier blind-spot denoisers did), B2U masks the *input*.
The image is tiled into non-overlapping width x width cells; within each
cell, exactly one pixel position is chosen and replaced by an interpolation
of its neighbors before the (ordinary, full-receptive-field) network sees
it. The network's prediction at that position is then a genuine blind-spot
estimate -- it never saw the true value there.

"Global-aware" specifically means: instead of masking only a random subset
of pixels each training step (as Noise2Void/Neighbor2Neighbor do), B2U runs
width**2 passes per step, one per cell-position, so that -- summed together
-- every single pixel in the image gets a blind-spot prediction every
step. That full coverage per step is what the name refers to, and it's why
training one step here costs width**2 forward passes.

Verified against the official implementation (github.com/zejinwang/
Blind2Unblind, Wang et al. CVPR 2022) for exact behavior; reimplemented
independently (that repo ships no LICENSE file, so nothing here is copied
from it) -- variable names and structure are our own.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def depth_to_space(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """(N, C*r^2, H, W) -> (N, C, H*r, W*r). Thin wrapper for readability."""
    return F.pixel_shuffle(x, block_size)


def _fixed_position_mask(n: int, h: int, w: int, width: int, index: int, device) -> torch.Tensor:
    """Boolean-as-int64 mask marking, in every width x width cell across the
    whole (n, h, w) volume, the single pixel at flat position `index` within
    that cell (0 <= index < width**2). Shape (n, 1, h, w).
    """
    cells_h, cells_w = h // width, w // width
    flat = torch.zeros(n * cells_h * cells_w * width * width, dtype=torch.int64, device=device)
    stride = width * width
    hit_positions = index + torch.arange(0, n * cells_h * cells_w * stride, step=stride, device=device)
    flat[hit_positions] = 1
    # (n, cells_h, cells_w, width**2) -> (n, width**2, cells_h, cells_w) -> pixel_shuffle -> (n, 1, h, w)
    grid = flat.view(n, cells_h, cells_w, width * width).permute(0, 3, 1, 2).float()
    return depth_to_space(grid, block_size=width).to(torch.int64)


def interpolate_masked(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace masked (blind) pixels with a weighted average of their 8
    neighbors (corners weighted half as much as edges; the pixel itself
    excluded), leaving unmasked pixels untouched.
    """
    n, c, h, w = image.shape
    kernel = torch.tensor(
        [[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.5]],
        dtype=image.dtype, device=image.device,
    )
    kernel = (kernel / kernel.sum()).view(1, 1, 3, 3)

    interpolated = F.conv2d(image.reshape(n * c, 1, h, w), kernel, padding=1)
    interpolated = interpolated.view_as(image)

    mask_f = mask.to(image.dtype)
    return interpolated * mask_f + image * (1 - mask_f)


class GlobalAwareMasker:
    """Generates the width**2 masked views of an image used for one B2U
    training step, and applies the corresponding fixed-position mask.
    """

    def __init__(self, width: int = 4):
        self.width = width

    def masked_view(self, image: torch.Tensor, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """One masked view: pixel `index` of every width x width cell is
        blinded (interpolated away). Returns (masked_image, mask).
        """
        n, c, h, w = image.shape
        mask = _fixed_position_mask(n, h, w, self.width, index, image.device)
        return interpolate_masked(image, mask), mask

    def all_views(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """All width**2 masked views, stacked into the batch dimension --
        this is what one B2U training/inference step actually runs.
        Returns (net_input, mask), each of batch size n * width**2.
        """
        n, c, h, w = image.shape
        cells = self.width * self.width
        views = torch.empty((n, cells, c, h, w), device=image.device, dtype=image.dtype)
        masks = torch.empty((n, cells, 1, h, w), device=image.device, dtype=torch.int64)
        for index in range(cells):
            masked, mask = self.masked_view(image, index)
            views[:, index] = masked
            masks[:, index] = mask
        return views.view(-1, c, h, w), masks.view(-1, 1, h, w)
