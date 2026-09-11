"""Convolutional VAE for unsupervised seafloor-background modeling.

Trained only on "hard negative" patches (real seafloor, no known marine
debris -- see train_vae.py), the reconstruction error at inference is the
anomaly signal: a patch the VAE reconstructs well looks like ordinary
seafloor it has seen the statistics of; a patch it reconstructs poorly
(a pipe, a wreck, debris) doesn't match anything in that learned
distribution. This is the standard VAE-for-anomaly-detection setup, not a
generative-quality target -- blurry reconstructions of normal seafloor are
fine and expected; the metric that matters downstream is reconstruction
error separation between normal and anomalous patches, not image sharpness.

Symmetric encoder/decoder, `channels` shared between both so the decoder is
a mirror of the encoder. Default channels give 6 stride-2 downsamples
(512 -> 256 -> 128 -> 64 -> 32 -> 16 -> 8), so `input_size` must be
divisible by 2**len(channels) (512 satisfies this by construction, matching
pipeline.py's target_size).

`channels` is deliberately exposed (like DenoiserUNet's base_width) so a
smaller variant can be trained for edge/embedded deployment -- narrower
channels shrink both the parameter count and the per-frame compute, at the
usual cost of reconstruction fidelity; that tradeoff should be revisited
once a specific target device is chosen (see prepare_vae_dataset.py and
train_vae.py docstrings for the same caveat on the denoiser side).
"""

from __future__ import annotations

import torch
from torch import nn


class ConvVAE(nn.Module):
    def __init__(
        self,
        input_size: int = 512,
        latent_dim: int = 256,
        channels: tuple[int, ...] = (32, 64, 128, 256, 256, 256),
    ):
        super().__init__()
        n_down = len(channels)
        if input_size % (2 ** n_down) != 0:
            raise ValueError(f"input_size={input_size} must be divisible by 2**{n_down}={2**n_down}")

        self.input_size = input_size
        self.latent_dim = latent_dim
        self.channels = channels
        self.bottleneck_size = input_size // (2 ** n_down)  # spatial extent at the deepest layer
        flat_dim = channels[-1] * self.bottleneck_size * self.bottleneck_size

        # --- encoder: 1 -> channels[0] -> ... -> channels[-1], each halving spatial size ---
        enc_layers = []
        in_ch = 1
        for out_ch in channels:
            enc_layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_ch = out_ch
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        # --- decoder: mirror image of the encoder ---
        self.fc_decode = nn.Linear(latent_dim, flat_dim)
        dec_layers = []
        rev_channels = list(reversed(channels))
        for i in range(len(rev_channels) - 1):
            dec_layers += [
                nn.ConvTranspose2d(rev_channels[i], rev_channels[i + 1], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(rev_channels[i + 1]),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        dec_layers += [
            nn.ConvTranspose2d(rev_channels[-1], 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # output in [0, 1], matching normalized input
        ]
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu  # deterministic at eval time -- reconstruction error should reflect the model, not sampling noise

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(-1, self.channels[-1], self.bottleneck_size, self.bottleneck_size)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample mean-squared reconstruction error, deterministic
        (uses mu directly, no sampling) -- this is the anomaly score used
        at inference, not a training-loss term.
        """
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x)
            recon = self.decode(mu)
            return ((recon - x) ** 2).flatten(1).mean(dim=1)
