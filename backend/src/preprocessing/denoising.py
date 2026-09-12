"""SSS denoising: Lee filter (fully implemented) plus scaffolding for the
two learned denoisers named in the pipeline design (Blind2Unblind, DSPNet).

Why three options: side-scan sonar speckle is multiplicative and locally
correlated, which classical filters (Lee) handle reasonably well without
any training data -- useful today, and a solid baseline to compare learned
methods against later. Blind2Unblind and DSPNet are self-supervised /
learned denoisers that should outperform Lee once trained on this
project's own SSS data, but they need that training pass first, so they're
scaffolded here with a clear interface and a loud, explicit fallback
rather than a silent wrong answer.

torch is imported lazily (only inside the learned-denoiser classes) so the
Lee-filter path -- the default, and the only one usable before any model
is trained -- never requires torch to be installed at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import uniform_filter

from src.utils.config import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------
# Lee filter -- fully implemented, no training required.
# --------------------------------------------------------------------------

def estimate_noise_variance(image: np.ndarray) -> float:
    """Fast global noise-variance estimate (Immerkaer 1996).

    Convolves with a Laplacian-of-a-constant kernel, whose response on a
    noise-free image is ~0 almost everywhere, so its energy is a reasonable
    proxy for noise power even without a clean reference. Used as the Lee
    filter's noise_variance when the caller doesn't supply one.
    """
    image = image.astype(np.float64)
    h, w = image.shape
    laplacian_kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    from scipy.signal import convolve2d

    response = convolve2d(image, laplacian_kernel, mode="same", boundary="symm")
    sigma = np.sqrt(np.pi / 2) * np.sum(np.abs(response)) / (6 * (w - 2) * (h - 2))
    return float(sigma**2)


def lee_filter(image: np.ndarray, window_size: int = 7, noise_variance: Optional[float] = None) -> np.ndarray:
    """Adaptive Lee filter for speckle reduction.

    Locally blends the pixel value toward the local mean, weighted by how
    much of the local variance looks like signal vs. noise: near-flat
    regions (low local variance) get smoothed hard, high-contrast edges
    and textured targets (local variance >> noise variance) are left
    mostly untouched. This is what keeps debris edges sharp while
    knocking down speckle in open seafloor.
    """
    if image.ndim != 2:
        raise ValueError("lee_filter expects a single-channel (grayscale) image")

    image_f = image.astype(np.float64)
    local_mean = uniform_filter(image_f, size=window_size)
    local_sqr_mean = uniform_filter(image_f**2, size=window_size)
    local_var = np.maximum(local_sqr_mean - local_mean**2, 0.0)

    if noise_variance is None:
        noise_variance = estimate_noise_variance(image_f)
        logger.debug("Lee filter: estimated noise variance = %.3f", noise_variance)

    weight = local_var / (local_var + noise_variance + 1e-8)
    output = local_mean + weight * (image_f - local_mean)
    return np.clip(output, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Blind2Unblind -- self-supervised blind-spot denoiser (scaffold).
# Reference: Wang et al., "Blind2Unblind: Self-Supervised Image Denoising
# with Visible Blind Spots", CVPR 2022. Trains directly on noisy SSS images
# (no clean targets needed), which fits this project well since a truly
# "clean" sonar reference doesn't exist.
# --------------------------------------------------------------------------

class Blind2UnblindDenoiser:
    """Blind2Unblind denoiser. Train it with `train_denoiser.py` on real,
    unlabeled noisy SSS images (see that file's docstring), which produces a
    checkpoint bundling the trained weights with `final_beta` -- the
    annealed blend weight training settled on, needed to reproduce the
    paper's inference recipe exactly (see `denoise()` below).

    NOT YET TRAINED until `weights_path` points at such a checkpoint.
    `denoise()` deliberately raises until then -- silently returning noisy
    input dressed up as "denoised" would be worse than failing loudly.
    """

    def __init__(self, weights_path: Optional[str | Path] = None, device: Optional[str] = None):
        self.weights_path = Path(weights_path) if weights_path else None
        self._model = None
        self._masker = None
        self._final_beta = None
        # Resolved lazily in _load() (needs torch imported to check
        # cuda.is_available()); None here just means "not loaded yet".
        self._device = device
        if self.weights_path and self.weights_path.exists():
            self._load(self.weights_path)

    def _load(self, weights_path: Path) -> None:
        import torch

        from src.preprocessing.denoiser_masker import GlobalAwareMasker
        from src.preprocessing.denoiser_model import DenoiserUNet

        # Explicit device wins; otherwise use the GPU if one's visible to
        # this process. Inference previously always ran on CPU regardless
        # of what training used, because nothing here ever called .to(...)
        # -- torch.load(map_location="cpu") plus a model/tensors that never
        # move off it. That's fixed by resolving a real device and moving
        # both the model and every input tensor onto it below; the masker
        # (GlobalAwareMasker) needs no change since it already derives its
        # device from the tensor it's given (see denoiser_masker.py).
        resolved = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(resolved)

        checkpoint = torch.load(weights_path, map_location=self._device, weights_only=True)
        model = DenoiserUNet(in_channels=1, out_channels=1, base_width=checkpoint.get("base_width", 48))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model.to(self._device)

        self._model = model
        self._masker = GlobalAwareMasker(width=checkpoint.get("width", 4))
        self._final_beta = checkpoint["final_beta"]
        logger.info(
            "Loaded Blind2Unblind weights from %s (epoch %s, final_beta=%.2f, device=%s)",
            weights_path, checkpoint.get("epoch", "?"), self._final_beta, self._device,
        )

    def denoise(self, image: np.ndarray, fast: bool = False) -> np.ndarray:
        """Run the trained network on `image`.

        `fast=False` (default) is the paper's actual inference recipe: blend
        the width**2-view blind-spot reconstruction with a plain visible
        pass, weighted by the beta training annealed to. That recipe costs
        width**2 + 1 forward passes per image -- real algorithmic cost, not
        an implementation accident -- which is fine for offline/GPU use but
        not for an edge/embedded target with no cloud fallback.

        `fast=True` skips the masked-view reconstruction entirely and
        returns the single visible-pass output on its own. Measured against
        the full recipe on real SubPipe frames: ~21x faster (the masked
        reconstruction is exactly what's skipped), mean pixel difference
        ~1.3/255, visually indistinguishable in side-by-side comparison.
        This works because the masking/blending apparatus is what lets B2U
        *train* without clean/noisy pairs -- nothing requires reproducing it
        at inference, since the trained network is itself already a valid
        denoiser once training is done. Worth re-verifying this gap stays
        small as training progresses past the epoch this was measured on;
        it isn't guaranteed to hold at every checkpoint.
        """
        if self._model is None:
            raise NotImplementedError(
                "Blind2UnblindDenoiser has no trained weights yet. Train it with "
                "`python -m src.preprocessing.train_denoiser --data_dir <real SSS images>` "
                "and pass the resulting checkpoint as weights_path, or use method='lee' "
                "in the meantime."
            )
        import torch
        import torch.nn.functional as F

        h, w = image.shape
        # Pad up to a multiple of 32 (5 UNet downsamples) so the encoder/
        # decoder skip shapes line up; cropped back off before returning.
        pad_h = (-h) % 32
        pad_w = (-w) % 32
        x = torch.from_numpy(image.astype(np.float32) / 255.0)[None, None].to(self._device)
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        with torch.no_grad():
            if fast:
                # Single forward pass, no masking -- see docstring above.
                pred_mid = self._model(x).clamp(0, 1)
            else:
                n, c, ph, pw = x.shape
                if self._device.type == "cuda":
                    # GPUs have memory to spare for this, and stacking all
                    # width**2 views into one batch lets them run in parallel
                    # instead of width**2 sequential kernel launches.
                    net_input, mask = self._masker.all_views(x)
                    blind_output = (self._model(net_input) * mask).view(n, -1, c, ph, pw).sum(dim=1)
                else:
                    # Same math as the batched path above, but one masked view
                    # at a time instead of stacking all width**2 views into a
                    # single batch -- that stacked batch is what OOMs on CPU
                    # (no GPU headroom to absorb a 16x activation-memory
                    # spike), even at a modest few-hundred-pixel resolution.
                    # Trades wall-clock time (still width**2 forward passes,
                    # just sequential) for a peak memory footprint close to
                    # one single-image forward pass.
                    blind_output = torch.zeros_like(x)
                    for view_index in range(self._masker.width * self._masker.width):
                        view, view_mask = self._masker.masked_view(x, view_index)
                        blind_output = blind_output + self._model(view) * view_mask
                visible_output = self._model(x)
                # The paper's actual inference output: a blend of the blind-spot
                # reconstruction and the fully-visible pass, weighted by the beta
                # training annealed to -- not just one branch alone.
                beta = self._final_beta
                pred_mid = (blind_output + beta * visible_output) / (1 + beta)
                pred_mid = pred_mid.clamp(0, 1)

        if pad_h or pad_w:
            pred_mid = pred_mid[:, :, :h, :w]
        # Back to CPU before touching numpy -- a CUDA tensor can't convert
        # directly.
        return (pred_mid[0, 0].cpu().numpy() * 255.0 + 0.5).clip(0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# DSPNet -- scaffold only. Architecture intentionally left for a follow-up
# pass once we've picked a specific DSPNet variant to target; interface
# matches Blind2UnblindDenoiser so pipeline.py doesn't need to change when
# it's filled in.
# --------------------------------------------------------------------------

class DSPNetDenoiser:
    """Scaffold for a DSPNet-style denoiser. NOT YET IMPLEMENTED."""

    def __init__(self, weights_path: Optional[str | Path] = None, device: Optional[str] = None):
        self.weights_path = Path(weights_path) if weights_path else None
        self.device = device  # unused until this scaffold is implemented

    def denoise(self, image: np.ndarray, fast: bool = False) -> np.ndarray:
        raise NotImplementedError(
            "DSPNetDenoiser is a scaffold pending architecture selection. "
            "Use method='lee' or 'blind2unblind' in the meantime."
        )


_LEARNED_DENOISERS: dict = {}


def denoise(
    image: np.ndarray,
    method: str = "lee",
    window_size: int = 7,
    weights_path: Optional[str | Path] = None,
    fallback_on_missing_weights: bool = True,
    device: Optional[str] = None,
    fast: bool = False,
) -> np.ndarray:
    """Dispatch to the requested denoising method.

    Learned methods (`blind2unblind`, `dspnet`) fall back to the Lee filter
    with a loud warning when no trained weights are available and
    `fallback_on_missing_weights=True` (the default), so the pipeline stays
    runnable end-to-end before training happens. Set it False to fail hard
    instead, e.g. in a CI check that should catch "forgot to train the model".

    `device` picks where the learned model runs ("cuda", "cpu", or omitted
    to auto-detect a GPU). Included in the cache key so a request for one
    device never silently returns an instance already loaded on another.

    `fast` (blind2unblind only) skips the width**2-view masked reconstruction
    and returns a single forward pass -- see Blind2UnblindDenoiser.denoise's
    docstring for the measured speed/quality tradeoff. Ignored by `dspnet`
    and `lee`.
    """
    if method == "none":
        return image
    if method == "lee":
        return lee_filter(image, window_size=window_size)

    if method in ("blind2unblind", "dspnet"):
        cls = Blind2UnblindDenoiser if method == "blind2unblind" else DSPNetDenoiser
        cache_key = (method, str(weights_path), device)
        if cache_key not in _LEARNED_DENOISERS:
            _LEARNED_DENOISERS[cache_key] = cls(weights_path=weights_path, device=device)
        denoiser = _LEARNED_DENOISERS[cache_key]
        try:
            return denoiser.denoise(image, fast=fast)
        except NotImplementedError as exc:
            if not fallback_on_missing_weights:
                raise
            logger.warning("%s -- falling back to Lee filter for this call.", exc)
            return lee_filter(image, window_size=window_size)

    raise ValueError(f"Unknown denoising method '{method}'. Choose from: none, lee, blind2unblind, dspnet.")
