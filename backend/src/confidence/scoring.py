"""Confidence scoring & noise-filtering module.

Produces a single 0-100% confidence score per detection, per the challenge
brief: "An algorithmic pipeline or pre-processing filter that minimizes
false positives caused by natural acoustic shadows or rock clusters,
outputting a clear confidence score (0% to 100%) for every detected
anomaly."

DESIGN NOTE -- what feeds the score, and what deliberately does NOT:

  - YOLO's own detection confidence is the primary signal (0.7 weight):
    it is the one number in this pipeline that's actually been trained
    end-to-end on this exact task.

  - VAE whole-image anomaly percentile is a secondary, CLASS-ROUTED
    modifier (up to +/-15 points), not a coequal signal. This weighting is
    not invented here -- it's carried over directly from the empirical
    cross-tab in claude/phase1-baseline-error-analysis.md: whole-image VAE
    reconstruction-error AUROC for "contains debris" was 0.70 (moderate),
    and when split by class it was informative for shipwreck (missed
    wrecks ranked MORE anomalous than TPs -- AUROC-consistent signal) and
    NOT informative for ghost_net (missed nets ranked at background-normal
    anomaly, i.e. no signal). So: shipwreck gets the full modifier weight,
    ghost_net gets ~0, other classes (human/cylinder/pipe) get a small
    default weight since Phase 1 didn't test them directly -- flagged as
    such in CLASS_VAE_WEIGHT below rather than silently assumed.

  - contrast_proxy / shadow_proxy (the pixel-only heuristics from
    src/evaluation/error_analysis.py) are deliberately NOT folded into the
    score as a positive signal. Phase 1 found contrast_proxy was actually
    HIGHER for Bkg-FP (23.3) than for TP (11.2) on this dataset -- the
    opposite of "more contrast = more confident." Baking that in as a
    score booster would contradict the project's own evidence. They are
    still computed and attached to every detection's breakdown (useful
    context for a human reviewer, and for future re-calibration), just not
    used to move the number.

  - "Noise filtering": rather than a hard reject rule (which would need
    more validation than this project currently has -- see
    claude/nadir-gap-validation.md for what happened the one time an
    unvalidated heuristic was nearly turned into a filter), low-confidence
    detections are labeled, not deleted. `LOW_CONFIDENCE_THRESHOLD` below
    is what the API/dashboard use to visually flag or filter them, but the
    detection and its evidence are always persisted.

This is a v1 heuristic, explicitly documented as such -- exactly the
project's standing convention (see error_analysis.py, nadir_gap.py). It
should be recalibrated once real field logs with known ground truth are
available; nothing here should be read as a validated probability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.evaluation.error_analysis import contrast_and_shadow_proxy, geometry_proxy

# Per-class VAE modifier weight (see module docstring). 1.0 = full weight
# (+/-15 pts based on whole-image anomaly percentile), 0.0 = ignored.
CLASS_VAE_WEIGHT = {
    "shipwreck": 1.0,   # Phase 1: informative (missed wrecks ranked more anomalous than TPs)
    "ghost_net": 0.0,   # Phase 1: NOT informative (missed nets ranked background-normal)
    "human": 0.4,       # not tested in Phase 1 -- conservative default, not a validated weight
    "cylinder": 0.4,    # not tested in Phase 1 -- conservative default, not a validated weight
    "pipe": 0.4,        # not tested in Phase 1 -- conservative default, not a validated weight
}
DEFAULT_VAE_WEIGHT = 0.4

YOLO_WEIGHT = 0.70
VAE_MODIFIER_MAX_POINTS = 15.0  # max +/- swing the VAE term can contribute, out of 100

LOW_CONFIDENCE_THRESHOLD = 40.0  # dashboard/report default for "flag as likely noise"


@dataclass
class ConfidenceResult:
    score: float  # 0-100
    label: str  # "high" | "medium" | "low"
    breakdown: dict = field(default_factory=dict)


def _vae_modifier(vae_percentile: Optional[float], class_name: str) -> float:
    """vae_percentile: 0 = most anomalous image in its population, 1 = least
    anomalous (same convention as claude/phase1-baseline-error-analysis.md).
    Returns a value in [-VAE_MODIFIER_MAX_POINTS, +VAE_MODIFIER_MAX_POINTS].

    More anomalous (lower percentile) -> higher modifier, scaled by how much
    Phase 1 evidence supports that relationship for this specific class.
    """
    if vae_percentile is None:
        return 0.0
    weight = CLASS_VAE_WEIGHT.get(class_name, DEFAULT_VAE_WEIGHT)
    # percentile 0 (most anomalous) -> +1.0, percentile 1 (least anomalous) -> -1.0
    centered = (0.5 - vae_percentile) * 2.0
    return centered * VAE_MODIFIER_MAX_POINTS * weight


def score_detection(
    yolo_conf: float,
    class_name: str,
    xyxy: list[float],
    gray_image=None,
    vae_whole_image_percentile: Optional[float] = None,
) -> ConfidenceResult:
    """Compute the 0-100 confidence score for one YOLO detection.

    yolo_conf: YOLO's own confidence, 0-1.
    class_name: detected class (drives the class-routed VAE weight).
    xyxy: detection box in absolute pixel coords, for the physics-proxy
        context computation (not scored, see module docstring).
    gray_image: the full grayscale frame the box came from -- needed to
        compute contrast_proxy/shadow_proxy context. If None, proxies are
        omitted from the breakdown (score is unaffected either way).
    vae_whole_image_percentile: this frame's VAE anomaly-error percentile
        rank within the current log/batch (0=most anomalous). Optional --
        if not supplied (e.g. VAE stage skipped), the VAE term is 0.
    """
    base = yolo_conf * 100.0 * YOLO_WEIGHT / 1.0  # scaled below with the 100-pt budget
    # YOLO contributes up to YOLO_WEIGHT*100 points, scaled by its own confidence.
    yolo_points = yolo_conf * 100.0 * YOLO_WEIGHT

    vae_points = _vae_modifier(vae_whole_image_percentile, class_name)

    # Remaining budget after YOLO's weighted share is filled by a neutral
    # baseline so a perfect-confidence YOLO detection with no VAE data still
    # lands near 100, not capped at YOLO_WEIGHT*100.
    neutral_fill = (1.0 - YOLO_WEIGHT) * 100.0 * yolo_conf

    raw_score = yolo_points + neutral_fill + vae_points
    score = max(0.0, min(100.0, raw_score))

    breakdown = {
        "yolo_conf": round(yolo_conf, 4),
        "yolo_points": round(yolo_points, 2),
        "neutral_fill_points": round(neutral_fill, 2),
        "vae_whole_image_percentile": vae_whole_image_percentile,
        "vae_points": round(vae_points, 2),
        "vae_class_weight_used": CLASS_VAE_WEIGHT.get(class_name, DEFAULT_VAE_WEIGHT),
    }

    if gray_image is not None:
        geo = geometry_proxy(xyxy)
        phys = contrast_and_shadow_proxy(gray_image, xyxy)
        breakdown["context_only_not_scored"] = {
            "aspect_ratio": round(geo["aspect_ratio"], 3),
            "contrast_proxy": phys["contrast_proxy"],
            "shadow_proxy": phys["shadow_proxy"],
            "shadow_side": phys["shadow_side"],
            "note": ("Not used to compute the score -- Phase 1 found contrast_proxy "
                     "was HIGHER for false positives than true positives on this "
                     "dataset. Shown for human review only."),
        }

    label = "high" if score >= 70 else ("medium" if score >= LOW_CONFIDENCE_THRESHOLD else "low")
    return ConfidenceResult(score=round(score, 1), label=label, breakdown=breakdown)
