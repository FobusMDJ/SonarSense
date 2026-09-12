"""Load all released checkpoints and run minimal CPU compatibility checks."""

from pathlib import Path
import sys

import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.backend.model_release import DETECTOR_CLASSES, get_vae_model, get_yolo_model, validate_present_checkpoints
from src.preprocessing.denoiser_model import DenoiserUNet
from src.vae.vae_analysis import run_vae

paths = {
    "detector": root / "best.pt",
    "vae": root / "src/vae/vae_epoch100.pth",
    "denoiser": root / "models/B2Ueph2.pth",
}
validate_present_checkpoints(paths)
device = torch.device("cpu")
fixture = np.zeros((512, 512), dtype=np.uint8)

yolo = get_yolo_model(str(paths["detector"]))
assert list(yolo.names.values()) == DETECTOR_CLASSES, yolo.names
_ = yolo.predict(source=fixture, conf=0.99, imgsz=640, device="cpu", verbose=False)

vae = get_vae_model(str(paths["vae"]), "cpu")
vae_result = run_vae(vae, fixture, device)
assert vae_result["reconstruction"].shape == fixture.shape

checkpoint = torch.load(paths["denoiser"], map_location="cpu", weights_only=True)
denoiser = DenoiserUNet(base_width=int(checkpoint.get("base_width", 32)))
denoiser.load_state_dict(checkpoint["model_state_dict"])
denoiser.eval()
with torch.no_grad():
    output = denoiser(torch.zeros(1, 1, 64, 64))
assert output.shape == (1, 1, 64, 64)
print("YOLO, VAE, and experimental B2U checkpoints are architecture-compatible on CPU.")
