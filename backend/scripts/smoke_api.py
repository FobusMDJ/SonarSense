"""Exercise upload, WebSocket progress, outputs, reports, and honest map behavior."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import httpx
import websockets

BASE = os.environ.get("SONARSENSE_API_URL", "http://127.0.0.1:8000").rstrip("/")
WS_BASE = BASE.replace("http://", "ws://").replace("https://", "wss://")
fixture_dir = Path(__file__).resolve().parents[1] / "runtime" / "smoke"


async def main() -> None:
    async with httpx.AsyncClient(timeout=120) as client:
        for attempt in range(30):
            try:
                health = await client.get(f"{BASE}/health")
                health.raise_for_status()
                break
            except httpx.HTTPError:
                if attempt == 29:
                    raise
                await asyncio.sleep(1)
        with (fixture_dir / "survey_fixture.zip").open("rb") as archive:
            response = await client.post(
                f"{BASE}/logs/upload_zip?yolo_conf=0.99&denoise_method=none&contrast_method=none",
                files={"archive": ("survey_fixture.zip", archive, "application/zip")},
            )
        response.raise_for_status()
        log_id = response.json()["log_id"]

        stages = []
        async with websockets.connect(f"{WS_BASE}/ws/logs/{log_id}") as socket:
            while True:
                event = json.loads(await asyncio.wait_for(socket.recv(), timeout=120))
                stages.append(event["stage"])
                if event["stage"] in {"done", "error", "closed"}:
                    break

        log = (await client.get(f"{BASE}/logs/{log_id}")).json()
        assert log["status"] == "done", log
        assert log["source_format"] == "zip+metadata", log
        assert "detect" in stages and "vae" in stages and "done" in stages, stages
        stats = (await client.get(f"{BASE}/logs/{log_id}/stats")).json()
        assert stats["confidence_threshold"] == 0.99
        assert stats["mean_inference_ms"] is not None and stats["fps"] is not None
        vae = (await client.get(f"{BASE}/logs/{log_id}/vae_stats")).json()
        assert vae["n_frames_analyzed"] == 1
        frame_id = vae["most_anomalous_frames"][0]["frame_record_id"]
        for filename in (
            "01_original.png", "02_reconstruction.png", "03_anomaly_overlay.png",
            "04_difference_heatmap.png", "05_edge_contour_map.png",
            "06_difference_map_legend.png", "07_3d_anomaly_surface.png",
        ):
            panel = await client.get(f"{BASE}/logs/{log_id}/frames/{frame_id}/vae/{filename}")
            assert panel.status_code == 200 and panel.headers["content-type"].startswith("image/")
        for extension in ("json", "csv", "geojson", "pdf"):
            assert (await client.get(f"{BASE}/logs/{log_id}/report.{extension}")).status_code == 200
        feature_collection = (await client.get(f"{BASE}/logs/{log_id}/map")).json()
        assert feature_collection["type"] == "FeatureCollection"
        print(json.dumps({"log_id": log_id, "stages": stages, "detections": stats["n_detections"]}))


if __name__ == "__main__":
    asyncio.run(main())
