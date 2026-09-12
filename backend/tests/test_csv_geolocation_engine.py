import csv
import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from src.backend import db_backend as db
from src.backend import main
from src.geolocation.csv_engine import geolocate_csv_detection, run_csv_pipeline_to_detections


class GeolocateCsvDetectionMathTests(unittest.TestCase):
    """Same worked example used to explain this engine (pixel_x=1450,
    bbox 80x30px, nav lat=19.0012/lon=72.8015/heading=45/altitude=8/depth=52,
    image_width=2000, range_per_pixel=0.05) -- verifies the lat/lon/ground-range
    math reproduces those numbers exactly, and that depth_m now includes
    altitude (60.0 = 52 + 8), which the originally uploaded script did not do
    (it would have returned 52.0)."""

    def setUp(self):
        self.det = {"pixel_x": 1450, "bbox_w": 80, "bbox_h": 30, "class": "pipe", "confidence": 0.9}
        self.nav = {"lat": 19.0012, "lon": 72.8015, "heading": 45.0, "altitude_m": 8.0, "depth_m": 52.0,
                    "ping_id": "P0001"}

    def test_worked_example_matches_by_hand_computation(self):
        geo = geolocate_csv_detection(self.det, self.nav, range_per_pixel_m=0.05, image_width=2000)
        self.assertEqual(geo["side"], "starboard")
        self.assertAlmostEqual(geo["ground_range_m"], 21.03, places=1)
        self.assertAlmostEqual(geo["lat"], 19.0010664, places=6)
        self.assertAlmostEqual(geo["lon"], 72.8016413, places=6)
        self.assertAlmostEqual(geo["width_m"], 4.0, places=2)
        self.assertAlmostEqual(geo["length_m"], 1.5, places=2)

    def test_depth_fix_adds_altitude_to_nav_depth(self):
        geo = geolocate_csv_detection(self.det, self.nav, range_per_pixel_m=0.05, image_width=2000)
        self.assertAlmostEqual(geo["depth_m"], 60.0, places=2)  # 52.0 + 8.0, NOT 52.0

    def test_port_side_when_pixel_left_of_center(self):
        det = dict(self.det, pixel_x=550)
        geo = geolocate_csv_detection(det, self.nav, range_per_pixel_m=0.05, image_width=2000)
        self.assertEqual(geo["side"], "port")

    def test_ground_range_clamped_to_zero_near_nadir(self):
        det = dict(self.det, pixel_x=1005)  # 5px off center * 0.05 = 0.25m slant, well under altitude
        geo = geolocate_csv_detection(det, self.nav, range_per_pixel_m=0.05, image_width=2000)
        self.assertEqual(geo["ground_range_m"], 0.0)


class RunCsvPipelineToDetectionsTests(unittest.TestCase):
    """run_csv_pipeline_to_detections() is what main.py's /logs/geolocate_csv
    endpoint calls -- checks the join/skip behavior and that every row is
    shaped correctly for db.insert_detection()."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.detections_csv = tmp / "detections.csv"
        self.nav_csv = tmp / "nav.csv"

        with open(self.detections_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ping_id", "pixel_x", "bbox_w", "bbox_h", "class", "confidence"])
            w.writerow(["1000", "1450", "80", "30", "class_6", "0.95"])
            w.writerow(["9999", "1450", "80", "30", "class_6", "0.95"])  # no matching nav row -> skipped

        with open(self.nav_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ping_id", "lat", "lon", "heading", "altitude_m", "depth_m", "timestamp"])
            w.writerow(["1000", "19.0012", "72.8015", "45.0", "8.0", "52.0", "2026-09-10T10:00:00Z"])

    def tearDown(self):
        self._tmp.cleanup()

    def test_unmatched_ping_id_is_skipped_not_stored(self):
        rows, n_skipped = run_csv_pipeline_to_detections(
            self.detections_csv, self.nav_csv, image_width=2000, range_per_pixel_m=0.05,
            log_id="log-1", created_at="2026-01-01T00:00:00Z",
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(n_skipped, 1)

    def test_row_is_shaped_for_insert_detection(self):
        rows, _ = run_csv_pipeline_to_detections(
            self.detections_csv, self.nav_csv, image_width=2000, range_per_pixel_m=0.05,
            log_id="log-1", created_at="2026-01-01T00:00:00Z",
        )
        row = rows[0]
        for key in ("id", "log_id", "frame_index", "frame_record_id", "class_name", "yolo_conf", "bbox",
                    "confidence_score", "confidence_label", "lat", "lon", "geo_method", "depth_m",
                    "footprint_geojson", "created_at"):
            self.assertIn(key, row)
        self.assertEqual(row["log_id"], "log-1")
        self.assertEqual(row["geo_method"], "nav_fix")
        self.assertIsNone(row["footprint_geojson"])
        self.assertAlmostEqual(row["depth_m"], 60.0, places=2)
        self.assertEqual(len(row["bbox"]), 4)


class GeolocateCsvEndpointTests(unittest.TestCase):
    """API-level: POST /logs/geolocate_csv actually writes into the same
    `detections` table the rest of the API reads from."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_db_path = main.DB_PATH
        self._orig_output_dir = main.OUTPUT_DIR
        main.DB_PATH = Path(self._tmp.name) / "test_csv_geo.db"
        main.OUTPUT_DIR = Path(self._tmp.name) / "outputs"

        self.client = TestClient(main.app)
        self.client.__enter__()

        self.detections_csv_bytes = (
            b"ping_id,pixel_x,bbox_w,bbox_h,class,confidence\n"
            b"1000,1450,80,30,class_6,0.95\n"
        )
        self.nav_csv_bytes = (
            b"ping_id,lat,lon,heading,altitude_m,depth_m,timestamp\n"
            b"1000,19.0012,72.8015,45.0,8.0,52.0,2026-09-10T10:00:00Z\n"
        )

    def tearDown(self):
        self.client.__exit__(None, None, None)
        main.DB_PATH = self._orig_db_path
        main.OUTPUT_DIR = self._orig_output_dir
        self._tmp.cleanup()

    def test_upload_geolocates_and_persists_into_detections_table(self):
        r = self.client.post(
            "/logs/geolocate_csv",
            files={
                "detections_csv": ("detections.csv", self.detections_csv_bytes, "text/csv"),
                "nav_csv": ("nav.csv", self.nav_csv_bytes, "text/csv"),
            },
            params={"image_width": 2000, "range_per_pixel": 0.05},
        )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["status"], "done")
        log_id = body["log_id"]

        detections = self.client.get(f"/logs/{log_id}/detections").json()
        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0]["depth_m"], 60.0, places=2)

        map_fc = self.client.get(f"/logs/{log_id}/map").json()
        self.assertEqual(len(map_fc["features"]), 1)


if __name__ == "__main__":
    unittest.main()
