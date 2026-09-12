import json
import tempfile
import unittest
from pathlib import Path

from src.geolocation.georeference import geolocate_detection
from src.geolocation.geojson_export import build_geojson
from src.geolocation.nav import NavFix

# main.py pulls in the full backend (torch, cv2, ...) just to import it --
# not needed for the math-only tests below, and not always installed (e.g.
# a lightweight dev machine that hasn't set up the ML deps yet). Guarded so
# a missing torch/cv2 only skips the one API-level test class further down,
# instead of failing collection for this entire file.
try:
    from fastapi.testclient import TestClient
    from src.backend import db_backend as db
    from src.backend import main
    _MAIN_IMPORTABLE = True
    _MAIN_IMPORT_ERROR = None
except ImportError as exc:
    _MAIN_IMPORTABLE = False
    _MAIN_IMPORT_ERROR = exc


class FootprintGeometryTests(unittest.TestCase):
    def setUp(self):
        self.nav = NavFix(frame_index=0, lat=40.70000, lon=-73.98500, heading_deg=0.0,
                           altitude_m=3.0, depth_m=24.0)
        self.xyxy = [200, 100, 260, 180]

    def test_footprint_is_a_closed_4_corner_ring(self):
        geo = geolocate_detection(xyxy=self.xyxy, image_width_px=512, image_height_px=512,
                                   nav_fix=self.nav, pixels_to_meters=0.15)
        self.assertEqual(len(geo.footprint), 5)
        self.assertEqual(geo.footprint[0], geo.footprint[-1])
        lats = [p[0] for p in geo.footprint[:-1]]
        lons = [p[1] for p in geo.footprint[:-1]]
        self.assertGreater(max(lats) - min(lats), 0)
        self.assertGreater(max(lons) - min(lons), 0)

    def test_no_nav_fix_means_no_footprint(self):
        geo = geolocate_detection(xyxy=self.xyxy, image_width_px=512, image_height_px=512,
                                   nav_fix=None, pixels_to_meters=0.15)
        self.assertEqual(geo.method, "placeholder")
        self.assertIsNone(geo.footprint)

    def test_omitting_image_height_px_keeps_old_point_only_behavior(self):
        with_footprint = geolocate_detection(xyxy=self.xyxy, image_width_px=512, image_height_px=512,
                                              nav_fix=self.nav, pixels_to_meters=0.15)
        without_footprint = geolocate_detection(xyxy=self.xyxy, image_width_px=512,
                                                 nav_fix=self.nav, pixels_to_meters=0.15)
        self.assertIsNone(without_footprint.footprint)
        self.assertEqual(without_footprint.lat, with_footprint.lat)
        self.assertEqual(without_footprint.lon, with_footprint.lon)

    def _detection(self):
        geo = geolocate_detection(xyxy=self.xyxy, image_width_px=512, image_height_px=512,
                                   nav_fix=self.nav, pixels_to_meters=0.15)
        return {
            "id": "d1", "log_id": "l1", "class_name": "pipe", "confidence_score": 91.2,
            "confidence_label": "High", "yolo_conf": 0.83, "frame_record_id": "f1", "frame_index": 0,
            "bbox_x1": 200, "bbox_y1": 100, "bbox_x2": 260, "bbox_y2": 180,
            "vae_whole_image_percentile": 0.2, "geo_method": "nav_fix", "created_at": "2026-01-01T00:00:00Z",
            "lat": geo.lat, "lon": geo.lon,
            "footprint_geojson": json.dumps([[lon, lat] for lat, lon in geo.footprint]),
        }

    def test_geojson_auto_mode_emits_geometry_collection(self):
        fc = build_geojson([self._detection()], geometry_mode="auto")
        geom = fc["features"][0]["geometry"]
        self.assertEqual(geom["type"], "GeometryCollection")
        self.assertEqual({g["type"] for g in geom["geometries"]}, {"Point", "Polygon"})
        self.assertTrue(fc["features"][0]["properties"]["has_footprint"])

    def test_geojson_point_only_mode_ignores_footprint(self):
        fc = build_geojson([self._detection()], geometry_mode="point_only")
        self.assertEqual(fc["features"][0]["geometry"]["type"], "Point")

    def test_geojson_footprint_only_mode(self):
        fc = build_geojson([self._detection()], geometry_mode="footprint_only")
        self.assertEqual(fc["features"][0]["geometry"]["type"], "Polygon")

    def test_detection_without_footprint_still_works_in_every_mode(self):
        det = self._detection()
        det["footprint_geojson"] = None
        for mode in ("auto", "point_only", "footprint_only"):
            fc = build_geojson([det], geometry_mode=mode)
            self.assertEqual(fc["features"][0]["geometry"]["type"], "Point")
            self.assertFalse(fc["features"][0]["properties"]["has_footprint"])


@unittest.skipUnless(_MAIN_IMPORTABLE, f"src.backend.main not importable ({_MAIN_IMPORT_ERROR})")
class MapEndpointGeometryModeTests(unittest.TestCase):
    """GET /logs/{log_id}/map's `geometry_mode` query param (see main.py's
    get_map_geojson) -- exercises the live API, not just build_geojson()
    directly, so a regression in the FastAPI wiring itself (missing param,
    swallowed ValueError, etc.) would actually be caught here."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self._orig_db_path = main.DB_PATH
        main.DB_PATH = Path(self._tmp.name) / "test_map.db"

        self.client = TestClient(main.app)
        self.client.__enter__()  # runs main.py's @app.on_event("startup") against the temp db

        nav = NavFix(frame_index=0, lat=40.70000, lon=-73.98500, heading_deg=0.0,
                     altitude_m=3.0, depth_m=24.0)
        geo = geolocate_detection(xyxy=[200, 100, 260, 180], image_width_px=512, image_height_px=512,
                                   nav_fix=nav, pixels_to_meters=0.15)
        with db.get_connection(main.DB_PATH) as conn:
            db.create_log(conn, "log-1", "fixture.png", "image", "2026-01-01T00:00:00Z")
            db.insert_detection(conn, {
                "id": "d1", "log_id": "log-1", "frame_index": 0, "frame_record_id": "f1",
                "class_name": "pipe", "yolo_conf": 0.9, "bbox": [200, 100, 260, 180],
                "confidence_score": 91.0, "confidence_label": "High",
                "lat": geo.lat, "lon": geo.lon, "geo_method": "nav_fix",
                "footprint_geojson": json.dumps([[lon, lat] for lat, lon in geo.footprint]),
                "created_at": "2026-01-01T00:00:00Z",
            })

    def tearDown(self):
        self.client.__exit__(None, None, None)
        main.DB_PATH = self._orig_db_path
        self._tmp.cleanup()

    def test_default_geometry_mode_is_auto(self):
        r = self.client.get("/logs/log-1/map")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["features"][0]["geometry"]["type"], "GeometryCollection")

    def test_point_only_mode(self):
        r = self.client.get("/logs/log-1/map", params={"geometry_mode": "point_only"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["features"][0]["geometry"]["type"], "Point")

    def test_footprint_only_mode(self):
        r = self.client.get("/logs/log-1/map", params={"geometry_mode": "footprint_only"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["features"][0]["geometry"]["type"], "Polygon")

    def test_invalid_geometry_mode_is_a_client_error_not_a_500(self):
        r = self.client.get("/logs/log-1/map", params={"geometry_mode": "bogus"})
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
