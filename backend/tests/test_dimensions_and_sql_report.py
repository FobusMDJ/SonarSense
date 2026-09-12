import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from src.backend import db_backend as db
from src.backend import main
from src.geolocation.postgis_export import build_postgis_sql


class PostgisSqlExportTests(unittest.TestCase):
    """build_postgis_sql() is a pure function -- no FastAPI/DB needed."""

    def _detection(self, **overrides):
        base = {
            "id": "d1", "log_id": "log-1", "frame_index": 0, "frame_record_id": "f1",
            "frame_image_path": None, "class_name": "pipe", "yolo_conf": 0.9,
            "bbox_x1": 10.0, "bbox_y1": 20.0, "bbox_x2": 30.0, "bbox_y2": 40.0,
            "confidence_score": 91.0, "confidence_label": "high", "confidence_breakdown": {"yolo_points": 63.0},
            "vae_box_error": None, "vae_whole_image_error": None, "vae_whole_image_percentile": None,
            "lat": 19.0012, "lon": 72.8015, "geo_method": "nav_fix", "depth_m": 60.0,
            "length_m": 1.5, "width_m": 4.0,
            "footprint_geojson": [[72.80, 19.00], [72.81, 19.00], [72.81, 19.01], [72.80, 19.01], [72.80, 19.00]],
            "vae_panel_dir": None, "created_at": "2026-01-01T00:00:00Z",
        }
        base.update(overrides)
        return base

    def test_output_has_create_table_and_one_insert_per_detection(self):
        sql = build_postgis_sql([self._detection(), self._detection(id="d2")])
        self.assertIn("CREATE TABLE IF NOT EXISTS detections", sql)
        self.assertEqual(sql.count("INSERT INTO detections"), 2)

    def test_string_values_are_quoted_and_escaped(self):
        sql = build_postgis_sql([self._detection(class_name="o'brien's net")])
        self.assertIn("'o''brien''s net'", sql)  # single quote doubled, standard SQL escaping

    def test_placeholder_detection_gets_null_geom_not_a_fake_point(self):
        sql = build_postgis_sql([self._detection(lat=0.0, lon=0.0, geo_method="placeholder")])
        # the INSERT's geom column (second-to-last value) must be NULL, not ST_MakePoint(0, 0)
        insert_line = [line for line in sql.splitlines() if line.startswith("INSERT")][0]
        self.assertNotIn("ST_MakePoint(0.0, 0.0)", insert_line)

    def test_footprint_produces_a_geom_footprint_polygon_expression(self):
        sql = build_postgis_sql([self._detection()])
        self.assertIn("ST_GeomFromGeoJSON", sql)
        self.assertIn("geometry(Polygon", sql)

    def test_missing_footprint_gets_null_geom_footprint(self):
        sql = build_postgis_sql([self._detection(footprint_geojson=None)])
        insert_line = [line for line in sql.splitlines() if line.startswith("INSERT")][0]
        # geom (real, since lat/lon are set) is followed by geom_footprint, which must be
        # NULL since there's no footprint -- i.e. the value list ends "..., <geom-expr>, NULL)".
        self.assertTrue(insert_line.rstrip(";").endswith(", NULL)"))
        self.assertIn("ST_MakePoint", insert_line)  # sanity: geom itself IS populated here

    def test_length_width_columns_carry_the_geolocation_engines_own_numbers(self):
        # The whole point of adding these columns: a CSV-engine-sourced detection's real
        # length_m/width_m (computed by csv_engine.geolocate_csv_detection, not a bbox
        # guess) must land in the SQL dump as plain numeric literals, not JSON/NULL.
        sql = build_postgis_sql([self._detection(length_m=1.5, width_m=4.0)])
        self.assertIn("CREATE TABLE IF NOT EXISTS detections", sql)
        self.assertIn("length_m DOUBLE PRECISION", sql)
        self.assertIn("width_m DOUBLE PRECISION", sql)
        insert_line = [line for line in sql.splitlines() if line.startswith("INSERT")][0]
        self.assertIn("1.5", insert_line)
        self.assertIn("4.0", insert_line)

    def test_length_width_null_when_not_computed(self):
        # An image/YOLO-pipeline detection has no length_m/width_m of its own at insert
        # time (see class_taxonomy.resolved_dimensions_m's fallback) -- must dump as SQL
        # NULL, never a fabricated 0.0.
        sql = build_postgis_sql([self._detection(length_m=None, width_m=None)])
        insert_line = [line for line in sql.splitlines() if line.startswith("INSERT")][0]
        values = insert_line.split("VALUES (", 1)[1]
        self.assertIn("NULL", values)


class DetectionsEndpointDimensionsTests(unittest.TestCase):
    """GET /logs/{id}/detections now includes length_m/width_m/height_m --
    the gap that left the frontend with no way to show debris geometry at
    all (dashboard/data.ts never had these fields to read)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self._orig_db_path = main.DB_PATH
        self._orig_output_dir = main.OUTPUT_DIR
        main.DB_PATH = Path(self._tmp.name) / "test_dims.db"
        main.OUTPUT_DIR = Path(self._tmp.name) / "outputs"
        self.client = TestClient(main.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        main.DB_PATH = self._orig_db_path
        main.OUTPUT_DIR = self._orig_output_dir
        self._tmp.cleanup()

    def _seed(self, log_id: str, pixels_to_meters):
        with db.get_connection(main.DB_PATH) as conn:
            db.create_log(conn, log_id, "fixture.png", "image", "2026-01-01T00:00:00Z",
                           pixels_to_meters=pixels_to_meters)
            db.insert_detection(conn, {
                "id": "d1", "log_id": log_id, "frame_index": 0, "frame_record_id": "f1",
                "class_name": "pipe", "yolo_conf": 0.9, "bbox": [100, 100, 180, 130],
                "confidence_score": 91.0, "confidence_label": "high",
                "lat": 19.0, "lon": 72.8, "geo_method": "nav_fix", "created_at": "2026-01-01T00:00:00Z",
            })

    def test_length_width_computed_from_bbox_when_pixels_to_meters_known(self):
        self._seed("log-1", pixels_to_meters=0.15)
        detections = self.client.get("/logs/log-1/detections").json()
        det = detections[0]
        self.assertAlmostEqual(det["length_m"], (180 - 100) * 0.15, places=3)
        self.assertAlmostEqual(det["width_m"], (130 - 100) * 0.15, places=3)
        self.assertIsNotNone(det["height_m"])
        self.assertFalse(det["dimensions_estimated"])

    def test_length_width_null_not_zero_when_pixels_to_meters_unknown(self):
        self._seed("log-2", pixels_to_meters=None)
        detections = self.client.get("/logs/log-2/detections").json()
        det = detections[0]
        self.assertIsNone(det["length_m"])
        self.assertIsNone(det["width_m"])
        self.assertTrue(det["dimensions_estimated"])

    def test_stored_engine_dimensions_win_over_bbox_recompute(self):
        # A detection that already has its own length_m/width_m (set at insert time by
        # the CSV geolocation engine's real slant-range/ground-range math) must come back
        # from GET /logs/{id}/detections EXACTLY as stored -- not silently overwritten by
        # the generic bbox * pixels_to_meters guess, which uses a different axis
        # convention (see class_taxonomy.resolved_dimensions_m). A large pixels_to_meters
        # is deliberately set on this log so a bbox-based recompute would produce very
        # different numbers if it were (wrongly) still in charge.
        log_id = "log-3"
        with db.get_connection(main.DB_PATH) as conn:
            db.create_log(conn, log_id, "fixture.csv", "csv_detections", "2026-01-01T00:00:00Z",
                           pixels_to_meters=0.05)
            db.insert_detection(conn, {
                "id": "d1", "log_id": log_id, "frame_index": 0, "frame_record_id": "P0001",
                "class_name": "pipe", "yolo_conf": 0.9, "bbox": [1410.0, 0.0, 1490.0, 30.0],
                "confidence_score": 91.0, "confidence_label": "high",
                "lat": 19.0011, "lon": 72.8016, "geo_method": "nav_fix",
                "length_m": 1.5, "width_m": 4.0,  # the engine's own numbers
                "created_at": "2026-01-01T00:00:00Z",
            })
        detections = self.client.get(f"/logs/{log_id}/detections").json()
        det = detections[0]
        self.assertEqual(det["length_m"], 1.5)
        self.assertEqual(det["width_m"], 4.0)
        self.assertFalse(det["dimensions_estimated"])


if __name__ == "__main__":
    unittest.main()
