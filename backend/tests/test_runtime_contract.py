import sqlite3
import tempfile
import unittest
import zipfile
from pathlib import Path

from src.backend import db
from src.backend.archive_ingestion import extract_survey_zip
from src.backend.model_release import DETECTOR_CLASSES, DETECTOR_METRICS, validate_present_checkpoints


class RuntimeContractTests(unittest.TestCase):
    def test_release_metadata_is_the_four_class_checkpoint(self):
        self.assertEqual(DETECTOR_CLASSES, ["aircraft", "human", "ship", "pipe"])
        self.assertEqual(DETECTOR_METRICS["map50"], 0.6926)
        self.assertEqual(DETECTOR_METRICS["map50_95"], 0.53874)

    def test_sqlite_persists_frame_and_runtime_measurements(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            path = Path(directory) / "runtime.db"
            db.init_db(path)
            with db.get_connection(path) as conn:
                db.create_log(conn, "log-1", "fixture.png", "image", "now",
                              yolo_confidence_threshold=0.25)
                db.insert_frame_analysis(conn, {
                    "log_id": "log-1", "frame_index": 0, "frame_record_id": "frame-1",
                    "whole_image_error": 0.12, "percentile": 0.5, "vae_panel_dir": "/tmp/panels",
                })
                db.update_log_performance(conn, "log-1", 80.0, 2)
            with sqlite3.connect(path) as conn:
                row = conn.execute("SELECT yolo_confidence_threshold, detector_inference_ms, detector_frames FROM logs").fetchone()
                self.assertEqual(row, (0.25, 80.0, 2))
            with db.get_connection(path) as conn:
                self.assertEqual(len(db.list_frame_analyses(conn, "log-1")), 1)

    def test_present_checkpoint_with_wrong_digest_is_rejected(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            bad = Path(directory) / "best.pt"
            bad.write_bytes(b"not the released checkpoint")
            with self.assertRaisesRegex(RuntimeError, "integrity verification failed"):
                validate_present_checkpoints({"detector": bad})

    def test_nested_survey_zip_is_flattened_and_metadata_is_validated(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            root = Path(directory)
            archive_path = root / "survey.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("survey/frames/frame_000.png", b"fixture")
                archive.writestr(
                    "survey/metadata.csv",
                    "frame_index,lat,lon,heading_deg\n0,15.29932,73.96301,142\n",
                )
            frames, metadata, count = extract_survey_zip(archive_path, root / "out")
            self.assertEqual(count, 1)
            self.assertTrue((frames / "frame_000.png").is_file())
            self.assertTrue(metadata.is_file())

    def test_survey_zip_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            root = Path(directory)
            archive_path = root / "unsafe.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../frame.png", b"fixture")
                archive.writestr("metadata.csv", "frame_index,lat,lon,heading_deg\n0,1,2,3\n")
            with self.assertRaisesRegex(ValueError, "Unsafe path"):
                extract_survey_zip(archive_path, root / "out")

    def test_nav_sidecar_is_selected_from_a_dataset_archive_with_multiple_csvs(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            root = Path(directory)
            archive_path = root / "dataset.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("dataset/Frames/frame_0001.png", b"fixture")
                archive.writestr("dataset/MetaData/dataset_metadata.csv", "label,source\npipe,test\n")
                archive.writestr(
                    "dataset/MetaData/nav_sidecar.csv",
                    "frame_index,lat,lon,heading_deg\n1,-5.0,75.0,0\n",
                )
            _, metadata, count = extract_survey_zip(archive_path, root / "out")
            self.assertEqual(count, 1)
            self.assertIn("frame_index,lat,lon,heading_deg", metadata.read_text())


if __name__ == "__main__":
    unittest.main()
