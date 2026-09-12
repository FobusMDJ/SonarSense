import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.geolocation.nav import load_nav_sidecar


class NavigationContractTests(unittest.TestCase):
    def test_iso_timestamp_and_one_based_single_frame_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.csv"
            path.write_text(
                "frame_index,lat,lon,heading_deg,altitude_m,depth_m,timestamp\n"
                "1,40.700008,-73.985,360,2.99,24.2,2026-06-14T09:00:00Z\n",
                encoding="utf-8",
            )
            fixes = load_nav_sidecar(path)
            self.assertEqual(len(fixes), 1)
            self.assertEqual(fixes[0].frame_index, 1)
            self.assertEqual(fixes[0].heading_deg, 0)
            self.assertEqual(
                fixes[0].timestamp,
                datetime(2026, 6, 14, 9, 0, tzinfo=timezone.utc).timestamp(),
            )

    def test_invalid_coordinate_reports_the_csv_row(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.csv"
            path.write_text(
                "frame_index,lat,lon,heading_deg\n0,120,-73.985,0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "CSV row 2"):
                load_nav_sidecar(path)


if __name__ == "__main__":
    unittest.main()
