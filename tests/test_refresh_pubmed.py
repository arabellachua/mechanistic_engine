import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import refresh_pubmed


class RefreshPubMedTests(unittest.TestCase):
    def test_resume_skips_complete_pairs_and_retries_partial_pairs(self):
        trials = pd.DataFrame({"primary_target": ["A", "A", "B", "UNKNOWN"],
                               "disease": ["cancer", "cancer", "cancer", "cancer"]})
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.json"
            raw.write_text("[]")
            out = Path(tmp) / "output"
            argv = ["refresh_pubmed.py", "--raw-trials", str(raw), "--output-dir", str(out),
                    "--as-of", "2026-09-14"]
            with patch("sys.argv", argv), patch.object(refresh_pubmed, "filter_trials"), patch.object(
                refresh_pubmed, "normalize_trials", return_value=trials
            ), patch.object(refresh_pubmed, "query_pubmed_pair", side_effect=[
                {"pubmed_status": "complete"}, {"pubmed_status": "partial"},
                {"pubmed_status": "complete"}
            ]) as query:
                self.assertEqual(refresh_pubmed.main(), 1)
                self.assertEqual(refresh_pubmed.main(), 0)
                self.assertEqual(query.call_count, 3)
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["complete_pairs"], 2)
            self.assertEqual(summary["unknown_target_trials"], 1)
            self.assertEqual(len((out / "pair_attempts.jsonl").read_text().splitlines()), 3)
            raw.write_text("[{}]")
            with patch("sys.argv", argv), self.assertRaises(ValueError):
                refresh_pubmed.main()

    def test_refuses_unrelated_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.json"
            raw.write_text("[]")
            argv = ["refresh_pubmed.py", "--raw-trials", str(raw), "--output-dir", tmp]
            with patch("sys.argv", argv), self.assertRaises(ValueError):
                refresh_pubmed.main()


if __name__ == "__main__":
    unittest.main()
