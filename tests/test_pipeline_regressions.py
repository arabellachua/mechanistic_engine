import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

import clients
import features
from modeling_v8 import build_stage2_features
from run_pipeline import classify_endpoint_type


class FakeResponse:
    def __init__(self, status_code, data=None, headers=None):
        self.status_code = status_code
        self._data = data or {}
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)

    def json(self):
        return self._data


class PipelineRegressionTests(unittest.TestCase):
    def test_hybrid_feature_set_is_distinct_and_contains_raw_features(self):
        df = pd.DataFrame({
            "CDS": [0.2],
            "ot_overall_score": [0.3],
            "raw_only_feature": [0.4],
        })

        feature_sets = features.build_feature_sets(
            df, ["ot_overall_score", "raw_only_feature"]
        )

        self.assertNotIn("raw_only_feature", feature_sets["composite"].columns)
        self.assertIn("raw_only_feature", feature_sets["hybrid"].columns)
        self.assertGreater(
            len(feature_sets["hybrid"].columns),
            len(feature_sets["composite"].columns),
        )

    def test_feature_computation_does_not_duplicate_normalized_columns(self):
        trials = pd.DataFrame({"drug_is_mapped": [1.0], "primary_target": ["EGFR"]})
        raw = {"drug_is_mapped": 0.0, "modality_score": 0.5}
        composite = {"CDS": 0.1}

        with patch.object(features, "compute_raw_features", return_value=raw), patch.object(
            features, "compute_composite_scores", return_value=composite
        ):
            result = features.compute_all_features(trials, {}, {})

        self.assertTrue(result.columns.is_unique)
        self.assertEqual(result.loc[0, "drug_is_mapped"], 1.0)

    def test_endpoint_classifier_handles_common_abbreviations(self):
        cases = {
            "Overall survival (OS) at 24 months": "survival",
            "Progression-free survival (PFS)": "pfs",
            "Objective response rate (ORR) by RECIST": "response",
            "Change in tumor biomarker expression": "biomarker",
            "Number of participants with adverse events": "other",
            "": None,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(classify_endpoint_type(text), expected)

    def test_stage2_columns_are_unique(self):
        df = pd.DataFrame({
            "drug_is_mapped": [1, 0],
            "endpoint_type": ["response", "survival"],
            "is_io_trial": [0, 1],
        })
        result = build_stage2_features(df, np.array([0.2, 0.8]), [])
        self.assertTrue(result.columns.is_unique)

    def test_cached_get_retries_rate_limit(self):
        responses = [
            FakeResponse(429, headers={"Retry-After": "0"}),
            FakeResponse(200, data={"ok": True}),
        ]
        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            clients, "_cache_key", return_value=Path(tmpdir) / "response.json"
        ), patch.object(clients.SESSION, "get", side_effect=responses) as get_mock, patch.object(
            clients.time, "sleep"
        ):
            result = clients.cached_get("https://example.test", retries=2)

        self.assertEqual(result, {"ok": True})
        self.assertEqual(get_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
