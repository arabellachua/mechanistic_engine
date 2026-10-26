import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

import features
import pubmed_client as pm


class Response:
    def __init__(self, data, status=200, headers=None):
        self.data, self.status_code, self.headers = data, status, headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self):
        return self.data


def count(value):
    return {"esearchresult": {"count": str(value)}}


class PubMedTests(unittest.TestCase):
    def test_error_response_is_not_successful_zero(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(pm, "_throttle"), patch.object(
            pm.time, "sleep"
        ), patch.object(pm.SESSION, "get", return_value=Response({"error": "API rate limit exceeded"})):
            result = pm.query_pubmed_pair("EGFR", "NSCLC", cache_dir=tmp)
            self.assertEqual(result["pubmed_status"], "failed")
            self.assertIsNone(result["pubmed_pair_count"])
            self.assertFalse(result["has_real_pubmed_data"])
            self.assertEqual(list(Path(tmp).glob("*.json")), [])

    def test_real_zero_and_cache_reuse(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(pm, "_throttle"), patch.object(
            pm.SESSION, "get", return_value=Response(count(0))
        ) as get:
            first = pm.query_pubmed_pair("EGFR", "NSCLC", as_of="2026-09-14", cache_dir=tmp)
            second = pm.query_pubmed_pair("EGFR", "NSCLC", as_of="2026-09-14", cache_dir=tmp)
            self.assertEqual(first, second)
            self.assertTrue(first["has_real_pubmed_data"])
            self.assertEqual(first["pubmed_pair_count"], 0)
            self.assertEqual(get.call_count, 3)

    def test_partial_failure_remains_missing(self):
        with patch.object(pm, "_pubmed_count", side_effect=[100, None, 20]):
            result = pm.query_pubmed_pair("EGFR", "NSCLC")
        self.assertEqual(result["pubmed_status"], "partial")
        self.assertIsNone(result["clinical_trial_pub_count"])
        self.assertEqual(result["pubmed_data_missing"], 1)

    def test_dates_are_explicit_and_cache_keys_include_dates(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(pm, "_throttle"), patch.object(
            pm.SESSION, "get", return_value=Response(count(1))
        ) as get:
            pm.query_pubmed_pair("EGFR", "NSCLC", as_of="2015-12-31", cache_dir=tmp)
            params = [call.kwargs["params"] for call in get.call_args_list]
            self.assertEqual(params[2]["mindate"], "2012/01/01")
            self.assertTrue(all(p["maxdate"] == "2015/12/31" for p in params))
            pm.query_pubmed_pair("EGFR", "NSCLC", as_of="2026-09-14", cache_dir=tmp)
            self.assertEqual(get.call_count, 6)

    def test_rate_limit_retry_respects_server_delay(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(pm, "_throttle") as throttle, patch.object(
            pm.time, "sleep"
        ) as sleep, patch.object(pm.SESSION, "get", side_effect=[
            Response({}, 429, {"Retry-After": "45"}), Response(count(7))
        ]):
            self.assertEqual(pm._pubmed_count({"term": "test"}, tmp), 7)
            sleep.assert_called_once_with(45.0)
            self.assertEqual(throttle.call_count, 2)

    def test_bad_cache_is_refetched(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(pm, "_throttle"), patch.object(
            pm.SESSION, "get", return_value=Response(count(7))
        ) as get:
            pm._pubmed_count({"term": "test"}, tmp)
            next(Path(tmp).glob("*.json")).write_text(json.dumps({"error": "bad cache"}))
            self.assertEqual(pm._pubmed_count({"term": "test"}, tmp), 7)
            self.assertEqual(get.call_count, 2)

    def test_subset_count_cannot_exceed_total(self):
        with patch.object(pm, "_pubmed_count", side_effect=[1, 5, 0]):
            result = pm.query_pubmed_pair("EGFR", "NSCLC")
        self.assertEqual(result["pubmed_status"], "inconsistent")
        self.assertFalse(result["has_real_pubmed_data"])

    def test_failed_feature_values_are_missing_not_zero(self):
        trial = pd.Series({"primary_target": "EGFR", "disease": "NSCLC"})
        result = features.compute_raw_features(trial, {}, {})
        self.assertTrue(np.isnan(result["pubmed_pair_count"]))
        self.assertTrue(np.isnan(result["human_study_fraction"]))
        self.assertEqual(result["pubmed_data_missing"], 1)
        matrices = features.build_feature_sets(pd.DataFrame([result]), list(result))
        self.assertEqual(matrices["composite"].pubmed_pair_count.iloc[0], -1)
        self.assertIn("pubmed_data_missing", matrices["composite"])


if __name__ == "__main__":
    unittest.main()
