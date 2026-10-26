import unittest

import numpy as np
import pandas as pd

from audit_research_snapshot import bootstrap_auc, overlap_rows


class ResearchSnapshotAuditTests(unittest.TestCase):
    def test_unknown_is_not_unseen_and_invalid_rows_are_counted(self):
        df = pd.DataFrame({
            "canonical_drug": [" A ", "a", "b", "unmapped", "c", "d"],
            "primary_target": ["T", "T", "U", "UNKNOWN", "V", "W"],
            "disease": ["cancer"] * 6,
            "start_year": [2015, 2016, 2017, 2018, np.nan, 2019],
            "label": [0, 1, 0, 1, 1, -1],
        })
        row = overlap_rows(df, "label")[0]
        self.assertEqual((row["test_seen"], row["test_unseen"], row["test_unknown"]), (1, 1, 1))
        self.assertEqual(row["excluded_invalid_label"], 1)
        self.assertEqual(row["excluded_missing_year_with_valid_label"], 1)

    def test_perfect_predictions_have_unit_auc_and_reproducible_intervals(self):
        first = bootstrap_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], repeats=100)
        self.assertEqual(first, bootstrap_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], repeats=100))
        self.assertEqual((first["auc"], first["ci_low"], first["ci_high"]), (1, 1, 1))

    def test_one_class_is_undefined_not_chance(self):
        self.assertIsNone(bootstrap_auc([1, 1], [0.3, 0.6], repeats=10)["auc"])

    def test_cluster_bootstrap_preserves_cluster_dependence(self):
        result = bootstrap_auc([0, 1, 0, 1], [0.1, 0.9, 0.1, 0.9],
                               groups=["a", "a", "b", "b"], repeats=100)
        self.assertEqual(result["valid_replicates"], 100)
        self.assertEqual(result["ci_low"], 1)

    def test_invalid_probabilities_fail(self):
        with self.assertRaises(ValueError):
            bootstrap_auc([0, 1], [0.5, np.nan])


if __name__ == "__main__":
    unittest.main()
