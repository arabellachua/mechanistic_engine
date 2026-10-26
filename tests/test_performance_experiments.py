import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone

from run_performance_experiments import candidates, refreshed_features, temporal_folds, paired_auc_interval


class PerformanceExperimentTests(unittest.TestCase):
    def test_pubmed_join_preserves_labels_and_missingness(self):
        data = pd.DataFrame({"nct_id": ["a", "b"], "primary_target": ["EGFR", "UNKNOWN"],
                             "disease": ["NSCLC", "NSCLC"], "label_strict": [1, 0],
                             "CDS": [.2, .2], "TDS": [.3, .3], "BFS": [.4, .4], "TWS": [.5, .5]})
        pm = pd.DataFrame({"primary_target": ["egfr"], "disease": ["nsclc"],
                           "pubmed_total_raw": [2000], "pubmed_clinical_trial_raw": [100],
                           "pubmed_recent_raw": [400], "pubmed_status": ["complete"]})
        updated, matched = refreshed_features(data, pm)
        self.assertEqual(updated.nct_id.tolist(), ["a", "b"])
        self.assertEqual(updated.label_strict.tolist(), [1, 0])
        self.assertEqual(matched.tolist(), [True, False])
        self.assertAlmostEqual(updated.clinical_publication_fraction.iloc[0], .05)
        self.assertTrue(np.isnan(updated.pubmed_pair_count.iloc[1]))
        self.assertEqual(updated.pubmed_data_missing.iloc[1], 1)
        with self.assertRaises(pd.errors.MergeError):
            refreshed_features(data, pd.concat([pm, pm]))

    def test_validation_is_chronological_and_disjoint(self):
        data = pd.DataFrame({"start_year": np.repeat(np.arange(2000, 2016), 10),
                             "label_strict": np.tile([0, 1], 80)})
        for train, val in temporal_folds(data):
            self.assertFalse(set(train) & set(val))
            self.assertLess(data.iloc[train].start_year.max(), data.iloc[val].start_year.min())
            self.assertLessEqual(data.iloc[val].start_year.max(), 2015)

    def test_scaler_is_fit_on_training_rows_only(self):
        _, estimator = candidates()["refreshed__scaled_lr_c01"]
        model = clone(estimator).fit(pd.DataFrame({"x": [0., 1., 2., 3.]}), [0, 0, 1, 1])
        before = model.named_steps["standardscaler"].mean_.copy()
        model.predict_proba(pd.DataFrame({"x": [10000.]}))
        np.testing.assert_array_equal(model.named_steps["standardscaler"].mean_, before)
        self.assertAlmostEqual(before[0], 1.5)

    def test_paired_bootstrap_identical_models_have_zero_difference(self):
        result = paired_auc_interval([0, 0, 1, 1], [.1, .2, .8, .9], [.1, .2, .8, .9], repeats=100)
        self.assertEqual(result["auc_difference"], {"low": 0., "high": 0.})


if __name__ == "__main__":
    unittest.main()
