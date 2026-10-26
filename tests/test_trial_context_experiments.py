import copy
import unittest

import numpy as np
import pandas as pd

from run_trial_context_experiments import (
    attach_protocol, choose_candidate, estimator_specs, inner_folds, paired_fold_interval, protocol_features,
)
from tests import test_context_experiments as context_tests
from run_context_experiments import make_folds, prepare_training_frame


class TrialContextExperimentTests(unittest.TestCase):
    def test_feature_allowlist_excludes_label_sources(self):
        study = {'protocolSection': {
            'conditionsModule': {'conditions': ['lung cancer']},
            'armsInterventionsModule': {'interventions': [
                {'type': 'DRUG', 'name': 'drug A', 'description': 'shouldnotappear'},
                {'type': 'PROCEDURE', 'name': 'shouldnotappear'}]},
            'eligibilityModule': {'eligibilityCriteria': 'EGFR mutation\\nrequired'},
        }}
        expected = protocol_features(study)
        changed = copy.deepcopy(study)
        for module in ['descriptionModule', 'outcomesModule', 'statusModule', 'identificationModule']:
            changed['protocolSection'][module] = {'arbitrary': 'met primary endpoint approved'}
        changed['resultsSection'] = {'outcomes': 'positive'}
        self.assertEqual(expected, protocol_features(changed))
        self.assertNotIn('shouldnotappear', expected['protocol_text'])
        self.assertIn('drug A', expected['protocol_text'])
        self.assertIn('EGFR mutation required', expected['protocol_text'])

    def test_missing_protocol_preserves_cohort_and_labels(self):
        source = pd.DataFrame({'nct_id': ['a', 'b'], 'label_strict': [1, 0]})
        studies = [{'protocolSection': {'identificationModule': {'nctId': 'a'},
                                       'conditionsModule': {'conditions': ['cancer']}}}]
        actual = attach_protocol(source, studies)
        pd.testing.assert_frame_equal(actual[source.columns], source)
        self.assertEqual(actual.protocol_missing.tolist(), [0, 1])
        self.assertEqual(actual.protocol_text.iloc[1], 'missingprotocol')
        with self.assertRaises(ValueError):
            attach_protocol(source, studies * 2)

    def test_nested_splits_are_disjoint_and_respect_outer_boundary(self):
        frame = prepare_training_frame(context_tests.ContextExperimentTests().fixture())
        frame['label_strict'] = np.tile(np.repeat([0, 1], 8), 10)
        for mode, partitions in make_folds(frame).items():
            for tr, va in partitions:
                training = frame.iloc[tr].reset_index(drop=True)
                for a, b in inner_folds(training, mode):
                    self.assertFalse(set(training.iloc[a].nct_id) & set(training.iloc[b].nct_id))
                    self.assertFalse(set(training.iloc[b].nct_id) & set(frame.iloc[va].nct_id))
                    if mode == 'temporal':
                        self.assertLess(training.iloc[a].start_year.max(), training.iloc[b].start_year.min())
                    else:
                        self.assertFalse(set(training.iloc[a].target_disease_pair) & set(training.iloc[b].target_disease_pair))

    def test_vocabulary_is_learned_from_training_only(self):
        frame = context_tests.ContextExperimentTests().fixture().iloc[:20].copy()
        frame['protocol_text'] = ['lung cancer egfr' if i % 2 else 'breast cancer her2' for i in range(20)]
        frame['protocol_missing'] = 0
        model = estimator_specs()['protocol_core_c1'].fit(frame, frame.label_strict)
        vectorizer = model[0].named_transformers_['text']
        vocabulary = vectorizer.vocabulary_.copy()
        later = frame.iloc[:1].copy()
        later['protocol_text'] = 'unseenvalidationword'
        self.assertTrue(np.isfinite(model.predict_proba(later)).all())
        self.assertEqual(vocabulary, vectorizer.vocabulary_)
        self.assertNotIn('unseenvalidationword', vectorizer.vocabulary_)

    def test_selection_uses_auc_then_brier_then_name(self):
        scores = pd.DataFrame({'candidate': ['a', 'b', 'c'], 'auc': [.7, .8, .8], 'brier': [.1, .2, .15]})
        self.assertEqual(choose_candidate(scores), 'c')

    def test_identical_predictions_have_zero_interval(self):
        pred = pd.DataFrame({'pair': ['a', 'a', 'b', 'b'] * 2, 'fold': [0] * 4 + [1] * 4,
                             'label': [0, 1, 0, 1] * 2, 'probability': [.1, .8, .2, .7] * 2,
                             'reference_probability': [.1, .8, .2, .7] * 2})
        interval = paired_fold_interval(pred, repeats=100)
        self.assertEqual(interval['low'], 0.)
        self.assertEqual(interval['high'], 0.)


if __name__ == '__main__':
    unittest.main()
