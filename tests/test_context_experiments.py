import unittest

import numpy as np
import pandas as pd

from run_context_experiments import CORE_COLUMNS, context_model, prepare_training_frame, make_folds


class ContextExperimentTests(unittest.TestCase):
    def fixture(self):
        frame = pd.DataFrame({"nct_id": [f"t{i}" for i in range(160)],
                              "start_year": np.repeat(np.arange(2000, 2016), 10),
                              "label_strict": np.tile([0, 1], 80),
                              "canonical_drug": [f"d{i % 8}" for i in range(160)],
                              "primary_target": [f"g{i % 8}" for i in range(160)],
                              "disease": "cancer", "modality": "small molecule"})
        for col in CORE_COLUMNS:
            frame[col] = np.arange(len(frame)) / len(frame)
        return frame

    def test_later_labels_do_not_change_training_input(self):
        frame = self.fixture()
        later = frame.iloc[[0]].copy()
        later["nct_id"], later["start_year"] = "later", 2020
        before = prepare_training_frame(pd.concat([frame, later]))
        later["label_strict"] = -999
        pd.testing.assert_frame_equal(before, prepare_training_frame(pd.concat([frame, later])))

    def test_grouped_validation_has_no_pair_overlap(self):
        frame = prepare_training_frame(self.fixture())
        # Ensure every synthetic group has both classes.
        frame["label_strict"] = np.tile(np.repeat([0, 1], 8), 10)
        for train, val in make_folds(frame)["unseen_target_disease_pair"]:
            self.assertFalse(set(frame.iloc[train].target_disease_pair) & set(frame.iloc[val].target_disease_pair))

    def test_unseen_category_does_not_refit_encoder(self):
        frame = prepare_training_frame(self.fixture())
        model = context_model(["primary_target", "disease", "modality"]).fit(frame, frame.label_strict)
        encoder = model[0].named_transformers_["context"]
        categories = [values.copy() for values in encoder.categories_]
        unseen = frame.iloc[[0]].copy()
        unseen["primary_target"] = "new_target"
        self.assertTrue(np.isfinite(model.predict_proba(unseen)).all())
        for old, new in zip(categories, encoder.categories_):
            np.testing.assert_array_equal(old, new)


if __name__ == "__main__":
    unittest.main()
