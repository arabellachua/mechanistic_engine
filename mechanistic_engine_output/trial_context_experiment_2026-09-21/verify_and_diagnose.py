"""Verify experiment lineage and inspect one explicitly post-hoc candidate."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from run_trial_context_experiments import estimator_specs, paired_fold_interval
from run_context_experiments import make_folds

output = Path(__file__).resolve().parent
plan = json.loads((output / 'experiment_plan.json').read_text())
for path, expected in plan['source_sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
frame = pd.read_csv(output / 'training_snapshot.csv').fillna({'structured_text': '', 'eligibility_text': '', 'protocol_text': ''})
original = pd.read_csv('mechanistic_engine_output/data/classification_matrix.csv')
expected = original.loc[original.start_year.le(2015) & original.label_strict.isin([0, 1])]
assert frame.nct_id.tolist() == expected.nct_id.tolist()
assert frame.label_strict.tolist() == expected.label_strict.tolist()
assert len(frame) == 472
splits = pd.read_csv(output / 'split_ids.csv')
assert not set(splits.nct_id) & set(original.loc[original.start_year.gt(2015), 'nct_id'])
for (mode, fold), block in splits.groupby(['mode', 'outer_fold']):
    outer = block.loc[block.stage.eq('outer')]
    outer_train = set(outer.loc[outer.role.eq('train'), 'nct_id'])
    outer_val = set(outer.loc[outer.role.eq('validation'), 'nct_id'])
    assert not outer_train & outer_val
    inner = block.loc[block.stage.eq('inner')]
    assert set(inner.nct_id) <= outer_train
    for _, part in inner.groupby('fold'):
        assert not set(part.loc[part.role.eq('train'), 'nct_id']) & set(part.loc[part.role.eq('validation'), 'nct_id'])

prediction = pd.read_csv(output / 'validation_predictions.csv')
# Chosen after viewing the outer results: these intervals are NOT confirmatory.
candidate = 'protocol_core_c1'
intervals = {'candidate': candidate, 'post_hoc': True, 'multiple_comparison_corrected': False,
             'interpretation': 'Descriptive only: current fixed predictions, previously used development folds.'}
for mode in prediction['mode'].unique():
    part = prediction.loc[prediction['mode'].eq(mode) & prediction.candidate.eq(candidate)].reset_index(drop=True)
    intervals[mode] = paired_fold_interval(part)
(output / 'posthoc_candidate_intervals.json').write_text(json.dumps(intervals, indent=2) + '\n')

terms = []
for mode, partitions in make_folds(frame).items():
    for fold, (train, _) in enumerate(partitions):
        model = clone(estimator_specs()[candidate]).fit(frame.iloc[train], frame.label_strict.iloc[train])
        names = model[0].named_transformers_['text'].get_feature_names_out()
        coefficients = model[-1].coef_[0][:len(names)]
        for sign, selected in [('negative', np.argsort(coefficients)[:15]),
                               ('positive', np.argsort(coefficients)[-15:][::-1])]:
            terms.extend({'mode': mode, 'fold': fold, 'sign': sign, 'term': names[i],
                          'coefficient': float(coefficients[i])} for i in selected)
pd.DataFrame(terms).to_csv(output / 'text_coefficient_diagnostics.csv', index=False)
verification = {'source_hashes_match': True, 'labels_and_trial_order_unchanged': True,
                'nested_split_isolation': True, 'post2015_rows_scored': 0,
                'n_training': len(frame), 'n_protocol_missing': int(frame.protocol_missing.sum())}
(output / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
print(json.dumps(intervals, indent=2))
print(json.dumps(verification, indent=2))
