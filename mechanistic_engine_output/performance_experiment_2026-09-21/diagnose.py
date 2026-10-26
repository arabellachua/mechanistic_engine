"""Post-hoc diagnostics only; do not use to select another holdout winner."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
frame = pd.read_csv(root / 'experiment_snapshot.csv')
plan = json.loads((root / 'experiment_plan.json').read_text())
frame['split'] = np.where(frame.start_year <= 2015, 'train', 'test')
provenance = frame.groupby(['split', 'label_provenance']).label_strict.agg(n='size', positive='sum').reset_index()
provenance.to_csv(root / 'label_source_diagnostics.csv', index=False)
rows=[]
for view, columns in plan['feature_columns'].items():
    # Hash exact numeric feature vectors; signature is not a biological identity.
    data = pd.read_csv('mechanistic_engine_output/data/classification_matrix.csv') if view == 'legacy' else frame
    keys = pd.util.hash_pandas_object(data[columns].fillna(-1), index=False).to_numpy()
    work = frame[['nct_id','split','label_strict']].copy()
    work['signature'] = keys
    for split, part in work.groupby('split'):
        groups = part.groupby('signature').label_strict.agg(n='size', positives='sum', classes='nunique')
        conflict = groups[groups.classes > 1]
        rows.append({'view':view,'split':split,'trials':len(part),'distinct_vectors':len(groups),
                     'mixed_label_vectors':len(conflict),'trials_in_mixed_label_vectors':int(conflict.n.sum())})
pd.DataFrame(rows).to_csv(root / 'feature_resolution_diagnostics.csv', index=False)
print(provenance.to_string(index=False))
print(pd.DataFrame(rows).to_string(index=False))
# Regression evidence: preserved inputs and chronology of every validation fold.
import hashlib
for name, digest in plan['source_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
folds = pd.read_csv(root / 'validation_split_ids.csv')
years = frame.set_index('nct_id').start_year
holdout = set(frame.loc[frame.split == 'test','nct_id'])
for fold, part in folds.groupby('fold'):
    assert not (set(part.nct_id) & holdout)
    tr, va = part.loc[part.role=='train','nct_id'], part.loc[part.role=='validation','nct_id']
    assert years.loc[tr].max() < years.loc[va].min()
assert frame.nct_id.nunique() == 638
print('Input hashes, cohort size and fold isolation verified.')
