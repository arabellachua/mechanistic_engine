"""Nested development validation of trial context and nonlinear biology models.

Only trials starting by 2015 are used. Existing labels and source files stay fixed.
Current protocol text is not a reconstructed historical registration snapshot.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC

from run_context_experiments import context_model, make_folds, prepare_training_frame
from run_performance_experiments import CORE_COLUMNS, SEED, metrics


SOURCE = Path('mechanistic_engine_output/data/classification_matrix.csv')
RAW_SOURCE = Path('mechanistic_engine_output/data/raw_trials.json')
REFERENCE = 'numeric_reference'


def protocol_features(study):
    """Explicit input allowlist: never use title, summary, outcomes or status."""
    protocol = study.get('protocolSection', {})
    conditions = protocol.get('conditionsModule', {}).get('conditions', [])
    interventions = protocol.get('armsInterventionsModule', {}).get('interventions', [])
    names = [item.get('name', '') for item in interventions
             if item.get('type') in {'DRUG', 'BIOLOGICAL'}]
    eligibility = protocol.get('eligibilityModule', {}).get('eligibilityCriteria', '') or ''
    structured = ' ; '.join(conditions + names)
    # Literal escaped newlines occur in this saved CT.gov export.
    eligibility = eligibility.replace('\\n', ' ').replace('\\t', ' ')
    return {'structured_text': structured or 'missingprotocol',
            'eligibility_text': eligibility or 'missingprotocol',
            'protocol_text': (structured + ' ; ' + eligibility).strip(' ;') or 'missingprotocol',
            'protocol_missing': int(not structured and not eligibility)}


def attach_protocol(frame, studies):
    lookup = {}
    for study in studies:
        trial_id = study.get('protocolSection', {}).get('identificationModule', {}).get('nctId')
        if trial_id in lookup:
            raise ValueError(f'Duplicate raw trial ID: {trial_id}')
        if trial_id:
            lookup[trial_id] = study
    rows = [protocol_features(lookup.get(trial_id, {})) for trial_id in frame.nct_id]
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def estimator_specs():
    def numeric_model(estimator, interaction=False):
        steps = [SimpleImputer(strategy='median', add_indicator=True), StandardScaler()]
        if interaction:
            steps += [PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
                      StandardScaler()]
        return make_pipeline(ColumnTransformer([('numeric', make_pipeline(*steps), CORE_COLUMNS)]), estimator)

    def text_model(column, c, numeric):
        transformers = [('text', TfidfVectorizer(min_df=2, max_features=5000,
                                                ngram_range=(1, 2), sublinear_tf=True,
                                                stop_words='english'), column)]
        if numeric:
            transformers.append(('numeric', make_pipeline(SimpleImputer(strategy='median', add_indicator=True),
                                                            StandardScaler()), CORE_COLUMNS + ['protocol_missing']))
        preprocessing = ColumnTransformer(transformers, transformer_weights={'text': 3.0} if numeric else None)
        return make_pipeline(preprocessing, LogisticRegression(C=c, max_iter=4000, random_state=SEED))

    specs = {REFERENCE: context_model([]),
             'target_reference': context_model(['primary_target', 'disease', 'modality'])}
    for c in [0.03, 0.3]:
        specs[f'core_lr_c{c}'] = numeric_model(LogisticRegression(C=c, max_iter=4000, random_state=SEED))
    for c in [1, 10]:
        specs[f'core_rbf_c{c}'] = numeric_model(SVC(C=c, gamma='scale', probability=True, random_state=SEED))
    specs['core_interactions'] = numeric_model(LogisticRegression(C=0.03, max_iter=4000, random_state=SEED), True)
    specs['structured_core_c1'] = text_model('structured_text', 1, True)
    specs['eligibility_only_c10'] = text_model('eligibility_text', 10, False)
    specs['protocol_only_c10'] = text_model('protocol_text', 10, False)
    specs['protocol_core_c1'] = text_model('protocol_text', 1, True)
    specs['protocol_core_c10'] = text_model('protocol_text', 10, True)
    return specs


def inner_folds(frame, mode):
    if mode == 'temporal':
        # Choose boundaries by row counts without consulting validation outcomes.
        sorted_years = np.sort(frame.start_year.to_numpy())
        boundaries = sorted(set(sorted_years[[len(frame) // 2, 3 * len(frame) // 4]]))
        partitions = []
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else np.inf
            tr = np.flatnonzero(frame.start_year.lt(start))
            va = np.flatnonzero(frame.start_year.ge(start) & frame.start_year.lt(end))
            partitions.append((tr, va))
    else:
        partitions = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED).split(
            frame, frame.label_strict, frame.target_disease_pair))
    for tr, va in partitions:
        if min(len(tr), len(va)) < 10 or any(frame.iloc[idx].label_strict.nunique() != 2 for idx in [tr, va]):
            raise ValueError('Insufficient inner-fold observations/classes; do not silently skip a fold')
        if set(tr) & set(va):
            raise ValueError('Inner training/validation overlap')
        if mode == 'temporal' and frame.iloc[tr].start_year.max() >= frame.iloc[va].start_year.min():
            raise ValueError('Inner temporal order violated')
        if mode != 'temporal' and set(frame.iloc[tr].target_disease_pair) & set(frame.iloc[va].target_disease_pair):
            raise ValueError('Inner grouped split contains shared pairs')
    return partitions


def choose_candidate(scores):
    ranking = scores.groupby('candidate', as_index=False).agg(auc=('auc', 'mean'), brier=('brier', 'mean'))
    return ranking.sort_values(['auc', 'brier', 'candidate'], ascending=[False, True, True]).iloc[0].candidate


def paired_fold_interval(predictions, repeats=2000):
    """Cluster bootstrap of mean fold AUC difference, conditional on predictions.

This is descriptive development uncertainty, not independent confirmation or a
correction for historical reuse of these outer validation windows.
"""
    clusters = [part.index.to_numpy() for _, part in predictions.groupby('pair', sort=True)]
    rng = np.random.default_rng(SEED)
    differences = []
    y = predictions.label.to_numpy()
    baseline = predictions.reference_probability.to_numpy()
    selected = predictions.probability.to_numpy()
    fold_ids = predictions.fold.to_numpy()
    from sklearn.metrics import roc_auc_score
    for _ in range(repeats):
        indices = np.concatenate([clusters[i] for i in rng.integers(0, len(clusters), len(clusters))])
        deltas = []
        for fold in np.unique(fold_ids):
            subset = indices[fold_ids[indices] == fold]
            if len(np.unique(y[subset])) != 2:
                break
            deltas.append(roc_auc_score(y[subset], selected[subset]) - roc_auc_score(y[subset], baseline[subset]))
        if len(deltas) == len(np.unique(fold_ids)):
            differences.append(np.mean(deltas))
    if not differences:
        raise ValueError('No valid bootstrap resamples')
    low, high = np.quantile(differences, [0.025, 0.975])
    return {'low': float(low), 'high': float(high), 'valid_resamples': len(differences),
            'requested_resamples': repeats, 'cluster': 'target_disease_pair'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    original = pd.read_csv(SOURCE)
    frame = attach_protocol(prepare_training_frame(original), json.loads(RAW_SOURCE.read_text()))
    folds, specs = make_folds(frame), estimator_specs()
    plan = {'seed': SEED, 'n_training': len(frame), 'n_positive': int(frame.label_strict.sum()),
            'n_post2015_scored': 0, 'n_protocol_missing': int(frame.protocol_missing.sum()),
            'excluded_post2015': int(original.start_year.gt(2015).sum()),
            'excluded_missing_year': int(original.start_year.isna().sum()),
            'excluded_invalid_training_label': int((original.start_year.le(2015) & ~original.label_strict.isin([0, 1])).sum()),
            'candidates': list(specs), 'primary_metric': 'mean outer-fold ROC AUC',
            'reference': REFERENCE, 'inner_selection': 'mean AUC descending, mean Brier ascending, name',
            'practical_gain_threshold': 0.05,
            'development_gate': 'nested selected delta AUC >= 0.05, cluster interval low > 0, and no worse Brier in BOTH modes',
            'text_allowlist': ['conditionsModule.conditions', 'armsInterventionsModule.interventions DRUG/BIOLOGICAL name',
                               'eligibilityModule.eligibilityCriteria'],
            'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                              [SOURCE, RAW_SOURCE, Path(__file__), Path('run_context_experiments.py'),
                               Path('run_performance_experiments.py')]},
            'limitations': ['Previously explored development windows; nested selection does not make them fresh data.',
                            'Heuristic labels unchanged; no confirmed trial endpoint labels.',
                            'Saved current protocol and biology features are not historical as-of snapshots.',
                            'Text model extends biology-only scope to trial context; operational language can confound it.',
                            'Unseen pairs can share individual targets/drugs; grouped splits are not chronological.',
                            'Bootstrap conditions on fitted predictions and omits training/selection uncertainty.']}
    (args.output_dir / 'experiment_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    frame.to_csv(args.output_dir / 'training_snapshot.csv', index=False)
    scores, predictions, inner_scores, splits, selected_rows = [], [], [], [], []
    for mode, partitions in folds.items():
        for fold, (tr, va) in enumerate(partitions):
            training, validation = frame.iloc[tr].reset_index(drop=True), frame.iloc[va]
            inner = inner_folds(training, mode)
            for stage, split_number, train_idx, val_idx, pool in [('outer', fold, tr, va, frame)] + [
                    ('inner', k, a, b, training) for k, (a, b) in enumerate(inner)]:
                for role, idx in [('train', train_idx), ('validation', val_idx)]:
                    splits.extend({'mode': mode, 'outer_fold': fold, 'stage': stage, 'fold': split_number,
                                   'role': role, 'nct_id': pool.iloc[i].nct_id} for i in idx)
            current = []
            for name, estimator in specs.items():
                for inner_fold, (a, b) in enumerate(inner):
                    model = clone(estimator).fit(training.iloc[a], training.label_strict.iloc[a])
                    p = model.predict_proba(training.iloc[b])[:, 1]
                    current.append({'mode': mode, 'outer_fold': fold, 'inner_fold': inner_fold,
                                    'candidate': name, **metrics(training.label_strict.iloc[b], p)})
            selected = choose_candidate(pd.DataFrame(current))
            inner_scores.extend(current)
            selected_rows.append({'mode': mode, 'fold': fold, 'selected': selected})
            # Persist the inner choice before scoring this outer validation fold.
            (args.output_dir / 'selected_by_fold.json').write_text(json.dumps(selected_rows, indent=2) + '\n')
            probabilities = {}
            for name, estimator in specs.items():
                model = clone(estimator).fit(training, training.label_strict)
                probabilities[name] = model.predict_proba(validation)[:, 1]
            probabilities['nested_selected'] = probabilities[selected]
            for name, p in probabilities.items():
                scores.append({'mode': mode, 'fold': fold, 'candidate': name, 'n_train': len(tr), 'n_val': len(va),
                               **metrics(validation.label_strict, p)})
                predictions.extend({'mode': mode, 'fold': fold, 'candidate': name, 'nct_id': row.nct_id,
                                    'pair': row.target_disease_pair, 'label': int(row.label_strict),
                                    'probability': float(prob), 'reference_probability': float(ref)}
                                   for (_, row), prob, ref in zip(validation.iterrows(), p, probabilities[REFERENCE]))
            print(f'Completed {mode} outer fold {fold}; inner selection: {selected}', flush=True)
    results = pd.DataFrame(scores)
    summary = results.groupby(['mode', 'candidate'], as_index=False).agg(
        mean_auc=('auc', 'mean'), sd_auc=('auc', 'std'), mean_brier=('brier', 'mean'),
        mean_average_precision=('average_precision', 'mean'))
    pred = pd.DataFrame(predictions)
    results.to_csv(args.output_dir / 'validation_folds.csv', index=False)
    summary.to_csv(args.output_dir / 'validation_summary.csv', index=False)
    pred.to_csv(args.output_dir / 'validation_predictions.csv', index=False)
    pd.DataFrame(inner_scores).to_csv(args.output_dir / 'inner_validation_scores.csv', index=False)
    pd.DataFrame(splits).to_csv(args.output_dir / 'split_ids.csv', index=False)
    checks = {}
    for mode in folds:
        table = summary.loc[summary['mode'].eq(mode)].set_index('candidate')
        selected, reference = table.loc['nested_selected'], table.loc[REFERENCE]
        subset = pred.loc[pred['mode'].eq(mode) & pred.candidate.eq('nested_selected')].reset_index(drop=True)
        interval = paired_fold_interval(subset)
        delta = selected.mean_auc - reference.mean_auc
        checks[mode] = {'mean_auc_difference': float(delta), 'conditional_95pct_interval': interval,
                        'brier_difference': float(selected.mean_brier - reference.mean_brier),
                        'passes_development_gate': bool(delta >= 0.05 and interval['low'] > 0
                                                        and selected.mean_brier <= reference.mean_brier)}
    checks['both_modes_pass'] = all(value['passes_development_gate'] for value in checks.values())
    (args.output_dir / 'development_gate.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == '__main__':
    main()
