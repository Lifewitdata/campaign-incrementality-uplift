"""Offline BigQuery contracts. Never regenerate record IDs inside the warehouse."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype

from .evaluation import _descending_rank, assign_deciles
from .features import make_splits
from .validation import load_raw, prepare_clean_table, run_all_checks


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_scores(clean: pd.DataFrame, scores: pd.DataFrame, config: dict,
                   learner: str = 't_learner') -> pd.DataFrame:
    """Select one learner and recreate the exact evaluation ordering and tie seed."""
    primary = config['modelling']['primary_comparison']
    comparison = next(c for c in config['experiment']['comparisons'] if c['name'] == primary)
    test = make_splits(clean, config)['test'].set_index('record_id')
    scores = scores[scores.comparison == primary]
    picked = scores[scores.learner == learner].set_index('record_id')
    response = scores[scores.learner == 'propensity'].set_index('record_id')
    for frame in (picked, response):
        if frame.index.has_duplicates or set(frame.index) != set(test.index):
            raise ValueError('Scores must contain each held-out record exactly once per learner')
    test = test[test.segment.isin([comparison['treatment'], comparison['control']])]
    out = picked.loc[test.index, ['comparison', 'uplift_score']].copy()
    out['response_score'] = response.loc[test.index, 'response_score']
    if not np.isfinite(out[['uplift_score', 'response_score']].to_numpy()).all():
        raise ValueError('Warehouse scores must be finite')
    if not out.response_score.between(0, 1).all() or not out.uplift_score.between(-1, 1).all():
        raise ValueError('Score probability outside its domain')
    seed = int(config['experiment']['bootstrap']['seed'])
    out['decile'] = assign_deciles(out.uplift_score, 10, seed)
    out['rank_position'] = _descending_rank(out.uplift_score, seed)
    out['learner'] = learner
    out['split'] = 'test'
    return out.reset_index()


def reconcile_scores(clean: pd.DataFrame, scores: pd.DataFrame, tables: Path) -> None:
    """Fail before upload if scores no longer reproduce the published evidence."""
    frame = scores.merge(clean, on='record_id', validate='one_to_one')
    learner = scores.learner.iloc[0]
    comparison = scores.comparison.iloc[0]
    treatment = {'mens_vs_control': 'mens', 'womens_vs_control': 'womens'}[comparison]
    deciles = pd.read_csv(tables / 'uplift_deciles_holdout.csv')
    expected = deciles[deciles.learner == learner].set_index('decile')
    for decile, group in frame.groupby('decile'):
        t = group[group.arm_code == treatment]
        c = group[group.arm_code == 'control']
        actual = [len(group), len(t), len(c), t.conversion.sum(), c.conversion.sum(), t.spend.sum(), c.spend.sum()]
        columns = ['n_customers', 'n_treated', 'n_control', 'conversions_treated', 'conversions_control', 'spend_total_treated', 'spend_total_control']
        np.testing.assert_allclose(actual, expected.loc[decile, columns].to_numpy(float), rtol=1e-9, atol=1e-8)
    policies = pd.read_csv(tables / 'targeting_policy_values.csv')
    policies = policies[(policies.strategy == 'uplift_' + learner) & (policies.comparison == comparison)]
    if len(policies) == 0:
        raise ValueError('Missing published policy evidence for selected learner')
    for row in policies.itertuples():
        selected = frame[frame.rank_position <= round(row.share_targeted * len(frame))]
        t, c = selected[selected.arm_code == treatment], selected[selected.arm_code == 'control']
        actual = [len(selected), len(t), len(c), (t.conversion.mean()-c.conversion.mean())*len(t), (t.spend.mean()-c.spend.mean())*len(t)]
        expected_values = [row.n_selected, row.n_selected_treated, row.n_selected_control, row.incremental_conversions_in_slice, row.incremental_spend_in_slice_usd]
        np.testing.assert_allclose(actual, expected_values, rtol=1e-9, atol=1e-8)


def write_table(frame: pd.DataFrame, table: str, directory: Path) -> dict:
    """NDJSON avoids CSV header, quoting and timestamp inference problems."""
    schema = []
    for name, dtype in frame.dtypes.items():
        kind = 'BOOLEAN' if is_bool_dtype(dtype) else 'INTEGER' if is_integer_dtype(dtype) else 'FLOAT' if is_numeric_dtype(dtype) else 'STRING'
        schema.append({'name': name, 'type': kind, 'mode': 'NULLABLE'})
    path = directory / f'{table}.jsonl'
    frame.to_json(path, orient='records', lines=True, double_precision=15)
    return {'table': table, 'file': path.name, 'rows': len(frame), 'sha256': digest(path), 'schema': schema}


def prepare_bundle(root: Path, config: dict, learner: str = 't_learner') -> dict:
    raw_path = root / config['data']['raw_path']
    if digest(raw_path) != config['data']['expected_sha256']:
        raise ValueError('Raw checksum differs from the pinned analysis input; refusing upload')
    raw = load_raw(raw_path)
    run_all_checks(raw, expected_rows=config['data']['expected_rows']).raise_if_failed()
    clean = prepare_clean_table(raw)
    indexed = raw.copy()
    indexed.insert(0, 'record_id', clean.record_id.to_numpy())
    tables = root / 'outputs/tables'
    source_scores = tables / 'holdout_customer_scores.csv'
    scores = prepare_scores(clean, pd.read_csv(source_scores), config, learner)
    reconcile_scores(clean, scores, tables)
    directory = root / 'outputs/bigquery'
    directory.mkdir(parents=True, exist_ok=True)
    exports = [write_table(indexed, 'landing_hillstrom_indexed', directory),
               write_table(scores, 'uplift_customer_scores', directory)]
    # Preserve all three learners, uncertainty and policy baselines in evidence tables.
    for path in sorted((root / 'outputs/dashboard').glob('*.csv')):
        if path.stem != 'uplift_customer_scores':
            exports.append(write_table(pd.read_csv(path), 'evidence_' + path.stem, directory))
    manifest = {'version': 1, 'executed': False, 'source_sha256': digest(raw_path),
                'scores_sha256': digest(source_scores), 'learner': learner,
                'record_id': 'zero-based original source file row; not customer identity',
                'local_score_reconciliation': 'passed', 'tables': exports}
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest
