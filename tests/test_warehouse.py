"""Integration contracts across real source records, published scores and SQL."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from campaign_ml.config import PROJECT_ROOT, load_config
from campaign_ml.validation import load_raw, prepare_clean_table
from campaign_ml.warehouse import prepare_scores, reconcile_scores, write_table

spec = importlib.util.spec_from_file_location('deploy_bigquery', PROJECT_ROOT/'scripts/deploy_bigquery.py')
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)

@pytest.fixture(scope='module')
def warehouse_data():
    config = load_config()
    raw = PROJECT_ROOT/config['data']['raw_path']
    if not raw.exists():
        pytest.skip('Run ingestion for real-data contract tests')
    clean = prepare_clean_table(load_raw(raw))
    original = pd.read_csv(PROJECT_ROOT/'outputs/tables/holdout_customer_scores.csv')
    return clean, original, config

@pytest.mark.parametrize('learner', ['s_learner', 't_learner'])
def test_export_reproduces_published_deciles_and_policies(warehouse_data, learner):
    clean, original, config = warehouse_data
    scores = prepare_scores(clean, original, config, learner)
    assert len(scores) == 8523
    assert scores.record_id.is_unique
    assert sorted(scores.rank_position) == list(range(1, 8524))
    assert set(scores.merge(clean).arm_code) == {'mens', 'control'}
    reconcile_scores(clean, scores, PROJECT_ROOT/'outputs/tables')


def test_offset_ids_are_rejected(warehouse_data):
    clean, original, config = warehouse_data
    original = original.copy()
    original.record_id += 1
    with pytest.raises(ValueError, match='held-out record'):
        prepare_scores(clean, original, config)


def test_permuted_scores_fail_evidence_reconciliation(warehouse_data):
    clean, original, config = warehouse_data
    scores = prepare_scores(clean, original, config)
    scores.record_id = np.roll(scores.record_id.to_numpy(), 1)
    with pytest.raises(AssertionError):
        reconcile_scores(clean, scores, PROJECT_ROOT/'outputs/tables')


def test_duplicate_scores_are_rejected(warehouse_data):
    clean, original, config = warehouse_data
    duplicate = pd.concat([original, original[original.learner == 't_learner'].head(1)])
    with pytest.raises(ValueError, match='exactly once'):
        prepare_scores(clean, duplicate, config)


def test_ndjson_schema_and_tamper_detection(tmp_path):
    frame = pd.DataFrame({'record_id': [0, 1], 'effect': [0.01, np.nan], 'label': ['a,b', 'c'], 'passed': [True, False]})
    table = write_table(frame, 'example', tmp_path)
    (tmp_path/'manifest.json').write_text(json.dumps({'tables': [table]}))
    assert [c['type'] for c in table['schema']] == ['INTEGER', 'FLOAT', 'STRING', 'BOOLEAN']
    deploy.verify_manifest(tmp_path)
    (tmp_path/table['file']).write_text('{}\n')
    with pytest.raises(ValueError, match='Changed'):
        deploy.verify_manifest(tmp_path)


def test_project_qualification_and_identifier_rejection():
    sql = 'SELECT record_id FROM `campaign_incrementality.curated_campaign_customers`; CREATE SCHEMA `campaign_incrementality`'
    rendered = deploy.qualified_sql(sql, 'customer-churn-clv', 'campaign_incrementality')
    assert '`customer-churn-clv.campaign_incrementality.curated_campaign_customers`' in rendered
    assert '`customer-churn-clv.campaign_incrementality`' in rendered
    with pytest.raises(ValueError):
        deploy.qualified_sql(sql, 'bad`;DROP', 'campaign_incrementality')
    with pytest.raises(ValueError):
        deploy.qualified_sql(sql, 'valid-project', 'bad.name')
