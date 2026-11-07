"""Preview by default; --execute loads and verifies a dedicated BigQuery dataset."""
from __future__ import annotations
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from campaign_ml.config import PROJECT_ROOT, load_config
from campaign_ml.validation import load_raw, prepare_clean_table
from campaign_ml.warehouse import digest


def qualified_sql(text: str, project: str, dataset: str) -> str:
    if not re.fullmatch(r'[a-z][a-z0-9-]{4,61}[a-z0-9]', project):
        raise ValueError('Invalid Google Cloud project ID')
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,1023}', dataset):
        raise ValueError('Invalid BigQuery dataset ID')
    return text.replace('`campaign_incrementality.', f'`{project}.{dataset}.').replace(
        '`campaign_incrementality`', f'`{project}.{dataset}`')


def verify_manifest(directory: Path) -> dict:
    manifest = json.loads((directory / 'manifest.json').read_text())
    for table in manifest['tables']:
        path = directory / table['file']
        if path.parent != directory or digest(path) != table['sha256']:
            raise ValueError(f'Changed or invalid export: {table["file"]}')
    return manifest


def run(args) -> dict:
    root = PROJECT_ROOT
    directory = root / 'outputs/bigquery'
    manifest = verify_manifest(directory)
    config = load_config()
    if digest(root / config['data']['raw_path']) != manifest['source_sha256']:
        raise ValueError('Raw data changed since preparation')
    scripts = sorted((root / 'sql/bigquery').glob('*.sql'))
    rendered = {p.name[:2]: qualified_sql(p.read_text(), args.project, args.dataset) for p in scripts}
    receipt = {'executed': False, 'project': args.project, 'dataset': args.dataset,
               'location': args.location, 'source_sha256': manifest['source_sha256'],
               'jobs': [], 'verification': 'not run',
               'writes': [t['table'] for t in manifest['tables']],
               'warning': 'Execute replaces project-owned tables in this dedicated dataset.'}
    if not args.execute:
        return receipt
    from google.cloud import bigquery
    client = bigquery.Client(project=args.project)
    dataset_id = f'{args.project}.{args.dataset}'
    receipt['executed'] = True
    receipt['started_at'] = datetime.now(timezone.utc).isoformat()
    receipt_path = directory / 'deployment_receipt.json'

    def save():
        receipt_path.write_text(json.dumps(receipt, indent=2, default=str)+'\n')

    def query(sql):
        job = client.query(sql, location=args.location,
                           job_config=bigquery.QueryJobConfig(maximum_bytes_billed=args.maximum_bytes_billed))
        receipt['jobs'].append({'id': job.job_id, 'kind': 'query'})
        save()
        return list(job.result())

    try:
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = args.location
        client.create_dataset(dataset, exists_ok=True)
        existing = client.get_dataset(dataset_id)
        if existing.location.upper() != args.location.upper():
            raise ValueError('Existing dataset location does not match --location')
        for stage in ('00', '01', '02', '05', '06'):
            query(rendered[stage])
        for table in manifest['tables']:
            schema = [bigquery.SchemaField.from_api_repr(c) for c in table['schema']]
            job_config = bigquery.LoadJobConfig(schema=schema,
                source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE)
            with (directory / table['file']).open('rb') as source:
                job = client.load_table_from_file(source, dataset_id+'.'+table['table'],
                    job_config=job_config, location=args.location)
            receipt['jobs'].append({'id': job.job_id, 'kind': 'load', 'table': table['table']})
            save()
            job.result()
            if client.get_table(dataset_id+'.'+table['table']).num_rows != table['rows']:
                raise ValueError('Cloud row-count mismatch: '+table['table'])
        query(rendered['03'])
        checks = [dict(row) for row in query(rendered['04'])]
        receipt['data_quality'] = checks
        if not checks or not all(row['passed'] for row in checks):
            raise ValueError('Warehouse data quality gate failed')
        # Whole-row reconciliation catches ID permutations that row counts cannot.
        clean = prepare_clean_table(load_raw(root / config['data']['raw_path'])).sort_values('record_id')
        columns = ', '.join(f'`{c}`' for c in clean.columns)
        actual = pd.DataFrame([dict(row) for row in query(
            f'SELECT {columns} FROM `{dataset_id}.campaign_customers_csv` ORDER BY record_id')])
        for column in clean:
            if pd.api.types.is_numeric_dtype(clean[column]):
                np.testing.assert_allclose(actual[column], clean[column], rtol=1e-10, atol=1e-8)
            elif actual[column].astype(str).tolist() != clean[column].astype(str).tolist():
                raise ValueError('Curated row mismatch: '+column)
        for stage in ('07', '08', '09'):
            query(rendered[stage])
        # Reconcile SQL scenario arithmetic with the published Python policies.
        policies = pd.read_csv(root / 'outputs/tables/targeting_policy_values.csv')
        policies = policies[policies.strategy == 'uplift_'+manifest['learner']]
        economics = query(f'SELECT comparison, targeting_share, modelled_emails_sent, modelled_incremental_revenue, modelled_incremental_conversions, email_cost_per_send, gross_margin_rate, modelled_net_contribution_scenario FROM `{dataset_id}.dashboard_targeting_scenario_economics`')
        if len(economics) != len(policies)*9:
            raise ValueError('Unexpected economics row count')
        for row in economics:
            matches = policies[(policies.comparison == row['comparison']) & np.isclose(policies.share_targeted, row['targeting_share'])]
            if len(matches) != 1:
                raise ValueError('Missing or duplicate policy reference')
            ref = matches.iloc[0]
            expected_net = ref.incremental_spend_in_slice_usd*row['gross_margin_rate'] - ref.n_selected_treated*row['email_cost_per_send']
            np.testing.assert_allclose([row['modelled_emails_sent'], row['modelled_incremental_revenue'], row['modelled_incremental_conversions'], row['modelled_net_contribution_scenario']], [ref.n_selected_treated, ref.incremental_spend_in_slice_usd, ref.incremental_conversions_in_slice, expected_net], rtol=1e-9, atol=1e-7)
        receipt['verification'] = 'passed: counts, DQ, every curated row, policy economics'
    except Exception as exc:
        receipt['verification'] = 'failed'
        receipt['error'] = str(exc)
        raise
    finally:
        save()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True)
    parser.add_argument('--dataset', default='campaign_incrementality')
    parser.add_argument('--location', default='US')
    parser.add_argument('--maximum-bytes-billed', type=int, default=1_000_000_000)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, default=str))

if __name__ == '__main__':
    main()
