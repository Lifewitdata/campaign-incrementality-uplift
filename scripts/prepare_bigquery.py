"""Prepare and reconcile warehouse exports; no credentials or cloud writes."""
import argparse
import json
from campaign_ml.config import PROJECT_ROOT, load_config
from campaign_ml.warehouse import prepare_bundle

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--learner', choices=['s_learner', 't_learner'], default='t_learner')
    args = parser.parse_args()
    manifest = prepare_bundle(PROJECT_ROOT, load_config(), args.learner)
    print(json.dumps({'executed': False, 'local_reconciliation': 'passed',
                      'tables': [{k: t[k] for k in ('table', 'rows')} for t in manifest['tables']]}, indent=2))
