"""Offline verification for ``scripts/load_bigquery.py``.

What these tests can and cannot prove, stated plainly: the loader is checked as
a *plan builder*. Nothing here contacts BigQuery - this machine has no gcloud, no
credentials and no ``google-cloud-bigquery`` installed - so the tests assert that
the dry-run plan carries the exact load-job fields, that those fields agree with
the DDL column lists in ``sql/bigquery`` (parsed from the file text, the same way
``tests/test_sql_bigquery.py`` reads them), that the ``bigquery`` config section
validates, and that the execute path refuses to pretend it loaded something.
They do NOT prove that a real load works; only a live project can.
"""

from __future__ import annotations

import importlib.util
import json
import socket
import sys
from pathlib import Path
from typing import Any

import pytest
import sqlglot
from sqlglot import exp

from campaign_ml.config import PROJECT_ROOT, load_config
from campaign_ml.validation import SOURCE_COLUMNS

SQL_DIR = PROJECT_ROOT / "sql" / "bigquery"
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.yaml"
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
DIALECT = "bigquery"

SCRIPT_PATH = PROJECT_ROOT / "scripts" / "load_bigquery.py"


def _load_script_module() -> Any:
    spec = importlib.util.spec_from_file_location("load_bigquery_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


lb = _load_script_module()


@pytest.fixture()
def config() -> dict[str, Any]:
    return load_config()


@pytest.fixture()
def no_cloud_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a machine with no project configured, as this one is."""
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("BIGQUERY_DATASET", raising=False)
    monkeypatch.delenv("BIGQUERY_LOCATION", raising=False)


@pytest.fixture()
def plan(config: dict[str, Any]) -> Any:
    return lb.build_plan(config, project="demo-project")


def parsed_create_columns(path: Path, table: str) -> list[tuple[str, str]]:
    """Independent re-parse of a DDL file, so the script is not graded against itself."""
    for statement in sqlglot.parse(path.read_text(encoding="utf-8"), dialect=DIALECT):
        if isinstance(statement, exp.Create) and isinstance(statement.this, exp.Schema):
            target = statement.this.this
            if isinstance(target, exp.Table) and target.name == table:
                return [
                    (column.name, column.args["kind"].sql(dialect=DIALECT).upper())
                    for column in statement.this.expressions
                ]
    raise AssertionError(f"{path.name} never creates {table!r}")


# --------------------------------------------------------------------------- #
# The dry-run plan: the exact job configuration, with no network and no import.
# --------------------------------------------------------------------------- #


def test_plan_carries_the_load_job_fields_it_would_submit(plan: Any, config: dict[str, Any]) -> None:
    assert plan.project == "demo-project"
    assert plan.dataset == config["bigquery"]["dataset_default"] == "campaign_incrementality"
    assert plan.landing_table == "landing_hillstrom_raw"
    assert plan.destination_table == "demo-project:campaign_incrementality.landing_hillstrom_raw"
    assert plan.source_format == "CSV"
    assert plan.field_delimiter == ","
    assert plan.skip_leading_rows == 1
    assert plan.write_disposition == "WRITE_TRUNCATE"
    assert plan.autodetect is False, "schema is explicit; autodetect would guess the 'Surburban' typo away"
    assert plan.source_uri == "data/raw/hillstrom.csv" == config["data"]["raw_path"]
    assert plan.dry_run is True and plan.executed is False


def test_plan_is_json_serialisable_and_marks_itself_not_executed(plan: Any) -> None:
    payload = json.loads(json.dumps(plan.as_dict()))
    assert payload["executed"] is False
    assert payload["dry_run"] is True
    assert payload["landing_columns_match_source_columns"] is True
    assert "this script creates nothing" in payload["create_or_overwrite"]
    assert payload["stages_after_load_not_run_by_this_script"][0] == "01_reference_arms_ddl.sql"


def test_plan_schema_equals_the_landing_ddl_column_list(plan: Any) -> None:
    ddl = parsed_create_columns(SQL_DIR / "00_dataset_and_landing_ddl.sql", "landing_hillstrom_raw")
    load_ddl = [(name, lb.DDL_TO_LOAD_TYPES[bq_type]) for name, bq_type in ddl]
    assert list(plan.schema) == list(lb.schema_from_columns(load_ddl))
    assert plan.schema[0] == "recency:INTEGER" and plan.schema[-1] == "spend:FLOAT"
    assert plan.landing_columns == tuple(name for name, _ in ddl)
    assert list(plan.landing_columns) == list(SOURCE_COLUMNS), (
        "the loader must reuse campaign_ml.validation.SOURCE_COLUMNS, not restate it"
    )


def test_legacy_bq_cli_equivalent_keeps_raw_only_contract(plan: Any) -> None:
    tokens = lb.bq_cli_command(plan)
    assert tokens[:2] == ["bq", "--project_id=demo-project"]
    assert "--source_format=CSV" in tokens and "--skip_leading_rows=1" in tokens
    assert "--replace" in tokens, "WRITE_TRUNCATE is the SDK spelling of bq's --replace"
    assert plan.destination_table in tokens
    readme = (SQL_DIR / "README.md").read_text(encoding="utf-8")
    assert "legacy raw-only loader" in readme
    assert ",".join(plan.schema) in tokens


def test_dry_run_needs_no_network_and_no_client(config: dict[str, Any], no_cloud_env, monkeypatch, capsys) -> None:
    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("dry run must not touch the network")

    monkeypatch.setattr(socket, "socket", _boom)
    monkeypatch.setattr(socket, "create_connection", _boom)
    assert lb.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "NOT EXECUTED AGAINST A LIVE BIGQUERY PROJECT" in out
    assert "nothing was loaded" in out
    assert "landing_hillstrom_raw" in out
    # The default, with no project id anywhere, is the same dry run.
    assert lb.main([]) == 0
    assert "Mode: dry run" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# The curated DDL is the contract the landing load has to stay compatible with.
# --------------------------------------------------------------------------- #

CURATED = SQL_DIR / "02_curated_campaign_customers_ddl.sql"


def test_curated_ddl_column_list_is_the_downstream_contract() -> None:
    curated = parsed_create_columns(CURATED, "curated_campaign_customers")
    names = [name for name, _ in curated]
    assert len(names) == 18
    for column in ["record_id", "zip_code_raw", "arm_code", "history_band", "campaign_weeks", "data_loaded_utc"]:
        assert column in names, f"{column} is part of the curated contract"
    landing_names = [name for name, _ in parsed_create_columns(
        SQL_DIR / "00_dataset_and_landing_ddl.sql", "landing_hillstrom_raw")]
    assert [name for name in names if name in set(landing_names)] == landing_names
    assert [name for name in names if name not in set(landing_names)] == [
        "record_id", "zip_code_raw", "arm_code", "history_band", "campaign_weeks", "data_loaded_utc"]


def test_plan_columns_agree_with_the_curated_ddl(plan: Any) -> None:
    curated = dict(parsed_create_columns(CURATED, "curated_campaign_customers"))
    assert plan.curated_columns == tuple(curated)
    landing = dict(zip(plan.landing_columns, (item.split(":", 1)[1] for item in plan.schema)))
    for name, load_type in landing.items():
        assert lb.DDL_TO_LOAD_TYPES[curated[name]] == load_type, f"type drift on {name}"


def test_ddl_contract_guard_rejects_drift() -> None:
    landing = [("recency", "INTEGER"), ("history_segment", "STRING")]
    curated = lb.table_columns(CURATED, "curated_campaign_customers")
    with pytest.raises(lb.BigQueryPlanError, match="SOURCE_COLUMNS"):
        # A landing list must be the published 12 columns, nothing shorter.
        lb.assert_ddl_contract(landing, curated)
    with pytest.raises(lb.BigQueryPlanError):
        # A curated list that cannot cover the landing contract.
        lb.assert_ddl_contract([(name, "INTEGER") for name in SOURCE_COLUMNS],
                               [("record_id", "INTEGER")])
    full_landing = lb.table_columns(SQL_DIR / "00_dataset_and_landing_ddl.sql", "landing_hillstrom_raw")
    lb.assert_ddl_contract(full_landing, curated)  # the shipped pair passes
    tampered = list(full_landing)
    tampered[2] = ("history", "INTEGER")  # FLOAT64 -> INT64 drift between the two DDLs
    with pytest.raises(lb.BigQueryPlanError, match="type"):
        lb.assert_ddl_contract(tampered, curated)


def test_table_columns_fails_loudly_on_a_wrong_table_name() -> None:
    with pytest.raises(lb.BigQueryPlanError, match="does not create"):
        lb.table_columns(CURATED, "no_such_table")


# --------------------------------------------------------------------------- #
# Config section.
# --------------------------------------------------------------------------- #


def test_bigquery_section_validates_and_lists_the_keys(config: dict[str, Any]) -> None:
    settings = lb.bigquery_settings(config)
    assert set(lb.REQUIRED_KEYS) <= set(settings)
    assert settings["project_env_var"] == "GOOGLE_CLOUD_PROJECT"
    assert settings["dataset_env_var"] == "BIGQUERY_DATASET"
    assert settings["landing_schema_sql"] == "sql/bigquery/00_dataset_and_landing_ddl.sql"
    assert settings["curated_schema_sql"] == "sql/bigquery/02_curated_campaign_customers_ddl.sql"
    assert settings["stages_after_load"][0] == "01_reference_arms_ddl.sql"
    assert "00_dataset_and_landing_ddl.sql" not in settings["stages_after_load"], (
        "stage 00 creates the landing table this script loads into; it is a prerequisite"
    )
    assert all((SQL_DIR / stage).exists() for stage in settings["stages_after_load"])


def test_config_comments_state_the_load_was_never_executed() -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    section = text.split("bigquery:", 1)[1]
    assert "not executed" in section.lower()
    assert "never been created" in section.lower() or "has ever" in section.lower()
    assert "dry-run" in section.lower()


def test_build_plan_revalidates_the_section(config: dict[str, Any]) -> None:
    broken = json.loads(json.dumps({k: str(v) for k, v in config["bigquery"].items()}))
    with pytest.raises(lb.BigQueryPlanError):
        lb.bigquery_settings({**config, "bigquery": broken})


@pytest.mark.parametrize("mutate", [
    lambda s: s.pop("project_env_var"),
    lambda s: s.pop("stages_after_load"),
    lambda s: s.update(project_id="a-real-project-id"),
    lambda s: s.update(credentials_path="~/.config/gcloud/application_default_credentials.json"),
    lambda s: s.update(project_env_var="google_cloud_project"),
    lambda s: s.update(skip_leading_rows=0),
    lambda s: s.update(field_delimiter="|"),
    lambda s: s.update(write_disposition="WRITE_APPEND"),
    lambda s: s.update(curated_schema_sql="sql/bigquery/99_nope.sql"),
    lambda s: s.update(sql_dir="sql/other"),
    lambda s: s.update(stages_after_load=["01_reference_arms_ddl.txt"]),
], ids=["missing-env-var", "missing-stages", "hardcoded-project", "credential-path",
        "lowercase-env-var", "header-loaded", "wrong-delimiter", "append-mode",
        "missing-ddl-file", "missing-sql-dir", "non-sql-stage"])
def test_bad_bigquery_sections_are_rejected(config: dict[str, Any], mutate) -> None:
    settings = dict(config["bigquery"])
    mutate(settings)
    with pytest.raises(lb.BigQueryPlanError):
        lb.bigquery_settings({**config, "bigquery": settings})


def test_dataset_and_location_can_be_overridden(config: dict[str, Any]) -> None:
    plan = lb.build_plan(config, project="p", dataset="my_dataset", location="EU", dry_run=True)
    assert plan.destination_table == "p:my_dataset.landing_hillstrom_raw"
    assert plan.location == "EU"
    assert lb.bq_cli_command(plan)[1] == "--project_id=p"


def test_gcs_uri_must_be_a_bucket_path(config: dict[str, Any]) -> None:
    with pytest.raises(lb.BigQueryPlanError, match="gs://"):
        lb.build_plan(config, project="p", gcs_uri="https://example.com/hillstrom.csv")
    plan = lb.build_plan(config, project="p", gcs_uri="gs://my-bucket/hillstrom.csv")
    assert plan.source_uri == "gs://my-bucket/hillstrom.csv"
    assert plan.source_path == "data/raw/hillstrom.csv", "the local pinned file stays recorded"


# --------------------------------------------------------------------------- #
# No project -> dry run. Missing client -> clear error. Nothing is ever claimed.
# --------------------------------------------------------------------------- #


def test_missing_project_id_forces_dry_run(config: dict[str, Any], no_cloud_env) -> None:
    settings = lb.bigquery_settings(config)
    project, source = lb.resolve_project(settings, None)
    assert project is None
    assert "unset" in source
    plan = lb.build_plan(config, project=project)
    assert plan.dry_run is True and plan.executed is False
    assert lb.UNSET_PROJECT in plan.destination_table


def test_execute_without_a_project_id_is_refused(config: dict[str, Any], no_cloud_env,
                                                 monkeypatch, capsys) -> None:
    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("--execute without a project must fail before importing anything")

    monkeypatch.setattr(lb, "import_bigquery", _boom)
    assert lb.main(["--execute"]) == 2
    err = capsys.readouterr().err
    assert "NOT EXECUTED" in err
    assert lb.BIGQUERY_INSTALL_HINT in err
    assert "--dry-run" in err


def test_submit_load_explains_the_missing_optional_dependency(no_cloud_env, monkeypatch) -> None:
    for name in ("google", "google.cloud", "google.cloud.bigquery"):
        monkeypatch.setitem(sys.modules, name, None)
    plan = lb.build_plan(load_config(), project="demo-project")
    with pytest.raises(lb.BigQueryDependencyError) as excinfo:
        lb.submit_load(plan, expected_rows=64000)
    message = str(excinfo.value)
    assert lb.BIGQUERY_INSTALL_HINT in message
    assert "nothing was loaded" in message
    assert "pyproject.toml" in message, "the error must say why it is not a hard dependency"


def test_execute_path_reports_the_missing_dependency_rather_than_a_load(
        no_cloud_env, monkeypatch, capsys) -> None:
    for name in ("google", "google.cloud", "google.cloud.bigquery"):
        monkeypatch.setitem(sys.modules, name, None)
    assert lb.main(["--execute", "--project", "demo-project"]) == 3
    captured = capsys.readouterr()
    # Before the job runs, the output must say "about to", never "done".
    assert "About to submit" in captured.out
    assert "nothing has been loaded" in captured.out
    assert "Load job finished" not in captured.out
    assert "Nothing was loaded" in captured.err
    assert "NOT EXECUTED AGAINST A LIVE BIGQUERY PROJECT" in captured.err


def test_client_is_imported_lazily_never_at_module_scope() -> None:
    lines = SCRIPT_PATH.read_text(encoding="utf-8").splitlines()
    statement = "from google.cloud import bigquery"
    imports = [index for index, line in enumerate(lines) if line.strip().startswith(statement)]
    assert len(imports) == 1, "exactly one import of the optional client"
    index = imports[0]
    assert lines[index].startswith("    "), "it must be indented inside a function, not at module scope"
    assert index > min(i for i, line in enumerate(lines) if line.startswith("def import_bigquery"))


def test_google_cloud_bigquery_is_not_a_hard_dependency() -> None:
    import tomllib
    config = tomllib.loads(PYPROJECT.read_text(encoding='utf-8'))['project']
    assert not any(d.startswith('google-cloud-bigquery') for d in config['dependencies'])
    assert any(d.startswith('google-cloud-bigquery') for d in config['optional-dependencies']['warehouse'])
    assert any(d.startswith('sqlglot') for d in config['dependencies'])


def test_no_credentials_or_personal_paths_are_committed() -> None:
    for path in (SCRIPT_PATH, CONFIG_PATH):
        text = path.read_text(encoding="utf-8")
        for forbidden in ("C:\\Users", "C:/Users", "/home/", "application_default_credentials.json",
                          "BEGIN PRIVATE KEY", "ya29.", "accounts.google.com"):
            assert forbidden not in text, f"{path.name} leaks {forbidden!r}"
    # Authentication is ADC and nothing more; no project id is written down.
    script = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "google.auth.default()" in script
    assert "Client" in script and "credentials=" not in script
    assert lb.UNSET_PROJECT == "$PROJECT_ID"


def test_the_not_executed_notice_is_in_the_module_docstring() -> None:
    doc = (lb.__doc__ or "").lower()
    assert "not executed against a live bigquery project" in doc
    assert "never been submitted" in doc or "has never" in doc
    assert lb.BIGQUERY_INSTALL_HINT in (lb.__doc__ or "")
    readme = (SQL_DIR / "README.md").read_text(encoding="utf-8").lower()
    assert "not executed against a live bigquery project" in readme, (
        "the script must match the caveat the SQL layer already publishes"
    )
