"""LEGACY RAW-ONLY LOADER: use prepare_bigquery.py and deploy_bigquery.py for dashboards.

Build - and by default only *print* - the one BigQuery load job this repo needs.

    # default, offline, no credentials required
    PYTHONPATH=src python scripts/load_bigquery.py --dry-run

    # real submission (needs an ADC-authenticated project and the optional client)
    export GOOGLE_CLOUD_PROJECT="your-gcp-project"
    PYTHONPATH=src python scripts/load_bigquery.py --execute

WHAT IT DOES
------------
Stage 00 of ``sql/bigquery`` creates ``landing_hillstrom_raw`` with the twelve
published source columns, verbatim. This script is the file-to-table copy for
that one table: ``data/raw/hillstrom.csv`` -> ``<dataset>.landing_hillstrom_raw``.
Nothing else. ``reference_arms``, ``curated_campaign_customers``, the DQ gate and
every dashboard view stay where they belong - in ``sql/bigquery/*.sql``, which you
run with ``bq query --batch``. This script never executes SQL, never creates a
dataset or table, and never re-derives a column.

NOT EXECUTED AGAINST A LIVE BIGQUERY PROJECT
--------------------------------------------
Matching the warning block in ``sql/bigquery/README.md``: this repository has no
cloud credentials, no ``gcloud``, no Docker and no BigQuery access, so **no load
job described here has ever been submitted - it has never run against a live
project - and no table anywhere was created or written by this script.** The only
mode that has ever run in this repo is ``--dry-run``, which performs no network
call and prints the job configuration it *would* submit. Every run prints a
caveat ahead of the plan, and the plan carries ``executed: false`` unless a job
really has been submitted and read back.

OPTIONAL DEPENDENCY (deliberately not in pyproject.toml)
--------------------------------------------------------
``google-cloud-bigquery`` is not a dependency of this project, so the offline
pipeline keeps running without it. Import it only if you want ``--execute``::

    python -m pip install "google-cloud-bigquery>=3.20"

The client is imported lazily inside :func:`import_bigquery`, so importing this
module - and running the tests - never needs it.

CREDENTIALS AND IDENTIFIERS
---------------------------
Authentication is whatever ``google.auth.default()`` (application-default
credentials) resolves, and nothing more: no tokens, no key files, no cookies are
read, written or printed. The project id and dataset are read from the *env var
names* declared in ``configs/experiment.yaml`` (``bigquery.project_env_var`` and
``bigquery.dataset_env_var``), so no project id, and no personal absolute path,
is ever committed here.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp

from campaign_ml.config import load_config, resolve
from campaign_ml.validation import SOURCE_COLUMNS

DIALECT = "bigquery"

#: Mirrors the ``bq load`` schema strings used in sql/bigquery/README.md.
DDL_TO_LOAD_TYPES = {"INT64": "INTEGER", "FLOAT64": "FLOAT", "STRING": "STRING",
                     "TIMESTAMP": "TIMESTAMP"}

#: Placeholder written into a dry-run plan when no project id resolved. It is
#: deliberately a shell variable, not a value: nothing here may look like a
#: project id that a reader could mistake for a real one.
UNSET_PROJECT = "$PROJECT_ID"

BIGQUERY_INSTALL_HINT = 'python -m pip install "google-cloud-bigquery>=3.20"'

NOT_EXECUTED_NOTICE = (
    "NOT EXECUTED AGAINST A LIVE BIGQUERY PROJECT - this repository has no cloud "
    "credentials and google-cloud-bigquery is not installed, so no load job here "
    "has ever been submitted and no table has been created or written. The block "
    "below is the job this script WOULD submit."
)

SUBMITTING_NOTICE = (
    "UNPROVEN PATH - no load job like this has ever been run in this repository "
    "(see sql/bigquery/README.md). The job below is about to be submitted against "
    "the project you supplied, using your own application-default credentials. "
    "Nothing is confirmed loaded until the final line reports a row count."
)

DRY_RUN_RESULT_LINE = "Dry run complete: no job submitted, no network call made, nothing was loaded."
SUBMITTING_RESULT_LINE = "About to submit. Until a row count is printed below, nothing has been loaded."
EXECUTED_RESULT_LINE = "Load job finished; the row count on the next line was read back from the " \
                       "destination table's metadata (stage 04 remains the real gate)."

ENV_VAR_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")
REQUIRED_KEYS = (
    "project_env_var",
    "dataset_env_var",
    "dataset_default",
    "location",
    "sql_dir",
    "landing_table",
    "curated_table",
    "landing_schema_sql",
    "curated_schema_sql",
    "skip_leading_rows",
    "field_delimiter",
    "write_disposition",
    "stages_after_load",
)
ALLOWED_WRITE_DISPOSITIONS = ("WRITE_TRUNCATE", "WRITE_EMPTY")


class BigQueryPlanError(RuntimeError):
    """The ``bigquery`` config section or a DDL contract does not hold."""


class BigQueryDependencyError(RuntimeError):
    """``google.cloud.bigquery`` is absent, so no job can be submitted."""


@dataclass(frozen=True)
class LoadJobPlan:
    """The exact configuration of the load job, whether or not it gets submitted."""

    dataset: str
    location: str
    landing_table: str
    curated_table: str
    project: str | None
    project_source: str
    source_path: str
    source_uri: str
    source_format: str
    field_delimiter: str
    skip_leading_rows: int
    write_disposition: str
    autodetect: bool
    schema: tuple[str, ...]
    landing_columns: tuple[str, ...]
    curated_columns: tuple[str, ...]
    dry_run: bool
    stages_after_load: tuple[str, ...]

    @property
    def destination_table(self) -> str:
        """``project:dataset.landing_hillstrom_raw``.

        With no project resolved it carries the ``$PROJECT_ID`` placeholder, so a
        dry-run plan can never be misread as "this table was written".
        """
        return f"{self.project or UNSET_PROJECT}:{self.dataset}.{self.landing_table}"

    @property
    def executed(self) -> bool:
        """Always False until a real job has been submitted and awaited."""
        return not self.dry_run

    def as_dict(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "project_source": self.project_source,
            "location": self.location,
            "dataset": self.dataset,
            "destination_table": self.destination_table,
            "source_uri": self.source_uri,
            "source_path": self.source_path,
            "source_format": self.source_format,
            "field_delimiter": self.field_delimiter,
            "skip_leading_rows": self.skip_leading_rows,
            "write_disposition": self.write_disposition,
            "autodetect": self.autodetect,
            "schema": list(self.schema),
            "create_or_overwrite": "the landing table must already exist (stage 00); "
                                   "this script creates nothing",
            "landing_columns_match_source_columns": list(self.landing_columns) == list(SOURCE_COLUMNS),
            "curated_table_columns": list(self.curated_columns),
            "dry_run": self.dry_run,
            "executed": self.executed,
            "stages_after_load_not_run_by_this_script": list(self.stages_after_load),
        }


def bigquery_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Return and validate the ``bigquery`` section of ``configs/experiment.yaml``.

    ``campaign_ml.config.validate_config`` covers the analysis sections; the
    warehouse section is checked here because its contracts are DDL files and
    environment-variable *names*, neither of which the analysis pipeline reads.
    """
    section = config.get("bigquery")
    if not isinstance(section, dict):
        raise BigQueryPlanError(
            "configs/experiment.yaml has no 'bigquery:' section. Add it before "
            "running this script; see sql/bigquery/README.md for the layer it serves."
        )
    missing = [key for key in REQUIRED_KEYS if key not in section]
    if missing:
        raise BigQueryPlanError(f"configs/experiment.yaml bigquery section is missing key(s): {missing}")

    # No hardcoded project id: the config may name an env var, never a value.
    for forbidden in ("project_id", "credentials", "credentials_path", "token", "api_key"):
        if forbidden in section:
            raise BigQueryPlanError(
                f"bigquery.{forbidden} is not allowed in a cloneable config - the project "
                "id must come from the env var named by bigquery.project_env_var."
            )

    errors: list[str] = []
    for key in ("project_env_var", "dataset_env_var"):
        name = section[key]
        if not isinstance(name, str) or not ENV_VAR_PATTERN.match(name):
            errors.append(f"bigquery.{key}={name!r} must be an UPPER_SNAKE env var name")
    for key in ("dataset_default", "location", "sql_dir", "landing_table",
                "curated_table", "landing_schema_sql", "curated_schema_sql"):
        if not str(section[key]).strip():
            errors.append(f"bigquery.{key} must be a non-empty string")
    skip = section["skip_leading_rows"]
    if not isinstance(skip, int) or isinstance(skip, bool) or skip < 1:
        errors.append(f"bigquery.skip_leading_rows={skip!r} must be an int >= 1 (the CSV has a header row)")
    if str(section["field_delimiter"]) != ",":
        errors.append("bigquery.field_delimiter must be ',' - hillstrom.csv is plain comma-delimited")
    if section["write_disposition"] not in ALLOWED_WRITE_DISPOSITIONS:
        errors.append(
            f"bigquery.write_disposition={section['write_disposition']!r} must be one of "
            f"{list(ALLOWED_WRITE_DISPOSITIONS)}"
        )
    stages = section["stages_after_load"]
    valid_stages = (isinstance(stages, list) and stages
                    and all(isinstance(stage, str) and stage.endswith(".sql") for stage in stages))
    if not valid_stages:
        errors.append("bigquery.stages_after_load must be a non-empty list of .sql filenames")
    if errors:
        raise BigQueryPlanError("invalid configs/experiment.yaml bigquery section:\n  - " + "\n  - ".join(errors))

    for key in ("landing_schema_sql", "curated_schema_sql", "sql_dir"):
        if not resolve(section[key]).exists():
            raise BigQueryPlanError(f"bigquery.{key}={section[key]!r} does not resolve to a file or directory")
    if resolve(section["landing_schema_sql"]) != resolve(section["sql_dir"]) / "00_dataset_and_landing_ddl.sql":
        raise BigQueryPlanError("bigquery.landing_schema_sql must be the stage 00 landing DDL")
    return dict(section)


def table_columns(sql_path: Path | str, table_name: str) -> list[tuple[str, str]]:
    """Read a ``CREATE TABLE`` column list out of a DDL file with sqlglot.

    Returns ``[(name, bigquery_load_type), ...]`` in declared order. The DDL files
    are the contract; parsing them keeps this script honest when they change.
    """
    path = Path(sql_path)
    if not path.is_absolute():
        path = resolve(path)
    try:
        statements = sqlglot.parse(path.read_text(encoding="utf-8"), dialect=DIALECT)
    except Exception as exc:  # noqa: BLE001 - a parse failure is the whole signal
        raise BigQueryPlanError(f"{path.name} does not parse as BigQuery SQL: {exc}") from exc

    for statement in statements:
        if not isinstance(statement, exp.Create):
            continue
        schema = statement.this
        if not isinstance(schema, exp.Schema) or not isinstance(schema.this, exp.Table):
            continue  # e.g. CREATE SCHEMA
        if schema.this.name != table_name:
            continue
        columns: list[tuple[str, str]] = []
        for column in schema.expressions:
            kind = column.args.get("kind")
            if kind is None:
                raise BigQueryPlanError(f"{path.name}: column {column.name!r} has no declared type")
            ddl_type = kind.sql(dialect=DIALECT).upper()
            try:
                load_type = DDL_TO_LOAD_TYPES[ddl_type]
            except KeyError as exc:
                raise BigQueryPlanError(
                    f"{path.name}: type {ddl_type!r} on {column.name!r} has no bq-load equivalent; "
                    f"extend DDL_TO_LOAD_TYPES {sorted(DDL_TO_LOAD_TYPES)}"
                ) from exc
            columns.append((str(column.name), load_type))
        if not columns:
            raise BigQueryPlanError(f"{path.name}: CREATE TABLE {table_name} declared no columns")
        return columns
    raise BigQueryPlanError(
        f"{path.name} does not create {table_name!r}. Check bigquery.landing_schema_sql / "
        "bigquery.curated_schema_sql in configs/experiment.yaml against sql/bigquery/."
    )


def assert_ddl_contract(landing: list[tuple[str, str]], curated: list[tuple[str, str]]) -> None:
    """The landing columns must be the curated ones minus the derived/audit columns.

    Stage 03 is what turns a landed row into a curated row, so if the two DDLs
    ever disagree on a shared column's name, order or type, the landing load is
    loading data the curated layer cannot consume. Fail loudly instead.
    """
    landing_names = [name for name, _ in landing]
    curated_names = [name for name, _ in curated]
    if landing_names != list(SOURCE_COLUMNS):
        raise BigQueryPlanError(
            f"landing DDL columns {landing_names} do not match "
            f"campaign_ml.validation.SOURCE_COLUMNS {list(SOURCE_COLUMNS)}"
        )
    shared = [name for name in curated_names if name in set(landing_names)]
    if shared != landing_names:
        raise BigQueryPlanError(
            f"curated DDL column order for the shared columns is {shared}, landing is {landing_names}"
        )
    curated_types = dict(curated)
    mismatched = [(name, load_type, curated_types[name])
                  for name, load_type in landing if curated_types[name] != load_type]
    if mismatched:
        raise BigQueryPlanError(f"landing vs curated type mismatches: {mismatched}")


def schema_from_columns(columns: list[tuple[str, str]]) -> tuple[str, ...]:
    """``[('recency', 'INTEGER'), ...]`` -> ``['recency:INTEGER', ...]`` (``bq load`` form)."""
    return tuple(f"{name}:{load_type}" for name, load_type in columns)


def build_plan(config: dict[str, Any], *, project: str | None = None,
               dataset: str | None = None, location: str | None = None,
               source_path: str | None = None, gcs_uri: str | None = None,
               dry_run: bool = True) -> LoadJobPlan:
    """Assemble the load job configuration. Pure: no import, no network, no files written."""
    settings = bigquery_settings(config)
    landing_columns = table_columns(resolve(settings["landing_schema_sql"]), settings["landing_table"])
    curated_columns = table_columns(resolve(settings["curated_schema_sql"]), settings["curated_table"])
    assert_ddl_contract(landing_columns, curated_columns)

    source = source_path or config["data"]["raw_path"]
    if gcs_uri and not gcs_uri.startswith("gs://"):
        raise BigQueryPlanError(f"--gcs-uri must start with gs://, got {gcs_uri!r}")
    return LoadJobPlan(
        dataset=dataset or settings["dataset_default"],
        location=location or str(settings["location"]),
        landing_table=str(settings["landing_table"]),
        curated_table=str(settings["curated_table"]),
        project=project,
        project_source=project_source_label(project),
        source_path=str(source),
        source_uri=gcs_uri or str(source),
        source_format="CSV",
        field_delimiter=str(settings["field_delimiter"]),
        skip_leading_rows=int(settings["skip_leading_rows"]),
        write_disposition=str(settings["write_disposition"]),
        autodetect=False,
        schema=schema_from_columns(landing_columns),
        landing_columns=tuple(name for name, _ in landing_columns),
        curated_columns=tuple(name for name, _ in curated_columns),
        dry_run=dry_run,
        stages_after_load=tuple(settings["stages_after_load"]),
    )


def project_source_label(project: str | None) -> str:
    if project:
        return "resolved from --project or the configured env var"
    return "unset -> dry-run only; this repo hardcodes no project id"


def bq_cli_command(plan: LoadJobPlan) -> list[str]:
    """The equivalent ``bq`` command, matching sql/bigquery/README.md step 2.

    ``--replace`` is the CLI spelling of WRITE_TRUNCATE. Emitted for review only;
    neither this script nor the tests run it.
    """
    project_flag = plan.project or "$PROJECT_ID"
    location_flag = plan.location
    replace = ["--replace"] if plan.write_disposition == "WRITE_TRUNCATE" else []
    return [
        "bq", f"--project_id={project_flag}", f"--location={location_flag}", "load",
        f"--source_format={plan.source_format}", f"--skip_leading_rows={plan.skip_leading_rows}",
        *replace,
        plan.destination_table, plan.source_uri, ",".join(plan.schema),
    ]


def import_bigquery():
    """Import the optional client lazily, or explain exactly how to install it."""
    try:
        from google.cloud import bigquery  # noqa: PLC0415 - optional dependency, lazy on purpose
    except ImportError as exc:
        raise BigQueryDependencyError(
            f"google-cloud-bigquery is not installed ({exc}), so no load job can be "
            f"submitted and nothing was loaded. Install the optional client with:\n"
            f"    {BIGQUERY_INSTALL_HINT}\n"
            "It is deliberately absent from pyproject.toml: the analysis pipeline runs "
            "offline. Use --dry-run to see the exact job without installing anything."
        ) from exc
    return bigquery


def submit_load(plan: LoadJobPlan, *, expected_rows: int | None = None) -> int:
    """Submit the plan's load job and wait for it. Never reached in this repository.

    Authentication is whatever ``google.auth.default()`` already resolved
    (``gcloud auth application-default login``); this function reads no secrets,
    and it runs no SQL - the stages in ``plan.stages_after_load`` are yours to run.
    """
    bigquery = import_bigquery()
    if not plan.project:
        raise BigQueryPlanError(
            f"no project id resolved ({plan.project_source}); export "
            "GOOGLE_CLOUD_PROJECT or pass --project, otherwise stay in --dry-run."
        )
    client = bigquery.Client(project=plan.project)  # ADC; no credentials handled here
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        field_delimiter=plan.field_delimiter,
        skip_leading_rows=plan.skip_leading_rows,
        write_disposition=getattr(bigquery.WriteDisposition, plan.write_disposition),
        schema=[bigquery.SchemaField(name, load_type) for name, load_type in
                ((item.split(":", 1)) for item in plan.schema)],
        autodetect=plan.autodetect,
    )
    if plan.source_uri.startswith("gs://"):
        job = client.load_table_from_uri([plan.source_uri], plan.destination_table, job_config=job_config)
    else:
        path = resolve(plan.source_uri)
        with open(path, "rb") as handle:
            job = client.load_table_from_file(handle, plan.destination_table, job_config=job_config)
    job.result()  # blocks until the job succeeds or raises
    rows = int(client.get_table(plan.destination_table).num_rows or 0)
    if expected_rows is not None and rows != int(expected_rows):
        raise BigQueryPlanError(
            f"{plan.destination_table} has {rows} rows, expected {expected_rows}. "
            "The landing load is incomplete; do not run the SQL stages on top of it."
        )
    return rows


def render_plan(plan: LoadJobPlan, config: dict[str, Any]) -> str:
    """Human-readable, review-only rendering of the job (and of what was NOT done)."""
    # Repo-relative paths only: an absolute path would be this machine's, not yours.
    sql_dir = str(config["bigquery"]["sql_dir"]).rstrip("/")
    lines = [
        NOT_EXECUTED_NOTICE if plan.dry_run else SUBMITTING_NOTICE,
        "",
        f"Mode: {'dry run' if plan.dry_run else 'execute'}",
        f"Project: {plan.project or 'UNSET'} ({plan.project_source})",
        f"Dataset/location: {plan.dataset} / {plan.location}",
        f"Columns landed: {len(plan.landing_columns)} (validation.SOURCE_COLUMNS, verbatim)",
        f"Downstream contract: {plan.curated_table} has {len(plan.curated_columns)} columns; "
        "stage 02/03 DDL owns them, this script writes none of them",
        "",
        f"Load job configuration {'it WOULD submit' if plan.dry_run else 'being submitted'}:",
        json.dumps(plan.as_dict(), indent=2),
        "",
        "Equivalent bq command (printed for review; not run here):",
        "  " + " ".join(bq_cli_command(plan)),
        "",
        "Then run the SQL yourself, in order - this script does not execute SQL:",
        *[f"  bq --project_id={plan.project or '$PROJECT_ID'} --location={plan.location} "
          f"query --use_legacy_sql=false --batch \"$(cat {sql_dir}/{stage})\""
          for stage in plan.stages_after_load],
        "",
    ]
    lines.append(DRY_RUN_RESULT_LINE if plan.dry_run else SUBMITTING_RESULT_LINE)
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="load_bigquery.py",
        description="Print (default) or submit the landing-table load job for the "
                    "Hillstrom BigQuery layer. See the module docstring: this repo "
                    "has never executed it against a live project.",
    )
    parser.add_argument("--config", default=None, help="path to configs/experiment.yaml")
    parser.add_argument("--project", default=None,
                        help="BigQuery project id; defaults to the env var named by bigquery.project_env_var")
    parser.add_argument("--dataset", default=None, help="override bigquery.dataset_default")
    parser.add_argument("--location", default=None, help="override bigquery.location (some regions require it)")
    parser.add_argument("--source", default=None, help="repo-relative CSV, defaults to data.raw_path")
    parser.add_argument("--gcs-uri", default=None,
                        help="gs:// URI to load instead of the local file (the SDK reads local files too)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="build and print the job only - no import, no network (the default)")
    mode.add_argument("--execute", action="store_true",
                      help="submit the job; requires a project id and google-cloud-bigquery")
    return parser.parse_args(argv)


def resolve_project(settings: dict[str, Any], cli_project: str | None) -> tuple[str | None, str]:
    """Project id from the flag, else the env var the config *names*. Never a literal."""
    if cli_project:
        return cli_project.strip() or None, "--project"
    env_var = settings["project_env_var"]
    value = os.environ.get(env_var, "").strip()
    return (value or None), f"${env_var}" if value else f"${env_var} is unset"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config) if args.config else load_config()
    settings = bigquery_settings(config)

    project, project_source = resolve_project(settings, args.project)
    dataset = args.dataset or os.environ.get(settings["dataset_env_var"], "").strip() or settings["dataset_default"]

    # --execute is the only way out of dry-run, and it needs a project id.
    dry_run = True
    if args.execute:
        if not project:
            print(
                f"{NOT_EXECUTED_NOTICE}\n\n--execute refused: no project id ({project_source}). "
                f"Export {settings['project_env_var']} (or pass --project) and install "
                f"{BIGQUERY_INSTALL_HINT}; otherwise use --dry-run.",
                file=sys.stderr,
            )
            return 2
        dry_run = False

    plan = build_plan(
        config, project=project, dataset=dataset, location=args.location,
        source_path=args.source, gcs_uri=args.gcs_uri, dry_run=dry_run,
    )
    print(render_plan(plan, config))

    if plan.dry_run:
        if not project:
            print(f"(no project id: {project_source} -> dry run)")
        return 0

    try:
        rows = submit_load(plan, expected_rows=int(config["data"]["expected_rows"]))
    except BigQueryDependencyError as exc:
        print(
            f"\nNOT EXECUTED AGAINST A LIVE BIGQUERY PROJECT\n{exc}\n\n"
            "Nothing was loaded. No job was submitted, no table was created or "
            "written, and the SQL stages listed above were not run either.",
            file=sys.stderr,
        )
        return 3
    print(f"{EXECUTED_RESULT_LINE}\n{plan.destination_table}: {rows:,} rows.")
    print(f"Next (NOT run by this script): {', '.join(plan.stages_after_load)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
