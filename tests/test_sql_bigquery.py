"""Static verification for the BigQuery SQL layer in ``sql/bigquery``.

What this can and cannot prove, stated plainly: sqlglot parsing plus a handful
of contract lints. Nothing here has been run against a live BigQuery project
(the author has no cloud credentials), so these tests check that every file is
parseable BigQuery standard SQL, that the expected stages and objects exist,
that column lists are explicit, and that each file documents its source and the
scenario caveat. They do NOT prove the queries return the rows we expect.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import sqlglot

SQL_DIR = Path(__file__).resolve().parents[1] / "sql" / "bigquery"
README = SQL_DIR / "README.md"
DIALECT = "bigquery"

EXPECTED_STAGES = [
    "00_dataset_and_landing_ddl.sql",
    "01_reference_arms_ddl.sql",
    "02_curated_campaign_customers_ddl.sql",
    "03_curated_campaign_customers_insert.sql",
    "04_data_quality_checks.sql",
    "05_reference_scenarios_ddl.sql",
    "06_uplift_customer_scores_ddl.sql",
    "07_views_dashboard_campaign_incrementality.sql",
    "08_views_dashboard_uplift.sql",
    "09_views_dashboard_economics.sql",
]
EXPECTED_TABLES = [
    "landing_hillstrom_raw",
    "reference_arms",
    "curated_campaign_customers",
    "reference_scenarios",
    "uplift_customer_scores",
]
EXPECTED_VIEWS = [
    "campaign_customers_csv",
    "dashboard_campaign_incrementality",
    "dashboard_campaign_incrementality_effects",
    "dashboard_uplift_deciles",
    "dashboard_uplift_summary",
    "dashboard_targeting_scenario_economics",
]
HEADER_TOKENS = ["Purpose:", "Inputs:", "Outputs:", "Assumptions:"]
DATASET = "campaign_incrementality"

STAR_PATTERN = re.compile(r"\bselect\s+(?:all\s+|distinct\s+)?(?:\w+\s*\.\s*)?\*", re.IGNORECASE)
LEGACY_PATTERN = re.compile(r"\b(join each|omit record|tablesample|table_sample)\b", re.IGNORECASE)


def sql_files() -> list[Path]:
    return sorted(SQL_DIR.glob("*.sql"))


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def split_comments(text: str) -> tuple[str, str]:
    """Return (code, comments) by stripping ``--`` line comments outside strings."""
    code_lines, comment_lines = [], []
    for line in text.splitlines():
        code_parts: list[str] = []
        in_string = False
        index = 0
        while index < len(line):
            char = line[index]
            if char == "'":
                in_string = not in_string
            if char == "-" and not in_string and line[index : index + 2] == "--":
                break
            code_parts.append(char)
            index += 1
        code_lines.append("".join(code_parts))
        if not in_string and "--" in line:
            comment_lines.append(line[line.index("--") + 2 :])
    return "\n".join(code_lines), "\n".join(comment_lines)


def statements(path: Path) -> list[sqlglot.exp.Expression]:
    parsed = sqlglot.parse(read(path), dialect=DIALECT)
    return [statement for statement in parsed if statement is not None]


FILE_IDS = [path.name for path in sql_files()]


def test_sql_directory_exists_and_is_populated() -> None:
    assert SQL_DIR.is_dir(), f"missing {SQL_DIR}"
    assert sql_files(), "no .sql files discovered"


def test_file_set_matches_expected_stages_in_load_order() -> None:
    found = [path.name for path in sql_files()]
    assert found == EXPECTED_STAGES, f"unexpected file set or order: {found}"
    assert found == sorted(found), "filename prefixes must sort into load order"


@pytest.mark.parametrize("path", sql_files(), ids=FILE_IDS)
def test_parses_as_bigquery_standard_sql(path: Path) -> None:
    """sqlglot must accept every statement, and regenerate it as BigQuery."""
    try:
        parsed = sqlglot.parse(read(path), dialect=DIALECT)
    except Exception as exc:  # noqa: BLE001 - any parse failure is the signal we want
        pytest.fail(f"{path.name} does not parse under dialect={DIALECT}: {exc}")
    kept = [statement for statement in parsed if statement is not None]
    assert kept, f"{path.name} produced no statements"
    for statement in kept:
        regenerated = statement.sql(dialect=DIALECT)
        assert regenerated.strip(), f"{path.name} regenerated an empty statement"
        assert sqlglot.parse_one(regenerated, dialect=DIALECT) is not None


@pytest.mark.parametrize("path", sql_files(), ids=FILE_IDS)
def test_no_legacy_dialect_markers(path: Path) -> None:
    code, _ = split_comments(read(path))
    match = LEGACY_PATTERN.search(code)
    assert match is None, f"{path.name} uses legacy-only syntax: {match and match.group(0)}"


@pytest.mark.parametrize("path", sql_files(), ids=FILE_IDS)
def test_no_star_in_any_select_list(path: Path) -> None:
    """Explicit column lists only - a star would silently break the CSV contract."""
    code, _ = split_comments(read(path))
    match = STAR_PATTERN.search(code)
    assert match is None, f"{path.name} has a star projection near: {code[match.start() - 30 :][:70]!r}"


@pytest.mark.parametrize("path", sql_files(), ids=FILE_IDS)
def test_every_file_documents_source_or_scenario_caveat(path: Path) -> None:
    _, comments = split_comments(read(path))
    lowered = comments.lower()
    documents_source = "hillstrom" in lowered
    documents_scenario = "modelled" in lowered and "realised roi" in lowered
    assert documents_source or documents_scenario, (
        f"{path.name} documents neither the Hillstrom source nor the modelled-scenario caveat"
    )


@pytest.mark.parametrize("path", sql_files(), ids=FILE_IDS)
def test_every_file_has_a_documented_header(path: Path) -> None:
    _, comments = split_comments(read(path))
    for token in HEADER_TOKENS:
        assert token in comments, f"{path.name} header is missing '{token}'"
    assert "not executed" in comments.lower(), f"{path.name} header must state it was not executed"


def test_expected_table_ddl_is_present() -> None:
    blob = "\n".join(read(path) for path in sql_files())
    for table in EXPECTED_TABLES:
        pattern = rf"CREATE TABLE IF NOT EXISTS\s+`{DATASET}\.{table}`"
        assert re.search(pattern, blob, re.IGNORECASE), f"missing DDL for {DATASET}.{table}"


def test_expected_view_definitions_are_present() -> None:
    blob = "\n".join(read(path) for path in sql_files())
    for view in EXPECTED_VIEWS:
        pattern = rf"CREATE OR REPLACE VIEW\s+`{DATASET}\.{view}`"
        assert re.search(pattern, blob, re.IGNORECASE), f"missing view definition for {DATASET}.{view}"


def test_curated_table_ddl_matches_the_published_column_contract() -> None:
    ddl = read(SQL_DIR / "02_curated_campaign_customers_ddl.sql")
    published = [
        "record_id", "recency", "history_segment", "history", "mens", "womens",
        "zip_code", "newbie", "channel", "segment", "visit", "conversion", "spend",
        "arm_code", "history_band", "campaign_weeks",
    ]
    for column in published + ["zip_code_raw", "data_loaded_utc"]:
        assert re.search(rf"^\s+{column}\s+(INT64|FLOAT64|STRING|TIMESTAMP)\b", ddl, re.IGNORECASE | re.MULTILINE), (
            f"curated DDL is missing typed column {column}"
        )
    assert "customer key" in ddl.lower(), "record_id comment must say it is not a customer key"


def test_curated_insert_fixes_the_source_typo_auditable() -> None:
    insert = read(SQL_DIR / "03_curated_campaign_customers_insert.sql")
    code, _ = split_comments(insert)
    assert "'Surburban'" in code, "the source typo must be named explicitly"
    assert "'Suburban'" in code, "the corrected label must be written"
    assert "zip_code_raw" in code, "the raw value must be preserved for audit"
    assert re.search(r"INSERT INTO\s+`campaign_incrementality\.curated_campaign_customers`", code, re.IGNORECASE)
    assert re.search(r"CURRENT_TIMESTAMP\(\)\s+AS\s+data_loaded_utc", code, re.IGNORECASE)
    for threshold in ["100.0", "200.0", "350.0", "500.0", "750.0", "1000.0"]:
        assert threshold in code, f"history band edge {threshold} is missing"


def test_data_quality_checks_return_the_contract_columns() -> None:
    path = SQL_DIR / "04_data_quality_checks.sql"
    code, _ = split_comments(read(path))
    parsed = statements(path)
    assert len(parsed) == 1, "stage 04 must be a single query"
    for column in ["check_name", "passed", "detail", "metric_value"]:
        assert re.search(rf"\bAS\s+{column}\b", code, re.IGNORECASE), f"missing output column {column}"
    assert code.lower().count("union all") >= 10, "expected one UNION ALL branch per gate"
    for gate in [
        "volume.row_count", "completeness.no_nulls", "uniqueness.record_id_no_duplicates",
        "binary.visit_in_zero_one", "binary.conversion_in_zero_one", "consistency.spend_non_negative",
        "consistency.spend_implies_conversion", "consistency.conversion_implies_visit",
        "consistency.history_band_matches_segment", "design.srm_arm_share_within_1pct",
        "treatment.arm_counts_match_published",
    ]:
        assert gate in code, f"missing check {gate}"


def test_incrementality_uses_the_published_adjusted_evidence() -> None:
    text = read(SQL_DIR / '07_views_dashboard_campaign_incrementality.sql')
    assert 'evidence_campaign_incrementality_effects' in text
    assert 'ci_low_fwer' in text and 'p_holm' in text
    assert '1.96' not in split_comments(text)[0]


def test_uplift_layer_is_aggregate_only_and_defines_scores() -> None:
    view = read(SQL_DIR / "08_views_dashboard_uplift.sql")
    code, _ = split_comments(view)
    assert re.search(r"GROUP BY comparison, split, decile", code, re.IGNORECASE)
    assert "cumulative_qini_gain_conversions" in code
    assert "cumulative_uplift" in code
    assert "control_scaling_factor" in code
    scores = read(SQL_DIR / "06_uplift_customer_scores_ddl.sql")
    for column in ["record_id", "comparison", "uplift_score", "response_score", "decile", "split"]:
        assert re.search(rf"^\s+{column}\s+(INT64|FLOAT64|STRING)\b", scores, re.IGNORECASE | re.MULTILINE)
    assert "PII" in view, "the view must state that per-customer rows stay out of the dashboard"


def test_economics_view_is_modelled_not_realised() -> None:
    text = read(SQL_DIR / "09_views_dashboard_economics.sql")
    code, comments = split_comments(text)
    for column in [
        "modelled_incremental_margin", "modelled_campaign_cost", "modelled_net_contribution_scenario",
        "modelled_emails_sent", "modelled_incremental_conversions",
    ]:
        assert column in code, f"missing modelled column {column}"
    assert re.search(r"UNNEST\(\[0\.05, 0\.10, 0\.20, 0\.30, 0\.50, 0\.75, 1\.00\]\)", code), "budget levels differ from config"
    assert "reference_scenarios" in code
    assert "not realised roi" in f"{code}\n{comments}".lower()


def test_readme_states_deployment_limits() -> None:
    text = read(README).lower()
    assert 'not executed against a live bigquery project' in text
    assert 'deploy_bigquery.py' in text and '--execute' in text
    assert 'sqlglot' in text


def test_no_unsupported_overwrite_or_regenerated_record_ids() -> None:
    for path in sql_files():
        code, _ = split_comments(read(path))
        assert not re.search(r'INSERT\s+OVERWRITE', code, re.I)
    curated = split_comments(read(SQL_DIR / '03_curated_campaign_customers_insert.sql'))[0]
    assert 'ROW_NUMBER' not in curated
    assert 'landing_hillstrom_indexed' in curated
    economics = split_comments(read(SQL_DIR / '09_views_dashboard_economics.sql'))[0]
    assert 'treated_targeted_customers AS modelled_emails_sent' in economics
    assert 'ROUND_HALF_EVEN' in economics
    assert 'rank_position' in economics
