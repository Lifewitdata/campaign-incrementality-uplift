"""Phase 1 entrypoint: download, validate and publish the clean campaign table.

    python scripts/ingest_data.py

Writes:
  data/raw/hillstrom.csv                  pinned source file (git-ignored)
  data/raw/hillstrom.csv.sha256           checksum sidecar
  data/processed/campaign_customers.csv   warehouse-ready table
  outputs/tables/data_quality_checks.csv  every gate, pass or fail
  outputs/tables/data_lineage.json        source, checksum, row count, versions
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from campaign_ml.config import PROJECT_ROOT, load_config, resolve
from campaign_ml.ingestion import ingest, write_checksum_file
from campaign_ml.validation import (
    DataQualityError,
    load_raw,
    prepare_clean_table,
    run_all_checks,
)


def main() -> int:
    config = load_config()
    data_cfg = config["data"]
    raw_path = resolve(data_cfg["raw_path"])

    result = ingest(
        url=data_cfg["source_url"],
        dest=raw_path,
        expected_sha256=data_cfg.get("expected_sha256"),
    )
    write_checksum_file(result.path, result.sha256)
    print(f"[1/4] source {'cached' if result.from_cache else 'downloaded'}: "
          f"{result.bytes:,} bytes, sha256 {result.sha256[:12]}...")

    raw = load_raw(raw_path)
    report = run_all_checks(
        raw,
        expected_rows=int(data_cfg["expected_rows"]),
        expected_share=float(config["experiment"]["srm"]["expected_shares"]),
    )
    out_dir = resolve("outputs/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    report.to_frame().to_csv(out_dir / "data_quality_checks.csv", index=False)
    print(f"[2/4] {len(report.checks)} quality gates: "
          f"{sum(c.passed for c in report.checks)} passed, {len(report.failed)} failed")

    report.raise_if_failed()

    clean = prepare_clean_table(raw)
    clean_path = resolve("data/processed/campaign_customers.csv")
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(clean_path, index=False)
    print(f"[3/4] clean table: {len(clean):,} rows x {len(clean.columns)} cols -> {clean_path}")

    lineage = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_name": data_cfg["source_name"],
        "source_url": data_cfg["source_url"],
        "attribution": data_cfg["attribution"],
        "sha256": result.sha256,
        "bytes": result.bytes,
        "rows": int(len(clean)),
        "columns": list(clean.columns),
        "arm_counts": raw["segment"].value_counts().sort_index().to_dict(),
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
        },
    }
    resolve("outputs/tables/data_lineage.json").write_text(
        json.dumps(lineage, indent=2), encoding="utf-8"
    )
    print(f"[4/4] lineage written -> {PROJECT_ROOT / 'outputs/tables/data_lineage.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
