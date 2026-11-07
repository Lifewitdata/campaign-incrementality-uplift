"""Thin MLflow wrapper: one run per learner, parameters and metrics only.

MLflow is an optional extra in ``pyproject.toml``. When it is not installed the
tracker degrades to a no-op so the pipeline and its tests still run offline;
when it is installed, every learner gets a run under the experiment named in
``configs/experiment.yaml`` and the metrics land in ``outputs/tables`` anyway,
so the repository's published numbers never depend on a tracking server.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from campaign_ml.config import PROJECT_ROOT

try:  # optional extra: .[tracking]
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra
    mlflow = None
    MLFLOW_AVAILABLE = False


def resolve_tracking_uri(uri: str) -> str:
    """Make a relative SQLite path repo-absolute.

    MLflow 3.x put the plain file store into maintenance mode, so runs land in a
    SQLite database. A relative URI would depend on whichever directory the
    script was launched from; pinning it to the repository root keeps the run
    history where a reviewer cloning this repo expects to find it.
    """
    prefix = "sqlite:///"
    if not uri.startswith(prefix):
        return uri
    database = uri[len(prefix):]
    if Path(database).is_absolute():
        return uri
    return prefix + (PROJECT_ROOT / database).as_posix()


@dataclass
class Tracker:
    experiment_name: str
    tracking_uri: str
    enabled: bool = True
    run_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: dict, force_disabled: bool = False) -> "Tracker":
        ml_cfg = config["mlflow"]
        return cls(
            experiment_name=ml_cfg["experiment_name"],
            tracking_uri=resolve_tracking_uri(ml_cfg["tracking_uri"]),
            enabled=MLFLOW_AVAILABLE and not force_disabled,
        )

    def start(self, run_name: str, params: Mapping[str, Any]) -> "RunHandle":
        if not self.enabled:
            return RunHandle(self, run_name, disabled=True)
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        active = mlflow.start_run(run_name=run_name)
        mlflow.log_params(_flatten(params))
        return RunHandle(self, run_name, disabled=False, run_id=active.info.run_id)


@dataclass
class RunHandle:
    tracker: Tracker
    run_name: str
    disabled: bool = True
    run_id: str | None = None

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        if self.disabled:
            return
        clean = {k: float(v) for k, v in metrics.items() if v is not None}
        mlflow.log_metrics(clean, step=step)

    def log_tags(self, tags: Mapping[str, str]) -> None:
        if self.disabled:
            return
        mlflow.set_tags({k: str(v) for k, v in tags.items()})

    def log_artifact(self, path: str) -> None:
        if self.disabled:
            return
        mlflow.log_artifact(path)

    def finish(self, status: str = "FINISHED") -> None:
        if self.disabled:
            return
        mlflow.end_run(status=status)
        self.tracker.run_ids.append(self.run_id or "")


def _flatten(params: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """MLflow wants flat string parameters, not nested dicts."""
    flat: dict[str, str] = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(_flatten(value, prefix=f"{name}."))
        elif isinstance(value, (list, tuple)):
            flat[name] = ",".join(str(item) for item in value)
        else:
            flat[name] = str(value)
    return flat


def recent_runs(tracker: Tracker, limit: int = 10) -> list[dict[str, Any]]:
    """Run table for the README - only meaningful when MLflow was used."""
    if not tracker.enabled:
        return []
    mlflow.set_tracking_uri(tracker.tracking_uri)
    experiment = mlflow.get_experiment_by_name(tracker.experiment_name)
    if experiment is None:
        return []
    found = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=limit,
        order_by=["attributes.start_time DESC"],
    )
    metric_columns = [c for c in found.columns if c.startswith("metrics.")]
    rows: list[dict[str, Any]] = []
    for _, row in found.iterrows():
        entry = {
            "run_id": str(row["run_id"])[:12],
            "run_name": row.get("tags.mlflow.runName"),
            "status": row["status"],
        }
        # search_runs returns a wide frame, so a metric one learner does not
        # have shows up as NaN here; the run table only carries what was logged.
        entry.update({
            column[len("metrics."):]: float(row[column])
            for column in metric_columns if pd.notna(row[column])
        })
        rows.append(entry)
    return rows
