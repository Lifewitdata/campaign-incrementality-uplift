"""Shared fixtures: a synthetic dataset with the published schema and design.

Nothing in ``tests/`` reads ``data/raw/hillstrom.csv``. The synthetic frame has
the same columns, the same three-arm randomised structure and a *known* effect,
so a test can assert a number instead of asserting that the code reproduced
itself. Tests that must run on the real 64k rows are marked ``slow`` and skip
when the file is absent, which keeps a fresh clone green before ingestion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from campaign_ml.validation import (
    ARMS,
    CHANNELS,
    HISTORY_SEGMENTS,
    SOURCE_COLUMNS,
    ZIP_CODE_LABELS,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = REPO_ROOT / "data" / "raw" / "hillstrom.csv"

ARM_LABELS = list(ZIP_CODE_LABELS.values())


def synthetic_experiment(n_per_arm: int = 3000, seed: int = 20261107,
                         uplift: float = 0.02) -> pd.DataFrame:
    """A three-arm randomised frame whose true conversion uplift is ``uplift``.

    The control arm converts at 6%; each emailed arm converts at 6% + uplift.
    ``spend`` is positive only for converters, so it inherits the same heavy
    skew as the published data without needing a distribution nobody recognises.
    """
    rng = np.random.default_rng(seed)
    n = n_per_arm * len(ARMS)
    arm = np.repeat(np.array(ARMS, dtype=object), n_per_arm)
    emailed = arm != ARMS[0]

    base = 0.06
    prob = np.where(emailed, base + uplift, base)
    conversion = (rng.random(n) < prob).astype(int)
    visiting = np.maximum(conversion, (rng.random(n) < np.where(emailed, 0.30, 0.15)))
    spend = np.where(conversion == 1, np.round(rng.lognormal(3.4, 1.0, n), 2), 0.0)

    history = np.round(rng.gamma(2.0, 120.0, n), 2)
    # history_segment must stay an exact function of history or the consistency
    # gate fails, so it is derived rather than drawn.
    bins = [0.0, 100.0, 200.0, 350.0, 500.0, 750.0, 1000.0, np.inf]
    history_segment = np.digitize(history, bins[1:-1], right=False)

    frame = pd.DataFrame({
        "recency": rng.integers(1, 13, n),
        "history_segment": [HISTORY_SEGMENTS[i] for i in history_segment],
        "history": history,
        "mens": rng.integers(0, 2, n),
        "womens": rng.integers(0, 2, n),
        "zip_code": rng.choice(list(ZIP_CODE_LABELS), n),
        "newbie": rng.integers(0, 2, n),
        "channel": rng.choice(CHANNELS, n),
        "segment": arm,
        "visit": visiting.astype(int),
        "conversion": conversion,
        "spend": spend,
    })
    return frame[SOURCE_COLUMNS].sample(frac=1.0, random_state=seed).reset_index(drop=True)


@pytest.fixture(scope="session")
def synth_raw() -> pd.DataFrame:
    return synthetic_experiment()


@pytest.fixture(scope="session")
def synth_clean(synth_raw) -> pd.DataFrame:
    from campaign_ml.validation import prepare_clean_table

    return prepare_clean_table(synth_raw)


@pytest.fixture(scope="session")
def real_data_available() -> bool:
    return RAW_CSV.exists()


@pytest.fixture(scope="session")
def real_clean() -> pd.DataFrame:
    if not RAW_CSV.exists():
        pytest.skip("data/raw/hillstrom.csv not present; run scripts/ingest_data.py")
    from campaign_ml.ingestion import verify_checksum_file
    from campaign_ml.validation import load_raw, prepare_clean_table

    verify_checksum_file(RAW_CSV)
    return prepare_clean_table(load_raw(RAW_CSV))
