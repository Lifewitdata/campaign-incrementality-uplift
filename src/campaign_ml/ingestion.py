"""Download the Hillstrom experiment file and record its checksum.

The source is a static 2008 publication, so the file is cached locally and
verified by SHA-256 on every run. Any drift from the pinned checksum is an
error rather than a silent re-download: analysis numbers in the README are
tied to a specific byte-for-byte input.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import requests

DEFAULT_URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
USER_AGENT = "campaign-incrementality-uplift/0.1 (portfolio project; data download)"
CHUNK_SIZE = 1 << 16


class ChecksumMismatch(RuntimeError):
    """The downloaded file does not match the pinned checksum."""


@dataclass(frozen=True)
class IngestResult:
    path: Path
    sha256: str
    bytes: int
    from_cache: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "bytes": self.bytes,
            "from_cache": self.from_cache,
        }


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str = DEFAULT_URL, dest: Path | str = "data/raw/hillstrom.csv",
             timeout: int = 120) -> Path:
    """Stream the CSV to ``dest`` atomically so a failed run leaves no partial file."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as raw, requests.get(
            url, stream=True, timeout=timeout, headers={"User-Agent": USER_AGENT}
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    raw.write(chunk)
        shutil.move(str(tmp_path), str(dest))
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return dest


def ingest(url: str = DEFAULT_URL, dest: Path | str = "data/raw/hillstrom.csv",
           expected_sha256: str | None = None, force: bool = False) -> IngestResult:
    """Return the local dataset file, downloading it only when needed.

    ``expected_sha256`` is the pin from ``configs/experiment.yaml``. When the
    cached file is present and matches, nothing is fetched from the network.
    """
    dest = Path(dest)
    from_cache = False
    if dest.exists() and not force:
        digest = sha256_of(dest)
        from_cache = expected_sha256 is None or digest == expected_sha256
    if not from_cache:
        download(url, dest)
        digest = sha256_of(dest)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ChecksumMismatch(
                f"{dest} sha256 {digest} does not match pinned {expected_sha256}. "
                "Refusing to continue: the published results are tied to the pinned file."
            )
    return IngestResult(dest, digest, dest.stat().st_size, from_cache=from_cache)


def write_checksum_file(path: Path | str, digest: str | None = None) -> Path:
    """Persist ``<file>.sha256`` next to the data so reviewers can re-verify it."""
    path = Path(path)
    digest = digest or sha256_of(path)
    out = path.with_name(path.name + ".sha256")
    out.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return out


def verify_checksum_file(path: Path | str) -> bool:
    """Compare the file against its sidecar checksum; True when they agree."""
    path = Path(path)
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.exists():
        return False
    expected = sidecar.read_text(encoding="utf-8").split()[0].strip()
    return sha256_of(path) == expected
