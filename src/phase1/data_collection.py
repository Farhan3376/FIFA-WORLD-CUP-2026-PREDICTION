"""Phase 1 - Step 1: Data collection.

Downloads the raw football datasets defined in ``config/config.yaml`` into
``data/raw/``. The pipeline is built around the open *International football
results from 1872* dataset (results, shootouts and goalscorers), plus an
optional historical FIFA-ranking file.

Design notes
------------
* **Resilient downloads** - each source is retried with exponential backoff;
  transient network failures do not crash the run.
* **Caching** - an existing, non-empty file in ``data/raw/`` is reused unless
  ``force=True`` is passed, so re-runs are cheap and offline-friendly.
* **Synthetic fallback** - if a *required* source cannot be obtained and no
  cached copy exists, a small synthetic match table is generated so the rest
  of the pipeline can still be demonstrated end-to-end (controlled by
  ``download.allow_synthetic_fallback`` in the config).
* **Extensible** - player statistics (``goalscorers.csv``) are collected now
  even though they are unused in Phase 1, so later phases can consume them
  without changing this module's architecture.

Run directly with::

    python -m src.data_collection
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from src.config import (
    RAW_DIR,
    Settings,
    ensure_directories,
    load_settings,
    set_global_seed,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DataSource:
    """A single downloadable dataset described in the configuration.

    Attributes:
        key: Short identifier (e.g. ``"results"``).
        url: Remote location of the file. May be empty to skip download.
        filename: Destination filename inside ``data/raw/``.
        required: If ``True``, failure to obtain the file aborts collection.
        description: Human-readable summary of the file's contents.
    """

    key: str
    url: str
    filename: str
    required: bool
    description: str = ""

    @property
    def destination(self) -> Path:
        """Absolute path the file is written to."""
        return RAW_DIR / self.filename


def _parse_sources(settings: Settings) -> List[DataSource]:
    """Flatten ``data_sources`` and ``rankings`` config blocks into objects.

    Args:
        settings: Parsed project settings.

    Returns:
        A list of :class:`DataSource` instances, ranking sources last.
    """
    sources: List[DataSource] = []

    for block in (settings.data_sources, settings.rankings):
        for key, spec in block.items():
            sources.append(
                DataSource(
                    key=key,
                    url=str(spec.get("url", "")).strip(),
                    filename=str(spec["filename"]),
                    required=bool(spec.get("required", False)),
                    description=str(spec.get("description", "")),
                )
            )
    return sources


def _download_file(url: str, destination: Path, download_cfg: Dict) -> None:
    """Stream a URL to disk with retries and exponential backoff.

    Args:
        url: The remote file URL.
        destination: Local path to write to.
        download_cfg: The ``download`` section of the configuration.

    Raises:
        requests.RequestException: If all retry attempts fail.
    """
    timeout = int(download_cfg.get("timeout_seconds", 60))
    max_retries = int(download_cfg.get("max_retries", 3))
    backoff = float(download_cfg.get("retry_backoff_seconds", 3))
    chunk_size = int(download_cfg.get("chunk_size_bytes", 32768))
    headers = {"User-Agent": str(download_cfg.get("user_agent", "pipeline/1.0"))}

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Downloading %s (attempt %d/%d)", url, attempt, max_retries)
            with requests.get(
                url, stream=True, timeout=timeout, headers=headers
            ) as response:
                response.raise_for_status()
                tmp_path = destination.with_suffix(destination.suffix + ".part")
                with tmp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            handle.write(chunk)
                tmp_path.replace(destination)  # atomic move on success
            logger.info("Saved -> %s", destination)
            return
        except requests.RequestException as exc:
            last_error = exc
            logger.warning("Download failed (attempt %d): %s", attempt, exc)
            if attempt < max_retries:
                sleep_for = backoff * attempt
                logger.info("Retrying in %.1fs ...", sleep_for)
                time.sleep(sleep_for)

    raise requests.RequestException(
        f"Exhausted {max_retries} attempts for {url}: {last_error}"
    )


def _is_cached(path: Path) -> bool:
    """Return ``True`` if a non-empty file already exists at ``path``."""
    return path.is_file() and path.stat().st_size > 0


def _generate_synthetic_results(n_rows: int) -> pd.DataFrame:
    """Create a small, schema-correct synthetic match table.

    Used only as a last-resort fallback so the pipeline remains runnable
    without network access. The data is *not* realistic and must never be
    used for genuine modelling - it merely exercises downstream code paths.

    Args:
        n_rows: Number of synthetic matches to generate.

    Returns:
        A DataFrame mirroring the columns of the real ``results.csv``.
    """
    import numpy as np

    teams = [
        "Brazil", "Argentina", "France", "Germany", "Spain", "England",
        "Italy", "Netherlands", "Portugal", "Belgium", "Croatia", "Uruguay",
        "Mexico", "USA", "Japan", "Morocco",
    ]
    tournaments = ["Friendly", "FIFA World Cup", "FIFA World Cup qualification",
                   "Copa America", "UEFA Euro"]

    rng = np.random.default_rng()
    dates = pd.to_datetime("1990-01-01") + pd.to_timedelta(
        rng.integers(0, 365 * 35, size=n_rows), unit="D"
    )
    home = rng.choice(teams, size=n_rows)
    away = rng.choice(teams, size=n_rows)
    # Avoid a team playing itself.
    mask = home == away
    away[mask] = rng.choice(teams, size=int(mask.sum()))

    frame = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "home_team": home,
            "away_team": away,
            "home_score": rng.integers(0, 6, size=n_rows),
            "away_score": rng.integers(0, 6, size=n_rows),
            "tournament": rng.choice(tournaments, size=n_rows),
            "city": "Unknown",
            "country": rng.choice(teams, size=n_rows),
            "neutral": rng.choice([True, False], size=n_rows),
        }
    )
    return frame


def collect_source(source: DataSource, download_cfg: Dict, force: bool) -> bool:
    """Obtain a single data source (cache-aware, with fallback handling).

    Args:
        source: The data source to collect.
        download_cfg: The ``download`` configuration section.
        force: If ``True``, re-download even when a cached copy exists.

    Returns:
        ``True`` if the file is available on disk afterwards, else ``False``.
    """
    if not force and _is_cached(source.destination):
        logger.info("Cached, skipping: %s", source.destination.name)
        return True

    if not source.url:
        logger.info(
            "No URL configured for '%s' - skipping (optional source).", source.key
        )
        return False

    try:
        _download_file(source.url, source.destination, download_cfg)
        return True
    except requests.RequestException as exc:
        logger.error("Could not download '%s': %s", source.key, exc)

    # Fallback handling for a required source we still don't have.
    if source.required and not _is_cached(source.destination):
        if download_cfg.get("allow_synthetic_fallback", False):
            n_rows = int(download_cfg.get("synthetic_rows", 1000))
            logger.warning(
                "Generating %d synthetic rows for required source '%s'.",
                n_rows,
                source.key,
            )
            synthetic = _generate_synthetic_results(n_rows)
            synthetic.to_csv(source.destination, index=False)
            logger.info("Synthetic data written -> %s", source.destination)
            return True
        logger.critical(
            "Required source '%s' unavailable and synthetic fallback disabled.",
            source.key,
        )
    return _is_cached(source.destination)


def collect_all(
    settings: Optional[Settings] = None, force: bool = False
) -> Dict[str, bool]:
    """Collect every configured data source into ``data/raw/``.

    Args:
        settings: Project settings; loaded from default config if omitted.
        force: If ``True``, re-download all sources even when cached.

    Returns:
        Mapping of source key to a boolean indicating availability on disk.

    Raises:
        RuntimeError: If any *required* source could not be obtained.
    """
    settings = settings or load_settings()
    ensure_directories()
    set_global_seed(settings.random_seed)

    download_cfg = settings.download
    sources = _parse_sources(settings)
    logger.info("Starting data collection for %d source(s).", len(sources))

    status: Dict[str, bool] = {}
    failures: List[str] = []

    for source in sources:
        available = collect_source(source, download_cfg, force)
        status[source.key] = available
        if source.required and not available:
            failures.append(source.key)

    if failures:
        raise RuntimeError(
            f"Required data source(s) could not be obtained: {', '.join(failures)}"
        )

    obtained = sum(status.values())
    logger.info("Data collection complete: %d/%d source(s) available.",
                obtained, len(sources))
    return status


def main() -> None:
    """CLI entry point: collect all data sources using the default config."""
    try:
        results = collect_all()
    except Exception as exc:  # noqa: BLE001 - top-level guard logs and re-raises
        logger.exception("Data collection failed: %s", exc)
        raise

    print("\nData collection summary")
    print("-" * 40)
    for key, available in results.items():
        marker = "OK " if available else "-- "
        print(f"  [{marker}] {key}")
    print(f"\nRaw files are in: {RAW_DIR}")


if __name__ == "__main__":
    main()
