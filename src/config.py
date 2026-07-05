"""Central configuration and project-wide constants.

This module is the single source of truth for:

* **Paths** - every directory is resolved relative to the project root with
  :mod:`pathlib`, so nothing is hard-coded and the project is portable.
* **Settings** - parsed from ``config/config.yaml`` into a typed object.
* **Constants** - reproducibility seed, canonical column names and the
  validation thresholds reused across the pipeline.

All other modules import from here rather than redefining paths or magic
numbers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml

# ---------------------------------------------------------------------------
# Project paths (resolved from this file's location: src/config.py -> root)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

CONFIG_DIR: Path = PROJECT_ROOT / "config"
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
EDA_REPORTS_DIR: Path = REPORTS_DIR / "eda"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
MODELS_DIR: Path = PROJECT_ROOT / "models"
TRAINED_MODELS_DIR: Path = MODELS_DIR / "trained"
METRICS_DIR: Path = MODELS_DIR / "metrics"
FEATURE_IMPORTANCE_DIR: Path = MODELS_DIR / "feature_importance"
ML_REPORTS_DIR: Path = REPORTS_DIR / "ml"

DEFAULT_CONFIG_PATH: Path = CONFIG_DIR / "config.yaml"

# Directories that must exist for the pipeline to run.
_REQUIRED_DIRS = (
    RAW_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    EDA_REPORTS_DIR,
    LOGS_DIR,
    MODELS_DIR,
    TRAINED_MODELS_DIR,
    METRICS_DIR,
    FEATURE_IMPORTANCE_DIR,
    ML_REPORTS_DIR,
)

# ---------------------------------------------------------------------------
# Canonical schema - the column names the pipeline standardizes everything to.
# Raw sources may use different names; preprocessing renames them to these.
# ---------------------------------------------------------------------------
COL_DATE = "date"
COL_HOME_TEAM = "home_team"
COL_AWAY_TEAM = "away_team"
COL_HOME_GOALS = "home_goals"
COL_AWAY_GOALS = "away_goals"
COL_TOURNAMENT = "tournament"
COL_CITY = "city"
COL_COUNTRY = "country"
COL_NEUTRAL = "neutral"
COL_WINNER = "winner"
COL_RESULT = "result"  # target: 0 home win, 1 draw, 2 away win

# Target-variable encoding (kept here so model code reuses the same mapping).
RESULT_HOME_WIN = 0
RESULT_DRAW = 1
RESULT_AWAY_WIN = 2
RESULT_LABELS: Dict[int, str] = {
    RESULT_HOME_WIN: "Home Win",
    RESULT_DRAW: "Draw",
    RESULT_AWAY_WIN: "Away Win",
}

# ---------------------------------------------------------------------------
# Validation thresholds (reused by preprocessing and validation).
# ---------------------------------------------------------------------------
MIN_MATCH_YEAR = 1872          # first recorded international match
MAX_GOALS_PER_TEAM = 31        # historical max is 31 (AUS 31-0 ASA, 2001)
RECENT_FORM_WINDOW = 5         # matches used for "recent form" features


@dataclass
class Settings:
    """Typed view over ``config.yaml`` plus the reproducibility seed."""

    raw: Dict[str, Any] = field(repr=False)

    @property
    def random_seed(self) -> int:
        return int(self.raw["project"]["random_seed"])

    @property
    def project_name(self) -> str:
        return str(self.raw["project"]["name"])

    @property
    def data_sources(self) -> Dict[str, Any]:
        return dict(self.raw.get("data_sources", {}))

    @property
    def rankings(self) -> Dict[str, Any]:
        return dict(self.raw.get("rankings", {}))

    @property
    def download(self) -> Dict[str, Any]:
        return dict(self.raw.get("download", {}))

    @property
    def outputs(self) -> Dict[str, Any]:
        return dict(self.raw.get("outputs", {}))

    @property
    def logging(self) -> Dict[str, Any]:
        return dict(self.raw.get("logging", {}))


def load_settings(config_path: Path | str = DEFAULT_CONFIG_PATH) -> Settings:
    """Load and parse the YAML configuration file.

    Args:
        config_path: Path to the YAML config. Defaults to ``config/config.yaml``.

    Returns:
        A :class:`Settings` instance wrapping the parsed configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the file is empty or not valid YAML.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Configuration in {path} must be a mapping.")

    return Settings(raw=data)


def ensure_directories() -> None:
    """Create all required project directories if they do not yet exist."""
    for directory in _REQUIRED_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    """Seed Python and NumPy RNGs for reproducible runs.

    Args:
        seed: The integer seed to apply.
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dependency anyway
        pass
