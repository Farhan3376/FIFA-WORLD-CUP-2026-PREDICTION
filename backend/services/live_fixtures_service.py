"""Live fixtures & scores integration.

Wraps the public worldcup2026 API (github.com/rezarahiminia/worldcup2026,
hosted at worldcup26.ir) to expose real tournament fixtures and in-play
scores. This is a third-party, unofficial, best-effort data source with no
SLA -- callers must tolerate timeouts/outages and degrade gracefully.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

from backend.config import settings

logger = logging.getLogger("backend")


class LiveFixturesService:
    """Fetches and normalizes fixture/live-score data from the worldcup2026 API."""

    @staticmethod
    def _get(path: str) -> Optional[Dict[str, Any]]:
        """Perform a GET request against the live fixtures API, returning None on failure."""
        url = f"{settings.LIVE_FIXTURES_API_URL.rstrip('/')}/{path.lstrip('/')}"
        try:
            response = requests.get(url, timeout=settings.LIVE_FIXTURES_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.warning("Live fixtures API request failed (%s): %s", url, e)
            return None

    @classmethod
    def get_fixtures(cls) -> List[Dict[str, Any]]:
        """Fetch and normalize all 104 tournament fixtures.

        Returns an empty list if the upstream API is unreachable, rather than raising,
        so calling routes can degrade to "live data unavailable" instead of failing.
        """
        data = cls._get("/get/games")
        if not data or "games" not in data:
            return []

        fixtures = []
        for g in data["games"]:
            try:
                fixtures.append(cls._normalize_fixture(g))
            except (KeyError, ValueError) as e:
                logger.debug("Skipping malformed fixture record: %s", e)
                continue
        return fixtures

    @classmethod
    def get_live_fixtures(cls) -> List[Dict[str, Any]]:
        """Return only fixtures currently in progress."""
        return [f for f in cls.get_fixtures() if f["status"] == "live"]

    @classmethod
    def get_qualified_teams(cls) -> List[str]:
        """Fetch the 48 real FIFA World Cup 2026 qualified team names.

        Distinct from analytics team names (which cover ~336 historical
        entities in the ELO database) -- this is the actual tournament
        field, sourced from the live fixtures API's /get/teams endpoint.
        Returns an empty list if the upstream API is unreachable.
        """
        data = cls._get("/get/teams")
        if not data or "teams" not in data:
            return []
        names = [t.get("name_en") for t in data["teams"] if t.get("name_en")]
        return sorted(names)

    @classmethod
    def get_fixture_by_teams(cls, home_team: str, away_team: str) -> Optional[Dict[str, Any]]:
        """Find a fixture matching the given home/away team names (case-insensitive)."""
        home_lower, away_lower = home_team.strip().lower(), away_team.strip().lower()
        for f in cls.get_fixtures():
            if f["home_team"].lower() == home_lower and f["away_team"].lower() == away_lower:
                return f
        return None

    @staticmethod
    def _normalize_fixture(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map the upstream game record onto a stable internal shape."""
        status_map = {
            "notstarted": "scheduled",
            "live": "live",
            "finished": "finished",
            "Finished": "finished",
        }
        status = status_map.get(raw.get("time_elapsed", ""), "scheduled")

        home_score = raw.get("home_score")
        away_score = raw.get("away_score")

        return {
            "fixture_id": raw.get("id"),
            "home_team": raw.get("home_team_name_en") or raw.get("home_team_label", "TBD"),
            "away_team": raw.get("away_team_name_en") or raw.get("away_team_label", "TBD"),
            "home_score": int(home_score) if home_score not in (None, "null") else None,
            "away_score": int(away_score) if away_score not in (None, "null") else None,
            "status": status,
            "group": raw.get("group"),
            "matchday": raw.get("matchday"),
            "kickoff_local": raw.get("local_date"),
            "stage": raw.get("type"),
        }
