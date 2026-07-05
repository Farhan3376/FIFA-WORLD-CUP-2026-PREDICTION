"""Pydantic configurations package init.

Exposes the settings singleton.
"""

from __future__ import annotations

from backend.config.config import settings

__all__ = ["settings"]
