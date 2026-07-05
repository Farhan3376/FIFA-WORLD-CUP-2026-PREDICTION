"""Database connection and session management.

Configures the SQLAlchemy engine and provides session dependencies for FastAPI.
"""

from __future__ import annotations

from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from backend.config.config import settings

# Determine if we are using SQLite (requires custom connection arguments)
is_sqlite = settings.DATABASE_URL.startswith("sqlite")
connect_args = {"check_same_thread": False} if is_sqlite else {}

# Create engine
engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,  # Test connection before using it
)

# Configure Session class
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# Declarative base model class
Base = declarative_base()


def get_db() -> Generator:
    """Dependency injection generator to yield a database session.

    Ensures the session is closed automatically after requests are completed.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
