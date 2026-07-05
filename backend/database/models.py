"""SQLAlchemy ORM Database models.

Defines the database schema mapping to tables: Users, Predictions, Simulations, and Logs.
"""

from __future__ import annotations

import datetime
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from backend.database.database import Base


class User(Base):
    """User accounts table for authentication and access control."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    predictions = relationship("PredictionHistory", back_populates="user", cascade="all, delete-orphan")
    simulations = relationship("SimulationHistory", back_populates="user", cascade="all, delete-orphan")


class PredictionHistory(Base):
    """Stores logs of individual match predictions run by users."""

    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Nullable if anonymous prediction is allowed
    
    # Inputs
    home_team = Column(String(50), nullable=False)
    away_team = Column(String(50), nullable=False)
    tournament = Column(String(100), nullable=False)
    venue = Column(String(20), nullable=False)
    match_date = Column(String(20), nullable=True)

    # Outputs
    predicted_winner = Column(String(50), nullable=False)
    prob_home_win = Column(Float, nullable=False)
    prob_draw = Column(Float, nullable=False)
    prob_away_win = Column(Float, nullable=False)
    confidence_score = Column(Float, nullable=False)
    
    # Complex metadata (SHAP values, feature importances, expected goals)
    feature_importance = Column(JSON, nullable=True)
    shap_explanation = Column(JSON, nullable=True)
    expected_goals = Column(JSON, nullable=True)
    
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="predictions")


class SimulationHistory(Base):
    """Stores summaries of Monte Carlo tournament simulations."""

    __tablename__ = "simulation_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    run_count = Column(Integer, nullable=False)
    
    # JSON containing the champion probabilities, Semifinal probabilities, etc.
    champion_odds = Column(JSON, nullable=False)
    stage_probabilities = Column(JSON, nullable=True)
    
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="simulations")


class ApplicationLog(Base):
    """Stores application request and error logs directly in DB."""

    __tablename__ = "application_logs"

    id = Column(Integer, primary_key=True, index=True)
    path = Column(String(255), nullable=False)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer, nullable=False)
    process_time_ms = Column(Float, nullable=False)
    error_message = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
