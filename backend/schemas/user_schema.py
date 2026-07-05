"""Pydantic schemas for User authentication.

Provides registration, login, profile, and JWT token schema validation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class UserBase(BaseModel):
    """Common properties shared across user schemas."""
    username: str = Field(..., min_length=3, max_length=50, description="Unique username")
    email: EmailStr = Field(..., description="Valid email address")


class UserCreate(UserBase):
    """Request schema to register a new user."""
    password: str = Field(..., min_length=6, max_length=100, description="Secret password (min 6 characters)")


class UserLogin(BaseModel):
    """Request schema to authenticate a user."""
    username: str = Field(..., description="Registered username")
    password: str = Field(..., description="Secret password")


class UserResponse(UserBase):
    """Response schema containing user profile data."""
    id: int
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    """Response schema containing access tokens."""
    access_token: str
    token_type: str


class TokenData(BaseModel):
    """Token payload contents."""
    username: Optional[str] = None
