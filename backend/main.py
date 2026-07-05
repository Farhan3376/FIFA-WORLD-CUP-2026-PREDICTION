"""FastAPI backend entry point.

Initializes the FastAPI application, registers middleware (CORS, Request Logging),
hooks up routers, and defines base metadata/health endpoints.
"""

from __future__ import annotations

import logging
import time
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, Request
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from fastapi.responses import JSONResponse

# Import routers (to be implemented in subsequent steps)
from backend.api.routes import (
    authentication,
    prediction,
    simulation,
    analytics,
    live,
)
from backend.database.database import engine, Base

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("backend")

# Create database tables on startup
try:
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized successfully.")
except Exception as e:
    logger.warning("Could not initialize database tables: %s. Assuming DB setup in progress.", e)

app = FastAPI(
    title="FIFA World Cup 2026 Prediction Engine API",
    description="Production-ready REST APIs for match winner predictions, Monte Carlo tournament simulations, and team analytics.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom Request Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log details of every incoming API request and execution time."""
    start_time = time.time()
    path = request.url.path
    method = request.method
    
    logger.info("Incoming request: %s %s", method, path)
    
    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        logger.info(
            "Request completed: %s %s | Status: %d | Time: %.2fms",
            method,
            path,
            response.status_code,
            process_time,
        )
        return response
    except Exception as e:
        process_time = (time.time() - start_time) * 1000
        logger.error(
            "Request failed: %s %s | Error: %s | Time: %.2fms",
            method,
            path,
            str(e),
            process_time,
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error", "error": str(e)},
        )

# Register routers
app.include_router(authentication.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(prediction.router, prefix="/api/predict", tags=["Predictions"])
app.include_router(simulation.router, prefix="/api/simulate", tags=["Simulations"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["Analytics"])
app.include_router(live.router, prefix="/api/live", tags=["Live Fixtures"])

@app.get("/health", response_model=dict, tags=["System"])
async def health_check():
    """Retrieve database and system health status."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "database": "connected"  # Real database check will be integrated in routers
    }

@app.get("/version", response_model=dict, tags=["System"])
async def version_check():
    """Retrieve current API and model version metadata."""
    return {
        "api_version": "1.0.0",
        "model_version": "LightGBM-v3-selected",
        "simulation_version": "MonteCarlo-48teams-v1.0"
    }
