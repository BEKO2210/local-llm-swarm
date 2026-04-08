"""FastAPI entry point for the Multi-Agent Swarm system."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.chat import router as chat_router
from backend.app.core.config import get_settings
from backend.app.database.db import init_db, close_db
from backend.app.runtimes.process_manager import get_process_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    logger.info("Starting Multi-Agent Swarm system...")

    settings = get_settings()
    logger.info(f"Loaded {len(settings.models.models)} models")
    logger.info(f"Loaded {len(settings.pools.pools)} pools")

    logger.info("Initializing database...")
    await init_db()
    logger.info("Database initialized successfully")

    yield

    logger.info("Shutting down Multi-Agent Swarm system...")

    process_manager = get_process_manager()
    await process_manager.stop_all()
    # Forceful cleanup for Windows zombies
    process_manager.cleanup_zombies()

    await close_db()
    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Multi-Agent Swarm API",
    description="Local multi-agent swarm system optimized for constrained hardware",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(chat_router, prefix="/api")


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    process_manager = get_process_manager()
    return {
        "status": "healthy",
        "version": "0.1.0",
        "runtime": {
            "running_models": process_manager.get_status()["running_processes"],
            "vram_used_mb": process_manager.total_vram_used_mb,
            "vram_available_mb": process_manager.available_vram_mb,
        },
    }


@app.get("/status")
async def get_system_status():
    process_manager = get_process_manager()
    settings = get_settings()
    return {
        "system": {
            "version": settings.app_version,
            "debug": settings.debug,
        },
        "models": {
            model_id: {
                "name": config.name,
                "vram_estimate_mb": config.vram_estimate_mb,
            }
            for model_id, config in settings.models.models.items()
        },
        "pools": {
            pool_id: {
                "name": pool.name,
                "models": pool.models,
                "max_concurrent": pool.max_concurrent,
            }
            for pool_id, pool in settings.pools.pools.items()
        },
        "runtime": process_manager.get_status(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
