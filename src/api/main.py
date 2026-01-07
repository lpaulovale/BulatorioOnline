"""
PharmaBula FastAPI Application

Main application entry point with scheduler integration.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes import chat, drugs
from src.config import get_settings
from src.scheduler.jobs import get_scheduler, run_initial_setup, setup_scheduler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown tasks:
    - Starts the background scheduler
    - Runs initial data scrape if needed
    - Gracefully shuts down scheduler on exit
    """
    settings = get_settings()
    
    # Startup
    logger.info("Starting PharmaBula API...")
    
    # Setup and start scheduler
    scheduler = setup_scheduler()
    if settings.enable_scheduler:
        scheduler.start()
        logger.info("Background scheduler started")
        
        # Load sample data if database is empty
        await run_initial_setup()
    
    yield
    
    # Shutdown
    logger.info("Shutting down PharmaBula API...")
    scheduler = get_scheduler()
    if scheduler.running:
        scheduler.shutdown(wait=True)
        logger.info("Scheduler stopped")


# Create FastAPI app
app = FastAPI(
    title="PharmaBula API",
    description=(
        "API para consulta de informações sobre medicamentos. "
        "Assistente inteligente baseado em bulas da ANVISA."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router)
app.include_router(drugs.router)

# Import and include router endpoint
from src.api.routes import router as router_routes
app.include_router(router_routes.router)


@app.get("/health")
async def health_check():
    """
    Overall application health check.
    
    Returns status of all components.
    """
    from src.database.metadata_cache import get_metadata_cache
    from src.database.vector_store import get_vector_store
    
    try:
        cache = get_metadata_cache()
        vector_store = get_vector_store()
        scheduler = get_scheduler()
        
        return {
            "status": "healthy",
            "components": {
                "database": "ok",
                "vector_store": f"{vector_store.count()} documents",
                "scheduler": "running" if scheduler.running else "stopped"
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# Serve frontend
frontend_path = Path(__file__).parent.parent / "frontend"


@app.get("/")
async def serve_frontend():
    """Serve the main frontend page."""
    index_path = frontend_path / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "PharmaBula API - Acesse /docs para documentação"}


# Mount static files if frontend exists
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
