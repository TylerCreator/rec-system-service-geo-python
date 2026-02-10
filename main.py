"""
FastAPI Recommendation System - Main Application
Система рекомендаций сервисов на основе истории вызовов
"""
import os
import sys
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.core.config import settings
from app.core.database import engine, init_db
from app.routers import calls, services, datasets, compositions, update, sequential
from app.routers import table_compositions

# Import models to ensure they're registered with SQLAlchemy
from app.models import models

# Scheduler instance
scheduler = AsyncIOScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    Handles startup and shutdown events
    """
    # Startup
    print("🚀 Starting application...")
    print(f"📅 Current time: {datetime.now().isoformat()}")
    
    # Initialize database
    await init_db()
    print("✅ Database initialized")
    
    # Initialize recommendation engine
    try:
        from app.core.database import get_db
        from app.services.recommendations_service import initialize_engine
        from app.services.sequential_recommendations_service import initialize_sequential_engine
        
        # Get DB session for initialization
        async for db in get_db():
            # Initialize main recommendation engine
            await initialize_engine(db)
            print("✅ Recommendation engine initialized")
            
            # Initialize sequential engine (loads model if available)
            try:
                await initialize_sequential_engine(db)
                print("✅ Sequential DAGNN engine initialized")
            except Exception as seq_err:
                print(f"⚠️  Sequential engine: {seq_err}")
                print("   Run POST /sequential/train to train the model")
            
            break
    except Exception as e:
        print(f"⚠️  Failed to initialize recommendation engine: {e}")
        print("   Engine will initialize on first request")
    
    # Setup cron job for daily updates
    if settings.ENABLE_CRON:
        from app.services.update_service import run_full_update
        
        scheduler.add_job(
            run_full_update,
            trigger=CronTrigger(hour=0, minute=0, timezone="Asia/Irkutsk"),
            id="daily_update",
            name="Daily full system update",
            replace_existing=True
        )
        scheduler.start()
        print("⏰ Cron job scheduled: Daily at 00:00 Asia/Irkutsk timezone")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down application...")
    if scheduler.running:
        scheduler.shutdown()
        print("⏰ Scheduler stopped")


# Create FastAPI application
app = FastAPI(
    title="Service Recommendation System",
    description="API для системы рекомендаций сервисов на основе истории вызовов",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене заменить на конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "message": "Service Recommendation System API",
        "status": "running",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    }


# Admin endpoint for manual cron trigger
@app.get("/admin/run-cron")
async def admin_run_cron():
    """
    Manual cron job trigger (for testing)
    Redirects to /update/full
    """
    from app.services.update_service import run_full_update
    
    print("🔧 Manual cron job triggered via /admin/run-cron")
    result = await run_full_update()
    
    return {
        "message": "Full update triggered manually",
        "result": result,
        "timestamp": datetime.now().isoformat()
    }


# Include routers
app.include_router(calls.router, prefix="/calls", tags=["calls"])
app.include_router(services.router, prefix="/services", tags=["services"])
app.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
app.include_router(compositions.router, prefix="/compositions", tags=["compositions"])
app.include_router(update.router, prefix="/update", tags=["update"])
app.include_router(sequential.router, prefix="/sequential", tags=["sequential-recommendations"])
app.include_router(table_compositions.router, prefix="/table-compositions", tags=["table-compositions"])


# 404 handler
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": "страница не найдена"}
    )


if __name__ == "__main__":
    import uvicorn
    
    # SSL configuration
    ssl_keyfile = settings.SSL_KEY_PATH if settings.SSL_ENABLED else None
    ssl_certfile = settings.SSL_CERT_PATH if settings.SSL_ENABLED else None
    
    if settings.SSL_ENABLED and ssl_keyfile and ssl_certfile:
        if os.path.exists(ssl_keyfile) and os.path.exists(ssl_certfile):
            print(f"🔒 Starting HTTPS server on port {settings.PORT}")
            uvicorn.run(
                "main:app",
                host="0.0.0.0",
                port=settings.PORT,
                ssl_keyfile=ssl_keyfile,
                ssl_certfile=ssl_certfile,
                reload=settings.DEBUG
            )
        else:
            print(f"⚠️  SSL certificates not found, starting HTTP server on port {settings.PORT}")
            uvicorn.run(
                "main:app",
                host="0.0.0.0",
                port=settings.PORT,
                reload=settings.DEBUG
            )
    else:
        print(f"🌐 Starting HTTP server on port {settings.PORT}")
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=settings.PORT,
            reload=settings.DEBUG
        )

