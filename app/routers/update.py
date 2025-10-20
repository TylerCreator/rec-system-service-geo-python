"""
Update router - endpoints for system updates
"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services import update_service

router = APIRouter()


@router.get("/all")
async def update_all(db: AsyncSession = Depends(get_db)):
    """
    Update all data (calls, datasets, services, compositions, recommendations)
    
    Steps:
    1. Update calls from external API
    2. Update datasets from external API
    3. Update services from external API
    4. Recover compositions from call history
    5. Refresh recommendation models (v2 engine)
    """
    result = await update_service.update_all(db)
    return {
        "message": "Update process completed",
        "results": result,
        "timestamp": update_service.datetime.now().isoformat()
    }


@router.get("/recomendations")
async def update_recomendations(db: AsyncSession = Depends(get_db)):
    """
    Update recommendations using KNN script
    """
    result = await update_service.update_recomendations()
    return result


@router.get("/statistic")
async def update_statistic(db: AsyncSession = Depends(get_db)):
    """
    Update user-service statistics
    """
    result = await update_service.update_statistics(db)
    return {
        "success": True,
        "message": "Statistics updated successfully",
        "result": result,
        "timestamp": update_service.datetime.now().isoformat()
    }


@router.get("/full")
async def run_full_update(db: AsyncSession = Depends(get_db)):
    """
    Full system update (for cron job and manual trigger)
    Runs all update operations in sequence
    
    Steps:
    1. Update all data (calls, datasets, services, compositions)
    2. Dump CSV file with latest calls
    3. Update user-service statistics
    4. Update recommendations (legacy KNN script for backward compatibility)
    5. Refresh recommendation models (v2 engine with cache)
    6. Train sequential DAGNN model (workflow predictions)
    """
    result = await update_service.run_full_update()
    return result


@router.get("/local")
async def update_local(db: AsyncSession = Depends(get_db)):
    """
    Local update (statistics only, no external API calls)
    """
    result = await update_service.update_statistics_internal(db)
    return {
        "message": "Local update completed successfully",
        "result": result,
        "timestamp": update_service.datetime.now().isoformat()
    }

