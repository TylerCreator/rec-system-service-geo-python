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
    Update all data (calls, datasets, services, compositions)
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

