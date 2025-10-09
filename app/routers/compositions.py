"""
Compositions router - endpoints for service composition analysis
"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services import compositions_service

router = APIRouter()


@router.get("/recover")
async def recover(db: AsyncSession = Depends(get_db)):
    """
    Recover service compositions from call history
    Analyzes call history to find service composition workflows
    """
    result = await compositions_service.recover(db)
    return result


@router.get("/recoverNew")
async def recover_new(db: AsyncSession = Depends(get_db)):
    """
    Advanced service composition recovery
    Uses improved algorithm for composition detection
    """
    result = await compositions_service.recover_new(db)
    return result


@router.get("/")
async def fetch_all_compositions(db: AsyncSession = Depends(get_db)):
    """
    Get all compositions from database
    """
    return await compositions_service.fetch_all_compositions(db)


@router.get("/stats")
async def get_composition_stats(db: AsyncSession = Depends(get_db)):
    """
    Get composition statistics
    Returns graph of service usage and connections
    """
    return await compositions_service.get_composition_stats(db)

