"""
Services router - endpoints for service management and recommendations
"""
from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services import services_service

router = APIRouter()


@router.get("/")
async def get_services(
    user: Optional[str] = Query(None),
    limit: Optional[int] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all services
    Can be filtered by user to show their most used services
    """
    return await services_service.get_services(db, user, limit)


@router.get("/getRecomendations")
async def get_recomendations(
    user_id: str = Query(..., description="User ID for recommendations"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get recommendations for a user (real-time KNN)
    Runs Python KNN script and returns results
    """
    return await services_service.get_recomendations(user_id)


@router.get("/getRecomendation")
async def get_recomendation(
    user_id: Optional[str] = Query(None, description="User ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get recommendations from pre-computed file
    Returns cached recommendations from recomendations.json
    """
    return await services_service.get_recomendation(user_id)


@router.get("/popular")
async def get_popular_services(
    type: str = Query("any", description="Filter by type: 'table', 'dataset', 'service', 'any'"),
    limit: int = Query(20, description="Number of results to return"),
    period: str = Query("all", description="Time period: 'week', 'month', 'year', 'all'"),
    min_calls: int = Query(1, description="Minimum number of calls"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    ids_only: bool = Query(False, description="Return only IDs"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get most popular services
    Returns list of services sorted by popularity with various filters
    """
    return await services_service.get_popular_services(
        db, type, limit, period, min_calls, user_id, ids_only
    )


@router.get("/parameters/{service_id}")
async def get_service_parameters(
    service_id: int,
    user: Optional[str] = Query(None, description="Filter by user"),
    limit: int = Query(100, description="Number of results"),
    unique: str = Query("true", description="Return only unique parameter combinations"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get service parameters history
    Returns historical parameter values used with this service
    """
    return await services_service.get_service_parameters(
        db, service_id, user, limit, unique
    )

