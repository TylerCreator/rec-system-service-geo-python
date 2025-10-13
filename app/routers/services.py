"""
Services router - endpoints for service management and recommendations
"""
from typing import Optional, List
from fastapi import APIRouter, Depends, Query, Body
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services import services_service, recommendations_service

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


# ===== LEGACY ENDPOINTS (DEPRECATED) =====

@router.get("/legacy/getRecomendations")
async def get_recomendations_legacy(
    user_id: str = Query(..., description="User ID for recommendations"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get recommendations for a user - LEGACY ENDPOINT
    
    ⚠️ DEPRECATED: Use /services/recommendations/{user_id} instead
    
    Uses new engine internally. Returns unified format with IDs only.
    """
    # Use new engine with ids_only=true for backward compatibility
    return await recommendations_service.get_recommendations_v2(
        user_id=user_id,
        n=15,
        ids_only=True,
        db=db
    )


@router.get("/legacy/getRecomendation")
async def get_recomendation_legacy(
    user_id: Optional[str] = Query(None, description="User ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get recommendations from pre-computed file - LEGACY ENDPOINT
    
    ⚠️ DEPRECATED: Use /services/recommendations/{user_id} instead
    
    Returns unified format with IDs only.
    """
    # Use new engine with ids_only=true for backward compatibility
    return await recommendations_service.get_recommendations_v2(
        user_id=user_id,
        n=15,
        ids_only=True,
        db=db
    )


# ===== NEW V2 ENDPOINTS =====

@router.get("/recommendations/{user_id}")
async def get_user_recommendations(
    user_id: str,
    n: int = Query(10, ge=1, le=100, description="Number of recommendations"),
    algorithm: Optional[str] = Query(None, description="Algorithm: knn, popularity, analytics_popularity, or auto"),
    period: Optional[str] = Query(None, description="Period for analytics_popularity: week, month, year, all"),
    min_calls: Optional[int] = Query(None, ge=1, description="Minimum calls for analytics_popularity"),
    ids_only: bool = Query(False, description="Return only service IDs (simple array)"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get personalized recommendations for a user (V2 API)
    
    Algorithms:
    - knn: Collaborative filtering (personalized)
    - popularity: Popular services excluding used (personalized)
    - analytics_popularity: Real-time popular services from DB (NOT personalized)
    - auto: Automatic selection based on user profile
    
    Parameters:
    - n: Number of recommendations (1-100)
    - algorithm: Algorithm to use (auto-select if not specified)
    - period: Time period for analytics_popularity (week/month/year/all)
    - min_calls: Minimum calls for analytics_popularity
    - ids_only: If true, returns only array of IDs [1001, 1002, ...]
    
    Returns:
    - ids_only=false: Full object with metadata
    - ids_only=true: Simple array [1001, 1002, 1003, ...]
    """
    return await recommendations_service.get_recommendations_v2(
        user_id=user_id,
        n=n,
        algorithm=algorithm,
        period=period,
        min_calls=min_calls,
        ids_only=ids_only,
        db=db
    )


@router.post("/recommendations/batch")
async def get_batch_recommendations(
    user_ids: List[str] = Body(..., description="List of user IDs"),
    n: int = Body(10, ge=1, le=100, description="Number of recommendations per user"),
    algorithm: Optional[str] = Body(None, description="Algorithm to use"),
    ids_only: bool = Body(False, description="Return only service IDs"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get recommendations for multiple users at once
    
    Efficiently generates recommendations for a batch of users.
    Useful for pre-computing recommendations or bulk operations.
    
    Returns:
    - ids_only=false: {"results": {user1: {full}, user2: {full}}, "total_users": 2}
    - ids_only=true: {user1: [1001, 1002], user2: [2001, 2002]}
    """
    return await recommendations_service.get_recommendations_batch(
        user_ids=user_ids,
        n=n,
        algorithm=algorithm,
        ids_only=ids_only,
        db=db
    )


@router.get("/recommendations/algorithms")
async def list_algorithms(
    algorithm: Optional[str] = Query(None, description="Get info for specific algorithm")
):
    """
    Get information about available recommendation algorithms
    
    Returns details about each algorithm including:
    - Name and type
    - Training status
    - Configuration parameters
    - Performance characteristics
    """
    return await recommendations_service.get_algorithm_info(algorithm)


@router.get("/recommendations/stats")
async def get_recommendation_stats():
    """
    Get recommendation engine statistics
    
    Returns metrics about:
    - Cache performance (hit rate, size)
    - Data loader status
    - Algorithm information
    - Overall system health
    """
    return await recommendations_service.get_engine_stats()


@router.post("/recommendations/refresh")
async def refresh_recommendation_models(
    db: AsyncSession = Depends(get_db)
):
    """
    Refresh recommendation models with latest data
    
    Reloads data from database and retrains all algorithms.
    Use this after significant data updates or on a schedule.
    """
    return await recommendations_service.refresh_recommendations(db)


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

