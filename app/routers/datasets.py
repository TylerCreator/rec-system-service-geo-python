"""
Datasets router - endpoints for dataset management and recommendations
"""
from typing import Optional, List
from fastapi import APIRouter, Depends, Query, Body
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services import datasets_service, recommendations_service

router = APIRouter()


@router.get("/update")
async def update_datasets(db: AsyncSession = Depends(get_db)):
    """
    Update datasets from remote server
    Synchronizes local database with remote CRIS server
    """
    await datasets_service.update_datasets(db)
    return {"message": "Datasets updated successfully"}


@router.get("/recommendations/{user_id}")
async def get_user_dataset_recommendations(
    user_id: str,
    n: int = Query(10, ge=1, le=100, description="Number of recommendations"),
    algorithm: Optional[str] = Query(None, description="Algorithm: knn, popularity, analytics_popularity, or auto"),
    period: Optional[str] = Query(None, description="Period for analytics_popularity: week, month, year, all"),
    min_calls: Optional[int] = Query(None, ge=1, description="Minimum calls for analytics_popularity"),
    ids_only: bool = Query(False, description="Return only dataset IDs"),
    service_id: Optional[int] = Query(None, description="Filter by service ID that has used these datasets"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get personalized dataset recommendations for a user (V2 API)
    
    Algorithms:
    - knn: Collaborative filtering (personalized)
    - popularity: Popular datasets excluding used (personalized)
    - analytics_popularity: Real-time popular datasets from DB (NOT personalized)
    - auto: Automatic selection based on user profile
    
    Parameters:
    - n: Number of recommendations (1-100)
    - algorithm: Algorithm to use (auto-select if not specified)
    - period: Time period for analytics_popularity (week/month/year/all)
    - min_calls: Minimum calls for analytics_popularity
    - ids_only: If true, returns only array of IDs [1001, 1002, ...]
    - service_id: Optional service filter to suggest only datasets used by this service
    
    Returns:
    - ids_only=false: Full object with metadata
    - ids_only=true: Simple array of original dataset IDs [42, 43, 44, ...]
    """
    return await recommendations_service.get_recommendations_v2(
        user_id=user_id,
        n=n,
        algorithm=algorithm,
        period=period,
        min_calls=min_calls,
        ids_only=ids_only,
        is_dataset=True,
        service_id=service_id,
        db=db
    )


@router.post("/recommendations/batch")
async def get_batch_dataset_recommendations(
    user_ids: List[str] = Body(..., description="List of user IDs"),
    n: int = Body(10, ge=1, le=100, description="Number of recommendations per user"),
    algorithm: Optional[str] = Body(None, description="Algorithm to use"),
    ids_only: bool = Body(False, description="Return only dataset IDs"),
    service_id: Optional[int] = Body(None, description="Filter by service ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get dataset recommendations for multiple users at once
    
    Efficiently generates dataset recommendations for a batch of users.
    
    Returns:
    - ids_only=false: {"results": {user1: {full}, user2: {full}}, "total_users": 2}
    - ids_only=true: {user1: [42, 43], user2: [44, 45]}
    """
    return await recommendations_service.get_recommendations_batch(
        user_ids=user_ids,
        n=n,
        algorithm=algorithm,
        ids_only=ids_only,
        is_dataset=True,
        service_id=service_id,
        db=db
    )

