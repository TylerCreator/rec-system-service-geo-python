"""
Recommendations service - high-level business logic for recommendations
Wraps the recommendation engine with additional business logic
"""
import json
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.recommendations import get_engine, RecommendationResult
from app.core.config import settings


async def initialize_engine(db: AsyncSession):
    """
    Initialize recommendation engine (call on startup)
    
    Args:
        db: Database session
    """
    engine = get_engine()
    if not engine.is_initialized:
        await engine.initialize(db)


async def get_recommendations_v2(
    user_id: str,
    n: int = 10,
    algorithm: Optional[str] = None,
    period: Optional[str] = None,
    min_calls: Optional[int] = None,
    ids_only: bool = False,
    db: Optional[AsyncSession] = None
) -> Dict[str, Any] | List[int]:
    """
    Get recommendations using new engine (v2 API)
    
    Args:
        user_id: User identifier
        n: Number of recommendations
        algorithm: Algorithm to use (auto-select if None)
        period: Period for analytics_popularity (week/month/year/all)
        min_calls: Minimum calls for analytics_popularity
        ids_only: If True, return simple array of service IDs
        db: Database session
        
    Returns:
        Full format (ids_only=false):
        {
            "user_id": "user123",
            "recommendations": [{service_id, score, ...}],
            "algorithm_used": "knn",
            "execution_time_ms": 12.3,
            ...
        }
        
        IDs only (ids_only=true):
        [1002120, 1001211, 1000253, ...]
    """
    engine = get_engine()
    
    # Initialize if needed
    if not engine.is_initialized and db:
        await engine.initialize(db)
    
    # Configure analytics_popularity if selected
    if algorithm == "analytics_popularity" and "analytics_popularity" in engine.algorithms:
        analytics_algo = engine.algorithms["analytics_popularity"]
        if period:
            analytics_algo.set_period(period)
        if min_calls:
            analytics_algo.set_min_calls(min_calls)
    
    # Get recommendations
    result = await engine.recommend(
        user_id=user_id,
        n=n,
        algorithm=algorithm
    )
    
    # Return only IDs if requested (simple array)
    if ids_only:
        return [rec.service_id for rec in result.recommendations]
    
    # Return full format
    return result.to_dict()


async def get_recommendations_batch(
    user_ids: List[str],
    n: int = 10,
    algorithm: Optional[str] = None,
    ids_only: bool = False,
    db: Optional[AsyncSession] = None
) -> Dict[str, Any] | Dict[str, List[int]]:
    """
    Get recommendations for multiple users
    
    Args:
        user_ids: List of user identifiers
        n: Number of recommendations per user
        algorithm: Algorithm to use
        ids_only: If True, return only IDs
        db: Database session
        
    Returns:
        Full format (ids_only=false):
        {
            "results": {
                "user1": {full response},
                "user2": {full response}
            },
            "total_users": 2,
            "algorithm_used": "knn"
        }
        
        IDs only (ids_only=true):
        {
            "user1": [1001, 1002, 1003],
            "user2": [2001, 2002, 2003]
        }
    """
    engine = get_engine()
    
    # Initialize if needed
    if not engine.is_initialized and db:
        await engine.initialize(db)
    
    # Get batch recommendations
    results = await engine.batch_recommend(
        user_ids=user_ids,
        n=n,
        algorithm=algorithm
    )
    
    # Convert to response format
    if ids_only:
        # Return only IDs for each user
        return {
            user_id: [rec.service_id for rec in result.recommendations]
            for user_id, result in results.items()
        }
    
    # Return full format
    return {
        "results": {
            user_id: result.to_dict()
            for user_id, result in results.items()
        },
        "total_users": len(results),
        "algorithm_used": algorithm or "auto"
    }


async def get_algorithm_info(algorithm: Optional[str] = None) -> Dict[str, Any]:
    """
    Get information about available algorithms
    
    Args:
        algorithm: Specific algorithm (None for all)
        
    Returns:
        Algorithm information
    """
    engine = get_engine()
    return engine.get_algorithm_info(algorithm)


async def get_engine_stats() -> Dict[str, Any]:
    """
    Get recommendation engine statistics
    
    Returns:
        Engine statistics
    """
    engine = get_engine()
    return engine.get_stats()


async def refresh_recommendations(db: AsyncSession) -> Dict[str, Any]:
    """
    Refresh recommendation models
    
    Args:
        db: Database session
        
    Returns:
        Refresh status
    """
    engine = get_engine()
    
    try:
        await engine.refresh_models(db)
        return {
            "success": True,
            "message": "Models refreshed successfully",
            "stats": engine.get_stats()
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to refresh models: {str(e)}"
        }


# Note: get_recommendations_legacy() removed
# Legacy endpoints now use get_recommendations_v2() with ids_only=True
# This provides unified response format across all endpoints

