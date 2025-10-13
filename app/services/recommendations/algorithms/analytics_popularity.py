"""
Analytics-based popularity recommendation algorithm
Uses real-time data from database without personalization
"""
from typing import List, Optional
from datetime import datetime, timedelta
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.recommendations.base import RecommendationAlgorithm
from app.services.recommendations.models import Recommendation
from app.models.models import Call


class AnalyticsPopularityAlgorithm(RecommendationAlgorithm):
    """
    Analytics-based popularity recommendations
    
    Uses real-time database queries to get popular services.
    NO personalization - returns same results for all users.
    Supports filtering by period, type, and minimum calls.
    """
    
    def __init__(self, db: AsyncSession):
        super().__init__(name="analytics_popularity")
        self.db = db
        self.period = "all"  # Default period
        self.min_calls = 1   # Default minimum calls
    
    async def train(self, data=None) -> None:
        """
        No training needed - queries DB in real-time
        """
        self.is_trained = True
        print(f"Analytics popularity algorithm ready (no training needed)")
    
    async def recommend(
        self,
        user_id: str,
        n: int = 10,
        exclude_services: Optional[List[int]] = None
    ) -> List[Recommendation]:
        """
        Generate analytics-based popularity recommendations
        
        Args:
            user_id: User identifier (NOT used for personalization)
            n: Number of recommendations
            exclude_services: Services to exclude (optional)
            
        Returns:
            List of popular services from real-time DB data
        """
        if not self.is_trained:
            raise ValueError("Algorithm not initialized")
        
        # Build time condition
        time_condition = None
        if self.period != "all":
            now = datetime.utcnow()
            if self.period == "week":
                start_date = now - timedelta(days=7)
            elif self.period == "month":
                start_date = now - timedelta(days=30)
            elif self.period == "year":
                start_date = now - timedelta(days=365)
            else:
                start_date = None
            
            if start_date:
                time_condition = Call.start_time >= start_date
        
        # Build WHERE clause
        where_conditions = [Call.status == "TASK_SUCCEEDED"]
        if time_condition is not None:
            where_conditions.append(time_condition)
        
        # Get all successful calls
        result = await self.db.execute(
            select(Call.mid, Call.owner).where(and_(*where_conditions))
        )
        calls = result.all()
        
        # Group statistics
        service_stats = {}
        for mid, owner in calls:
            if mid not in service_stats:
                service_stats[mid] = {
                    "mid": mid,
                    "call_count": 0,
                    "unique_users": set()
                }
            service_stats[mid]["call_count"] += 1
            service_stats[mid]["unique_users"].add(owner)
        
        # Convert to list and filter
        call_stats = [
            {
                "mid": stat["mid"],
                "call_count": stat["call_count"],
                "unique_users": len(stat["unique_users"])
            }
            for stat in service_stats.values()
            if stat["call_count"] >= self.min_calls
        ]
        
        # Sort by call count
        call_stats.sort(key=lambda x: x["call_count"], reverse=True)
        
        # Apply exclusions
        exclude_set = set(exclude_services) if exclude_services else set()
        
        # Build recommendations
        recommendations = []
        max_calls = call_stats[0]["call_count"] if call_stats else 1
        
        for stat in call_stats:
            if stat["mid"] in exclude_set:
                continue
            
            # Normalize score
            score = stat["call_count"] / max_calls
            
            # Calculate popularity score
            popularity = stat["call_count"] / max(stat["unique_users"], 1)
            
            recommendations.append(Recommendation(
                service_id=stat["mid"],
                score=score,
                algorithm=self.name,
                confidence=0.9,  # High confidence - based on real data
                reason="analytics_popular",
                metadata={
                    "call_count": stat["call_count"],
                    "unique_users": stat["unique_users"],
                    "popularity": round(popularity, 2),
                    "period": self.period
                }
            ))
            
            if len(recommendations) >= n:
                break
        
        return recommendations
    
    def set_period(self, period: str) -> None:
        """
        Set time period for popularity calculation
        
        Args:
            period: "week", "month", "year", or "all"
        """
        self.period = period
    
    def set_min_calls(self, min_calls: int) -> None:
        """
        Set minimum calls threshold
        
        Args:
            min_calls: Minimum number of calls required
        """
        self.min_calls = min_calls
    
    def get_info(self) -> dict:
        """Get algorithm information"""
        info = super().get_info()
        info.update({
            "period": self.period,
            "min_calls": self.min_calls,
            "personalized": False,
            "real_time": True,
            "source": "database"
        })
        return info





