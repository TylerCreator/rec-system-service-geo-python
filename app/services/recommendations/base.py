"""
Base classes and interfaces for recommendation algorithms
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from app.services.recommendations.models import Recommendation


class RecommendationAlgorithm(ABC):
    """
    Abstract base class for recommendation algorithms
    
    All recommendation algorithms must inherit from this class
    and implement the recommend method.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.is_trained = False
    
    @abstractmethod
    async def train(self, data) -> None:
        """
        Train/prepare the algorithm with data
        
        Args:
            data: Training data (format depends on algorithm)
        """
        pass
    
    @abstractmethod
    async def recommend(
        self,
        user_id: str,
        n: int = 10,
        exclude_services: Optional[List[int]] = None
    ) -> List[Recommendation]:
        """
        Generate recommendations for a user
        
        Args:
            user_id: User identifier
            n: Number of recommendations to return
            exclude_services: Services to exclude from recommendations
            
        Returns:
            List of Recommendation objects sorted by score (descending)
        """
        pass
    
    async def batch_recommend(
        self,
        user_ids: List[str],
        n: int = 10
    ) -> dict[str, List[Recommendation]]:
        """
        Generate recommendations for multiple users
        
        Default implementation calls recommend() for each user.
        Can be overridden for more efficient batch processing.
        
        Args:
            user_ids: List of user identifiers
            n: Number of recommendations per user
            
        Returns:
            Dictionary mapping user_id to list of recommendations
        """
        results = {}
        for user_id in user_ids:
            results[user_id] = await self.recommend(user_id, n)
        return results
    
    def get_info(self) -> dict:
        """Get information about the algorithm"""
        return {
            "name": self.name,
            "is_trained": self.is_trained,
            "type": self.__class__.__name__
        }





