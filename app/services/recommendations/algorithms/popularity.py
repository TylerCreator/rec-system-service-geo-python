"""
Popularity-based recommendation algorithm
Simple but effective baseline
"""
import numpy as np
from typing import List, Optional

from app.services.recommendations.base import RecommendationAlgorithm
from app.services.recommendations.models import Recommendation
from app.services.recommendations.data_loader import DataLoader


class PopularityRecommendationAlgorithm(RecommendationAlgorithm):
    """
    Popularity-based recommendations
    
    Recommends most popular services (by call count)
    excluding services already used by the user.
    """
    
    def __init__(self, data_loader: DataLoader):
        super().__init__(name="popularity")
        self.data_loader = data_loader
        self.popular_services: Optional[np.ndarray] = None
        self.service_ids: Optional[np.ndarray] = None
        self.popularity_scores: Optional[np.ndarray] = None
    
    async def train(self, data=None) -> None:
        """
        Prepare popularity rankings
        
        Args:
            data: Optional pre-loaded data
        """
        print("Training popularity-based recommender...")
        
        # Prepare data if needed
        if data is None:
            matrix, users, services = self.data_loader.prepare_user_item_matrix(
                normalize=False  # Don't normalize for popularity
            )
        else:
            matrix, users, services = data
        
        self.service_ids = services
        
        # Calculate popularity (total calls per service)
        total_calls = np.sum(matrix, axis=0)
        
        # Get sorted indices by popularity
        sorted_indices = np.argsort(total_calls)[::-1]
        
        # Filter out services with zero calls
        non_zero_mask = total_calls[sorted_indices] > 0
        self.popular_services = sorted_indices[non_zero_mask]
        
        # Calculate normalized popularity scores
        max_calls = np.max(total_calls)
        self.popularity_scores = total_calls / max_calls if max_calls > 0 else total_calls
        
        self.is_trained = True
        print(f"Popularity model trained. Total services: {len(self.popular_services)}")
    
    async def recommend(
        self,
        user_id: str,
        n: int = 10,
        exclude_services: Optional[List[int]] = None
    ) -> List[Recommendation]:
        """
        Generate popularity-based recommendations
        
        Args:
            user_id: User identifier
            n: Number of recommendations
            exclude_services: Services to exclude
            
        Returns:
            List of popular services as recommendations
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making recommendations")
        
        # Get user's used services
        user_profile = self.data_loader.get_user_profile(user_id)
        used_services = user_profile.used_services if user_profile else set()
        
        # Combine with exclude_services
        if exclude_services:
            used_services = used_services.union(set(exclude_services))
        
        # Filter popular services
        recommendations = []
        for idx in self.popular_services:
            service_id = int(self.service_ids[idx])
            
            # Skip if already used or excluded
            if service_id in used_services:
                continue
            
            score = float(self.popularity_scores[idx])
            
            recommendations.append(Recommendation(
                service_id=service_id,
                score=score,
                algorithm=self.name,
                confidence=0.8,  # High confidence for popular items
                reason="globally_popular"
            ))
            
            if len(recommendations) >= n:
                break
        
        return recommendations
    
    def get_info(self) -> dict:
        """Get algorithm information"""
        info = super().get_info()
        info.update({
            "total_popular_services": len(self.popular_services) if self.popular_services is not None else 0
        })
        return info





