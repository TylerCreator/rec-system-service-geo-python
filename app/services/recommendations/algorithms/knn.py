"""
KNN-based collaborative filtering algorithm
Migrated and improved from app/static/knn.py
"""
import numpy as np
from typing import List, Optional, Tuple
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator

from app.services.recommendations.base import RecommendationAlgorithm
from app.services.recommendations.models import Recommendation
from app.services.recommendations.data_loader import DataLoader


class KNNModel(BaseEstimator):
    """
    Custom KNN model for collaborative filtering
    Based on the original implementation in knn.py
    """
    
    def __init__(self, n_neighbors: int = 3, metric: str = 'cosine'):
        self.n_neighbors = n_neighbors + 1  # +1 because user itself is included
        self.metric = metric
        self.is_fitted_ = False
        self.X = None
        self.nbrs = None
        self.distances = None
        self.indices = None
    
    def fit(self, X: np.ndarray):
        """Fit the KNN model"""
        self.X = X
        self.nbrs = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, len(X)),
            metric=self.metric
        ).fit(X)
        self.distances, self.indices = self.nbrs.kneighbors(X)
        self.is_fitted_ = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict ratings for users"""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        preds = np.zeros(X.shape)
        
        # For each user
        for i in range(self.indices.shape[0]):
            temp = np.zeros(X[0].shape)
            
            # Average the ratings of nearest neighbors (excluding self)
            for neighbor_idx in self.indices[i][1:]:
                temp += X[neighbor_idx]
            
            # Average
            n_neighbors_used = min(self.n_neighbors - 1, len(self.indices[i]) - 1)
            if n_neighbors_used > 0:
                temp /= n_neighbors_used
            
            preds[i] = temp
        
        return preds


class KNNRecommendationAlgorithm(RecommendationAlgorithm):
    """
    KNN-based collaborative filtering recommendation algorithm
    
    Uses user-item interaction matrix and finds similar users
    to generate recommendations.
    """
    
    def __init__(
        self,
        data_loader: DataLoader,
        n_neighbors: int = 4,
        metric: str = 'cosine'
    ):
        super().__init__(name="knn")
        self.data_loader = data_loader
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.model: Optional[KNNModel] = None
        self.predictions: Optional[np.ndarray] = None
        self.user_ids: Optional[np.ndarray] = None
        self.service_ids: Optional[np.ndarray] = None
        self.popular_services: Optional[np.ndarray] = None
    
    async def train(self, data=None) -> None:
        """
        Train the KNN model
        
        Args:
            data: Optional pre-loaded data (tuple of matrix, users, services)
        """
        print(f"Training KNN model with n_neighbors={self.n_neighbors}, metric={self.metric}")
        
        # Prepare data
        if data is None:
            matrix, users, services = self.data_loader.prepare_user_item_matrix(
                normalize=True
            )
        else:
            matrix, users, services = data
        
        self.user_ids = users
        self.service_ids = services
        
        # Train KNN model
        self.model = KNNModel(
            n_neighbors=self.n_neighbors,
            metric=self.metric
        )
        self.model.fit(matrix)
        
        # Generate predictions
        self.predictions = self.model.predict(matrix)
        
        # Calculate popular services
        self.popular_services = self.data_loader.get_popular_services(n=100)
        
        self.is_trained = True
        print(f"KNN model trained successfully. Users: {len(users)}, Services: {len(services)}")
    
    async def recommend(
        self,
        user_id: str,
        n: int = 10,
        exclude_services: Optional[List[int]] = None
    ) -> List[Recommendation]:
        """
        Generate KNN-based recommendations for a user
        
        Args:
            user_id: User identifier
            n: Number of recommendations
            exclude_services: Services to exclude
            
        Returns:
            List of recommendations sorted by score
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making recommendations")
        
        # Check if user exists
        if user_id not in self.user_ids:
            # Return popular services for unknown users
            return self._get_popular_recommendations(n, exclude_services)
        
        # Get user index
        user_idx = np.where(self.user_ids == user_id)[0][0]
        
        # Get user's used services
        user_profile = self.data_loader.get_user_profile(user_id)
        used_services = user_profile.used_services if user_profile else set()
        
        # Combine with exclude_services
        if exclude_services:
            used_services = used_services.union(set(exclude_services))
        
        # Get predicted scores for this user
        user_predictions = self.predictions[user_idx]
        
        # Get services sorted by predicted score
        sorted_indices = np.argsort(user_predictions)[::-1]
        
        # Filter out zero predictions
        eps = 1e-10
        sorted_indices = sorted_indices[user_predictions[sorted_indices] > eps]
        
        # Get used service indices
        used_service_indices = set()
        for service in used_services:
            if service in self.service_ids:
                service_idx = np.where(self.service_ids == service)[0][0]
                used_service_indices.add(service_idx)
        
        # Filter out used services
        sorted_indices = [idx for idx in sorted_indices if idx not in used_service_indices]
        
        # If not enough predictions, add popular services
        if len(sorted_indices) < n:
            # Get popular services that are not used and not already recommended
            recommended_services = set(self.service_ids[sorted_indices])
            popular_to_add = [
                idx for idx in self.popular_services
                if self.service_ids[idx] not in used_services
                and self.service_ids[idx] not in recommended_services
            ]
            sorted_indices = list(sorted_indices) + popular_to_add
        
        # Take top N
        top_indices = sorted_indices[:n]
        
        # Create recommendations
        recommendations = []
        for idx in top_indices:
            service_id = int(self.service_ids[idx])
            score = float(user_predictions[idx])
            
            # Determine if this is from KNN or popular fallback
            reason = "knn_prediction" if score > eps else "popular_fallback"
            
            recommendations.append(Recommendation(
                service_id=service_id,
                score=score if score > eps else 0.5,  # Default score for popular
                algorithm=self.name,
                confidence=min(score * 1.2, 1.0) if score > eps else 0.3,
                reason=reason
            ))
        
        return recommendations
    
    def _get_popular_recommendations(
        self,
        n: int,
        exclude_services: Optional[List[int]] = None
    ) -> List[Recommendation]:
        """
        Get popular services as fallback for unknown users
        
        Args:
            n: Number of recommendations
            exclude_services: Services to exclude
            
        Returns:
            List of popular service recommendations
        """
        exclude_set = set(exclude_services) if exclude_services else set()
        
        recommendations = []
        for idx in self.popular_services:
            service_id = int(self.service_ids[idx])
            
            if service_id in exclude_set:
                continue
            
            recommendations.append(Recommendation(
                service_id=service_id,
                score=0.5,  # Default score for popular
                algorithm=self.name,
                confidence=0.3,
                reason="popular_fallback"
            ))
            
            if len(recommendations) >= n:
                break
        
        return recommendations
    
    def get_info(self) -> dict:
        """Get algorithm information"""
        info = super().get_info()
        info.update({
            "n_neighbors": self.n_neighbors,
            "metric": self.metric,
            "total_users": len(self.user_ids) if self.user_ids is not None else 0,
            "total_services": len(self.service_ids) if self.service_ids is not None else 0
        })
        return info





