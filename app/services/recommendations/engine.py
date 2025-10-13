"""
Main recommendation engine
Coordinates algorithms, caching, and strategies
"""
import time
from typing import List, Optional, Dict
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.recommendations.base import RecommendationAlgorithm
from app.services.recommendations.data_loader import DataLoader
from app.services.recommendations.cache import get_cache
from app.services.recommendations.models import RecommendationResult, Recommendation
from app.services.recommendations.algorithms import (
    KNNRecommendationAlgorithm,
    PopularityRecommendationAlgorithm,
    AnalyticsPopularityAlgorithm
)


class RecommendationEngine:
    """
    Main recommendation engine
    
    Manages multiple algorithms, handles caching,
    and provides unified interface for recommendations.
    """
    
    def __init__(
        self,
        default_algorithm: str = "knn",
        cache_ttl: int = 3600,
        min_calls_for_knn: int = 3
    ):
        """
        Initialize recommendation engine
        
        Args:
            default_algorithm: Default algorithm to use
            cache_ttl: Cache time-to-live in seconds
            min_calls_for_knn: Minimum calls required to use KNN
        """
        self.default_algorithm = default_algorithm
        self.cache_ttl = cache_ttl
        self.min_calls_for_knn = min_calls_for_knn
        
        # Components
        self.data_loader = DataLoader()
        self.cache = get_cache()
        self.algorithms: Dict[str, RecommendationAlgorithm] = {}
        
        # State
        self.is_initialized = False
    
    async def initialize(self, db: Optional[AsyncSession] = None):
        """
        Initialize the engine and train algorithms
        
        Args:
            db: Database session (optional, can use CSV)
        """
        if self.is_initialized:
            print("Engine already initialized")
            return
        
        print("Initializing recommendation engine...")
        
        # Load data
        if db:
            await self.data_loader.load_from_db(db)
        else:
            # Fallback to CSV if no DB provided
            from app.core.config import settings
            await self.data_loader.load_from_csv(settings.CSV_FILE_PATH)
        
        # Prepare matrix
        matrix, users, services = self.data_loader.prepare_user_item_matrix()
        data = (matrix, users, services)
        
        # Initialize algorithms
        print("Initializing algorithms...")
        
        # KNN algorithm
        knn = KNNRecommendationAlgorithm(
            data_loader=self.data_loader,
            n_neighbors=4,
            metric='cosine'
        )
        await knn.train(data)
        self.algorithms["knn"] = knn
        
        # Popularity algorithm (personalized - excludes used services)
        popularity = PopularityRecommendationAlgorithm(
            data_loader=self.data_loader
        )
        await popularity.train(data)
        self.algorithms["popularity"] = popularity
        
        # Analytics popularity algorithm (non-personalized - real-time DB queries)
        if db:
            analytics_pop = AnalyticsPopularityAlgorithm(db=db)
            await analytics_pop.train()
            self.algorithms["analytics_popularity"] = analytics_pop
            print("Analytics popularity algorithm initialized")
        
        self.is_initialized = True
        print(f"Engine initialized with algorithms: {list(self.algorithms.keys())}")
    
    async def recommend(
        self,
        user_id: str,
        n: int = 10,
        algorithm: Optional[str] = None,
        use_cache: bool = True,
        exclude_services: Optional[List[int]] = None
    ) -> RecommendationResult:
        """
        Get recommendations for a user
        
        Args:
            user_id: User identifier
            n: Number of recommendations
            algorithm: Algorithm to use (None for auto-select)
            use_cache: Use cached results if available
            exclude_services: Services to exclude
            
        Returns:
            RecommendationResult with recommendations and metadata
        """
        if not self.is_initialized:
            raise ValueError("Engine not initialized. Call initialize() first")
        
        start_time = time.time()
        
        # Check cache
        cache_key = f"rec:{user_id}:{n}:{algorithm}:{','.join(map(str, exclude_services or []))}"
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                print(f"Cache hit for user {user_id}")
                cached.execution_time_ms = (time.time() - start_time) * 1000
                cached.metadata["cache_hit"] = True
                return cached
        
        # Select algorithm
        selected_algorithm = algorithm or self._select_algorithm(user_id)
        fallback_used = False
        
        # Get recommendations
        try:
            algo = self.algorithms.get(selected_algorithm)
            if algo is None:
                raise ValueError(f"Algorithm '{selected_algorithm}' not found")
            
            recommendations = await algo.recommend(
                user_id=user_id,
                n=n,
                exclude_services=exclude_services
            )
        except Exception as e:
            print(f"Error with {selected_algorithm}: {e}. Using fallback...")
            # Fallback to popularity
            algo = self.algorithms.get("popularity")
            recommendations = await algo.recommend(
                user_id=user_id,
                n=n,
                exclude_services=exclude_services
            )
            selected_algorithm = "popularity"
            fallback_used = True
        
        # Create result
        execution_time = (time.time() - start_time) * 1000
        result = RecommendationResult(
            user_id=user_id,
            recommendations=recommendations,
            algorithm_used=selected_algorithm,
            fallback_used=fallback_used,
            execution_time_ms=execution_time,
            metadata={
                "cache_hit": False,
                "requested_algorithm": algorithm,
                "n_requested": n,
                "n_returned": len(recommendations)
            }
        )
        
        # Cache result
        if use_cache:
            self.cache.set(cache_key, result, ttl=self.cache_ttl)
        
        return result
    
    async def batch_recommend(
        self,
        user_ids: List[str],
        n: int = 10,
        algorithm: Optional[str] = None
    ) -> Dict[str, RecommendationResult]:
        """
        Get recommendations for multiple users
        
        Args:
            user_ids: List of user identifiers
            n: Number of recommendations per user
            algorithm: Algorithm to use
            
        Returns:
            Dictionary mapping user_id to RecommendationResult
        """
        results = {}
        for user_id in user_ids:
            results[user_id] = await self.recommend(
                user_id=user_id,
                n=n,
                algorithm=algorithm
            )
        return results
    
    def _select_algorithm(self, user_id: str) -> str:
        """
        Auto-select best algorithm for user
        
        Args:
            user_id: User identifier
            
        Returns:
            Algorithm name
        """
        # Get user profile
        profile = self.data_loader.get_user_profile(user_id)
        
        # If user is new (cold start), use popularity
        if profile is None or profile.is_new_user(self.min_calls_for_knn):
            return "popularity"
        
        # Otherwise use default (typically KNN)
        return self.default_algorithm
    
    async def refresh_models(self, db: Optional[AsyncSession] = None):
        """
        Refresh all models with latest data
        
        Args:
            db: Database session (optional, will use CSV if not provided)
        """
        print("Refreshing recommendation models...")
        
        # Clear caches
        self.data_loader.clear_cache()
        self.cache.clear()
        
        # Reload data from DB or fallback to CSV
        if db:
            await self.data_loader.load_from_db(db, force_refresh=True)
        else:
            from app.core.config import settings
            await self.data_loader.load_from_csv(settings.CSV_FILE_PATH)
        
        # Prepare fresh matrix
        matrix, users, services = self.data_loader.prepare_user_item_matrix()
        data = (matrix, users, services)
        
        # Retrain ALL algorithms
        for name, algo in self.algorithms.items():
            print(f"Retraining algorithm: {name}...")
            
            # Analytics popularity doesn't need data (queries DB directly)
            if name == "analytics_popularity":
                await algo.train()
            else:
                await algo.train(data)
            
            print(f"✓ {name} retrained successfully")
        
        # Re-initialize analytics_popularity if we have DB now and didn't before
        if db and "analytics_popularity" not in self.algorithms:
            analytics_pop = AnalyticsPopularityAlgorithm(db=db)
            await analytics_pop.train()
            self.algorithms["analytics_popularity"] = analytics_pop
            print("✓ analytics_popularity initialized")
        
        print(f"✓ All {len(self.algorithms)} models refreshed successfully")
    
    def get_algorithm_info(self, algorithm: Optional[str] = None) -> Dict:
        """
        Get information about algorithms
        
        Args:
            algorithm: Specific algorithm name (None for all)
            
        Returns:
            Algorithm information
        """
        if algorithm:
            algo = self.algorithms.get(algorithm)
            if algo:
                return algo.get_info()
            return {"error": f"Algorithm '{algorithm}' not found"}
        
        return {
            name: algo.get_info()
            for name, algo in self.algorithms.items()
        }
    
    def get_stats(self) -> Dict:
        """Get engine statistics"""
        return {
            "is_initialized": self.is_initialized,
            "algorithms": list(self.algorithms.keys()),
            "default_algorithm": self.default_algorithm,
            "data_loader": self.data_loader.get_stats(),
            "cache": self.cache.get_stats()
        }


# Global engine instance
_engine_instance: Optional[RecommendationEngine] = None


def get_engine() -> RecommendationEngine:
    """Get or create the global engine instance"""
    global _engine_instance
    if _engine_instance is None:
        from app.core.config import settings
        _engine_instance = RecommendationEngine(
            default_algorithm=getattr(settings, 'RECOMMENDATION_DEFAULT_ALGORITHM', 'knn'),
            cache_ttl=getattr(settings, 'RECOMMENDATION_CACHE_TTL', 3600),
            min_calls_for_knn=getattr(settings, 'RECOMMENDATION_MIN_USER_CALLS', 3)
        )
    return _engine_instance

