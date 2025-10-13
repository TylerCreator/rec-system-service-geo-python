"""
Data loader for recommendations with caching
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Call
from app.services.recommendations.models import UserProfile


class DataLoader:
    """
    Loads and prepares data for recommendation algorithms
    Implements caching to avoid repeated database queries
    """
    
    def __init__(self):
        self._calls_df: Optional[pd.DataFrame] = None
        self._user_item_matrix: Optional[np.ndarray] = None
        self._user_ids: Optional[np.ndarray] = None
        self._service_ids: Optional[np.ndarray] = None
        self._user_profiles: Dict[str, UserProfile] = {}
        self._last_load_time: Optional[datetime] = None
        self._cache_ttl: int = 3600  # 1 hour
    
    @property
    def is_cached(self) -> bool:
        """Check if data is cached and still valid"""
        if self._last_load_time is None:
            return False
        age = (datetime.utcnow() - self._last_load_time).total_seconds()
        return age < self._cache_ttl
    
    async def load_from_db(
        self,
        db: AsyncSession,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Load call data from database
        
        Args:
            db: Database session
            force_refresh: Force reload even if cached
            
        Returns:
            DataFrame with call data
        """
        if self.is_cached and not force_refresh and self._calls_df is not None:
            print("Using cached call data")
            return self._calls_df
        
        print("Loading call data from database...")
        
        # Load successful calls
        result = await db.execute(
            select(Call.id, Call.mid, Call.owner, Call.start_time)
            .where(Call.status == "TASK_SUCCEEDED")
            .order_by(Call.start_time.desc())
        )
        calls = result.all()
        
        # Convert to DataFrame
        self._calls_df = pd.DataFrame(
            calls,
            columns=["id", "mid", "owner", "start_time"]
        )
        
        self._last_load_time = datetime.utcnow()
        print(f"Loaded {len(self._calls_df)} calls")
        
        return self._calls_df
    
    async def load_from_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load call data from CSV file (for backward compatibility)
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            DataFrame with call data
        """
        print(f"Loading call data from CSV: {csv_path}")
        self._calls_df = pd.read_csv(csv_path, sep=';')
        self._last_load_time = datetime.utcnow()
        print(f"Loaded {len(self._calls_df)} calls from CSV")
        return self._calls_df
    
    def prepare_user_item_matrix(
        self,
        df: Optional[pd.DataFrame] = None,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare user-item interaction matrix
        
        Args:
            df: DataFrame with calls (uses cached if None)
            normalize: Normalize by user (convert to frequencies)
            
        Returns:
            Tuple of (matrix, user_ids, service_ids)
        """
        if df is None:
            df = self._calls_df
        
        if df is None:
            raise ValueError("No data loaded. Call load_from_db() or load_from_csv() first")
        
        print("Preparing user-item matrix...")
        
        # Get unique users and services
        self._user_ids = df['owner'].unique()
        self._service_ids = df['mid'].unique()
        
        # Create pivot table
        pivot = df.pivot_table(
            values='id',
            index='owner',
            columns='mid',
            aggfunc='count'
        ).fillna(0)
        
        # Initialize matrix
        matrix = np.zeros((len(self._user_ids), len(self._service_ids)))
        
        # Fill matrix
        for i, user in enumerate(self._user_ids):
            for j, service in enumerate(self._service_ids):
                if user in pivot.index and service in pivot.columns:
                    matrix[i, j] = pivot.loc[user, service]
        
        # Normalize if requested
        if normalize:
            for i in range(len(self._user_ids)):
                row_sum = np.sum(matrix[i])
                if row_sum > 0:
                    matrix[i] /= row_sum
        
        self._user_item_matrix = matrix
        
        print(f"Matrix shape: {matrix.shape} (users x services)")
        return matrix, self._user_ids, self._service_ids
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Get or create user profile
        
        Args:
            user_id: User identifier
            
        Returns:
            UserProfile object or None if user not found
        """
        # Return cached profile if exists
        if user_id in self._user_profiles:
            return self._user_profiles[user_id]
        
        # Check if data is loaded
        if self._calls_df is None:
            return None
        
        # Filter user calls
        user_calls = self._calls_df[self._calls_df['owner'] == user_id]
        
        if len(user_calls) == 0:
            return None
        
        # Create profile
        profile = UserProfile(user_id=user_id)
        profile.used_services = set(user_calls['mid'].unique())
        profile.service_frequencies = user_calls['mid'].value_counts().to_dict()
        profile.total_calls = len(user_calls)
        
        if 'start_time' in user_calls.columns:
            profile.first_call = user_calls['start_time'].min()
            profile.last_call = user_calls['start_time'].max()
        
        # Cache profile
        self._user_profiles[user_id] = profile
        
        return profile
    
    def get_popular_services(self, n: int = 100) -> np.ndarray:
        """
        Get most popular services by call count
        
        Args:
            n: Number of services to return
            
        Returns:
            Array of service IDs sorted by popularity
        """
        if self._user_item_matrix is None:
            raise ValueError("Matrix not prepared. Call prepare_user_item_matrix() first")
        
        # Calculate average popularity across all users
        popularity = np.mean(self._user_item_matrix, axis=0)
        
        # Get indices sorted by popularity
        sorted_indices = np.argsort(popularity)[::-1]
        
        # Filter out zero popularity
        eps = 1e-10
        non_zero_indices = sorted_indices[popularity[sorted_indices] > eps]
        
        return non_zero_indices[:n]
    
    def clear_cache(self):
        """Clear all cached data"""
        self._calls_df = None
        self._user_item_matrix = None
        self._user_ids = None
        self._service_ids = None
        self._user_profiles.clear()
        self._last_load_time = None
        print("Cache cleared")
    
    def get_stats(self) -> Dict:
        """Get statistics about loaded data"""
        return {
            "is_cached": self.is_cached,
            "last_load_time": self._last_load_time.isoformat() if self._last_load_time else None,
            "total_calls": len(self._calls_df) if self._calls_df is not None else 0,
            "total_users": len(self._user_ids) if self._user_ids is not None else 0,
            "total_services": len(self._service_ids) if self._service_ids is not None else 0,
            "matrix_prepared": self._user_item_matrix is not None,
            "cached_profiles": len(self._user_profiles)
        }





