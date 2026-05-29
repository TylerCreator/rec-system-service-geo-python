"""
Data loader for recommendations with caching
"""
import numpy as np
import pandas as pd
import json
from typing import Dict, Optional, Tuple, Set, Any, List
from datetime import datetime, timedelta
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Call, Dataset
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
        self._service_dataset_map: Dict[int, Set[int]] = {}
        self._dataset_service_map: Dict[int, Set[int]] = {}
    
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

        # Load service-dataset connection mappings
        print("Loading service-dataset connections...")
        self._service_dataset_map.clear()
        self._dataset_service_map.clear()
        
        try:
            # Build dataset GUID-to-ID mapping
            result = await db.execute(select(Dataset.id, Dataset.guid))
            guid_map = {guid: ds_id for ds_id, guid in result.all() if guid}
            
            # Query successful service calls (mid < 1,000,000) that reference dataset_id in inputs
            result = await db.execute(
                select(Call.mid, Call.input)
                .where(
                    and_(
                        Call.status == "TASK_SUCCEEDED",
                        Call.mid < 1000000,
                        Call.input.like("%dataset_id%")
                    )
                )
            )
            
            for mid, input_str in result.all():
                if not input_str:
                    continue
                try:
                    dataset_ids = self._extract_dataset_ids(input_str, guid_map)
                    for ds_id in dataset_ids:
                        # populate maps
                        if ds_id not in self._dataset_service_map:
                            self._dataset_service_map[ds_id] = set()
                        self._dataset_service_map[ds_id].add(mid)
                        
                        if mid not in self._service_dataset_map:
                            self._service_dataset_map[mid] = set()
                        self._service_dataset_map[mid].add(ds_id)
                except Exception as parse_err:
                    print(f"Error parsing call connection inputs: {parse_err}")
            
            print(f"Service-dataset connections loaded: {len(self._service_dataset_map)} services, {len(self._dataset_service_map)} datasets")
        except Exception as db_err:
            print(f"Error loading service-dataset connections: {db_err}")
        
        return self._calls_df

    def _extract_dataset_ids(self, input_data: Any, guid_map: Dict[str, int]) -> Set[int]:
        """Recursively extract dataset IDs from call input JSON"""
        dataset_ids = set()
        
        if isinstance(input_data, str):
            try:
                input_data = json.loads(input_data)
            except Exception:
                return dataset_ids
                
        if isinstance(input_data, dict):
            for k, v in input_data.items():
                if k == "dataset_id":
                    try:
                        if isinstance(v, int):
                            dataset_ids.add(v)
                        elif isinstance(v, str):
                            if v in guid_map:
                                dataset_ids.add(guid_map[v])
                            else:
                                dataset_ids.add(int(v))
                    except Exception:
                        pass
                else:
                    dataset_ids.update(self._extract_dataset_ids(v, guid_map))
        elif isinstance(input_data, list):
            for item in input_data:
                dataset_ids.update(self._extract_dataset_ids(item, guid_map))
                
        return dataset_ids

    def get_services_using_dataset(self, dataset_id: int) -> Set[int]:
        """Get services that have used the specified dataset"""
        # Normalize dataset_id if it contains the offset
        if dataset_id >= 1000000:
            dataset_id -= 1000000
        return self._dataset_service_map.get(dataset_id, set())

    def get_datasets_using_service(self, service_id: int) -> Set[int]:
        """Get datasets that have been used by the specified service"""
        return self._service_dataset_map.get(service_id, set())
    
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
        self._service_dataset_map.clear()
        self._dataset_service_map.clear()
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
            "cached_profiles": len(self._user_profiles),
            "service_dataset_connections": len(self._service_dataset_map)
        }





