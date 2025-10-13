"""
Cache manager for recommendations
"""
import time
from typing import Optional, Dict, Any
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    data: Any
    timestamp: float
    ttl: int
    hits: int = 0
    
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        return time.time() - self.timestamp > self.ttl
    
    def touch(self):
        """Update hit count"""
        self.hits += 1


class RecommendationCache:
    """
    LRU cache for recommendations with TTL support
    
    Singleton pattern to ensure single cache instance
    """
    
    _instance: Optional['RecommendationCache'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize cache
        
        Args:
            max_size: Maximum number of entries
            default_ttl: Default time-to-live in seconds
        """
        if self._initialized:
            return
        
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0
        }
        self._initialized = True
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        if key not in self._cache:
            self._stats["misses"] += 1
            return None
        
        entry = self._cache[key]
        
        # Check if expired
        if entry.is_expired():
            self._stats["expirations"] += 1
            self._stats["misses"] += 1
            del self._cache[key]
            return None
        
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        entry.touch()
        self._stats["hits"] += 1
        
        return entry.data
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        # Remove oldest if at capacity
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._evict_oldest()
        
        # Create entry
        entry = CacheEntry(
            data=value,
            timestamp=time.time(),
            ttl=ttl if ttl is not None else self.default_ttl
        )
        
        # Update cache
        self._cache[key] = entry
        self._cache.move_to_end(key)
    
    def delete(self, key: str) -> bool:
        """
        Delete key from cache
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False if not found
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self):
        """Clear all cache entries"""
        self._cache.clear()
        self._stats["evictions"] += len(self._cache)
    
    def _evict_oldest(self):
        """Evict the oldest (least recently used) entry"""
        if self._cache:
            self._cache.popitem(last=False)
            self._stats["evictions"] += 1
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries
        
        Returns:
            Number of entries removed
        """
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        self._stats["expirations"] += len(expired_keys)
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "utilization": len(self._cache) / self.max_size if self.max_size > 0 else 0,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": round(hit_rate, 4),
            "evictions": self._stats["evictions"],
            "expirations": self._stats["expirations"],
            "total_requests": total_requests
        }
    
    def get_entry_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get information about a cache entry"""
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        age = time.time() - entry.timestamp
        
        return {
            "key": key,
            "age_seconds": round(age, 2),
            "ttl": entry.ttl,
            "remaining_seconds": round(entry.ttl - age, 2),
            "hits": entry.hits,
            "is_expired": entry.is_expired()
        }


# Singleton instance
_cache_instance = RecommendationCache()


def get_cache() -> RecommendationCache:
    """Get the global cache instance"""
    return _cache_instance





