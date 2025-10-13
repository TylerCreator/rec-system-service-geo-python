"""
User profile models
"""
from typing import List, Dict, Set
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class UserProfile:
    """User profile for recommendations"""
    user_id: str
    used_services: Set[int] = field(default_factory=set)
    service_frequencies: Dict[int, int] = field(default_factory=dict)
    total_calls: int = 0
    first_call: datetime | None = None
    last_call: datetime | None = None
    favorite_categories: List[str] = field(default_factory=list)
    
    def is_new_user(self, min_calls: int = 3) -> bool:
        """Check if user is new (cold start problem)"""
        return self.total_calls < min_calls
    
    def get_top_services(self, n: int = 5) -> List[int]:
        """Get user's top N most used services"""
        sorted_services = sorted(
            self.service_frequencies.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [service_id for service_id, _ in sorted_services[:n]]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "user_id": self.user_id,
            "used_services": list(self.used_services),
            "total_calls": self.total_calls,
            "first_call": self.first_call.isoformat() if self.first_call else None,
            "last_call": self.last_call.isoformat() if self.last_call else None,
            "favorite_categories": self.favorite_categories,
            "top_services": self.get_top_services()
        }





