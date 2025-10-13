"""
Data models for recommendations
"""
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Recommendation:
    """Single recommendation item"""
    service_id: int
    score: float
    algorithm: str
    confidence: float = 1.0
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "service_id": self.service_id,
            "score": round(self.score, 4),
            "algorithm": self.algorithm,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "metadata": self.metadata
        }


@dataclass
class RecommendationResult:
    """Result with multiple recommendations"""
    user_id: str
    recommendations: list[Recommendation]
    algorithm_used: str
    fallback_used: bool = False
    execution_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "user_id": self.user_id,
            "recommendations": [rec.to_dict() for rec in self.recommendations],
            "algorithm_used": self.algorithm_used,
            "fallback_used": self.fallback_used,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "timestamp": self.timestamp.isoformat(),
            "count": len(self.recommendations),
            "metadata": self.metadata
        }





