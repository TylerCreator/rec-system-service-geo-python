"""
Business logic services package.

Keep package initialization lightweight to avoid circular-import issues during
application startup. Submodules should be imported explicitly by consumers, e.g.:
    from app.services import calls_service
"""
from . import (
    calls_service,
    compositions_service,
    datasets_service,
    services_service,
    update_service,
    recommendations_service,
    sequential_recommendations_service,
    table_compositions_service
)

__all__ = [
    "calls_service",
    "compositions_service",
    "datasets_service",
    "services_service",
    "update_service",
    "recommendations_service",
    "sequential_recommendations_service",
    "table_compositions_service",
]
