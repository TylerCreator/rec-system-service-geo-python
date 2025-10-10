"""
Compositions service - business logic for service composition analysis

This is a refactored wrapper that re-exports functionality from specialized modules.
All implementation has been moved to app/services/compositions/ for better organization.
"""
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List

# Import from refactored modules
from .compositions import (
    # Recovery algorithms
    recover,
    recover_new,
    # Database operations
    create_compositions,
    create_users,
    fetch_all_compositions,
    get_composition_stats,
    # Service mapping
    build_service_connection_map,
    build_dataset_guid_map,
    # Composition building
    build_composition_for_task,
    normalize_composition,
    # Helper functions
    create_composition_node,
    add_task_link,
    normalize_dataset_id,
)

# Re-export for backwards compatibility
__all__ = [
    'recover',
    'recover_new',
    'create_compositions',
    'create_users',
    'fetch_all_compositions',
    'get_composition_stats',
    'build_service_connection_map',
    'build_dataset_guid_map',
    'build_composition_for_task',
    'normalize_composition',
    'create_composition_node',
    'add_task_link',
    'normalize_dataset_id',
]
