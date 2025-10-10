"""
Database operations for compositions
"""
from typing import Dict, List, Any
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Composition, User


async def create_compositions(db: AsyncSession, compositions: List[Dict]) -> None:
    """
    Save compositions to database
    
    Args:
        db: Database session
        compositions: List of composition dictionaries
    """
    try:
        new_count = 0
        existing_count = 0
        
        for composition_data in compositions:
            # Check if exists
            result = await db.execute(
                select(Composition).where(Composition.id == composition_data["id"])
            )
            existing = result.scalar_one_or_none()
            
            if not existing:
                composition = Composition(
                    id=composition_data["id"],
                    nodes=composition_data["nodes"],
                    links=composition_data["links"]
                )
                db.add(composition)
                new_count += 1
            else:
                existing_count += 1
        
        await db.commit()
        print(f"Compositions saved: {new_count} new, {existing_count} already existed")
    except Exception as e:
        print(f"Error saving compositions: {e}")
        await db.rollback()
        raise


async def create_users(db: AsyncSession, users: Dict) -> None:
    """
    Create users from composition analysis
    
    Args:
        db: Database session
        users: Dictionary of user IDs
    """
    try:
        for user_id in users.keys():
            result = await db.execute(
                select(User).where(User.id == user_id)
            )
            existing = result.scalar_one_or_none()
            
            if not existing:
                user = User(id=user_id)
                db.add(user)
        
        await db.commit()
        print("Users successfully created")
    except Exception as e:
        print(f"Error creating users: {e}")
        await db.rollback()


async def fetch_all_compositions(db: AsyncSession) -> List[Dict[str, Any]]:
    """
    Get all compositions from database
    
    Args:
        db: Database session
    
    Returns:
        List of composition dictionaries
    """
    result = await db.execute(select(Composition))
    compositions = result.scalars().all()
    
    return [
        {
            "id": comp.id,
            "nodes": comp.nodes,
            "links": comp.links
        }
        for comp in compositions
    ]


async def get_composition_stats(db: AsyncSession) -> Dict[str, Any]:
    """
    Get composition statistics and build graph
    
    Args:
        db: Database session
    
    Returns:
        Dictionary with composition statistics
    """
    compositions = await fetch_all_compositions(db)
    
    service_stats = {}
    total_nodes = 0
    
    for comp in compositions:
        nodes = comp["nodes"]
        total_nodes += len(nodes)
        
        for node in nodes:
            mid = node.get("mid")
            if mid:
                if mid not in service_stats:
                    service_stats[mid] = {
                        "count": 0,
                        "service_name": node.get("service")
                    }
                service_stats[mid]["count"] += 1
    
    # Sort by usage
    top_services = sorted(
        [
            {"mid": mid, "name": info["service_name"], "usageCount": info["count"]}
            for mid, info in service_stats.items()
        ],
        key=lambda x: x["usageCount"],
        reverse=True
    )[:10]
    
    return {
        "totalCompositions": len(compositions),
        "totalNodes": total_nodes,
        "avgNodesPerComposition": total_nodes / len(compositions) if compositions else 0,
        "topServices": top_services
    }

