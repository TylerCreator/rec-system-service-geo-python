"""
Sequential recommendations service
Handles workflow-based sequential service recommendations
"""
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.recommendations.algorithms import SequentialDAGNNAlgorithm


# Global instance
_sequential_engine: Optional[SequentialDAGNNAlgorithm] = None


def get_sequential_engine(db: AsyncSession) -> SequentialDAGNNAlgorithm:
    """
    Get or create sequential recommendation engine
    
    Args:
        db: Database session
        
    Returns:
        SequentialDAGNNAlgorithm instance
    """
    global _sequential_engine
    
    if _sequential_engine is None:
        _sequential_engine = SequentialDAGNNAlgorithm(db=db)
    
    return _sequential_engine


async def initialize_sequential_engine(db: AsyncSession):
    """
    Initialize sequential recommendation engine
    
    Args:
        db: Database session
    """
    engine = get_sequential_engine(db)
    
    # Try to load saved model first
    if engine._load_model():
        print("✓ Sequential DAGNN model loaded from disk")
    else:
        print("⚠️  No saved model found. Please train using /sequential/train")


async def predict_next_service(
    sequence: List[int],
    n: int = 5,
    db: Optional[AsyncSession] = None
) -> Dict[str, Any]:
    """
    Predict next services in a workflow sequence
    
    Args:
        sequence: List of service IDs in current sequence
        n: Number of predictions
        db: Database session
        
    Returns:
        Predictions with scores
    """
    if db is None:
        raise ValueError("Database session required")
    
    engine = get_sequential_engine(db)
    
    if not engine.is_trained:
        # Try to load model
        if not engine._load_model():
            return {
                "error": "Model not trained",
                "message": "Please train the model first using POST /sequential/train",
                "sequence": sequence
            }
    
    # Get predictions
    recommendations = engine.predict_next(sequence=sequence, n=n)
    
    return {
        "sequence": sequence,
        "next_services": [
            {
                "service_id": rec.service_id,
                "score": rec.score,
                "confidence": rec.confidence,
                "reason": rec.reason,
                "metadata": rec.metadata
            }
            for rec in recommendations
        ],
        "count": len(recommendations),
        "algorithm": "sequential_dagnn"
    }


async def predict_next_service_ids_only(
    sequence: List[int],
    n: int = 5,
    db: Optional[AsyncSession] = None
) -> List[int]:
    """
    Predict next services (IDs only)
    
    Args:
        sequence: List of service IDs in current sequence
        n: Number of predictions
        db: Database session
        
    Returns:
        List of service IDs
    """
    result = await predict_next_service(sequence, n, db)
    
    if "error" in result:
        return []
    
    return [rec["service_id"] for rec in result["next_services"]]


async def get_possible_next_services(
    sequence: List[int],
    db: Optional[AsyncSession] = None
) -> Dict[str, Any]:
    """
    Get possible next services based on DAG structure only (no ML)
    
    Args:
        sequence: List of service IDs in current sequence
        db: Database session
        
    Returns:
        Possible next services from DAG
    """
    if db is None:
        raise ValueError("Database session required")
    
    engine = get_sequential_engine(db)
    
    if not engine.is_trained and not engine._load_model():
        return {
            "error": "Model not loaded",
            "message": "DAG data not available",
            "sequence": sequence
        }
    
    # Get possible services from DAG
    possible_services = engine.get_possible_next_services(sequence)
    
    return {
        "sequence": sequence,
        "possible_next_services": possible_services,
        "count": len(possible_services),
        "source": "dag_structure"
    }


async def train_sequential_model(db: AsyncSession) -> Dict[str, Any]:
    """
    Train sequential recommendation model
    
    Args:
        db: Database session
        
    Returns:
        Training status
    """
    try:
        engine = get_sequential_engine(db)
        
        # Train model
        await engine.train(data=db)
        
        return {
            "success": True,
            "message": "Sequential model trained successfully",
            "model_info": engine.get_info()
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to train model: {str(e)}",
            "error": str(e)
        }


async def get_sequential_model_info(db: AsyncSession) -> Dict[str, Any]:
    """
    Get information about sequential model
    
    Args:
        db: Database session
        
    Returns:
        Model information
    """
    engine = get_sequential_engine(db)
    return engine.get_info()


async def predict_next_table(
    table_sequence: List[int],
    n: int = 5,
    db: Optional[AsyncSession] = None
) -> Dict[str, Any]:
    """
    Predict next tables in a dataset workflow sequence
    
    Analyzes only table-to-table connections in DAG,
    ignoring intermediate services.
    
    Args:
        table_sequence: List of table/dataset IDs in current sequence
        n: Number of predictions
        db: Database session
        
    Returns:
        Predictions with scores
    """
    if db is None:
        raise ValueError("Database session required")
    
    engine = get_sequential_engine(db)
    
    if not engine.is_trained:
        if not engine._load_model():
            return {
                "error": "Model not trained",
                "message": "Please train the model first using POST /sequential/train",
                "table_sequence": table_sequence
            }
    
    # Get predictions
    recommendations = engine.predict_next_table(table_sequence=table_sequence, n=n)
    
    return {
        "table_sequence": table_sequence,
        "next_tables": [
            {
                "table_id": rec.service_id,  # Note: service_id field contains table_id for tables
                "score": rec.score,
                "confidence": rec.confidence,
                "reason": rec.reason,
                "metadata": rec.metadata
            }
            for rec in recommendations
        ],
        "count": len(recommendations),
        "algorithm": "sequential_dagnn",
        "type": "table_recommendations"
    }


async def predict_next_table_ids_only(
    table_sequence: List[int],
    n: int = 5,
    db: Optional[AsyncSession] = None
) -> List[int]:
    """
    Predict next tables (IDs only)
    
    Args:
        table_sequence: List of table IDs in current sequence
        n: Number of predictions
        db: Database session
        
    Returns:
        List of table IDs
    """
    result = await predict_next_table(table_sequence, n, db)
    
    if "error" in result:
        return []
    
    return [rec["table_id"] for rec in result["next_tables"]]


async def get_possible_next_tables(
    table_sequence: List[int],
    db: Optional[AsyncSession] = None
) -> Dict[str, Any]:
    """
    Get possible next tables based on DAG structure only (no ML)
    
    Args:
        table_sequence: List of table IDs in current sequence
        db: Database session
        
    Returns:
        Possible next tables from DAG
    """
    if db is None:
        raise ValueError("Database session required")
    
    engine = get_sequential_engine(db)
    
    if not engine.is_trained and not engine._load_model():
        return {
            "error": "Model not loaded",
            "message": "DAG data not available",
            "table_sequence": table_sequence
        }
    
    # Get possible tables from DAG
    possible_tables = engine.get_possible_next_tables(table_sequence)
    
    return {
        "table_sequence": table_sequence,
        "possible_next_tables": possible_tables,
        "count": len(possible_tables),
        "source": "dag_structure",
        "type": "table_recommendations"
    }

