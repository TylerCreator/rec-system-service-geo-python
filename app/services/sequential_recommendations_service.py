"""
Sequential recommendations service
Handles workflow-based sequential service recommendations
"""
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.recommendations.algorithms import SequentialDAGNNAlgorithm
from app.services.recommendations.algorithms.sr_gnn import SRGNNAlgorithm
from app.services.recommendations.algorithms.dag_transformer import DAGTransformerAlgorithm


# Global instances registry
_sequential_engines: Dict[str, Any] = {}


def get_sequential_engine(db: AsyncSession, model_name: str = "dagnn") -> Any:
    """
    Get or create sequential recommendation engine
    
    Args:
        db: Database session
        model_name: "dagnn", "sr-gnn", or "dag-transformer"
        
    Returns:
        RecommendationAlgorithm instance
    """
    global _sequential_engines
    
    if model_name not in _sequential_engines:
        if model_name == "dagnn":
            _sequential_engines[model_name] = SequentialDAGNNAlgorithm(db=db)
        elif model_name == "sr-gnn":
            _sequential_engines[model_name] = SRGNNAlgorithm(db=db)
        elif model_name == "dag-transformer":
            _sequential_engines[model_name] = DAGTransformerAlgorithm(db=db)
        else:
            raise ValueError(f"Unknown sequential model: {model_name}")
            
    return _sequential_engines[model_name]


async def initialize_sequential_engine(db: AsyncSession):
    """
    Initialize all sequential recommendation engines
    
    Args:
        db: Database session
    """
    models = ["dagnn", "sr-gnn", "dag-transformer"]
    for model_name in models:
        engine = get_sequential_engine(db, model_name=model_name)
        if engine._load_model():
            print(f"✓ Sequential {model_name} model loaded from disk")
        else:
            print(f"⚠️  No saved {model_name} model found. Please train using /sequential/train")


async def predict_next_service(
    sequence: List[int],
    n: int = 5,
    db: Optional[AsyncSession] = None,
    model: str = "dagnn"
) -> Dict[str, Any]:
    """
    Predict next services in a workflow sequence
    
    Args:
        sequence: List of service IDs in current sequence
        n: Number of predictions
        db: Database session
        model: Algorithm to use ("dagnn", "sr-gnn", "dag-transformer")
        
    Returns:
        Predictions with scores
    """
    if db is None:
        raise ValueError("Database session required")
    
    engine = get_sequential_engine(db, model_name=model)
    
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
        "algorithm": model
    }


async def predict_next_service_ids_only(
    sequence: List[int],
    n: int = 5,
    db: Optional[AsyncSession] = None,
    model: str = "dagnn"
) -> List[int]:
    """
    Predict next services (IDs only)
    
    Args:
        sequence: List of service IDs in current sequence
        n: Number of predictions
        db: Database session
        model: Algorithm name
        
    Returns:
        List of service IDs
    """
    result = await predict_next_service(sequence, n, db, model)
    
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


async def train_sequential_model(db: AsyncSession, model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Train sequential recommendation models
    
    Args:
        db: Database session
        model_name: Optional, specific model to train. If None, trains all.
        
    Returns:
        Training status
    """
    try:
        models_to_train = [model_name] if model_name else ["dagnn", "sr-gnn", "dag-transformer"]
        results = []
        
        for name in models_to_train:
            engine = get_sequential_engine(db, model_name=name)
            await engine.train(data=db)
            results.append(engine.get_info())
        
        return {
            "success": True,
            "message": f"Sequential models ({', '.join(models_to_train)}) trained successfully",
            "model_info": results
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to train model: {str(e)}",
            "error": str(e)
        }


async def get_sequential_model_info(db: AsyncSession, model_name: str = "dagnn") -> Dict[str, Any]:
    """
    Get information about sequential model
    
    Args:
        db: Database session
        model_name: model to check
        
    Returns:
        Model information
    """
    engine = get_sequential_engine(db, model_name=model_name)
    return engine.get_info()


async def predict_next_table(
    table_sequence: List[int],
    n: int = 5,
    db: Optional[AsyncSession] = None,
    model: str = "dagnn"
) -> Dict[str, Any]:
    """
    Predict next tables in a dataset workflow sequence
    
    Analyzes only table-to-table connections in DAG,
    ignoring intermediate services.
    
    Args:
        table_sequence: List of table/dataset IDs in current sequence
        n: Number of predictions
        db: Database session
        model: Algorithm
        
    Returns:
        Predictions with scores
    """
    if db is None:
        raise ValueError("Database session required")
    
    engine = get_sequential_engine(db, model_name=model)
    
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
        "algorithm": model,
        "type": "table_recommendations"
    }


async def predict_next_table_ids_only(
    table_sequence: List[int],
    n: int = 5,
    db: Optional[AsyncSession] = None,
    model: str = "dagnn"
) -> List[int]:
    """
    Predict next tables (IDs only)
    
    Args:
        table_sequence: List of table IDs in current sequence
        n: Number of predictions
        db: Database session
        model: Algorithm
        
    Returns:
        List of table IDs
    """
    result = await predict_next_table(table_sequence, n, db, model)
    
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

