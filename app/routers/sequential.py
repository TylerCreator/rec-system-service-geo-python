"""
Sequential recommendations router
Endpoints for workflow-based sequential service recommendations
"""
from typing import List
from fastapi import APIRouter, Depends, Query, Body
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services import sequential_recommendations_service

router = APIRouter()


@router.post("/predict")
async def predict_next_service(
    sequence: List[int] = Body(..., description="Current sequence of service IDs"),
    n: int = Body(5, ge=1, le=20, description="Number of predictions"),
    ids_only: bool = Body(False, description="Return only service IDs"),
    db: AsyncSession = Depends(get_db)
):
    """
    Predict next service(s) in a workflow sequence
    
    Uses DAGNN (Graph Neural Network) trained on composition DAG structure
    to predict the most likely next service based on the current sequence.
    
    Example:
    ```json
    {
        "sequence": [123, 456, 789],
        "n": 5,
        "ids_only": false
    }
    ```
    
    Returns:
    - ids_only=false: Full predictions with scores and confidence
    - ids_only=true: Simple array of service IDs
    
    This is for **strict sequence continuation** based on existing DAG connections.
    """
    if ids_only:
        return await sequential_recommendations_service.predict_next_service_ids_only(
            sequence=sequence,
            n=n,
            db=db
        )
    
    return await sequential_recommendations_service.predict_next_service(
        sequence=sequence,
        n=n,
        db=db
    )


@router.post("/possible")
async def get_possible_next_services(
    sequence: List[int] = Body(..., description="Current sequence of service IDs"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get possible next services based on DAG structure only (no ML prediction)
    
    Returns services that have direct connections in the composition DAG.
    This is faster than ML prediction and guarantees valid connections.
    
    Use this when you need:
    - Only valid/existing connections
    - Fast response without ML inference
    - Strict workflow validation
    
    Example:
    ```json
    {
        "sequence": [123, 456]
    }
    ```
    
    Returns list of service IDs that can follow the last service in DAG.
    """
    return await sequential_recommendations_service.get_possible_next_services(
        sequence=sequence,
        db=db
    )


@router.post("/train")
async def train_sequential_model(db: AsyncSession = Depends(get_db)):
    """
    Train sequential recommendation model
    
    This endpoint:
    1. Recovers compositions using recover_new()
    2. Builds DAG from compositions
    3. Trains DAGNN model on the DAG
    4. Saves model to disk
    
    Training takes ~2-5 minutes depending on data size.
    Model is saved and will be loaded automatically on next request.
    
    Should be called:
    - After initial setup
    - Daily (via cron or /update/full)
    - After significant composition changes
    """
    return await sequential_recommendations_service.train_sequential_model(db)


@router.get("/info")
async def get_model_info(db: AsyncSession = Depends(get_db)):
    """
    Get information about sequential recommendation model
    
    Returns:
    - Model status (trained/not trained)
    - DAG statistics (nodes, edges)
    - Model parameters
    - Training info
    """
    return await sequential_recommendations_service.get_sequential_model_info(db)


@router.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    """
    Health check for sequential recommendation system
    
    Returns status of model and DAG data.
    """
    engine = sequential_recommendations_service.get_sequential_engine(db)
    
    return {
        "status": "ready" if engine.is_trained else "not_trained",
        "model_loaded": engine.is_trained,
        "dag_available": engine.dag is not None,
        "nodes_count": len(engine.node_map) if engine.node_map else 0
    }


# ===== TABLE-BASED SEQUENTIAL RECOMMENDATIONS =====

@router.post("/tables/predict")
async def predict_next_table(
    table_sequence: List[int] = Body(..., description="Current sequence of table/dataset IDs"),
    n: int = Body(5, ge=1, le=20, description="Number of predictions"),
    ids_only: bool = Body(False, description="Return only table IDs"),
    db: AsyncSession = Depends(get_db)
):
    """
    Predict next table (dataset) in a workflow sequence
    
    Analyzes ONLY table-to-table connections in the DAG,
    ignoring intermediate service nodes in the path.
    
    Use case:
    - User selected tables: [table_123, table_456]
    - System predicts next table: table_789
    - Ignores which services connect them
    
    Example:
    ```json
    {
        "table_sequence": [1002120, 1001211],
        "n": 5,
        "ids_only": false
    }
    ```
    
    Returns:
    - ids_only=false: Full predictions with scores, distance, frequency
    - ids_only=true: Simple array of table IDs
    
    The algorithm considers:
    1. DAGNN model predictions (ML)
    2. Distance in DAG (shortest path between tables)
    3. Frequency of table usage
    """
    if ids_only:
        return await sequential_recommendations_service.predict_next_table_ids_only(
            table_sequence=table_sequence,
            n=n,
            db=db
        )
    
    return await sequential_recommendations_service.predict_next_table(
        table_sequence=table_sequence,
        n=n,
        db=db
    )


@router.post("/tables/possible")
async def get_possible_next_tables(
    table_sequence: List[int] = Body(..., description="Current sequence of table IDs"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get possible next tables based on DAG structure only (no ML)
    
    Returns tables that are reachable from the last table in sequence
    through any path in the composition DAG (may go through services).
    
    Use this when you need:
    - Only valid/existing table connections
    - Fast response without ML inference
    - Strict workflow validation for tables
    
    Example:
    ```json
    {
        "table_sequence": [1002120, 1001211]
    }
    ```
    
    Returns list of table IDs that can follow the last table.
    """
    return await sequential_recommendations_service.get_possible_next_tables(
        table_sequence=table_sequence,
        db=db
    )

