"""
Table compositions router
Endpoints for table-centric recovered workflows and recommendations
"""

from typing import List
from fastapi import APIRouter, Depends, Query, Body
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services import table_compositions_service

router = APIRouter()


@router.get("/")
async def list_table_compositions(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """
    List TableCompositions stored in DB.
    Produced by GET /compositions/recover (recover).
    """
    return await table_compositions_service.list_table_compositions(db=db, limit=limit, offset=offset)


@router.post("/train")
async def train(db: AsyncSession = Depends(get_db)):
    """
    Train table recommendation model from TableCompositions.
    
    Builds a transition model (Markov chain) from table_ids sequences.
    Must be called after /compositions/recover populates TableCompositions.
    """
    return await table_compositions_service.train(db=db)


@router.post("/predict")
async def predict_next(
    table_sequence: List[int] = Body(..., description="Current sequence of table/dataset IDs"),
    n: int = Body(5, ge=1, le=20, description="Number of predictions"),
    db: AsyncSession = Depends(get_db),
):
    """
    Predict next table in a workflow sequence.
    
    Uses the Markov chain model trained on TableCompositions.
    Auto-trains if not trained yet.

    Example:
    ```json
    {"table_sequence": [1003284, 1002118], "n": 5}
    ```
    """
    return await table_compositions_service.predict_next(
        db=db,
        table_sequence=table_sequence,
        n=n,
    )
