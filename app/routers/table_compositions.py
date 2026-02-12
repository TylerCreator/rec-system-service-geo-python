"""
Table compositions router
Endpoints for table-centric recovered workflows and sequential recommendations
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

    These are produced by GET /compositions/recoverNew (recover_new),
    which persists both Compositions and TableCompositions.
    """
    return await table_compositions_service.list_table_compositions(db=db, limit=limit, offset=offset)


@router.post("/predict")
async def predict_next(
    table_sequence: List[int] = Body(..., description="Current sequence of table/dataset IDs in the workflow"),
    n: int = Body(5, ge=1, le=20, description="Number of predictions to return"),
    db: AsyncSession = Depends(get_db),
):
    """
    Predict how to continue a table sequence.

    Given a sequence of table IDs (e.g. [table_1, table_2]),
    recommend the next service and/or next table to add to the workflow.

    Learned from TableCompositions extracted by /compositions/recoverNew.

    Example:
    ```json
    {
        "table_sequence": [1003093, 1003086],
        "n": 5
    }
    ```

    Returns predictions ranked by frequency:
    - next_service_mid: which service to call next
    - next_table_id: which table to add next
    - score: how many real compositions support this prediction
    """
    return await table_compositions_service.predict_next(
        db=db,
        table_sequence=table_sequence,
        n=n,
    )
