"""
Table compositions router
Endpoints for table-centric recovered workflows and substitution recommendations
"""

from typing import Optional
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
    which now persists both Compositions and TableCompositions.
    """
    return await table_compositions_service.list_table_compositions(db=db, limit=limit, offset=offset)


@router.post("/recommend/substitute-table")
async def recommend_substitute_table(
    upstream_service_id: int = Body(..., description="Service mid that produced the existing artifact/output"),
    new_table_id: int = Body(..., description="New table/dataset id to substitute into the workflow"),
    existing_table_id: Optional[int] = Body(None, description="Optional existing table id already in the workflow"),
    n: int = Body(5, ge=1, le=20, description="Number of recommendations"),
    db: AsyncSession = Depends(get_db),
):
    """
    Recommend a service chain for the scenario "substitute a new table into an existing workflow".

    The model is learned from TableCompositions extracted from call logs.
    """
    return await table_compositions_service.recommend_substitute_table(
        db=db,
        upstream_service_id=upstream_service_id,
        new_table_id=new_table_id,
        existing_table_id=existing_table_id,
        n=n,
    )

