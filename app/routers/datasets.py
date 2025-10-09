"""
Datasets router - endpoints for dataset management
"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services import datasets_service

router = APIRouter()


@router.get("/update")
async def update_datasets(db: AsyncSession = Depends(get_db)):
    """
    Update datasets from remote server
    Synchronizes local database with remote CRIS server
    """
    await datasets_service.update_datasets(db)
    return {"message": "Datasets updated successfully"}

