"""
Calls router - endpoints for service call management
"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services import calls_service

router = APIRouter()


@router.get("/")
async def get_calls(db: AsyncSession = Depends(get_db)):
    """
    Get all service calls
    Returns list of all calls in descending order by ID
    """
    return await calls_service.get_calls(db)


@router.get("/incr")
async def incr_calls(db: AsyncSession = Depends(get_db)):
    """
    Incremental calls update (legacy endpoint)
    """
    return await calls_service.incr_calls(db)


@router.get("/update-calls")
async def update_calls(db: AsyncSession = Depends(get_db)):
    """
    Update calls from remote server
    Synchronizes local database with remote CRIS server
    """
    await calls_service.update_calls(db)
    return {"message": "Calls updated successfully"}


@router.get("/dump-csv")
async def dump_csv(db: AsyncSession = Depends(get_db)):
    """
    Export calls to CSV file
    Creates calls.csv with id, mid, owner, start_time columns
    """
    await calls_service.dump_csv(db)
    return {"message": "CSV file created successfully", "status": 200}

