"""
Datasets service - business logic for dataset management
"""
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
import httpx

from app.models.models import Dataset
from app.core.config import settings


async def update_datasets(db: AsyncSession):
    """
    Update datasets from remote server
    """
    base_url = f"{settings.CRIS_BASE_URL}/dataset/list"
    request_data = {"sort": [{"fieldname": "id", "dir": False}]}
    
    # Get total count
    url = f"{base_url}?f=100&count_rows=true&iDisplayStart=0&iDisplayLength=1"
    
    async with httpx.AsyncClient(timeout=settings.API_TIMEOUT) as client:
        response = await client.post(url, json=request_data)
        response_data = response.json()
    
    total_records = int(response_data.get("iTotalDisplayRecords", 0))
    
    # Get current DB stats
    result = await db.execute(select(func.max(Dataset.id), func.min(Dataset.id)))
    max_id, min_id = result.one()
    
    print(f"minmax id DB: max={max_id}, min={min_id}")
    print(f"datasets on RS: {total_records}")
    
    result = await db.execute(select(func.count(Dataset.id)))
    db_count = result.scalar()
    print(f"Database datasets count: {db_count}")
    
    # Sync in batches
    display_length = 500
    i_display_start = 0
    counter = 1
    
    async with httpx.AsyncClient(timeout=settings.API_TIMEOUT) as client:
        while i_display_start < total_records:
            print(f"datasetData update counter {counter}")
            print(f"{i_display_start} ----------------------------------")
            
            url = f"{base_url}?f=100&count_rows=true&iDisplayStart={i_display_start}&iDisplayLength={display_length}"
            
            try:
                response = await client.post(url, json=request_data)
                data = response.json().get("aaData", [])
                
                if not data:
                    print("no data")
                    break
                
                # Process each dataset
                for item in data:
                    # Check if exists
                    result = await db.execute(
                        select(Dataset).where(Dataset.id == item.get("id"))
                    )
                    existing = result.scalar_one_or_none()
                    
                    if not existing:
                        dataset = Dataset(
                            id=item.get("id"),
                            guid=item.get("guid")
                        )
                        db.add(dataset)
                    else:
                        print("dataset already exist")
                
                await db.commit()
                
                i_display_start += display_length
                
                if len(data) < display_length:
                    print("datasets закончились")
                    break
                
                counter += 1
                
            except Exception as e:
                print(f"Error: {e}")
                await db.rollback()
                raise
    
    print("Data synchronization completed.")

