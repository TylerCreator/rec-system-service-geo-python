"""
Calls service - business logic for service calls
"""
import csv
from typing import List, Dict, Any
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
import httpx

from app.models.models import Call
from app.core.config import settings
from app.services.utils.parsers import parse_datetime, to_string


async def get_calls(db: AsyncSession) -> List[Dict[str, Any]]:
    """Get all calls from database"""
    print("Getting calls from database...")
    
    result = await db.execute(
        select(Call).order_by(Call.id.desc())
    )
    calls = result.scalars().all()
    
    print(f"Call data from the database: {len(calls)}")
    
    return [
        {
            "id": call.id,
            "classname": call.classname,
            "console_output": call.console_output,
            "created_by": call.created_by,
            "created_on": call.created_on.isoformat() if call.created_on else None,
            "edited_by": call.edited_by,
            "edited_on": call.edited_on.isoformat() if call.edited_on else None,
            "end_time": call.end_time.isoformat() if call.end_time else None,
            "error_output": call.error_output,
            "input": call.input,
            "input_data": call.input_data,
            "input_params": call.input_params,
            "is_deleted": call.is_deleted,
            "mid": call.mid,
            "os_pid": call.os_pid,
            "owner": call.owner,
            "result": call.result,
            "start_time": call.start_time.isoformat() if call.start_time else None,
            "status": call.status,
        }
        for call in calls
    ]


async def incr_calls(db: AsyncSession) -> Dict[str, Any]:
    """Incremental calls update (legacy)"""
    url = f"{settings.CRIS_BASE_URL}/dataset/list?f=186&count_rows=true&unique=undefined&count_rows=1&iDisplayStart=0&iDisplayLength=100"
    
    async with httpx.AsyncClient(timeout=settings.API_TIMEOUT) as client:
        response = await client.get(url)
        data = response.json()
    
    calls_data = data.get("aaData", [])
    
    # Bulk insert using SQLAlchemy
    for item in calls_data:
        call = Call(
            id=item.get("id"),
            classname=item.get("classname"),
            console_output=item.get("console_output"),
            created_by=item.get("created_by"),
            created_on=parse_datetime(item.get("created_on")),
            edited_by=item.get("edited_by"),
            edited_on=parse_datetime(item.get("edited_on")),
            end_time=parse_datetime(item.get("end_time")),
            error_output=item.get("error_output"),
            input=item.get("input"),
            input_data=item.get("input_data"),
            input_params=item.get("input_params"),
            is_deleted=to_string(item.get("is_deleted")),
            mid=item.get("mid"),
            os_pid=item.get("os_pid"),
            owner=item.get("owner"),
            result=item.get("result"),
            start_time=parse_datetime(item.get("start_time")),
            status=item.get("status"),
        )
        db.add(call)
    
    await db.commit()
    print("Data saved to the database.")
    
    return {"data": calls_data}


async def update_calls(db: AsyncSession):
    """
    Update calls from remote server
    Synchronizes local database with remote CRIS server
    """
    base_url = f"{settings.CRIS_BASE_URL}/dataset/list"
    request_data = {
        "sort": [{"fieldname": "id", "dir": False}]
    }
    
    # Get total records count
    url = f"{base_url}?f=186&count_rows=true&iDisplayStart=0&iDisplayLength=1"
    
    async with httpx.AsyncClient(timeout=settings.API_TIMEOUT) as client:
        response = await client.post(url, json=request_data)
        response_data = response.json()
    
    total_records = int(response_data.get("iTotalDisplayRecords", 0))
    
    # Get current database stats
    result = await db.execute(select(func.max(Call.id), func.min(Call.id)))
    max_id, min_id = result.one()
    
    print(f"minmax id DB: max={max_id}, min={min_id}")
    print(f"calls on RS: {total_records}")
    
    result = await db.execute(select(func.count(Call.id)))
    db_count = result.scalar()
    print(f"Database calls count: {db_count}")
    
    # Sync data in batches
    display_length = 500
    i_display_start = 0
    counter = 1
    
    async with httpx.AsyncClient(timeout=settings.API_TIMEOUT) as client:
        while i_display_start < total_records:
            print(f"calls update counter {counter}")
            print(f"{i_display_start} ----------------------------------")
            
            url = f"{base_url}?f=186&count_rows=true&iDisplayStart={i_display_start}&iDisplayLength={display_length}"
            
            try:
                response = await client.post(url, json=request_data)
                data = response.json().get("aaData", [])
                
                if not data:
                    print("no data")
                    break
                
                # Process each call
                for item in data:
                    # Check if call exists
                    result = await db.execute(
                        select(Call).where(Call.id == item.get("id"))
                    )
                    existing_call = result.scalar_one_or_none()
                    
                    if not existing_call:
                        # Create new call
                        call = Call(
                            id=item.get("id"),
                            classname=item.get("classname"),
                            console_output=item.get("console_output"),
                            created_by=item.get("created_by"),
                            created_on=parse_datetime(item.get("created_on")),
                            edited_by=item.get("edited_by"),
                            edited_on=parse_datetime(item.get("edited_on")),
                            end_time=parse_datetime(item.get("end_time")),
                            error_output=item.get("error_output"),
                            input=item.get("input"),
                            input_data=item.get("input_data"),
                            input_params=item.get("input_params"),
                            is_deleted=to_string(item.get("is_deleted")),
                            mid=item.get("mid"),
                            os_pid=item.get("os_pid"),
                            owner=item.get("owner"),
                            result=item.get("result"),
                            start_time=parse_datetime(item.get("start_time")),
                            status=item.get("status"),
                        )
                        db.add(call)
                
                await db.commit()
                
                i_display_start += display_length
                
                if len(data) < display_length:
                    print("calls закончились")
                    break
                    
                counter += 1
                
            except Exception as e:
                print(f"Error: {e}")
                await db.rollback()
                raise
    
    print("Data synchronization completed.")


async def dump_csv(db: AsyncSession):
    """Export calls to CSV file"""
    print("dump-csv")
    
    result = await db.execute(select(Call))
    calls = result.scalars().all()
    
    print(f"all found: {len(calls)}")
    
    # Write to CSV
    with open(settings.CSV_FILE_PATH, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'mid', 'owner', 'start_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
        
        writer.writeheader()
        for call in calls:
            writer.writerow({
                'id': call.id,
                'mid': call.mid,
                'owner': call.owner,
                'start_time': call.start_time.isoformat() if call.start_time else ''
            })
    
    print(f"Write to {settings.CSV_FILE_PATH} successfully!")

