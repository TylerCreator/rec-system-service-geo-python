"""
Service connection map building
"""
import json
import os
from typing import Dict, Any
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Service, Dataset
from app.services.utils.constants import WIDGET_FILE, WIDGET_FILE_SAVE, WIDGET_THEME_SELECT
from app.services.utils.parsers import parse_service_params
from app.services.utils.validators import categorize_params


async def build_service_connection_map(db: AsyncSession) -> Dict[int, Any]:
    """
    Build map of service connections (inputs/outputs)
    Extended with data from in_and_out_settings.json file if exists
    
    Args:
        db: Database session
    
    Returns:
        Dictionary mapping service ID to connection info
    """
    print("Building service connection map...")
    
    # Load data from file
    in_and_out_file = "app/static/in_and_out_settings.json"
    file_data = {}
    
    if os.path.exists(in_and_out_file):
        try:
            print(f"Loading data from {in_and_out_file}...")
            with open(in_and_out_file, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
            print(f"Loaded {len(file_data)} services from file")
        except Exception as e:
            print(f"Warning: Could not load {in_and_out_file}: {e}")
            file_data = {}
    
    result = await db.execute(select(Service))
    services = result.scalars().all()
    
    in_and_out = {}
    
    for service in services:
        params = parse_service_params(service.params)
        output_params = parse_service_params(service.output_params)
        
        input_categorized = categorize_params(
            params,
            [WIDGET_FILE, WIDGET_THEME_SELECT]
        )
        
        output_categorized = categorize_params(
            output_params,
            [],
            [WIDGET_FILE_SAVE]
        )
        
        # Check if service has data in file
        service_id_str = str(service.id)
        if service_id_str in file_data:
            # Merge data from DB with data from file
            file_service_data = file_data[service_id_str]
            
            # Merge input (file data extends DB data)
            combined_input = input_categorized["internal"].copy()
            if "input" in file_service_data:
                combined_input.update(file_service_data["input"])
            
            # Merge output (file data extends DB data)
            combined_output = output_categorized["internal"].copy()
            if "output" in file_service_data:
                combined_output.update(file_service_data["output"])
            
            in_and_out[service.id] = {
                "type": service.type,
                "name": service.name,
                "input": combined_input,
                "externalInput": input_categorized["external"],
                "output": combined_output,
                "externalOutput": output_categorized["external"]
            }
        else:
            # Use only DB data
            in_and_out[service.id] = {
                "type": service.type,
                "name": service.name,
                "input": input_categorized["internal"],
                "externalInput": input_categorized["external"],
                "output": output_categorized["internal"],
                "externalOutput": output_categorized["external"]
            }
    
    print(f"Service connection map built: {len(in_and_out)} services")
    return in_and_out


async def build_dataset_guid_map(db: AsyncSession) -> Dict[str, int]:
    """
    Build GUID to ID mapping for datasets
    
    Args:
        db: Database session
    
    Returns:
        Dictionary mapping dataset GUID to ID
    """
    print("Building dataset GUID to ID mapping...")
    
    result = await db.execute(select(Dataset))
    datasets = result.scalars().all()
    
    guid_map = {}
    for dataset in datasets:
        if dataset.guid:
            guid_map[dataset.guid] = dataset.id
    
    return guid_map

