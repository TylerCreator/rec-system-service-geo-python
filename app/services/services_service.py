"""
Services service - business logic for service management and recommendations
"""
import json
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
import httpx

from app.models.models import Service, Call, User, UserService, Dataset
from app.core.config import settings
from app.services.utils.parsers import parse_datetime, to_string


async def get_services(
    db: AsyncSession,
    user: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Get all services, optionally filtered by user"""
    print("Getting services")
    
    if user:
        # Get services for specific user with call counts
        query = (
            select(Service, UserService.number_of_calls)
            .join(UserService, Service.id == UserService.service_id)
            .where(UserService.user_id == user)
            .order_by(UserService.number_of_calls.desc())
        )
        
        if limit:
            query = query.limit(limit)
        
        result = await db.execute(query)
        services_with_counts = result.all()
        
        return [
            {
                **_service_to_dict(service),
                "number_of_calls": count
            }
            for service, count in services_with_counts
        ]
    else:
        # Get all services
        result = await db.execute(
            select(Service).order_by(Service.id.desc())
        )
        services = result.scalars().all()
        
        return [_service_to_dict(service) for service in services]


async def update_services(db: AsyncSession):
    """Update services from remote server"""
    base_url = f"{settings.CRIS_BASE_URL}/dataset/list"
    request_data = {"sort": [{"fieldname": "id", "dir": False}]}
    
    # Get total count
    url = f"{base_url}?f=185&count_rows=true&iDisplayStart=0&iDisplayLength=1"
    
    async with httpx.AsyncClient(timeout=settings.API_TIMEOUT) as client:
        response = await client.post(url, json=request_data)
        response_data = response.json()
    
    total_records = int(response_data.get("iTotalDisplayRecords", 0))
    
    # Get current DB stats
    result = await db.execute(select(func.max(Service.id), func.min(Service.id)))
    max_id, min_id = result.one()
    print(f"Max ID: {max_id}, Min ID: {min_id}")
    
    # Sync in batches
    display_length = 100
    i_display_start = 0
    counter = 1
    
    async with httpx.AsyncClient(timeout=settings.API_TIMEOUT) as client:
        while i_display_start < total_records:
            print(f"services update counter {counter}")
            
            url = f"{base_url}?f=185&count_rows=true&unique=undefined&count_rows=1&iDisplayStart={i_display_start}&iDisplayLength={i_display_start + display_length}"
            
            response = await client.post(url, json=request_data)
            data = response.json().get("aaData", [])
            
            if not data:
                print("services data empty")
                break
            
            for item in data:
                # Check if exists
                result = await db.execute(
                    select(Service).where(Service.id == item.get("id"))
                )
                existing = result.scalar_one_or_none()
                
                if not existing:
                    service = Service(
                        id=item.get("id"),
                        name=item.get("name"),
                        subject=item.get("subject"),
                        type=item.get("type"),
                        description=item.get("description"),
                        actionview=item.get("actionview"),
                        actionmodify=item.get("actionmodify"),
                        map_reduce_specification=item.get("map_reduce_specification"),
                        params=item.get("params"),
                        js_body=item.get("js_body"),
                        wpsservers=item.get("wpsservers"),
                        wpsmethod=item.get("wpsmethod"),
                        status=item.get("status"),
                        output_params=item.get("output_params"),
                        wms_link=item.get("wms_link"),
                        wms_layer_name=item.get("wms_layer_name"),
                        is_deleted=to_string(item.get("is_deleted")),
                        created_by=item.get("created_by"),
                        edited_by=item.get("edited_by"),
                        edited_on=parse_datetime(item.get("edited_on")),
                        created_on=parse_datetime(item.get("created_on")),
                        classname=item.get("classname"),
                    )
                    db.add(service)
            
            await db.commit()
            
            i_display_start += display_length
            
            if len(data) < display_length:
                print("services completed")
                break
            
            counter += 1
    
    print("Data synchronization completed.")


async def get_recomendations(user_id: str) -> Dict[str, Any]:
    """
    Get real-time recommendations using KNN script
    """
    print("Running KNN recommendations...")
    
    # Run Python KNN script
    proc = await asyncio.create_subprocess_exec(
        "python3",
        settings.KNN_SCRIPT_PATH,
        settings.CSV_FILE_PATH,
        user_id,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await proc.communicate()
    
    if proc.returncode != 0:
        print(f"KNN script error: {stderr.decode()}")
        raise Exception(f"KNN script failed: {stderr.decode()}")
    
    # Parse output
    output = stdout.decode()
    result = json.loads(output)
    
    return result


async def get_recomendation(user_id: Optional[str] = None) -> List[int]:
    """
    Get recommendations from pre-computed file
    """
    print("Getting recommendation from file")
    
    try:
        with open(settings.RECOMMENDATIONS_FILE_PATH, 'r', encoding='utf-8') as f:
            recommendations = json.load(f)
        
        if user_id and user_id in recommendations.get('prediction', {}):
            return recommendations['prediction'][user_id]
        else:
            return []
    except Exception as e:
        print(f"Error reading recommendations: {e}")
        return []


async def get_popular_services(
    db: AsyncSession,
    type: str = "any",
    limit: int = 20,
    period: str = "all",
    min_calls: int = 1,
    user_id: Optional[str] = None,
    ids_only: bool = False
) -> Dict[str, Any]:
    """
    Get most popular services with various filters
    """
    print(f"Getting popular services: type={type}, limit={limit}, period={period}, user_id={user_id or 'all users'}")
    
    # Build time condition
    time_condition = None
    if period != "all":
        now = datetime.utcnow()
        if period == "week":
            start_date = now - timedelta(days=7)
        elif period == "month":
            start_date = now - timedelta(days=30)
        elif period == "year":
            start_date = now - timedelta(days=365)
        else:
            start_date = None
        
        if start_date:
            time_condition = Call.start_time >= start_date
    
    # Build user condition
    user_condition = None
    if user_id:
        user_condition = Call.owner == user_id
    
    # Build WHERE clause
    where_conditions = [Call.status == "TASK_SUCCEEDED"]
    if time_condition is not None:
        where_conditions.append(time_condition)
    if user_condition is not None:
        where_conditions.append(user_condition)
    
    # Get all successful calls
    result = await db.execute(
        select(Call.mid, Call.owner).where(and_(*where_conditions))
    )
    calls = result.all()
    
    # Group statistics manually
    service_stats = {}
    for mid, owner in calls:
        if mid not in service_stats:
            service_stats[mid] = {
                "mid": mid,
                "call_count": 0,
                "unique_users": set()
            }
        service_stats[mid]["call_count"] += 1
        service_stats[mid]["unique_users"].add(owner)
    
    # Convert to list and filter by min_calls
    call_stats = [
        {
            "mid": stat["mid"],
            "call_count": stat["call_count"],
            "unique_users": len(stat["unique_users"])
        }
        for stat in service_stats.values()
        if stat["call_count"] >= min_calls
    ]
    
    # Sort by call count
    call_stats.sort(key=lambda x: x["call_count"], reverse=True)
    call_stats = call_stats[:limit * 3]  # Take more for filtering by type
    
    # Separate service IDs and table IDs
    service_ids = [stat["mid"] for stat in call_stats if stat["mid"] < 1000000]
    table_ids = [stat["mid"] - 1000000 for stat in call_stats if stat["mid"] >= 1000000]
    
    # Fetch services and datasets
    services = []
    datasets = []
    
    if type in ["any", "service"] and service_ids:
        result = await db.execute(
            select(Service).where(Service.id.in_(service_ids))
        )
        services = result.scalars().all()
    
    if type in ["any", "table", "dataset"] and table_ids:
        result = await db.execute(
            select(Dataset).where(Dataset.id.in_(table_ids))
        )
        datasets = result.scalars().all()
    
    # Create service map
    service_map = {}
    
    for service in services:
        service_map[service.id] = {
            "id": service.id,
            "name": service.name or f"Service {service.id}",
            "type": service.type or "service",
            "description": service.description or "",
            "subject": service.subject or "",
            "itemType": "service"
        }
    
    for dataset in datasets:
        service_map[dataset.id + 1000000] = {
            "id": dataset.id,
            "name": dataset.guid or f"Dataset {dataset.id}",
            "type": "table",
            "description": f"Dataset with GUID: {dataset.guid or 'Unknown'}",
            "subject": "",
            "itemType": "dataset"
        }
    
    # Filter and apply limit
    filtered_stats = [
        stat for stat in call_stats
        if stat["mid"] in service_map
    ][:limit]
    
    # If ids_only, return simple array
    if ids_only:
        return [stat["mid"] for stat in filtered_stats]
    
    # Build popular services response
    popular_services = []
    for i, stat in enumerate(filtered_stats):
        item = service_map[stat["mid"]]
        is_dataset = stat["mid"] >= 1000000
        
        popular_services.append({
            "itemId": stat["mid"],
            "originalId": stat["mid"] - 1000000 if is_dataset else stat["mid"],
            "itemName": item["name"],
            "itemType": item["itemType"],
            "serviceType": item["type"],
            "itemDescription": item["description"],
            "itemSubject": item["subject"],
            "callCount": stat["call_count"],
            "uniqueUsers": stat["unique_users"],
            "popularity": round(stat["call_count"] / max(stat["unique_users"], 1), 2),
            "rank": i + 1,
            # Backward compatibility
            "serviceId": stat["mid"],
            "serviceName": item["name"]
        })
    
    # Get additional statistics
    total_services_result = await db.execute(select(func.count(Service.id)))
    total_services = total_services_result.scalar()
    
    total_datasets_result = await db.execute(select(func.count(Dataset.id)))
    total_datasets = total_datasets_result.scalar()
    
    return {
        "items": popular_services,
        "services": popular_services,  # Backward compatibility
        "meta": {
            "total_services_in_db": total_services,
            "total_datasets_in_db": total_datasets,
            "total_items_in_db": total_services + total_datasets,
            "total_successful_calls": len(calls),
            "filtered_by_type": type,
            "time_period": period,
            "filtered_by_user": user_id,
            "user_specific": bool(user_id),
            "min_calls_threshold": min_calls,
            "limit": limit,
            "returned_count": len(popular_services),
            "breakdown": {
                "services": sum(1 for item in popular_services if item["itemType"] == "service"),
                "datasets": sum(1 for item in popular_services if item["itemType"] == "dataset")
            },
            "time_filter_applied": period != "all",
            "generated_at": datetime.utcnow().isoformat()
        }
    }


def _deep_parse_json(data):
    """Deep parse JSON with nested JSON strings"""
    if isinstance(data, dict):
        return data
    
    if not isinstance(data, str):
        return data
    
    try:
        parsed = json.loads(data)
        
        if isinstance(parsed, dict):
            deep_parsed = {}
            for key, value in parsed.items():
                if isinstance(value, str):
                    try:
                        inner_parsed = json.loads(value)
                        if isinstance(inner_parsed, (dict, list)):
                            deep_parsed[key] = _deep_parse_json(inner_parsed)
                        else:
                            deep_parsed[key] = inner_parsed
                    except:
                        deep_parsed[key] = value
                elif isinstance(value, (dict, list)):
                    deep_parsed[key] = _deep_parse_json(value)
                else:
                    deep_parsed[key] = value
            
            return deep_parsed
        
        return parsed
    except:
        return data


async def get_service_parameters(
    db: AsyncSession,
    service_id: int,
    user: Optional[str] = None,
    limit: int = 100,
    unique: str = "true"
) -> Dict[str, Any]:
    """
    Get service parameters history
    """
    print(f"Getting parameters for service {service_id}")
    
    # Check if service exists
    result = await db.execute(
        select(Service).where(Service.id == service_id)
    )
    service = result.scalar_one_or_none()
    
    if not service:
        return {
            "error": "Service not found",
            "serviceId": service_id
        }
    
    # Build query
    where_conditions = [
        Call.mid == service_id,
        Call.input.isnot(None)
    ]
    
    if user:
        where_conditions.append(Call.owner == user)
    
    # Get service calls
    result = await db.execute(
        select(Call)
        .where(and_(*where_conditions))
        .order_by(Call.start_time.desc())
        .limit(limit * 2)
    )
    service_calls = result.scalars().all()
    
    if not service_calls:
        return {
            "service": {
                "id": service.id,
                "name": service.name,
                "description": service.description
            },
            "parameters": [],
            "schema": service.params,
            "totalCalls": 0,
            "message": "No calls found for this service"
        }
    
    # Process parameters
    processed_parameters = []
    unique_parameter_hashes = set()
    
    for call in service_calls:
        try:
            input_params = _deep_parse_json(call.input) if call.input else None
            
            if not input_params:
                continue
            
            parameter_set = {
                "callId": call.id,
                "owner": call.owner,
                "timestamp": call.start_time.isoformat() if call.start_time else None,
                "status": call.status,
                "parameters": input_params
            }
            
            # Check uniqueness
            if unique == "true":
                param_hash = json.dumps(input_params, sort_keys=True)
                if param_hash in unique_parameter_hashes:
                    continue
                unique_parameter_hashes.add(param_hash)
            
            processed_parameters.append(parameter_set)
            
            if len(processed_parameters) >= limit:
                break
        except Exception as e:
            print(f"Error processing call {call.id}: {e}")
            continue
    
    # Analyze parameters
    analysis = _analyze_parameters(processed_parameters)
    sets_analysis = _analyze_parameter_sets(processed_parameters)
    
    return {
        "service": {
            "id": service.id,
            "name": service.name,
            "description": service.description,
            "type": service.type
        },
        "parameters": processed_parameters,
        "analysis": analysis,
        "setsAnalysis": sets_analysis,
        "schema": service.params,
        "totalCalls": len(service_calls),
        "returnedParameters": len(processed_parameters),
        "filters": {
            "user": user,
            "limit": limit,
            "unique": unique == "true"
        }
    }


def _analyze_parameters(parameter_sets):
    """Analyze parameters and return statistics"""
    analysis = {
        "parameterNames": {},
        "parameterTypes": {},
        "mostCommonValues": {},
        "totalUniqueCombinations": len(parameter_sets)
    }
    
    for param_set in parameter_sets:
        params = param_set["parameters"]
        
        for key, value in params.items():
            # Count parameter names
            analysis["parameterNames"][key] = analysis["parameterNames"].get(key, 0) + 1
            
            # Analyze types
            value_type = "array" if isinstance(value, list) else type(value).__name__
            if key not in analysis["parameterTypes"]:
                analysis["parameterTypes"][key] = {}
            analysis["parameterTypes"][key][value_type] = analysis["parameterTypes"][key].get(value_type, 0) + 1
            
            # Count common values
            if key not in analysis["mostCommonValues"]:
                analysis["mostCommonValues"][key] = {}
            
            if isinstance(value, (str, int, float, bool)):
                value_for_counting = value
            elif isinstance(value, (list, dict)):
                try:
                    value_for_counting = json.dumps(value, sort_keys=True)
                except:
                    continue
            else:
                value_for_counting = str(value)
            
            analysis["mostCommonValues"][key][str(value_for_counting)] = \
                analysis["mostCommonValues"][key].get(str(value_for_counting), 0) + 1
    
    # Sort common values
    for key in analysis["mostCommonValues"]:
        sorted_values = sorted(
            analysis["mostCommonValues"][key].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        analysis["mostCommonValues"][key] = dict(sorted_values)
    
    return analysis


def _analyze_parameter_sets(parameter_sets):
    """Analyze popular parameter sets"""
    analysis = {
        "totalSets": len(parameter_sets),
        "uniqueSets": 0,
        "popularSets": {},
        "setFrequency": {},
        "mostPopularParameters": {},
        "setsByOwner": {}
    }
    
    parameter_hash_groups = {}
    owner_stats = {}
    
    for param_set in parameter_sets:
        param_hash = json.dumps(param_set["parameters"], sort_keys=True)
        owner = param_set["owner"]
        
        if param_hash not in parameter_hash_groups:
            parameter_hash_groups[param_hash] = {
                "count": 0,
                "parameters": param_set["parameters"],
                "owners": set(),
                "firstUsed": param_set["timestamp"],
                "lastUsed": param_set["timestamp"],
                "callIds": []
            }
        
        parameter_hash_groups[param_hash]["count"] += 1
        parameter_hash_groups[param_hash]["owners"].add(owner)
        parameter_hash_groups[param_hash]["callIds"].append(param_set["callId"])
        
        if owner not in owner_stats:
            owner_stats[owner] = {
                "totalCalls": 0,
                "uniqueSets": set(),
                "mostUsedSet": {"hash": None, "count": 0}
            }
        owner_stats[owner]["totalCalls"] += 1
        owner_stats[owner]["uniqueSets"].add(param_hash)
    
    analysis["uniqueSets"] = len(parameter_hash_groups)
    
    # Sort by popularity
    sorted_sets = sorted(
        parameter_hash_groups.items(),
        key=lambda x: x[1]["count"],
        reverse=True
    )[:20]
    
    for param_hash, data in sorted_sets:
        analysis["popularSets"][param_hash] = {
            "count": data["count"],
            "parameters": data["parameters"],
            "uniqueOwners": len(data["owners"]),
            "owners": list(data["owners"]),
            "firstUsed": data["firstUsed"],
            "lastUsed": data["lastUsed"],
            "totalCalls": len(data["callIds"]),
            "callIds": data["callIds"][:10]
        }
    
    return analysis


def _service_to_dict(service: Service) -> Dict[str, Any]:
    """Convert Service model to dictionary"""
    return {
        "id": service.id,
        "name": service.name,
        "subject": service.subject,
        "type": service.type,
        "description": service.description,
        "number_of_calls": service.number_of_calls,
        "actionview": service.actionview,
        "actionmodify": service.actionmodify,
        "map_reduce_specification": service.map_reduce_specification,
        "params": service.params,
        "js_body": service.js_body,
        "wpsservers": service.wpsservers,
        "wpsmethod": service.wpsmethod,
        "status": service.status,
        "output_params": service.output_params,
        "wms_link": service.wms_link,
        "wms_layer_name": service.wms_layer_name,
        "is_deleted": service.is_deleted,
        "created_by": service.created_by,
        "edited_by": service.edited_by,
        "edited_on": service.edited_on.isoformat() if service.edited_on else None,
        "created_on": service.created_on.isoformat() if service.created_on else None,
        "classname": service.classname,
    }

