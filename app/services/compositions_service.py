"""
Compositions service - business logic for service composition analysis
This module analyzes service call history to identify composition workflows
"""
import json
from typing import Dict, Any, List, Set
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Call, Service, Composition, User, Dataset
from app.core.config import settings

# Constants
DATASET_ID_OFFSET = 1000000
WIDGET_FILE = 'file'
WIDGET_FILE_SAVE = 'file_save'
WIDGET_THEME_SELECT = 'theme_select'
TASK_SUCCEEDED = 'TASK_SUCCEEDED'


def safe_json_parse(json_string, default_value=None):
    """Safe JSON parsing with error handling"""
    if default_value is None:
        default_value = {}
    
    try:
        return json.loads(json_string) if isinstance(json_string, str) else json_string
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return default_value


def parse_service_params(params_string):
    """Parse service parameters from JSON string"""
    try:
        return json.loads(params_string) if isinstance(params_string, str) else params_string or []
    except:
        return []


def categorize_params(params, input_widget_types=None, output_widget_types=None):
    """Categorize parameters into internal (file/dataset) and external"""
    if input_widget_types is None:
        input_widget_types = [WIDGET_FILE]
    if output_widget_types is None:
        output_widget_types = [WIDGET_FILE_SAVE]
    
    external = {}
    internal = {}
    
    for param in params:
        widget_name = param.get('widget', {}).get('name') if isinstance(param.get('widget'), dict) else None
        fieldname = param.get('fieldname')
        
        if widget_name in input_widget_types or widget_name in output_widget_types:
            internal[fieldname] = widget_name
        else:
            external[fieldname] = param.get('type')
    
    return {"internal": internal, "external": external}


async def build_service_connection_map(db: AsyncSession) -> Dict[int, Any]:
    """
    Build map of service connections (inputs/outputs)
    Extended with data from in_and_out_settings.json file if exists
    """
    print("Building service connection map...")
    
    # Загружаем данные из файла перед циклом
    import os
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
        
        # Проверяем наличие данных для этого сервиса в файле
        service_id_str = str(service.id)
        if service_id_str in file_data:
            # Объединяем данные из БД с данными из файла
            file_service_data = file_data[service_id_str]
            
            # Объединяем input (данные из файла расширяют данные из БД)
            combined_input = input_categorized["internal"].copy()
            if "input" in file_service_data:
                combined_input.update(file_service_data["input"])
            
            # Объединяем output (данные из файла расширяют данные из БД)
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
            # Используем только данные из БД
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
    """Build GUID to ID mapping for datasets"""
    print("Building dataset GUID to ID mapping...")
    
    result = await db.execute(select(Dataset))
    datasets = result.scalars().all()
    
    guid_map = {}
    for dataset in datasets:
        if dataset.guid:
            guid_map[dataset.guid] = dataset.id
    
    return guid_map


def normalize_dataset_id(dataset_id, guid_map):
    """Normalize dataset ID (convert GUID to ID and add offset)"""
    normalized_id = dataset_id
    
    # Convert GUID to ID if necessary
    if dataset_id in guid_map:
        normalized_id = guid_map[dataset_id]
    
    # Convert to int if string
    if isinstance(normalized_id, str):
        try:
            normalized_id = int(normalized_id)
        except:
            return dataset_id
    
    # Add offset to distinguish from service IDs
    return normalized_id + DATASET_ID_OFFSET


async def create_compositions(db: AsyncSession, compositions: List[Dict]):
    """Save compositions to database"""
    try:
        new_count = 0
        existing_count = 0
        
        for composition_data in compositions:
            # Check if exists
            result = await db.execute(
                select(Composition).where(Composition.id == composition_data["id"])
            )
            existing = result.scalar_one_or_none()
            
            if not existing:
                composition = Composition(
                    id=composition_data["id"],
                    nodes=composition_data["nodes"],
                    links=composition_data["links"]
                )
                db.add(composition)
                new_count += 1
            else:
                existing_count += 1
        
        await db.commit()
        print(f"Compositions saved: {new_count} new, {existing_count} already existed")
    except Exception as e:
        print(f"Error saving compositions: {e}")
        await db.rollback()
        raise


async def create_users(db: AsyncSession, users: Dict):
    """Create users from composition analysis"""
    try:
        for user_id in users.keys():
            result = await db.execute(
                select(User).where(User.id == user_id)
            )
            existing = result.scalar_one_or_none()
            
            if not existing:
                user = User(id=user_id)
                db.add(user)
        
        await db.commit()
        print("Users successfully created")
    except Exception as e:
        print(f"Error creating users: {e}")
        await db.rollback()


def add_task_link(links, task_id, source_id, source_param_name, param_name):
    """Add link between tasks"""
    link_data = {
        "source": source_id,
        "target": task_id,
        "value": f"{source_param_name}:{param_name}"
    }
    
    # Use string key for consistency with build_composition_for_task
    task_id_str = str(task_id)
    if task_id_str in links:
        links[task_id_str].append(link_data)
    else:
        links[task_id_str] = [link_data]


def create_composition_node(task, in_and_out):
    """Create composition node from task"""
    inputs = safe_json_parse(task.input, {})
    outputs = safe_json_parse(task.result, {})
    service_info = in_and_out.get(task.mid)
    
    if not service_info:
        return None
    
    local_inputs = [
        {
            "name": input_name,
            "value": inputs.get(input_name),
            "type": service_info["externalInput"].get(input_name)
        }
        for input_name in service_info["externalInput"].keys()
    ]
    
    local_outputs = [
        {
            "name": output_name,
            "value": outputs.get(output_name),
            "type": service_info["externalOutput"].get(output_name)
        }
        for output_name in service_info["externalOutput"].keys()
    ]
    
    return {
        "mid": task.mid,
        "taskId": task.id,
        "type": service_info["type"],
        "service": service_info["name"],
        "owner": task.owner,
        "inputs": local_inputs,
        "outputs": local_outputs,
        "end_time": task.end_time.isoformat() if task.end_time else None
    }


def build_composition_for_task(task, links, tasks, task_id_to_index, in_and_out):
    """Build composition starting from successful task"""
    stack = [task.id]
    nodes = []
    local_links = {}
    
    while stack:
        current_task_id = stack.pop()
        current_task = tasks[task_id_to_index.get(current_task_id)]
        
        if not current_task:
            continue
        
        node = create_composition_node(current_task, in_and_out)
        if not node:
            continue
        
        # Process links to this task
        task_links = links.get(str(current_task_id))
        if task_links:
            for link in task_links:
                source_task = tasks[task_id_to_index.get(link["source"])]
                if not source_task:
                    continue
                
                link_key = f"{source_task.id}:{current_task.id}"
                
                if link_key in local_links:
                    local_links[link_key]["value"].append(link["value"])
                else:
                    local_links[link_key] = {
                        "source": source_task.id,
                        "sourceMid": source_task.mid,
                        "target": current_task.id,
                        "targetMid": current_task.mid,
                        "value": [link["value"]]
                    }
                
                stack.append(link["source"])
        
        # Add reference inputs for current node BEFORE adding to nodes
        for link in local_links.values():
            if link["target"] == current_task.id:  # Only add inputs for current task
                for params in link["value"]:
                    source_param_name, target_param_name = params.split(':')
                    node["inputs"].append({
                        "name": target_param_name,
                        "value": f"ref::{link['source']}::{source_param_name}",
                        "type": in_and_out.get(link["targetMid"], {}).get("input", {}).get(target_param_name)
                    })
        
        nodes.append(node)
    
    return {"nodes": nodes, "localLinks": local_links}


def normalize_composition(nodes, local_links):
    """Normalize composition (assign local IDs, sort by time)"""
    # Sort by end time
    nodes.sort(key=lambda x: x["end_time"] or "")
    
    task_id_to_local_id = {}
    composition_id = ""
    
    # Assign local IDs
    for index, node in enumerate(nodes):
        node["id"] = f"task/{index + 1}"
        task_id_to_local_id[node["taskId"]] = node["id"]
        
        if composition_id:
            composition_id += "_"
        composition_id += str(node["taskId"])
        
        # Update reference inputs
        updated_inputs = []
        for input_item in node["inputs"]:
            if input_item.get("value") and isinstance(input_item["value"], str) and "ref::" in input_item["value"]:
                parts = input_item["value"].split("::")
                if len(parts) == 3:
                    ref, task_id, source_param_name = parts
                    local_id = task_id_to_local_id.get(int(task_id))
                    if local_id:
                        input_item["value"] = f"{ref}::{local_id}::{source_param_name}"
            updated_inputs.append(input_item)
        node["inputs"] = updated_inputs
    
    # Normalize links
    normalized_links = [
        {
            **link,
            "source": task_id_to_local_id.get(link["source"]),
            "target": task_id_to_local_id.get(link["target"])
        }
        for link in local_links.values()
    ]
    
    return {
        "id": composition_id,
        "nodes": nodes,
        "links": normalized_links
    }


async def recover(db: AsyncSession) -> Dict[str, Any]:
    """
    Recover service compositions from call history
    Original algorithm
    """
    try:
        print("Starting service composition recovery...")
        
        # Build service connection map
        in_and_out = await build_service_connection_map(db)
        
        # Get all tasks
        print("Loading tasks...")
        result = await db.execute(select(Call).order_by(Call.id.asc()))
        tasks = result.scalars().all()
        tasks_list = list(tasks)
        
        compositions = []
        file_value_tracker = {}
        task_links = {}
        task_id_to_index = {task.id: idx for idx, task in enumerate(tasks_list)}
        users = {}
        
        # Build users dict
        for task in tasks_list:
            if task.owner:
                users[task.owner] = True
        
        print(f"Processing {len(tasks_list)} tasks...")
        
        # Main processing loop
        for task in tasks_list:
            inputs = safe_json_parse(task.input, {})
            result_data = safe_json_parse(task.result, {})
            
            if task.mid not in in_and_out:
                continue
            
            service_inputs = in_and_out[task.mid].get("input", {})
            service_outputs = in_and_out[task.mid].get("output", {})
            
            # Check if successful with WMS
            is_successful_with_wms = (
                result_data and
                ((result_data.get("status") == "success" and
                "wms_link" in result_data) or 
                (task.mid == 399 and
                "map" in result_data and
                task.status == "TASK_SUCCEEDED"))
            )
            
            if is_successful_with_wms:
                # Process inputs to find links
                for param_name in service_inputs.keys():
                    input_value = inputs.get(param_name) if isinstance(inputs, dict) else None
                    widget_type = service_inputs[param_name]
                    
                    # For 'edit' widget type, convert value to string
                    if widget_type == 'edit' and input_value is not None:
                        input_value = str(input_value)
                    
                    # Skip if input_value is not hashable (dict, list, etc.)
                    if input_value and isinstance(input_value, (str, int, float, bool, type(None))):
                        if input_value in file_value_tracker and widget_type != WIDGET_THEME_SELECT:
                            tracker_info = file_value_tracker[input_value]
                            add_task_link(
                                task_links,
                                task.id,
                                tracker_info["value"],
                                tracker_info["name"],
                                param_name
                            )
                
                # Build composition
                comp_data = build_composition_for_task(
                    task, task_links, tasks_list, task_id_to_index, in_and_out
                )
                
                nodes_count = len(comp_data["nodes"])
                if nodes_count > 1:
                    composition = normalize_composition(comp_data["nodes"], comp_data["localLinks"])
                    compositions.append(composition)
            else:
                # Process intermediate task
                for param_name in service_inputs.keys():
                    input_value = inputs.get(param_name) if isinstance(inputs, dict) else None
                    widget_type = service_inputs[param_name]
                    
                    # For 'edit' widget type, convert value to string
                    if widget_type == 'edit' and input_value is not None:
                        input_value = str(input_value)
                    
                    # Skip if input_value is not hashable (dict, list, etc.)
                    if input_value and isinstance(input_value, (str, int, float, bool, type(None))):
                        if input_value in file_value_tracker and widget_type != WIDGET_THEME_SELECT:
                            tracker_info = file_value_tracker[input_value]
                            add_task_link(
                                task_links,
                                task.id,
                                tracker_info["value"],
                                tracker_info["name"],
                                param_name
                            )
            
            # Register output files (for ALL tasks, not just intermediate ones)
            if result_data and isinstance(result_data, dict):
                for param_name in service_outputs.keys():
                    output_value = result_data.get(param_name)
                    widget_type = service_outputs[param_name]
                    
                    # For 'edit' widget type, convert value to string and track
                    if widget_type == 'edit' and output_value is not None:
                        output_value = str(output_value)
                        file_value_tracker[output_value] = {
                            "value": task.id,
                            "name": param_name
                        }
                    # Track hashable values for other widget types (matching JS behavior)
                    elif output_value and isinstance(output_value, (str, int, float, bool)):
                        file_value_tracker[output_value] = {
                            "value": task.id,
                            "name": param_name
                        }
        
        print(f"Created {len(compositions)} compositions")
        
        # Save results
        await create_compositions(db, compositions)
        await create_users(db, users)
        
        return {
            "success": True,
            "message": "Service composition recovery completed",
            "compositionsCount": len(compositions),
            "usersCount": len(users)
        }
        
    except Exception as e:
        print(f"Error in recover function: {e}")
        raise


async def recover_new(db: AsyncSession) -> Dict[str, Any]:
    """
    Advanced service composition recovery
    Improved algorithm with dataset tracking
    """
    try:
        print("Starting advanced service composition recovery...")
        
        # Build maps in parallel
        in_and_out = await build_service_connection_map(db)
        guid_map = await build_dataset_guid_map(db)
        
        # Get all calls
        print("Loading calls...")
        result = await db.execute(select(Call).order_by(Call.id.asc()))
        calls = result.scalars().all()
        calls_list = list(calls)
        
        # Initialize data structures
        dataset_links = {}
        service_dataset_edges = {}
        file_tracker = {}
        call_edges = {}
        call_id_to_index = {call.id: idx for idx, call in enumerate(calls_list)}
        users = {call.owner: True for call in calls_list if call.owner}
        
        print(f"Processing {len(calls_list)} calls...")
        
        # First pass: analyze connections
        for call in calls_list:
            if call.status != TASK_SUCCEEDED:
                continue
            
            inputs = safe_json_parse(call.input, {})
            outputs = safe_json_parse(call.result, {})
            
            if call.mid not in in_and_out:
                continue
            
            service_inputs = in_and_out[call.mid].get("input", {})
            service_outputs = in_and_out[call.mid].get("output", {})
            
            if not service_inputs or not service_outputs:
                continue
            
            # Process inputs
            for param_name in service_inputs.keys():
                input_value = inputs.get(param_name) if isinstance(inputs, dict) else None
                if not input_value:
                    continue
                
                widget_type = service_inputs[param_name]
                
                if widget_type == WIDGET_THEME_SELECT:
                    # Dataset connection
                    parsed_input = safe_json_parse(input_value, input_value) if isinstance(input_value, str) else input_value
                    
                    if isinstance(parsed_input, dict) and "dataset_id" in parsed_input:
                        normalized_id = normalize_dataset_id(parsed_input["dataset_id"], guid_map)
                        dataset_links[call.id] = f"{normalized_id}:{param_name}"
                        
                        # Update service-dataset edges
                        if normalized_id not in service_dataset_edges:
                            service_dataset_edges[normalized_id] = {}
                        if call.mid not in service_dataset_edges[normalized_id]:
                            service_dataset_edges[normalized_id][call.mid] = {"total": 0}
                        if call.owner not in service_dataset_edges[normalized_id][call.mid]:
                            service_dataset_edges[normalized_id][call.mid][call.owner] = 0
                        
                        service_dataset_edges[normalized_id][call.mid][call.owner] += 1
                        service_dataset_edges[normalized_id][call.mid]["total"] += 1
                        
                elif widget_type == WIDGET_FILE or widget_type == 'edit':
                    # File connection or edit widget
                    # For 'edit' widget type, convert value to string
                    if widget_type == 'edit':
                        input_value = str(input_value)
                    
                    file_info = file_tracker.get(input_value)
                    if file_info and file_info.get("source_call_id") and file_info.get("source_param_name"):
                        if call.id not in call_edges:
                            call_edges[call.id] = {}
                        if file_info["source_call_id"] not in call_edges[call.id]:
                            call_edges[call.id][file_info["source_call_id"]] = []
                        
                        call_edges[call.id][file_info["source_call_id"]].append(
                            f"{file_info['source_param_name']}:{param_name}"
                        )
            
            # Register output files
            for param_name in service_outputs.keys():
                output_value = outputs.get(param_name) if isinstance(outputs, dict) else None
                widget_type = service_outputs[param_name]
                
                # For 'edit' widget type, convert value to string and track
                if widget_type == 'edit' and output_value is not None:
                    output_value = str(output_value)
                    file_tracker[output_value] = {
                        "source_call_id": call.id,
                        "source_service_id": call.mid,
                        "source_param_name": param_name
                    }
                # Track file_save widget types
                elif output_value and widget_type == WIDGET_FILE_SAVE:
                    file_tracker[output_value] = {
                        "source_call_id": call.id,
                        "source_service_id": call.mid,
                        "source_param_name": param_name
                    }
        
        # Second pass: build compositions
        raw_compositions = {}
        
        for call in calls_list:
            if call.status != TASK_SUCCEEDED:
                continue
            
            # Dataset connections
            if call.id in dataset_links:
                dataset_id_str, param_name = dataset_links[call.id].split(':')
                
                dataset_node = {
                    "id": dataset_id_str,
                    "start_date": call.start_time.isoformat() if call.start_time else None
                }
                
                dataset_link = {
                    "source": dataset_id_str,
                    "target": call.id,
                    "fields": f"{dataset_id_str}:{param_name}"
                }
                
                raw_compositions[call.id] = {
                    "nodes": [dataset_node, call],
                    "links": [dataset_link]
                }
            
            # Call edges
            if call.id in call_edges:
                for source_call_id, fields in call_edges[call.id].items():
                    link = {"source": source_call_id, "target": call.id, "fields": fields}
                    
                    if source_call_id in raw_compositions:
                        # Inherit composition
                        raw_compositions[call.id] = {
                            "nodes": raw_compositions[source_call_id]["nodes"] + [call],
                            "links": raw_compositions[source_call_id]["links"] + [link]
                        }
                    else:
                        # Create new composition
                        source_call = calls_list[call_id_to_index[source_call_id]]
                        raw_compositions[call.id] = {
                            "nodes": [source_call, call],
                            "links": [link]
                        }
        
        # Extract sequences
        call_sequences = []
        service_sequences = []
        
        for composition in raw_compositions.values():
            call_ids = []
            service_ids = []
            
            for node in composition["nodes"]:
                if hasattr(node, 'mid'):  # It's a Call object
                    call_ids.append(node.id)
                    service_ids.append(node.mid)
            
            if call_ids:
                call_sequences.append('_'.join(map(str, call_ids)))
                service_sequences.append('_'.join(map(str, service_ids)))
        
        # Filter non-prefix sequences
        def filter_non_prefix(sequences):
            return [
                seq for i, seq in enumerate(sequences)
                if not any(i != j and other.startswith(seq) for j, other in enumerate(sequences))
            ]
        
        longest_call_sequences = filter_non_prefix(call_sequences)
        longest_service_sequences = filter_non_prefix(list(set(service_sequences)))
        
        # Build final compositions
        final_compositions = [
            raw_compositions[int(seq.split('_')[-1])]
            for seq in longest_call_sequences
            if int(seq.split('_')[-1]) in raw_compositions
        ]
        
        # Save compositions DAG to file
        output_path = "app/static/compositionsDAG.json"
        
        print(f"Preparing to save {len(final_compositions)} compositions to file...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Convert Call objects to dicts for JSON serialization
            serializable_compositions = []
            for i, comp in enumerate(final_compositions):
                try:
                    print(f"Processing composition {i+1}/{len(final_compositions)}, nodes count: {len(comp['nodes'])}")
                    
                    serializable_nodes = []
                    for j, node in enumerate(comp["nodes"]):
                        try:
                            node_dict = {
                                "id": node.id if hasattr(node, 'id') else (node.get("id") if isinstance(node, dict) else str(node)),
                                "mid": node.mid if hasattr(node, 'mid') else None,
                                "owner": node.owner if hasattr(node, 'owner') else None,
                                "start_time": node.start_time.isoformat() if hasattr(node, 'start_time') and node.start_time else (node.get("start_date") if isinstance(node, dict) else None)
                            }
                            serializable_nodes.append(node_dict)
                        except Exception as e:
                            print(f"Error processing node {j} in composition {i}: {e}, node type: {type(node)}")
                            raise
                    
                    serializable_comp = {
                        "nodes": serializable_nodes,
                        "links": comp["links"]
                    }
                    serializable_compositions.append(serializable_comp)
                except Exception as e:
                    print(f"Error processing composition {i}: {e}")
                    raise
            
            json.dump(serializable_compositions, f, indent=2)
            print(f"Successfully saved compositions to {output_path}")
        
        print(f"Created {len(final_compositions)} final compositions")
        
        return {
            "success": True,
            "message": "Advanced composition recovery completed",
            "longest_service_seq": longest_service_sequences,
            "longest_comp": longest_call_sequences,
            "res_compositions": len(final_compositions),
            "stats": {
                "totalCalls": len(calls_list),
                "finalCompositions": len(final_compositions),
                "users": len(users)
            }
        }
        
    except Exception as e:
        print(f"Error in recoverNew function: {e}")
        raise


async def fetch_all_compositions(db: AsyncSession) -> List[Dict[str, Any]]:
    """Get all compositions from database"""
    result = await db.execute(select(Composition))
    compositions = result.scalars().all()
    
    return [
        {
            "id": comp.id,
            "nodes": comp.nodes,
            "links": comp.links
        }
        for comp in compositions
    ]


async def get_composition_stats(db: AsyncSession) -> Dict[str, Any]:
    """Get composition statistics and build graph"""
    # Get all tasks
    result = await db.execute(select(Call).order_by(Call.id.desc()))
    tasks = result.scalars().all()
    tasks_list = list(tasks)
    
    task_id_to_index = {task.id: idx for idx, task in enumerate(tasks_list)}
    nodes = {}
    links = {}
    
    # Process tasks for dataset connections
    for task in tasks_list:
        inputs = safe_json_parse(task.input, {})
        
        theme = inputs.get("theme")
        if isinstance(theme, dict) and "dataset_id" in theme:
            dataset_id = theme["dataset_id"]
            mid = task.mid
            owner = task.owner
            
            # Update nodes
            if dataset_id not in nodes:
                nodes[dataset_id] = {"id": dataset_id}
            if owner not in nodes[dataset_id]:
                nodes[dataset_id][owner] = 0
            nodes[dataset_id][owner] += 1
            
            if mid not in nodes:
                nodes[mid] = {"id": mid}
            if owner not in nodes[mid]:
                nodes[mid][owner] = 0
            nodes[mid][owner] += 1
            
            # Update links
            link_key = f"{dataset_id}:{mid}"
            if link_key not in links:
                links[link_key] = {
                    "source": dataset_id,
                    "target": mid,
                    "stats": {"total": 0}
                }
            if owner not in links[link_key]["stats"]:
                links[link_key]["stats"][owner] = 0
            links[link_key]["stats"][owner] += 1
            links[link_key]["stats"]["total"] += 1
    
    print(f"nodes from datasets: {len(nodes)}")
    print(f"links from datasets: {len(links)}")
    
    # Process compositions
    result = await db.execute(select(Composition))
    compositions = result.scalars().all()
    
    path = {}
    for composition in compositions:
        composition_elements = composition.nodes
        if not composition_elements:
            continue
        
        last_task_id = composition_elements[-1].get("taskId")
        if last_task_id not in task_id_to_index:
            continue
        
        owner = tasks_list[task_id_to_index[last_task_id]].owner
        path_str = ""
        
        for i, node in enumerate(composition_elements):
            mid = node.get("mid")
            path_str = path_str + str(mid) + "."
            
            if mid not in nodes:
                nodes[mid] = {"id": mid}
            if owner not in nodes[mid]:
                nodes[mid][owner] = 0
            nodes[mid][owner] += 1
            
            if i < len(composition_elements) - 1:
                source_mid = composition_elements[i].get("mid")
                target_mid = composition_elements[i + 1].get("mid")
                link_key = f"{source_mid}:{target_mid}"
                
                if link_key not in links:
                    links[link_key] = {
                        "source": source_mid,
                        "target": target_mid,
                        "stats": {"total": 0}
                    }
                if owner not in links[link_key]["stats"]:
                    links[link_key]["stats"][owner] = 0
                links[link_key]["stats"][owner] += 1
                links[link_key]["stats"]["total"] += 1
        
        path[path_str] = True
    
    print(f"total nodes: {len(nodes)}")
    print(f"total links: {len(links)}")
    
    # Save to file
    import os
    file_path = os.path.join(os.path.dirname(settings.CSV_FILE_PATH), "statsGraph.json")
    result_data = {
        "nodes": nodes,
        "links": links
    }
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2)
    
    print("write to statsGraph")
    
    return result_data

