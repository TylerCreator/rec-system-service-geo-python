"""
Composition building and normalization
"""
from typing import Dict, Any, List
from .helpers import create_composition_node


def build_composition_for_task(task: Any, links: Dict, tasks: List[Any], 
                                task_id_to_index: Dict[int, int], 
                                in_and_out: Dict[int, Any]) -> Dict[str, Any]:
    """
    Build composition starting from successful task
    
    Args:
        task: Starting task
        links: Dictionary of task links
        tasks: List of all tasks
        task_id_to_index: Mapping of task ID to index in tasks list
        in_and_out: Service connection map
    
    Returns:
        Dictionary with 'nodes' and 'localLinks'
    """
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


def normalize_composition(nodes: List[Dict], local_links: Dict) -> Dict[str, Any]:
    """
    Normalize composition (assign local IDs, sort by time)
    
    Args:
        nodes: List of composition nodes
        local_links: Dictionary of links between nodes
    
    Returns:
        Normalized composition with id, nodes, and links
    """
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

