"""
Table composition extraction utilities.

Derives "table compositions" (table/dataset-centric workflows) from recovered
service compositions (nodes/links DAG).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def _dedupe_preserve_order(items: List[Any]) -> List[Any]:
    seen = set()
    out = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        # Handles "2026-01-30T17:15:42" as produced by the API
        return datetime.fromisoformat(value)
    except Exception:
        return None


def extract_table_compositions_from_service_compositions(
    compositions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build TableCompositions records from recovered compositions.

    Expected input composition format:
    {
      "id": "3985_3986_4010",
      "nodes": [{"id": "1002132", "mid": None, ...}, {"id": 3985, "mid": 308, ...}, ...],
      "links": [{"source": "1002132", "target": 3985, "fields": "1002132:theme"}, ...]
    }
    """

    table_compositions: List[Dict[str, Any]] = []

    for comp in compositions:
        comp_id = comp.get("id")
        nodes = comp.get("nodes") or []
        links = comp.get("links") or []

        # Build node maps
        call_id_to_mid: Dict[int, int] = {}
        call_id_to_owner: Dict[int, Optional[str]] = {}
        call_id_to_start: Dict[int, Optional[datetime]] = {}
        table_ids_in_order: List[int] = []
        call_ids_in_order: List[int] = []
        service_mids_in_order: List[int] = []

        for node in nodes:
            mid = node.get("mid")
            node_id = node.get("id")

            if mid is None:
                # Table/dataset node: id is numeric string in recover_new
                try:
                    table_id = int(str(node_id))
                    table_ids_in_order.append(table_id)
                except Exception:
                    continue
                continue

            # Call/service node
            try:
                call_id = int(node_id)
            except Exception:
                continue

            try:
                mid_int = int(mid)
            except Exception:
                continue

            call_id_to_mid[call_id] = mid_int
            call_id_to_owner[call_id] = node.get("owner")
            call_id_to_start[call_id] = _parse_iso_datetime(node.get("start_time"))
            call_ids_in_order.append(call_id)
            service_mids_in_order.append(mid_int)

        table_ids = _dedupe_preserve_order(table_ids_in_order)
        call_ids = _dedupe_preserve_order(call_ids_in_order)
        service_mids = _dedupe_preserve_order(service_mids_in_order)

        # Determine owner/start/end heuristically from call nodes
        owner = None
        for cid in call_ids:
            if call_id_to_owner.get(cid):
                owner = call_id_to_owner[cid]
                break

        times = [t for t in (call_id_to_start.get(cid) for cid in call_ids) if t is not None]
        start_time = min(times) if times else None
        end_time = max(times) if times else None

        # Index links by target for join-step detection
        links_by_target: Dict[str, List[Dict[str, Any]]] = {}
        # Track dataset producers: dataset_id -> source_call_id
        dataset_producers: Dict[str, int] = {}
        for link in links:
            target = str(link.get("target"))
            if not target:
                continue
            links_by_target.setdefault(target, []).append(link)

            # If link is call -> dataset, remember producer
            try:
                src = link.get("source")
                if src is None:
                    continue
                src_call_id = int(str(src))
                if src_call_id in call_id_to_mid and target.isdigit():
                    dataset_producers[target] = src_call_id
            except Exception:
                continue

        join_steps: List[Dict[str, Any]] = []

        for cid in call_ids:
            incoming = links_by_target.get(str(cid), [])
            table_inputs: List[Dict[str, Any]] = []
            upstream_calls: List[Dict[str, Any]] = []

            for link in incoming:
                src = link.get("source")
                if src is None:
                    continue

                # Table input: source is numeric string (dataset id) and node has mid None
                is_table_src = False
                try:
                    _ = int(str(src))
                    # Only treat as table if it exists in this composition's table set
                    is_table_src = int(str(src)) in set(table_ids)
                except Exception:
                    is_table_src = False

                if is_table_src:
                    table_id_int = int(str(src))
                    table_inputs.append(
                        {
                            "table_id": table_id_int,
                            "fields": link.get("fields"),
                        }
                    )
                    # If this table was produced by a prior call, treat it as upstream too
                    producer_call_id = dataset_producers.get(str(src))
                    if producer_call_id:
                        upstream_calls.append(
                            {
                                "source_call_id": producer_call_id,
                                "source_service_mid": call_id_to_mid.get(producer_call_id),
                                "fields": f"{table_id_int}:produced"
                            }
                        )
                    continue

                # Otherwise consider it an upstream call (service output)
                try:
                    src_call_id = int(str(src))
                except Exception:
                    continue

                upstream_calls.append(
                    {
                        "source_call_id": src_call_id,
                        "source_service_mid": call_id_to_mid.get(src_call_id),
                        "fields": link.get("fields"),
                    }
                )

            if not table_inputs:
                continue

            join_steps.append(
                {
                    "target_call_id": cid,
                    "target_service_mid": call_id_to_mid.get(cid),
                    "table_inputs": table_inputs,
                    "upstream_calls": upstream_calls,
                    "is_join": bool(upstream_calls),  # table + upstream service output
                }
            )

        if not comp_id:
            # Fallback if caller didn't generate one
            comp_id = "_".join(map(str, call_ids)) if call_ids else None

        if not comp_id:
            continue

        # Always store full nodes/links for audit/debug
        table_compositions.append(
            {
                "id": comp_id,
                "owner": owner,
                "start_time": start_time,
                "end_time": end_time,
                "table_ids": table_ids,
                "call_ids": call_ids,
                "service_mids": service_mids,
                "join_steps": join_steps,
                "nodes": nodes,
                "links": links,
            }
        )

    return table_compositions

