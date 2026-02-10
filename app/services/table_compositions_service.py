"""
Table compositions service.

Provides:
- listing table compositions stored in DB
- recommendation model for "substitute a new table" scenarios

The model is trained from recovered compositions (recover_new) stored in TableCompositions.
It does NOT require CRIS calls to be fetched at request time.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import TableComposition
from app.services.compositions.service_map import build_service_connection_map
from app.services.utils.constants import WIDGET_THEME_SELECT, WIDGET_FILE, WIDGET_EDIT


def _is_intish(x: Any) -> bool:
    try:
        int(str(x))
        return True
    except Exception:
        return False


def _build_graph_from_record(record: Dict[str, Any]) -> Tuple[Dict[int, int], Dict[int, List[int]], Dict[int, List[str]], List[Dict[str, Any]]]:
    """
    Returns:
    - call_id_to_mid
    - outgoing_call_edges: call_id -> [next_call_id]
    - incoming_table_edges: call_id -> [table_id_str]
    - raw_links
    """
    nodes = record.get("nodes") or []
    links = record.get("links") or []

    call_id_to_mid: Dict[int, int] = {}
    for n in nodes:
        if n.get("mid") is None:
            continue
        try:
            call_id_to_mid[int(str(n.get("id")))] = int(str(n.get("mid")))
        except Exception:
            continue

    outgoing_call_edges: Dict[int, List[int]] = defaultdict(list)
    incoming_table_edges: Dict[int, List[str]] = defaultdict(list)

    # Links are either:
    # - table_id(intish, typically >= 1_000_000) -> call_id(intish)
    # - call_id(intish) -> call_id(intish)
    for link in links:
        src = link.get("source")
        tgt = link.get("target")
        if src is None or tgt is None:
            continue
        if not _is_intish(tgt):
            continue

        tgt_int = int(str(tgt))
        if tgt_int not in call_id_to_mid:
            continue

        if not _is_intish(src):
            continue

        src_int = int(str(src))
        if src_int in call_id_to_mid:
            # call -> call
            outgoing_call_edges[src_int].append(tgt_int)
        else:
            # table -> call
            incoming_table_edges[tgt_int].append(str(src_int))

    # Deduplicate adjacency lists
    for k, v in outgoing_call_edges.items():
        outgoing_call_edges[k] = list(dict.fromkeys(v))
    for k, v in incoming_table_edges.items():
        incoming_table_edges[k] = list(dict.fromkeys(v))

    return call_id_to_mid, outgoing_call_edges, incoming_table_edges, links


def _bfs_shortest_path_calls(out_edges: Dict[int, List[int]], start: int, goal: int) -> Optional[List[int]]:
    """
    BFS shortest path in call graph (call_id -> call_id).
    """
    if start == goal:
        return [start]
    queue = [start]
    parent: Dict[int, Optional[int]] = {start: None}
    i = 0
    while i < len(queue):
        cur = queue[i]
        i += 1
        for nxt in out_edges.get(cur, []):
            if nxt in parent:
                continue
            parent[nxt] = cur
            if nxt == goal:
                # reconstruct
                path = [goal]
                p = cur
                while p is not None:
                    path.append(p)
                    p = parent[p]
                path.reverse()
                return path
            queue.append(nxt)
    return None


async def list_table_compositions(db: AsyncSession, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
    q = select(TableComposition).offset(offset).limit(limit)
    result = await db.execute(q)
    comps = result.scalars().all()
    return {
        "items": [
            {
                "id": c.id,
                "owner": c.owner,
                "start_time": c.start_time.isoformat() if c.start_time else None,
                "end_time": c.end_time.isoformat() if c.end_time else None,
                "table_ids": c.table_ids,
                "service_mids": c.service_mids,
                "join_steps_count": len(c.join_steps or []),
            }
            for c in comps
        ],
        "limit": limit,
        "offset": offset,
        "returned": len(comps),
    }


async def recommend_substitute_table(
    db: AsyncSession,
    upstream_service_id: int,
    new_table_id: int,
    existing_table_id: Optional[int] = None,
    n: int = 5,
) -> Dict[str, Any]:
    """
    Recommend a service chain to integrate a NEW table into an existing workflow.

    Inputs:
    - upstream_service_id: service mid that produced an artifact we already have
    - new_table_id: the new table/dataset id (may be unseen in training; used for output/validation)
    - existing_table_id: optional table already present in workflow context (tightens matching)

    Output:
    - ranked candidate service chains (list of mids), where the chain starts with a table-processing step
      (theme_select input) and ends with a "join" step that consumes upstream artifact.
    """
    # Load model data
    result = await db.execute(select(TableComposition))
    comps = result.scalars().all()

    # Service IO map to validate that suggested services can accept a table (theme_select)
    # NOTE: In local/offline setups this may be sparse; we relax filtering if missing.
    in_and_out = await build_service_connection_map(db)

    # Build pattern counts: (chain_mids_tuple) -> count
    def _process_compositions(filter_by_existing: bool):
        pattern_counts: Counter[Tuple[int, ...]] = Counter()
        pattern_examples: Dict[Tuple[int, ...], List[Dict[str, Any]]] = defaultdict(list)
        fallback_multi_table_counts: Counter[Tuple[int, ...]] = Counter()
        fallback_multi_table_examples: Dict[Tuple[int, ...], List[Dict[str, Any]]] = defaultdict(list)
        fallback_multi_table_any_counts: Counter[Tuple[int, ...]] = Counter()
        fallback_multi_table_any_examples: Dict[Tuple[int, ...], List[Dict[str, Any]]] = defaultdict(list)

        for c in comps:
            record = {
                "id": c.id,
                "table_ids": c.table_ids,
                "nodes": c.nodes,
                "links": c.links,
            }
            if filter_by_existing and existing_table_id is not None:
                try:
                    if existing_table_id not in (record.get("table_ids") or []):
                        continue
                except Exception:
                    continue

            # If join_steps were stored, use them for direct matching (more reliable than call graph alone)
            if c.join_steps:
                for step in c.join_steps:
                    if not step.get("is_join"):
                        # still useful for patterns where join is table-only
                        pass

                    upstream_calls = step.get("upstream_calls") or []
                    if upstream_service_id is not None:
                        if not any(u.get("source_service_mid") == upstream_service_id for u in upstream_calls):
                            continue

                    join_mid = step.get("target_service_mid")
                    if join_mid is None:
                        continue

                    chain = (int(join_mid),)
                    pattern_counts[chain] += 1
                    if len(pattern_examples[chain]) < 3:
                        pattern_examples[chain].append(
                            {
                                "composition_id": record["id"],
                                "join_call_id": step.get("target_call_id"),
                                "join_mid": join_mid,
                                "table_inputs": step.get("table_inputs"),
                                "upstream_calls": upstream_calls,
                            }
                        )
                # We already extracted candidates from join_steps; still allow graph logic for richer chains

                # Fallback: table-only joins (multiple table inputs in a single call)
                for step in c.join_steps:
                    table_inputs = step.get("table_inputs") or []
                    if len(table_inputs) < 2:
                        continue
                    if filter_by_existing and existing_table_id is not None:
                        if not any(t.get("table_id") == existing_table_id for t in table_inputs):
                            continue

                    join_mid = step.get("target_service_mid")
                    if join_mid is None:
                        continue

                    chain = (int(join_mid),)
                    fallback_multi_table_counts[chain] += 1
                    if len(fallback_multi_table_examples[chain]) < 3:
                        fallback_multi_table_examples[chain].append(
                            {
                                "composition_id": record["id"],
                                "join_call_id": step.get("target_call_id"),
                                "join_mid": join_mid,
                                "table_inputs": table_inputs,
                            }
                        )

                # Also track ANY multi-table patterns (used if existing_table_id doesn't match anything)
                for step in c.join_steps:
                    table_inputs = step.get("table_inputs") or []
                    if len(table_inputs) < 2:
                        continue
                    join_mid = step.get("target_service_mid")
                    if join_mid is None:
                        continue
                    chain = (int(join_mid),)
                    fallback_multi_table_any_counts[chain] += 1
                    if len(fallback_multi_table_any_examples[chain]) < 3:
                        fallback_multi_table_any_examples[chain].append(
                            {
                                "composition_id": record["id"],
                                "join_call_id": step.get("target_call_id"),
                                "join_mid": join_mid,
                                "table_inputs": table_inputs,
                            }
                        )

            call_id_to_mid, out_edges, incoming_table_edges, links = _build_graph_from_record(record)

            # Build incoming call edges for join detection
            incoming_call_edges: Dict[int, List[int]] = defaultdict(list)
            for link in links:
                src = link.get("source")
                tgt = link.get("target")
                if src is None or tgt is None:
                    continue
                if _is_intish(src) and _is_intish(tgt):
                    src_int = int(str(src))
                    tgt_int = int(str(tgt))
                    if src_int in call_id_to_mid and tgt_int in call_id_to_mid:
                        incoming_call_edges[tgt_int].append(src_int)

            # Candidate join calls: any call that has an incoming edge from a call with mid=upstream_service_id
            for join_call_id, incoming_sources in incoming_call_edges.items():
                if not incoming_sources:
                    continue

                if not any(call_id_to_mid.get(src) == upstream_service_id for src in incoming_sources):
                    continue

                join_mid = call_id_to_mid.get(join_call_id)
                if join_mid is None:
                    continue

                # Determine "other branches" besides the upstream source(s)
                other_call_sources = [src for src in incoming_sources if call_id_to_mid.get(src) != upstream_service_id]
                direct_tables_into_join = incoming_table_edges.get(join_call_id, [])

                # Direct table into join (join consumes table directly)
                for _table_src in direct_tables_into_join:
                    chain = (join_mid,)
                    pattern_counts[chain] += 1
                    if len(pattern_examples[chain]) < 3:
                        pattern_examples[chain].append(
                            {
                                "composition_id": record["id"],
                                "join_call_id": join_call_id,
                                "join_mid": join_mid,
                                "direct_table_into_join": _table_src,
                            }
                        )

                # Table enters via an upstream branch: find table-consuming entry calls in that branch
                for branch_src_call_id in other_call_sources:
                    # Find closest ancestor call(s) in this branch that directly consume a table
                    # We search backward in the call graph using incoming_call_edges.
                    queue = [branch_src_call_id]
                    visited = set()
                    entry_calls = []

                    while queue:
                        cur = queue.pop(0)
                        if cur in visited:
                            continue
                        visited.add(cur)

                        if incoming_table_edges.get(cur):
                            entry_calls.append(cur)
                            # Do not stop: a branch might include multiple table inputs

                        for prev in incoming_call_edges.get(cur, []):
                            # Stay within this composition's call graph
                            if prev in call_id_to_mid:
                                queue.append(prev)

                    # For each entry call, compute shortest call path entry -> join and map to mids
                    for entry_call_id in entry_calls:
                        path = _bfs_shortest_path_calls(out_edges, entry_call_id, join_call_id)
                        if not path:
                            continue
                        mids_path = []
                        for call_id in path:
                            mid = call_id_to_mid.get(call_id)
                            if mid is not None:
                                mids_path.append(mid)

                        # Ensure join_mid is last
                        if not mids_path or mids_path[-1] != join_mid:
                            mids_path.append(join_mid)

                        chain = tuple(mids_path)
                        pattern_counts[chain] += 1
                        if len(pattern_examples[chain]) < 3:
                            pattern_examples[chain].append(
                                {
                                    "composition_id": record["id"],
                                    "entry_call_id": entry_call_id,
                                    "entry_tables": incoming_table_edges.get(entry_call_id),
                                    "branch_src_call_id": branch_src_call_id,
                                    "join_call_id": join_call_id,
                                }
                            )

        return (
            pattern_counts,
            pattern_examples,
            fallback_multi_table_counts,
            fallback_multi_table_examples,
            fallback_multi_table_any_counts,
            fallback_multi_table_any_examples,
        )

    # First pass: respect existing_table_id if provided
    (pattern_counts,
     pattern_examples,
     fallback_multi_table_counts,
     fallback_multi_table_examples,
     fallback_multi_table_any_counts,
     fallback_multi_table_any_examples) = _process_compositions(filter_by_existing=existing_table_id is not None)

    # Second pass: if nothing found and existing_table_id was provided, retry without filtering
    if existing_table_id is not None and not pattern_counts and not fallback_multi_table_counts and not fallback_multi_table_any_counts:
        (pattern_counts,
         pattern_examples,
         fallback_multi_table_counts,
         fallback_multi_table_examples,
         fallback_multi_table_any_counts,
         fallback_multi_table_any_examples) = _process_compositions(filter_by_existing=False)

    # Rank patterns
    ranked = pattern_counts.most_common()
    # If no upstream-based patterns found, fall back to multi-table joins
    if not ranked and fallback_multi_table_counts:
        ranked = fallback_multi_table_counts.most_common()
        pattern_examples = fallback_multi_table_examples
    # If still empty (no match for existing_table_id), fall back to ANY multi-table joins
    if not ranked and fallback_multi_table_any_counts:
        ranked = fallback_multi_table_any_counts.most_common()
        pattern_examples = fallback_multi_table_any_examples

    # Validate patterns by service IO (must start with a table-processing service and end with a join-capable service)
    candidates = []
    for chain, cnt in ranked:
        if not chain:
            continue
        first_mid = chain[0]
        last_mid = chain[-1]

        first_io = in_and_out.get(int(first_mid), {}) if in_and_out else {}
        last_io = in_and_out.get(int(last_mid), {}) if in_and_out else {}
        first_inputs = first_io.get("input") or {}
        last_inputs = last_io.get("input") or {}

        first_table_params = [k for k, v in first_inputs.items() if v == WIDGET_THEME_SELECT]
        last_table_params = [k for k, v in last_inputs.items() if v == WIDGET_THEME_SELECT]
        last_artifact_params = [k for k, v in last_inputs.items() if v in (WIDGET_FILE, WIDGET_EDIT)]

        # If we have service IO info, enforce input compatibility.
        # Otherwise, relax filters to avoid dropping all candidates in local/offline setups.
        if in_and_out:
            # Only enforce filters when we have IO data for the candidate services
            has_first_info = bool(first_io)
            has_last_info = bool(last_io)
            if has_first_info and not first_table_params:
                continue
            if has_last_info and (not last_artifact_params and not last_table_params):
                continue

        candidates.append(
            {
                "service_chain": list(chain),
                "score": cnt,
                "evidence_count": cnt,
                "first_service": {
                    "service_id": first_mid,
                    "table_input_params": first_table_params,
                },
                "join_service": {
                    "service_id": last_mid,
                    "table_input_params": last_table_params,
                    "artifact_input_params": last_artifact_params,
                },
                "examples": pattern_examples.get(chain, []),
            }
        )
        if len(candidates) >= n:
            break

    # Final fallback: if still empty, recommend most common multi-table join services
    if not candidates:
        fallback_counts = Counter()
        fallback_examples: Dict[Tuple[int, ...], List[Dict[str, Any]]] = defaultdict(list)
        for c in comps:
            if not c.join_steps:
                continue
            for step in c.join_steps:
                table_inputs = step.get("table_inputs") or []
                if len(table_inputs) < 2:
                    continue
                join_mid = step.get("target_service_mid")
                if join_mid is None:
                    continue
                chain = (int(join_mid),)
                fallback_counts[chain] += 1
                if len(fallback_examples[chain]) < 3:
                    fallback_examples[chain].append(
                        {
                            "composition_id": c.id,
                            "join_call_id": step.get("target_call_id"),
                            "join_mid": join_mid,
                            "table_inputs": table_inputs,
                        }
                    )

        for chain, cnt in fallback_counts.most_common():
            first_mid = chain[0]
            last_mid = chain[-1]
            candidates.append(
                {
                    "service_chain": list(chain),
                    "score": cnt,
                    "evidence_count": cnt,
                    "first_service": {"service_id": first_mid, "table_input_params": []},
                    "join_service": {"service_id": last_mid, "table_input_params": [], "artifact_input_params": []},
                    "examples": fallback_examples.get(chain, []),
                }
            )
            if len(candidates) >= n:
                break

    return {
        "upstream_service_id": upstream_service_id,
        "new_table_id": new_table_id,
        "existing_table_id": existing_table_id,
        "candidates": candidates,
        "raw_patterns_found": len(pattern_counts),
        "table_compositions_used": len(comps),
        "note": (
            "Chains are learned from TableCompositions extracted by /compositions/recoverNew. "
            "Filtering ensures the first service consumes a table (theme_select) and the last service "
            "can consume an upstream artifact (file/edit)."
        ),
    }

