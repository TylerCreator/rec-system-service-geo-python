"""
Table compositions service.

Provides:
- listing table compositions stored in DB
- recommendation: given a sequence of tables, predict the next service + next table

The model is built from recovered compositions (recover_new) stored in TableCompositions.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import TableComposition


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


async def predict_next(
    db: AsyncSession,
    table_sequence: List[int],
    n: int = 5,
) -> Dict[str, Any]:
    """
    Given a sequence of table IDs, predict how to continue the workflow.

    Algorithm:
    1. Load all TableCompositions from DB
    2. Find compositions whose table_ids contain the input sequence (as a subsequence)
    3. For each match, look at what comes AFTER the matched prefix:
       - next service (mid) that processes the continuation
       - next table that appears after the last matched table
    4. Rank by frequency and return top-n

    Input:  table_sequence = [1003093, 1003086]  (tables already in workflow)
    Output: [
      { "next_service_mid": 399, "next_table_id": 1002118, "score": 5, ... },
      ...
    ]
    """
    if not table_sequence:
        return {
            "table_sequence": table_sequence,
            "predictions": [],
            "message": "Empty sequence provided",
        }

    result = await db.execute(select(TableComposition))
    comps = result.scalars().all()

    # Convert input to a set for fast prefix matching, and keep ordered list
    seq_set = set(table_sequence)
    last_table = table_sequence[-1]

    # Track: (next_service_mid, next_table_id) -> count
    next_service_counts: Counter[int] = Counter()
    next_table_counts: Counter[int] = Counter()
    next_pair_counts: Counter[Tuple[int, int]] = Counter()  # (service_mid, table_id)
    examples: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)

    for c in comps:
        comp_table_ids: List[int] = c.table_ids or []
        comp_service_mids: List[int] = c.service_mids or []
        join_steps: List[Dict[str, Any]] = c.join_steps or []
        nodes: List[Dict[str, Any]] = c.nodes or []
        links: List[Dict[str, Any]] = c.links or []

        # Skip compositions that don't contain any of our sequence tables
        if not seq_set.intersection(set(comp_table_ids)):
            continue

        # Strategy 1: Subsequence matching on table_ids
        # Find the position of the last table from our sequence in this composition's table_ids
        # Then see what comes after it
        try:
            # Find last occurrence of last_table in comp_table_ids
            last_idx = -1
            for i, tid in enumerate(comp_table_ids):
                if tid == last_table:
                    last_idx = i

            if last_idx >= 0 and last_idx < len(comp_table_ids) - 1:
                # There are tables after the matched position
                next_tid = comp_table_ids[last_idx + 1]
                next_table_counts[next_tid] += 1
        except Exception:
            pass

        # Strategy 2: Use join_steps to find what service processes the next table
        # Look at join_steps where one of the table_inputs matches a table in our sequence
        for step in join_steps:
            table_inputs = step.get("table_inputs") or []
            step_mid = step.get("target_service_mid")
            if step_mid is None:
                continue

            step_table_ids = [t.get("table_id") for t in table_inputs if t.get("table_id")]

            # If this step consumes one of our sequence tables
            matched_tables = seq_set.intersection(set(step_table_ids))
            if not matched_tables:
                continue

            # This step's service is a candidate "next service"
            next_service_counts[step_mid] += 1

            # Other tables in this step (not in our sequence) are candidate "next tables"
            other_tables = [tid for tid in step_table_ids if tid not in seq_set]
            for other_tid in other_tables:
                next_table_counts[other_tid] += 1
                pair = (step_mid, other_tid)
                next_pair_counts[pair] += 1
                if len(examples[pair]) < 3:
                    examples[pair].append({
                        "composition_id": c.id,
                        "step": step,
                    })

        # Strategy 3: Graph-based — look at what services/tables follow after nodes
        # that consume the last table in our sequence
        # Build a quick node lookup
        call_id_to_mid: Dict[int, int] = {}
        table_node_ids: set = set()
        for node in nodes:
            mid = node.get("mid")
            nid = node.get("id")
            if mid is not None:
                try:
                    call_id_to_mid[int(str(nid))] = int(str(mid))
                except Exception:
                    pass
            else:
                try:
                    table_node_ids.add(int(str(nid)))
                except Exception:
                    pass

        # Find calls that consume our last_table (via links)
        consuming_calls: List[int] = []
        for link in links:
            src = link.get("source")
            tgt = link.get("target")
            try:
                src_int = int(str(src))
                tgt_int = int(str(tgt))
            except Exception:
                continue

            if src_int == last_table and tgt_int in call_id_to_mid:
                consuming_calls.append(tgt_int)

        # For each consuming call, find what comes next in the graph
        for call_id in consuming_calls:
            mid = call_id_to_mid.get(call_id)
            if mid is not None:
                next_service_counts[mid] += 1

            # Find outgoing links from this call
            for link in links:
                src = link.get("source")
                tgt = link.get("target")
                try:
                    src_int = int(str(src))
                    tgt_int = int(str(tgt))
                except Exception:
                    continue

                if src_int != call_id:
                    continue

                # Target is another call → its mid is a "next service after the service that consumed our table"
                if tgt_int in call_id_to_mid:
                    next_mid = call_id_to_mid[tgt_int]
                    next_service_counts[next_mid] += 1

                    # Does this next call consume a table we haven't seen?
                    for link2 in links:
                        src2 = link2.get("source")
                        tgt2 = link2.get("target")
                        try:
                            src2_int = int(str(src2))
                            tgt2_int = int(str(tgt2))
                        except Exception:
                            continue
                        if tgt2_int == tgt_int and src2_int in table_node_ids and src2_int not in seq_set:
                            next_table_counts[src2_int] += 1
                            pair = (next_mid, src2_int)
                            next_pair_counts[pair] += 1
                            if len(examples[pair]) < 3:
                                examples[pair].append({
                                    "composition_id": c.id,
                                    "consuming_call": call_id,
                                    "next_call": tgt_int,
                                    "next_table": src2_int,
                                })

                # Target is a table node (not in our sequence) → candidate next table
                elif tgt_int in table_node_ids and tgt_int not in seq_set:
                    next_table_counts[tgt_int] += 1

    # Build predictions
    predictions: List[Dict[str, Any]] = []

    # Primary: paired predictions (service + table)
    for (svc_mid, tbl_id), cnt in next_pair_counts.most_common():
        predictions.append({
            "next_service_mid": svc_mid,
            "next_table_id": tbl_id,
            "score": cnt,
            "type": "service_and_table",
            "examples": examples.get((svc_mid, tbl_id), []),
        })
        if len(predictions) >= n:
            break

    # If not enough pairs, add service-only predictions
    if len(predictions) < n:
        seen_services = {p["next_service_mid"] for p in predictions}
        for svc_mid, cnt in next_service_counts.most_common():
            if svc_mid in seen_services:
                continue
            predictions.append({
                "next_service_mid": svc_mid,
                "next_table_id": None,
                "score": cnt,
                "type": "service_only",
                "examples": [],
            })
            if len(predictions) >= n:
                break

    # If still not enough, add table-only predictions
    if len(predictions) < n:
        seen_tables = {p["next_table_id"] for p in predictions if p["next_table_id"]}
        for tbl_id, cnt in next_table_counts.most_common():
            if tbl_id in seen_tables:
                continue
            predictions.append({
                "next_service_mid": None,
                "next_table_id": tbl_id,
                "score": cnt,
                "type": "table_only",
                "examples": [],
            })
            if len(predictions) >= n:
                break

    return {
        "table_sequence": table_sequence,
        "predictions": predictions,
        "compositions_searched": len(comps),
    }
