"""
Table compositions service.

Provides:
- listing table compositions stored in DB
- train: build a transition model from table_ids sequences in TableCompositions
- predict: given a sequence of tables, predict the next table(s)

The model is a first-order Markov chain learned from TableCompositions.table_ids.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import TableComposition


# In-memory model (singleton)
_transition_counts: Dict[int, Counter] = {}
_table_popularity: Counter = Counter()
_is_trained: bool = False


def _build_model(sequences: List[List[int]]) -> None:
    """
    Build transition model from table sequences.
    For each consecutive pair (A, B) in a sequence, count A→B transitions.
    Also count overall table popularity.
    """
    global _transition_counts, _table_popularity, _is_trained
    
    _transition_counts = defaultdict(Counter)
    _table_popularity = Counter()
    
    for seq in sequences:
        for tid in seq:
            _table_popularity[tid] += 1
        for i in range(len(seq) - 1):
            _transition_counts[seq[i]][seq[i + 1]] += 1
    
    _is_trained = True


async def list_table_compositions(db: AsyncSession, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
    q = select(TableComposition).offset(offset).limit(limit)
    result = await db.execute(q)
    comps = result.scalars().all()
    return {
        "items": [
            {
                "id": c.id,
                "table_ids": c.table_ids,
            }
            for c in comps
        ],
        "limit": limit,
        "offset": offset,
        "returned": len(comps),
    }


async def train(db: AsyncSession) -> Dict[str, Any]:
    """
    Train transition model from TableCompositions in DB.
    """
    result = await db.execute(select(TableComposition))
    comps = result.scalars().all()
    
    sequences = [c.table_ids for c in comps if c.table_ids and len(c.table_ids) >= 2]
    
    if not sequences:
        return {
            "success": False,
            "message": "No table compositions with 2+ tables found. Run /compositions/recover first.",
            "sequences_count": 0,
        }
    
    _build_model(sequences)
    
    unique_tables = set()
    for seq in sequences:
        unique_tables.update(seq)
    
    total_transitions = sum(sum(c.values()) for c in _transition_counts.values())
    
    return {
        "success": True,
        "message": "Table composition model trained",
        "sequences_count": len(sequences),
        "unique_tables": len(unique_tables),
        "total_transitions": total_transitions,
    }


async def predict_next(
    db: AsyncSession,
    table_sequence: List[int],
    n: int = 5,
) -> Dict[str, Any]:
    """
    Predict next table(s) given a sequence.
    
    Uses the trained transition model:
    - looks at the last table in the sequence
    - returns tables that most often follow it in recovered compositions
    - if the last table has no transitions, falls back to global popularity
    """
    global _is_trained, _transition_counts, _table_popularity
    
    if not _is_trained:
        # Auto-train if not trained yet
        train_result = await train(db)
        if not train_result.get("success"):
            return {
                "table_sequence": table_sequence,
                "predictions": [],
                "message": train_result.get("message", "Model not trained"),
            }
    
    if not table_sequence:
        return {
            "table_sequence": table_sequence,
            "predictions": [],
            "message": "Empty sequence",
        }
    
    last_table = table_sequence[-1]
    exclude_set = set(table_sequence)
    
    predictions: List[Dict[str, Any]] = []
    
    # Primary: transitions from last table
    if last_table in _transition_counts:
        transitions = _transition_counts[last_table]
        total = sum(transitions.values())
        
        for next_table, count in transitions.most_common():
            if next_table in exclude_set:
                continue
            predictions.append({
                "table_id": next_table,
                "score": round(count / total, 4) if total > 0 else 0,
                "count": count,
                "type": "transition",
            })
            if len(predictions) >= n:
                break
    
    # Fallback: popular tables not in sequence
    if len(predictions) < n:
        seen = {p["table_id"] for p in predictions}
        total_pop = sum(_table_popularity.values()) or 1
        
        for tid, count in _table_popularity.most_common():
            if tid in exclude_set or tid in seen:
                continue
            predictions.append({
                "table_id": tid,
                "score": round(count / total_pop, 4),
                "count": count,
                "type": "popularity",
            })
            if len(predictions) >= n:
                break
    
    return {
        "table_sequence": table_sequence,
        "predictions": predictions,
        "model": "table_markov_chain",
    }
