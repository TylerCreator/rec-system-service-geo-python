"""
Sequential Transformer-based recommendation algorithm (Standard NLP Approach)
Predicts next service in a workflow sequence using Self-Attention.
"""
import json
import pickle
import math
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import collections

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.recommendations.base import RecommendationAlgorithm
from app.services.recommendations.models import Recommendation
from app.services.compositions.recovery import recover_new
from app.core.config import settings

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Adding batch dimension for shape (1, max_len, d_model)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (batch_size, seq_len, d_model)"""
        return x + self.pe[:, :x.size(1), :]

class SequenceTransformerRecommender(nn.Module):
    """Sequence Transformer Model"""
    
    def __init__(self, num_nodes: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(num_nodes, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.predictor = nn.Linear(d_model, num_nodes)

    def forward(self, src: torch.Tensor, src_pad_mask: torch.Tensor = None):
        """
        src shape: (batch_size, seq_len)
        src_pad_mask shape: (batch_size, seq_len) - True for padding tokens
        """
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        
        # Transformer output shape: (batch_size, seq_len, d_model)
        output = self.transformer_encoder(src_emb, src_key_padding_mask=src_pad_mask)
        
        # Use the representation of the last step to predict next
        # (batch_size, d_model)
        # However, because of padding, we must select the valid last token representation.
        # But for simpler training, we can just use training sequences without padding if batch=1,
        # or properly select. Let's assume input is padded with 0 and we find lengths.
        
        if src_pad_mask is not None:
            lengths = (~src_pad_mask).sum(dim=1) - 1
        else:
            lengths = torch.full((src.size(0),), src.size(1) - 1, dtype=torch.long, device=src.device)
            
        # Gather the last non-padded output for each sequence in the batch
        batch_idx = torch.arange(src.size(0), device=src.device)
        last_outputs = output[batch_idx, lengths, :]
        
        logits = self.predictor(last_outputs)
        return logits


class DAGTransformerAlgorithm(RecommendationAlgorithm):
    """
    Sequence Transformer Algorithm for sequential recommendations.
    Uses Standard NLP Approach (Self-Attention over linear sequence of tasks).
    Bypasses the cycle problem perfectly by treating workflow as an ordered sequence.
    """
    
    def __init__(
        self,
        db: AsyncSession,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        epochs: int = 100,
        learning_rate: float = 0.001
    ):
        super().__init__(name="dag-transformer")
        self.db = db
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        self.model: Optional[SequenceTransformerRecommender] = None
        self.node_map: Optional[Dict] = None
        self.reverse_node_map: Optional[Dict] = None
        
        self.popular_services = []
        self.popular_tables = []
        
        self.model_path = Path("app/static/dag_transformer_model.pth")
        self.metadata_path = Path("app/static/dag_transformer_metadata.pkl")
    
    async def train(self, data=None) -> None:
        """Train Sequence Transformer Model on paths"""
        print(f"Training DAGTransformer model...")
        db = data if data else self.db
        
        recovery_result = await recover_new(db)
        if not recovery_result.get("success"):
            raise ValueError("Failed to recover compositions")
        
        dag_path = Path(settings.CSV_FILE_PATH).parent / "compositionsDAG.json"
        if not dag_path.exists():
            raise FileNotFoundError(f"Compositions DAG file not found: {dag_path}")
            
        paths, all_nodes = self._extract_sequences(dag_path)
        if len(paths) == 0:
            raise ValueError("No paths found for DAGTransformer")
            
        print(f"Extracted {len(paths)} sequences for DAGTransformer")

        # 0 is reserved for padding, so encode from 1.
        all_nodes = sorted(list(all_nodes)) # Ensure deterministic order
        self.node_map = {node: idx + 1 for idx, node in enumerate(all_nodes)}
        self.node_map["<PAD>"] = 0
        self.reverse_node_map = {idx: node for node, idx in self.node_map.items()}

        # Create training pairs (context, target)
        X = []
        Y = []
        for path in paths:
            if len(path) < 2:
                continue
            for i in range(1, len(path)):
                context = [self.node_map[n] for n in path[:i]]
                target = self.node_map[path[i]]
                X.append(context)
                Y.append(target)
                
        if not X:
            raise ValueError("No training pairs generated.")
            
        self._calculate_popularities(paths)
        
        self.model = SequenceTransformerRecommender(
            num_nodes=len(self.node_map),
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers
        )
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Batch preparation logic
        batch_size = 64
        self.model.train()
        
        num_batches = int(np.ceil(len(X) / batch_size))
        indices = np.arange(len(X))
        
        for epoch in range(self.epochs):
            np.random.shuffle(indices)
            total_loss = 0
            
            for b_idx in range(num_batches):
                batch_indices = indices[b_idx * batch_size:(b_idx + 1) * batch_size]
                
                # Get lengths and find max_length
                batch_X_raw = [X[i] for i in batch_indices]
                batch_y = torch.tensor([Y[i] for i in batch_indices], dtype=torch.long)
                
                max_len = max(len(seq) for seq in batch_X_raw)
                
                # Pad sequences
                batch_X_padded = []
                batch_mask = []
                for seq in batch_X_raw:
                    pad_len = max_len - len(seq)
                    padded_seq = seq + [0] * pad_len
                    mask = [False] * len(seq) + [True] * pad_len
                    
                    batch_X_padded.append(padded_seq)
                    batch_mask.append(mask)
                    
                batch_X = torch.tensor(batch_X_padded, dtype=torch.long)
                batch_pad_mask = torch.tensor(batch_mask, dtype=torch.bool)
                
                optimizer.zero_grad()
                logits = self.model(batch_X, src_pad_mask=batch_pad_mask)
                loss = F.cross_entropy(logits, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f"   DAGTransformer Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/num_batches:.4f}")
                
        self.is_trained = True
        self._save_model()
        print("✓ DAGTransformer trained successfully")

    def predict_next(
        self,
        sequence: List[int],
        n: int = 5,
        exclude_services: Optional[List[int]] = None
    ) -> List[Recommendation]:
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        exclude_set = set(exclude_services) if exclude_services else set()
        exclude_set.update(sequence)
        
        node_seq = [self.node_map[f"service_{sid}"] for sid in sequence if f"service_{sid}" in self.node_map]
        
        if not node_seq:
            return self._get_fallback_recommendations(n, exclude_set, "service")
            
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor([node_seq], dtype=torch.long)
            logits = self.model(x, src_pad_mask=None)
            probs = F.softmax(logits[0], dim=0).numpy()
            
        top_indices = np.argsort(probs)[::-1]
        
        recommendations = []
        for idx in top_indices:
            if idx == 0: continue # PAD token
            node_name = self.reverse_node_map[idx]
            if not node_name.startswith("service_"): continue
                
            service_id = int(node_name.split("_")[1])
            if service_id in exclude_set: continue
                
            recommendations.append(Recommendation(
                service_id=service_id,
                score=float(probs[idx]),
                algorithm=self.name,
                confidence=0.8,
                reason="transformer_prediction",
                metadata={"sequence_length": len(sequence)}
            ))
            
            if len(recommendations) >= n:
                break
                
        return recommendations
        
    def predict_next_table(
        self,
        table_sequence: List[int],
        n: int = 5,
        exclude_tables: Optional[List[int]] = None
    ) -> List[Recommendation]:
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        exclude_set = set(exclude_tables) if exclude_tables else set()
        exclude_set.update(table_sequence)
        
        node_seq = [self.node_map[f"table_{tid}"] for tid in table_sequence if f"table_{tid}" in self.node_map]
        
        if not node_seq:
            return self._get_fallback_recommendations(n, exclude_set, "table")
            
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor([node_seq], dtype=torch.long)
            logits = self.model(x, src_pad_mask=None)
            probs = F.softmax(logits[0], dim=0).numpy()
            
        top_indices = np.argsort(probs)[::-1]
        
        recommendations = []
        for idx in top_indices:
            if idx == 0: continue
            node_name = self.reverse_node_map[idx]
            if not node_name.startswith("table_"): continue
                
            table_id = int(node_name.split("_")[1])
            if table_id in exclude_set: continue
                
            recommendations.append(Recommendation(
                service_id=table_id,
                score=float(probs[idx]),
                algorithm=self.name,
                confidence=0.7,
                reason="transformer_prediction_table",
                metadata={"table_sequence_length": len(table_sequence), "type": "table"}
            ))
            
            if len(recommendations) >= n:
                break
                
        return recommendations

    async def recommend(self, user_id: str, n: int = 10, exclude_services: Optional[List[int]] = None) -> List[Recommendation]:
        return []

    def _extract_sequences(self, json_path: Path) -> Tuple[List[List[str]], List[str]]:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        paths = []
        all_nodes = set()
        id_to_mid = {}

        for composition in data:
            for node in composition["nodes"]:
                node_id = str(node["id"])
                if node.get("mid") is not None:
                    node_name = f"service_{node['mid']}"
                else:
                    node_name = f"table_{node['id']}"
                    
                id_to_mid[node_id] = node_name
                all_nodes.add(node_name)
                
            local_dag = nx.DiGraph()
            for link in composition["links"]:
                src = id_to_mid[str(link["source"])]
                tgt = id_to_mid[str(link["target"])]
                local_dag.add_edge(src, tgt)
                
            for start_node in [n for n in local_dag.nodes if local_dag.in_degree(n) == 0]:
                for end_node in [n for n in local_dag.nodes if local_dag.out_degree(n) == 0]:
                    if start_node != end_node and nx.has_path(local_dag, start_node, end_node):
                        for path in nx.all_simple_paths(local_dag, source=start_node, target=end_node):
                            if len(path) > 1:
                                paths.append(path)
                        
        return paths, list(all_nodes)
        
    def _calculate_popularities(self, paths: List[List[str]]):
        services_counter = collections.Counter()
        tables_counter = collections.Counter()
        
        for path in paths:
            for item in path:
                if item.startswith("service_"):
                    services_counter[int(item.split("_")[1])] += 1
                elif item.startswith("table_"):
                    tables_counter[int(item.split("_")[1])] += 1
                    
        self.popular_services = [val[0] for val in services_counter.most_common()]
        self.popular_tables = [val[0] for val in tables_counter.most_common()]
        
    def _get_fallback_recommendations(self, n: int, exclude_set: set, item_type: str) -> List[Recommendation]:
        recs = []
        popular_list = self.popular_services if item_type == "service" else self.popular_tables
        
        for item_id in popular_list:
            if item_id in exclude_set:
                continue
            recs.append(Recommendation(
                service_id=item_id,
                score=0.5,
                algorithm=self.name,
                confidence=0.5,
                reason="popular_start",
                metadata={"type": item_type}
            ))
            if len(recs) >= n:
                break
        return recs
        
    def _save_model(self):
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'd_model': self.d_model,
                'nhead': self.nhead,
                'num_layers': self.num_layers,
                'num_nodes': len(self.node_map)
            }, self.model_path)
            
            with open(self.metadata_path, 'wb') as f:
                pickle.dump({
                    'node_map': self.node_map,
                    'reverse_node_map': self.reverse_node_map,
                    'popular_services': self.popular_services,
                    'popular_tables': self.popular_tables
                }, f)
        except Exception as e:
            print(f"Warning: Failed to save DAGTransformer model: {e}")
            
    def _load_model(self) -> bool:
        try:
            if not self.model_path.exists() or not self.metadata_path.exists():
                return False
                
            with open(self.metadata_path, 'rb') as f:
                md = pickle.load(f)
                
            self.node_map = md['node_map']
            self.reverse_node_map = md['reverse_node_map']
            self.popular_services = md['popular_services']
            self.popular_tables = md['popular_tables']
            
            checkpoint = torch.load(self.model_path)
            self.model = SequenceTransformerRecommender(
                num_nodes=checkpoint['num_nodes'],
                d_model=checkpoint['d_model'],
                nhead=checkpoint['nhead'],
                num_layers=checkpoint['num_layers']
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Failed to load DAGTransformer model: {e}")
            return False
