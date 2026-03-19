"""
Sequential Transformer-based recommendation algorithm (GraphGPS Approach)
Predicts next service in a workflow sequence using SOTA Graph Transformers.
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
from sqlalchemy.ext.asyncio import AsyncSession

# PyTorch Geometric imports for SOTA Graph Transformer
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GPSConv, GINConv, global_mean_pool

from app.services.recommendations.base import RecommendationAlgorithm
from app.services.recommendations.models import Recommendation
from app.services.compositions.recovery import recover_new
from app.core.config import settings

class GraphGPSTransformerRecommender(nn.Module):
    """
    SOTA Graph Transformer Model (GraphGPS)
    Combines local Message Passing (GINEConv) with global Multi-Head Attention.
    """
    
    def __init__(self, num_nodes: int, d_model: int = 64, num_layers: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Node embedding layer
        self.embedding = nn.Embedding(num_nodes, d_model)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            # Local MPNN
            nn_local = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model)
            )
            local_conv = GINConv(nn_local)
            
            # SOTA GPS Layer combining MPNN and Global Attention
            gps = GPSConv(
                channels=d_model,
                conv=local_conv,
                heads=heads,
                dropout=dropout,
                attn_type='multihead'  # Standard PyTorch MHA
            )
            self.layers.append(gps)
            
        # Classifier producing logits for all possible next nodes
        self.predictor = nn.Linear(d_model, num_nodes)

    def forward(self, x, edge_index, batch):
        # x is (num_total_nodes_in_batch,)
        h = self.embedding(x.squeeze(-1) if x.dim() > 1 else x)
        
        for layer in self.layers:
            h = layer(h, edge_index, batch=batch)
            
        # Readout: Collapse the entire subgraph into one graph embedding
        hz = global_mean_pool(h, batch)
        
        # Predict multi-label logits for next nodes
        logits = self.predictor(hz)
        return logits


class DAGTransformerAlgorithm(RecommendationAlgorithm):
    """
    Graph Transformer Algorithm for sequential recommendations.
    Uses True Graph Representation (Subgraphs + GraphGPS) to predict next missing edges.
    """
    
    def __init__(
        self,
        db: AsyncSession,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        epochs: int = 50,
        learning_rate: float = 0.001
    ):
        super().__init__(name="dag-transformer")
        self.db = db
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        self.model: Optional[GraphGPSTransformerRecommender] = None
        self.node_map: Optional[Dict] = None
        self.reverse_node_map: Optional[Dict] = None
        
        self.popular_services = []
        self.popular_tables = []
        
        self.model_path = Path("app/static/dag_transformer_model.pth")
        self.metadata_path = Path("app/static/dag_transformer_metadata.pkl")
    
    async def train(self, data=None) -> None:
        """Train GraphGPS Model on incremental subgraphs"""
        print(f"Training DAGTransformer (GraphGPS) model...")
        db = data if data else self.db
        
        recovery_result = await recover_new(db)
        if not recovery_result.get("success"):
            raise ValueError("Failed to recover compositions")
        
        dag_path = Path(settings.CSV_FILE_PATH).parent / "compositionsDAG.json"
        if not dag_path.exists():
            raise FileNotFoundError(f"Compositions DAG file not found: {dag_path}")
            
        # Generate true subgraphs
        graphs_data, all_nodes = self._extract_incremental_subgraphs(dag_path)
        if len(graphs_data) == 0:
            raise ValueError("No graph pairs found for DAGTransformer")
            
        print(f"Extracted {len(graphs_data)} incremental subgraphs for GraphGPS")

        # 0 is reserved for padding/unknown
        all_nodes = sorted(list(all_nodes))
        self.node_map = {node: idx + 1 for idx, node in enumerate(all_nodes)}
        self.node_map["<PAD>"] = 0
        self.reverse_node_map = {idx: node for node, idx in self.node_map.items()}

        # Build PyG Data objects
        data_list = []
        for g in graphs_data:
            # Map nodes to IDs
            x = torch.tensor([self.node_map[n] for n in g["nodes"]], dtype=torch.long)
            
            # Map edges to local relative index within the subgraph
            node_to_idx = {n: i for i, n in enumerate(g["nodes"])}
            edge_list = []
            if not edge_list:
                for i in range(len(g["nodes"])):
                    edge_list.append([i, i])
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                
            y_indices = [self.node_map[n] for n in g["targets"]]
            y = torch.zeros(len(self.node_map))
            y[y_indices] = 1.0 # Multi-hot encoding for Multi-Label target
            
            data_list.append(Data(x=x, edge_index=edge_index, y=y.unsqueeze(0)))
            
        self._calculate_popularities(graphs_data)
        
        self.model = GraphGPSTransformerRecommender(
            num_nodes=len(self.node_map),
            d_model=self.d_model,
            heads=self.nhead,
            num_layers=self.num_layers
        )
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        
        # Batch preparation logic using PyG DataLoader principles
        batch_size = 32
        self.model.train()
        
        num_batches = int(np.ceil(len(data_list) / batch_size))
        indices = np.arange(len(data_list))
        
        for epoch in range(self.epochs):
            np.random.shuffle(indices)
            total_loss = 0
            
            for b_idx in range(num_batches):
                batch_indices = indices[b_idx * batch_size:(b_idx + 1) * batch_size]
                batch_graphs = [data_list[i] for i in batch_indices]
                
                # Use PyG Batch to collate disjoint graphs
                batch = Batch.from_data_list(batch_graphs)
                
                optimizer.zero_grad()
                logits = self.model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(logits, batch.y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"   DAGTransformer (GraphGPS) Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/num_batches:.4f}")
                
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
            
        # Reconstruct subgraph from plain sequence for inference.
        # Without full graph context, we assume a chain for the given flat sequence.
        x = torch.tensor(node_seq, dtype=torch.long)
        edges = [[i, i+1] for i in range(len(node_seq)-1)]
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            
        batch_vector = torch.zeros(len(node_seq), dtype=torch.long)
            
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x, edge_index, batch_vector)
            # Use Sigmoid for multi-label probabilities
            probs = torch.sigmoid(logits[0]).numpy()
            
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
                reason="gps_transformer_prediction",
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
            
        x = torch.tensor(node_seq, dtype=torch.long)
        edges = [[i, i+1] for i in range(len(node_seq)-1)]
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            
        batch_vector = torch.zeros(len(node_seq), dtype=torch.long)
            
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x, edge_index, batch_vector)
            probs = torch.sigmoid(logits[0]).numpy()
            
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
                reason="gps_transformer_prediction_table",
                metadata={"table_sequence_length": len(table_sequence), "type": "table"}
            ))
            
            if len(recommendations) >= n:
                break
                
        return recommendations

    async def recommend(self, user_id: str, n: int = 10, exclude_services: Optional[List[int]] = None) -> List[Recommendation]:
        return []

    def _extract_incremental_subgraphs(self, json_path: Path) -> Tuple[List[Dict], List[str]]:
        """
        True Graph Representation Builder
        Generates snapshots of developing subgraphs and targets next execution nodes.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        training_graphs = []
        all_nodes = set()

        for composition in data:
            id_to_mid = {}
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
                
            try:
                topo_order = list(nx.topological_sort(local_dag))
            except nx.NetworkXUnfeasible:
                continue # Skip cyclic graphs
                
            for i in range(1, len(topo_order)):
                executed_nodes = topo_order[:i]
                
                # Active subgraph: edges between currently executed nodes only
                sub_edges = [(u, v) for u, v in local_dag.edges if u in executed_nodes and v in executed_nodes]
                
                # Targets: Valid next nodes are those whose parents are all in executed_nodes
                # (Meaning they are fully ready to be executed now)
                valid_next = [
                    n for n in local_dag.nodes 
                    if n not in executed_nodes and all(pred in executed_nodes for pred in local_dag.predecessors(n))
                ]
                
                if not valid_next:
                    continue
                    
                training_graphs.append({
                    "nodes": list(executed_nodes),
                    "edges": list(sub_edges),
                    "targets": list(valid_next)
                })
                        
        return training_graphs, list(all_nodes)
        
    def _extract_sequences(self, json_path: Path) -> Tuple[List[List[str]], List[str]]:
        """Legacy sequence flattener (kept for backward compatibility)"""
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
        
    def _calculate_popularities(self, graphs: List[Dict]):
        services_counter = collections.Counter()
        tables_counter = collections.Counter()
        
        for g in graphs:
            for item in g["targets"]:
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
            self.model = GraphGPSTransformerRecommender(
                num_nodes=checkpoint['num_nodes'],
                d_model=checkpoint['d_model'],
                heads=checkpoint['nhead'],
                num_layers=checkpoint['num_layers']
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Failed to load DAGTransformer model: {e}")
            return False
