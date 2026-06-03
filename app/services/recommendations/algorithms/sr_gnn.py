"""
Sequential SR-GNN-based recommendation algorithm
Predicts next service in a workflow sequence based on Session-based Graph Neural Networks
Adapted for True Graph Representations (Incremental Subgraphs)
"""
import json
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import collections

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GatedGraphConv
from torch_geometric.utils import scatter
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.recommendations.base import RecommendationAlgorithm
from app.services.recommendations.models import Recommendation
from app.services.compositions.recovery import recover_new
from app.core.config import settings


class SRGNNRecommender(nn.Module):
    """SR-GNN-based recommender model adapted for Incremental DAG Subgraphs"""
    
    def __init__(self, num_nodes: int, hidden_channels: int, step: int = 1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_nodes = num_nodes
        
        self.embedding = nn.Embedding(num_nodes, hidden_channels)
        self.ggnn = GatedGraphConv(out_channels=hidden_channels, num_layers=step)
        
        self.w_1 = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.w_2 = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.q = nn.Linear(hidden_channels, 1, bias=False)
        self.w_3 = nn.Linear(2 * hidden_channels, hidden_channels, bias=False)
        
        self.predictor = nn.Linear(hidden_channels, num_nodes)

    def forward(self, x, edge_index, batch, leaves_mask):
        h = self.embedding(x.squeeze(-1) if x.dim() > 1 else x)
        
        # Graph convolution (Message Passing)
        h = self.ggnn(h, edge_index)
        
        # Local intent: mean of "leaf" nodes (active current nodes of the subgraph)
        h_leaves = h * leaves_mask.unsqueeze(-1).float()
        h_local = scatter(h_leaves, batch, dim=0, dim_size=int(batch.max()) + 1, reduce='sum')
        leaf_counts = scatter(leaves_mask.float(), batch, dim=0, dim_size=int(batch.max()) + 1, reduce='sum')
        leaf_counts = leaf_counts.clamp(min=1).unsqueeze(-1)
        # Use index_select to avoid PyTorch returning a view with stride=0,
        # which crashes ONEDNN (CPU backend) with "matmul primitive descriptor" error.
        h_local_expanded = h_local.index_select(0, batch)
        
        # Enforce clean tensors (float32, nan-free)
        h = torch.nan_to_num(h.to(torch.float32))
        h_local_expanded = torch.nan_to_num(h_local_expanded.to(torch.float32))
        
        # Global intent: attention over all nodes in the subgraph
        # Using explicit matmul + transpose, avoiding F.linear to bypass CPU bugs
        term1 = torch.matmul(h_local_expanded, self.w_1.weight.t())
        term2 = torch.matmul(h, self.w_2.weight.t())
        
        # Use sum instead of matmul for [N, 64] @ [64, 1] to avoid ONEDNN 1D matmul crash
        sig = torch.sigmoid(term1 + term2)
        alpha = torch.sum(sig * self.q.weight, dim=-1, keepdim=True)
        
        h_global = scatter(alpha * h, batch, dim=0, dim_size=h_local.size(0), reduce='sum')
        
        # Combined session representation
        h_concat = torch.cat([h_global, h_local], dim=1)
        h_session = torch.matmul(h_concat, self.w_3.weight.t())
        
        # Predict multi-label logits for next nodes
        logits = torch.matmul(h_session, self.predictor.weight.t())
        if hasattr(self.predictor, 'bias') and self.predictor.bias is not None:
            logits = logits + self.predictor.bias
            
        return logits


class SRGNNAlgorithm(RecommendationAlgorithm):
    """
    Session-based Graph Neural Network Algorithm for sequential recommendations.
    Uses True Graph Representation (Subgraphs) to predict next missing edges.
    """
    
    def __init__(
        self,
        db: AsyncSession,
        hidden_channels: int = 64,
        step: int = 1,
        epochs: int = 50,
        learning_rate: float = 0.001
    ):
        super().__init__(name="sr-gnn")
        self.db = db
        self.hidden_channels = hidden_channels
        self.step = step
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        self.model: Optional[SRGNNRecommender] = None
        self.node_map: Optional[Dict] = None
        self.reverse_node_map: Optional[Dict] = None
        
        self.popular_services = []
        self.popular_tables = []
        
        self.model_path = Path("app/static/sr_gnn_model.pth")
        self.metadata_path = Path("app/static/sr_gnn_metadata.pkl")
    
    async def train(self, data=None) -> None:
        """Train SR-GNN on incremental subgraphs"""
        print(f"Training SR-GNN model...")
        db = data if data else self.db
        
        recovery_result = await recover_new(db)
        if not recovery_result.get("success"):
            raise ValueError("Failed to recover compositions")
        
        dag_path = Path(settings.CSV_FILE_PATH).parent / "compositionsDAG.json"
        if not dag_path.exists():
            raise FileNotFoundError(f"Compositions DAG file not found: {dag_path}")
            
        graphs_data, all_nodes = self._extract_incremental_subgraphs(dag_path)
        if len(graphs_data) == 0:
            raise ValueError("No graphs found for SR-GNN")
            
        print(f"Extracted {len(graphs_data)} subgraph sequences for SR-GNN")

        # Encode nodes (0 reserved for PAD/Unknown)
        all_nodes = sorted(list(all_nodes))
        self.node_map = {node: idx + 1 for idx, node in enumerate(all_nodes)}
        self.node_map["<PAD>"] = 0
        self.reverse_node_map = {idx: node for node, idx in self.node_map.items()}

        # Build PyG Data objects
        data_list = []
        for g in graphs_data:
            x = torch.tensor([self.node_map[n] for n in g["nodes"]], dtype=torch.long)
            
            node_to_idx = {n: i for i, n in enumerate(g["nodes"])}
            edge_list = []
            out_degrees = {n: 0 for n in g["nodes"]}
            
            for u, v in g["edges"]:
                edge_list.append([node_to_idx[u], node_to_idx[v]])
                out_degrees[u] += 1
                
            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                
            y_indices = [self.node_map[n] for n in g["targets"]]
            y = torch.zeros(len(self.node_map))
            y[y_indices] = 1.0 # Multi-hot encoding for multi-label prediction
            
            # Identify leaf nodes (active tip of the DAG) for local intent
            leaf_indices = [i for i, n in enumerate(g["nodes"]) if out_degrees[n] == 0]
            if not leaf_indices:
                leaf_indices = [len(g["nodes"]) - 1]
            leaves_mask = torch.zeros(len(g["nodes"]), dtype=torch.bool)
            leaves_mask[leaf_indices] = True
            
            data_list.append(Data(x=x, edge_index=edge_index, y=y.unsqueeze(0), leaves_mask=leaves_mask))

        self._calculate_popularities(graphs_data)
        
        self.model = SRGNNRecommender(
            num_nodes=len(self.node_map),
            hidden_channels=self.hidden_channels,
            step=self.step
        )
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        
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
                
                batch = Batch.from_data_list(batch_graphs)
                
                optimizer.zero_grad()
                logits = self.model(batch.x, batch.edge_index, batch.batch, batch.leaves_mask)
                
                loss = criterion(logits, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"   SR-GNN Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/num_batches:.4f}")
                
        self.is_trained = True
        self._save_model()
        print("✓ SR-GNN trained successfully")
        
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
            
        x = torch.tensor(node_seq, dtype=torch.long)
        edges = [[i, i+1] for i in range(len(node_seq)-1)]
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            
        batch_vector = torch.zeros(len(node_seq), dtype=torch.long)
        
        # Only the last node is the leaf in a linear sequence
        leaves_mask = torch.zeros(len(node_seq), dtype=torch.bool)
        leaves_mask[-1] = True
            
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x, edge_index, batch_vector, leaves_mask)
            probs = torch.sigmoid(logits[0]).numpy()
            
        top_indices = np.argsort(probs)[::-1]
        
        recommendations = []
        for idx in top_indices:
            if idx == 0: continue
            node_name = self.reverse_node_map[idx]
            if not node_name.startswith("service_"):
                continue
                
            service_id = int(node_name.split("_")[1])
            if service_id in exclude_set:
                continue
                
            recommendations.append(Recommendation(
                service_id=service_id,
                score=float(probs[idx]),
                algorithm=self.name,
                confidence=0.8,
                reason="sr_gnn_prediction",
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
        leaves_mask = torch.zeros(len(node_seq), dtype=torch.bool)
        leaves_mask[-1] = True
            
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x, edge_index, batch_vector, leaves_mask)
            probs = torch.sigmoid(logits[0]).numpy()
            
        top_indices = np.argsort(probs)[::-1]
        
        recommendations = []
        for idx in top_indices:
            if idx == 0: continue
            node_name = self.reverse_node_map[idx]
            if not node_name.startswith("table_"):
                continue
                
            table_id = int(node_name.split("_")[1])
            if table_id in exclude_set:
                continue
                
            recommendations.append(Recommendation(
                service_id=table_id,
                score=float(probs[idx]),
                algorithm=self.name,
                confidence=0.7,
                reason="sr_gnn_prediction_table",
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
                continue 
                
            for i in range(1, len(topo_order)):
                executed_nodes = topo_order[:i]
                
                sub_edges = [(u, v) for u, v in local_dag.edges if u in executed_nodes and v in executed_nodes]
                
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
                'hidden_channels': self.hidden_channels,
                'step': self.step,
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
            print(f"Warning: Failed to save SR-GNN model: {e}")
            
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
            self.model = SRGNNRecommender(
                num_nodes=checkpoint['num_nodes'],
                hidden_channels=checkpoint['hidden_channels'],
                step=checkpoint['step']
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Failed to load SR-GNN model: {e}")
            return False
