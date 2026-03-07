"""
Sequential SR-GNN-based recommendation algorithm
Predicts next service in a workflow sequence based on Session-based Graph Neural Networks
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
from sklearn.preprocessing import LabelEncoder
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.recommendations.base import RecommendationAlgorithm
from app.services.recommendations.models import Recommendation
from app.services.compositions.recovery import recover_new
from app.core.config import settings


class SRGNNRecommender(nn.Module):
    """SR-GNN-based recommender model (Simplified)"""
    
    def __init__(self, num_nodes: int, hidden_channels: int, step: int = 1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_nodes = num_nodes
        
        # Node embeddings
        self.embedding = nn.Embedding(num_nodes, hidden_channels)
        
        # Gated Graph Neural Network
        self.ggnn = GatedGraphConv(out_channels=hidden_channels, num_layers=step)
        
        # Attention parameters
        self.w_1 = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.w_2 = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.q = nn.Linear(hidden_channels, 1, bias=False)
        self.w_3 = nn.Linear(2 * hidden_channels, hidden_channels, bias=False)

    def forward(self, x, edge_index, batch):
        # x are node indices
        h = self.embedding(x)
        
        # Graph convolution
        h = self.ggnn(h, edge_index)
        
        # We need to compute local and global session representation
        # Find the last node in each session
        # PyG batch gives us batch indices.
        _, counts = torch.unique(batch, return_counts=True)
        # Calculate indices of last elements
        last_indices = counts.cumsum(dim=0) - 1
        
        h_last = h[last_indices] # Global intent representation (last node)
        
        # Attention
        # Expand h_last to match h's size
        h_last_expanded = h_last[batch]
        
        alpha = self.q(torch.sigmoid(self.w_1(h_last_expanded) + self.w_2(h)))
        
        # Sum with attention weights
        # We can use scatter_add to sum over batches
        from torch_scatter import scatter_add
        h_local = scatter_add(alpha * h, batch, dim=0, dim_size=h_last.size(0))
        
        # Combined representation
        h_g = self.w_3(torch.cat([h_local, h_last], dim=1))
        
        # Compute scores for all nodes
        scores = torch.matmul(h_g, self.embedding.weight.t())
        
        return scores


class SRGNNAlgorithm(RecommendationAlgorithm):
    """
    Session-based Graph Neural Network Algorithm for sequential recommendations.
    Treats each historical sequence as a standalone directed graph.
    """
    
    def __init__(
        self,
        db: AsyncSession,
        hidden_channels: int = 64,
        step: int = 1, # Number of GNN steps
        epochs: int = 100,
        learning_rate: float = 0.001
    ):
        super().__init__(name="sr-gnn")
        self.db = db
        self.hidden_channels = hidden_channels
        self.step = step
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        # Model components
        self.model: Optional[SRGNNRecommender] = None
        self.node_map: Optional[Dict] = None
        self.reverse_node_map: Optional[Dict] = None
        
        # Popularity fallback models
        self.popular_services = []
        self.popular_tables = []
        
        # Cache file paths
        self.model_path = Path("app/static/sr_gnn_model.pth")
        self.metadata_path = Path("app/static/sr_gnn_metadata.pkl")
    
    async def train(self, data=None) -> None:
        """Train SR-GNN on sequence paths"""
        print(f"Training SR-GNN model...")
        db = data if data else self.db
        
        # Recover compositions
        recovery_result = await recover_new(db)
        if not recovery_result.get("success"):
            raise ValueError("Failed to recover compositions")
        
        dag_path = Path(settings.CSV_FILE_PATH).parent / "compositionsDAG.json"
        if not dag_path.exists():
            raise FileNotFoundError(f"Compositions DAG file not found: {dag_path}")
            
        # Instead of global DAG logic, extract full realistic paths.
        paths, all_nodes = self._extract_sequences(dag_path)
        if len(paths) == 0:
            raise ValueError("No paths found for SR-GNN")
            
        print(f"Extracted {len(paths)} sequences for SR-GNN")

        # Encode nodes
        node_encoder = LabelEncoder()
        node_ids = node_encoder.fit_transform(all_nodes)
        self.node_map = {node: idx for node, idx in zip(all_nodes, node_ids)}
        self.reverse_node_map = {idx: node for node, idx in self.node_map.items()}

        # Create session graphs PyG Data objects
        graphs = []
        for path in paths:
            if len(path) < 2:
                continue
            for i in range(1, len(path)):
                # Context is the path up to i
                context = path[:i]
                target = path[i]
                
                # Create edges for the context
                edges = []
                for j in range(len(context) - 1):
                    edges.append([self.node_map[context[j]], self.node_map[context[j+1]]])
                
                if len(edges) == 0:
                    # Single node context
                    edges = [[self.node_map[context[0]], self.node_map[context[0]]]] # self-loop
                    
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                
                # Nodes in the graph are just the unique nodes in the context sequence
                unique_nodes = list(collections.OrderedDict.fromkeys(context))
                local_node_map = {global_id: local_id for local_id, global_id in enumerate([self.node_map[n] for n in unique_nodes])}
                
                # Map edge index to local IDs
                if edge_index.size(0) > 0:
                    edge_index[0] = torch.tensor([local_node_map[n.item()] for n in edge_index[0]])
                    edge_index[1] = torch.tensor([local_node_map[n.item()] for n in edge_index[1]])
                
                x = torch.tensor([self.node_map[n] for n in unique_nodes], dtype=torch.long)
                
                # Target is global id
                y = torch.tensor([self.node_map[target]], dtype=torch.long)
                
                # Need to keep track of the original sequence for attention last node
                item_seq = torch.tensor([local_node_map[self.node_map[n]] for n in context], dtype=torch.long)
                
                data_obj = Data(x=x, edge_index=edge_index, y=y, item_seq=item_seq)
                graphs.append(data_obj)

        if not graphs:
            raise ValueError("No valid session graphs created")
        
        # Populate popularities for fallback
        self._calculate_popularities(paths)
        
        print(f"Created {len(graphs)} session graphs. Training model...")
        
        self.model = SRGNNRecommender(
            num_nodes=len(self.node_map),
            hidden_channels=self.hidden_channels,
            step=self.step
        )
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        
        # Simple training loop with batching
        batch_size = 64
        self.model.train()
        
        from torch_geometric.loader import DataLoader
        loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
        
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in loader:
                optimizer.zero_grad()
                out = self.model(batch.x, batch.edge_index, batch.batch)
                
                # The scatter_add in forward changes order sometimes, but DataLoader batching keeps indices aligned.
                # 'out' is a score vector of size (batch_size, num_nodes)
                loss = F.cross_entropy(out, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f"   SR-GNN Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(loader):.4f}")
                
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
        
        # Convert to string node names
        node_seq = [f"service_{sid}" for sid in sequence if f"service_{sid}" in self.node_map]
        
        if not node_seq:
            return self._get_fallback_recommendations(n, exclude_set, "service")
            
        # Create session graph
        try:
            batch = self._sequence_to_batch(node_seq)
        except Exception:
            return self._get_fallback_recommendations(n, exclude_set, "service")
            
        self.model.eval()
        with torch.no_grad():
            scores = self.model(batch.x, batch.edge_index, batch.batch)
            probs = F.softmax(scores[0], dim=0).numpy()
            
        top_indices = np.argsort(probs)[::-1]
        
        recommendations = []
        for idx in top_indices:
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
        
        node_seq = [f"table_{tid}" for tid in table_sequence if f"table_{tid}" in self.node_map]
        
        if not node_seq:
            return self._get_fallback_recommendations(n, exclude_set, "table")
            
        # Create session graph
        try:
            batch = self._sequence_to_batch(node_seq)
        except Exception:
            return self._get_fallback_recommendations(n, exclude_set, "table")
            
        self.model.eval()
        with torch.no_grad():
            scores = self.model(batch.x, batch.edge_index, batch.batch)
            probs = F.softmax(scores[0], dim=0).numpy()
            
        top_indices = np.argsort(probs)[::-1]
        
        recommendations = []
        for idx in top_indices:
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

    # Helper methods
    def _sequence_to_batch(self, context: List[str]) -> Batch:
        """Convert a sequence of node names to a PyG Batch with 1 item"""
        edges = []
        for j in range(len(context) - 1):
            edges.append([self.node_map[context[j]], self.node_map[context[j+1]]])
            
        if len(edges) == 0:
            edges = [[self.node_map[context[0]], self.node_map[context[0]]]]
            
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        unique_nodes = list(collections.OrderedDict.fromkeys(context))
        local_node_map = {global_id: local_id for local_id, global_id in enumerate([self.node_map[n] for n in unique_nodes])}
        
        if edge_index.size(0) > 0:
            edge_index[0] = torch.tensor([local_node_map[n.item()] for n in edge_index[0]])
            edge_index[1] = torch.tensor([local_node_map[n.item()] for n in edge_index[1]])
            
        x = torch.tensor([self.node_map[n] for n in unique_nodes], dtype=torch.long)
        batch_vec = torch.zeros(len(unique_nodes), dtype=torch.long)
        
        return Batch(x=x, edge_index=edge_index, batch=batch_vec)
        
    def _extract_sequences(self, json_path: Path) -> Tuple[List[List[str]], List[str]]:
        """Extract historical linear sequences from compositions"""
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
                
            # Build a local DAG just for this composition
            local_dag = nx.DiGraph()
            for link in composition["links"]:
                src = id_to_mid[str(link["source"])]
                tgt = id_to_mid[str(link["target"])]
                local_dag.add_edge(src, tgt)
                
            # Find paths in this local DAG
            for start_node in [n for n in local_dag.nodes if local_dag.in_degree(n) == 0]:
                for end_node in [n for n in local_dag.nodes if local_dag.out_degree(n) == 0]:
                    if start_node != end_node and nx.has_path(local_dag, start_node, end_node):
                        for path in nx.all_simple_paths(local_dag, source=start_node, target=end_node):
                            if len(path) > 1:
                                paths.append(path)
                        
        return paths, list(all_nodes)
        
    def _calculate_popularities(self, paths: List[List[str]]):
        """Fallback list of popular items"""
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
        """Return popular items as fallback"""
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
