import numpy as np

from torch_geometric.data import HeteroData, Dataset
from dataset_generator.feature_extraction import NODE_TYPE_MAP, EDGE_TYPE_MAP, get_node_index_from_id, compute_node_features, compute_edge_features
import torch
from neural_models.codebert_utils import load_codebert_model
from repository.database import DBSession
from repository.db.models import GraphEdge, GraphNode
from repository.db.repository import GraphEdgeRepository, GraphMappingCrossChainRepository, GraphNodeRepository
from utils.utils import CustomException, load_module
import os


class GraphDatasetGenerator:
    CLASS_NAME = "GraphDatasetGenerator"
    
    def __init__(self, bridges, output_file):
        self.bridges = bridges
        self.output_file = output_file
        self.load_db_modules()
        self.load_repositories()

    def load_db_modules(self):
        function_name = "load_db_models"

        try:
            load_module("repository.db")

            # It's expected that the database tables have been created
            # by XChainDataGen, so we don't need to call create_tables() here.
        except Exception as e:
            raise CustomException(GraphDatasetGenerator.CLASS_NAME, function_name, "Failed to load database module") from e
        
    def load_repositories(self):
        self.session = DBSession()
        self.cross_chain_mapping_repo = GraphMappingCrossChainRepository(DBSession)
        self.blockchain_mapping_repo = GraphMappingCrossChainRepository(DBSession)
        self.graph_nodes_repo = GraphNodeRepository(DBSession)
        self.graph_edges_repo = GraphEdgeRepository(DBSession)

    def generate_graph_dataset(self):
        storage_folder = os.path.join(os.path.dirname(__file__), "data")
        dataset = Dataset(root=storage_folder)

        # Load the graphs in batches to avoid memory issues
        batch_size = 50
        offset = 0

        tokenizer, model = load_codebert_model()
        total_node_features = []
        while True:
            cctx_mappings = self.cross_chain_mapping_repo.get_batch(batch_size, offset)
            if not cctx_mappings or len(cctx_mappings) == 0:
                break

            for cctx in cctx_mappings:
                nodes = self.graph_nodes_repo.get_by_cctx_graph_id(cctx.cctx_graph_id)
                edges = self.graph_edges_repo.get_by_cctx_graph_id(cctx.cctx_graph_id)

                graph_data = self.build_heterogeneous_graph(nodes, edges, tokenizer, model)
                # TODO Save to the dataset

            # Get all nodes and edges for the batch of graphs and compute features
            nodes = self.session.query(GraphNode).filter(GraphNode.cctx_graph_id.in_([cctx.cctx_graph_id for cctx in cctx_mappings])).all()
            edges = self.session.query(GraphEdge).filter(GraphEdge.cctx_graph_id.in_([cctx.cctx_graph_id for cctx in cctx_mappings])).all()

            node_features = compute_node_features(nodes, edges, tokenizer, model)
            total_node_features.append(node_features)

            offset += batch_size
        
        # Combine all node features into a single array
        total_node_features = np.concatenate(total_node_features, axis=0)

    def build_heterogeneous_graph(self, nodes: list[GraphNode], edges: list[GraphEdge], tokenizer, model, label=None) -> HeteroData:
        """
        Build a torch-geometric HeteroData object for a single cross-chain graph.
        nodes: list of GraphNode objects
        edges: list of GraphEdge objects
        tokenizer, model: CodeBERT tokenizer and model
        label: optional graph-level label
        """
        data = HeteroData()

        # 1. Node features by type
        node_type_to_indices = {ntype: [] for ntype in NODE_TYPE_MAP}
        for idx, node in enumerate(nodes):
            node_type_to_indices[node.node_type].append(idx)

        node_features = compute_node_features(nodes, edges, tokenizer, model)
        for ntype, indices in node_type_to_indices.items():
            if not indices:
                continue
            feats = torch.tensor(node_features[indices], dtype=torch.float)
            data[ntype].x = feats

        # 2. Edge indices and features by type
        edge_types_to_indices = {}
        edge_types_to_node_indexes = {}
        edge_types_to_attrs = {}
        for idx, edge in enumerate(edges):
            edge_type = edge.edge_type
            src_idx = get_node_index_from_id(nodes, edge.source_id)
            src_type = nodes[src_idx].node_type
            dst_idx = get_node_index_from_id(nodes, edge.target_id)
            dst_type = nodes[dst_idx].node_type

            edge_attr = [EDGE_TYPE_MAP[edge_type], 0]  # Add more features as needed

            edge_types_to_indices.setdefault((src_type, edge_type, dst_type), []).append(idx)
            edge_types_to_node_indexes.setdefault((src_type, edge_type, dst_type), []).append(np.array([[src_idx], [dst_idx]]))
            edge_types_to_attrs.setdefault((src_type, edge_type, dst_type), []).append(np.array(edge_attr))

        for etype, node_indexes in edge_types_to_node_indexes.items():
            if not node_indexes:
                continue
            edge_index = torch.tensor(np.hstack(node_indexes), dtype=torch.long)
            data[etype].edge_index = edge_index

        for etype, attrs in edge_types_to_attrs.items():
            if not attrs:
                continue
            data[etype].edge_attr = torch.tensor(np.vstack(attrs), dtype=torch.float)

        # 3. Graph-level label
        if label is not None:
            data['label'] = torch.tensor([label], dtype=torch.long)

        return data