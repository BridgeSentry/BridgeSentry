import numpy as np

from nn_models.codebert_utils import get_codebert_embedding, load_codebert_model
from repository.db.models import GraphEdgeType, GraphNodeType

NODE_TYPE_MAP = {
    GraphNodeType.USER.value: 0,
    GraphNodeType.ROUTER.value: 1,
    GraphNodeType.TOKEN.value: 2,
    GraphNodeType.OTHER_ACCOUNT.value: 3,
    GraphNodeType.LOG_EVENT.value: 4,
    GraphNodeType.VALIDATOR.value: 5,
}

EDGE_TYPE_MAP = {
    GraphEdgeType.TRANSACTION.value: 0,
    GraphEdgeType.TOKEN_TRANSFER.value: 1,
    GraphEdgeType.TOKEN_AUTH.value: 2,
    GraphEdgeType.FUNCTION_CALL.value: 3,
    GraphEdgeType.LOG_RELATION.value: 4,
    GraphEdgeType.CROSS_CHAIN_RELATION.value: 5,
}

BLOCKCHAIN_MAP = {
    "ethereum": 1,
    "bsc": 2,
    "polygon": 3,
    "avalanche": 4,
    "arbitrum": 5,
    "optimism": 6,
    "solana": 7,
    "base": 8,
    "scroll": 9,
    "linea": 10,
    "gnosis": 11,
    "ronin": 12,
    "unichain": 13,
}

def encode_node_types(nodes):
    # Using -1 for unknown node types
    return np.array([NODE_TYPE_MAP.get(node.node_type, -1) for node in nodes], dtype=int).reshape(-1, 1)

def encode_edge_type(edge_type: str) -> int:
    res = EDGE_TYPE_MAP.get(edge_type)
    if res is None:
        raise ValueError(f"Unknown edge type: {edge_type}")
    return res

def encode_edge_types(edges):
    return np.array([encode_edge_type(edge.edge_type) for edge in edges], dtype=int).reshape(-1, 1)

def encode_blockchains(objects):
    # Onjects can be either nodes or edges, both have a 'blockchain' attribute.
    # Using 0 for unknown/none blockchains
    return np.array([BLOCKCHAIN_MAP.get(obj.blockchain, 0) for obj in objects], dtype=int).reshape(-1, 1)

def compute_node_degrees(nodes, edges):
    num_nodes = len(nodes)
    in_deg = np.zeros(num_nodes, dtype=int)
    out_deg = np.zeros(num_nodes, dtype=int)
    for edge in edges:
        out_deg[get_node_index_from_id(nodes, edge.source_id)] += 1
        in_deg[get_node_index_from_id(nodes, edge.target_id)] += 1
    return in_deg.reshape(-1, 1), out_deg.reshape(-1, 1)

def get_node_index_from_id(nodes, node_id):
    for i, node in enumerate(nodes):
        if node.node_id == node_id:
            return i
    raise ValueError(f"Node ID {node_id} not found in nodes list")

def compute_node_features(nodes, edges, tokenizer, model):
    in_deg, out_deg = compute_node_degrees(nodes, edges)
    node_types_encoded = encode_node_types(nodes)
    node_blockchains_encoded = encode_blockchains(nodes)

    codebert_embeddings = np.zeros((len(nodes), 768), dtype=np.float32)
    for i, node in enumerate(nodes):        
        attributes_text = node.attributes_text if node.attributes_text else None
        codebert_embeddings[i] = get_codebert_embedding(attributes_text, tokenizer, model)

    features = np.concatenate((node_types_encoded, node_blockchains_encoded, codebert_embeddings, in_deg, out_deg), axis=1)
    return features

def compute_edge_features(edges):
    edge_types_encoded = encode_edge_types(edges)
    blockchain_encoded = encode_blockchains(edges)
    return np.concatenate((edge_types_encoded, blockchain_encoded), axis=1)
