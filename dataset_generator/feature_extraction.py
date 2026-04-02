import numpy as np

from neural_models.codebert_utils import get_codebert_embedding, load_codebert_model
from repository.db.graph_label import BlockchainType, GraphEdgeType, GraphNodeType
from repository.db.models import GraphLabel

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

BLOCKCHAIN_TYPE_MAP = {
    BlockchainType.SOURCE.value: 0,
    BlockchainType.DESTINATION.value: 1,
    BlockchainType.OFFCHAIN.value: 2,
}

LABEL_MAP = {
    GraphLabel.NORMAL.value: 0,
    GraphLabel.ANOMALY_SOURCE.value: 1,
    GraphLabel.ANOMALY_OFFCHAIN.value: 2,
    GraphLabel.ANOMALY_DESTINATION.value: 3,
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

def encode_blockchain_types(objects):
    # Use 0 for source, 1 for destination, 2 for offchain, and -1 for unknown/none
    return np.array([BLOCKCHAIN_TYPE_MAP.get(obj.blockchain_type, -1) for obj in objects], dtype=int).reshape(-1, 1)

def encode_blockchains_from_attributes(nodes, key='blockchain'):
    return np.array([BLOCKCHAIN_MAP.get(node.attributes.get(key, None), 0) for node in nodes], dtype=int).reshape(-1, 1)

def encode_amounts(nodes):
    return np.array([float(node.amount) if node.amount is not None else -1.0 for node in nodes], dtype=np.float32).reshape(-1, 1)

def encode_graph_label(label):
    res = LABEL_MAP.get(label)
    if res is None:
        raise ValueError(f"Unknown graph label: {label}")
    return res

def compute_node_degrees(ntype_nodes, edges):
    num_nodes = len(ntype_nodes)
    in_deg = np.zeros(num_nodes, dtype=int)
    out_deg = np.zeros(num_nodes, dtype=int)
    for edge in edges:
        if edge.source_id in [node.node_id for node in ntype_nodes]:
            out_deg[get_node_index_from_id(ntype_nodes, edge.source_id)] += 1
        if edge.target_id in [node.node_id for node in ntype_nodes]:
            in_deg[get_node_index_from_id(ntype_nodes, edge.target_id)] += 1

    return in_deg.reshape(-1, 1), out_deg.reshape(-1, 1)

def get_node_index_from_id(nodes, node_id):
    for i, node in enumerate(nodes):
        if node.node_id == node_id:
            return i
    raise ValueError(f"Node ID {node_id} not found in nodes list")

def compute_node_features_type(ntype_nodes, edges, tokenizer, model, ntype):
    in_deg, out_deg = compute_node_degrees(ntype_nodes, edges)
    feature_elements = [in_deg, out_deg]

    if ntype not in [GraphNodeType.VALIDATOR.value]:
        # In order to prevent 
        feature_elements.append(encode_blockchain_types(ntype_nodes))
    else:
        feature_elements.append(encode_blockchains_from_attributes(ntype_nodes, key="source_chain"))
        feature_elements.append(encode_blockchains_from_attributes(ntype_nodes, key="target_chain"))
    
    if ntype == GraphNodeType.LOG_EVENT.value:
        feature_elements.append(encode_amounts(ntype_nodes))

    if ntype in [GraphNodeType.LOG_EVENT.value, GraphNodeType.TOKEN.value]:
        codebert_embeddings = np.zeros((len(ntype_nodes), 768), dtype=np.float32)
        for i, node in enumerate(ntype_nodes):        
            attributes_text = node.attributes_text if node.attributes_text else None
            codebert_embeddings[i] = get_codebert_embedding(attributes_text, tokenizer, model)
        feature_elements.append(codebert_embeddings)

    return np.concatenate(feature_elements, axis=1)
