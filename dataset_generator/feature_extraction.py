import numpy as np
import mmh3
from pandas import DataFrame

from neural_models.codebert_utils import get_codebert_embedding
from repository.db.graph_label import BlockchainType, EventType, GraphEdgeType, GraphLabel, GraphNodeType
import json

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

# Binary encoding for blockchains
BLOCKCHAIN_MAP = {
    "ethereum": [0, 0, 0, 1],
    "bsc": [0, 0, 1, 0],
    "polygon": [0, 0, 1, 1],
    "avalanche": [0, 1, 0, 0],
    "arbitrum": [0, 1, 0, 1],
    "optimism": [0, 1, 1, 0],
    "solana": [0, 1, 1, 1],
    "base": [1, 0, 0, 0],
    "scroll": [1, 0, 0, 1],
    "linea": [1, 0, 1, 0],
    "gnosis": [1, 0, 1, 1],
    "ronin": [1, 1, 0, 0],
    "unichain": [1, 1, 0, 1],
}

BLOCKCHAIN_STAGE_MAP = {
    BlockchainType.SOURCE.value: 0,
    BlockchainType.DESTINATION.value: 1,
    BlockchainType.OFFCHAIN.value: 2,
}

EVENT_TYPE_MAP = {
    EventType.TRANSFER.value: [0, 0, 0, 0],
    EventType.APPROVAL.value: [0, 0, 0, 1],
    EventType.BURN.value: [0, 0, 1, 0],
    EventType.MINT.value: [0, 0, 1, 1],
    EventType.OPERATION_REQUEST_SIGNING.value: [0, 1, 0, 0],
    EventType.OPERATION_FINALIZED.value: [0, 1, 0, 1],
    EventType.DEPOSIT_REQUEST.value: [1, 0, 0, 0],
    EventType.DEPOSIT_CONFIRMATION.value: [1, 0, 0, 1],
    EventType.WITHDRAWAL_REQUEST.value: [1, 0, 1, 0],
    EventType.WITHDRAWAL_CONFIRMATION.value: [1, 0, 1, 1],
    EventType.ROUTER_UNKNOWN.value: [1, 1, 0, 0],
    EventType.TOKEN_UNKNOWN.value: [1, 1, 0, 1],
    EventType.UNKNOWN.value: [1, 1, 1, 1],
}

LABEL_MAP = {
    GraphLabel.NORMAL.value: 0,
    GraphLabel.ANOMALY_SOURCE.value: 1,
    GraphLabel.ANOMALY_OFFCHAIN.value: 2,
    GraphLabel.ANOMALY_DESTINATION.value: 3,
}

IN_DEGREE_INDEX = 0
OUT_DEGREE_INDEX = 1
ARGS_NUM_INDEX = 14
INPUT_SIZE_INDEX = 15
AMOUNTS_INDEX = 16

class FeatureExtractor:
    def __init__(self, nodes_df: DataFrame, edges_df: DataFrame):
        self.nodes_df = nodes_df
        self.edges_df = edges_df

        self.calculate_common_stats()
        #self.tokenizer, self.model = load_codebert_model()

    def calculate_common_stats(self):
        self.max_in_degree = self.nodes_df['in_degree'].max()
        self.max_out_degree = self.nodes_df['out_degree'].max()
        #print(f"Max in degree: {self.max_in_degree}, Max out degree: {self.max_out_degree}")


    # ======== Common node feature encoding methods ========
    def compute_degrees(self, ntype_nodes: list):
        return np.array([node.in_degree for node in ntype_nodes], dtype=np.float32).reshape(-1, 1), \
            np.array([node.out_degree for node in ntype_nodes], dtype=np.float32).reshape(-1, 1)

    def encode_node_blockchains(self, ntype_nodes: list):
        # Use a binary encoding for blockchains, with one bit for each blockchain type
        # (this is possible since we have a fixed set of blockchains in our dataset)
        return np.array([BLOCKCHAIN_MAP.get(node.blockchain, [0, 0, 0, 0]) for node in ntype_nodes])

    def encode_blockchain_stages(self, objects):
        # Use a one-hot encoding for source/offchain/destination
        return np.array([
            [1 if obj.blockchain_type == b_type else 0 for b_type in BLOCKCHAIN_STAGE_MAP.keys()]
            for obj in objects], dtype=int)

    def compute_token_symbols(self, nodes):
        # Use MurmurHash3 to hash the token symbol string into a fixed-size integer, and then split into a set of 8 features, normalizing each one to be between 0 and 1
        token_symbol_features = np.zeros((len(nodes), 8), dtype=np.float32)
        for i, node in enumerate(nodes):
            if node.token_symbol is not None and isinstance(node.token_symbol, str):
                hash_value = mmh3.hash(node.token_symbol, 12, signed=False) # Returns a 32-bit unsigned int
                for j in range(8):
                    # Extract 4 bits at a time to create 8 features, and normalize to [0, 1]
                    token_symbol_features[i, j] = ((hash_value >> (j * 4)) & 0xF) / 15.0
            else:
                token_symbol_features[i] = np.zeros(8, dtype=np.float32)

        return token_symbol_features

    # ======== Log-specific node feature encoding methods ========
    def encode_event_orders(self, nodes):
        # In this case, we will be normalizing the event order in a per-transaction basis,
        # so the event order will be represented as a float between 0 and 1 (inclusive) in each stage
        num_events_source = len([n for n in nodes if n.blockchain_type == BlockchainType.SOURCE.value])
        num_events_destination = len([n for n in nodes if n.blockchain_type == BlockchainType.DESTINATION.value])
        
        if num_events_source == 0 and num_events_destination == 0:
            raise ValueError("No nodes provided for encoding event orders")
        return np.array([
            -1 if node.event_order is None 
            else 1 if node.blockchain_type == BlockchainType.SOURCE.value and num_events_source == 1
            else node.event_order / (num_events_source - 1) if node.blockchain_type == BlockchainType.SOURCE.value 
            else 1 if node.blockchain_type == BlockchainType.DESTINATION.value and num_events_destination == 1
            else node.event_order / (num_events_destination - 1) if node.blockchain_type == BlockchainType.DESTINATION.value 
            else 0
            for node in nodes], dtype=np.float32).reshape(-1, 1)

    def encode_event_types(self, node_attributes):
        # Use Binary encoding for event types, with 4 bits to allow for up to 16 different event types (we currently have 11)
        return np.array([EVENT_TYPE_MAP.get(attr.get("event_type"), EVENT_TYPE_MAP[EventType.UNKNOWN.value]) for attr in node_attributes], dtype=int)

    def encode_args_num(self, node_attributes):
        return np.array([attr.get("num_args", 0) for attr in node_attributes], dtype=int).reshape(-1, 1)

    def encode_input_size(self, node_attributes):
        return np.array([attr.get("input_size", 0) for attr in node_attributes], dtype=int).reshape(-1, 1)

    def encode_amounts(self, nodes):
        return np.array([float(node.amount_usd) if node.amount_usd is not None and not np.isnan(float(node.amount_usd)) else 0 for node in nodes], dtype=np.float32).reshape(-1, 1)

    # ======== Validator-specific node feature encoding methods ========
    def encode_src_dst_blockchains_and_orders(self, node_attributes):
        blockchain_src = np.array([BLOCKCHAIN_MAP.get(attr.get("source_chain"), [0, 0, 0, 0]) for attr in node_attributes], dtype=int)
        blockchain_dst = np.array([BLOCKCHAIN_MAP.get(attr.get("target_chain"), [0, 0, 0, 0]) for attr in node_attributes], dtype=int)
        
        # For the order, we will be solely be considering pair-wise bridges, and depending on the timestamp order,
        # we can have 2 possible orders: source -> destination and destination -> source.
        # These are categorical features
        blockchains_order = np.array([
            [1 if attr.get("source_timestamp") < attr.get("destination_timestamp") else 0,
            1 if attr.get("source_timestamp") > attr.get("destination_timestamp") else 0]
            for attr in node_attributes], dtype=int)        
        
        return [blockchain_src, blockchain_dst, blockchains_order]
    

    # ======== Graph label encoding ========
    def encode_graph_label(self, label):
        res = LABEL_MAP.get(label)
        if res is None:
            raise ValueError(f"Unknown graph label: {label}")
        return res


    # ======== Miscellaneous (if needed in the future) ========
    def compute_cobebert_embeddings(self, nodes, tokenizer, model):
        embeddings = np.zeros((len(nodes), 768), dtype=np.float32)
        for i, node in enumerate(nodes):
            attributes_text = node.attributes_text if node.attributes_text else None
            embeddings[i] = get_codebert_embedding(attributes_text, tokenizer, model)
        return embeddings


    # ======= Node feature computation methods by node type ========
    def compute_user_node_features(self, user_nodes: list):
        # features: [in_deg, out_deg, blockchain (binary, size 4), blockchain type (one-hot, size 3)]
        in_deg, out_deg = self.compute_degrees(user_nodes)
        feature_elements = [in_deg, out_deg]

        feature_elements.append(self.encode_blockchain_stages(user_nodes))
        feature_elements.append(self.encode_node_blockchains(user_nodes))

        return np.concatenate(feature_elements, axis=1)

    def compute_router_node_features(self, router_nodes: list):
        in_deg, out_deg = self.compute_degrees(router_nodes)
        feature_elements = [in_deg, out_deg]

        feature_elements.append(self.encode_blockchain_stages(router_nodes))
        feature_elements.append(self.encode_node_blockchains(router_nodes))

        return np.concatenate(feature_elements, axis=1)

    def compute_token_node_features(self, token_nodes: list):
        in_deg, out_deg = self.compute_degrees(token_nodes)
        feature_elements = [in_deg, out_deg]

        feature_elements.append(self.encode_blockchain_stages(token_nodes))
        feature_elements.append(self.encode_node_blockchains(token_nodes))

        feature_elements.append(self.compute_token_symbols(token_nodes))
        return np.concatenate(feature_elements, axis=1)

    def compute_other_account_node_features(self, other_account_nodes: list):
        in_deg, out_deg = self.compute_degrees(other_account_nodes)
        feature_elements = [in_deg, out_deg]

        feature_elements.append(self.encode_blockchain_stages(other_account_nodes))
        feature_elements.append(self.encode_node_blockchains(other_account_nodes))

        return np.concatenate(feature_elements, axis=1)

    def compute_log_event_node_features(self, log_event_nodes: list):
        attr_json = [json.loads(node.attributes) if node.attributes else {} for node in log_event_nodes]

        in_deg, out_deg = self.compute_degrees(log_event_nodes)
        feature_elements = [in_deg, out_deg] # 2 features

        feature_elements.append(self.encode_blockchain_stages(log_event_nodes)) # 3 features (one-hot encoding for source/offchain/destination)
        feature_elements.append(self.encode_node_blockchains(log_event_nodes)) # 4 features (binary encoding for blockchains)

        feature_elements.append(self.encode_event_orders(log_event_nodes)) # 1 feature (normalized event order in transaction)

        feature_elements.append(self.encode_event_types(attr_json)) # 4 features (binary encoding for event types, with 4 bits allowing for up to 16 event types - we currently have 11)
        feature_elements.append(self.encode_args_num(attr_json)) # 1 feature (number of arguments in the event, extracted from attributes JSON)
        feature_elements.append(self.encode_input_size(attr_json)) # 1 feature (input size in bytes for the event, extracted from attributes JSON)

        feature_elements.append(self.encode_amounts(log_event_nodes)) # 1 feature (amount for the event, extracted from attributes JSON)
        feature_elements.append(self.compute_token_symbols(log_event_nodes)) # 8 features (token symbol, encoded using MurmurHash3 as described above)
        return np.concatenate(feature_elements, axis=1)

    def compute_validator_node_features(self, validator_nodes):
        attr_json = [json.loads(node.attributes) if node.attributes else {} for node in validator_nodes]

        in_deg, out_deg = self.compute_degrees(validator_nodes)
        feature_elements = [in_deg, out_deg]

        feature_elements.extend(self.encode_src_dst_blockchains_and_orders(attr_json))
        return np.concatenate(feature_elements, axis=1)

    def compute_node_features_type(self, ntype_nodes: list, ntype: str):
        if ntype == GraphNodeType.USER.value:
            return self.compute_user_node_features(ntype_nodes)
        elif ntype == GraphNodeType.ROUTER.value:
            return self.compute_router_node_features(ntype_nodes)
        elif ntype == GraphNodeType.TOKEN.value:
            return self.compute_token_node_features(ntype_nodes)
        elif ntype == GraphNodeType.OTHER_ACCOUNT.value:
            return self.compute_other_account_node_features(ntype_nodes)
        elif ntype == GraphNodeType.LOG_EVENT.value:
            return self.compute_log_event_node_features(ntype_nodes)
        elif ntype == GraphNodeType.VALIDATOR.value:
            return self.compute_validator_node_features(ntype_nodes)
        else:
            raise ValueError(f"Unknown node type: {ntype}")