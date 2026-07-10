from sqlalchemy import func, or_

from repository.base import BaseRepository
from .graph_label import BlockchainGraphLabel, BlockchainType, GraphEdgeType, GraphNodeType

from .models import (
    GraphEdge,
    GraphMappingBlockchain,
    GraphMappingCrossChain,
    GraphNode,
)


class GraphMappingBlockchainRepository(BaseRepository):
    def __init__(self, session_factory):
        super().__init__(GraphMappingBlockchain, session_factory)

    def get_all_non_cctx(self):
        with self.get_session() as session:
            return session.query(GraphMappingBlockchain).filter(
                GraphMappingBlockchain.cctx_graph_id.is_(None),
                or_(
                    GraphMappingBlockchain.discard_flag != 1,
                    GraphMappingBlockchain.discard_flag.is_(None),
                )
            ).all()
        
    def get_all_cctx(self):
        with self.get_session() as session:
            return session.query(GraphMappingBlockchain).filter(
                GraphMappingBlockchain.cctx_graph_id.is_not(None),
                or_(
                    GraphMappingBlockchain.discard_flag != 1,
                    GraphMappingBlockchain.discard_flag.is_(None),
                )
            ).all()

    def get_by_id(self, graph_id: int):
        with self.get_session() as session:
            return session.query(GraphMappingBlockchain).filter(GraphMappingBlockchain.graph_id == graph_id).first()

    def graph_exists(self, bridge: str, blockchain: str, tx_hash: str):
        with self.get_session() as session:
            return session.query(GraphMappingBlockchain).filter(GraphMappingBlockchain.bridge == bridge, GraphMappingBlockchain.blockchain == blockchain, GraphMappingBlockchain.tx_hash == tx_hash).first()

    def assign_cctx_id(self, graph_id: int, cctx_graph_id: int):
        with self.get_session() as session:
            graph_mapping = session.query(GraphMappingBlockchain).filter(GraphMappingBlockchain.graph_id == graph_id).first()
            if graph_mapping:
                graph_mapping.cctx_graph_id = cctx_graph_id
                session.commit()
                return graph_mapping
            return None
        
    def get_by_bridge(self, bridge: str):
        with self.get_session() as session:
            return session.query(GraphMappingBlockchain).filter(GraphMappingBlockchain.bridge == bridge).all()
    
    def get_cctxs_source_anomalies(self, bridges: list[str]):
        with self.get_session() as session:
            return session.query(GraphMappingBlockchain).filter(
                or_(
                    GraphMappingBlockchain.discard_flag != 1,
                    GraphMappingBlockchain.discard_flag.is_(None),
                ),
                GraphMappingBlockchain.bridge.in_(bridges),
                GraphMappingBlockchain.cctx_graph_id.isnot(None),
                GraphMappingBlockchain.label == BlockchainGraphLabel.ANOMALY.value
            ).all()

class GraphMappingCrossChainRepository(BaseRepository):
    def __init__(self, session_factory):
        super().__init__(GraphMappingCrossChain, session_factory)

    def get_by_id(self, cctx_graph_id: int):
        with self.get_session() as session:
            return session.query(GraphMappingCrossChain).filter(GraphMappingCrossChain.cctx_graph_id == cctx_graph_id).first()

    def graph_exists(self, bridge: str, cctx_id: int):
        with self.get_session() as session:
            return session.query(GraphMappingCrossChain).filter(GraphMappingCrossChain.bridge == bridge, GraphMappingCrossChain.cctx_id == cctx_id).first()

    def get_by_chain_tx_hash(self, bridge: str, chain: str, tx_hash: str):
        with self.get_session() as session:
            source = session.query(GraphMappingCrossChain).filter(
                GraphMappingCrossChain.bridge == bridge,
                GraphMappingCrossChain.source_chain == chain, 
                GraphMappingCrossChain.source_tx_hash == tx_hash
            ).first()
            if source is not None:
                return source
            else:
                return session.query(GraphMappingCrossChain).filter(
                    GraphMappingCrossChain.bridge == bridge,
                    GraphMappingCrossChain.target_chain == chain, 
                    GraphMappingCrossChain.destination_tx_hash == tx_hash
                ).first()
            
    def get_by_bridge(self, bridge: str):
        with self.get_session() as session:
            return session.query(GraphMappingCrossChain).filter(GraphMappingCrossChain.bridge == bridge).all()

class GraphNodeRepository(BaseRepository):
    def __init__(self, session_factory):
        super().__init__(GraphNode, session_factory)

    def get_all_cctx(self):
        with self.get_session() as session:
            return session.query(GraphNode).filter(
                or_(
                    GraphNode.discard_flag != 1,
                    GraphNode.discard_flag.is_(None),
                ),
                GraphNode.cctx_graph_id.is_not(None)
            ).all()
        
    def get_all_non_cctx(self):
        with self.get_session() as session:
            return session.query(GraphNode).filter(
                GraphNode.cctx_graph_id.is_(None),
                or_(
                    GraphNode.discard_flag != 1,
                    GraphNode.discard_flag.is_(None),
                )
            ).all()

    def get_by_address(self, graph_id: int, address: str):
        with self.get_session() as session:
            return session.query(GraphNode).filter(GraphNode.chain_graph_id == graph_id, GraphNode.address == address).first()

    def get_by_chain_graph_ids(self, chain_graph_ids: list[str], no_offchain: bool = False):
        with self.get_session() as session:
            query = session.query(GraphNode).filter(GraphNode.chain_graph_id.in_(chain_graph_ids))
            if no_offchain:
                query = query.filter(
                    or_(
                        GraphNode.discard_flag != 1,
                        GraphNode.discard_flag.is_(None),
                    ),
                    GraphNode.blockchain_type != BlockchainType.OFFCHAIN.value,
                    GraphNode.node_type != GraphNodeType.VALIDATOR.value
                )
            return query.all()

    def get_by_cctx_graph_id(self, cctx_graph_id: int):
        with self.get_session() as session:
            return session.query(GraphNode).filter(GraphNode.cctx_graph_id == cctx_graph_id).all()

    def update_node_type(self, node_id: int, new_type: str):
        with self.get_session() as session:
            node = session.query(GraphNode).filter(GraphNode.node_id == node_id).first()
            if node:
                node.node_type = new_type
                session.commit()
                return node
            return None
        
    def assign_cctx_id(self, graph_id: int, cctx_id: int):
        with self.get_session() as session:
            nodes = session.query(GraphNode).filter(GraphNode.chain_graph_id == graph_id).all()
            for node in nodes:
                node.cctx_graph_id = cctx_id
            session.commit()
            return nodes
        
    def get_router_node_by_graph_id(self, graph_id: int):
        with self.get_session() as session:
            return session.query(GraphNode).filter(GraphNode.chain_graph_id == graph_id, GraphNode.node_type == GraphNodeType.ROUTER.value).first()

    def get_by_bridge(self, bridge: str):
        with self.get_session() as session:
            return session.query(GraphNode).filter(GraphNode.bridge == bridge).all()

    def get_max_in_degree_excluding_non_normal_cross_chain(self):
        with self.get_session() as session:
            max_in_degree = (
                session.query(func.max(GraphNode.in_degree))
                .outerjoin(
                    GraphMappingCrossChain,
                    GraphNode.cctx_graph_id == GraphMappingCrossChain.cctx_graph_id,
                )
                .filter(
                    or_(
                        GraphNode.cctx_graph_id.is_(None),
                        GraphMappingCrossChain.label == "normal",
                    )
                )
                .scalar()
            )
            return max_in_degree or 0

    def get_max_out_degree_excluding_non_normal_cross_chain(self):
        with self.get_session() as session:
            max_out_degree = (
                session.query(func.max(GraphNode.out_degree))
                .outerjoin(
                    GraphMappingCrossChain,
                    GraphNode.cctx_graph_id == GraphMappingCrossChain.cctx_graph_id,
                )
                .filter(
                    or_(
                        GraphNode.cctx_graph_id.is_(None),
                        GraphMappingCrossChain.label == "normal",
                    )
                )
                .scalar()
            )
            return max_out_degree or 0

class GraphEdgeRepository(BaseRepository):
    def __init__(self, session_factory):
        super().__init__(GraphEdge, session_factory)

    def get_all_cctx(self):
        with self.get_session() as session:
            return session.query(GraphEdge).filter(GraphEdge.cctx_graph_id.is_not(None)).all()
        
    def get_all_non_cctx(self):
        with self.get_session() as session:
            return session.query(GraphEdge).filter(
                GraphEdge.cctx_graph_id.is_(None),
                or_(
                    GraphEdge.discard_flag != 1,
                    GraphEdge.discard_flag.is_(None),
                )
            ).all()

    def get_by_chain_graph_ids(self, chain_graph_ids: list[str], no_offchain: bool = False):
        with self.get_session() as session:
            query = session.query(GraphEdge).filter(GraphEdge.chain_graph_id.in_(chain_graph_ids))
            if no_offchain:
                query = query.filter(
                    or_(
                        GraphEdge.discard_flag != 1,
                        GraphEdge.discard_flag.is_(None),
                    ),
                    GraphEdge.blockchain_type != BlockchainType.OFFCHAIN.value,
                    GraphEdge.edge_type != GraphEdgeType.CROSS_CHAIN_RELATION.value
                )
            return query.all()

    def get_by_connections(self, graph_id: int, source_id: int, target_id: int):
        with self.get_session() as session:
            return session.query(GraphEdge).filter(GraphEdge.chain_graph_id == graph_id, GraphEdge.source_id == source_id, GraphEdge.target_id == target_id).first()

    def get_by_cctx_graph_id(self, cctx_graph_id: int):
        with self.get_session() as session:
            return session.query(GraphEdge).filter(GraphEdge.cctx_graph_id == cctx_graph_id).all()

    def assign_cctx_id(self, graph_id: int, cctx_id: int):
        with self.get_session() as session:
            edges = session.query(GraphEdge).filter(GraphEdge.chain_graph_id == graph_id).all()
            for edge in edges:
                edge.cctx_graph_id = cctx_id
            session.commit()
            return edges

    def get_by_bridge(self, bridge: str):
        with self.get_session() as session:
            return session.query(GraphEdge).filter(GraphEdge.bridge == bridge).all()