import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

from repository.db.graph_label import GraphNodeType
from repository.db.graph_label import GraphEdgeType

# Define colors for nodes and edges
node_type_colors = {
    GraphNodeType.USER.value: "#4599C3",
    GraphNodeType.ROUTER.value: "#ED8546",
    GraphNodeType.TOKEN.value: "#F2C14E",
    GraphNodeType.OTHER_ACCOUNT.value: "#9B5DE5",
    GraphNodeType.LOG_EVENT.value: "#F15BB5",
    GraphNodeType.VALIDATOR.value: "#00BBF9",
}

def visualize_graph(graph_data):
    graph = to_networkx(graph_data, to_undirected=False)
    node_colors = []
    labels = {}
    for node, attrs in graph.nodes(data=True):
        node_type = attrs.get("type")
        color = node_type_colors.get(node_type, "#CCCCCC")  # Default to gray if type is unknown
        node_colors.append(color)
        if attrs.get("type") == GraphNodeType.USER.value:
            labels[node] = f"U{node}"
        elif attrs.get("type") == GraphNodeType.ROUTER.value:
            labels[node] = f"R{node}"
        elif attrs.get("type") == GraphNodeType.TOKEN.value:
            labels[node] = f"T{node}"
        elif attrs.get("type") == GraphNodeType.OTHER_ACCOUNT.value:
            labels[node] = f"O{node}"
        elif attrs.get("type") == GraphNodeType.LOG_EVENT.value:
            labels[node] = f"L{node}"
        elif attrs.get("type") == GraphNodeType.VALIDATOR.value:
            labels[node] = f"V{node}"
        
    # Define colors for the edges
    edge_type_colors = {
        GraphEdgeType.TRANSACTION.value: "#000000",
        GraphEdgeType.TOKEN_TRANSFER.value: "#FF0000",
        GraphEdgeType.TOKEN_AUTH.value: "#00FF00",
        GraphEdgeType.FUNCTION_CALL.value: "#0000FF",
        GraphEdgeType.LOG_RELATION.value: "#FFA500",
        GraphEdgeType.CROSS_CHAIN_RELATION.value: "#800080",
    }

    edge_colors = []
    edge_labels = {}
    for u, v, attrs in graph.edges(data=True):
        edge_type = attrs.get("type")
        color = edge_type_colors.get(edge_type[1], "#CCCCCC")  # Default to gray if type is unknown
        edge_colors.append(color)
        if edge_type[1] == GraphEdgeType.TRANSACTION.value:
            edge_labels[(u, v)] = "TX"
        elif edge_type[1] == GraphEdgeType.TOKEN_TRANSFER.value:
            edge_labels[(u, v)] = "TT"
        elif edge_type[1] == GraphEdgeType.TOKEN_AUTH.value:
            edge_labels[(u, v)] = "TA"
        elif edge_type[1] == GraphEdgeType.FUNCTION_CALL.value:
            edge_labels[(u, v)] = "FC"
        elif edge_type[1] == GraphEdgeType.LOG_RELATION.value:
            edge_labels[(u, v)] = "LR"
        elif edge_type[1] == GraphEdgeType.CROSS_CHAIN_RELATION.value:
            edge_labels[(u, v)] = "CCR"

    pos = nx.spring_layout(graph, k=2, iterations=50, seed=1)
    nx.draw_networkx(
        graph,
        pos=pos,
        labels=labels,
        with_labels=True,
        node_color=node_colors,
        edge_color=edge_colors,
        node_size=500,
    )
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    plt.show()
    input("Press Enter to continue...")