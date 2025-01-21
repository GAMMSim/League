# This is the utility file for the basic_example game.
import networkx as nx


class Graph:
    def __init__(self, graph=None):
        self.graph = graph
        self.agent_positions = {}
        self.flag_positions = []
        self.flag_weights = []

    def attach_networkx_graph(self, nodes_data, edges_data):
        self.graph = nx.DiGraph()

        for node_id, attrs in nodes_data.items():
            if not isinstance(attrs, dict):
                try:
                    attrs = attrs.__dict__
                except Exception:
                    attrs = {}
            self.graph.add_node(node_id, **attrs)

        for edge_id, edge_info in edges_data.items():
            if isinstance(edge_info, dict):
                source = edge_info.get("source")
                target = edge_info.get("target")
                edge_attrs = edge_info
            else:
                source = getattr(edge_info, "source", None)
                target = getattr(edge_info, "target", None)
                try:
                    edge_attrs = edge_info.__dict__
                except Exception:
                    edge_attrs = {}

            if source is None or target is None:
                continue  # Skip edges with invalid endpoints

            self.graph.add_edge(source, target, **edge_attrs)

    def update_networkx_graph(self, nodes_data, edges_data):
        for node_id, attrs in nodes_data.items():
            if not isinstance(attrs, dict):
                try:
                    attrs = attrs.__dict__
                except Exception:
                    attrs = {}
            if self.graph.has_node(node_id):
                self.graph.nodes[node_id].update(attrs)
            else:
                self.graph.add_node(node_id, **attrs)

        for edge_id, edge_info in edges_data.items():
            if isinstance(edge_info, dict):
                source = edge_info.get("source")
                target = edge_info.get("target")
                edge_attrs = edge_info
            else:
                source = getattr(edge_info, "source", None)
                target = getattr(edge_info, "target", None)
                try:
                    edge_attrs = edge_info.__dict__
                except Exception:
                    edge_attrs = {}
            if source is None or target is None:
                continue  # Skip edges with invalid endpoints

            if self.graph.has_edge(source, target):
                self.graph[source][target].update(edge_attrs)
            else:
                self.graph.add_edge(source, target, **edge_attrs)

    def set_agent_positions(self, agent_info):
        self.agent_positions = {name: info.get("position") for name, info in agent_info.items()}

    def set_flag_positions(self, FLAG_POSITIONS):
        self.flag_positions = FLAG_POSITIONS

    def set_flag_weights(self, FLAG_WEIGHTS):
        self.flag_weights = FLAG_WEIGHTS
        
    def shortest_path_to(self, source_node, target_node, speed=1):
        if not isinstance(target_node, (list, tuple, set)):
            target_nodes = [target_node]
        else:
            target_nodes = list(target_node)

        best_path = None
        best_length = float('inf')

        # Compute the shortest path for each candidate target
        for t in target_nodes:
            try:
                length, path = nx.single_source_dijkstra(self.graph, source=source_node, target=t, weight='length')
                # Use the length of the path as the comparison metric
                if length < best_length:
                    best_length = length
                    best_path = path
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

        if best_path is None:
            return None
        
        index = speed if speed < len(best_path) else len(best_path) - 1
        return best_path[index]
    
    def get_neighbors(self, source_node, speed):
        # Get all nodes within the given distance (speed) from the source_node
        neighbors = nx.single_source_shortest_path_length(self.graph, source_node, cutoff=speed)
        # Remove the source node itself from the neighbors
        neighbors.pop(source_node, None)
        return list(neighbors.keys())
        
        
