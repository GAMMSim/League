# This is the utility file for the basic_example game.
import networkx as nx


class Graph:
    def __init__(self, graph=None):
        self.graph = graph
        self.agent_positions = {}
        self.flag_positions = {}
        self.flag_weights = {}

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
                continue

        self.graph.add_edge(source, target, **edge_attrs)
        
    def set_agent_positions(self, agent_info):
        self.agent_positions = {name: info.get("position") for name, info in agent_info.items()}
        
    def set_flag_positions(self, flag_positions):
