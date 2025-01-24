# This is the utility file for the basic_example game.
import networkx as nx
from dataclasses import dataclass


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
        if self.graph is None:
            self.attach_networkx_graph(nodes_data, edges_data)
            return
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
        self.agent_positions = {name: info for name, info in agent_info.items()}

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
    
    def get_team_positions(self, team):
        if not hasattr(self, 'agent_positions') or not self.agent_positions:
            return []
        return [pos for name, pos in self.agent_positions.items() if team in name]
  
@dataclass
class AgentParams:
    speed: float
    capture_radius: float
    map: Graph  
          
def extract_map_sensor_data(state):
    sensor_data = state.get('sensor', {})
    map_sensor = sensor_data.get('map')
    if map_sensor is None:
        raise ValueError("No map sensor data found in state.")
    
    sensor_type, map_data = map_sensor
    if not (isinstance(map_data, tuple) and len(map_data) == 2):
        raise ValueError("Map sensor data is not in the expected format (nodes_data, edges_data).")
    
    nodes_data, edges_data = map_data
    return nodes_data, edges_data

def extract_neighbor_sensor_data(state):
    sensor_data = state.get('sensor', {})
    neighbor_sensor = sensor_data.get('neighbor')
    if neighbor_sensor is None:
        raise ValueError("No neighbor sensor data found in state.")
    
    # Unpack the neighbor sensor tuple.
    sensor_type, neighbor_data = neighbor_sensor
    return neighbor_data

def extract_agent_sensor_data(state):
    sensor_data = state.get('sensor', {})
    agent_sensor = sensor_data.get('agent')
    if agent_sensor is None:
        raise ValueError("No agent sensor data found in state.")
    
    # Unpack the sensor tuple: sensor_type, agent_info.
    sensor_type, agent_info = agent_sensor
    return agent_info

def extract_sensor_data(state, flag_pos, flag_weight, agent_params):
    nodes_data, edges_data =  extract_map_sensor_data(state)
    agent_info = extract_agent_sensor_data(state)
    agent_params.map.update_networkx_graph(nodes_data, edges_data)
    agent_params.map.set_agent_positions(agent_info)
    agent_params.map.set_flag_positions(flag_pos)
    agent_params.map.set_flag_weights(flag_weight)
    attacker_positions = agent_params.map.get_team_positions("attacker")
    defender_positions = agent_params.map.get_team_positions("defender")
    return attacker_positions, defender_positions