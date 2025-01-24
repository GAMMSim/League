# This is the utility file for the basic_example game.
import networkx as nx
from dataclasses import dataclass
import gamms
import os
import pickle


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

def initialize_game_context(vis_engine, graph_path, location, resolution):
    """
    Initialize the game context and load or create the graph used in the simulation.

    Args:
        vis_engine: Visualization engine to be used for the game context.
        graph_path (str): Path to the graph file.
        location (str): Location to generate a new graph if the file does not exist.
        resolution (float): Resolution for generating a new graph.

    Returns:
        ctx: The initialized game context.
        G: The graph object used in the simulation.
    """
    ctx = gamms.create_context(vis_engine=vis_engine)

    if os.path.exists(graph_path):
        with open(graph_path, "rb") as f:
            G = pickle.load(f)
        print("Graph loaded from file.")
    else:
        G = gamms.osm.create_osm_graph(location, resolution=resolution)
        with open(graph_path, "wb") as f:
            pickle.dump(G, f)
        print("Graph created and saved to file.")

    ctx.graph.attach_networkx_graph(G)
    return ctx, G

def configure_agents(ctx, attacker_config, defender_config, global_params):
    """
    Configure attacker and defender agents with their respective parameters and create them in the game context.

    Args:
        ctx: The initialized game context.
        attacker_config (dict): Configuration for attackers.
        defender_config (dict): Configuration for defenders.
        global_params (dict): Dictionary containing global parameters such as speed, sensors, and colors.

    Returns:
        agent_config (dict): Dictionary of agent configurations.
        agent_params_map (dict): Dictionary of agent parameters mapped by agent name.
    """
    agent_config = {}
    agent_params_map = {}

    for name, config in attacker_config.items():
        agent_entry = config.copy()
        agent_entry["team"] = "attacker"
        agent_entry.setdefault("sensors", global_params["attacker_sensors"])
        agent_entry.setdefault("color", global_params["attacker_color"])
        agent_entry["start_node_id"] = config.get("start_node_id", None)

        param_obj = AgentParams(
            speed=config.get("speed", global_params["attacker_speed"]),
            capture_radius=config.get("capture_radius", global_params["attacker_capture_radius"]),
            map=Graph(),
        )
        agent_params_map[name] = param_obj
        agent_config[name] = agent_entry

    for name, config in defender_config.items():
        agent_entry = config.copy()
        agent_entry["team"] = "defender"
        agent_entry.setdefault("sensors", global_params["defender_sensors"])
        agent_entry.setdefault("color", global_params["defender_color"])
        agent_entry["start_node_id"] = config.get("start_node_id", None)

        param_obj = AgentParams(
            speed=config.get("speed", global_params["defender_speed"]),
            capture_radius=config.get("capture_radius", global_params["defender_capture_radius"]),
            map=Graph(),
        )
        agent_params_map[name] = param_obj
        agent_config[name] = agent_entry

    for name, config in agent_config.items():
        ctx.agent.create_agent(name, **config)

    print("Agents created.")
    return agent_config, agent_params_map

def assign_strategies(ctx, agent_config, attacker_strategy_module, defender_strategy_module):
    """
    Assign strategies to agents based on their team.

    Args:
        ctx: The initialized game context.
        agent_config (dict): Dictionary of agent configurations.
        attacker_strategy_module: Module for attacker strategies.
        defender_strategy_module: Module for defender strategies.
    """
    strategies = {}
    strategies.update(attacker_strategy_module.map_strategy({name: config for name, config in agent_config.items() if config.get("team") == "attacker"}))
    strategies.update(defender_strategy_module.map_strategy({name: config for name, config in agent_config.items() if config.get("team") == "defender"}))

    for agent in ctx.agent.create_iter():
        agent.register_strategy(strategies.get(agent.name, None))

    print("Strategies set.")

def configure_visualization(ctx, agent_config, global_params):
    """
    Configure visualization settings for the graph and agents.

    Args:
        ctx: The initialized game context.
        agent_config (dict): Dictionary of agent configurations.
        global_params (dict): Dictionary containing global visualization parameters.
    """
    ctx.visual.set_graph_visual(
        width=global_params["width"],
        height=global_params["height"],
        draw_id=global_params["draw_node_id"],
        node_color=global_params["node_color"],
        edge_color=global_params["edge_color"]
    )
    ctx.visual._sim_time_constant = global_params["game_speed"]

    for name, config in agent_config.items():
        color = config.get("color", global_params["default_color"])
        size = config.get("size", global_params["default_size"])
        ctx.visual.set_agent_visual(name, color=color, size=size)

def initialize_flags(ctx, flag_positions, flag_size, flag_color):
    """
    Initialize flags in the simulation based on their positions and visualization parameters.

    Args:
        ctx: The initialized game context.
        flag_positions (list): List of node IDs where flags should be placed.
        flag_size (int): Size of the flags.
        flag_color: Color of the flags.
    """
    for index, flag_node_id in enumerate(flag_positions):
        node = ctx.graph.graph.get_node(flag_node_id)
        flag_data = {"x": node.x, "y": node.y, "scale": flag_size, "color": flag_color}
        ctx.visual.add_artist(f"flag_{index}", flag_data)

    print("Flags initialized.")