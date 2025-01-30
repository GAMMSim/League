# This is the utility file for the basic_example game.
import networkx as nx
from dataclasses import dataclass
import gamms
import os
import pickle
from lib.interface import *
import yaml


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
        best_length = float("inf")

        # Compute the shortest path for each candidate target
        for t in target_nodes:
            try:
                length, path = nx.single_source_dijkstra(self.graph, source=source_node, target=t, weight="length")
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
        if not hasattr(self, "agent_positions") or not self.agent_positions:
            return []
        return [pos for name, pos in self.agent_positions.items() if team in name]


class AgentParams:
    def __init__(self, speed: float, capture_radius: float, map: Graph, start_node_id: int, **kwargs):
        self.speed = speed
        self.capture_radius = capture_radius
        self.map = map
        self.start_node_id = start_node_id
                
        # Handle any additional custom kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

# ------------------------------------------------------------------------------
# RAW HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def extract_map_sensor_data(state):
    sensor_data = state.get("sensor", {})
    map_sensor = sensor_data.get("map")
    if map_sensor is None:
        raise ValueError("No map sensor data found in state.")

    sensor_type, map_data = map_sensor
    if not (isinstance(map_data, tuple) and len(map_data) == 2):
        raise ValueError("Map sensor data is not in the expected format (nodes_data, edges_data).")

    nodes_data, edges_data = map_data
    return nodes_data, edges_data


def extract_neighbor_sensor_data(state):
    sensor_data = state.get("sensor", {})
    neighbor_sensor = sensor_data.get("neighbor")
    if neighbor_sensor is None:
        raise ValueError("No neighbor sensor data found in state.")

    # Unpack the neighbor sensor tuple.
    sensor_type, neighbor_data = neighbor_sensor
    return neighbor_data


def extract_agent_sensor_data(state):
    sensor_data = state.get("sensor", {})
    agent_sensor = sensor_data.get("agent")
    if agent_sensor is None:
        raise ValueError("No agent sensor data found in state.")

    # Unpack the sensor tuple: sensor_type, agent_info.
    sensor_type, agent_info = agent_sensor
    return agent_info


def extract_sensor_data(state, flag_pos, flag_weight, agent_params):
    nodes_data, edges_data = extract_map_sensor_data(state)
    agent_info = extract_agent_sensor_data(state)
    agent_params.map.update_networkx_graph(nodes_data, edges_data)
    agent_params.map.set_agent_positions(agent_info)
    agent_params.map.set_flag_positions(flag_pos)
    agent_params.map.set_flag_weights(flag_weight)
    attacker_positions = agent_params.map.get_team_positions("attacker")
    defender_positions = agent_params.map.get_team_positions("defender")
    return attacker_positions, defender_positions

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------


def load_game_rule(config, game_rule: str) -> None:
    """
    Load a game rule from a YAML file and update the configuration parameters accordingly.

    Args:
        config: The current configuration object.
        game_rule (str): The name of the game rule to load.
    """
    print(colored(f"Loading game rule: {game_rule}", "blue"))

    # Construct path to game rule file
    rule_path = os.path.join("game_rules", f"{game_rule}.yaml")

    config_dir = os.path.dirname(os.path.abspath(config.__file__))
    rule_path = os.path.join(os.path.dirname(config_dir), "game_rules", f"{game_rule}.yaml")
    print(colored(f"Looking for game rule at: {rule_path}", "blue"))  # Debug print

    try:
        # Load YAML file
        with open(rule_path, "r") as file:
            rule_data = yaml.safe_load(file)

        if "gamerule" not in rule_data:
            print(colored("Invalid game rule file format", "red"))
            return

        rule = rule_data["gamerule"]

        # Override config parameters based on game rule
        if "max_time" in rule and rule["max_time"] is not None:
            config.MAX_TIME = rule["max_time"]
            print(colored(f"Max time overrided to {config.MAX_TIME}", "yellow"))

        if "interaction" in rule and rule["interaction"] is not None:
            config.INTERACTION = {"tagging": rule["interaction"]["tagging"], "capture": rule["interaction"]["capture"], "prioritize": rule["interaction"]["prioritize"]}
            print(colored(f"Interaction model overrided to {config.INTERACTION}", "yellow"))

        if "payoff" in rule and rule["payoff"] is not None:
            config.PAYOFF = {"model": rule["payoff"]["model"], "constants": rule["payoff"]["constants"]}
            print(colored(f"Payoff model overrided to {config.PAYOFF}", "yellow"))

        if "agent" in rule:
            # Update attacker parameters
            if "attacker" in rule["agent"]:
                attacker = rule["agent"]["attacker"]
                config.ATTACKER_GLOBAL_SPEED = attacker.get("speed", config.ATTACKER_GLOBAL_SPEED)
                config.ATTACKER_GLOBAL_CAPTURE_RADIUS = attacker.get("capture_radius", config.ATTACKER_GLOBAL_CAPTURE_RADIUS)
                config.ATTACKER_GLOBAL_SENSORS = attacker.get("sensors", config.ATTACKER_GLOBAL_SENSORS)
                print(colored(f"Attacker global parameters overrided to: {config.ATTACKER_GLOBAL_SPEED}, {config.ATTACKER_GLOBAL_CAPTURE_RADIUS}, {config.ATTACKER_GLOBAL_SENSORS}", "yellow"))

                # Update attacker configs
                for key in config.ATTACKER_CONFIG:
                    update_dict = {}

                    # Only update parameters that exist in original config
                    if "speed" in config.ATTACKER_CONFIG[key]:
                        update_dict["speed"] = attacker.get("speed", config.ATTACKER_GLOBAL_SPEED)

                    if "capture_radius" in config.ATTACKER_CONFIG[key]:
                        update_dict["capture_radius"] = attacker.get("capture_radius", config.ATTACKER_GLOBAL_CAPTURE_RADIUS)

                    if "sensors" in config.ATTACKER_CONFIG[key]:
                        update_dict["sensors"] = attacker.get("sensors", config.ATTACKER_GLOBAL_SENSORS)

                    if update_dict:  # Only update if there are parameters to update
                        config.ATTACKER_CONFIG[key].update(update_dict)
                print(colored(f"Attacker parameters overrided by global.", "yellow"))

            # Update defender parameters
            if "defender" in rule["agent"]:
                defender = rule["agent"]["defender"]
                config.DEFENDER_GLOBAL_SPEED = defender.get("speed", config.DEFENDER_GLOBAL_SPEED)
                config.DEFENDER_GLOBAL_CAPTURE_RADIUS = defender.get("capture_radius", config.DEFENDER_GLOBAL_CAPTURE_RADIUS)
                config.DEFENDER_GLOBAL_SENSORS = defender.get("sensors", config.DEFENDER_GLOBAL_SENSORS)
                print(colored(f"Defender global parameters overrided to: {config.DEFENDER_GLOBAL_SPEED}, {config.DEFENDER_GLOBAL_CAPTURE_RADIUS}, {config.DEFENDER_GLOBAL_SENSORS}", "yellow"))

                # Update defender configs
                for key in config.DEFENDER_CONFIG:
                    update_dict = {}

                    # Only update parameters that exist in original config
                    if "speed" in config.DEFENDER_CONFIG[key]:
                        update_dict["speed"] = defender.get("speed", config.DEFENDER_GLOBAL_SPEED)

                    if "capture_radius" in config.DEFENDER_CONFIG[key]:
                        update_dict["capture_radius"] = defender.get("capture_radius", config.DEFENDER_GLOBAL_CAPTURE_RADIUS)

                    if "sensors" in config.DEFENDER_CONFIG[key]:
                        update_dict["sensors"] = defender.get("sensors", config.DEFENDER_GLOBAL_SENSORS)

                    if update_dict:  # Only update if there are parameters to update
                        config.DEFENDER_CONFIG[key].update(update_dict)
                print(colored(f"Defender parameters overrided by global.", "yellow"))

        print(colored("Game rule loaded successfully\n", "green"))
        return config

    except FileNotFoundError:
        print(colored(f"Game rule file not found: {rule_path}", "red"))
        return
    except yaml.YAMLError as e:
        print(colored(f"Invalid YAML file: {e}", "red"))
        raise
    except Exception as e:
        print(colored(f"Cannot load game rule: {e}", "red"))
        raise


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
        print(colored("Graph loaded from file.", "green"))
    else:
        G = gamms.osm.create_osm_graph(location, resolution=resolution)
        with open(graph_path, "wb") as f:
            pickle.dump(G, f)
        print(colored("New graph generated and saved to file.", "green"))

    ctx.graph.attach_networkx_graph(G)
    return ctx, G


def initialize_agents(ctx, attacker_config, defender_config, global_params):
    """
    Configure and create agents in the game context.
    Individual agent parameters override global defaults when specified.

    Args:
        ctx: Game context
        attacker_config (dict): Individual attacker configurations
        defender_config (dict): Individual defender configurations
        global_params (dict): Global default parameters
    """

    def get_agent_param(config, param_name, team):
        """Get parameter with priority: individual config > global params"""
        individual_value = config.get(param_name)
        if individual_value is not None:
            return individual_value
        return global_params[f"{team}_{param_name}"]

    def create_agent_entries(configs, team):
        print(colored(f"Creating {team} agents...", "blue"))
        entries = {}
        params = {}

        for name, config in configs.items():
            start_node_id = config.get("start_node_id")
            
            # Create agent entry
            entries[name] = {
                "team": team,
                "sensors": get_agent_param(config, "sensors", team),
                "color": get_agent_param(config, "color", team),
                "current_node_id": start_node_id,  # Set initial position
                "start_node_id": start_node_id
            }

            # Create parameter object
            params[name] = AgentParams(
                speed=get_agent_param(config, "speed", team),
                capture_radius=get_agent_param(config, "capture_radius", team),
                map=Graph(),
                start_node_id=start_node_id
            )

        return entries, params

    # Create entries for both teams
    attacker_entries, attacker_params = create_agent_entries(attacker_config, "attacker")
    defender_entries, defender_params = create_agent_entries(defender_config, "defender")

    # Combine configurations
    agent_config = {**attacker_entries, **defender_entries}
    agent_params_map = {**attacker_params, **defender_params}

    # Create agents in context
    for name, config in agent_config.items():
        ctx.agent.create_agent(name, **config)
    print(colored("Agents created.", "green"))

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

    print(colored("Strategies set.", "green"))


def configure_visualization(ctx, agent_config, global_params):
    """
    Configure visualization settings for the graph and agents.

    Args:
        ctx: The initialized game context.
        agent_config (dict): Dictionary of agent configurations.
        global_params (dict): Dictionary containing global visualization parameters.
    """
    print(colored("Configuring visualization...", "blue"))
    ctx.visual.set_graph_visual(width=global_params["width"], height=global_params["height"], draw_id=global_params["draw_node_id"], node_color=global_params["node_color"], edge_color=global_params["edge_color"])
    ctx.visual._sim_time_constant = global_params["game_speed"]

    for name, config in agent_config.items():
        color = config.get("color", global_params["default_color"])
        size = config.get("size", global_params["default_size"])
        ctx.visual.set_agent_visual(name, color=color, size=size)
    print(colored("Visualization configured.", "green"))


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

    print(colored("Flags initialized.", "green"))
