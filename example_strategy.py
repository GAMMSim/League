from lib.utils.sensor_utils import extract_agent_info, extract_team_and_enemy_info, extract_map_and_territory_info
from lib.utils.graph_utils import compute_x_neighbors
from lib.core.console import *
import networkx as nx
from typing import Dict, List


def strategy(state):
    """
    Enhanced strategy that extracts all game information.
    Currently agent stays at current position.
    """
    # Extract agent information
    name, capture_radius, tagging_radius, speed, current_position, time, team = extract_agent_info(state)
    
    # Extract team and enemy information (this also updates the map with sensor data)
    teammate_positions, team_flags, enemy_positions, enemy_flags = extract_team_and_enemy_info(state)
    
    # Extract map and territory information (includes neighbor sensor processing)
    graph, team_territory, enemy_territory, neighbor_nodes = extract_map_and_territory_info(state)
    
    # LIST OF ALL INFORMATION AVAILABLE:
    # name: str - Agent's name
    # capture_radius: int - Distance needed to capture flags
    # tagging_radius: int - Distance needed to tag enemies
    # speed: int - Movement speed per turn
    # current_position: int - Current node ID
    # time: int - Current game timestep
    # team: str - Team membership ("alpha" or "beta")
    # teammate_positions: Dict[str, int] - Teammate names to positions
    # team_flags: List[int] - Flag positions this team should defend
    # team_territory: List[int] - Nodes controlled by this team (from partition)
    # enemy_positions: Dict[str, int] - Enemy names to positions
    # enemy_flags: List[int] - Enemy flag positions to attack
    # enemy_territory: List[int] - Nodes controlled by enemy team (from partition)
    # graph: nx.MultiDiGraph - NetworkX graph of the game map
    # neighbor_nodes: List[int] - Neighboring node IDs from neighbor sensor
    
    # Debug info for alpha_0 agent only
    if name == "alpha_0":
        success(f"=== Info for {name} ===")
        success(f"Agent: {name}")
        success(f"Capture radius: {capture_radius}")
        success(f"Tagging radius: {tagging_radius}")
        success(f"Speed: {speed}")
        success(f"Current position: {current_position}")
        success(f"Time: {time}")
        success(f"Team: {team}")
        success(f"Teammate positions: {teammate_positions}")
        success(f"Team flags: {team_flags}")
        success(f"Team territory: {team_territory}")
        success(f"Enemy positions: {enemy_positions}")
        success(f"Enemy flags: {enemy_flags}")
        success(f"Enemy territory: {enemy_territory}")
        success(f"Graph nodes: {len(graph.nodes()) if graph else 0}")
        success(f"Graph edges: {len(graph.edges()) if graph else 0}")
        success(f"Neighbor nodes: {neighbor_nodes}")
        success("=== Debug Info ===")
    
    # Strategy logic goes here
    # Pick a random neighbor node
    if neighbor_nodes:
        import random
        state["action"] = random.choice(neighbor_nodes)
    else:
        state["action"] = current_position


def map_strategy(agent_config):
    """
    Map strategy to all agents in the config.
    """
    return {name: strategy for name in agent_config}