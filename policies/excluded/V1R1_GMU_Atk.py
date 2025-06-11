import random
import networkx as nx
from lib.graph_utils import compute_node_dominance_region, compute_convex_hull_and_perimeter, compute_x_neighbors, compute_shortest_path_step
from lib.game_utils import extract_sensor_data, extract_neighbor_sensor_data

def strategy(state):
    """
    Defines the attacker's strategy by computing the advantage index versus all defenders 
    for each candidate target node, summing these indices, and choosing the node with the best (lowest) sum.
    
    Parameters:
        state (dict): The current state of the game, including positions and parameters.
    """
    # --- Retrieve state parameters ---
    current_node = state["curr_pos"]
    flag_positions = state["flag_pos"]
    flag_weights = state["flag_weight"]
    agent_params = state["agent_params"]
    agent_params_dict = state["agent_params_dict"]
    _, _ = extract_sensor_data(state, flag_positions, flag_weights, agent_params)

    # --- Extract attacker and defender positions ---
    attacker_dict = agent_params.map.attacker_dict
    attacker_list = list(attacker_dict.items())  # [(attacker_name, attacker_position), ...]
    # Identify the current attacker by matching its position.
    current_attacker_idx = None
    for idx, (attacker_name, attacker_position) in enumerate(attacker_list):
        if attacker_position == current_node:
            current_attacker_idx = idx
            break
    if current_attacker_idx is None:
        print("Current attacker not found in attacker list. Staying in place.")
        state["action"] = current_node
        return

    current_attacker_name, attacker_position = attacker_list[current_attacker_idx]
    attacker_speed = agent_params_dict.get(current_attacker_name, {}).speed
    attacker_capture_radius = agent_params_dict.get(current_attacker_name, {}).capture_radius
    
    # --- Extract defender positions ---
    defender_dict = agent_params.map.defender_dict
    defender_list = list(defender_dict.items())  # [(defender_name, defender_position), ...]

    # --- Compute candidate target region ---
    # Use the attacker's capture radius to compute the target region.
    target_region = compute_x_neighbors(agent_params.map.graph, flag_positions, attacker_capture_radius)

    # --- For each candidate node in P, compute aggregated advantage index ---
    aggregated_advantages = {}
    for node in target_region:
        sum_advantage = 0
        sum_sign = 1
        for defender_name, defender_position in defender_list:
            try:
                # Compute advantage dictionary between attacker and defender.
                _, _, _, advantage_dict = compute_node_dominance_region(
                    attacker_position, defender_position,
                    attacker_speed,
                    agent_params_dict.get(defender_name, {}).speed,
                    agent_params.map.graph
                )
                # Add advantage index for the candidate node; default to inf if node missing.\
                current_advantage = advantage_dict.get(node, float("inf"))
                sum_advantage += current_advantage
                if current_advantage > 0:
                    sum_sign = -1
            except Exception as e:
                print(f"Error computing advantage for attacker {current_attacker_name} vs defender {defender_name} at node {node}: {e}")
                sum_advantage += float("inf")
        aggregated_advantages[node] = (sum_advantage, sum_sign)

    if not aggregated_advantages:
        print("No aggregated advantage computed. Attacker remains in place.")
        state["action"] = current_node
        return
    

    # First, check flag_positions.
    closest_flag = None
    min_flag_distance = float("inf")
    for node in flag_positions:
        if aggregated_advantages[node][1] > 0:
            try:
                distance = nx.shortest_path_length(agent_params.map.graph, current_node, node)
            except Exception as e:
                print(f"Error computing distance from {current_node} to flag node {node}: {e}")
                continue
            if distance < min_flag_distance:
                min_flag_distance = distance
                closest_flag = node

    if closest_flag is not None:
        next_node = compute_shortest_path_step(agent_params.map.graph, current_node, closest_flag, step=attacker_speed)
        if next_node is not None:
            state["action"] = next_node
            return

    # --- Select the candidate node with the best (lowest) aggregated advantage ---
    best_node = min(aggregated_advantages, key=lambda node: aggregated_advantages[node][0])
    best_sum = aggregated_advantages[best_node]
    # print(f"Attacker {current_attacker_name} computed aggregated advantages: {aggregated_advantages}")
    # print(f"Selected target node {best_node} with aggregated advantage {best_sum}")

    # --- Compute the next step towards the best node ---
    next_node = compute_shortest_path_step(agent_params.map.graph, current_node, best_node, step=attacker_speed)
    if next_node is None:
        print("No valid path found; attacker remains in place.")
        next_node = current_node
    state["action"] = next_node

def map_strategy(agent_config):
    """
    Maps each attacker agent to the defined strategy.
    
    Parameters:
        agent_config (dict): Configuration dictionary for all agents.
        
    Returns:
        dict: A dictionary mapping agent names to their strategies.
    """
    strategies = {}
    for name in agent_config.keys():
        strategies[name] = strategy
    return strategies
