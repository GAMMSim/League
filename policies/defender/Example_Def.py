import random
import networkx as nx
from lib.utils.sensor_utils import extract_sensor_data, extract_neighbor_sensor_data

def strategy(state):
    """
    Defines the defender's strategy to move towards the closest attacker.
    
    Parameters:
        state (dict): The current state of the game, including positions and parameters.
    """
    current_node = state['curr_pos']
    flag_positions = state['flag_pos']
    flag_weights = state['flag_weight']
    agent_params = state['agent_params']
    
    # Extract positions of attackers and defenders from sensor data
    attacker_positions, defender_positions = extract_sensor_data(
        state, flag_positions, flag_weights, agent_params
    )
    
    closest_attacker = None
    min_distance = float('inf')
    
    # Find the closest attacker based on shortest path distance
    for attacker in attacker_positions:
        for flag in flag_positions:
            try:
                # Compute the unweighted shortest path length to the attacker
                dist = nx.shortest_path_length(
                    agent_params.map.graph, source=attacker, target=flag
                )
                if dist < min_distance:
                    min_distance = dist
                    closest_attacker = attacker
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # Skip if no path exists or node is not found
                continue

    if closest_attacker is None:
        # Fallback: move to a random neighboring node if no attacker is found
        neighbor_data = extract_neighbor_sensor_data(state)
        state['action'] = random.choice(neighbor_data)
        return

    try:
        # Determine the next node towards the closest attacker
        next_node = agent_params.map.shortest_path_to(
            current_node, closest_attacker, agent_params.speed
        )
        state['action'] = next_node
    except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
        # Handle cases where the path cannot be found
        print(f"No path found from blue agent at node {current_node} to attacker at node {closest_attacker}: {e}")
        neighbor_data = extract_neighbor_sensor_data(state)
        state['action'] = random.choice(neighbor_data)

def map_strategy(agent_config):
    """
    Maps each defender agent to the defined strategy.
    
    Parameters:
        agent_config (dict): Configuration dictionary for all agents.
        
    Returns:
        dict: A dictionary mapping agent names to their strategies.
    """
    strategies = {}
    for name in agent_config.keys():
        strategies[name] = strategy
    return strategies
