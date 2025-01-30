import random
import networkx as nx
from lib.utilities import extract_sensor_data, extract_neighbor_sensor_data

def strategy(state):
    """
    Defines the attacker's strategy to move towards the closest flag.
    
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
    
    closest_flag = None
    min_distance = float('inf')
    
    # Find the closest flag based on shortest path distance
    for flag in flag_positions:
        try:
            # Compute the unweighted shortest path length to the flag
            dist = nx.shortest_path_length(
                agent_params.map.graph, source=current_node, target=flag
            )
            if dist < min_distance:
                min_distance = dist
                closest_flag = flag
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # Skip if no path exists or node is not found
            continue

    if closest_flag is None:
        # Fallback: move to a random neighboring node if no flag is reachable
        neighbor_data = extract_neighbor_sensor_data(state)
        state['action'] = random.choice(neighbor_data)
        return

    try:
        # Determine the next node towards the closest flag
        next_node = agent_params.map.shortest_path_to(
            current_node, closest_flag, agent_params.speed
        )
        state['action'] = next_node
    except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
        # Handle cases where the path cannot be found
        print(f"No path found from red agent at node {current_node} to flag at node {closest_flag}: {e}")
        neighbor_data = extract_neighbor_sensor_data(state)
        state['action'] = random.choice(neighbor_data)

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
