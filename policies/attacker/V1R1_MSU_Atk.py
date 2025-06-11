import random
import networkx as nx
from lib.utils.sensor_utils import extract_sensor_data, extract_neighbor_sensor_data

'''Mild revision of closest flag strategy where we retreat if a defenser gets too close'''

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
    
    closest_defender_dist = float('inf')
    closest_defender_pos = None
    for defender in defender_positions:
        try:
            d_dist = nx.shortest_path_length(agent_params.map.graph, source=current_node, target=defender)
            if  d_dist < closest_defender_dist:
                closest_defender_dist = d_dist
                closest_defender_pos = defender
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
    


    try:
        # Determine the next node towards the closest flag
        if closest_defender_dist >= 3:
            next_node = agent_params.map.shortest_path_to(
                current_node, closest_flag, agent_params.speed
            )
            state['action'] = next_node
        else:
            neighbors = nx.neighbors(agent_params.map.graph, current_node)
            safe_node = None
            for neighbour in neighbors:
                n_dist = nx.shortest_path_length(agent_params.map.graph, source=neighbour, target=closest_defender_pos)
                if n_dist > 2:
                    safe_node = neighbour
                    break
            # if there is a safe node move there
            if safe_node is not None:
                state['action'] = neighbour
            else:
                # we're doomed probably
                neighbor_data = extract_neighbor_sensor_data(state)
                state['action'] = random.choice(neighbor_data)

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
