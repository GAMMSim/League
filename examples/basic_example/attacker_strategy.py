import random
from utilities import *

def strategy(state, FLAG_POSITIONS, FLAG_WEIGHTS, agent):
    current_node = state['curr_pos']
    attacker_positions, defender_positions = extract_sensor_data(state, FLAG_POSITIONS, FLAG_WEIGHTS, agent)
    
    closest_flag = None
    min_distance = float('inf')
    for flag in FLAG_POSITIONS:
        try:
            # Compute the unweighted shortest path length.
            dist = nx.shortest_path_length(agent.map.graph, source=current_node, target=flag)
            if dist < min_distance:
                min_distance = dist
                closest_flag = flag
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

    if closest_flag is None:
        # Fallback: if no flag is reachable, use a random neighbor.
        neighbor_data = extract_neighbor_sensor_data(state)
        state['action'] = random.choice(neighbor_data)
        return

    # Compute the shortest path from the agent's node to the chosen flag.
    try:
        next_node = agent.map.shortest_path_to(current_node, closest_flag, agent.speed)
        state['action'] = next_node
    except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
        print(f"No path found from red agent at node {current_node} to flag at node {closest_flag}: {e}")
        neighbor_data = extract_neighbor_sensor_data(state)
        state['action'] = random.choice(neighbor_data)
    

def map_strategy(agent_config):
    strategies = {}
    for name in agent_config.keys():
        strategies[name] = strategy
    return strategies