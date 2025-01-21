import random
from utilities import *

def strategy(state, FLAG_POSITIONS, FLAG_WEIGHTS, agent):
    current_node = state['curr_pos']
    attacker_positions, defender_positions = extract_sensor_data(state, FLAG_POSITIONS, FLAG_WEIGHTS, agent)
    
    # Find the attacker that is closest to any flag.
    closest_attacker = None
    min_distance = float('inf')
    for attacker in attacker_positions:
        for flag in FLAG_POSITIONS:
            try:
                # Compute the unweighted shortest path length.
                dist = nx.shortest_path_length(agent.map.graph, source=attacker, target=flag)
                if dist < min_distance:
                    min_distance = dist
                    closest_attacker = attacker
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

    if closest_attacker is None:
        # If no attacker is connected to any flag, fallback to a random neighbor.
        neighbor_data = extract_neighbor_sensor_data(state)
        state['action'] = random.choice(neighbor_data)
        return
    
    try:
        next_node = agent.map.shortest_path_to(current_node, closest_attacker, agent.speed)
        state['action'] = next_node
    except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
        print(f"No path found from blue agent at node {current_node} to attacker at node {closest_attacker}: {e}")
        neighbor_data = extract_neighbor_sensor_data(state)
        state['action'] = random.choice(neighbor_data)
    

def map_strategy(agent_config):
    strategies = {}
    for name in agent_config.keys():
        strategies[name] = strategy
    return strategies