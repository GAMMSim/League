import random
import networkx as nx
from lib.utils.sensor_utils import *

def find_key_nodes(graph, flag_positions):
    key_nodes = []
    for flag in flag_positions:
        neighbors = list(graph.neighbors(flag))
        high_degree_neighbors = sorted(neighbors, key=lambda n: graph.degree[n], reverse=True)
        if high_degree_neighbors:
            key_nodes.append(high_degree_neighbors[0])
    return key_nodes

def get_patrol_path(graph, flag_positions, key_nodes):
    if graph.is_directed():
        graph = graph.to_undirected()  
    
    subgraph = graph.subgraph(flag_positions + key_nodes)
    mst = nx.minimum_spanning_tree(subgraph)
    path = list(nx.dfs_preorder_nodes(mst, source=random.choice(flag_positions)))
    return path + [path[0]]  # Close the loop

def strategy(state):
    """Defines the defender's strategy"""
    current_node = state['curr_pos']
    flag_positions = state['flag_pos']
    flag_weights = state['flag_weight']
    agent_params = state['agent_params']
    
    # Extract attacker and defender positions
    attacker_positions, defender_positions = extract_sensor_data(
        state, flag_positions, flag_weights, agent_params
    )
    
    graph = agent_params.map.graph
    key_nodes = find_key_nodes(graph, flag_positions)
    patrol_path = get_patrol_path(graph, flag_positions, key_nodes)
    
    for attacker in attacker_positions:
        for flag in flag_positions:
            if nx.shortest_path_length(graph, source=attacker, target=flag) <= len(attacker_positions + flag_positions): 
                closest_defender = min(
                    defender_positions,
                    key=lambda d: nx.shortest_path_length(graph, source=d, target=attacker),
                    default=None
                )
                if closest_defender == current_node:
                    try:
                        next_node = agent_params.map.shortest_path_to(current_node, attacker, agent_params.speed)
                        state['action'] = next_node
                        return
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass 
    
    if current_node in patrol_path:
        next_index = (patrol_path.index(current_node) + 1) % len(patrol_path)
        state['action'] = patrol_path[next_index]
    else:
        nearest_patrol_node = min(patrol_path, key=lambda n: nx.shortest_path_length(graph, source=current_node, target=n))
        state['action'] = agent_params.map.shortest_path_to(current_node, nearest_patrol_node, agent_params.speed)

def map_strategy(agent_config):
    strategies = {}
    for name in agent_config.keys():
        strategies[name] = strategy
    return strategies
