import random
import networkx as nx
from lib.utils.sensor_utils import *

def find_midway_nodes(graph, attacker_pos, flag_positions):
    midway_nodes = []
    for flag in flag_positions:
        path_lengths = {
            node: nx.shortest_path_length(graph, source=attacker_pos, target=node)
            for node in graph.nodes
        }
        sorted_nodes = sorted(path_lengths.keys(), key=lambda n: (path_lengths[n], -graph.degree[n]))
        midway_nodes.append(sorted_nodes[0]) 
    return midway_nodes

def count_defenders_near_flags(graph, flag_positions, defender_positions, radius=2):
    flag_defender_count = {}
    for flag in flag_positions:
        flag_defender_count[flag] = sum(
            1 for defender in defender_positions
            if nx.shortest_path_length(graph, source=flag, target=defender) <= radius
        )
    return flag_defender_count

def distribute_attackers(attacker_positions, flag_positions, flag_defender_count):
    flag_ratios = {flag: 1 / (flag_defender_count.get(flag, 1) + 1) for flag in flag_positions}
    sorted_flags = sorted(flag_positions, key=lambda f: flag_ratios[f], reverse=True)
    assigned_targets = {att: sorted_flags[i % len(sorted_flags)] for i, att in enumerate(attacker_positions)}
    return assigned_targets

def strategy(state):
    """Defines the attacker's strategy"""
    current_node = state['curr_pos']
    flag_positions = state['flag_pos']
    flag_weights = state['flag_weight']
    agent_params = state['agent_params']
    
    # Extract attacker and defender positions
    attacker_positions, defender_positions = extract_sensor_data(
        state, flag_positions, flag_weights, agent_params
    )
    
    graph = agent_params.map.graph
    midway_nodes = find_midway_nodes(graph, current_node, flag_positions)
    
    flag_defender_count = count_defenders_near_flags(graph, flag_positions, defender_positions, len(defender_positions))
    assigned_targets = distribute_attackers(attacker_positions, flag_positions, flag_defender_count)
    target_flag = assigned_targets[current_node]
    
    safe_midway_nodes = [node for node in midway_nodes if node not in defender_positions]
    target_midway_node = random.choice(safe_midway_nodes) if safe_midway_nodes else random.choice(midway_nodes)
    
    if len(attacker_positions) > len(defender_positions):
        extra_attackers = len(attacker_positions) - len(defender_positions)
        defender_dense_regions = sorted(defender_positions, key=lambda d: len(list(graph.neighbors(d))), reverse=True)
        
        for i, attacker in enumerate(attacker_positions[:extra_attackers]):
            distraction_target = defender_dense_regions[i % len(defender_dense_regions)]
            state['action'] = agent_params.map.shortest_path_to(current_node, distraction_target, agent_params.speed)
            return
    
    if current_node == target_midway_node:
        state['action'] = agent_params.map.shortest_path_to(current_node, target_flag, agent_params.speed)
        return

    state['action'] = agent_params.map.shortest_path_to(current_node, target_midway_node, agent_params.speed)


def map_strategy(agent_config):
    strategies = {}
    for name in agent_config.keys():
        strategies[name] = strategy
    return strategies
