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
    env_graph=agent_params.map.graph
    closest_attacker = None
    min_distance = float('inf')
    alpha=0.5
    new_distance=[]
    DEFENDER_GLOBAL_CAPTURE_RADIUS = 1
    reachable_attacker=[]
    for flag in flag_positions:
        time_reach_flag=nx.shortest_path_length(
                    agent_params.map.graph, source=current_node, target=flag
                )
        reachable_nodes = [
            node for node, dist in nx.single_source_shortest_path_length(env_graph, flag).items()
            if 0 < dist <= time_reach_flag + DEFENDER_GLOBAL_CAPTURE_RADIUS
        ]
        # print(reachable_nodes)
    for attacker in attacker_positions:
        if attacker in reachable_nodes:
            print('not reachable')
        else:
            reachable_attacker.append(attacker)
    print(reachable_attacker)
    if not reachable_attacker:
        print('Go to the closest flag')
        for flag in flag_positions:
            new_distance.append(nx.shortest_path_length(
                agent_params.map.graph, source=current_node, target=flag
            ))
        min_index = new_distance.index(min(new_distance))
        path=nx.shortest_path(
                            agent_params.map.graph, source=current_node, target=flag_positions[min_index]
                        )
        next_node = agent_params.map.shortest_path_to(
            current_node, path[-1], agent_params.speed
        )
        state['action'] = next_node
        return


    else:
        print('Go to the closest attacker')
        for attacker in reachable_attacker:
            print(attacker)
            distance2 = nx.shortest_path_length(
                agent_params.map.graph, source=current_node, target=attacker)
            for flag in flag_positions:
                neighbors = list(env_graph.neighbors(flag))
                print(neighbors)
                try:
                    # Compute the unweighted shortest path length to the attacker
                    distance = nx.shortest_path_length(
                        agent_params.map.graph, source=attacker, target=flag
                    )
                    total_distance = alpha*distance + (1-alpha)*distance2
                    if total_distance < min_distance:
                        min_distance = total_distance
                        closest_attacker = attacker
                        path = nx.shortest_path(
                            agent_params.map.graph, source=closest_attacker, target=flag
                        )
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    # Skip if no path exists or node is not found
                    continue

        for node in path:
            time_reach_defender=nx.shortest_path_length(
                        agent_params.map.graph, source=current_node, target=node
                    )
            time_reach_attacker= nx.shortest_path_length(
                agent_params.map.graph, source=closest_attacker, target=node
            )
            if time_reach_defender <= time_reach_attacker:
                defender_target=node
                break
        if time_reach_defender <= time_reach_attacker:
            next_node = agent_params.map.shortest_path_to(
                        current_node, defender_target, agent_params.speed
                    )
            state['action'] = next_node
            return
        else:
            next_node = agent_params.map.shortest_path_to(
                current_node, path[-1], agent_params.speed
            )
            state['action'] = next_node
            return

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
