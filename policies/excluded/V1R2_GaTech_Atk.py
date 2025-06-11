import random
from lib.utils.game_utils import *
import pickle
import numpy as np

with open("policy_0_59_v2_2v1.pkl", "rb") as f:
    def1_policy_2v1, def2_policy_2v1, att_policy_2v1, superdefender_policy = pickle.load(f)

with open("policy_0_59_v2_1v1.pkl", "rb") as f:
    att_policy_1v1, def_policy_1v1 = pickle.load(f)

def nearest_attackers(state):
    flag_positions = state['flag_pos']
    flag_weights = state['flag_weight']
    agent_params = state['agent_params']

    # Extract positions of attackers and defenders from sensor data
    attacker_positions, defender_positions = extract_sensor_data(state, flag_positions, flag_weights, agent_params)

    min_distance = float('inf')
    closest_attackers = []  # List to store indices of all attackers at minimum distance

    for flag in flag_positions:
        for attacker_idx, attacker_pos in enumerate(attacker_positions):
            try:
                # Compute the unweighted shortest path length to the flag
                dist = nx.shortest_path_length(agent_params.map.graph, source=attacker_pos, target=flag)

                if dist < min_distance:
                    # Found new minimum, clear list and add this attacker
                    min_distance = dist
                    closest_attackers = [attacker_idx]
                elif dist == min_distance:
                    # This attacker is tied for minimum distance, add to list
                    closest_attackers.append(attacker_idx)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # Skip if no path exists or node is not found
                continue

    return closest_attackers

def nearest_two_defenders(state,attacker_idx):
    flag_positions = state['flag_pos']
    flag_weights = state['flag_weight']
    agent_params = state['agent_params']

    # Extract positions of attackers and defenders from sensor data
    attacker_positions, defender_positions = extract_sensor_data(state, flag_positions, flag_weights, agent_params)
    #closest_attacker = nearest_attacker(state)
    distances = np.full(len(defender_positions),np.inf)
    for defender_idx,defender_pos in enumerate(defender_positions):
        try :
            distances[defender_idx] = nx.shortest_path_length(agent_params.map.graph, source=defender_pos, target=attacker_positions[attacker_idx])
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # Skip if no path exists or node is not found
            continue
    closest_defender_indices = np.argsort(distances)[:2]
    return closest_defender_indices

def nearest_attacker(state):
    flag_positions = state['flag_pos']
    flag_weights = state['flag_weight']
    agent_params = state['agent_params']

    # Extract positions of attackers and defenders from sensor data
    attacker_positions, defender_positions = extract_sensor_data(state, flag_positions, flag_weights, agent_params)
    closest_attacker = None
    min_distance = float('inf')


    for flag in flag_positions:
        for attacker_idx,attacker_pos in enumerate(attacker_positions):
            try:
                # Compute the unweighted shortest path length to the flag
                dist = nx.shortest_path_length(agent_params.map.graph, source=attacker_pos, target=flag)
                if dist < min_distance:
                    min_distance = dist
                    closest_attacker= attacker_idx
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # Skip if no path exists or node is not found
                continue
    return closest_attacker


#Dynamic assignment of the strategies at evry moment in the game
def strategy(state):
    """
    Attacker strategy that reassesses roles on every call, adapting to changes in the game state.
    """
    current_node = state['curr_pos']
    attacker_positions, defender_positions = extract_sensor_data(state, state['flag_pos'], state['flag_weight'],state['agent_params'])
    node_data, _ = extract_map_sensor_data(state)
    no_of_nodes = len(node_data)
    neighbor_data = extract_neighbor_sensor_data(state)

    # Find this attacker's index
    current_attacker_idx = next((i for i, pos in enumerate(attacker_positions) if pos == current_node), None)

    # Determine primary attacker (nearest to goal)
    primary_attacker_idx = nearest_attackers(state)[0]
    print("primary_attacker_idx", primary_attacker_idx)

    # Assign defenders
    if len(defender_positions) >= 2:
        assigned_defenders_idx = nearest_two_defenders(state, primary_attacker_idx)
    else:
        assigned_defenders_idx = list(range(len(defender_positions)))  # Assign all available defenders

    # Prepare neighbor data for policy lookup
    neighbor_data_copy = [n for n in neighbor_data if n != current_node]
    sorted_neighbor = sorted(neighbor_data_copy) + [current_node]

    # Determine if this is the primary attacker
    if current_attacker_idx == primary_attacker_idx:
        # This is the primary attacker
        defenders_against_primary = [defender_positions[i] for i in assigned_defenders_idx if i < len(defender_positions)]

        if len(defenders_against_primary) >= 2:
            # Use 2v1 policy against two defenders
            d1 = min(defenders_against_primary)
            d2 = max(defenders_against_primary)
            defender_index = (d2 * (d2 + 1) // 2) + (d2 - d1)
            joint_state = defender_index * no_of_nodes + current_node
            vector = att_policy_2v1[joint_state]
            print(f'Primary attacker using 2v1 evasion strategy',vector)
        elif defenders_against_primary:
            # Use 1v1 policy against one defender
            defender_pos = defenders_against_primary[0]
            joint_state = defender_pos * no_of_nodes + current_node
            vector = att_policy_1v1[joint_state]
            print(f'Primary attacker using 1v1 evasion strategy',vector)
        else:
            # No defenders assigned to primary - unlikely but handle it
            vector = [1.0 / len(sorted_neighbor)] * len(sorted_neighbor)
            print(f'Primary attacker using random strategy (no defenders)')
    else:
        # This is a secondary attacker
        # Find defenders not assigned to primary attacker
        unassigned_defenders = [defender_positions[i] for i in range(len(defender_positions)) if i not in assigned_defenders_idx]

        if unassigned_defenders:
            # Use 1v1 policy against first unassigned defender
            defender_pos = unassigned_defenders[0]
            joint_state = defender_pos * no_of_nodes + current_node
            vector = att_policy_1v1[joint_state]
            print(f'Secondary attacker using 1v1 evasion against unassigned defender',vector)
        elif defender_positions:
            # No unassigned defenders, use any defender
            defender_pos = defender_positions[0]
            joint_state = defender_pos * no_of_nodes + current_node
            vector = att_policy_1v1[joint_state]
            print(f'Secondary attacker using 1v1 evasion against any defender',vector)
        else:
            # No defenders at all
            vector = [1.0 / len(sorted_neighbor)] * len(sorted_neighbor)
            print(f'Secondary attacker using random strategy (no defenders)')

    # Sample action according to probability distribution
    action_index = np.random.choice(len(vector), p=vector)
    state['action'] = sorted_neighbor[action_index]
    return state


def map_strategy(agent_config):
    strategies = {}
    for name in agent_config.keys():
        strategies[name] = strategy
    return strategies


#import random
# from lib.utilities import *
# import pickle
# import numpy as np
#
# from games.lib.utilities import nearest_attacker, nearest_two_defenders
#
# with open("policy_grid_7*7new.pkl", "rb") as f:
#     def1_policy_2v1, def2_policy_2v1, att_policy_2v1, superdefender_policy = pickle.load(f)
#
# with open("policy_grid_7*7new_1v1.pkl", "rb") as f:
#     att_policy_1v1,def_policy_1v1 = pickle.load(f)
#
# # Global variables
# primary_attacker_idx = None
# assigned_defenders_idx = None
# initial_assignment_done = False
#
#
# def strategy(state):
#     global primary_attacker_idx, assigned_defenders_idx, initial_assignment_done
#
#     current_node = state['curr_pos']
#     attacker_positions, defender_positions = extract_sensor_data(state, state['flag_pos'], state['flag_weight'],state['agent_params'])
#     node_data, _ = extract_map_sensor_data(state)
#     no_of_nodes = len(node_data)
#     neighbor_data = extract_neighbor_sensor_data(state)
#
#     # Create active masks (True for active agents)
#     active_attackers = [pos is not None for pos in attacker_positions]
#     active_defenders = [pos is not None for pos in defender_positions]
#
#     # Only run initial assignment once
#     if not initial_assignment_done:
#         # Find active attackers and select primary
#         active_attacker_indices = [i for i, active in enumerate(active_attackers) if active]
#         if active_attacker_indices:
#             primary_attacker_idx = nearest_attacker(state)
#
#             # Assign defenders to primary attacker
#             active_defender_indices = [i for i, active in enumerate(active_defenders) if active]
#             if len(active_defender_indices) >= 2:
#                 assigned_defenders_idx = nearest_two_defenders(state, primary_attacker_idx)
#             else:
#                 assigned_defenders_idx = active_defender_indices
#
#             initial_assignment_done = True
#
#     # Find this attacker's index
#     current_attacker_idx = next((i for i, pos in enumerate(attacker_positions)
#                                  if pos == current_node and active_attackers[i]), None)
#
#     # Skip if this attacker is inactive (shouldn't happen since it's the current agent)
#     if current_attacker_idx is None:
#         state['action'] = current_node  # Stay in place
#
#     # Prepare neighbor data for policy lookup
#     neighbor_data_copy = [n for n in neighbor_data if n != current_node]
#     sorted_neighbor = sorted(neighbor_data_copy) + [current_node]
#
#     # Determine strategy based on whether this is the primary attacker
#     if current_attacker_idx == primary_attacker_idx:
#         # Primary attacker strategy
#         active_def_positions = [pos for i, pos in enumerate(defender_positions) if active_defenders[i]]
#
#         if len(active_def_positions) >= 2:
#             # Use 2v1 strategy with two defenders
#             d1 = min(active_def_positions)
#             d2 = max(active_def_positions)
#             defender_index = (d2 * (d2 + 1) // 2) + (d2 - d1)
#             joint_state = defender_index * no_of_nodes + current_node
#             vector = att_policy_2v1[joint_state]
#             print('primary_attacker_2v1', current_attacker_idx)
#         elif len(active_def_positions) == 1:
#             # Use 1v1 strategy with one defender
#             defender_pos = active_def_positions[0]
#             joint_state = defender_pos * no_of_nodes + current_node
#             vector = att_policy_1v1[joint_state]
#             print('primary_attacker_1v1', current_attacker_idx)
#         else:
#             # No active defenders, use simple strategy
#             vector = [1.0 / len(sorted_neighbor)] * len(sorted_neighbor)
#             print('primary_attacker_no_defenders', current_attacker_idx)
#     else:
#         # Secondary attacker strategy
#         active_def_positions = [pos for i, pos in enumerate(defender_positions)
#                                 if active_defenders[i] and (
#                                             assigned_defenders_idx is None or i not in assigned_defenders_idx)]
#
#         if active_def_positions:
#             # Use 1v1 strategy with first unassigned defender
#             defender_pos = active_def_positions[0]
#             joint_state = defender_pos * no_of_nodes + current_node
#             vector = att_policy_1v1[joint_state]
#             print("secondary_attacker_1v1", current_attacker_idx)
#         else:
#             # No unassigned defenders available, use any active defender
#             active_def_positions = [pos for i, pos in enumerate(defender_positions) if active_defenders[i]]
#             if active_def_positions:
#                 defender_pos = active_def_positions[0]
#                 joint_state = defender_pos * no_of_nodes + current_node
#                 vector = att_policy_1v1[joint_state]
#                 print("secondary_attacker_any_defender", current_attacker_idx)
#             else:
#                 # No active defenders at all
#                 vector = [1.0 / len(sorted_neighbor)] * len(sorted_neighbor)
#                 print("secondary_attacker_no_defenders", current_attacker_idx)
#
#     # Sample action according to probability distribution
#     action_index = np.random.choice(len(vector), p=vector)
#     state['action'] = sorted_neighbor[action_index]
#
# def map_strategy(agent_config):
#     strategies = {}
#     for name in agent_config.keys():
#         strategies[name] = strategy
#     return strategies


# # Global variable to store primary attacker index
# primary_attacker_idx = None
# initial_assignment_done = False
#
#
# def strategy(state):
#     global primary_attacker_idx, initial_assignment_done
#
#     current_node = state['curr_pos']
#     flag_positions = state['flag_pos']
#     flag_weights = state['flag_weight']
#     agent_params = state['agent_params']
#     agent_name = state['name']
#
#     print("state", state.keys())
#     print('state-name', state['name'])
#     print("agent_params", vars(agent_params))
#     print("agent_parameters.maps", vars(agent_params.map))
#     print('agent_positions', state['agent_params'].map.agent_positions)
#
#     attacker_positions, defender_positions = extract_sensor_data(state, flag_positions, flag_weights, agent_params)
#     node_data, edge_data = extract_map_sensor_data(state)
#     no_of_nodes = len(node_data)
#
#     print('attacker_po', attacker_positions)
#     print('defender_po', defender_positions)
#     neighbor_data = extract_neighbor_sensor_data(state)
#     print("attacker_neighbor", neighbor_data)
#
#     # One-time role assignment at the beginning of the game
#     if not initial_assignment_done:
#         # Find the nearest attacker at the beginning
#         primary_attacker_idx = nearest_attacker(state)
#         initial_assignment_done = True
#         print(f"Primary attacker assigned: attacker index {primary_attacker_idx}")
#
#     # Determine current attacker's index
#     current_attacker_idx = None
#     for idx, pos in enumerate(attacker_positions):
#         if pos == current_node:
#             current_attacker_idx = idx
#             break
#     # Get indices of the two closest defenders
#     closest_def_idx = nearest_two_defenders(state, primary_attacker_idx)
#
#     # Create boolean mask for all defenders
#     mask = np.ones(len(defender_positions), dtype=bool)
#     mask[closest_def_idx] = False  # Set the closest defenders to False in the mask
#
#     # Use the mask to separate the arrays
#     closest_defenders = np.array(defender_positions)[closest_def_idx]  # These are the closest defenders
#     other_defender = np.array(defender_positions)[mask]  # These are all the other defenders
#
#     # Check if this is the primary attacker
#     if current_attacker_idx == primary_attacker_idx:
#         # Strategy for the primary attacker
#         print("I am the primary attacker, implementing primary strategy")
#         # Implement the primary attacker strategy
#         d1 = min(closest_defenders)
#         d2 = max(closest_defenders)
#         defender_index = (d2 * (d2 + 1) // 2) + (d2 - d1)
#         joint_state = defender_index * no_of_nodes + attacker_positions[current_attacker_idx]
#
#         neighbor_data_copy = neighbor_data.copy()
#         if current_node in neighbor_data_copy:
#             neighbor_data_copy.remove(current_node)
#
#         sorted_neighbor = sorted(neighbor_data_copy)
#         sorted_neighbor.append(current_node)
#
#         vector = att_policy_2v1[joint_state]
#         print('Primary_attacker_policies', vector)
#
#         # For a mixed strategy, sample an action according to the probability distribution
#         action_index = np.random.choice(len(vector), p=vector)
#         state['action'] = sorted_neighbor[action_index]
#     else:
#         # Strategy for secondary attackers
#         print("I am a secondary attacker, implementing secondary strategy")
#
#         joint_state = other_defender * no_of_nodes + attacker_positions[current_attacker_idx]
#         neighbor_data_copy = neighbor_data.copy()
#         if current_node in neighbor_data_copy:
#             neighbor_data_copy.remove(current_node)
#
#         sorted_neighbor = sorted(neighbor_data_copy)
#         sorted_neighbor.append(current_node)
#         vector = att_policy_1v1[joint_state]
#         print('secondary_attacker_policies', vector)
#
#         # For a mixed strategy, sample an action according to the probability distribution
#         action_index = np.random.choice(len(vector), p=vector)
#         state['action'] = sorted_neighbor[action_index]
#
#     print("att_finish")
# def strategy(state):
#     current_node = state['curr_pos']
#     flag_positions = state['flag_pos']
#     flag_weights = state['flag_weight']
#     agent_params = state['agent_params']
#     print("state",state.keys())
#     print('state-name',state['name'])
#     print("agent_params",vars(agent_params))
#     print("agent_parameters.maps",vars(agent_params.map))
#     print('agent_positions',state['agent_params'].map.agent_positions)
#     attacker_positions, defender_positions = extract_sensor_data(state, flag_positions, flag_weights, agent_params)
#     print("closest _attacker")
#     node_data, edge_data = extract_map_sensor_data(state)
#     no_of_nodes = len(node_data)
#     #print("Number of nodes:", no_of_nodes)
#     print('attacker_po',attacker_positions)
#     print('defender_po',defender_positions)
#     neighbor_data = extract_neighbor_sensor_data(state)
#     print("attacker_neighbor",neighbor_data)
#     #joint_state = defender_positions[0]*no_of_nodes*no_of_nodes + defender_positions[1]*no_of_nodes+attacker_positions[0]
#     closest_attacker = nearest_attacker(state)
#     d1 = min(defender_positions)
#     d2 = max(defender_positions)
#     defender_index = (d2 * (d2 + 1) // 2) + (d2 - d1)
#     joint_state = defender_index*no_of_nodes+attacker_positions[0]
#     neighbor_data.remove(attacker_positions[0])
#     sorted_neighbor = sorted(neighbor_data)
#     sorted_neighbor.append(attacker_positions[0])
#     vector = att_policy_2v1[joint_state]
#     print('attacker_policies',vector)
#     non_zero_index = 0
#     vector = att_policy_2v1[joint_state]
#     # For a mixed strategy, sample an action according to the probability distribution
#     action_index = np.random.choice(len(vector), p=vector)
#     state['action'] = sorted_neighbor[action_index]
#
#     print("att_finish")
#     # sensor_data = state['sensor']
#     # print("Sensor data:", sensor_data.keys())
#     # print("Neighbor Data",neighbor_data)
#     #state['action'] = random.choice(neighbor_data)