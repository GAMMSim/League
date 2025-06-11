import random
import pickle
from lib.utils.game_utils import *
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


def strategy(state):
    """
    Simplified defender strategy ensuring all defenders are active.
    """
    current_node = state['curr_pos']
    attacker_positions, defender_positions = extract_sensor_data(state, state['flag_pos'], state['flag_weight'],state['agent_params'])
    node_data, _ = extract_map_sensor_data(state)
    no_of_nodes = len(node_data)
    neighbor_data = extract_neighbor_sensor_data(state)
    agent_name = state.get('name', '')

    # Find this defender's index
    current_defender_idx = next((i for i, pos in enumerate(defender_positions) if pos == current_node), None)

    # Identify primary attacker (nearest to goal)
    primary_attacker_idx = nearest_attackers(state)[0]

    # Assign two nearest defenders to primary attacker (if possible)
    assigned_defenders = nearest_two_defenders(state, primary_attacker_idx)

    # Prepare neighbor data for policy lookup
    neighbor_data_copy = [n for n in neighbor_data if n != current_node]
    sorted_neighbor = sorted(neighbor_data_copy) + [current_node]

    # Determine if this defender is assigned to primary attacker
    is_primary_defender = current_defender_idx in assigned_defenders

    # IMPORTANT: Even if we're not in assigned_defenders, we need a valid target
    target_attacker_idx = primary_attacker_idx  # Default to primary

    # If this defender isn't assigned to primary, target a secondary attacker
    if not is_primary_defender:
        # Find any active secondary attacker
        secondary_attackers = [i for i, pos in enumerate(attacker_positions)
                               if i != primary_attacker_idx and pos is not None]
        if secondary_attackers:
            target_attacker_idx = secondary_attackers[0]

    # Get target position
    target_pos = attacker_positions[target_attacker_idx]

    if is_primary_defender and len(assigned_defenders) >= 2:
        # This is part of a 2v1 strategy
        # Get positions of assigned defenders
        active_assigned_defenders = [defender_positions[i] for i in assigned_defenders if defender_positions[i] is not None]

        if len(active_assigned_defenders) >= 2:
            d1 = min(active_assigned_defenders)
            d2 = max(active_assigned_defenders)
            defender_index = (d2 * (d2 + 1) // 2) + (d2 - d1)
            joint_state = defender_index * no_of_nodes + target_pos

            if d1 == d2:
                # Both defenders at same position
                vector = def2_policy_2v1[joint_state]  # Both use same policy when at same position
                print(f'{agent_name} strategies (equal positions)', vector)
            elif current_node == d1:
                # This defender is at min position
                vector = def1_policy_2v1[joint_state]
                print(f'{agent_name} strategies (min position)', vector)
            else:
                # This defender is at max position
                vector = def2_policy_2v1[joint_state]
                print(f'{agent_name} strategies (max position)', vector)
        else:
            # Fallback to 1v1
            joint_state = current_node * no_of_nodes + target_pos
            vector = def_policy_1v1[joint_state]
            print(f'{agent_name} using 1v1 strategy against primary attacker', vector)
    else:
        # Use 1v1 strategy
        joint_state = current_node * no_of_nodes + target_pos
        vector = def_policy_1v1[joint_state]
        print(f'{agent_name} using 1v1 strategy against secondary attacker', vector)

    # Sample action
    action_index = np.random.choice(len(vector), p=vector)
    state['action'] = sorted_neighbor[action_index]

def map_strategy(agent_config):
    strategies = {}
    for name in agent_config.keys():
        strategies[name] = strategy
    return strategies


#Dynamic assignment of the strategies at evry moment in the game
# def strategy(state):
#     """
#     Defender strategy that reassigns roles on every call, adapting to changes in the game state.
#     """
#     current_node = state['curr_pos']
#     attacker_positions, defender_positions = extract_sensor_data(state, state['flag_pos'], state['flag_weight'],state['agent_params'])
#     node_data, _ = extract_map_sensor_data(state)
#     no_of_nodes = len(node_data)
#     neighbor_data = extract_neighbor_sensor_data(state)
#     agent_name = state.get('name', '')
#
#     # Find this defender's index by matching current position
#     current_defender_idx = next((i for i, pos in enumerate(defender_positions) if pos == current_node), None)
#
#     # Assign primary attacker (nearest to goal)
#     primary_attacker_idx = nearest_attacker(state)
#     print("primary_attacker_idx_defender", primary_attacker_idx)
#
#     # Assign two nearest defenders to primary attacker
#     if len(defender_positions) >= 2:
#         assigned_defenders_idx = nearest_two_defenders(state, primary_attacker_idx)
#     else:
#         assigned_defenders_idx = list(range(len(defender_positions)))  # Assign all available defenders
#
#     # Prepare neighbor data for policy lookup
#     neighbor_data_copy = [n for n in neighbor_data if n != current_node]
#     sorted_neighbor = sorted(neighbor_data_copy) + [current_node]
#
#     # Determine if this defender is assigned to primary attacker
#     is_assigned_to_primary = current_defender_idx in assigned_defenders_idx
#
#     if is_assigned_to_primary and len(assigned_defenders_idx) >= 2:
#         # This defender is part of a 2v1 strategy against primary attacker
#         active_assigned_defenders = [defender_positions[i] for i in assigned_defenders_idx]
#
#         # Use 2v1 strategy
#         d1 = min(active_assigned_defenders)
#         d2 = max(active_assigned_defenders)
#         defender_index = (d2 * (d2 + 1) // 2) + (d2 - d1)
#         primary_pos = attacker_positions[primary_attacker_idx]
#         joint_state = defender_index * no_of_nodes + primary_pos
#
#         # Choose policy based on position
#         if d1 == d2:
#             # Both defenders at same position
#             vector = def2_policy_2v1[joint_state]
#             # Both use same policy when at same position
#             print(f'{agent_name} using 2v1 strategy (equal positions)',vector)
#         elif current_node == d1:
#             # This defender is at min position
#             vector = def1_policy_2v1[joint_state]
#             print(f'{agent_name} using 2v1 strategy (min position)',vector)
#         else:
#             # This defender is at max position
#             vector = def2_policy_2v1[joint_state]
#             print(f'{agent_name} using 2v1 strategy (max position)',vector)
#     elif is_assigned_to_primary:
#         # Only one defender assigned to primary attacker, use 1v1
#         primary_pos = attacker_positions[primary_attacker_idx]
#         joint_state = current_node * no_of_nodes + primary_pos
#         vector = def_policy_1v1[joint_state]
#         print(f'{agent_name} using 1v1 strategy against primary attacker',vector)
#     else:
#         # This defender targets secondary attackers
#         secondary_attackers = [pos for i, pos in enumerate(attacker_positions) if i != primary_attacker_idx]
#
#         if secondary_attackers:
#             # Target first secondary attacker
#             target_attacker = secondary_attackers[0]
#             joint_state = current_node * no_of_nodes + target_attacker
#             vector = def_policy_1v1[joint_state]
#             print(f'{agent_name} using 1v1 strategy against secondary attacker',vector)
#         else:
#             # No secondary attackers, help with primary
#             primary_pos = attacker_positions[primary_attacker_idx]
#             joint_state = current_node * no_of_nodes + primary_pos
#             vector = def_policy_1v1[joint_state]
#             print(f'{agent_name} helping with primary attacker',vector)
#
#     # Sample action according to probability distribution
#     action_index = np.random.choice(len(vector), p=vector)
#     state['action'] = sorted_neighbor[action_index]
#
################################################################################################################3
# import random
# import pickle
# from lib.utilities import *
# import numpy as np
#
# with open("policy_grid_7*7new.pkl", "rb") as f:
#     def1_policy_2v1, def2_policy_2v1, att_policy_2v1, superdefender_policy = pickle.load(f)
#
# with open("policy_grid_7*7new_1v1.pkl", "rb") as f:
#     att_policy_1v1, def_policy_1v1 = pickle.load(f)
#
# # Global variables to track attacker/defender assignment
# primary_attacker_idx = None
# assigned_defenders_idx = None
# last_active_attackers = None
#
#
# def strategy(state):
#     global primary_attacker_idx, assigned_defenders_idx, last_active_attackers
#
#     current_node = state['curr_pos']
#     attacker_positions, defender_positions = extract_sensor_data(state, state['flag_pos'], state['flag_weight'],state['agent_params'])
#     node_data, _ = extract_map_sensor_data(state)
#     no_of_nodes = len(node_data)
#     neighbor_data = extract_neighbor_sensor_data(state)
#     agent_name = state.get('name', '')
#     # Get active attacker indices
#     active_attacker_indices = [i for i, active in enumerate(active_attackers) if active]
#
#     # Check if attacker status has changed (attacker caught or reactivated)
#     attacker_status_changed = (last_active_attackers is not None and
#                                active_attackers != last_active_attackers)
#
#     # Update last_active_attackers for future comparison
#     last_active_attackers = active_attackers.copy()
#
#     # Reassign roles if initial assignment hasn't been done OR attacker status changed
#     if primary_attacker_idx is None or attacker_status_changed:
#         if active_attacker_indices:
#             # Select the nearest active attacker as primary
#             primary_attacker_idx = nearest_attacker(state)
#
#             # Assign defenders to primary attacker
#             active_defender_indices = [i for i, active in enumerate(active_defenders) if active]
#             if len(active_defender_indices) >= 2:
#                 assigned_defenders_idx = nearest_two_defenders(state, primary_attacker_idx)
#             else:
#                 assigned_defenders_idx = active_defender_indices
#
#             print(f"{agent_name}: Reassigning roles - Primary attacker: {primary_attacker_idx}, "
#                   f"Assigned defenders: {assigned_defenders_idx}")
#         else:
#             # No active attackers, reset assignments
#             primary_attacker_idx = None
#             assigned_defenders_idx = None
#
#     # Find this defender's index
#     current_defender_idx = next((i for i, pos in enumerate(defender_positions)
#                                  if pos == current_node and active_defenders[i]), None)
#
#     # Skip if this defender is inactive (shouldn't happen)
#     if current_defender_idx is None:
#         state['action'] = current_node  # Stay in place
#         return state
#
#     # Get positions of active attackers
#     active_att_positions = [pos for i, pos in enumerate(attacker_positions) if active_attackers[i]]
#
#     # Skip if no active attackers
#     if not active_att_positions:
#         state['action'] = random.choice(neighbor_data)
#         return state
#
#     # Prepare neighbor data
#     neighbor_data_copy = [n for n in neighbor_data if n != current_node]
#     sorted_neighbor = sorted(neighbor_data_copy) + [current_node]
#
#     # Determine if this defender is assigned to primary attacker
#     is_assigned_to_primary = (assigned_defenders_idx is not None and
#                               current_defender_idx in assigned_defenders_idx and
#                               primary_attacker_idx is not None and
#                               active_attackers[primary_attacker_idx])
#
#     if is_assigned_to_primary:
#         # This defender is assigned to primary attacker
#         active_assigned_defenders = [defender_positions[i] for i in assigned_defenders_idx
#                                      if i < len(defender_positions) and active_defenders[i]]
#
#         if len(active_assigned_defenders) >= 2:
#             # Use 2v1 strategy
#             d1 = min(active_assigned_defenders)
#             d2 = max(active_assigned_defenders)
#             defender_index = (d2 * (d2 + 1) // 2) + (d2 - d1)
#             primary_pos = attacker_positions[primary_attacker_idx]
#             joint_state = defender_index * no_of_nodes + primary_pos
#
#             # Choose policy based on position
#             if d1 == d2:
#                 # Both defenders at same position
#                 vector = def1_policy_2v1[joint_state]  # Both use same policy when at same position
#                 print(f'{agent_name} strategies (equal positions)', vector)
#             elif current_node == d1:
#                 # This defender is at min position
#                 vector = def1_policy_2v1[joint_state]
#                 print(f'{agent_name} strategies (min position)', vector)
#             else:
#                 # This defender is at max position
#                 vector = def2_policy_2v1[joint_state]
#                 print(f'{agent_name} strategies (max position)', vector)
#         else:
#             # Only one active assigned defender, use 1v1
#             primary_pos = attacker_positions[primary_attacker_idx]
#             joint_state = current_node * no_of_nodes + primary_pos
#             vector = def_policy_1v1[joint_state]
#             print(f'{agent_name} strategies (1v1 with primary)', vector)
#     else:
#         # This defender targets secondary attackers
#         secondary_attackers = [pos for i, pos in enumerate(attacker_positions)
#                                if active_attackers[i] and i != primary_attacker_idx]
#
#         if secondary_attackers:
#             # Target first secondary attacker
#             target_attacker = secondary_attackers[0]
#             joint_state = current_node * no_of_nodes + target_attacker
#             vector = def_policy_1v1[joint_state]
#             print(f'{agent_name} strategies (1v1 with secondary)', vector)
#         elif active_att_positions:
#             # No secondary attackers, help with primary
#             primary_pos = attacker_positions[primary_attacker_idx]
#             joint_state = current_node * no_of_nodes + primary_pos
#             vector = def_policy_1v1[joint_state]
#             print(f'{agent_name} strategies (helping with primary)', vector)
#         else:
#             # No active attackers (unlikely to reach here)
#             state['action'] = random.choice(neighbor_data)
#             return state
#
#     # Sample action according to probability distribution
#     action_index = np.random.choice(len(vector), p=vector)
#     state['action'] = sorted_neighbor[action_index]
#     return state
#
#
# def map_strategy(agent_config):
#     strategies = {}
#     for name in agent_config.keys():
#         strategies[name] = strategy
#     return strategies


# import random
# import pickle
# from lib.utilities import *
# import numpy as np
#
# with open("policy_grid_7*7new.pkl", "rb") as f:
#     def1_policy_2v1, def2_policy_2v1, att_policy_2v1, superdefender_policy = pickle.load(f)
#
# with open("policy_grid_7*7new_1v1.pkl", "rb") as f:
#     att_policy_1v1,def_policy_1v1 = pickle.load(f)
#
# primary_attacker_idx = None
# assigned_defenders_idx = None
# initial_assignment_done = False
# def strategy(state):
#     global primary_attacker_idx, assigned_defenders_idx, initial_assignment_done
#
#     current_node = state['curr_pos']
#     attacker_positions, defender_positions = extract_sensor_data(state, state['flag_pos'], state['flag_weight'],
#                                                                  state['agent_params'])
#     node_data, _ = extract_map_sensor_data(state)
#     no_of_nodes = len(node_data)
#     neighbor_data = extract_neighbor_sensor_data(state)
#     agent_name = state.get('name', '')
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
#     # Find this defender's index
#     current_defender_idx = next((i for i, pos in enumerate(defender_positions)
#                                  if pos == current_node and active_defenders[i]), None)
#
#     # Skip if this defender is inactive (shouldn't happen)
#     if current_defender_idx is None:
#         state['action'] = current_node  # Stay in place
#         return state
#
#     # Active attacker positions
#     active_att_positions = [pos for i, pos in enumerate(attacker_positions) if active_attackers[i]]
#
#     # Skip if no active attackers
#     if not active_att_positions:
#         state['action'] = random.choice(neighbor_data)
#         return state
#
#     # Prepare neighbor data
#     neighbor_data_copy = [n for n in neighbor_data if n != current_node]
#     sorted_neighbor = sorted(neighbor_data_copy) + [current_node]
#
#     # Determine if this defender is assigned to primary attacker
#     is_assigned_to_primary = (assigned_defenders_idx is not None and
#                               current_defender_idx in assigned_defenders_idx and
#                               primary_attacker_idx is not None and
#                               active_attackers[primary_attacker_idx])
#
#     if is_assigned_to_primary:
#         # This defender is assigned to primary attacker
#         active_assigned_defenders = [defender_positions[i] for i in assigned_defenders_idx
#                                      if i < len(defender_positions) and active_defenders[i]]
#
#         if len(active_assigned_defenders) >= 2:
#             # Use 2v1 strategy
#             d1 = min(active_assigned_defenders)
#             d2 = max(active_assigned_defenders)
#             defender_index = (d2 * (d2 + 1) // 2) + (d2 - d1)
#             primary_pos = attacker_positions[primary_attacker_idx]
#             joint_state = defender_index * no_of_nodes + primary_pos
#
#             # Choose policy based on position
#             if d1 == d2:
#                 # Both defenders at same position
#                 vector = def1_policy_2v1[joint_state]  # Both use same policy when at same position
#                 print(f'{agent_name} strategies (equal positions)', vector)
#             elif current_node == d1:
#                 # This defender is at min position
#                 vector = def1_policy_2v1[joint_state]
#                 print(f'{agent_name} strategies (min position)', vector)
#             else:
#                 # This defender is at max position
#                 vector = def2_policy_2v1[joint_state]
#                 print(f'{agent_name} strategies (max position)', vector)
#         else:
#             # Only one active assigned defender, use 1v1
#             primary_pos = attacker_positions[primary_attacker_idx]
#             joint_state = current_node * no_of_nodes + primary_pos
#             vector = def_policy_1v1[joint_state]
#             print(f'{agent_name} strategies (1v1 with primary)', vector)
#     else:
#         # This defender targets secondary attackers
#         secondary_attackers = [pos for i, pos in enumerate(attacker_positions)
#                                if active_attackers[i] and i != primary_attacker_idx]
#
#         if secondary_attackers:
#             # Target first secondary attacker
#             target_attacker = secondary_attackers[0]
#             joint_state = current_node * no_of_nodes + target_attacker
#             vector = def_policy_1v1[joint_state]
#             print(f'{agent_name} strategies (1v1 with secondary)', vector)
#         elif active_att_positions:
#             # No secondary attackers, help with primary
#             primary_pos = attacker_positions[primary_attacker_idx]
#             joint_state = current_node * no_of_nodes + primary_pos
#             vector = def_policy_1v1[joint_state]
#             print(f'{agent_name} strategies (helping with primary)', vector)
#         else:
#             # No active attackers (unlikely to reach here)
#             state['action'] = random.choice(neighbor_data)
#             return state
#
#     # Sample action according to probability distribution
#     action_index = np.random.choice(len(vector), p=vector)
#     state['action'] = sorted_neighbor[action_index]
#
#
#
# # The map_strategy function remains unchanged
# def map_strategy(agent_config):
#     strategies = {}
#     for name in agent_config.keys():
#         strategies[name] = strategy
#     return strategies
#    # """
#     # Unified strategy function that handles both defender roles based on the agent's name.
#     # """
#     # # Extract agent name to determine if this is defender 0 or defender 1
#     # agent_name = state.get('name', None)
#     #
#     # # Extract common data needed for both defenders
#     # current_node = state['curr_pos']
#     # flag_positions = state['flag_pos']
#     # flag_weights = state['flag_weight']
#     # agent_params = state['agent_params']
#     # attacker_positions, defender_positions = extract_sensor_data(state, flag_positions, flag_weights, agent_params)
#     # node_data, edge_data = extract_map_sensor_data(state)
#     # no_of_nodes = len(node_data)
#     # neighbor_data = extract_neighbor_sensor_data(state)
#     #
#     # # print(f"Agent name: {agent_name}, Current position: {current_node}")
#     # # print(f"Defender positions: {defender_positions}, Attacker positions: {attacker_positions}")
#     # # print(f"Neighbor data: {neighbor_data}")
#     #
#     # # Calculate the joint state
#     # d1 = min(defender_positions)
#     # d2 = max(defender_positions)
#     # defender_index = (d2 * (d2 + 1) // 2) + (d2 - d1)
#     # joint_state = defender_index * no_of_nodes + attacker_positions[0]
#     #
#     # # Determine if this agent is defender 0 or defender 1
#     # is_defender_0 = agent_name == "defender_0"
#     # is_defender_1 = agent_name == "defender_1"
#     #
#     # # Handle equal defender positions case
#     # if d1 == d2:
#     #     neighbor_data.remove(d1)
#     #     sorted_neighbor = sorted(neighbor_data)
#     #     sorted_neighbor.append(current_node)
#     #     vector = def1_policy_2v1[joint_state]  # Both defenders use the same policy when in same position
#     #     print(f'Defender strategies (equal positions): {vector}')
#     #     action_index = np.random.choice(len(vector), p=vector)
#     #     state['action'] = sorted_neighbor[action_index]
#     #
#     # # Handle defender 0 (assuming defender 0 is always the first in the defender_positions list)
#     # elif is_defender_0:
#     #     this_defender_pos = defender_positions[0]
#     #     neighbor_data.remove(this_defender_pos)
#     #     sorted_neighbor = sorted(neighbor_data)
#     #     sorted_neighbor.append(this_defender_pos)
#     #
#     #     # If defender 0 is at the minimum position
#     #     if this_defender_pos == d1:
#     #         vector = def1_policy_2v1[joint_state]
#     #         print('defender 0 strategies (min position)', vector)
#     #     # If defender 0 is at the maximum position
#     #     else:
#     #         vector = def2_policy_2v1[joint_state]
#     #         print('defender 0 strategies (max position)', vector)
#     #
#     #     action_index = np.random.choice(len(vector), p=vector)
#     #     state['action'] = sorted_neighbor[action_index]
#     #
#     # # Handle defender 1 (assuming defender 1 is always the second in the defender_positions list)
#     # elif is_defender_1:
#     #     this_defender_pos = defender_positions[1]
#     #     neighbor_data.remove(this_defender_pos)
#     #     sorted_neighbor = sorted(neighbor_data)
#     #     sorted_neighbor.append(this_defender_pos)
#     #
#     #     # If defender 1 is at the minimum position
#     #     if this_defender_pos == d1:
#     #         vector = def1_policy_2v1[joint_state]
#     #         print('defender 1 strategies (min position)', vector)
#     #     # If defender 1 is at the maximum position
#     #     else:
#     #         vector = def2_policy_2v1[joint_state]
#     #         print('defender 1 strategies (max position)', vector)
#     #
#     #     action_index = np.random.choice(len(vector), p=vector)
#     #     state['action'] = sorted_neighbor[action_index]
#     #
#     # # Fallback option (shouldn't normally reach here)
#     # else:
#     #     print(f"Warning: Unknown defender name: {agent_name}. Using random move.")
#     #     state['action'] = random.choice(neighbor_data)
#     #
#     # print(f'Defender {agent_name} at position {current_node} moves to {state["action"]}')
#     # return state
#     #
#
# #print(superdefender_policy)
#
# # def strategy_def1(state):
# #     current_node = state['curr_pos']
# #     flag_positions = state['flag_pos']
# #     flag_weights = state['flag_weight']
# #     agent_params = state['agent_params']
# #     attacker_positions, defender_positions = extract_sensor_data(state, flag_positions, flag_weights, agent_params)
# #     node_data, edge_data = extract_map_sensor_data(state)
# #     no_of_nodes = len(node_data)
# #     neighbor_data = extract_neighbor_sensor_data(state)
# #     print("Neighbour data def1",neighbor_data)
# #     print("Attacker positions",attacker_positions)
# #     print("Defender positions",defender_positions)
# #     # joint_state = defender_positions[0]*no_of_nodes*no_of_nodes + defender_positions[1]*no_of_nodes+attacker_positions[0]
# #     d1 = min(defender_positions)
# #     d2 = max(defender_positions)
# #     defender_index = (d2 * (d2 + 1) // 2) + (d2 - d1)
# #     joint_state = defender_index * no_of_nodes + attacker_positions[0]
# #     #neighbor_data.remove(defender_positions[0])
# #     if d1 == d2:
# #         neighbor_data.remove(d1)
# #         sorted_neighbor = sorted(neighbor_data)
# #         sorted_neighbor.append(defender_positions[0])
# #         vector = def1_policy_2v1[joint_state]
# #         print('defender_1 strategies_equal', vector)
# #         action_index = np.random.choice(len(vector), p=vector)
# #         state['action'] = sorted_neighbor[action_index]
# #
# #     elif d1 == defender_positions[0] and d1 != defender_positions[1]:
# #         neighbor_data.remove(d1)
# #         sorted_neighbor = sorted(neighbor_data)
# #         sorted_neighbor.append(defender_positions[0])
# #         vector = def1_policy_2v1[joint_state]
# #         print('defender_1 strategies_min', vector)
# #         action_index = np.random.choice(len(vector), p=vector)
# #         state['action'] = sorted_neighbor[action_index]
# #
# #     else :
# #         neighbor_data.remove(d2)
# #         sorted_neighbor = sorted(neighbor_data)
# #         sorted_neighbor.append(defender_positions[1])
# #         vector = def2_policy_2v1[joint_state]
# #         print('defender_1 strategies_max', vector)
# #         action_index = np.random.choice(len(vector), p=vector)
# #         state['action'] = sorted_neighbor[action_index]
# #     print("def_1_finish")
# #
# #     # neighbor_data = extract_neighbor_sensor_data(state)
# #     # state['action'] = random.choice(neighbor_data)
# #
# # def strategy_def2(state):
# #     current_node = state['curr_pos']
# #     flag_positions = state['flag_pos']
# #     flag_weights = state['flag_weight']
# #     agent_params = state['agent_params']
# #     attacker_positions, defender_positions = extract_sensor_data(state, flag_positions, flag_weights, agent_params)
# #     node_data, edge_data = extract_map_sensor_data(state)
# #     no_of_nodes = len(node_data)
# #     neighbor_data = extract_neighbor_sensor_data(state)
# #     print("Neighbour data defender-2",neighbor_data)
# #     # joint_state = defender_positions[0]*no_of_nodes*no_of_nodes + defender_positions[1]*no_of_nodes+attacker_positions[0]
# #     d1 = min(defender_positions)
# #     d2 = max(defender_positions)
# #     defender_index = (d2 * (d2 + 1) // 2) + (d2 - d1)
# #     joint_state = defender_index * no_of_nodes + attacker_positions[0]
# #     # neighbor_data.remove(defender_positions[1])
# #     if d1 == d2:
# #         neighbor_data.remove(d1)
# #         sorted_neighbor = sorted(neighbor_data)
# #         sorted_neighbor.append(defender_positions[1])
# #         vector = def2_policy_2v1[joint_state]
# #         print('defender_2 strategies_equal', vector)
# #         action_index = np.random.choice(len(vector), p=vector)
# #         state['action'] = sorted_neighbor[action_index]
# #
# #     elif d1 == defender_positions[1] and d1 != defender_positions[0]:
# #         neighbor_data.remove(d1)
# #         sorted_neighbor = sorted(neighbor_data)
# #         sorted_neighbor.append(defender_positions[1])
# #         vector = def1_policy_2v1[joint_state]
# #         print('defender_2 strategies_min', vector)
# #         action_index = np.random.choice(len(vector), p=vector)
# #         state['action'] = sorted_neighbor[action_index]
# #     else:
# #         neighbor_data.remove(d2)
# #         sorted_neighbor = sorted(neighbor_data)
# #         sorted_neighbor.append(defender_positions[1])
# #         vector = def2_policy_2v1[joint_state]
# #         print('defender_2 strategies_max', vector)
# #         action_index = np.random.choice(len(vector), p=vector)
# #         state['action'] = sorted_neighbor[action_index]
# #     print('def_2 finish')