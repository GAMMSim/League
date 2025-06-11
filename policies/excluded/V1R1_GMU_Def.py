import random
import networkx as nx
from lib.game_utils import extract_sensor_data, extract_neighbor_sensor_data
from lib.graph_utils import compute_node_dominance_region, compute_convex_hull_and_perimeter, compute_x_neighbors, compute_shortest_path_step
import numpy as np
from scipy.optimize import linear_sum_assignment


def strategy(state):
    """
    Defines the defender's strategy to move towards the closest attacker based on a maximum matching.
    Intermediate results are computed once and cached for later reuse.

    Parameters:
        state (dict): The current state of the game, including positions and parameters.
    """
    
    # --- Retrieve state parameters ---
    current_node = state["curr_pos"]
    flag_positions = state["flag_pos"]
    flag_weights = state["flag_weight"]
    agent_params = state["agent_params"]
    agent_params_dict = state["agent_params_dict"]
    # Next Position
    next_node = current_node
        
    try:
        # --- Extract sensor data and agent positions ---
        _, _ = extract_sensor_data(state, flag_positions, flag_weights, agent_params)
        attacker_dict = agent_params.map.attacker_dict
        defender_dict = agent_params.map.defender_dict

        num_of_defenders = len(defender_dict)
        num_of_attackers = len(attacker_dict)

        # --- Prepare lists and initialize maximum matching matrix ---
        defender_list = list(defender_dict.items())  # [(defender_name, defender_position), ...]
        attacker_list = list(attacker_dict.items())  # [(attacker_name, attacker_position), ...]
        maximum_matching_matrix = [[0 for _ in range(num_of_attackers)] for _ in range(num_of_defenders)]

        # Cache for storing intermediate results keyed by (defender_name, attacker_name)
        computed_results = {}
        
        # Fast computation of target region
        attacker_capture_radius_history = agent_params_dict.get(attacker_list[0][0], {}).capture_radius
        target_region = compute_x_neighbors(agent_params.map.graph, flag_positions, attacker_capture_radius_history)
        H, P = compute_convex_hull_and_perimeter(agent_params.map.graph, target_region, False)

        # --- Build the maximum matching matrix with caching ---
        for d_idx, (defender_name, defender_position) in enumerate(defender_list):
            for a_idx, (attacker_name, attacker_position) in enumerate(attacker_list):
                try:
                    # Compute dominance regions and advantage dictionary.
                    attacker_DR, defender_DR, contested, advantage_dict = compute_node_dominance_region(
                        attacker_position, defender_position, agent_params_dict.get(defender_name, {}).speed, agent_params_dict.get(attacker_name, {}).speed, agent_params.map.graph
                    )
                    if agent_params_dict.get(attacker_name, {}).speed != attacker_capture_radius_history:
                        target_region = compute_x_neighbors(agent_params.map.graph, flag_positions, agent_params_dict.get(attacker_name, {}).capture_radius)
                        H, P = compute_convex_hull_and_perimeter(agent_params.map.graph, target_region, False)
                        attacker_capture_radius_history = agent_params_dict.get(attacker_name, {}).capture_radius

                    # Determine the node in P with the smallest advantage.
                    min_node = min(P, key=lambda node: advantage_dict[node], default=None)
                    min_advantage = advantage_dict[min_node] if min_node is not None else float("inf")

                    # Record the computed advantage in the matrix.
                    maximum_matching_matrix[d_idx][a_idx] = min_advantage

                    # Cache all computed intermediate results for later reuse.
                    computed_results[(defender_name, attacker_name)] = {
                        "attacker_DR": attacker_DR,
                        "defender_DR": defender_DR,
                        "contested": contested,
                        "advantage_dict": advantage_dict,
                        "target_region": target_region,
                        "H": H,
                        "P": P,
                        "min_node": min_node,
                        "min_advantage": min_advantage,
                    }
                except Exception as e:
                    raise Exception(f"Error computing dominance region for attacker {attacker_name} and defender {defender_name}: {e}")

        print("Maximum matching matrix:")
        for row in maximum_matching_matrix:
            print(row)

        # --- Compute defender-to-attacker assignments using cost-based matching ---
        penalty = agent_params.map.graph.number_of_nodes()
        assignments = assign_defenders(maximum_matching_matrix, penalty=penalty)

        # --- Find the index corresponding to the current defender ---
        current_defender_idx = None
        for idx, (defender_name, defender_position) in enumerate(defender_list):
            if defender_position == current_node:
                current_defender_idx = idx
                break

        if current_defender_idx is None:
            raise Exception("Current defender not found in defender list.")

        current_defender_name, _ = defender_list[current_defender_idx]
        defender_speed = agent_params_dict.get(current_defender_name, {}).speed

        # --- Determine the final action based on assignment ---
        if current_defender_idx in assignments:
            assigned_attacker_idx = assignments[current_defender_idx]
            assigned_attacker_name, assigned_attacker_position = attacker_list[assigned_attacker_idx]
            # print(f"Defender {current_defender_name} assigned to attacker {assigned_attacker_name}.")

            # If the current defender's position has not changed, try to use cached data.
            cached = computed_results.get((current_defender_name, assigned_attacker_name))
            if cached:
                min_node = cached["min_node"]
                min_advantage = cached["min_advantage"]
                advantage_dict = cached["advantage_dict"]
                P = cached["P"]
            else:
                # Recompute if the cached result is not available.
                attacker_DR, defender_DR, contested, advantage_dict = compute_node_dominance_region(assigned_attacker_position, current_node, defender_speed, agent_params_dict.get(assigned_attacker_name, {}).get("speed"), agent_params.map.graph)
                target_region = compute_x_neighbors(agent_params.map.graph, flag_positions, agent_params_dict.get(assigned_attacker_name, {}).get("capture_radius"))
                H, P = compute_convex_hull_and_perimeter(agent_params.map.graph, target_region, False)
                min_node = min(P, key=lambda node: advantage_dict[node], default=None)
                min_advantage = advantage_dict[min_node] if min_node is not None else float("inf")

            # --- Compute the next step toward min_node along a path that stays within P ---
            if min_node is None:
                raise Exception("Defender have a valid assignment but no target node found on the perimeter.")
            else:
                if current_node in P:
                    # Restrict movement to P by working on the subgraph induced by P.
                    subgraph = agent_params.map.graph.subgraph(P)
                    next_node = compute_shortest_path_step(subgraph, current_node, min_node, step=defender_speed)
                else:
                    next_node = compute_shortest_path_step(agent_params.map.graph, current_node, min_node, step=defender_speed)

            if next_node is None:
                raise Exception("No valid path found to the target node.")
        else:
            # Go to the closest attacker if no assignment is found.
            min_distance = float("inf")
            closest_attacker = None
            for attacker_name, attacker_position in attacker_list:
                distance = nx.shortest_path_length(agent_params.map.graph, current_node, attacker_position)
                if distance < min_distance:
                    min_distance = distance
                    closest_attacker = attacker_position
            if closest_attacker is not None:
                next_node = compute_shortest_path_step(agent_params.map.graph, current_node, closest_attacker, step=defender_speed)
        state["action"] = next_node
    except Exception as e:
        print(f"Error executing defender strategy for defender {current_defender_name}: {e}")
        next_node = current_node
        state["action"] = next_node


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


def compute_assignment(cost_matrix):
    """
    Given a cost matrix, compute the minimum cost assignment using the Hungarian algorithm.

    Args:
        cost_matrix (np.ndarray): A 2D array where each entry is a non-negative cost.

    Returns:
        row_ind, col_ind: Indices for the optimal assignment.
    """
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind


def assign_defenders(maximum_matching_matrix, penalty=10000):
    """
    Uses a cost-based matching approach to assign defenders to attackers.

    The cost is defined as:
      - 0 if the entry in maximum_matching_matrix is >= 0 (good assignment).
      - (penalty + (-entry)) if the entry is negative, so that even if negative assignments are forced,
        the one with the least negative value is chosen.

    This ensures that if a non-negative assignment exists, it will be chosen rather than any negative assignment.

    Args:
        maximum_matching_matrix (list of lists or np.ndarray): Matrix with defender rows and attacker columns.
        penalty (int, optional): A large constant used to penalize negative entries. Defaults to 10000.

    Returns:
        dict: Mapping of defender index to attacker index for valid assignments.
    """
    # Convert the input matrix to a NumPy array.
    M = np.array(maximum_matching_matrix)

    # Create the cost matrix:
    # For non-negative entries, cost is 0.
    # For negative entries, cost is penalty + (-value).
    cost_matrix = np.where(M >= 0, 0, penalty - M)  # -M gives the absolute value for negatives
    # print("Cost matrix after transformation:")
    # print(cost_matrix)

    # Get the original dimensions.
    num_defenders, num_attackers = M.shape

    # Pad the matrix to square if necessary.
    if num_defenders > num_attackers:
        pad_width = num_defenders - num_attackers
        cost_matrix = np.hstack([cost_matrix, np.ones((num_defenders, pad_width)) * (penalty + 1)])
    elif num_attackers > num_defenders:
        pad_height = num_attackers - num_defenders
        cost_matrix = np.vstack([cost_matrix, np.ones((pad_height, cost_matrix.shape[1])) * (penalty + 1)])

    # Optionally, you can print the cost matrix for debugging:
    # print("Cost matrix:\n", cost_matrix)

    # Compute the assignment.
    row_ind, col_ind = compute_assignment(cost_matrix)
    # print("Row indices:", row_ind)
    # print("Column indices:", col_ind)
    # # Filter out the padded assignments.
    # # Only keep assignments where the row index is less than the number of defenders
    # # and the column index is less than the number of attackers.
    # # This ensures we only consider valid assignments.
    # # Print the filtered assignments for debugging.
    # print("Filtered assignments:")
    # print(list(zip(row_ind, col_ind)))

    # Build the assignment dictionary for the original defenders and attackers.
    assignments = {}
    for r, c in zip(row_ind, col_ind):
        # Only record assignments where the column is within the range of real attackers.
        if r < num_defenders and c < num_attackers:
            assignments[r] = c
    return assignments
