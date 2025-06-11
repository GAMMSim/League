import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
from lib.utils.sensor_utils import extract_sensor_data, extract_neighbor_sensor_data
from lib.utils.graph_utils import compute_x_neighbors
from lib.utils.strategy_utils import compute_node_dominance_region, compute_convex_hull_and_perimeter, compute_shortest_path_step
from lib.core.core import *
from rich.console import Console
from rich.table import Table

console = Console()

def build_advantage_matrix(state, attacker_list, defender_list, graph, target_region):
    """
    Computes the advantage matrix for all defender-attacker pairs.
    Returns:
      - matrix: 2D list [num_defenders][num_attackers]
      - cache: dict[(def_name, att_name)] -> {min_node, adv_dict, perimeter, target_region}
    """
    num_def, num_att = len(defender_list), len(attacker_list)
    matrix = [[float("inf")] * num_att for _ in range(num_def)]
    adv_cache = {}
    perimeter_cache = {}
    min_node_cache = {}

    # initial target region based on first attacker radius
    first_att = attacker_list[0][0]
    history_radius = state["agent_params_dict"][first_att].capture_radius
    hull, perimeter = compute_convex_hull_and_perimeter(graph, target_region, False)

    for d_idx, (def_name, def_pos) in enumerate(defender_list):
        d_speed = state["agent_params_dict"][def_name].speed
        for a_idx, (att_name, att_pos) in enumerate(attacker_list):
            att_params = state["agent_params_dict"][att_name]
            # update region if radius changed
            if att_params.capture_radius != history_radius:
                history_radius = att_params.capture_radius
                hull, perimeter = compute_convex_hull_and_perimeter(graph, target_region, False)

            # compute advantage dict
            _, _, _, adv_dict = compute_node_dominance_region(att_pos, def_pos, att_params.speed, d_speed, graph)
            # find minimal advantage on perimeter
            min_node = min(perimeter, key=lambda n: adv_dict.get(n, float("inf")))
            min_adv = adv_dict.get(min_node, float("inf"))

            # fill the matrix
            matrix[d_idx][a_idx] = min_adv

            # cache exactly as before
            adv_cache[(att_pos, def_pos)] = adv_dict
            # now also stash perimeter and min_node separately
            perimeter_cache[(att_pos, def_pos)] = perimeter
            min_node_cache[(att_pos, def_pos)] = min_node
    return matrix, adv_cache, perimeter_cache, min_node_cache


def assign_pairs(matrix, penalty=10000):
    """
    Hungarian assignment favoring non-negatives, then least-negative.
    Returns dict[def_idx] -> att_idx.
    """
    M = np.array(matrix)
    cost = np.where(M >= 0, 0, penalty - M)
    d, a = M.shape
    if d > a:
        pad = np.full((d, d - a), penalty + 1)
        cost = np.hstack((cost, pad))
    elif a > d:
        pad = np.full((a - d, a), penalty + 1)
        cost = np.vstack((cost, pad))

    row_ind, col_ind = linear_sum_assignment(cost)
    return {r: c for r, c in zip(row_ind, col_ind) if r < d and c < a}


def compute_next_step(state, adv_cache, perimeter_cache, min_node_cache, target_region, defender_idx, assignments, attacker_list, defender_list):
    """
    Determine the next move for this defender.
    Uses cached adv, perimeter, and min_node.
    """
    graph = state["agent_params"].map.graph
    def_name, def_pos = defender_list[defender_idx]
    speed = state["agent_params_dict"][def_name].speed

    # if unassigned ⇒ pure pursuit of the closest attacker
    if defender_idx not in assignments:
        min_dist, closest = min((nx.shortest_path_length(graph, def_pos, a_pos), a_pos) for _, a_pos in attacker_list)
        return compute_shortest_path_step(graph, def_pos, closest, speed)

    # assigned ⇒ look up cached data
    att_idx = assignments[defender_idx]
    att_name, att_pos = attacker_list[att_idx]

    adv = adv_cache[(att_pos, def_pos)]
    perimeter = perimeter_cache[(att_pos, def_pos)]
    node = min_node_cache[(att_pos, def_pos)]
    val = adv.get(node, float("inf"))

    # if defender is winning inside the perimeter (val < 0),
    # steer toward the weakest point in the full target region
    if val < 0:
        region = target_region
        node = min(region, key=lambda n: adv.get(n, float("inf")))
        return compute_shortest_path_step(graph, def_pos, node, speed)
    else:
        # otherwise push against the perimeter
        region = perimeter

    # if already within the chosen region, plan on the subgraph
    if def_pos in region:
        sub = graph.subgraph(region)
        return compute_shortest_path_step(sub, def_pos, node, speed)
    # else, plan on the full graph
    return compute_shortest_path_step(graph, def_pos, node, speed)

def visualize_assignment(assignments, atk_list, def_list):
    """
    Prints a table of defender -> assigned attacker pairs.
    """
    table = Table(title="Assignment")
    table.add_column("Defender", style="red")
    table.add_column("Attacker", style="cyan")
    for def_idx, att_idx in assignments.items():
        def_name, _ = def_list[def_idx]
        att_name, _ = atk_list[att_idx]
        table.add_row(f"D{def_idx}:{def_name}", f"A{att_idx}:{att_name}")
    console.print(table)
    
def render_advantage_matrix(def_list, atk_list, matrix, assignments=None):
    """
    Prints the defender × attacker advantage matrix, and if `assignments`
    is provided (def_idx→att_idx), highlights the assigned cell.
    """
    table = Table(title="Advantage Matrix")
    table.add_column("Def\\Att", style="red")
    for j, (att_name, _) in enumerate(atk_list):
        table.add_column(f"A{j}", justify="right")

    for i, row in enumerate(matrix):
        cells = []
        for j, val in enumerate(row):
            text = "inf" if val == float("inf") else f"{val:.2f}"
            if assignments and assignments.get(i) == j:
                # highlight assigned pair
                text = f"[bold yellow on blue]{text}[/]"
            cells.append(text)
        table.add_row(f"D{i}", *cells)

    console.print(table)
    


def strategy(state):
    ap = state["agent_params"]
    ap_dict = state["agent_params_dict"]
    
    extract_sensor_data(state, state["flag_pos"], state.get("flag_weight"), ap)
    graph = ap.map.graph
    flags = state["flag_pos"]
    params = state["agent_params_dict"]
    atk_list        = list(ap.map.attacker_dict.items())
    atk_radii       = [params[name].capture_radius for name, _ in atk_list]
    region_radius   = max(atk_radii)

    curr_flags = tuple(flags)
    curr_time = state.get("time")
    atk_list = list(ap.map.attacker_dict.items())
    def_list = list(ap.map.defender_dict.items())
    print(atk_list)
    print(def_list)
    last_stamp = ap_dict.get("defender_time_stamp")
    last_flags = ap_dict.get("cached_flags")

    if last_stamp is None or curr_time != last_stamp:
        # only recompute region when flags change
        if last_flags is None or curr_flags != last_flags:
            target_region = compute_x_neighbors(graph, flags, region_radius)
            ap_dict["cached_target_region"] = target_region
            ap_dict["cached_flags"] = curr_flags
        else:
            target_region = ap_dict["cached_target_region"]
        matrix, adv_dict, perimeter_cache, min_node_cache = build_advantage_matrix(state, atk_list, def_list, graph, target_region)
        assignments = assign_pairs(matrix, penalty=10000)
        # Visualize the advantage matrix
        # Visualize the assignment
        # visualize_assignment(assignments, atk_list, def_list)
        # render_advantage_matrix(def_list, atk_list, matrix, assignments)
        
        
        ap_dict["defender_time_stamp"] = curr_time
        ap_dict["cached_time"]           = curr_time
        ap_dict["defender_advantage_matrix"] = matrix
        ap_dict["defender_assignments"]      = assignments
        ap_dict["adv_cache"]                = adv_dict
        ap_dict["perimeter_cache"]          = perimeter_cache
        ap_dict["min_node_cache"]           = min_node_cache
    else:
        target_region   = ap_dict["cached_target_region"]
        matrix          = ap_dict["defender_advantage_matrix"]
        adv_dict        = ap_dict["adv_cache"]
        perimeter_cache = ap_dict["perimeter_cache"]
        min_node_cache  = ap_dict["min_node_cache"]
        assignments     = ap_dict["defender_assignments"]

    # determine this defender's index
    current = state["curr_pos"]
    name_to_idx = {name: i for i,(name,pos) in enumerate(def_list)}
    idx = name_to_idx.get(state["name"])
    if idx is None:
        raise ValueError(f"Defender {state['name']} not found in defender list {def_list}")
    # compute action
    state["action"] = compute_next_step(state, adv_dict, perimeter_cache, min_node_cache, target_region, idx, assignments, atk_list, def_list)
    # state["action"] = current


def map_strategy(agent_config):
    return {name: strategy for name in agent_config}
