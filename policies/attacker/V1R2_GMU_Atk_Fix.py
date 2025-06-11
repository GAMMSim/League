import networkx as nx
import numpy as np
from itertools import combinations
from lib.utils.sensor_utils import extract_sensor_data
from lib.utils.graph_utils import compute_x_neighbors
from lib.utils.strategy_utils import compute_node_dominance_region, compute_shortest_path_step
from rich.console import Console
from rich.table import Table
from lib.core.core import *

console = Console()


def build_win_matrix_and_cache(state, graph, target_region):
    atk_list = list(state["agent_params"].map.attacker_dict.items())
    def_list = list(state["agent_params"].map.defender_dict.items())
    params = state["agent_params_dict"]

    nA, nD = len(atk_list), len(def_list)
    win = [[False] * nD for _ in range(nA)]
    adv_cache = {}

    for i, (atk_name, atk_pos) in enumerate(atk_list):
        a_speed = params[atk_name].speed
        # target_region same for all defenders of this attacker
        for j, (def_name, def_pos) in enumerate(def_list):
            d_speed = params[def_name].speed
            try:
                _, _, _, adv = compute_node_dominance_region(atk_pos, def_pos, a_speed, d_speed, graph)
            except Exception as err:
                error(f"Error computing dominance region: {err}")
                adv = {}
            can_win = any(adv.get(node, np.inf) < 0 for node in target_region)
            win[i][j] = can_win
            adv_cache[(atk_pos, def_pos)] = adv
    return win, adv_cache, atk_list, def_list


def constellate_partitions(elems):
    if not elems:
        return [[]]
    first, rest = elems[0], elems[1:]
    parts = []
    for p in constellate_partitions(rest):
        parts.append([[first]] + [grp.copy() for grp in p])
        for idx in range(len(p)):
            newp = [grp.copy() for grp in p]
            newp[idx].append(first)
            parts.append(newp)
    # dedupe
    unique = []
    for p in parts:
        key = tuple(sorted(tuple(sorted(g)) for g in p))
        if key not in unique:
            unique.append(key)
    return [[list(g) for g in part] for part in unique]


def brute_force_attacker_grouping(win):
    nA = len(win)
    nD = len(win[0]) if nA else 0
    # uniform rows: all-win or all-lose
    uniform = []
    mixed = []
    for i, row in enumerate(win):
        if all(row) or not any(row):
            C = [j for j in range(nD) if row[j]]
            uniform.append(([i], C, 1 - len(C)))
        else:
            mixed.append(i)
    # brute-force mixed partitions
    mixed_groups = []
    if mixed:
        best_score = -np.inf
        best_part = []
        for part in constellate_partitions(mixed):
            total = 0
            info = []
            for S in part:
                C = [j for j in range(nD) if all(win[i][j] for i in S)]
                score = len(S) - len(C)
                total += score
                info.append((S, C, score))
            if total > best_score:
                best_score = total
                best_part = info
        mixed_groups = best_part
    return uniform + mixed_groups


def render_win_matrix(atk_list, def_list, win):
    table = Table(title="Win Matrix")
    table.add_column("Att\\Def", style="cyan")
    for j in range(len(def_list)):
        table.add_column(f"D{j}", justify="center")
    for i, row in enumerate(win):
        table.add_row(f"A{i}", *["✓" if c else "✗" for c in row])
    console.print(table)


def visualize_grouping(atk_list, def_list, grouping):
    table = Table(title="Grouping")
    table.add_column("#", style="bold green")
    table.add_column("Attackers", style="magenta")
    table.add_column("Defenders", style="red")
    table.add_column("Score", justify="right")
    for idx, (S, C, sc) in enumerate(grouping, start=1):
        table.add_row(str(idx), ",".join(map(str, S)), ",".join(map(str, C)), str(sc))
    console.print(table)


def compute_attacker_next_step(
    state, attacker_idx: int, grouping: list[tuple[list[int], list[int], float]], adv_cache: dict[tuple, dict[int, float]], target_region: list[int], attacker_list: list[tuple[str, int]], defender_list: list[tuple[str, int]]
):
    """
    For attacker #attacker_idx, find all nodes where *all* its assigned defenders
    lose (adv<0), then pick the safe node whose graph‐distance to any node in
    target_region is minimal, and return the next step toward it.
    """
    graph = state["agent_params"].map.graph
    curr_pos = state["curr_pos"]
    speed = state["agent_params_dict"][attacker_list[attacker_idx][0]].speed

    # 1) find which defenders this attacker cares about
    #    grouping entries are (S, C, score)
    S, C, _ = next((S, C, sc) for S, C, sc in grouping if attacker_idx in S)

    # 2) gather each defender's adv_dict for this attacker
    att_pos = attacker_list[attacker_idx][1]
    adv_dicts = [adv_cache[(att_pos, defender_list[d][1])] for d in C]

    # 3) safe nodes = those where every adv_dict[n] < 0
    safe_nodes = [n for n in graph.nodes() if all(d.get(n, np.inf) < 0 for d in adv_dicts)]
    if not safe_nodes:
        return compute_shortest_path_step(graph, curr_pos, target_region, speed)

    # 4) among safe_nodes, find the one minimizing distance to target_region
    best_node, best_dist = None, float("inf")
    for n in safe_nodes:
        dist = min(nx.shortest_path_length(graph, n, t) for t in target_region)
        if dist < best_dist:
            best_node, best_dist = n, dist

    # 5) step toward that best_node
    return compute_shortest_path_step(graph, curr_pos, best_node, speed)


def strategy(state):
    ap = state["agent_params"]
    ap_dict = state["agent_params_dict"]
    # ensure sensors build map
    extract_sensor_data(state, state["flag_pos"], state.get("flag_weight"), ap)
    graph = ap.map.graph
    flags = state["flag_pos"]
    params = state["agent_params_dict"]

    # pick an “attacker speed” to use for target‐region
    atk_list = list(ap.map.attacker_dict.items())
    radius = [params[name].capture_radius for name, _ in atk_list]
    a_radius = max(radius)

    # check if grouping needs recompute based on flags, positions, or time
    curr_flags = tuple(flags)
    curr_time = state.get("time")
    last_stamp = ap_dict.get("attacker_time_stamp")
    last_flags = ap_dict.get("cached_flags")

    # outer check: do we need to rebuild grouping at all?
    if last_stamp is None or curr_time != last_stamp:
        # print(f"Attacker {state['name']} updating grouping")
        if last_flags is None or curr_flags != last_flags:
            target_region = compute_x_neighbors(graph, flags, a_radius)
            ap_dict["cached_target_region"] = target_region
            ap_dict["cached_flags"] = curr_flags
        else:
            # reuse the old region
            target_region = ap_dict["cached_target_region"]

        win, adv_cache, atk_list, def_list = build_win_matrix_and_cache(state, graph, target_region)
        grouping = brute_force_attacker_grouping(win)

        # render_win_matrix(atk_list, def_list, win)
        # visualize_grouping(atk_list, def_list, grouping)
        ap_dict["attacker_time_stamp"] = curr_time
        ap_dict["cached_time"] = curr_time
        ap_dict["attacker_win_matrix"] = win
        ap_dict["attacker_grouping"] = grouping
        ap_dict["adv_cache"] = adv_cache
    else:
        win = ap_dict["attacker_win_matrix"]
        grouping = ap_dict["attacker_grouping"]
        adv_cache = ap_dict["adv_cache"]
        atk_list = list(ap.map.attacker_dict.items())
        def_list = list(ap.map.defender_dict.items())
        target_region = ap_dict["cached_target_region"]

    # determine this agent's action (placeholder)
    current = state["curr_pos"]
    # no action modification yet
    # state["action"] = current
    my_name = state["name"]
    attacker_idx = next(i for i, (nm, pos) in enumerate(atk_list) if nm == my_name)

    # 2) call the helper
    state["action"] = compute_attacker_next_step(state, attacker_idx=attacker_idx, grouping=grouping, adv_cache=adv_cache, target_region=target_region, attacker_list=atk_list, defender_list=def_list)


def map_strategy(agent_config):
    return {name: strategy for name in agent_config}
