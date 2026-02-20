import networkx as nx
import itertools
import random
import math
from typing import Dict, List, Any
from lib.core.apsp_cache import get_apsp_length_cache


# ---------- cost functions ----------
def attacker_stage_cost(pos, defenders, flags, lookup, w_goal=1.0, w_risk=5.0, epsilon=1e-2):
    """
    Immediate cost for an attacker at node `pos`.
    Lower is better for attacker (goal: minimize this).
    cost = w_goal * dist_to_closest_flag + w_risk * sum(1 / (dist_to_def + eps))
    """
    if not flags:
        # no flags known — encourage exploration (small constant)
        return 0.0
    dist_to_flag = min(lookup[pos][f] for f in flags)
    risk_penalty = 0.0
    if defenders:
        risk_penalty = sum(1.0 / (lookup[pos][d] + epsilon) for d in defenders)
    return w_goal * dist_to_flag + w_risk * risk_penalty


def attacker_terminal_cost(pos, defenders, flags, lookup, w_goal=1.0, w_risk=5.0, epsilon=1e-2):
    """
    Terminal evaluation for attacker at node `pos`.
    """
    if not flags:
        return 0.0
    min_flag_dist = min(lookup[pos][f] for f in flags)
    risk_term = 0.0
    if defenders:
        risk_term = sum(1.0 / (lookup[pos][d] + epsilon) for d in defenders)
    return w_goal * min_flag_dist + w_risk * risk_term


# ---------- minimax (1-attacker vs all-defenders) ----------
def minimax_lookahead(attacker_pos, defenders, flags, graph, lookup, depth, is_attacker_turn, beta=1.0):
    """
    Minimax recursion:
      - attacker node: MIN (minimize cost)
      - defenders node: MAX (maximize cost)
    Depth semantics: depth counts defender+attacker turns pairs; callers typically use depth=1
    """
    # base case
    if depth == 0:
        return attacker_terminal_cost(attacker_pos, defenders, flags, lookup)

    # Attacker's turn: choose neighbor (or stay) to minimize
    if is_attacker_turn:
        best_value = float('inf')
        moves = list(graph.neighbors(attacker_pos)) + [attacker_pos]
        for npos in moves:
            stage = attacker_stage_cost(npos, defenders, flags, lookup)
            future = minimax_lookahead(npos, defenders, flags, graph, lookup, depth, False, beta)
            value = stage + beta * future
            if value < best_value:
                best_value = value
        return best_value

    # Defender's turn: choose joint defender move that is worst for attacker (maximize)
    else:
        worst_value = float('-inf')
        # build move options for each defender
        defender_move_options = []
        for dpos in defenders:
            defender_move_options.append(list(graph.neighbors(dpos)) + [dpos])

        # cartesian product of defender moves
        for joint in itertools.product(*defender_move_options):
            next_defs = list(joint)
            # after defenders move, it's attacker's turn; decrement depth by 1 (a full cycle complete)
            value = minimax_lookahead(attacker_pos, next_defs, flags, graph, lookup, depth - 1, True, beta)
            if value > worst_value:
                worst_value = value
        return worst_value


# ---------- main policy ----------
def strategy(state):
    """
    Updated attacker strategy:
    - initializes / shares candidate flag set in team_cache
    - identifies real flags when within sensing region and prunes fake candidates
    - tracks defenders using egocentric and stationary sensors
    - runs 1-attacker vs all-defenders minimax (depth=1) using lookup table
    - random tie-break among equally good actions
    - returns set of detected real flags
    """
    # --- agent/environment ---
    agent_ctrl = state["agent_controller"]
    current_pos: int = state["curr_pos"]
    current_time: int = state["time"]
    team: str = agent_ctrl.team  # 'red' or 'blue' (should be 'red' for attackers)

    cache = agent_ctrl.cache
    team_cache = agent_ctrl.team_cache

    # --- sensors & map ---
    sensors = state.get("sensor", {})
    # graph: try to get it from sensors similar to defender or from agent_ctrl.map (fallback)
    graph = None
    global_map_apsp = None
    if "global_map" in sensors:
        try:
            global_map_payload = sensors["global_map"][1]
            graph = global_map_payload["graph"]
            global_map_apsp = global_map_payload.get("apsp")
        except Exception:
            graph = agent_ctrl.map.graph
    else:
        graph = agent_ctrl.map.graph

    agent_map = agent_ctrl.map
    # attach graph + APSP once (safe)
    if agent_map.graph is None and graph is not None:
        try:
            nodes_data = {n: graph.nodes[n] for n in graph.nodes()}
            edges_data = {}
            for idx, (u, v, data) in enumerate(graph.edges(data=True)):
                edges_data[idx] = {"source": u, "target": v, **data}
            agent_map.attach_networkx_graph(
                nodes_data,
                edges_data,
                apsp_lookup=global_map_apsp if isinstance(global_map_apsp, dict) else None,
            )
        except Exception:
            # if graph is None or malformed, default to no-op
            pass
    if agent_map.graph is not None and agent_map.apsp_lookup is None:
        if isinstance(global_map_apsp, dict):
            agent_map.set_apsp_lookup(global_map_apsp)
        else:
            agent_map.set_apsp_lookup(get_apsp_length_cache(agent_map.graph))

    if graph is None:
        graph = agent_map.graph

    # --- initialize / load candidate flags into team cache ---
    # candidate_flag sensor contains both real + fake initially
    if "candidate_flag" in sensors:
        cand_list = sensors["candidate_flag"][1].get("candidate_flags", [])
        # store as set in team cache for shared access
        team_cache.set("candidate_flags", set(cand_list))
    else:
        # if not present, ensure a key exists
        if team_cache.get("candidate_flags", None) is None:
            team_cache.set("candidate_flags", set())

    candidate_flags = set(team_cache.get("candidate_flags", set()))

    # --- detect real flags when within sensing region, and remove fake candidates ---
    # If agent has egocentric_flag sensor, it reports real flags within sensing radius.
    detected_real_flags = set()
    if "egocentric_flag" in sensors:
        # expected structure: sensors["egocentric_flag"][1]["detected_flags"] -> List[int]
        detected = sensors["egocentric_flag"][1].get("detected_flags", [])
        detected_real_flags.update(detected)

        # Now, for any candidate that is inside the agent's sensing neighborhood but not in detected,
        # we can treat it as fake and remove from candidate set.
        # If egocentric_map exists we can check which nodes are within sensing radius.
        sensing_neighborhood = set()
        if "egocentric_map" in sensors:
            em_nodes = sensors["egocentric_map"][1].get("nodes", {})
            # em_nodes is expected to be dict of node_id->data; take keys as visible nodes
            sensing_neighborhood.update(em_nodes.keys())
        else:
            # fallback: we only know the agent's own node: if current_pos is candidate and not detected -> fake
            sensing_neighborhood.add(current_pos)

        # For each candidate in sensing neighborhood, if it's not in detected_real_flags => fake => drop it.
        to_remove = set()
        for c in candidate_flags:
            if c in sensing_neighborhood and c not in detected_real_flags:
                to_remove.add(c)
        if to_remove:
            candidate_flags.difference_update(to_remove)
            team_cache.set("candidate_flags", candidate_flags)

    # If we see at least one real flag, update the shared candidate flags to only contain real flags
    if detected_real_flags:
        # It's reasonable to narrow the candidate set to only known real flags (others are irrelevant)
        candidate_flags = set(detected_real_flags)
        team_cache.set("candidate_flags", candidate_flags)

    # --- now we have candidate_flags (shared) and detected_real_flags (locally sensed real ones) ---
    # for planning, use the candidate_flags set (if empty, default to fallback behavior)
    flags_for_planning = list(candidate_flags) if candidate_flags else []

    # --- track defender positions from sensors ---
    defender_positions = []
    # from egocentric_agent sensor (gives visible agents)
    if "egocentric_agent" in sensors:
        for name, pos in sensors["egocentric_agent"][1].items():
            # defender team should be the opposite of attacker team
            enemy_team = "blue" if team == "red" else "red"
            if name.startswith(enemy_team):
                defender_positions.append(pos)
    # from global or 'agent' sensor (if available)
    if "agent" in sensors:
        all_agents = sensors["agent"][1]
        enemy_team = "blue" if team == "red" else "red"
        for name, pos in all_agents.items():
            if name.startswith(enemy_team):
                # avoid duplicates
                if pos not in defender_positions:
                    defender_positions.append(pos)
    # from stationary sensors
    for s in sensors:
        if s.startswith("stationary_"):
            det = sensors[s][1].get("detected_agents", {})
            for name, info in det.items():
                enemy_team = "blue" if team == "red" else "red"
                if name.startswith(enemy_team):
                    node_id = info.get("node_id", None)
                    if node_id is not None and node_id not in defender_positions:
                        defender_positions.append(node_id)

    # --- fallback behavior: no flags known (move to explore) ---
    if not flags_for_planning:
        # simple: move to a random neighbor to explore (or stay)
        neighbors = list(graph.neighbors(current_pos)) if graph is not None else []
        if neighbors:
            chosen = random.choice(neighbors + [current_pos])
            state["action"] = chosen
        else:
            state["action"] = current_pos
        # return any detected real flags
        return set(detected_real_flags)

    # --- load lookup table for fast distances ---
    lookup = (
        agent_map.apsp_lookup
        if isinstance(agent_map.apsp_lookup, dict)
        else global_map_apsp if isinstance(global_map_apsp, dict)
        else get_apsp_length_cache(graph)
    )

    # --- prepare minimax search (depth=1) ---
    depth = 1
    beta = 1.0

    # candidate defender list for minimax: if no defenders observed, assume none (empty list)
    defenders_for_planning = defender_positions.copy()

    # Evaluate possible attacker moves (neighbors + stay)
    attacker_moves = list(graph.neighbors(current_pos)) + [current_pos]

    best_value = float('inf')
    best_moves = []

    # Precompute attacker move options for defenders: each defender can move to neighbors + stay
    # (used inside minimax for defender response)
    def get_defender_move_options(def_positions):
        opts = []
        for d in def_positions:
            opts.append(list(graph.neighbors(d)) + [d])
        return opts

    defender_move_options = get_defender_move_options(defenders_for_planning)

    # Evaluate each attacker candidate move
    for a_move in attacker_moves:
        sc = attacker_stage_cost(a_move, defenders_for_planning, flags_for_planning, lookup)
        # attackers assume defenders will choose moves to maximize the cost (worst case for attacker)
        max_value = -float('inf')
        # defender joint moves product
        if defender_move_options:
            for joint in itertools.product(*defender_move_options):
                next_defs = list(joint)
                tc = attacker_terminal_cost(a_move, next_defs, flags_for_planning, lookup)
                value = sc + beta * tc
                if value > max_value:
                    max_value = value
        else:
            # no defenders observed -> terminal cost only depends on attacker
            tc = attacker_terminal_cost(a_move, [], flags_for_planning, lookup)
            max_value = sc + beta * tc

        # attacker wants to minimize max_value
        if max_value < best_value:
            best_value = max_value
            best_moves = [a_move]
        elif math.isclose(max_value, best_value, rel_tol=1e-9):
            best_moves.append(a_move)

    # choose randomly among best moves
    chosen_action = random.choice(best_moves) if best_moves else current_pos
    state["action"] = chosen_action

    # return detected real flags (for reward counting in env)
    return set(detected_real_flags)


def map_strategy(agent_config):
    return {name: strategy for name in agent_config.keys()}
