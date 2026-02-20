import networkx as nx
import itertools
import random
import math
from typing import Dict, List, Any
from lib.core.apsp_cache import get_apsp_length_cache, get_cached_distance


# ---- Helper: compute interception targets (prefix nodes of shortest paths) ----
def compute_interception_nodes(graph, attackers: List[int], flags: List[int], lookup, prefix_len: int = 3, capture_R: int = 2):
    """
    For each attacker -> each flag shortest path, take the first `prefix_len` nodes
    (excluding the attacker's current node if desired) as interception candidates.

    Also include shell nodes at distance capture_R+1 (e.g., 3) from each flag.
    Returns a set of node IDs representing good interception positions.
    """
    intercept_nodes = set()
    # shell distance (just outside capture radius)
    shell_dist = capture_R + 1

    # add shell nodes for each flag
    for f in flags:
        for n in graph.nodes():
            dfn = get_cached_distance(lookup, n, f)
            if dfn is not None and dfn == shell_dist:
                intercept_nodes.add(n)

    # path prefixes
    for a in attackers:
        for f in flags:
            try:
                path_af = nx.shortest_path(graph, a, f)
                # we want nodes that attacker must go through early: take prefix after attacker
                # skip the first element (attacker's current node) so prefix nodes are later hops
                for node in path_af[1:1 + prefix_len]:
                    intercept_nodes.add(node)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

    return intercept_nodes


# ---- Stage & Terminal cost revised to favor interception and avoid sitting on flag ----
def stage_cost_intercept(pos: int, attackers: List[int], flags: List[int], lookup: Dict[int, Dict[int, int]],
                         intercept_nodes: set, capture_R: int = 2,
                         w_intercept=4.0, w_flag_guard=1.0, w_penalty_inside=100.0, epsilon=1e-6):
    """
    Stage cost for defender at pos:
     - large penalty if defender sits within capture radius of ANY real flag (unsafe)
     - reward (negative cost) if pos is an interception node (shell / path)
     - cost component to move towards closest threatening attacker relative to flags
    Lower is better for defender.
    """
    if not flags:
        return 0.0

    cost = 0.0

    # if defender sits too close to a flag (within capture radius), heavy penalty
    for f in flags:
        if lookup[pos][f] <= capture_R:
            cost += w_penalty_inside

    # reward for being on an interception node (preferable)
    if pos in intercept_nodes:
        cost -= w_intercept  # negative reduces cost: good thing

    # guard score: being closer to flags is good but not if inside capture radius
    flag_guard_score = sum(1.0 / (lookup[pos][f] + epsilon) for f in flags)
    cost -= w_flag_guard * flag_guard_score

    # threat proximity: prefer to be closer to the most threatening attacker
    if attackers:
        # threat defined as attacker closest to any flag
        threat = min(attackers, key=lambda a: min(lookup[a][f] for f in flags))
        dist_to_threat = lookup[pos][threat]
        cost += 0.5 * dist_to_threat  # closer is better (lower cost), so adding positive cost penalizes distance

    return cost


def terminal_cost_intercept(pos: int, attackers: List[int], defenders: List[int], flags: List[int],
                            lookup: Dict[int, Dict[int, int]], intercept_nodes: set, capture_R: int = 2,
                            mu=5.0):
    """
    Terminal evaluation: sum over attackers of (closest defender dist - attacker dist) to flags,
    with additional bonus if defenders are on intercept nodes.
    Lower is better (defender perspective).
    """
    if not attackers or not defenders or not flags:
        return 0.0

    cost = 0.0
    for a in attackers:
        ad = min(lookup[a][f] for f in flags)
        dd = min(lookup[d][f] for d in defenders for f in flags)
        cost += (dd - ad)

    # reward defenders who occupy intercept nodes (reduce cost)
    for d in defenders:
        if d in intercept_nodes:
            cost -= mu

    return cost


# ---- Main strategy function ----
def strategy(state):
    """
    Defender strategy combining shortest-path interception + minimax planning.
    """
    # local imports & types
    import math
    import itertools

    agent_ctrl = state["agent_controller"]
    current_pos: int = state["curr_pos"]
    current_time: int = state["time"]
    team: str = agent_ctrl.team

    cache = agent_ctrl.cache
    team_cache = agent_ctrl.team_cache

    # map & graph
    agent_map = agent_ctrl.map
    # get graph from sensor/global_map as in example_def
    global_map_sensor = None
    global_map_apsp = None
    try:
        global_map_payload = state["sensor"]["global_map"][1]
        global_map_sensor = global_map_payload["graph"]
        global_map_apsp = global_map_payload.get("apsp")
    except Exception:
        global_map_sensor = agent_ctrl.map.graph

    # attach networkx graph + APSP once
    if agent_map.graph is None and global_map_sensor is not None:
        nodes_data = {n: global_map_sensor.nodes[n] for n in global_map_sensor.nodes()}
        edges_data = {}
        for idx, (u, v, data) in enumerate(global_map_sensor.edges(data=True)):
            edges_data[idx] = {"source": u, "target": v, **data}
        agent_map.attach_networkx_graph(
            nodes_data,
            edges_data,
            apsp_lookup=global_map_apsp if isinstance(global_map_apsp, dict) else None,
        )
    if agent_map.graph is not None and agent_map.apsp_lookup is None:
        if isinstance(global_map_apsp, dict):
            agent_map.set_apsp_lookup(global_map_apsp)
        else:
            agent_map.set_apsp_lookup(get_apsp_length_cache(agent_map.graph))
    graph = agent_map.graph if agent_map.graph is not None else global_map_sensor

    agent_map.update_agent_position(team, agent_ctrl.name, current_pos, current_time)

    # read flags from sensors
    sensors = state["sensor"]
    real_flags: List[int] = []
    if "flag" in sensors:
        real_flags = sensors["flag"][1]["real_flags"]

    # collect visible attackers from egocentric and stationary sensors
    visible_attackers = {}
    if "egocentric_agent" in sensors:
        for name, pos in sensors["egocentric_agent"][1].items():
            if name.startswith("red"):
                visible_attackers[name] = pos

    for s in sensors:
        if s.startswith("stationary_"):
            det = sensors[s][1].get("detected_agents", {})
            for name, info in det.items():
                if name.startswith("red"):
                    visible_attackers[name] = info.get("node_id", info.get("node", None))

    # update seen attackers in cache
    seen_attackers = cache.get("seen_attackers", {})
    for name, pos in visible_attackers.items():
        seen_attackers[name] = pos
    cache.set("seen_attackers", seen_attackers)
    attacker_positions = list(seen_attackers.values())

    # if no real flags known, fallback: move to shell of candidate flags if available in team_cache
    if not real_flags:
        # try to use candidate flags from team_cache (if defenders may know)
        cand_flags = team_cache.get("candidate_flags", [])
        if cand_flags:
            # choose nearest candidate's shell node
            lookup_tmp = (
                agent_map.apsp_lookup
                if isinstance(agent_map.apsp_lookup, dict)
                else global_map_apsp if isinstance(global_map_apsp, dict)
                else get_apsp_length_cache(graph)
            )
            # pick flag whose shell is nearest
            best_target = None
            best_dist = float('inf')
            for f in cand_flags:
                # find nodes at distance 3 (shell) and pick nearest
                for n in graph.nodes():
                    dfn = lookup_tmp.get(n, {}).get(f, None)
                    if dfn is None:
                        continue
                    if dfn == 3 and lookup_tmp[current_pos][n] < best_dist:
                        best_dist = lookup_tmp[current_pos][n]
                        best_target = n
            if best_target is not None:
                nxt = agent_map.shortest_path_step(current_pos, best_target, agent_ctrl.speed)
                state['action'] = nxt
                return set()
        # else just stay
        state['action'] = current_pos
        return set()

    # load fast lookup
    lookup = (
        agent_map.apsp_lookup
        if isinstance(agent_map.apsp_lookup, dict)
        else global_map_apsp if isinstance(global_map_apsp, dict)
        else get_apsp_length_cache(graph)
    )

    # build interception set using shortest path prefixes and shell nodes
    intercept_nodes = compute_interception_nodes(graph, attacker_positions, real_flags, lookup, prefix_len=3, capture_R=2)

    # minimax (1-defender vs all attackers) with depth=1
    depth = 1
    beta = 1.0

    # possible defender moves (neighbors + stay)
    defender_moves = list(graph.neighbors(current_pos)) + [current_pos]

    # prepare attacker move options (neighbors + stay)
    attacker_move_sets = []
    for atk in attacker_positions:
        # if attacker node not in graph (shouldn't happen) skip
        if atk not in graph:
            attacker_move_sets.append([atk])
        else:
            attacker_move_sets.append(list(graph.neighbors(atk)) + [atk])

    best_score = float('inf')
    best_moves = []

    # find index of current defender in defender list (we treat single defender, others remain same)
    # get current defenders positions from agent_map (teams)
    try:
        defenders_list = [pos for (_, pos, _) in agent_map.get_team_agents(team)]
    except Exception:
        # fallback: estimate defenders_list contains current_pos only
        defenders_list = [current_pos]

    # If multiple defenders exist and current_pos not in defenders_list, attempt to find and insert
    if current_pos not in defenders_list:
        # try to find some defender position slot to represent this defender
        # we will just append current_pos to defenders_list to simulate its move in minimax
        defenders_list.append(current_pos)

    # choose index for this defender (first occurrence)
    try:
        my_idx = defenders_list.index(current_pos)
    except ValueError:
        my_idx = 0

    for dmove in defender_moves:
        # simulate moving this defender to dmove (others hold)
        temp_defenders = list(defenders_list)
        temp_defenders[my_idx] = dmove

        # stage cost (using intercept nodes & capture radius logic)
        sc = stage_cost_intercept(dmove, attacker_positions, real_flags, lookup, intercept_nodes, capture_R=2)

        # attackers will maximize (choose worst response)
        max_value = -float('inf')

        # iterate over all attacker joint moves
        for att_joint in itertools.product(*attacker_move_sets):
            future_attackers = list(att_joint)
            tc = terminal_cost_intercept(dmove, future_attackers, temp_defenders, real_flags, lookup, intercept_nodes, capture_R=2)
            value = sc + beta * tc
            if value > max_value:
                max_value = value

        # defender wants minimal max_value
        if max_value < best_score:
            best_score = max_value
            best_moves = [dmove]
        elif math.isclose(max_value, best_score, rel_tol=1e-9):
            best_moves.append(dmove)

    # tie-break randomly
    chosen = random.choice(best_moves) if best_moves else current_pos
    state['action'] = chosen

    return set()

# wrapper
def map_strategy(agent_config):
    return {name: strategy for name in agent_config.keys()}
