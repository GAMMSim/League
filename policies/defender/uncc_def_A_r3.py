# defender_strategy.py
from __future__ import annotations

from typing import Dict, List, Any, Tuple, Set, Optional
import math
import random
import collections
import networkx as nx


# ============================================================
# Tunables (DEFENDER)
# ============================================================

# Voronoi / territory
REASSIGN_TICKS = 30                 # how often to refresh assignments / partitions if needed

# Threat / heat
HEAT_DECAY = 0.92
HEAT_ADD_DETECTION = 1.0
HEAT_SPREAD_HOPS = 2

# Defend behavior
DEFEND_THRESHOLD = 1.6              # higher => less twitchy
INTERCEPT_HOPS = 3                  # search hot intercept nodes around threatened flag

# Multi-scout (deny find reward)
SCOUT_HOLD_TICKS = 12
SCOUT_FRACTION = 0.34
MIN_SCOUTS = 1
MAX_SCOUTS = 2

# Commitment / hysteresis
MODE_HOLD_TICKS = 10

# patrol
PATROL_WANDER_PROB = 0.25

EPS = 1e-12


# ============================================================
# Helpers
# ============================================================

def _k_hop_ball(G: nx.Graph, center: int, k: int) -> Set[int]:
    if k <= 0:
        return {center}
    seen = {center}
    frontier = [center]
    for _ in range(k):
        nxt = []
        for u in frontier:
            for v in G.neighbors(u):
                if v not in seen:
                    seen.add(v)
                    nxt.append(v)
        frontier = nxt
        if not frontier:
            break
    return seen


def _best_step_toward(agent_map, curr: int, goal: int, speed: float) -> int:
    step = agent_map.shortest_path_step(curr, goal, speed)
    return curr if step is None else step


def _nearest(G: nx.Graph, src: int, goals: List[int]) -> Optional[int]:
    best_g, best_d = None, None
    for g in goals:
        try:
            d = nx.shortest_path_length(G, src, g)
        except nx.NetworkXNoPath:
            continue
        if best_d is None or d < best_d:
            best_d, best_g = d, g
    return best_g


def _multi_source_voronoi(G: nx.Graph, sources: List[int]) -> Dict[int, int]:
    """
    Multi-source BFS Voronoi assignment on an unweighted graph:
    returns dict node -> closest_source (ties broken by smaller source id).
    """
    if not sources:
        return {}

    sources_sorted = sorted(sources)
    owner: Dict[int, int] = {}
    dist: Dict[int, int] = {}

    q = collections.deque()
    for s in sources_sorted:
        owner[s] = s
        dist[s] = 0
        q.append(s)

    while q:
        u = q.popleft()
        for v in G.neighbors(u):
            cand_owner = owner[u]
            cand_dist = dist[u] + 1

            if v not in dist:
                dist[v] = cand_dist
                owner[v] = cand_owner
                q.append(v)
            else:
                # tie-break: if equal dist and smaller owner id
                if cand_dist == dist[v] and cand_owner < owner[v]:
                    owner[v] = cand_owner

    return owner


def _get_team_mem(team_cache) -> Dict[str, Any]:
    mem = team_cache.get("DEFENDER_MEM", None)
    if mem is None or not isinstance(mem, dict):
        mem = {}
        team_cache.set("DEFENDER_MEM", mem)
    return mem


def _ensure_mem(mem: Dict[str, Any], home_flags: List[int]):
    if mem.get("heat", None) is None:
        mem["heat"] = {}

    if mem.get("partition_owner", None) is None:
        mem["partition_owner"] = {}  # node -> flag

    if mem.get("assignments", None) is None:
        mem["assignments"] = {}      # agent -> flag

    if mem.get("scouts", None) is None:
        mem["scouts"] = []
        mem["scout_until"] = -1

    if mem.get("last_refresh", None) is None:
        mem["last_refresh"] = -999

    # record current flag set for refresh detection
    mem["home_flags"] = tuple(sorted(home_flags))


def _decay_and_update_heat(mem: Dict[str, Any], G: nx.Graph, attacker_nodes: List[int]) -> None:
    heat: Dict[int, float] = mem["heat"]

    # decay
    for k in list(heat.keys()):
        heat[k] *= HEAT_DECAY
        if heat[k] < 1e-4:
            del heat[k]

    # add detections spread
    for a in attacker_nodes:
        region = _k_hop_ball(G, a, HEAT_SPREAD_HOPS)
        for n in region:
            heat[n] = heat.get(n, 0.0) + HEAT_ADD_DETECTION

    mem["heat"] = heat


def _threat_for_flag(mem: Dict[str, Any], G: nx.Graph, flag_node: int) -> float:
    """
    Threat score for a flag = sum heat in its neighborhood (distance implicit via spread+decay).
    """
    heat: Dict[int, float] = mem["heat"]
    region = _k_hop_ball(G, flag_node, INTERCEPT_HOPS)
    return sum(heat.get(n, 0.0) for n in region)


def _choose_scouts(mem: Dict[str, Any], names: List[str], t: int, calm: bool) -> None:
    # If threatened, we reduce scouting; if calm, we run multi-scout for denial.
    if not calm:
        mem["scouts"] = []
        mem["scout_until"] = -1
        return

    if t <= mem.get("scout_until", -1) and mem.get("scouts", []):
        return

    if not names:
        mem["scouts"] = []
        mem["scout_until"] = -1
        return

    base = int(math.ceil(SCOUT_FRACTION * len(names)))
    K = max(MIN_SCOUTS, min(MAX_SCOUTS, base, len(names)))

    scouts = sorted(names)[-K:]  # deterministic but different from "interceptor"
    mem["scouts"] = scouts
    mem["scout_until"] = t + SCOUT_HOLD_TICKS


def _assign_defenders_to_flags(mem: Dict[str, Any], names: List[str], home_flags: List[int]) -> None:
    """
    Coverage assignment: distribute defenders across flags by name-hash.
    """
    if not home_flags:
        mem["assignments"] = {}
        return

    flags_sorted = sorted(home_flags)
    assignments = {}
    for nm in names:
        assignments[nm] = flags_sorted[abs(hash(nm)) % len(flags_sorted)]
    mem["assignments"] = assignments


def _pick_interceptor(names: List[str], scouts: List[str]) -> Optional[str]:
    """
    Choose one non-scout defender to act as interceptor reserve.
    """
    candidates = [n for n in sorted(names) if n not in set(scouts)]
    return candidates[0] if candidates else (sorted(names)[0] if names else None)


def _hot_intercept_node(mem: Dict[str, Any], G: nx.Graph, flag_node: int) -> int:
    """
    Pick the hottest node in a neighborhood around the threatened flag.
    """
    heat: Dict[int, float] = mem["heat"]
    region = list(_k_hop_ball(G, flag_node, INTERCEPT_HOPS))
    # include the flag as fallback
    best = flag_node
    best_h = heat.get(flag_node, 0.0)
    for n in region:
        h = heat.get(n, 0.0)
        if h > best_h:
            best_h = h
            best = n
    return best


# ============================================================
# Main defender strategy
# ============================================================

def strategy(state):
    agent_ctrl = state["agent_controller"]
    name: str = agent_ctrl.name
    curr: int = state["curr_pos"]
    t: int = state["time"]
    team: str = agent_ctrl.team
    speed: float = agent_ctrl.speed

    cache = agent_ctrl.cache
    team_cache = agent_ctrl.team_cache

    # ----- Map attach -----
    agent_map = agent_ctrl.map
    G: nx.Graph = state["sensor"]["global_map"][1]["graph"]

    nodes_data = {nid: G.nodes[nid] for nid in G.nodes()}
    edges_data = {}
    for idx, (u, v, data) in enumerate(G.edges(data=True)):
        edges_data[idx] = {"source": u, "target": v, **data}
    agent_map.attach_networkx_graph(nodes_data, edges_data)
    agent_map.update_time(t)
    agent_map.update_agent_position(team, name, curr, t)

    sensors = state["sensor"]

    # ----- True home flags -----
    home_flags: List[int] = []
    if "flag" in sensors:
        home_flags = sensors["flag"][1].get("real_flags", [])

    # ----- Candidate nodes (deny find reward) -----
    candidates: List[int] = []
    if "candidate_flag" in sensors:
        candidates = sensors["candidate_flag"][1].get("candidate_flags", [])

    # ----- Collect attacker detections -----
    enemy_team = "red" if team == "blue" else "blue"
    attacker_nodes: List[int] = []

    # local egocentric attacker sensor
    if "egocentric_agent" in sensors:
        nearby = sensors["egocentric_agent"][1]
        for nm, node in nearby.items():
            if nm.startswith(enemy_team):
                attacker_nodes.append(node)

    # stationary sensors (very valuable)
    for sensor_name in sensors:
        if sensor_name.startswith("stationary_"):
            sd = sensors[sensor_name][1]
            det = sd.get("detected_agents", {})
            for nm, info in det.items():
                if nm.startswith(enemy_team):
                    attacker_nodes.append(info.get("node_id", info.get("node", None)))

    attacker_nodes = [x for x in attacker_nodes if x is not None]

    # ----- Team memory -----
    mem = _get_team_mem(team_cache)
    _ensure_mem(mem, home_flags)

    # refresh partitions / assignments periodically or if flags changed
    refresh_needed = (t - mem.get("last_refresh", -999) >= REASSIGN_TICKS) or (mem.get("home_flags") != tuple(sorted(home_flags)))
    team_names = [nm for (nm, _, _) in agent_map.get_team_agents(team)]
    if name not in team_names:
        team_names.append(name)
    team_names = sorted(set(team_names))

    if refresh_needed and home_flags:
        mem["partition_owner"] = _multi_source_voronoi(G, home_flags)
        _assign_defenders_to_flags(mem, team_names, home_flags)
        mem["last_refresh"] = t

    # update attacker heatmap (memory)
    _decay_and_update_heat(mem, G, attacker_nodes)

    # compute threats
    target = curr

    if home_flags:
        threats = [(f, _threat_for_flag(mem, G, f)) for f in home_flags]
        f_star, thr_star = max(threats, key=lambda x: x[1])

        calm = (thr_star < DEFEND_THRESHOLD)

        # scouts in calm mode
        _choose_scouts(mem, team_names, t, calm=calm)
        scouts: List[str] = mem.get("scouts", [])
        is_scout = name in set(scouts)

        # interceptor reserve
        interceptor = _pick_interceptor(team_names, scouts)
        is_interceptor = (name == interceptor)

        # assignment (coverage via Voronoi-flags)
        assigned_flag = mem.get("assignments", {}).get(name, f_star)

        # commitment on mode to avoid thrashing
        mode = cache.get("mode", "calm")
        mode_until = cache.get("mode_until", -1)
        if t > mode_until:
            mode = "threat" if not calm else "calm"
            cache.set("mode", mode)
            cache.set("mode_until", t + MODE_HOLD_TICKS)

        # ---------------- Action selection ----------------
        if mode == "threat":
            if assigned_flag == f_star:
                # primary defender: protect threatened flag
                target = _best_step_toward(agent_map, curr, f_star, speed)

            elif is_interceptor:
                # interceptor: move to hottest intercept node around threatened flag
                hot = _hot_intercept_node(mem, G, f_star)
                target = _best_step_toward(agent_map, curr, hot, speed)

            else:
                # maintain coverage at assigned flag (do not all dogpile)
                target = _best_step_toward(agent_map, curr, assigned_flag, speed)

        else:
            # calm mode: deny "find" reward by patrolling candidates in our Voronoi territory
            if is_scout:
                # patrol candidate nodes inside my flag's Voronoi cell (or near it)
                owner = mem.get("partition_owner", {})
                territory = [v for v in candidates if owner.get(v, assigned_flag) == assigned_flag]

                if territory:
                    # choose the "hottest" candidate in territory (if attackers likely there), else nearest
                    heat = mem.get("heat", {})
                    # prefer top heat, but keep it reachable by also considering distance
                    best = None
                    best_score = -1e18
                    for c in territory:
                        h = heat.get(c, 0.0)
                        try:
                            d = nx.shortest_path_length(G, curr, c)
                        except nx.NetworkXNoPath:
                            d = 10**9
                        score = h - 0.05 * d
                        if score > best_score:
                            best_score = score
                            best = c
                    if best is not None:
                        target = _best_step_toward(agent_map, curr, best, speed)
                    else:
                        target = _best_step_toward(agent_map, curr, assigned_flag, speed)

                else:
                    # no candidates in territory -> orbit around flag
                    if random.random() < PATROL_WANDER_PROB:
                        nbrs = list(agent_map.graph.neighbors(curr)) if agent_map.graph else []
                        if nbrs:
                            target = random.choice(nbrs)
                    else:
                        target = _best_step_toward(agent_map, curr, assigned_flag, speed)

            else:
                # non-scout: hold near assigned flag inside territory (light orbit)
                if random.random() < PATROL_WANDER_PROB:
                    nbrs = list(agent_map.graph.neighbors(curr)) if agent_map.graph else []
                    if nbrs:
                        target = random.choice(nbrs)
                else:
                    target = _best_step_toward(agent_map, curr, assigned_flag, speed)

    state["action"] = target
    return set()  # must remain empty


def map_strategy(agent_config):
    return {name: strategy for name in agent_config.keys()}
