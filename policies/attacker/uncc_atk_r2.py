# attacker_strategy.py
from __future__ import annotations
from typing import Dict, List, Tuple, Any
import math
import random
import networkx as nx
from lib.core.apsp_cache import get_apsp_length_cache, get_cached_distance

# ===================== Tunables =====================
DIFFUSION_ALPHA = 0.65
STICKINESS = 0.05

SENSOR_LOCK = 0.9
SENSOR_SPREAD_HOPS = 2

SIGHT_DECAY = 0.9
RECENT_MIX = 0.2

ENTROPY_EXPLORE_THRESHOLD = 4.5
SCOUT_MIN_HOLD = 8

EPS = 1e-12

# Per-team shared memory (module global)
_TEAM_MEMORY: Dict[str, Dict[str, Any]] = {
    "red":  {"belief": None, "last_graph_id": None, "recent": {}, "entropy": 0.0,
             "scout": None, "scout_since": -10},
    "blue": {"belief": None, "last_graph_id": None, "recent": {}, "entropy": 0.0,
             "scout": None, "scout_since": -10},
}

# ----------------- small helpers -----------------
def _graph_uid(G: nx.Graph) -> Tuple[int, int]:
    return (G.number_of_nodes(), G.number_of_edges())


def _init_uniform(G: nx.Graph) -> Dict[int, float]:
    p = 1.0 / max(1, G.number_of_nodes())
    return {n: p for n in G.nodes()}


def _full_support(b: Dict[int, float], G: nx.Graph) -> Dict[int, float]:
    if not b:
        return _init_uniform(G)
    for n in G.nodes():
        b.setdefault(n, 0.0)
    return b


def _normalize(b: Dict[int, float]) -> Dict[int, float]:
    s = sum(b.values()) + EPS
    if s > 0:
        for k in b:
            b[k] /= s
    return b


def _k_hop_ball(G: nx.Graph, center: int, k: int) -> List[int]:
    if k <= 0:
        return [center]
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
    return list(seen)


def _diffuse_once(G: nx.Graph, b: Dict[int, float]) -> Dict[int, float]:
    b = _full_support(b, G)
    out = {n: 0.0 for n in G.nodes()}
    for n, p in b.items():
        neigh = list(G.neighbors(n))
        out[n] += STICKINESS * p
        if neigh:
            share = (1.0 - STICKINESS) * p / len(neigh)
            for v in neigh:
                out[v] += share
        else:
            out[n] += (1.0 - STICKINESS) * p
    return _normalize(out)


def _diffuse_mixture(G: nx.Graph, b: Dict[int, float]) -> Dict[int, float]:
    b = _full_support(b, G)
    step = _diffuse_once(G, b)
    mixed = {n: (1.0 - DIFFUSION_ALPHA) * b.get(n, 0.0) +
                DIFFUSION_ALPHA * step.get(n, 0.0) for n in G.nodes()}
    return _normalize(mixed)


def _apply_sensor_clamps(G: nx.Graph, b: Dict[int, float],
                         obs_nodes: List[int], apsp=None) -> Dict[int, float]:
    if not obs_nodes:
        return b
    like = {n: 0.0 for n in G.nodes()}
    for o in obs_nodes:
        support = _k_hop_ball(G, o, SENSOR_SPREAD_HOPS)
        for n in support:
            d = get_cached_distance(apsp, o, n) if apsp else None
            if d is None:
                continue
            like[n] = max(like[n], math.exp(-0.8 * d))
    _normalize(like)
    post = {n: (1.0 - SENSOR_LOCK) * b.get(n, 0.0) +
               SENSOR_LOCK * like.get(n, 0.0) for n in G.nodes()}
    return _normalize(post)


def _update_recent(team: str, obs_nodes: List[int]) -> None:
    mem = _TEAM_MEMORY[team]["recent"]
    for k in list(mem.keys()):
        mem[k] *= SIGHT_DECAY
        if mem[k] < 1e-6:
            del mem[k]
    for n in obs_nodes:
        mem[n] = mem.get(n, 0.0) + 1.0


def _inject_recent(team: str, b: Dict[int, float]) -> Dict[int, float]:
    R = _TEAM_MEMORY[team]["recent"]
    if not R:
        return b
    s = sum(R.values()) + EPS
    Rn = {k: v / s for k, v in R.items()}
    mixed = {n: (1.0 - RECENT_MIX) * b.get(n, 0.0) +
                RECENT_MIX * Rn.get(n, 0.0) for n in b}
    return _normalize(mixed)


def _entropy(b: Dict[int, float]) -> float:
    return -sum(p * math.log(p + 1e-12) for p in b.values())


def _ensure_team_belief(team: str, G: nx.Graph) -> None:
    mem = _TEAM_MEMORY[team]
    gid = _graph_uid(G)
    if mem["belief"] is None or mem["last_graph_id"] != gid:
        mem["belief"] = _init_uniform(G)
        mem["recent"].clear()
        mem["entropy"] = 0.0
        mem["scout"] = None
        mem["scout_since"] = -10
        mem["last_graph_id"] = gid


def _maybe_assign_scout(team: str, agent_names: List[str], t: int) -> None:
    mem = _TEAM_MEMORY[team]
    if mem["entropy"] >= ENTROPY_EXPLORE_THRESHOLD:
        if mem["scout"] is None or (t - mem["scout_since"]) >= SCOUT_MIN_HOLD:
            mem["scout"] = sorted(agent_names)[0] if agent_names else None
            mem["scout_since"] = t
    else:
        if mem["scout"] is not None and (t - mem["scout_since"]) >= SCOUT_MIN_HOLD:
            mem["scout"] = None


def _frontier_node(G: nx.Graph, b: Dict[int, float]) -> int | None:
    best, arg = -1.0, None
    for n in G.nodes():
        neigh = list(G.neighbors(n))
        if not neigh:
            continue
        grad = max(b.get(v, 0.0) - b.get(n, 0.0) for v in neigh)
        if grad > best:
            best, arg = grad, n
    return arg


def _nearest_target_distance(G: nx.Graph, src: int, targets: List[int], apsp=None) -> int | None:
    best = None
    for t in targets:
        d = get_cached_distance(apsp, src, t) if apsp else None
        if d is None:
            continue
        best = d if best is None or d < best else best
    return best


def _best_step_toward(agent_map, G: nx.Graph, curr: int,
                      targets: List[int], speed: float, apsp=None) -> int:
    if not targets:
        return curr
    dists = []
    for t in targets:
        d = _nearest_target_distance(G, curr, [t], apsp=apsp)
        if d is not None:
            dists.append((d, t))
    if not dists:
        return curr
    _, t_star = min(dists, key=lambda x: x[0])
    step = agent_map.shortest_path_step(curr, t_star, speed)
    return curr if step is None else step

# ===================== MAIN STRATEGY =====================

def strategy(state):
    """
    Attacker: diffusion-based belief over defenders and entropy-aware scout,
    attacking candidate/confirmed flags.
    """
    # --- Agent controller & basic state ---
    agent_ctrl = state["agent_controller"]
    current_pos: int = state["curr_pos"]
    current_time: int = state["time"]
    team: str = agent_ctrl.team           # 'red' or 'blue'
    speed: float = agent_ctrl.speed

    cache = agent_ctrl.cache
    cache.set("last_position", current_pos)
    cache.set("visit_count", cache.get("visit_count", 0) + 1)
    cache.update(last_time=current_time,
                 patrol_index=cache.get("patrol_index", 0) + 1)

    team_cache = agent_ctrl.team_cache
    team_cache.set("last_update", current_time)
    team_cache.update(total_captures=team_cache.get("total_captures", 0),
                      formation="spread")

    # --- Map setup (same pattern as example_atk) ---
    agent_map = agent_ctrl.map
    global_map_payload: Dict[str, Any] = state["sensor"]["global_map"][1]
    global_map_sensor: nx.Graph = global_map_payload["graph"]
    global_map_apsp = global_map_payload.get("apsp")

    if agent_map.graph is None:
        nodes_data: Dict[int, Dict[str, Any]] = {
            nid: global_map_sensor.nodes[nid] for nid in global_map_sensor.nodes()
        }
        edges_data: Dict[int, Dict[str, Any]] = {}
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
    agent_map.update_time(current_time)
    agent_map.update_agent_position(team, agent_ctrl.name, current_pos, current_time)

    G: nx.Graph = global_map_sensor
    apsp = (
        agent_map.apsp_lookup
        if isinstance(agent_map.apsp_lookup, dict)
        else global_map_apsp if isinstance(global_map_apsp, dict)
        else get_apsp_length_cache(G)
    )

    # --- Observed defenders from sensors (for belief) ---
    sensors: Dict[str, Tuple[Any, Dict[str, Any]]] = state["sensor"]
    enemy_team: str = "blue" if team == "red" else "red"
    observed_defenders: List[int] = []

    if "agent" in sensors:
        # global agent sensor
        for name, node in sensors["agent"][1].items():
            if name.startswith(enemy_team):
                observed_defenders.append(node)

    if "egocentric_agent" in sensors:
        # local sensor
        for name, node in sensors["egocentric_agent"][1].items():
            if name.startswith(enemy_team):
                observed_defenders.append(node)

    # --- Update team belief over defender locations ---
    _ensure_team_belief(team, G)
    mem = _TEAM_MEMORY[team]

    b = mem["belief"]
    _update_recent(team, observed_defenders)
    b = _diffuse_mixture(G, b)
    b = _apply_sensor_clamps(G, b, observed_defenders, apsp=apsp)
    b = _inject_recent(team, b)
    b = _normalize(b)
    mem["belief"] = b
    mem["entropy"] = _entropy(b)

    # --- Scout assignment based on entropy ---
    teammates_names = [nm for (nm, _, _) in agent_map.get_team_agents(team)]
    if agent_ctrl.name not in teammates_names:
        teammates_names.append(agent_ctrl.name)
    _maybe_assign_scout(team, teammates_names, current_time)
    is_scout = (mem["scout"] == agent_ctrl.name)

    # --- Flags for attackers ---
    candidate_flags: List[int] = []
    detected_flags: List[int] = []

    if "candidate_flag" in sensors:
        candidate_flags = sensors["candidate_flag"][1]["candidate_flags"]
    if "egocentric_flag" in sensors:
        detected_flags = sensors["egocentric_flag"][1]["detected_flags"]

    target = current_pos  # default

    if is_scout:
        # Move toward frontier (uncertainty-reducing) region
        frontier = _frontier_node(G, b)
        if frontier is not None:
            target = _best_step_toward(agent_map, G, current_pos, [frontier], speed, apsp=apsp)
    else:
        # Main attackers: go for nearest (detected or candidate) flag
        flag_targets = detected_flags or candidate_flags
        if flag_targets:
            target = _best_step_toward(agent_map, G, current_pos, flag_targets, speed, apsp=apsp)
        elif agent_map.graph is not None:
            # mild exploration when we know nothing
            neighbors = list(agent_map.graph.neighbors(current_pos))
            if neighbors:
                target = random.choice(neighbors)

    state["action"] = target
    # Return discovered real flags (if any), as in example_atk.py
    return set(detected_flags)


def map_strategy(agent_config):
    strategies = {}
    for name in agent_config.keys():
        strategies[name] = strategy
    return strategies
