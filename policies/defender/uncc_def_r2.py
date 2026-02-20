# defender_strategy.py
from __future__ import annotations
from typing import Dict, List, Tuple, Any
import math
import random
import networkx as nx
from lib.core.apsp_cache import get_apsp_length_cache, get_cached_distance

# ===================== Tunables =====================
DEFEND_RISK_HOPS = 3
RISK_DECAY = 0.6
DEFEND_RISK_THRESHOLD = 0.20

DIFFUSION_ALPHA = 0.65
STICKINESS = 0.05

SENSOR_LOCK = 0.9
SENSOR_SPREAD_HOPS = 2

SIGHT_DECAY = 0.9
RECENT_MIX = 0.2

ENTROPY_EXPLORE_THRESHOLD = 4.5
EXPLORER_MIN_HOLD = 8

EPS = 1e-12

_TEAM_MEMORY: Dict[str, Dict[str, Any]] = {
    "red":  {"belief": None, "last_graph_id": None, "recent": {}, "entropy": 0.0,
             "explorer": None, "explorer_since": -10},
    "blue": {"belief": None, "last_graph_id": None, "recent": {}, "entropy": 0.0,
             "explorer": None, "explorer_since": -10},
}

# ----------------- helpers -----------------
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
        mem["explorer"] = None
        mem["explorer_since"] = -10
        mem["last_graph_id"] = gid


def _maybe_assign_explorer(team: str, names: List[str], t: int) -> None:
    mem = _TEAM_MEMORY[team]
    if mem["entropy"] >= ENTROPY_EXPLORE_THRESHOLD:
        if mem["explorer"] is None or (t - mem["explorer_since"]) >= EXPLORER_MIN_HOLD:
            mem["explorer"] = sorted(names)[0] if names else None
            mem["explorer_since"] = t
    else:
        if mem["explorer"] is not None and (t - mem["explorer_since"]) >= EXPLORER_MIN_HOLD:
            mem["explorer"] = None


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


def _nearest_target_distance(G: nx.Graph, src: int,
                             targets: List[int], apsp=None) -> int | None:
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


def _flag_risk(G: nx.Graph, b: Dict[int, float], flag_node: int, apsp=None) -> float:
    region = _k_hop_ball(G, flag_node, DEFEND_RISK_HOPS)
    tot = 0.0
    for n in region:
        d = get_cached_distance(apsp, n, flag_node) if apsp else None
        if d is None:
            continue
        tot += b.get(n, 0.0) * math.exp(-RISK_DECAY * d)
    return tot

# ===================== MAIN STRATEGY =====================

def strategy(state):
    """
    Defender: diffusion belief over attackers, risk = belief × consequence
    around home flags, plus one explorer when uncertainty is high.
    """
    agent_ctrl = state["agent_controller"]
    current_pos: int = state["curr_pos"]
    current_time: int = state["time"]
    team: str = agent_ctrl.team
    speed: float = agent_ctrl.speed

    cache = agent_ctrl.cache
    cache.set("last_position", current_pos)
    cache.set("visit_count", cache.get("visit_count", 0) + 1)
    cache.update(last_time=current_time,
                 patrol_index=cache.get("patrol_index", 0) + 1)

    team_cache = agent_ctrl.team_cache
    team_cache.set("last_update", current_time)
    team_cache.update(total_tags=team_cache.get("total_tags", 0),
                      formation="defensive")

    # --- Map setup (like example_def.py) ---
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
    sensors: Dict[str, Tuple[Any, Dict[str, Any]]] = state["sensor"]

    # --- Home flags (real ones) ---
    home_flags: List[int] = []
    if "flag" in sensors:
        home_flags = sensors["flag"][1]["real_flags"]

    # --- Observed attackers from sensors ---
    enemy_team: str = "red" if team == "blue" else "blue"
    observed_attackers: List[int] = []

    if "egocentric_agent" in sensors:
        for name, node in sensors["egocentric_agent"][1].items():
            if name.startswith(enemy_team):
                observed_attackers.append(node)

    if "agent" in sensors:
        for name, node in sensors["agent"][1].items():
            if name.startswith(enemy_team):
                observed_attackers.append(node)

    # --- Update team belief over attacker locations ---
    _ensure_team_belief(team, G)
    mem = _TEAM_MEMORY[team]

    b = mem["belief"]
    _update_recent(team, observed_attackers)
    b = _diffuse_mixture(G, b)
    b = _apply_sensor_clamps(G, b, observed_attackers, apsp=apsp)
    b = _inject_recent(team, b)
    b = _normalize(b)
    mem["belief"] = b
    mem["entropy"] = _entropy(b)

    # --- Explorer assignment when entropy is high ---
    teammates = [nm for (nm, _, _) in agent_map.get_team_agents(team)]
    if agent_ctrl.name not in teammates:
        teammates.append(agent_ctrl.name)
    _maybe_assign_explorer(team, teammates, current_time)
    is_explorer = (mem["explorer"] == agent_ctrl.name)

    # --- Risk for each home flag ---
    defend_targets: List[int] = []
    if home_flags:
        risks = [(f, _flag_risk(G, b, f, apsp=apsp)) for f in home_flags]
        f_star, r_star = max(risks, key=lambda x: x[1])
        if r_star >= DEFEND_RISK_THRESHOLD:
            defend_targets = [f_star]

    target = current_pos

    if is_explorer:
        frontier = _frontier_node(G, b)
        if frontier is not None:
            target = _best_step_toward(agent_map, G, current_pos, [frontier], speed, apsp=apsp)
        elif home_flags:
            target = _best_step_toward(agent_map, G, current_pos, home_flags, speed, apsp=apsp)
    elif defend_targets:
        target = _best_step_toward(agent_map, G, current_pos, defend_targets, speed, apsp=apsp)
    elif home_flags:
        # gentle patrol near closest home flag
        target = _best_step_toward(agent_map, G, current_pos, home_flags, speed, apsp=apsp)
    elif agent_map.graph is not None:
        neighbors = list(agent_map.graph.neighbors(current_pos))
        if neighbors:
            target = random.choice(neighbors)

    state["action"] = target
    # defenders don't "discover flags" -> return empty set (like example_def.py)
    return set()


def map_strategy(agent_config):
    strategies = {}
    for name in agent_config.keys():
        strategies[name] = strategy
    return strategies
