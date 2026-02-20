# attacker_strategy.py
from __future__ import annotations

from typing import Dict, List, Any, Set, Tuple, Optional
import math
import random
import networkx as nx


# ============================================================
# Tunables (ATTACKER)
# ============================================================

# --- Multi-explorer control (instead of only one scout) ---
EXPLORER_HOLD_TICKS = 12                 # fixed commitment window for explorer assignment
EXPLORER_FRACTION = 0.4                  # fraction of attackers to be explorers when uncertainty high
MIN_EXPLORERS = 1
MAX_EXPLORERS = 3

# --- Candidate belief update ---
BELIEF_POS_BOOST = 3.0                   # boost candidates that are detected flags
BELIEF_NEG_DECAY = 0.05                  # how much to reduce belief for candidates in visible area w/ no flag

# --- Two-layer belief: danger heat (defender risk map) ---
DANGER_DECAY = 0.93                      # decay danger heat each tick
DANGER_ADD_VISIBLE_DEF = 1.0             # add heat around visible defenders
DANGER_ADD_TAG_SPIKE = 3.0               # spike heat if we infer a tag/death-like event (best-effort)
DANGER_SPREAD_HOPS = 2                   # spread danger around sources
RISK_WEIGHT = 1.2                        # how strongly to avoid danger during path selection

# --- Exploration scoring (info gain frontier) ---
INFO_HOPS = 2                            # coverage neighborhood for info gain at candidate
INFO_WEIGHT = 0.8                        # weight of "new info" vs pure belief
OVERLAP_PENALTY = 0.7                    # reduce score if candidate overlaps already assigned explorer region

# --- Commitment / hysteresis ---
TARGET_COMMIT_TICKS = 10                 # hold a chosen target for fixed ticks to avoid thrashing
SWITCH_IMPROVEMENT_RATIO = 1.25          # only switch if new target is >= 1.25x better

# --- Capture behavior ---
CAPTURE_SPREAD = True                    # distribute capture targets across known flags

EPS = 1e-12


# ============================================================
# Small helpers
# ============================================================

def _normalize_dict(d: Dict[int, float]) -> Dict[int, float]:
    s = sum(d.values())
    if s <= 0:
        return d
    inv = 1.0 / (s + EPS)
    for k in list(d.keys()):
        d[k] *= inv
    return d


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


def _best_step_toward_with_risk(agent_map, G: nx.Graph, curr: int, goal: int, speed: float,
                               danger: Dict[int, float]) -> int:
    """
    Pick a single step that progresses to goal but avoids danger.
    We evaluate candidate neighbors with:
        dist_to_goal(neighbor) + RISK_WEIGHT * danger(neighbor)
    """
    if curr == goal:
        return curr

    # If engine supports shortest_path_step use it as baseline,
    # but we improve locally using neighbor scoring.
    neighbors = list(agent_map.graph.neighbors(curr)) if agent_map.graph else []
    if not neighbors:
        step = agent_map.shortest_path_step(curr, goal, speed)
        return curr if step is None else step

    best_n = curr
    best_score = float("inf")
    for n in neighbors:
        try:
            d = nx.shortest_path_length(G, n, goal)
        except nx.NetworkXNoPath:
            d = 10**9
        score = d + RISK_WEIGHT * danger.get(n, 0.0)
        if score < best_score:
            best_score = score
            best_n = n

    # If we're totally stuck, fallback to shortest_path_step
    if best_n == curr:
        step = agent_map.shortest_path_step(curr, goal, speed)
        return curr if step is None else step

    return best_n


def _stable_hash_pick(items: List[int], name: str) -> int:
    return items[abs(hash(name)) % len(items)]


def _infer_tag_spike(state: Dict, cache) -> Optional[int]:
    """
    Best-effort: try to infer a 'tag/death' event.
    This game version doesn't always expose events. We do conservative checks:
    - If position jumps (teleport) compared to last position, treat last position as risky.
    - If state has something like state['killed'] or state['respawned'], use it.
    If nothing is available, returns None.
    """
    curr = state.get("curr_pos", None)
    last = cache.get("last_position", None)

    # explicit flags if present
    for key in ("killed", "respawned", "was_tagged", "tagged"):
        if key in state and state[key]:
            return last if last is not None else curr

    # teleport heuristic: if it looks like a respawn jump
    if curr is not None and last is not None and curr != last:
        # if graph exists, we can measure distance; huge jump suggests respawn
        try:
            G = state["sensor"]["global_map"][1]["graph"]
            d = nx.shortest_path_length(G, last, curr)
            if d >= 25:  # heuristic threshold
                return last
        except Exception:
            pass

    return None


# ============================================================
# Team-shared state in team_cache keys
# ============================================================

def _get_team_mem(team_cache) -> Dict[str, Any]:
    # store everything under a single dict so we don't pollute cache too much
    mem = team_cache.get("ATTACKER_MEM", None)
    if mem is None or not isinstance(mem, dict):
        mem = {}
        team_cache.set("ATTACKER_MEM", mem)
    return mem


def _ensure_beliefs(mem: Dict[str, Any], candidates: List[int]):
    # candidate belief
    cb = mem.get("cand_belief", None)
    cand_set = tuple(sorted(candidates))
    if cb is None or mem.get("cand_set", None) != cand_set:
        if candidates:
            p = 1.0 / len(candidates)
            cb = {c: p for c in candidates}
        else:
            cb = {}
        mem["cand_belief"] = cb
        mem["cand_set"] = cand_set

    # danger heat
    if mem.get("danger_heat", None) is None:
        mem["danger_heat"] = {}

    # seen nodes (team wide)
    if mem.get("seen_nodes", None) is None:
        mem["seen_nodes"] = set()

    # known flags (team wide)
    if mem.get("known_flags", None) is None:
        mem["known_flags"] = set()

    # explorer bookkeeping
    if mem.get("explorers", None) is None:
        mem["explorers"] = []
        mem["explore_until"] = -1
        mem["explore_targets"] = {}  # name -> candidate

    # capture targets bookkeeping
    if mem.get("capture_targets", None) is None:
        mem["capture_targets"] = {}  # name -> flag


def _update_candidate_belief(mem: Dict[str, Any],
                            candidates: List[int],
                            detected_flags: List[int],
                            visible_nodes: Set[int]) -> None:
    cb: Dict[int, float] = mem["cand_belief"]

    # Keep support aligned with candidates
    cb = {c: cb.get(c, 0.0) for c in candidates}

    # Positive update: detected flags
    for f in detected_flags:
        if f in cb:
            cb[f] = cb.get(f, 0.0) + BELIEF_POS_BOOST

    # Negative info: if we see the area but do NOT detect a flag,
    # reduce candidates that lie in visible_nodes.
    if not detected_flags and visible_nodes:
        for c in candidates:
            if c in visible_nodes:
                cb[c] = max(0.0, cb.get(c, 0.0) * (1.0 - BELIEF_NEG_DECAY))

    mem["cand_belief"] = _normalize_dict(cb)


def _decay_and_update_danger(mem: Dict[str, Any],
                            G: nx.Graph,
                            visible_defenders: List[int],
                            spike_node: Optional[int]) -> None:
    danger: Dict[int, float] = mem["danger_heat"]

    # decay
    for k in list(danger.keys()):
        danger[k] *= DANGER_DECAY
        if danger[k] < 1e-4:
            del danger[k]

    # add heat around visible defenders
    for dpos in visible_defenders:
        region = _k_hop_ball(G, dpos, DANGER_SPREAD_HOPS)
        for n in region:
            danger[n] = danger.get(n, 0.0) + DANGER_ADD_VISIBLE_DEF

    # spike if we infer a tag-like event
    if spike_node is not None:
        region = _k_hop_ball(G, spike_node, DANGER_SPREAD_HOPS)
        for n in region:
            danger[n] = danger.get(n, 0.0) + DANGER_ADD_TAG_SPIKE

    mem["danger_heat"] = danger


def _info_gain(mem: Dict[str, Any], G: nx.Graph, candidate: int) -> int:
    seen: Set[int] = mem["seen_nodes"]
    ball = _k_hop_ball(G, candidate, INFO_HOPS)
    return len([n for n in ball if n not in seen])


def _choose_explorers(mem: Dict[str, Any], names: List[str], t: int, candidates: List[int]) -> None:
    """
    Assign multiple explorers for fixed ticks when uncertainty is high.
    Uncertainty proxy: number of plausible candidates remaining (belief mass not concentrated).
    """
    if not names:
        mem["explorers"] = []
        mem["explore_until"] = -1
        mem["explore_targets"] = {}
        return

    # If still within hold window, keep existing assignment
    if t <= mem.get("explore_until", -1) and mem.get("explorers", []):
        return

    cb: Dict[int, float] = mem["cand_belief"]
    if not cb:
        # no candidates => no need for explorers
        mem["explorers"] = []
        mem["explore_until"] = -1
        mem["explore_targets"] = {}
        return

    # entropy proxy from candidate belief
    H = 0.0
    for p in cb.values():
        if p > 0:
            H -= p * math.log(p + EPS)

    # choose K based on entropy (high entropy => more explorers)
    base = int(math.ceil(EXPLORER_FRACTION * len(names)))
    K = base
    if H < 1.0:
        K = max(1, base - 1)
    if H > 2.0:
        K = min(len(names), base + 1)

    K = max(MIN_EXPLORERS, min(MAX_EXPLORERS, K, len(names)))

    explorers = sorted(names)[:K]  # deterministic
    mem["explorers"] = explorers
    mem["explore_until"] = t + EXPLORER_HOLD_TICKS
    mem["explore_targets"] = {}


def _assign_explore_targets(mem: Dict[str, Any],
                            G: nx.Graph,
                            candidates: List[int],
                            t: int) -> None:
    """
    Greedy assignment:
      score(candidate) = belief + INFO_WEIGHT * info_gain - overlap_penalty
    """
    explorers: List[str] = mem["explorers"]
    if not explorers or not candidates:
        mem["explore_targets"] = {}
        return

    # do not reassign within the hold window if already assigned
    if mem.get("explore_targets") and t <= mem.get("explore_until", -1):
        return

    cb: Dict[int, float] = mem["cand_belief"]
    assigned_regions: List[Set[int]] = []
    explore_targets: Dict[str, int] = {}

    # precompute base scores
    base_scores: List[Tuple[int, float, Set[int]]] = []
    for c in candidates:
        p = cb.get(c, 0.0)
        gain = _info_gain(mem, G, c)
        region = _k_hop_ball(G, c, INFO_HOPS)
        score = p + INFO_WEIGHT * float(gain)
        base_scores.append((c, score, region))

    # greedy assign best non-overlapping-ish targets
    remaining = base_scores[:]
    for ex in explorers:
        best_c = None
        best_s = -1e18
        best_region = None

        for c, s, region in remaining:
            overlap = 0.0
            for r0 in assigned_regions:
                if r0:
                    overlap += len(region & r0) / max(1, len(region))
            penalized = s - OVERLAP_PENALTY * overlap
            if penalized > best_s:
                best_s = penalized
                best_c = c
                best_region = region

        if best_c is None:
            break

        explore_targets[ex] = best_c
        assigned_regions.append(best_region if best_region is not None else set())

        # optionally remove that candidate to reduce duplicates
        remaining = [(c, s, r) for (c, s, r) in remaining if c != best_c]

    mem["explore_targets"] = explore_targets


def _assign_capture_targets(mem: Dict[str, Any], names: List[str]) -> None:
    known_flags = list(mem["known_flags"])
    if not known_flags:
        mem["capture_targets"] = {}
        return

    # deterministic stable distribution
    known_flags_sorted = sorted(known_flags)
    capture_targets = {}
    for nm in names:
        if CAPTURE_SPREAD and len(known_flags_sorted) > 1:
            # rotate by hash(name)
            capture_targets[nm] = _stable_hash_pick(known_flags_sorted, nm)
        else:
            capture_targets[nm] = known_flags_sorted[0]
    mem["capture_targets"] = capture_targets


def _commit_or_choose(cache,
                      curr_t: int,
                      proposed_goal: Optional[int],
                      proposed_score: float) -> Optional[int]:
    """
    Commitment rule:
      - keep existing goal until commit_until
      - if expired, accept proposed
      - if still committed, only switch if proposed is significantly better
    """
    if proposed_goal is None:
        return cache.get("goal", None)

    goal = cache.get("goal", None)
    commit_until = cache.get("commit_until", -1)
    goal_score = cache.get("goal_score", None)

    if goal is None or curr_t > commit_until:
        cache.set("goal", proposed_goal)
        cache.set("goal_score", proposed_score)
        cache.set("commit_until", curr_t + TARGET_COMMIT_TICKS)
        return proposed_goal

    # still committed
    if goal_score is None:
        return goal

    if proposed_score >= SWITCH_IMPROVEMENT_RATIO * float(goal_score):
        cache.set("goal", proposed_goal)
        cache.set("goal_score", proposed_score)
        cache.set("commit_until", curr_t + TARGET_COMMIT_TICKS)
        return proposed_goal

    return goal


# ============================================================
# Main attacker strategy
# ============================================================

def strategy(state):
    agent_ctrl = state["agent_controller"]
    name: str = agent_ctrl.name
    curr: int = state["curr_pos"]
    t: int = state["time"]
    team: str = agent_ctrl.team  # 'red' or 'blue'
    speed: float = agent_ctrl.speed

    cache = agent_ctrl.cache
    team_cache = agent_ctrl.team_cache

    # ----- Build map (same safe pattern as your working example) -----
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

    # ----- Candidate nodes -----
    candidates: List[int] = []
    if "candidate_flag" in sensors:
        candidates = sensors["candidate_flag"][1].get("candidate_flags", [])

    # ----- True flags visible to attacker -----
    detected_flags: List[int] = []
    if "egocentric_flag" in sensors:
        detected_flags = sensors["egocentric_flag"][1].get("detected_flags", [])

    # ----- Visible map region (state-sense) -----
    visible_nodes: Set[int] = set()
    if "egocentric_map" in sensors:
        visible_nodes = set(sensors["egocentric_map"][1].get("nodes", {}).keys())

    # ----- Visible defenders (if "agent" sensor gives all agents) -----
    visible_defenders: List[int] = []
    enemy_team = "blue" if team == "red" else "red"
    if "agent" in sensors:
        all_agents: Dict[str, int] = sensors["agent"][1]
        for nm, pos in all_agents.items():
            if nm.startswith(enemy_team):
                visible_defenders.append(pos)

    # ----- Team memory -----
    mem = _get_team_mem(team_cache)
    _ensure_beliefs(mem, candidates)

    # update team seen nodes
    if visible_nodes:
        mem["seen_nodes"] = set(mem["seen_nodes"]) | set(visible_nodes)

    # update known flags (team)
    if detected_flags:
        kf = set(mem["known_flags"])
        for f in detected_flags:
            kf.add(f)
        mem["known_flags"] = kf

    # infer tag/danger spike node (best effort)
    spike_node = _infer_tag_spike(state, cache)

    # update beliefs
    _update_candidate_belief(mem, candidates, detected_flags, visible_nodes)
    _decay_and_update_danger(mem, G, visible_defenders, spike_node)

    # choose explorers and targets (multi-explorer)
    team_names = [nm for (nm, _, _) in agent_map.get_team_agents(team)]
    if name not in team_names:
        team_names.append(name)

    _choose_explorers(mem, sorted(team_names), t, candidates)
    _assign_explore_targets(mem, G, candidates, t)
    _assign_capture_targets(mem, sorted(team_names))

    explorers: List[str] = mem["explorers"]
    is_explorer = (name in explorers)

    cb: Dict[int, float] = mem["cand_belief"]
    danger: Dict[int, float] = mem["danger_heat"]
    known_flags = list(mem["known_flags"])

    # ========================================================
    # Decide goal (with commitment / hysteresis)
    # ========================================================

    proposed_goal: Optional[int] = None
    proposed_score: float = 0.0

    if is_explorer or not known_flags:
        # --- Explore mode: go to assigned explore target; fallback to best candidate by belief+info
        tgt = mem.get("explore_targets", {}).get(name, None)

        if tgt is None and candidates:
            # fallback choose best by belief + info gain
            best_c, best_s = None, -1e18
            for c in candidates:
                s = cb.get(c, 0.0) + INFO_WEIGHT * float(_info_gain(mem, G, c))
                if s > best_s:
                    best_s = s
                    best_c = c
            tgt = best_c

        proposed_goal = tgt
        if proposed_goal is not None:
            proposed_score = cb.get(proposed_goal, 0.0) + INFO_WEIGHT * float(_info_gain(mem, G, proposed_goal))

    else:
        # --- Capture mode: go to assigned flag (spread), but still avoid danger on the path
        assigned_flag = mem.get("capture_targets", {}).get(name, None)
        if assigned_flag is None and known_flags:
            assigned_flag = _stable_hash_pick(sorted(known_flags), name)

        proposed_goal = assigned_flag
        proposed_score = 1.0  # capture intent, not comparable to explore score

    goal = _commit_or_choose(cache, t, proposed_goal, proposed_score)

    # ========================================================
    # Move one step toward goal (risk-aware)
    # ========================================================

    target = curr
    if goal is not None:
        target = _best_step_toward_with_risk(agent_map, G, curr, goal, speed, danger)
    else:
        # fallback: random neighbor
        nbrs = list(agent_map.graph.neighbors(curr)) if agent_map.graph else []
        if nbrs:
            target = random.choice(nbrs)

    # store last position for tag inference
    cache.set("last_position", curr)

    state["action"] = target

    # IMPORTANT: attacker returns discovered real flags for reward (N_flag_found)
    return set(detected_flags)


def map_strategy(agent_config):
    return {name: strategy for name in agent_config.keys()}
