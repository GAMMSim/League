import math
import random
import networkx as nx
from typing import Dict, List, Tuple, Any

# ---------- Cost Function ----------
def attacker_stage_cost(pos: int, defenders: List[int], flags: List[int], lookup: Dict[int, Dict[int, float]], w_goal: float = 1.0, w_risk: float = 5.0, epsilon: float = 1e-3) -> float:
    """Immediate cost for a single attacker evaluating a specific node. Lower is better."""
    if not flags:
        return 0.0
    dist_to_flag = min(lookup[pos][f] for f in flags)
    risk_penalty = sum(1.0 / (lookup[pos][d] + epsilon) for d in defenders) if defenders else 0.0
    return w_goal * dist_to_flag + w_risk * risk_penalty

# ---------- Main Strategy ----------
def strategy(state: dict) -> str:
    # ===== AGENT CONTROLLER =====
    agent_ctrl = state["agent_controller"]
    current_pos: int = state["curr_pos"]
    current_time: int = state["time"]
    team: str = agent_ctrl.team
    enemy_team: str = agent_ctrl.enemy_team

    # ===== SENSORS & MAP =====
    sensors: Dict[str, Tuple[Any, Dict[str, Any]]] = state.get("sensor", {})
    
    # Flag Pruning Logic
    raw_candidates = agent_ctrl.sensor_data(state, "candidate_flag").get("candidate_flags", []) if "candidate_flag" in sensors else []
    detected_flags = agent_ctrl.sensor_data(state, "egocentric_flag").get("detected_flags", []) if "egocentric_flag" in sensors else []
    visible_nodes_dict = agent_ctrl.sensor_data(state, "egocentric_flag_region").get("nodes", {}) if "egocentric_flag_region" in sensors else {}

    shared_candidates = agent_ctrl.get_team("candidate_flags", set())
    if raw_candidates: shared_candidates.update(raw_candidates)
    
    visible_nodes = set(visible_nodes_dict.keys()) if visible_nodes_dict else {current_pos}
    to_remove = {c for c in shared_candidates if c in visible_nodes and c not in detected_flags}
    if to_remove: shared_candidates.difference_update(to_remove)
    if detected_flags: shared_candidates = set(detected_flags)
    
    agent_ctrl.set_team("candidate_flags", shared_candidates)
    flags_for_planning = list(shared_candidates)

    # ===== AGENT MAP INIT & SYNC =====
    agent_map = agent_ctrl.map
    if agent_map.graph is None and "global_map" in sensors:
        global_map_payload = agent_ctrl.sensor_data(state, "global_map")
        global_map_sensor = global_map_payload["graph"]
        agent_map.attach_networkx_graph(
            {node_id: global_map_sensor.nodes[node_id] for node_id in global_map_sensor.nodes()},
            {idx: {"source": u, "target": v, **data} for idx, (u, v, data) in enumerate(global_map_sensor.edges(data=True))},
            apsp_lookup=global_map_payload.get("apsp")
        )

    agent_map.update_time(current_time)
    agent_map.update_agent_position(team, agent_ctrl.name, current_pos, current_time)

    agent_sensor = agent_ctrl.sensor_data(state, "agent") if "agent" in sensors else {}
    if enemies := agent_sensor.get("enemies", {}):
        agent_map.update_team_agents(enemy_team, enemies, current_time)
    if teammates := agent_sensor.get("teammates", {}):
        agent_map.update_team_agents(team, teammates, current_time)

    if not agent_map.apsp_lookup:
        agent_map.apsp_lookup = dict(nx.all_pairs_shortest_path_length(agent_map.graph))
    lookup = agent_map.apsp_lookup

    known_defenders = agent_map.get_team_agents(enemy_team)
    def_positions = list({pos for _, pos, _ in known_defenders})

    # ===== SEQUENTIAL CLAIMING (APF LOGIC) =====
    # 1. Reset claimed nodes if this is the start of a new timestep
    tick_tracker = agent_ctrl.get_team("claimed_nodes_time", -1)
    if tick_tracker < current_time:
        claimed_nodes = set()
        agent_ctrl.set_team("claimed_nodes_time", current_time)
    else:
        claimed_nodes = agent_ctrl.get_team("claimed_nodes", set())

    # 2. Evaluate Independent Moves
    target = current_pos
    
    if flags_for_planning and agent_map.graph is not None:
        best_score = float('inf')
        best_moves = []
        my_moves = list(agent_map.graph.neighbors(current_pos)) + [current_pos]

        for move in my_moves:
            score = attacker_stage_cost(move, def_positions, flags_for_planning, lookup)
            
            # Heavy penalty if a teammate has already decided to go here this turn
            if move in claimed_nodes:
                score += 1000.0
                
            if score < best_score:
                best_score = score
                best_moves = [move]
            elif math.isclose(score, best_score, rel_tol=1e-9):
                best_moves.append(move)
                
        if best_moves:
            target = random.choice(best_moves)
            
    elif agent_map.graph is not None:
        # Fallback Exploration (Avoid teammates if possible)
        neighbors = list(agent_map.graph.neighbors(current_pos)) + [current_pos]
        unclaimed = [n for n in neighbors if n not in claimed_nodes]
        target = random.choice(unclaimed) if unclaimed else random.choice(neighbors)

    # 3. Claim the node for subsequent teammates and execute
    claimed_nodes.add(target)
    agent_ctrl.set_team("claimed_nodes", claimed_nodes)
    
    state["action"] = target
    return "executing greedy apf" if target != current_pos else "exploring / holding"

def map_strategy(agent_config):
    return {name: strategy for name in agent_config.keys()}