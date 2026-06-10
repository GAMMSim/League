import math
import random
import networkx as nx
from typing import Dict, List, Tuple, Any

# ---------- Cost Function ----------
def defender_stage_cost(pos: int, attackers: List[int], flags: List[int], lookup: Dict[int, Dict[int, float]], w_intercept: float = 1.0, w_position: float = 5.0, epsilon: float = 1e-3) -> float:
    """Immediate cost for a single defender. Lower is better."""
    if not attackers or not flags:
        return 0.0
        
    attacker_threat = min(attackers, key=lambda a: min(lookup[a][f] for f in flags))
    intercept_cost = lookup[pos][attacker_threat]
    flag_guard_score = sum(1.0 / (lookup[pos][f] + epsilon) for f in flags)
    
    return w_intercept * intercept_cost - w_position * flag_guard_score

# ---------- Main Strategy ----------
def strategy(state: dict) -> str:
    # ===== AGENT CONTROLLER =====
    agent_ctrl = state["agent_controller"]
    current_pos: int = state["curr_pos"]
    current_time: int = state["time"]
    team: str = agent_ctrl.team
    enemy_team: str = agent_ctrl.enemy_team

    # ===== SENSORS & MAP SYNC =====
    sensors: Dict[str, Tuple[Any, Dict[str, Any]]] = state.get("sensor", {})
    real_flags: List[int] = agent_ctrl.sensor_data(state, "flag").get("real_flags", []) if "flag" in sensors else []
    
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

    # Compile visible enemies
    nearby_enemies = agent_ctrl.sensor_data(state, "egocentric_agent").get("enemies", {}) if "egocentric_agent" in sensors else {}
    stationary = agent_ctrl.sensor_data(state, "stationary") or {"enemies": {}}
    all_visible_enemies = {**nearby_enemies, **stationary.get("enemies", {})}
    
    if all_visible_enemies:
        agent_map.update_team_agents(enemy_team, all_visible_enemies, current_time)
    if teammates_sensor := agent_ctrl.sensor_data(state, "custom_team") if "custom_team" in sensors else {}:
        agent_map.update_team_agents(team, teammates_sensor, current_time)

    if not agent_map.apsp_lookup:
        agent_map.apsp_lookup = dict(nx.all_pairs_shortest_path_length(agent_map.graph))
    lookup = agent_map.apsp_lookup

    known_attackers = agent_map.get_team_agents(enemy_team)
    att_positions = list({pos for _, pos, _ in known_attackers})

    # ===== SEQUENTIAL CLAIMING (APF LOGIC) =====
    # 1. Reset claimed nodes at the start of the tick
    tick_tracker = agent_ctrl.get_team("claimed_nodes_time", -1)
    if tick_tracker < current_time:
        claimed_nodes = set()
        agent_ctrl.set_team("claimed_nodes_time", current_time)
    else:
        claimed_nodes = agent_ctrl.get_team("claimed_nodes", set())

    # 2. Evaluate Independent Moves
    target = current_pos

    if real_flags and agent_map.graph is not None:
        best_score = float('inf')
        best_moves = []
        my_moves = list(agent_map.graph.neighbors(current_pos)) + [current_pos]

        for move in my_moves:
            score = defender_stage_cost(move, att_positions, real_flags, lookup)
            
            # Massive penalty for stepping on a node another defender is moving to
            if move in claimed_nodes:
                score += 1000.0

            if score < best_score:
                best_score = score
                best_moves = [move]
            elif math.isclose(score, best_score, rel_tol=1e-9):
                best_moves.append(move)

        if best_moves:
            target = random.choice(best_moves)

    # 3. Claim the node and execute
    claimed_nodes.add(target)
    agent_ctrl.set_team("claimed_nodes", claimed_nodes)
    
    state["action"] = target
    return "executing greedy intercept" if target != current_pos else "holding position"

def map_strategy(agent_config):
    return {name: strategy for name in agent_config.keys()}