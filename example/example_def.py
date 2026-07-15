def strategy(state: dict) -> str:
    from typing import Dict, List, Tuple, Any
    import networkx as nx

    # ===== AGENT CONTROLLER =====
    # DO NOT directly call agent_ctrl methods unless you understand the library
    agent_ctrl = state["agent_controller"]  # Wrapper containing agent state and methods
    current_pos: int = state["curr_pos"]    # Current node ID where agent is located
    current_time: int = state["time"]       # Current game timestep
    team: str = agent_ctrl.team             # Team identifier ('red' or 'blue')
    red_payoff: float = state["payoff"]["red"]   # Red team accumulated score
    blue_payoff: float = state["payoff"]["blue"]  # Blue team accumulated score

    # Agent parameters
    speed: float = agent_ctrl.speed                    # Movement speed (max nodes per turn)
    tagging_radius: float = agent_ctrl.tagging_radius  # Distance to tag attackers

    # ===== RULE CONFIG =====
    rule_config = state["rule_config"]  # Read-only view of red_global, blue_global, environment
    # Opponent (red/attacker) parameters
    opp_capture_radius: float = rule_config["red_global"]["capture_radius"]  # flag capture range
    opp_sensing_radius: float = rule_config["red_global"]["sensing_radius"]  # red vision radius
    # Environment (stationary sensor network)
    stationary_radius: float = rule_config["environment"]["blue_stationary_sensor_radius"]
    stationary_positions: list = rule_config["environment"]["blue_static_sensor_positions"]

    # ===== INDIVIDUAL CACHE =====
    cache = agent_ctrl.cache  # Per-agent storage, not shared with teammates

    example_val = cache.get("example_key", None)    # get a value (returns default if missing)
    cache.set("example_key", example_val)            # set a value
    cache.update(example_a=0, example_b=1)           # set multiple values at once

    # ===== TEAM CACHE (SHARED) =====
    example_shared = agent_ctrl.get_team("example_key", 0)   # get a shared team value
    agent_ctrl.set_team("example_key", current_time)          # set a shared team value
    agent_ctrl.update_team(example_a=0, example_b="val")      # set multiple shared values at once

    # ===== SENSORS =====
    # All sensor data is read here once; variables are reused in map updates and decision logic below.
    sensors: Dict[str, Tuple[Any, Dict[str, Any]]] = state["sensor"]  # All sensor data

    real_flags: List[int] = agent_ctrl.sensor_data(state, "flag")["real_flags"] if "flag" in sensors else []  # True flag locations
    fake_flags: List[int] = agent_ctrl.sensor_data(state, "flag")["fake_flags"] if "flag" in sensors else []  # Fake flag locations

    teammates_sensor: Dict[str, int] = agent_ctrl.sensor_data(state, "custom_team") if "custom_team" in sensors else {}  # Teammates only {name: node_id}

    nearby_enemies: Dict[str, int]   = agent_ctrl.sensor_data(state, "egocentric_agent")["enemies"]   if "egocentric_agent" in sensors else {}  # Enemy agents within sensing radius
    nearby_teammates: Dict[str, int] = agent_ctrl.sensor_data(state, "egocentric_agent")["teammates"] if "egocentric_agent" in sensors else {}  # Teammates within sensing radius

    egocentric_agent_visibility_graph: Dict[int, Any] = agent_ctrl.sensor_data(state, "egocentric_agent").get("table", {}) if "egocentric_agent" in sensors else {}  # SAME sensor as above, extra key: FULL node -> visible-nodes table for blue_agent_r250 (line-of-sight), static for the whole game
    visible_from_here = egocentric_agent_visibility_graph.get(current_pos, frozenset())  # Visible nodes from curr_pos specifically — look up any node the same way

    # Stationary sensors — pre-consolidated by game engine and pre-filtered by team
    # {"enemies": {name: node_id}, "teammates": {name: node_id}, "detections": [per-sensor entries], "table": full blue_tower_r450 visibility table}
    # Each per-sensor entry: {"fixed_position": int, "detected_agents": {name: {node_id, distance}}, "agent_count": int, "covered_nodes": [...]}
    stationary: Dict[str, Any] = agent_ctrl.sensor_data(state, "stationary") or {"enemies": {}, "teammates": {}, "detections": []}
    stationary_visibility_graph: Dict[int, Any] = stationary.get("table", {})  # A second, DIFFERENT visibility graph (blue_tower_r450) — same "stationary" sensor read below, name matches the sensor it came from
    stationary_enemies: Dict[str, int] = stationary["enemies"]      # {name: node_id} of attackers seen by any stationary sensor
    stationary_detections: List[Dict[str, Any]] = stationary["detections"]  # raw per-sensor entries

    # ===== AGENT MAP (SHARED) =====
    agent_map = agent_ctrl.map  # Team-shared map with positions and graph
    global_map_payload: Dict[str, Any] = agent_ctrl.sensor_data(state, "global_map")
    global_map_sensor: nx.Graph = global_map_payload["graph"]  # Full graph topology from sensor
    global_map_apsp = global_map_payload.get("apsp")

    # The graph is static — attach it once on the first turn and reuse every turn after.
    # agent_map is shared across teammates, so only the first agent to run pays this cost.
    if agent_map.graph is None:
        nodes_data: Dict[int, Dict[str, Any]] = {node_id: global_map_sensor.nodes[node_id] for node_id in global_map_sensor.nodes()}
        edges_data: Dict[int, Dict[str, Any]] = {}
        for idx, (u, v, data) in enumerate(global_map_sensor.edges(data=True)):
            edges_data[idx] = {"source": u, "target": v, **data}
        agent_map.attach_networkx_graph(
            nodes_data,
            edges_data,
            apsp_lookup=global_map_apsp if isinstance(global_map_apsp, dict) else None,
        )
    agent_map.update_time(current_time)  # Sync map time with game time for age tracking

    # Update own position in map (call this every turn to track your position)
    agent_map.update_agent_position(team, agent_ctrl.name, current_pos, current_time)

    # Update teammate positions from custom_team sensor
    agent_map.update_team_agents(team, teammates_sensor, current_time)

    # Update enemy positions from egocentric_agent sensor
    agent_map.update_team_agents(agent_ctrl.enemy_team, nearby_enemies, current_time)

    # How to get all positions of a team from agent map
    teammates_data: List[Tuple[str, int, int]] = agent_map.get_team_agents(team)            # [(name, pos, age)] of teammates

    # ===== DECISION LOGIC =====
    target: int = current_pos  # Default action is to stay at current position
    known_attackers = agent_map.get_team_agents(agent_ctrl.enemy_team)
    if known_attackers:
        _, attacker_pos, _ = min(
            known_attackers,
            key=lambda item: agent_map.shortest_path_length(current_pos, item[1]),
        )
        target = agent_map.shortest_path_step(current_pos, attacker_pos, speed)

    # ===== OUTPUT =====
    state["action"] = target  # Required: set action for this turn
    return f"chasing attacker → node {target}" if target != current_pos else "holding position"  # avoid f-strings here if possible — string construction runs every call even when logging is off and will slow down mass eval


def map_strategy(agent_config):
    """
    Maps each agent to the defender strategy.

    Parameters:
        agent_config (dict): Configuration dictionary for all agents.

    Returns:
        dict: A dictionary mapping agent names to the defender strategy.
    """
    strategies = {}
    for name in agent_config.keys():
        strategies[name] = strategy
    return strategies
