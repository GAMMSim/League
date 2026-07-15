def strategy(state: dict) -> str:
    from typing import Dict, List, Tuple, Any
    import networkx as nx
    import random

    # ===== AGENT CONTROLLER =====
    # DO NOT directly call agent_ctrl methods unless you understand the library
    agent_ctrl = state["agent_controller"]  # Wrapper containing agent state and methods
    current_pos: int = state["curr_pos"]    # Current node ID where agent is located
    current_time: int = state["time"]       # Current game timestep
    team: str = agent_ctrl.team             # Team identifier ('red' or 'blue')
    red_payoff: float = state["payoff"]["red"]   # Red team accumulated score
    blue_payoff: float = state["payoff"]["blue"]  # Blue team accumulated score

    # Agent parameters
    speed: float = agent_ctrl.speed                  # Movement speed (max nodes per turn)
    capture_radius: float = agent_ctrl.capture_radius  # Distance to capture flags

    # ===== RULE CONFIG =====
    rule_config = state["rule_config"]  # Read-only view of red_global, blue_global, environment
    # Opponent (blue/defender) parameters
    opp_tagging_radius: float = rule_config["blue_global"]["tagging_radius"]  # tagging interaction range
    opp_sensing_radius: float = rule_config["blue_global"]["sensing_radius"]  # blue vision radius
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

    candidates: List[int] = agent_ctrl.sensor_data(state, "candidate_flag")["candidate_flags"] if "candidate_flag" in sensors else []  # Possible flag locations

    enemies: Dict[str, int]   = agent_ctrl.sensor_data(state, "agent")["enemies"]   if "agent" in sensors else {}  # Enemy agents in game {name: node_id}
    teammates: Dict[str, int] = agent_ctrl.sensor_data(state, "agent")["teammates"] if "agent" in sensors else {}  # Teammate agents in game {name: node_id}

    detected_flags: List[int] = agent_ctrl.sensor_data(state, "egocentric_flag")["detected_flags"] if "egocentric_flag" in sensors else []  # Real flags within range; flags visible to agent
    flag_count: int           = agent_ctrl.sensor_data(state, "egocentric_flag").get("flag_count", len(detected_flags)) if "egocentric_flag" in sensors else 0   # Number of detected flags (region-sensor payloads don't carry this key; derive it)

    egocentric_flag_visibility_graph: Dict[int, Any] = agent_ctrl.sensor_data(state, "egocentric_flag").get("table", {}) if "egocentric_flag" in sensors else {}  # SAME sensor as above, extra key: FULL node -> visible-nodes table (line-of-sight), static for the whole game
    visible_from_here = egocentric_flag_visibility_graph.get(current_pos, frozenset())  # Visible nodes from curr_pos specifically — look up any node the same way

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

    # Update enemy positions from sensor data (if you have visibility)
    agent_map.update_team_agents(agent_ctrl.enemy_team, enemies, current_time)
    # You can update your teammates similarly via agent_ctrl.sensor_data(state, "agent")["teammates"]

    # How to get all positions of a team from agent map
    teammates_data: List[Tuple[str, int, int]] = agent_map.get_team_agents(team)            # [(name, pos, age)] of teammates

    # ===== DECISION LOGIC =====
    target: int = current_pos  # Default action is to stay at current position

    if detected_flags:
        target = agent_map.shortest_path_step(current_pos, detected_flags[0], speed)  # Move toward first visible flag
    elif agent_map.graph is not None:
        neighbors: List[int] = list(agent_map.graph.neighbors(current_pos))  # Adjacent nodes
        if neighbors:
            target = random.choice(neighbors)  # No flags visible — wander to a neighbor

    # ===== OUTPUT =====
    state["action"] = target  # Required: set action for this turn
    return f"moving to {target}" if target != current_pos else "holding position"  # avoid f-strings here if possible — string construction runs every call even when logging is off and will slow down mass eval


def map_strategy(agent_config):
    """
    Maps each agent to the attacker strategy.

    Parameters:
        agent_config (dict): Configuration dictionary for all agents.

    Returns:
        dict: A dictionary mapping agent names to the attacker strategy.
    """
    strategies = {}
    for name in agent_config.keys():
        strategies[name] = strategy
    return strategies
