def strategy(state):
    from typing import Dict, List, Tuple, Any
    import networkx as nx

    # ===== AGENT CONTROLLER =====
    # DO NOT directly call agent_ctrl methods unless you understand the library
    agent_ctrl = state["agent_controller"]
    current_pos: int = state["curr_pos"]
    current_time: int = state["time"]
    team: str = agent_ctrl.team
    red_payoff: float = state["payoff"]["red"]
    blue_payoff: float = state["payoff"]["blue"]

    # Agent parameters
    speed: float = agent_ctrl.speed
    tagging_radius: float = agent_ctrl.tagging_radius

    # ===== INDIVIDUAL CACHE =====
    cache = agent_ctrl.cache

    last_target: int = cache.get("last_target", None)
    visit_count: int = cache.get("visit_count", 0)

    cache.set("last_position", current_pos)
    cache.set("visit_count", visit_count + 1)
    cache.update(last_time=current_time, patrol_index=cache.get("patrol_index", 0) + 1)

    # ===== TEAM CACHE (SHARED) =====
    team_cache = agent_ctrl.team_cache

    priority_targets: List[int] = team_cache.get("priority_targets", [])

    team_cache.set("last_update", current_time)
    team_cache.update(total_tags=team_cache.get("total_tags", 0), formation="defensive")

    # ===== AGENT MAP (SHARED) =====
    agent_map = agent_ctrl.map
    global_map_sensor: nx.Graph = state["sensor"]["global_map"][1]["graph"]

    nodes_data: Dict[int, Dict[str, Any]] = {node_id: global_map_sensor.nodes[node_id] for node_id in global_map_sensor.nodes()}

    edges_data: Dict[int, Dict[str, Any]] = {}
    for idx, (u, v, data) in enumerate(global_map_sensor.edges(data=True)):
        edges_data[idx] = {"source": u, "target": v, **data}

    agent_map.attach_networkx_graph(nodes_data, edges_data)
    agent_map.update_time(current_time)
    agent_map.update_agent_position(team, agent_ctrl.name, current_pos, current_time)

    # Update teammate positions from custom_team sensor
    if "custom_team" in state["sensor"]:
        teammates_sensor: Dict[str, int] = state["sensor"]["custom_team"][1]
        for teammate_name, teammate_pos in teammates_sensor.items():
            agent_map.update_agent_position(team, teammate_name, teammate_pos, current_time)

    # Update enemy positions from egocentric_agent sensor
    if "egocentric_agent" in state["sensor"]:
        nearby_agents: Dict[str, int] = state["sensor"]["egocentric_agent"][1]
        enemy_team: str = "red" if team == "blue" else "blue"
        for agent_name, node_id in nearby_agents.items():
            if agent_name.startswith(enemy_team):
                agent_map.update_agent_position(enemy_team, agent_name, node_id, current_time)

    teammates_data: List[Tuple[str, int, int]] = agent_map.get_team_agents(team)
    enemy_pos, enemy_age = agent_map.get_agent_position("red", "red_0")

    # ===== SENSORS =====
    sensors: Dict[str, Tuple[Any, Dict[str, Any]]] = state["sensor"]

    if "flag" in sensors:
        real_flags: List[int] = sensors["flag"][1]["real_flags"]  # True flag locations
        fake_flags: List[int] = sensors["flag"][1]["fake_flags"]  # Fake flag locations

    if "custom_team" in sensors:
        teammates_sensor: Dict[str, int] = sensors["custom_team"][1]  # Teammates only

    if "egocentric_agent" in sensors:
        nearby_agents: Dict[str, int] = sensors["egocentric_agent"][1]  # Agents within sensing radius

    if "egocentric_map" in sensors:
        visible_nodes: Dict[int, Any] = sensors["egocentric_map"][1]["nodes"]
        visible_edges: List[Any] = sensors["egocentric_map"][1]["edges"]

    # Stationary sensors (multiple sensors at different positions)
    stationary_detections: List[Dict[str, Any]] = []
    for sensor_name in sensors:
        if sensor_name.startswith("stationary_"):
            sensor_data: Dict[str, Any] = sensors[sensor_name][1]
            fixed_pos: int = sensor_data["fixed_position"]  # Sensor location
            detected_agents: Dict[str, Dict[str, Any]] = sensor_data["detected_agents"]  # {agent_name: {node_id, distance}}
            agent_count: int = sensor_data["agent_count"]  # Number detected
            stationary_detections.append({"position": fixed_pos, "detected": detected_agents, "count": agent_count})

    # ===== DECISION LOGIC =====
    target: int = current_pos
    # ===== OUTPUT =====
    state["action"] = target
    
    return set() # Empty set for code integrity, do not touch


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
