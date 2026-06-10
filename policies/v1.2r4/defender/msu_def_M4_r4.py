def strategy(state: dict) -> str:
    from typing import Dict, List, Tuple, Any
    import networkx as nx
    import numpy as np
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

    visible_nodes: Dict[int, Any] = agent_ctrl.sensor_data(state, "egocentric_agent_region")["nodes"] if "egocentric_agent_region" in sensors else {}  # Nodes within sensing radius
    visible_edges: List[Any]      = agent_ctrl.sensor_data(state, "egocentric_agent_region")["edges"] if "egocentric_agent_region" in sensors else []   # Edges within sensing radius

    stationary_data: Dict[str, Any] = agent_ctrl.sensor_data(state, "stationary") or {}

    # ===== AGENT MAP (SHARED) =====
    agent_map = agent_ctrl.map  # Team-shared map with positions and graph
    global_map_payload: Dict[str, Any] = agent_ctrl.sensor_data(state, "global_map")
    global_map_sensor: nx.Graph = global_map_payload["graph"]  # Full graph topology from sensor
    global_map_apsp = global_map_payload.get("apsp")

    nodes_data: Dict[int, Dict[str, Any]] = {node_id: global_map_sensor.nodes[node_id] for node_id in global_map_sensor.nodes()}  # Convert graph nodes to dict format

    edges_data: Dict[int, Dict[str, Any]] = {}  # Convert graph edges to dict format
    for idx, (u, v, data) in enumerate(global_map_sensor.edges(data=True)):
        edges_data[idx] = {"source": u, "target": v, **data}

    agent_map.attach_networkx_graph(
        nodes_data,
        edges_data,
        apsp_lookup=global_map_apsp if isinstance(global_map_apsp, dict) else None,
    )  # Initialize map's internal graph (+ APSP if available)
    agent_map.update_time(current_time)  # Sync map time with game time for age tracking

    # Update own position in map (call this every turn to track your position)
    agent_map.update_agent_position(team, agent_ctrl.name, current_pos, current_time)

    # Update teammate positions from custom_team sensor
    agent_map.update_team_agents(team, teammates_sensor, current_time)

    # Update enemy positions from egocentric_agent sensor
    agent_map.update_team_agents(agent_ctrl.enemy_team, nearby_enemies, current_time)

    # How to get all positions of a team from agent map
    teammates_data: List[Tuple[str, int, int]] = agent_map.get_team_agents(team)            # [(name, pos, age)] of teammates
    # How to get a specific agent's position from agent map
    enemy_pos, enemy_age = agent_map.get_agent_position(agent_ctrl.enemy_team, "red_0")  # (position, age_in_timesteps) or (None, None)


    # ===== First Time Step Caching =====
    # shared distance matrix
    dist_lookup : dict = agent_ctrl.get_team('distance_matrix', None)

    if dist_lookup is None:
        dist_lookup = dict(nx.all_pairs_dijkstra_path_length(agent_map.graph.to_undirected(as_view=True)))
        agent_ctrl.set_team('distance_matrix', dist_lookup)

    # undirected version of graph structure 
    u_graph : nx.Graph = agent_ctrl.get_team('undirected_graph', None)
    if u_graph is None:
        u_graph = agent_map.graph.to_undirected()
        agent_ctrl.set_team('undirected_graph', u_graph)
    
    # a modified adjacency matrix used for score propegation 
    adj_mask : np.ndarray = agent_ctrl.get_team('adj_mask', None)
    adj_matrix : np.ndarray = agent_ctrl.get_team('adj_matrix', None)
    if adj_mask is None:
        adj_matrix = nx.to_numpy_array(u_graph)
        adj_mask = np.where(adj_matrix==1, 0.0, np.inf)
        agent_ctrl.set_team('adj_mask', adj_mask)
        agent_ctrl.set_team('adj_matrix', adj_matrix)
    
    degree_vector : np.ndarray = agent_ctrl.get_team('degrees', None)
    if degree_vector is None:
        degrees = [(1 / nx.degree(u_graph, n)) for n in u_graph.nodes]
        degree_vector = np.array(degrees)
        degree_vector = np.c_[degree_vector]
        agent_ctrl.set_team('degrees', degree_vector)



    # ===== Per Iteration Cache ===== 
    attacker_tracking : dict = agent_ctrl.get_team('attacker_tracking', None)
    if attacker_tracking is None:
        attacker_tracking = {}

    # last target information (personal cache)
    previous_node : int = cache.get("previous", None)

    # ===== DECISION LOGIC =====
    enemy_team: str = "blue" if team == "red" else "red"
    THREAT_WEIGHT = 1
    GUARD_WEIGHT = 2
    EPSILON = 1e-2
    SCORE_PROPAGATION_HORIZON = 10
    GROUPING_WEIGHT = 10
    BACKTRACKING_PENALTY = 10



    for enemy_name, location in nearby_enemies.items():
        
        attacker_tracking[enemy_name] = (location, current_time)
    
    for agent_name, node_id in stationary_data.get("enemies", {}).items():
        attacker_tracking[agent_name] = (node_id, current_time)
    
    old_agents = []
    for agent, (_, time_seen) in attacker_tracking.items():
        if (current_time - time_seen) >= 10: # if we havent seen the attacker in 10 time steps stop trcking it
            old_agents.append(agent)
    
    for agent in old_agents:
        attacker_tracking.pop(agent, None)


    agent_ctrl.set_team('attacker_tracking', attacker_tracking)

    current_enemy_locs = [v[0] for v in attacker_tracking.values() if v[1] == current_time ] # enemies we can see now
    past_enemy_locs = [k for k,v in attacker_tracking.items() if v[1] != current_time ]  # enemy NAMES that have moved out of sensor range
    
    teammate_neighbours = set()
    for teammate_loc in nearby_teammates.values():
        neighbourhood = set(u_graph.neighbors(teammate_loc))
        teammate_neighbours.union(neighbourhood)



    current_enemy_scores = []
    guarding_scores = []
    grouping_scores = []

    for node in u_graph.nodes:

        if current_enemy_locs:
            closest_attacker = min(current_enemy_locs, key=lambda x: min(dist_lookup[x][f] for f in real_flags))
            current_enemy_scores.append(dist_lookup[closest_attacker][node])
        else:
            current_enemy_scores.append(0)

        guarding_score = sum(1 / (dist_lookup[node][f]+EPSILON) for f in real_flags)
        guarding_scores.append(guarding_score)

        grouping_scores.append(1 if node in teammate_neighbours else 0)
    
    current_enemy_scores = np.array(current_enemy_scores)
    guarding_scores = np.array(guarding_scores)
    grouping_scores = np.array(grouping_scores)

    # handle past detections
    past_enemy_scores = np.zeros((u_graph.number_of_nodes(), 1))
    if past_enemy_locs:
        
        for past_enemy in past_enemy_locs:
            graph_vec = np.zeros((u_graph.number_of_nodes(), 1))
            location, timestep = attacker_tracking[past_enemy]
            age = current_time - timestep

            graph_vec[location] = min(dist_lookup[location][f] for f in real_flags)

            for _ in range(age):
                graph_vec = adj_matrix @ (degree_vector * graph_vec)
            past_enemy_scores += graph_vec

    past_enemy_scores = past_enemy_scores.T[0]

    enemy_vec = current_enemy_scores + past_enemy_scores

    weight_vec = (THREAT_WEIGHT * enemy_vec) + (GUARD_WEIGHT * guarding_score) + (GROUPING_WEIGHT * grouping_scores)


    for _ in range(SCORE_PROPAGATION_HORIZON):
        pool = np.min(adj_mask + weight_vec.reshape(1, -1), axis=1)
        weight_vec += pool

    if previous_node is not None:
        weight_vec[previous_node] += BACKTRACKING_PENALTY

    neighbour_scores = [(n, float(weight_vec[n])) for n in nx.neighbors(u_graph, current_pos)]
    best_neighbour = min(neighbour_scores, key=lambda x : x[1])

    target , score = best_neighbour

    cache.set("previous", current_pos)
    # ===== OUTPUT =====
    state["action"] = target  # Required: set action for this turn
    return f"moving to {target} with weight {score}" if target != current_pos else "holding position"  # avoid f-strings here if possible — string construction runs every call even when logging is off and will slow down mass eval


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
