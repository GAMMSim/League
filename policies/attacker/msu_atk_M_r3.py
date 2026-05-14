def strategy(state: dict) -> str:
    from typing import Dict, List, Tuple, Any
    import networkx as nx
    import random
    import numpy as np


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
    # example_shared = agent_ctrl.get_team("example_key", 0)   # get a shared team value
    # agent_ctrl.set_team("example_key", current_time)          # set a shared team value
    # agent_ctrl.update_team(example_a=0, example_b="val")      # set multiple shared values at once

    # ===== SENSORS =====
    # All sensor data is read here once; variables are reused in map updates and decision logic below.
    sensors: Dict[str, Tuple[Any, Dict[str, Any]]] = state["sensor"]  # All sensor data

    candidates: List[int] = agent_ctrl.sensor_data(state, "candidate_flag")["candidate_flags"] if "candidate_flag" in sensors else []  # Possible flag locations

    enemies: Dict[str, int]   = agent_ctrl.sensor_data(state, "agent")["enemies"]   if "agent" in sensors else {}  # Enemy agents in game {name: node_id}
    teammates: Dict[str, int] = agent_ctrl.sensor_data(state, "agent")["teammates"] if "agent" in sensors else {}  # Teammate agents in game {name: node_id}

    visible_nodes: Dict[int, Any] = agent_ctrl.sensor_data(state, "egocentric_flag_region")["nodes"] if "egocentric_flag_region" in sensors else {}  # Nodes within sensing radius
    visible_edges: List[Any]      = agent_ctrl.sensor_data(state, "egocentric_flag_region")["edges"] if "egocentric_flag_region" in sensors else []   # Edges within sensing radius

    detected_flags: List[int] = agent_ctrl.sensor_data(state, "egocentric_flag")["detected_flags"] if "egocentric_flag" in sensors else []  # Real flags within range; flags visible to agent
    flag_count: int           = agent_ctrl.sensor_data(state, "egocentric_flag")["flag_count"]      if "egocentric_flag" in sensors else 0   # Number of detected flags

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

    # Update enemy positions from sensor data (if you have visibility)
    agent_map.update_team_agents(agent_ctrl.enemy_team, enemies, current_time)
    # You can update your teammates similarly via agent_ctrl.sensor_data(state, "agent")["teammates"]

    # How to get all positions of a team from agent map
    teammates_data: List[Tuple[str, int, int]] = agent_map.get_team_agents(team)            # [(name, pos, age)] of teammates
    # How to get a specific agent's position from agent map
    enemy_pos, enemy_age = agent_map.get_agent_position(agent_ctrl.enemy_team, "blue_0")  # (position, age_in_timesteps) or (None, None)



    # ===== First Time Step Caching =====
    # shared list of candidate flags that will be updated
    possible_targets : list[int] = agent_ctrl.get_team('possible_targets', None)
    if possible_targets is None:
        possible_targets = [f for f in candidates]
        agent_ctrl.set_team('possible_targets', possible_targets)
    
    # shared path and distance matricies
    dist_lookup : dict = agent_ctrl.get_team('distance_matrix', None)
    path_lookup : dict = agent_ctrl.get_team('path_matrix', None)

    if dist_lookup is None:
        dist_lookup = dict(nx.all_pairs_dijkstra_path_length(agent_map.graph.to_undirected(as_view=True)))
        agent_ctrl.set_team('distance_matrix', dist_lookup)
    
    if path_lookup is None:
        path_lookup = dict(nx.all_pairs_shortest_path(agent_map.graph.to_undirected(as_view=True)))
        agent_ctrl.set_team('path_matrix', path_lookup)
    
    # undirected version of graph structure 
    u_graph : nx.Graph = agent_ctrl.get_team('undirected_graph', None)
    if u_graph is None:
        u_graph = agent_map.graph.to_undirected()
        agent_ctrl.set_team('undirected_graph', u_graph)
    
    # a modified adjacency matrix used for score propegation 
    adj_mask : np.ndarray = agent_ctrl.get_team('adj_mask', None)
    if adj_mask is None:
        adj_matrix = nx.to_numpy_array(u_graph)
        adj_mask = np.where(adj_matrix==1, 0.0, np.inf)
        agent_ctrl.set_team('adj_mask', adj_mask)

    # ===== Per Iteration Caching ===== 
    # update the list of possible flag locations based on current sensor information
    num_possible_targets = len(possible_targets)
    for visible_node in visible_nodes.keys():
        # if the visible node is candidate flag, but is not detected as a flag, and is target we are still tracking remove it from consideration
        if (visible_node in candidates) and (visible_node not in detected_flags) and (visible_node in possible_targets):
            possible_targets.remove(visible_node)
    # if we removed a candidate flag update the list of possible targets for the entire team
    if len(possible_targets) < num_possible_targets:
        agent_ctrl.set_team('possible_targets', possible_targets)
    
    # last target information (personal cache)
    last_taget : int = cache.get("last_target", None)
    # cache.set("last_position", current_pos)

    # ===== PARAMETERS =====
    EPSILON = 1e-2 # division by zero protection
    GOAL_WEIGHT = 2 # weigh associated with moving twoards a flag and exploration
    RISK_WEIGHT = 3 # weight associated with avoiding tagging
    SCORE_PROPAGATION_HORIZON = 10

    # ===== DECISION LOGIC =====
    target: int = current_pos  # Default action is to stay at current position
    
    # State information
    enemy_locations = list(enemies.values())
    remaining_candidates = set(candidates).difference(set(possible_targets))
    graph_nodes = list(u_graph.nodes)



    # compute scores for each graph node
    candidate_scores = [] # score for investigating an unconfirmed candidate
    target_scores = [] # score for targeting a possbile target
    risk_scores = [] # score for risk of node ie closeness to defense

    # score calculation
    for node in graph_nodes:

        # min distance to a remaining candidate
        if len(remaining_candidates):
            min_cand_distance = min(dist_lookup[node][f] for f in remaining_candidates)
        else:
            min_cand_distance = 0
        
        # min distance to a possible target
        if len(possible_targets):
            min_target_distance = min(dist_lookup[node][f] for f in possible_targets)
        else:
            min_target_distance = 0
        
        # risk score based on distance of node from all enemies
        risk = sum(1 / (dist_lookup[node][d]+EPSILON) for d in enemy_locations)

        candidate_scores.append(min_cand_distance)
        target_scores.append(min_target_distance)
        risk_scores.append(risk)
    
    # cast to numpy
    candidate_scores = np.array(candidate_scores)
    target_scores = np.array(target_scores)
    risk_scores = np.array(risk_scores)

    # calculate explore vs target fractions 
    cand_frac = len(remaining_candidates) / len(candidates)
    target_frac = len(possible_targets) / len(candidates)

    # calculate weighed goal and risk scores for all nodes
    goal_weight_vector = GOAL_WEIGHT * ((cand_frac * candidate_scores) + (target_frac * target_scores))
    risk_weight_vector = RISK_WEIGHT * risk_scores
    weighted_scores = goal_weight_vector + risk_weight_vector

    # propegate scores between neighbours across the specified horizon
    for _ in range(SCORE_PROPAGATION_HORIZON):
        pool = np.min(adj_mask + weighted_scores.reshape(1,-1), axis=1)
        weighted_scores += pool
    
    # find the score for each neighbour node
    neighbour_scores = [(n, float(weighted_scores[n])) for n in nx.neighbors(u_graph, current_pos)]
    best_neighbour = min(neighbour_scores, key=lambda x : x[1])

    target , score = best_neighbour

    print(f'{nx.neighbors(u_graph, current_pos)} {target=} {score=}')
    
    # ===== OUTPUT =====
    state["action"] = target  # Required: set action for this turn
    return f"moving to {target} with weight {score}" if target != current_pos else "holding position"  # avoid f-strings here if possible — string construction runs every call even when logging is off and will slow down mass eval


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
