def strategy(state):
    from typing import Dict, List, Tuple, Any
    import networkx as nx
    import random
    import numpy as np

    # ===== AGENT CONTROLLER =====
    # DO NOT directly call agent_ctrl methods unless you understand the library
    agent_ctrl = state["agent_controller"]  # Wrapper containing agent state and methods
    current_pos: int = state["curr_pos"]  # Current node ID where agent is located
    current_time: int = state["time"]  # Current game timestep
    team: str = agent_ctrl.team  # Team identifier ('red' or 'blue')
    red_payoff: float = state['payoff']['red']  # Red team accumulated score
    blue_payoff: float = state['payoff']['blue']  # Blue team accumulated score
    
    # Agent parameters
    speed: float = agent_ctrl.speed  # Movement speed (max nodes per turn)
    capture_radius: float = agent_ctrl.capture_radius  # Distance to capture flags

    # ===== INDIVIDUAL CACHE =====
    cache = agent_ctrl.cache  # Per-agent storage, not shared with teammates

    last_target: int = cache.get("last_target", None)  # Previously chosen target node
    visit_count: int = cache.get("visit_count", 0)  # Number of strategy calls for this agent

    cache.set("last_position", current_pos)  # Store current position for next turn
    cache.set("visit_count", visit_count + 1)  # Increment visit counter
    cache.update(last_time=current_time, patrol_index=cache.get("patrol_index", 0) + 1)  # Batch update multiple cache values
    

    # ===== TEAM CACHE (SHARED) =====
    team_cache = agent_ctrl.team_cache  # Shared storage across all teammates

    priority_targets: List[int] = team_cache.get("priority_targets", [])  # How to get data from team cache

    team_cache.set("last_update", current_time)  # Track when team cache was last modified
    team_cache.update(total_captures=team_cache.get("total_captures", 0), formation="spread")  # Update team-wide statistics

    

    

    # ===== AGENT MAP (SHARED) =====
    agent_map = agent_ctrl.map  # Team-shared map with positions and graph
    global_map_sensor: nx.Graph = state["sensor"]["global_map"][1]["graph"]  # Full graph topology from sensor

    nodes_data: Dict[int, Dict[str, Any]] = {node_id: global_map_sensor.nodes[node_id] for node_id in global_map_sensor.nodes()}  # Convert graph nodes to dict format # coords stored here also in agent map

    edges_data: Dict[int, Dict[str, Any]] = {}  # Convert graph edges to dict format
    for idx, (u, v, data) in enumerate(global_map_sensor.edges(data=True)):
        edges_data[idx] = {"source": u, "target": v, **data}

    agent_map.attach_networkx_graph(nodes_data, edges_data)  # Initialize map's internal graph
    agent_map.update_time(current_time)  # Sync map time with game time for age tracking

    # Update own position in map (call this every turn to track your position)
    agent_map.update_agent_position(team, agent_ctrl.name, current_pos, current_time)

    # Update enemy positions from sensor data (if you have visibility)
    if "agent" in state["sensor"]:
        all_agents: Dict[str, int] = state["sensor"]["agent"][1]
        enemy_team: str = "blue" if team == "red" else "red"
        for agent_name, node_id in all_agents.items():
            if agent_name.startswith(enemy_team):
                agent_map.update_agent_position(enemy_team, agent_name, node_id, current_time)
    # You can update your teammates similarly if desired

    # How to get all position of a team from agent map
    teammates_data: List[Tuple[str, int, int]] = agent_map.get_team_agents(team)  # [(name, pos, age)] of teammates
    # How to get a specific agent's position from agent map
    enemy_pos, enemy_age = agent_map.get_agent_position("blue", "blue_0")  # (position, age_in_timesteps) or (None, None)

    # ===== SENSORS =====
    sensors: Dict[str, Tuple[Any, Dict[str, Any]]] = state["sensor"]  # All sensor data

    if "candidate_flag" in sensors:
        candidates: List[int] = sensors["candidate_flag"][1]["candidate_flags"]  # Possible flag locations

    if "agent" in sensors:
        all_agents: Dict[str, int] = sensors["agent"][1]  # All agents in game {name: node_id}

    if "egocentric_map" in sensors:
        visible_nodes: Dict[int, Any] = sensors["egocentric_map"][1]["nodes"]  # Nodes within sensing radius
        visible_edges: List[Any] = sensors["egocentric_map"][1]["edges"]  # Edges within sensing radius

    if "egocentric_flag" in sensors:
        detected: List[int] = sensors["egocentric_flag"][1]["detected_flags"]  # Real flags within range
        count: int = sensors["egocentric_flag"][1]["flag_count"]  # Number of detected flags
        flags: List[int] = sensors["egocentric_flag"][1]["detected_flags"]

    
    # ===== CACHE MODIFICATIONS =====
    possible_targets : list = team_cache.get('possible', None)
    if possible_targets is None:
        possible_targets = [f  for f in candidates]
        team_cache.set('possible', possible_targets)
    
    # update possible flags 
    poss_len = len(possible_targets)
    for n in visible_nodes:
        if (n in candidates) and (n not in detected) and (n in possible_targets):
            possible_targets.remove(n)
    # if we lost a candidate, update it for everyone
    if len(possible_targets) < poss_len:
        # team_cache.update(possible=possible_targets)
        team_cache.set('possible', possible_targets)

    # cache a distance matrix and path matrix if one does not exist
    graph = agent_map.graph
    dist_lookup : dict = team_cache.get('distance_matrix', None)
    path_lookup : dict = team_cache.get('path_matrix', None)
    if dist_lookup is None:
        dist_lookup = dict(nx.all_pairs_dijkstra_path_length(graph.to_undirected(as_view=True)))
        team_cache.set('distance_matrix', dist_lookup)
    if path_lookup is None:
        path_lookup = dict(nx.all_pairs_shortest_path(graph.to_undirected(as_view=True)))
        team_cache.set('path_matrix', path_lookup)

    # cache a graph and adj_mask if none
    u_graph : nx.Graph = team_cache.get('u-graph', None)
    adj_mask : np.ndarray = team_cache.get('adj-mask', None)
    if u_graph is None:
        u_graph = graph.to_undirected()
        team_cache.set('u-graph', u_graph)
    if adj_mask is None:
        adj = nx.to_numpy_array(u_graph)
        adj_mask = np.where(adj==1, 0., np.inf) 
        team_cache.set('adj-mask', adj_mask)

    # ===== DECISION LOGIC =====
    target: int = current_pos  # Default action is to stay at current position

    # if we dont have a target or we found our target to be fake 
    if (last_target is None) or (last_target not in possible_targets):
        # find the closest flag
        closest_flag, f_dist = None, float('inf')
        for flag in possible_targets:
            dist = dist_lookup[current_pos][flag]
            if dist < f_dist:
                f_dist = dist
                closest_flag = flag
            elif dist == f_dist:
                closest_flag = random.choice([closest_flag, flag])
        
        last_target = closest_flag
        cache.set("last_position", last_target)
        # we use last position to hold our current flag value 
    






    candidate_scores, flag_scores, risk_scores = [], [], []
    # u_graph : nx.Graph = graph.to_undirected(as_view=True)
    defender_postions = [def_loc for (_, def_loc, _) in agent_map.get_team_agents(enemy_team)]
    remaining_candidates = set(candidates).difference(set(possible_targets))
    graph_nodes = list(u_graph.nodes)
    epsilon = 1e-2
    goal_weight = 2
    risk_weight = 3


    for node in graph_nodes:

        # calculate a score for the min distance to all remaining candidate flags
        if len(remaining_candidates):
            min_cand_dist = min(dist_lookup[node][f] for f in remaining_candidates)
        else:
            min_cand_dist = 0
        

        # calculate a score for the min distance to all known real flags. 
        # the length here might be 0 so account for that
        if len(possible_targets):
            min_flag_dist = min(dist_lookup[node][f] for f in possible_targets)
        else:
            min_flag_dist = 0

        # calculate a risk score for the node
        risk = sum(1 / (dist_lookup[node][d]+epsilon) for d in defender_postions)

        candidate_scores.append(min_cand_dist)
        flag_scores.append(min_flag_dist)
        risk_scores.append(risk)
    
    candidate_scores = np.array(candidate_scores)
    flag_scores = np.array(flag_scores)
    risk_scores = np.array(risk_scores)

    cand_frac = len(remaining_candidates) / len(candidates)
    flag_frac = len(possible_targets) / len(candidates)

    goal_vec = goal_weight*((cand_frac*candidate_scores) + (flag_frac*flag_scores))
    weight_vec = goal_vec + risk_weight*risk_scores

    
    
    

    HORIZON = 10

    for _ in range(HORIZON):
        pool = np.min(adj_mask + weight_vec.reshape(1, -1), axis=1)
        weight_vec += pool



    neighbour_scores = [(n, weight_vec[n]) for n in nx.neighbors(u_graph, current_pos)]
    # print(neighbour_scores)
    best_neighbour = min(neighbour_scores, key=lambda x: x[1])

    target = best_neighbour[0]


    # ===== OUTPUT =====
    state["action"] = target  # Required: set target node for this turn
    
    # Return discovered flags for potential reward calculation
    return set(flags)



def map_strategy(agent_config):
    """
    Maps each agent to the 'do nothing' strategy.

    Parameters:
        agent_config (dict): Configuration dictionary for all agents.

    Returns:
        dict: A dictionary mapping agent names to the stationary strategy.
    """
    strategies = {}
    for name in agent_config.keys():
        strategies[name] = strategy
    return strategies
