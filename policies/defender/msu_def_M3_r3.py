def strategy(state):
    from typing import Dict, List, Tuple, Any
    import networkx as nx
    import random
    import numpy as np

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

    # ===============
    # ignore node data and edge data this is boilerplate basically that gets unpacked elsewhere
    nodes_data: Dict[int, Dict[str, Any]] = {node_id: global_map_sensor.nodes[node_id] for node_id in global_map_sensor.nodes()}

    edges_data: Dict[int, Dict[str, Any]] = {}
    for idx, (u, v, data) in enumerate(global_map_sensor.edges(data=True)):
        edges_data[idx] = {"source": u, "target": v, **data}

    agent_map.attach_networkx_graph(nodes_data, edges_data)
    agent_map.update_time(current_time)
    agent_map.update_agent_position(team, agent_ctrl.name, current_pos, current_time)
    # ==============

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

    if "custom_team" in sensors: # only gives locations of defenders
        teammates_sensor: Dict[str, int] = sensors["custom_team"][1]  # Teammates only

    if "egocentric_agent" in sensors: # gives all agents need to filter 
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

    # ===== CACHE MODIFICATIONS =====
    graph = agent_map.graph
    dist_lookup : dict = team_cache.get('distance_matrix', None)
    
    if dist_lookup is None:
        dist_lookup = dict(nx.all_pairs_dijkstra_path_length(graph.to_undirected(as_view=True)))
        team_cache.set('distance_matrix', dist_lookup)
    
    # cache a graph and adj_mask if none
    u_graph : nx.Graph = team_cache.get('u-graph', None)
    adj_mask : np.ndarray = team_cache.get('adj-mask', None)
    adj : np.ndarray = team_cache.get('adj', None)
    if u_graph is None:
        u_graph = graph.to_undirected()
        team_cache.set('u-graph', u_graph)
    if adj_mask is None:
        adj = nx.to_numpy_array(u_graph)
        adj_mask = np.where(adj==1, 0., np.inf) 
        team_cache.set('adj-mask', adj_mask)
        team_cache.set('adj', adj)

        


    # ===== DECISION LOGIC =====
    enemy_team: str = "blue" if team == "red" else "red"

    ego_enemies = [v for k, v in nearby_agents.items() if k.startswith(enemy_team)]
    static_sensors = [x["position"] for x in stationary_detections]
    non_ego_enemies = []
    for sensor_dict in stationary_detections:
        detected = sensor_dict['detected']
        if len(detected):
            non_ego_enemies += [v['node_id'] for k, v in detected.items() if k.startswith(enemy_team)]
    
    # get old positions 
    old_ego : Dict[str , int] = team_cache.get('old-ego', None)
    old_static : Dict[str , int] = team_cache.get('old-static', None)
    team_cache.set('old-ego', {k : v for k, v in nearby_agents.items() if k.startswith(enemy_team)} )
    team_cache.set('old-static', {k : v['node_id'] for k, v in detected.items() if k.startswith(enemy_team)})
    

    vanishing_attacker_locations = []
    if old_ego is not None:
        current_names = [k for k in nearby_agents.keys() if k.startswith(enemy_team)] 
        for sensor_dict in stationary_detections:
            detected = sensor_dict['detected']
            if len(detected):
                current_names += [k for k in detected.keys() if k.startswith(enemy_team)]
        old_names = [x for x in old_ego.keys()] + [x for x in old_static.keys()]

        # nodes where we saw an attacker last step but not this step
        
        vanishing_names = [name for name in old_names if name not in current_names]

        # if there are agents we no longer see ... we will assume they arent captured 
        if len(vanishing_names):
            for name in vanishing_names:
                if name in old_ego.keys():
                    vanishing_attacker_locations.append(old_ego[name])
                else:
                    vanishing_attacker_locations.append(old_static[name])
    
    threat_scores, guard_scores, vanishing_scores = [], [], []
    attacker_locs = set(ego_enemies + non_ego_enemies)

    epsilon = 1e-2
    threat_weight = 1
    guard_weight = 1

    for node in u_graph.nodes:
        
        if len(attacker_locs):
            closest_attacker = min(attacker_locs, key=lambda x: min(dist_lookup[x][f] for f in real_flags))
            threat_scores.append(dist_lookup[closest_attacker][node])
        else:
            threat_scores.append(0)
        
        guarding_score = sum(1 / (dist_lookup[node][f]+epsilon) for f in real_flags + fake_flags)
        guard_scores.append(guarding_score)

        if len(vanishing_attacker_locations):
            closest_van = min(vanishing_attacker_locations, key=lambda x: min(dist_lookup[x][f] for f in real_flags))
            vanishing_scores.append(dist_lookup[closest_van][node])
        else:
            vanishing_scores.append(0)

    
    vanishing_scores = np.array(vanishing_scores)
    guard_scores = np.array(guard_scores)
    threat_scores = np.array(threat_scores)


    diffused_vans = adj @ np.c_[vanishing_scores]
    diffused_vans = diffused_vans.T[0]

    attacker_vec = threat_scores + diffused_vans

    weight_vec = threat_weight*attacker_vec - guard_weight*guard_scores

    HORIZON = 10

    for _ in range(HORIZON):
        pool = np.min(adj_mask + weight_vec.reshape(1, -1), axis=1)
        weight_vec += pool
    
    neighbour_scores = [(n, weight_vec[n]) for n in nx.neighbors(u_graph, current_pos)]
    best_neighbour = min(neighbour_scores, key=lambda x: x[1])
    target = best_neighbour[0]
    
    # print(f'{chasing=} {moved_to_sensor=} {target=}')
    
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
