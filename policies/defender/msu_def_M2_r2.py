def strategy(state):
    from typing import Dict, List, Tuple, Any
    import networkx as nx
    import random
    from lib.core.apsp_cache import get_apsp_length_cache

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
    global_map_payload: Dict[str, Any] = state["sensor"]["global_map"][1]
    global_map_sensor: nx.Graph = global_map_payload["graph"]
    global_map_apsp = global_map_payload.get("apsp")

    # ===============
    # ignore node data and edge data this is boilerplate basically that gets unpacked elsewhere
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
    if agent_map.graph is not None and agent_map.apsp_lookup is None:
        if isinstance(global_map_apsp, dict):
            agent_map.set_apsp_lookup(global_map_apsp)
        else:
            agent_map.set_apsp_lookup(get_apsp_length_cache(agent_map.graph))
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
    graph = agent_map.graph if agent_map.graph is not None else global_map_sensor
    dist_lookup = (
        agent_map.apsp_lookup
        if isinstance(agent_map.apsp_lookup, dict)
        else global_map_apsp if isinstance(global_map_apsp, dict)
        else get_apsp_length_cache(graph)
    )
    
    moved_to_sensor : bool = cache.get('moved_to_sensor', None)
    if moved_to_sensor is None:
        if current_pos in real_flags:
            moved_to_sensor = True
            cache.set('moved_to_sensor', True)
        else:
            moved_to_sensor = False
            cache.set('moved_to_sensor', False)
    
    selected_targets : set = team_cache.get('selected_targets', None)
    if selected_targets is None:
        selected_targets = set()
        selected_targets.add(max(list(graph.nodes))+10)
        team_cache.set('selected_targets', selected_targets)
    
    cutoff : float = team_cache.get('cutoff', None)
    if cutoff == None:
        nf_nodes = set(graph.nodes).difference(set(real_flags))
        flag_distances = []
        max_nf_dist = 0
        for flag in real_flags:
            for nf in nf_nodes:
                d = dist_lookup[flag][nf]
                flag_distances.append(d)
                max_nf_dist = max(max_nf_dist, d)
        avg = sum(flag_distances) / len(flag_distances)
        cutoff = ((max_nf_dist - avg) * 0) + avg
        team_cache.set('cutoff', cutoff)

    chasing : bool = cache.get('chasing', None)
    if chasing is None:
        chasing = False
        cache.set('chasing', chasing)


        


    # ===== DECISION LOGIC =====
    enemy_team: str = "blue" if team == "red" else "red"

    ego_enemies = [v for k, v in nearby_agents.items() if k.startswith(enemy_team)]
    static_sensors = [x["position"] for x in stationary_detections]
    non_ego_enemies = []
    for sensor_dict in stationary_detections:
        detected = sensor_dict['detected']
        if len(detected):
            non_ego_enemies += [v['node_id'] for k, v in detected.items() if k.startswith(enemy_team)]
    # print(non_ego_enemies)
    

    if chasing and len(ego_enemies) == 0:
        chasing = False
        cache.set('chasing', False)

    nearest_flag, nearest_flag_distance = None, float('inf')
    for f in real_flags:
        f_dist = dist_lookup[f][current_pos]
        if f_dist < nearest_flag_distance:
            nearest_flag = f
            nearest_flag_distance = f_dist

    nearest_sensor, nearest_sensor_distance = None, float('inf')
    for s in static_sensors:
        s_dist = dist_lookup[s][current_pos]
        if s_dist < nearest_sensor_distance:
            nearest_sensor_distance = s_dist
            nearest_sensor = s
    
    if nearest_flag_distance >= cutoff:
        moved_to_sensor = False
        cache.set('moved_to_sensor', False)
    

    if not moved_to_sensor and not chasing:
        if current_pos in static_sensors:
            cache.set('moved_to_sensor', True)
            moved_to_sensor = True
            last_target = None
            cache.set('last_target', None)
        else:
            if last_target is None:
                # nearest_flag, nearest_flag_distance = None, float('inf')
                # for f in real_flags:
                #     f_dist = dist_lookup[f][current_pos]
                #     if f_dist < nearest_flag_distance:
                #         nearest_flag = f
                #         nearest_flag_distance = f_dist
                last_target = nearest_sensor
                cache.set('last_target', last_target)
            target = agent_map.shortest_path_step(current_pos, last_target, speed)
    
    if moved_to_sensor and not chasing:
        # if we have a target in mind, and havent reached it, go there
        if last_target is not None and current_pos != last_target:
            target = agent_map.shortest_path_step(current_pos, last_target, speed)
        else:
            # otherwise find a new target to travel to
            if current_pos == last_target:
                if current_pos in selected_targets:
                    selected_targets.remove(current_pos)
            
            visible_targets = [(x, dist_lookup[current_pos][x]) for x in visible_nodes.keys()]
            visible_targets.sort(key= lambda x : x[1])
            max_distance  = visible_targets[-1][1]

            selected = False
            while not selected:
                if max_distance == 0:
                    goal_node = random.choice(visible_targets)
                    selected = True
                candidates = set([n[0] for n in visible_targets if n[1] == max_distance])
                candidates = candidates.difference(selected_targets)
                if len(candidates) != 0:
                    goal_node = random.choice(list(candidates))
                    selected = True
                else:
                    max_distance -= 1
                if max_distance == 0:
                    goal_node = random.choice(visible_targets)
                    selected = True
            cache.set('last_target', goal_node)
            selected_targets.add(goal_node)
            team_cache.set('selected_targets', selected_targets)

            target = agent_map.shortest_path_step(current_pos, goal_node, speed)
    
    # so far defined normal movement in the absence of attackers 
    # if we do observe attackers in our ego sensor ... move twoards them
    
    # if there is an enemy nearby move twoards the closest one
    if len(ego_enemies) != 0:
        # print('chasing')
        # closest_enemy, closest_enemy_dist = None, float('inf')
        # for en in ego_enemies:
        #     if dist_lookup[current_pos][en] < closest_enemy_dist:
        #         closest_enemy = en
        #         closest_enemy_dist = dist_lookup[current_pos][en]
        selected_enemy = random.choice(ego_enemies)
        # cache.set('last_target', selected_enemy)
        if not chasing:
            cache.set('chasing', True)
        target = agent_map.shortest_path_step(current_pos, selected_enemy, speed)
    else:
        cache.set('chasing', False)

    if len(non_ego_enemies) != 0:
        distance_list = [dist_lookup[x][current_pos] for x in non_ego_enemies]
        closest_dist = min(distance_list)
        en_node = non_ego_enemies[distance_list.index(closest_dist)]

        en_flag_dists = [dist_lookup[x][en_node] for x in real_flags]
        if min(en_flag_dists) > closest_dist:
            if not chasing:
                cache.set('chasing', True)
            target = agent_map.shortest_path_step(current_pos, en_node, speed)
    else:
        cache.set('chasing', False)
            

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
