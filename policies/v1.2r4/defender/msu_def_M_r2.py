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

    # Stationary sensors — pre-consolidated and pre-filtered by team
    stationary: Dict[str, Any] = agent_ctrl.sensor_data(state, "stationary") or {"enemies": {}, "teammates": {}, "detections": []}
    stationary_enemies: Dict[str, int] = stationary["enemies"]  # {name: node_id} of attackers

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
    
    cutoff : float = agent_ctrl.get_team('cutoff', None)
    if cutoff is None:
        nf_nodes = set(u_graph.nodes).difference(set(real_flags))
        
        flag_distances = [dist_lookup[flag][not_flag] for flag in real_flags for not_flag in nf_nodes]
        cutoff = sum(flag_distances) / len(flag_distances)
        agent_ctrl.set_team('cutoff', cutoff)


    # ===== Per Iteration Cache ===== 
    moved_to_sensor : bool = cache.get('moved_to_sensor', None)
    if moved_to_sensor is None:
        if current_pos in real_flags:
            moved_to_sensor = True
            cache.set('moved_to_sensor', True)
        else:
            moved_to_sensor = False
            cache.set('moved_to_sensor', False)

    selected_targets : set = agent_ctrl.get_team('selected_targets', None)
    if selected_targets is None:
        selected_targets = set()
        selected_targets.add(max(list(u_graph.nodes))+10)
        agent_ctrl.set_team('selected_targets', selected_targets)
    
    chasing : bool = cache.get('chasing', None)
    if chasing is None:
        chasing = False
        cache.set('chasing', chasing)

    last_target: int = cache.get("last_target", None)

    # ===== DECISION LOGIC =====
    ego_enemies = list(nearby_enemies.values())
    non_ego_enemies = list(stationary_enemies.values())

    # if we are chasing but there are no longer nearby enemies stop chasing
    if chasing and len(ego_enemies) == 0:
        chasing = False
        cache.set('chasing', False)
    
    closest_flag = min(real_flags, key= lambda x : dist_lookup[x][current_pos])
    closest_flag_dist = dist_lookup[closest_flag][current_pos]

    closest_sensor = min(stationary_positions, key= lambda x : dist_lookup[x][current_pos])
    

    # if we have strayed too far from the flags we trigger the move to sensor actions
    if closest_flag_dist >= cutoff:
        moved_to_sensor = False
        cache.set('moved_to_sensor', False)
    
    # if we are not chaing and have specified the move to sensor action 
    # check if we have reached a sensor 
    # if so update the state so a new action can be chosen
    # otherwise take a step to the sensor
    if not moved_to_sensor and not chasing:
        if current_pos in stationary_positions:
            cache.set('moved_to_sensor', True)
            moved_to_sensor = True
            last_target = None
            cache.set('last_target', None)
        else:
            last_target = closest_sensor
            cache.set('last_target', last_target)
            target = agent_map.shortest_path_step(current_pos, last_target, speed)
        
        # action string 
        if closest_flag_dist >= cutoff:
            # print(f'{stationary_positions=}')
            action_string = f'Too far from flags! Moving to sensor at {last_target} current pos: {current_pos} {target=}'
        else:
            action_string = f'Moving to sensor at {last_target}'
    

    # if we are not moving to the sensor and not chasing and enemy
    if moved_to_sensor and not chasing:
        # if we already know our target and haven't yet reached it, take a step twoards it
        if last_target is not None and current_pos != last_target:
            target = agent_map.shortest_path_step(current_pos, last_target, speed)
            action_string = f'Continuing to target node: {last_target}'
        else:
            # determine a new target to move to 
            if (current_pos == last_target) and (current_pos in selected_targets):
                selected_targets.remove(current_pos)

            # look at the set of all nodes in our sensing area
            visible_targets = [(x, dist_lookup[current_pos][x]) for x in visible_nodes.keys()]
            # sort them by distance in node hops
            visible_targets.sort(key= lambda x : x[1])
            max_distance  = visible_targets[-1][1]

            # we will select one of the visibile nodes to move to
            selected = False
            while not selected:
                # we will select one of the nodes max distance away that isnt already the target of another teammate
                candidates = set([n[0] for n in visible_targets if n[1] == max_distance])
                candidates = candidates.difference(selected_targets)
                if len(candidates) != 0:
                    goal_node = random.choice(list(candidates))
                    selected = True
                else:
                    max_distance -= 1

                # an edge case where for some reason we are on a competely unconnected node or all visibile nodes are the target of another agent
                if max_distance == 0: 
                    goal_node = random.choice(visible_targets)
                    selected = True
            cache.set('last_target', goal_node)
            selected_targets.add(goal_node)
            agent_ctrl.set_team('selected_targets', selected_targets)

            target = agent_map.shortest_path_step(current_pos, goal_node, speed)
            action_string = f'New target selected: {last_target}'
        
    # normal behavior is overridden in there is an enemy within the ego sensor range
    if len(ego_enemies):
        # choose a random detected enemy to pursue
        selected_enemy = random.choice(ego_enemies)
        if not chasing:
            cache.set('chasing', True)
        target = agent_map.shortest_path_step(current_pos, selected_enemy, speed)
        action_string = f'Pursuing enemy at location {selected_enemy}'
    else:
        cache.set('chasing', False)
    
    # 
    if len(non_ego_enemies) != 0:
        closest_non_ego_node = min(non_ego_enemies, key= lambda x : dist_lookup[x][current_pos])
        closest_non_ego_dist = dist_lookup[closest_non_ego_node][current_pos]
        # adjust based on our tagging radius
        tag_steps =  closest_non_ego_dist - tagging_radius # number of steps it will take us to tag enemy

        # distance of the closest non ego enemy to the nearest flag 
        non_ego_flag_dist = min(dist_lookup[flag][closest_non_ego_node] for flag in real_flags)
        # adjust based on the capture radius
        capture_steps = non_ego_flag_dist - opp_capture_radius

        if tag_steps < capture_steps:
            if not chasing:
                cache.set('chasing', True)
            target = agent_map.shortest_path_step(current_pos, closest_non_ego_node, speed)
            action_string = f'Pursuing enemy at location {closest_non_ego_node}'
    else:
        cache.set('chasing', False)



    # ===== OUTPUT =====
    state["action"] = target  # Required: set action for this turn
    return action_string  # avoid f-strings here if possible — string construction runs every call even when logging is off and will slow down mass eval


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
