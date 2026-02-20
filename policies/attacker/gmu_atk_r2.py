def attacker_strategy(state):
    """
    Attacker strategy implementation based on Strategy Proposal V1.3
    
    Mode Priority:
    1. Flag Capture (highest): Direct capture when safe
    2. Emergency Flee: Run away if defender is within dynamic tolerance
    3. Prep Stage: Position for optimal capture
    4. Exploration: One agent explores unknown flags
    """
    from typing import Dict, List, Tuple, Set, Optional
    import networkx as nx
    from lib.core.apsp_cache import get_apsp_length_cache, get_cached_distance

    # ===== CONFIGURATION (HARDCODED VALUES) =====
    # Assumptions about enemy capabilities (used if not found in cache)
    DEFAULT_DEFENDER_SPEED = 1
    DEFAULT_DEFENDER_TAG_RADIUS = 1

    # Sensing assumptions for calculating safe zones (fallback only)
    DEFAULT_STATIONARY_SENSE_RADIUS = 450
    DEFAULT_DEFENDER_SENSE_RADIUS = 250
    
    # Flee Logic: Scoring weights for movement selection
    FLEE_SAFE_ZONE_BONUS = 1000       # Bonus for moving into H_sense_c
    FLEE_DIST_WEIGHT = 10             # Multiplier for distance to defender
    
    # Flee Logic: Triggers
    TEAMING_CHECK_RADIUS = 5          # Radius to check for multiple defenders (teaming)
    BASE_FLEE_TOLERANCE = 2           # Base safety margin (Tag radius + Buffer)
    
    # Explorer Logic
    EXPLORER_SAFETY_BUFFER = 2        # Extra buffer added to tag radius for path safety
    
    # ============================================

    # ===== INITIALIZATION =====
    agent_ctrl = state["agent_controller"]
    current_pos: int = state["curr_pos"]
    current_time: int = state["time"]
    team: str = agent_ctrl.team
    agent_name: str = agent_ctrl.name
    
    # Agent parameters
    speed: int = int(agent_ctrl.speed)
    capture_radius: int = int(agent_ctrl.capture_radius)
    sense_radius = agent_ctrl.sensing_radius
    
    # Caches
    cache = agent_ctrl.cache
    team_cache = agent_ctrl.team_cache
    agent_map = agent_ctrl.map
    
    # Sensors
    sensors = state["sensor"]
    global_map_payload = sensors["global_map"][1]
    global_map_sensor: nx.Graph = global_map_payload["graph"]
    global_map_apsp = global_map_payload.get("apsp")
    
    # ===== INITIALIZE MAP/APSP (ONLY AT t=0) =====
    if agent_map.graph is None:
        nodes_data = {node_id: global_map_sensor.nodes[node_id] for node_id in global_map_sensor.nodes()}
        edges_data = {}
        for idx, (u, v, data) in enumerate(global_map_sensor.edges(data=True)):
            edges_data[idx] = {"source": u, "target": v, **data}
        agent_map.attach_networkx_graph(
            nodes_data,
            edges_data,
            apsp_lookup=global_map_apsp if isinstance(global_map_apsp, dict) else None,
        )

    if agent_map.graph is not None and agent_map.apsp_lookup is None:
        agent_map.set_apsp_lookup(get_apsp_length_cache(agent_map.graph))

    apsp = agent_map.apsp_lookup if isinstance(agent_map.apsp_lookup, dict) else None

    agent_map.update_time(current_time)
    
    # ===== UPDATE AGENT POSITIONS =====
    agent_map.update_agent_position(team, agent_name, current_pos, current_time)
    
    # Update defender positions from sensor
    enemy_team = "blue" if team == "red" else "red"
    if "agent" in sensors:
        all_agents: Dict[str, int] = sensors["agent"][1]
        for name, pos in all_agents.items():
            if name.startswith(enemy_team):
                agent_map.update_agent_position(enemy_team, name, pos, current_time)
    
    # ===== GET DEFENDER PARAMETERS =====
    defender_speed = team_cache.get("defender_speed", DEFAULT_DEFENDER_SPEED)
    defender_tag_radius = team_cache.get("defender_tag_radius", DEFAULT_DEFENDER_TAG_RADIUS)
    
    # ===== PROCESS FLAGS =====
    candidate_flags: List[int] = sensors["candidate_flag"][1]["candidate_flags"]
    real_flags: Set[int] = set(team_cache.get("real_flags", []))
    fake_flags: Set[int] = set(team_cache.get("fake_flags", []))
    
    if "egocentric_flag" in sensors:
        detected_flags: List[int] = sensors["egocentric_flag"][1]["detected_flags"]
        for flag in detected_flags:
            if flag not in real_flags:
                real_flags.add(flag)
        
        for flag_pos in candidate_flags:
            if flag_pos not in real_flags and flag_pos not in fake_flags:
                flag_node = global_map_sensor.nodes[flag_pos]
                current_node = global_map_sensor.nodes[current_pos]
                euclidean_dist = ((flag_node['x'] - current_node['x'])**2 + 
                                 (flag_node['y'] - current_node['y'])**2)**0.5
                if euclidean_dist <= sense_radius:
                    if flag_pos not in detected_flags:
                        fake_flags.add(flag_pos)
    
    team_cache.set("real_flags", list(real_flags))
    team_cache.set("fake_flags", list(fake_flags))
    unknown_flags = set(candidate_flags) - real_flags - fake_flags
    
    # ===== HELPER FUNCTIONS =====
    
    def graph_distance(node1: int, node2: int) -> int:
        if apsp is not None:
            d = get_cached_distance(apsp, node1, node2)
            if d is not None:
                return d
        return agent_map.shortest_path_length(node1, node2)
    
    def advantage_index(target_node: int, attacker_node: int, defender_node: int) -> float:
        return graph_distance(defender_node, target_node) - graph_distance(attacker_node, target_node)
    
    def capture_condition(flag_pos: int, attacker_pos: int, defenders: List[int]) -> float:
        if not defenders:
            return float('inf')
        min_adv = min(advantage_index(flag_pos, attacker_pos, d_pos) for d_pos in defenders)
        return min_adv - defender_tag_radius
    
    def get_sensing_free_set(defenders: List[int], defender_sense_radius: float) -> Set[int]:
        sensing_set = set()
        for d_pos in defenders:
            d_node = global_map_sensor.nodes[d_pos]
            for node_id in global_map_sensor.nodes():
                node = global_map_sensor.nodes[node_id]
                euclidean_dist = ((node['x'] - d_node['x'])**2 +
                                 (node['y'] - d_node['y'])**2)**0.5
                if euclidean_dist <= defender_sense_radius:
                    sensing_set.add(node_id)

        # Use stationary sensor covered_nodes if available (preferred)
        stationary_covered = set()
        for sensor_name, sensor_payload in sensors.items():
            if sensor_name.startswith("stationary_"):
                covered = sensor_payload[1].get("covered_nodes", [])
                stationary_covered.update(covered)

        if stationary_covered:
            sensing_set.update(stationary_covered)
        else:
            # Fallback: Euclidean radius around candidate flags
            stationary_sense_radius = team_cache.get("stationary_sense_radius", DEFAULT_STATIONARY_SENSE_RADIUS)
            for flag_pos in candidate_flags:
                flag_node = global_map_sensor.nodes[flag_pos]
                for node_id in global_map_sensor.nodes():
                    node = global_map_sensor.nodes[node_id]
                    euclidean_dist = ((node['x'] - flag_node['x'])**2 +
                                     (node['y'] - flag_node['y'])**2)**0.5
                    if euclidean_dist <= stationary_sense_radius:
                        sensing_set.add(node_id)
        return set(global_map_sensor.nodes()) - sensing_set
    
    def get_tagging_region(defenders: List[int], tag_radius: int) -> Set[int]:
        tagging_set = set()
        for d_pos in defenders:
            for node_id in global_map_sensor.nodes():
                if graph_distance(d_pos, node_id) <= tag_radius:
                    tagging_set.add(node_id)
        return tagging_set
    
    def path_exists_in_subgraph(subgraph_nodes: Set[int], start: int, end: int) -> bool:
        if start not in subgraph_nodes or end not in subgraph_nodes:
            return False
        subgraph = agent_map.graph.subgraph(subgraph_nodes)
        try:
            nx.shortest_path(subgraph, start, end)
            return True
        except:
            return False
    
    def shortest_path_in_subgraph(subgraph_nodes: Set[int], start: int, end: int, max_speed: int) -> int:
        if start not in subgraph_nodes or end not in subgraph_nodes:
            return start
        subgraph = agent_map.graph.subgraph(subgraph_nodes)
        try:
            path = nx.shortest_path(subgraph, start, end)
            next_index = min(max_speed, len(path) - 1)
            return path[next_index]
        except:
            return start

    def get_flee_move(start_node: int, defenders: List[int]) -> int:
        """
        Get neighbor that maximizes distance to nearest defenders.
        Prioritizes nodes in H_sense_c (safe zones).
        """
        neighbors = list(agent_map.graph.neighbors(start_node))
        best_move = start_node
        max_score = -float('inf')
        
        for n in neighbors:
            score = 0
            
            # Priority 1: Move into H_sense_c if possible
            if n in H_sense_c:
                score += FLEE_SAFE_ZONE_BONUS
            
            # Priority 2: Maximize distance to closest defender
            if defenders:
                min_d_dist = min(graph_distance(n, d) for d in defenders)
                score += min_d_dist * FLEE_DIST_WEIGHT  # Weight distance heavily
            
            if score > max_score:
                max_score = score
                best_move = n
                
        return best_move

    # ===== GET DEFENDER POSITIONS =====
    defender_positions: List[int] = []
    defenders_data = agent_map.get_team_agents(enemy_team)
    for name, pos, age in defenders_data:
        if pos is not None:
            defender_positions.append(pos)
    
    # ===== COMPUTE KEY SETS =====
    defender_sense_radius = team_cache.get("defender_sense_radius", DEFAULT_DEFENDER_SENSE_RADIUS)
    H_sense_c = get_sensing_free_set(defender_positions, defender_sense_radius)
    H_tag = get_tagging_region(defender_positions, defender_tag_radius)
    H_tag_c = set(global_map_sensor.nodes()) - H_tag
    
    # ===== EXPLORER ASSIGNMENT (EARLY CHECK) =====
    # Ensure at least one agent explores if there are unknown flags
    # Do this BEFORE any mode execution to handle explorer being forced to flee
    EXPLORER_COOLDOWN = 10  # Number of steps an agent is blocked from exploring after failing
    
    if unknown_flags:
        explorer_name = team_cache.get("explorer_name", None)
        explorer_cooldowns = team_cache.get("explorer_cooldowns", {})
        
        # Clean up expired cooldowns
        active_cooldowns = {name: expire_time for name, expire_time in explorer_cooldowns.items() 
                           if expire_time > current_time}
        team_cache.set("explorer_cooldowns", active_cooldowns)
        
        teammates_data = agent_map.get_team_agents(team)
        all_agent_names = [name for name, pos, _ in teammates_data if pos is not None]
        explorer_alive = explorer_name in all_agent_names
        
        # Check if current explorer is blocked (would flee or not in safe zone)
        explorer_blocked = False
        if explorer_name and explorer_alive:
            # Get explorer's position
            explorer_pos = None
            for name, pos, _ in teammates_data:
                if name == explorer_name:
                    explorer_pos = pos
                    break
            
            if explorer_pos:
                # Check if explorer would flee
                if defender_positions:
                    nearby_defenders = sum(1 for d in defender_positions if graph_distance(explorer_pos, d) <= TEAMING_CHECK_RADIUS)
                    dynamic_tolerance = BASE_FLEE_TOLERANCE + max(0, nearby_defenders - 1)
                    closest_def_dist = min(graph_distance(explorer_pos, d) for d in defender_positions)
                    if closest_def_dist <= dynamic_tolerance:
                        explorer_blocked = True
                
                # Check if explorer is not in safe zone
                if explorer_pos not in H_sense_c:
                    explorer_blocked = True
        
        # Reassign if explorer is dead, None, or blocked
        if explorer_name is None or not explorer_alive or explorer_blocked:
            # If current explorer was blocked, add to cooldown
            if explorer_blocked and explorer_name:
                active_cooldowns[explorer_name] = current_time + EXPLORER_COOLDOWN
                team_cache.set("explorer_cooldowns", active_cooldowns)
                # print(f"[EXPLORER] {explorer_name} failed, cooldown until t={current_time + EXPLORER_COOLDOWN}")
            
            # Find safe agents who won't flee and are not on cooldown
            safe_agents = []
            for name, pos, _ in teammates_data:
                # Skip agents on cooldown
                if name in active_cooldowns:
                    continue
                    
                if pos is not None and pos in H_sense_c:
                    # Check this agent won't flee
                    will_flee = False
                    if defender_positions:
                        nearby_defenders = sum(1 for d in defender_positions if graph_distance(pos, d) <= TEAMING_CHECK_RADIUS)
                        dynamic_tolerance = BASE_FLEE_TOLERANCE + max(0, nearby_defenders - 1)
                        closest_def_dist = min(graph_distance(pos, d) for d in defender_positions)
                        if closest_def_dist <= dynamic_tolerance:
                            will_flee = True
                    
                    if not will_flee:
                        safe_agents.append(name)
            
            if safe_agents:
                import random
                new_explorer = random.choice(safe_agents)
                team_cache.set("explorer_name", new_explorer)
                # print(f"[EXPLORER] Assigned {new_explorer} as new explorer")
            elif all_agent_names:
                # No safe agents, pick any not on cooldown
                available = [n for n in all_agent_names if n not in active_cooldowns]
                if available:
                    import random
                    new_explorer = random.choice(available)
                    team_cache.set("explorer_name", new_explorer)
                    # print(f"[EXPLORER] No safe agents, assigned {new_explorer}")
                elif not explorer_alive:
                    # All on cooldown or dead, pick any alive
                    import random
                    team_cache.set("explorer_name", random.choice(all_agent_names))
    
    # ===== MODE DETERMINATION =====
    target = current_pos
    discovered_flags = set()
    
    current_mode = cache.get("mode", "exploration")
    is_retreating = cache.get("is_retreating", False)
    retreat_target = cache.get("retreat_target", None)
    
    # ===== MODE 1: FLAG CAPTURE (HIGHEST PRIORITY) =====
    best_capture_flag = None
    best_capture_score = -float('inf')
    
    for flag_pos in real_flags:
        score = capture_condition(flag_pos, current_pos, defender_positions)
        if score > 0 and score > best_capture_score:
            best_capture_flag = flag_pos
            best_capture_score = score
    
    if best_capture_flag is None:
        for flag_pos in unknown_flags:
            score = capture_condition(flag_pos, current_pos, defender_positions)
            if score > 0 and score > best_capture_score:
                best_capture_flag = flag_pos
                best_capture_score = score
    
    # HANDLE RETREAT FROM FAKE FLAG
    if is_retreating and retreat_target is not None:
        if current_pos in H_sense_c:
            cache.set("is_retreating", False)
            cache.set("retreat_target", None)
            cache.set("mode", "prep")
        else:
            target = agent_map.shortest_path_step(current_pos, retreat_target, speed)
            cache.set("mode", "retreat")
            state["action"] = target
            # print(f"[{agent_name}] Mode: RETREAT")
            return discovered_flags
    
    # CAPTURE MODE
    if best_capture_flag is not None:
        if best_capture_flag in fake_flags:
            if H_sense_c:
                closest_safe = min(H_sense_c, key=lambda n: graph_distance(current_pos, n))
                cache.set("is_retreating", True)
                cache.set("retreat_target", closest_safe)
                cache.set("mode", "retreat")
                target = agent_map.shortest_path_step(current_pos, closest_safe, speed)
            else:
                target = current_pos
        else:
            cache.set("mode", "capture")
            cache.set("is_retreating", False)
            target = agent_map.shortest_path_step(current_pos, best_capture_flag, speed)
        
        state["action"] = target
        current_mode = cache.get("mode", "unknown")
        # print(f"[{agent_name}] Mode: {current_mode.upper()}")
        if "egocentric_flag" in sensors:
            discovered_flags = set(sensors["egocentric_flag"][1]["detected_flags"])
        return discovered_flags

    # ===== MODE 1.5: EMERGENCY FLEE (HIGH PRIORITY) =====
    # "On top of all other modes... if defender is too close... fleed"
    if defender_positions:
        # Calculate dynamic tolerance based on defender density (teaming)
        # Base tolerance +1 for every extra defender nearby
        nearby_defenders_count = sum(1 for d in defender_positions if graph_distance(current_pos, d) <= TEAMING_CHECK_RADIUS)
        dynamic_tolerance = BASE_FLEE_TOLERANCE + max(0, nearby_defenders_count - 1)
        
        closest_defender_dist = min(graph_distance(current_pos, d) for d in defender_positions)
        
        if closest_defender_dist <= dynamic_tolerance:
            cache.set("mode", "flee")
            cache.set("is_retreating", False)
            target = get_flee_move(current_pos, defender_positions)
            state["action"] = target
            # print(f"[{agent_name}] Mode: FLEE (defender_dist={closest_defender_dist}, tolerance={dynamic_tolerance}, nearby_defenders={nearby_defenders_count})")
            return discovered_flags
    
    # ===== MODE 2: PREP STAGE =====
    if not unknown_flags and real_flags and current_pos in H_sense_c:
        cache.set("mode", "prep")
        cache.set("is_retreating", False)
        
        best_flag = None
        best_flag_score = -float('inf')
        for flag_pos in real_flags:
            if defender_positions:
                min_adv = min(advantage_index(flag_pos, current_pos, d) for d in defender_positions)
                if min_adv > best_flag_score:
                    best_flag = flag_pos
                    best_flag_score = min_adv
        
        if best_flag is not None:
            best_position = None
            best_pos_score = -float('inf')
            for node in H_sense_c:
                if defender_positions:
                    min_adv = min(advantage_index(best_flag, node, d) for d in defender_positions)
                    if min_adv > best_pos_score:
                        best_position = node
                        best_pos_score = min_adv
            
            if best_position is not None:
                if path_exists_in_subgraph(H_sense_c, current_pos, best_position):
                    target = shortest_path_in_subgraph(H_sense_c, current_pos, best_position, speed)
                else:
                    target = current_pos
            else:
                target = current_pos
        else:
            target = current_pos
        
        state["action"] = target
        # print(f"[{agent_name}] Mode: PREP")
        if "egocentric_flag" in sensors:
            discovered_flags = set(sensors["egocentric_flag"][1]["detected_flags"])
        return discovered_flags
    
    # ===== MODE 3: EXPLORATION =====
    # Explorer assignment done earlier, just check if I'm the explorer
    explorer_name = team_cache.get("explorer_name", None)
    
    if agent_name == explorer_name and unknown_flags:
            cache.set("mode", "exploration")
            cache.set("is_retreating", False)
            
            closest_unknown = min(unknown_flags, key=lambda f: graph_distance(current_pos, f))
            next_step = agent_map.shortest_path_step(current_pos, closest_unknown, speed)
            
            # REVISED CHECK WITH SAFETY BUFFER
            safety_buffer = defender_tag_radius + EXPLORER_SAFETY_BUFFER
            is_safe_step = True
            
            if defender_positions:
                dist_to_closest = min(graph_distance(next_step, d) for d in defender_positions)
                if dist_to_closest <= safety_buffer:
                    is_safe_step = False
            
            # If next step is safe (distance > buffer) AND outside tag range
            if is_safe_step and next_step in H_tag_c:
                target = next_step
            else:
                # Path blocked by safety buffer or tag zone.
                # Trigger flee logic to reposition safely.
                target = get_flee_move(current_pos, defender_positions)
            
            state["action"] = target
            # print(f"[{agent_name}] Mode: EXPLORATION")
            if "egocentric_flag" in sensors:
                discovered_flags = set(sensors["egocentric_flag"][1]["detected_flags"])
            return discovered_flags
    
    # ===== DEFAULT: PREP OR STAY SAFE =====
    # Debug why not in prep
    in_safe_zone = current_pos in H_sense_c
    has_unknown = len(unknown_flags) > 0
    has_real = len(real_flags) > 0
    
    if current_pos not in H_sense_c and H_sense_c:
        closest_safe = min(H_sense_c, key=lambda n: graph_distance(current_pos, n))
        target = agent_map.shortest_path_step(current_pos, closest_safe, speed)
        cache.set("mode", "moving_to_safety")
    elif current_pos not in H_sense_c:
        target = current_pos
        cache.set("mode", "stuck")
    else:
        target = current_pos
        cache.set("mode", "idle")
    
    cache.set("is_retreating", False)
    state["action"] = target
    
    # Print current mode with debug info
    current_mode = cache.get("mode", "unknown")
    # print(f"[{agent_name}] Mode: {current_mode.upper()} (in_safe={in_safe_zone}, unknown_flags={has_unknown}, real_flags={has_real})")
    
    if "egocentric_flag" in sensors:
        discovered_flags = set(sensors["egocentric_flag"][1]["detected_flags"])
    return discovered_flags


def map_strategy(agent_config):
    strategies = {}
    for name in agent_config.keys():
        strategies[name] = attacker_strategy
    return strategies
