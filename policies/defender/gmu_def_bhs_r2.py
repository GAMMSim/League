def defender_strategy(state):
    """
    Defender strategy: Full Version V1.5 (Corrected Stationary Sensing)
    - Threat-Based Assignment for Blockers
    - 1-1 Hunter Coordination
    - Diversity in Seeker Targets
    - DYNAMIC agent sensing radius
    - SEPARATE stationary sensor radius
    """
    from typing import Dict, List, Tuple, Set, Optional, Any
    import networkx as nx
    import random
    from lib.core.apsp_cache import get_apsp_length_cache, get_cached_distance
    
    # ===== CONFIGURATION (HARDCODED VALUES) =====
    # Vision parameters
    DEFAULT_STATIONARY_SENSE_RADIUS = 450   # Radius for Flags/Stationary sensors
                                            # MODIFY THIS based on game rules!
    
    # Assumptions about enemy capabilities (used if not found in cache)
    DEFAULT_ATTACKER_CAPTURE_RADIUS = 2
    DEFAULT_ATTACKER_SPEED = 1
    
    # Memory / State tracking
    ATTACKER_MEMORY_AGE = 5          # How many turns to remember an attacker's last known pos
    
    # ============================================

    # ===== INITIALIZATION =====
    agent_ctrl = state["agent_controller"]
    current_pos: int = state["curr_pos"]
    current_time: int = state["time"]
    team: str = agent_ctrl.team
    agent_name: str = agent_ctrl.name
    
    speed: int = int(agent_ctrl.speed)
    tagging_radius: int = int(agent_ctrl.tagging_radius)
    
    # Dynamic Agent Sensing (from Controller)
    agent_sense_radius = float(agent_ctrl.sensing_radius)
    
    # Stationary Sensing (from Cache or Config)
    # We try to load it from cache in case the team shares discovered stats, otherwise use config
    stationary_sense_radius = agent_ctrl.team_cache.get("stationary_sense_radius", DEFAULT_STATIONARY_SENSE_RADIUS)
    
    cache = agent_ctrl.cache
    team_cache = agent_ctrl.team_cache
    agent_map = agent_ctrl.map
    
    sensors = state["sensor"]
    global_map_payload = sensors["global_map"][1]
    global_map_sensor: nx.Graph = global_map_payload["graph"]
    global_map_apsp = global_map_payload.get("apsp")
    
    # Initialize map/APSP only on the first timestep.
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

    # Update Positions
    agent_map.update_agent_position(team, agent_name, current_pos, current_time)
    
    if "custom_team" in sensors:
        for t_name, t_pos in sensors["custom_team"][1].items():
            agent_map.update_agent_position(team, t_name, t_pos, current_time)
    
    enemy_team = "blue" if team == "red" else "red"
    if "egocentric_agent" in sensors:
        for name, node_id in sensors["egocentric_agent"][1].items():
            if name.startswith(enemy_team):
                agent_map.update_agent_position(enemy_team, name, node_id, current_time)
    
    for sensor_name in sensors:
        if sensor_name.startswith("stationary_"):
            detected = sensors[sensor_name][1]["detected_agents"]
            for d_name, d_info in detected.items():
                if d_name.startswith(enemy_team):
                    agent_map.update_agent_position(enemy_team, d_name, d_info["node_id"], current_time)
    
    attacker_capture_radius = team_cache.get("attacker_capture_radius", DEFAULT_ATTACKER_CAPTURE_RADIUS)
    attacker_speed = team_cache.get("attacker_speed", DEFAULT_ATTACKER_SPEED)
    
    real_flags: List[int] = sensors["flag"][1]["real_flags"]
    fake_flags: List[int] = sensors["flag"][1]["fake_flags"]
    candidate_flags: List[int] = real_flags + fake_flags
    
    # ===== HELPER FUNCTIONS =====
    
    def graph_distance(node1: int, node2: int) -> int:
        if apsp is not None:
            d = get_cached_distance(apsp, node1, node2)
            if d is not None:
                return d
        return agent_map.shortest_path_length(node1, node2)
    
    def get_attacker_positions(max_age: int = ATTACKER_MEMORY_AGE) -> List[Tuple[str, int, int]]:
        attackers = []
        for name, pos, age in agent_map.get_team_agents(enemy_team):
            if pos is not None and age <= max_age:
                attackers.append((name, pos, age))
        return attackers
    
    def get_defender_positions() -> List[Tuple[str, int]]:
        defenders = []
        for name, pos, age in agent_map.get_team_agents(team):
            if pos is not None:
                defenders.append((name, pos))
        return defenders
    
    def get_target_set(flag_pos: int, capture_radius: int) -> Set[int]:
        target_set = set()
        for node_id in agent_map.graph.nodes():
            if graph_distance(node_id, flag_pos) <= capture_radius:
                target_set.add(node_id)
        return target_set
    
    def get_boundary_nodes(target_set: Set[int], flag_pos: int, capture_radius: int) -> List[int]:
        boundary = []
        for node_id in agent_map.graph.nodes():
            if graph_distance(node_id, flag_pos) == capture_radius:
                boundary.append(node_id)
        return boundary
    
    def get_sensing_free_set(defenders: List[Tuple[str, int]]) -> Set[int]:
        sensing_set = set()
        
        # 1. Add DYNAMIC defender regions (using agent_sense_radius)
        for _, d_pos in defenders:
            d_node = global_map_sensor.nodes[d_pos]
            for node_id in global_map_sensor.nodes():
                node = global_map_sensor.nodes[node_id]
                dist = ((node['x'] - d_node['x'])**2 + (node['y'] - d_node['y'])**2)**0.5
                if dist <= agent_sense_radius:
                    sensing_set.add(node_id)
                    
        # 2. Add STATIONARY sensor regions from sensor payload (preferred)
        stationary_covered_nodes = set()
        for sensor_name, sensor_payload in sensors.items():
            if sensor_name.startswith("stationary_"):
                sensor_data = sensor_payload[1]
                # print(f"Processing sensor {sensor_name}: {sensor_data}")
                covered_nodes = sensor_data.get("covered_nodes", [])
                # print(f"Covered nodes for {sensor_name}: {covered_nodes}")
                stationary_covered_nodes.update(covered_nodes)

        if stationary_covered_nodes:
            sensing_set.update(stationary_covered_nodes)
        else:
            # Fallback for older sensor payloads without covered_nodes
            for flag_pos in candidate_flags:
                flag_node = global_map_sensor.nodes[flag_pos]
                for node_id in global_map_sensor.nodes():
                    node = global_map_sensor.nodes[node_id]
                    dist = ((node['x'] - flag_node['x'])**2 + (node['y'] - flag_node['y'])**2)**0.5
                    if dist <= stationary_sense_radius:
                        sensing_set.add(node_id)
                    
        return set(global_map_sensor.nodes()) - sensing_set
    
    def get_reachable_set(agent_pos: int, agent_speed: int) -> Set[int]:
        reachable = set()
        for node_id in agent_map.graph.nodes():
            if graph_distance(agent_pos, node_id) <= agent_speed:
                reachable.add(node_id)
        return reachable
    
    def threat_based_assignment(defenders: List[Tuple[str, int]], flags: List[int], attackers: List[Tuple[str, int, int]]) -> Dict[str, int]:
        """Assign defenders based on attacker threat density per flag."""
        assignment = {}
        defender_dict = {name: pos for name, pos in defenders}
        available_defenders = set(name for name, _ in defenders)
        
        # 1. Calculate Threat
        flag_threats = {f: 0 for f in flags}
        for _, att_pos, _ in attackers:
            if flags:
                closest_f = min(flags, key=lambda f: graph_distance(att_pos, f))
                flag_threats[closest_f] += 1
        
        # 2. Create Tasks
        tasks = []
        for f, threat in flag_threats.items():
            for _ in range(threat):
                tasks.append(f)
        
        # 3. Assign to Threats
        while tasks and available_defenders:
            target_flag = tasks.pop(0)
            best_def = min(available_defenders, key=lambda d: graph_distance(defender_dict[d], target_flag))
            assignment[best_def] = target_flag
            available_defenders.remove(best_def)
            
        # 4. Handle Surplus
        uncovered = [f for f in flags if f not in assignment.values()]
        while available_defenders:
            def_name = available_defenders.pop()
            pos = defender_dict[def_name]
            if uncovered:
                target = min(uncovered, key=lambda f: graph_distance(pos, f))
                assignment[def_name] = target
            else:
                target = min(flags, key=lambda f: graph_distance(pos, f))
                assignment[def_name] = target
        return assignment

    # ===== STATE UPDATE =====
    attackers = get_attacker_positions(max_age=ATTACKER_MEMORY_AGE)
    all_defenders = get_defender_positions()
    H_sense_c = get_sensing_free_set(all_defenders)
    
    defender_modes = team_cache.get("defender_modes", {})
    hunter_targets = team_cache.get("hunter_targets", {})
    flag_assignments = team_cache.get("flag_assignments", {})
    seekers = team_cache.get("seekers", [])
    seeker_targets = team_cache.get("seeker_targets", {})
    known_attackers = team_cache.get("known_attackers", set())
    
    current_known = set(name for name, _, _ in attackers)
    newly_discovered = current_known - known_attackers
    team_cache.set("known_attackers", current_known)
    
    # Calculate desired seekers
    # If attackers < defenders, only send seekers if we have enough blockers for real flags
    num_real_flags = len(real_flags) if real_flags else 0
    
    # Count current hunters (who are not blocking)
    current_hunters = sum(1 for n, _ in all_defenders if defender_modes.get(n, "blocking") == "hunting")
    
    if len(attackers) < len(all_defenders):
        # We have surplus defenders
        # Calculate how many blockers we need (at least one per real flag)
        available_for_seeking = len(all_defenders) - num_real_flags - current_hunters
        desired_seekers = max(0, available_for_seeking)
    else:
        # Attackers >= Defenders, no seekers needed
        desired_seekers = 0
    
    current_mode = defender_modes.get(agent_name, "blocking")
    target = current_pos
    
    # ===== MODE TRANSITIONS =====
    
    # Hunting -> Blocking (Lost Track)
    if current_mode == "hunting":
        target_attacker = hunter_targets.get(agent_name)
        if target_attacker:
            visible = any(name == target_attacker for name, _, _ in attackers)
            if not visible:
                current_mode = "blocking"
                hunter_targets.pop(agent_name, None)
    
    # Blocking -> Seeking (Need Seekers)
    if current_mode == "blocking" and len(seekers) < desired_seekers:
        if agent_name not in seekers:
            # Only switch if we are the least valuable blocker (furthest from real flags)
            blockers = [n for n, p in all_defenders if defender_modes.get(n, "blocking") == "blocking"]
            if blockers:
                # Calculate distance to nearest REAL flag
                def dist_to_real(b_name):
                    # Find pos
                    pos = next((p for n, p in all_defenders if n == b_name), None)
                    if not pos: return 0
                    if not real_flags: return 0
                    return min(graph_distance(pos, f) for f in real_flags)
                
                worst_blocker = max(blockers, key=dist_to_real)
                if agent_name == worst_blocker:
                    seekers.append(agent_name)
                    current_mode = "seeking"
    
    # Reduce Seekers
    if len(seekers) > desired_seekers and agent_name in seekers:
        seekers.remove(agent_name)
        if current_mode == "seeking":
            current_mode = "blocking"
            
    # ===== MODE EXECUTION =====
    
    # --- BLOCKING ---
    if current_mode == "blocking":
        blockers = [(n, p) for n, p in all_defenders if defender_modes.get(n, "blocking") == "blocking"]
        defend_targets = real_flags if real_flags else candidate_flags
        
        if blockers and defend_targets:
            flag_assignments = threat_based_assignment(blockers, defend_targets, attackers)
            
        assigned_flag = flag_assignments.get(agent_name)
        
        if assigned_flag is None:
            target = current_pos
        else:
            target_set = get_target_set(assigned_flag, attacker_capture_radius)
            closest_attacker = None
            closest_dist = float('inf')
            
            for att_name, att_pos, _ in attackers:
                d = graph_distance(current_pos, att_pos)
                if d < closest_dist:
                    closest_dist = d
                    closest_attacker = (att_name, att_pos)
            
            if closest_attacker:
                att_name, att_pos = closest_attacker
                R_a = get_reachable_set(att_pos, attacker_speed)
                outside = att_pos not in target_set
                safe_escape = len(R_a & H_sense_c) > 0
                already_hunted = att_name in hunter_targets.values()
                
                if outside and safe_escape and not already_hunted:
                    current_mode = "hunting"
                    hunter_targets[agent_name] = att_name
                    target = agent_map.shortest_path_step(current_pos, att_pos, speed)
                else:
                    # Intercept
                    crit_node = min(target_set, key=lambda n: graph_distance(att_pos, n), default=None)
                    if crit_node: target = agent_map.shortest_path_step(current_pos, crit_node, speed)
            else:
                # Patrol
                boundary = get_boundary_nodes(target_set, assigned_flag, attacker_capture_radius)
                if boundary:
                    idx = cache.get("patrol_index", 0)
                    t_bound = boundary[idx % len(boundary)]
                    if current_pos == t_bound:
                        idx = (idx + 1) % len(boundary)
                        cache.set("patrol_index", idx)
                        t_bound = boundary[idx]
                    target = agent_map.shortest_path_step(current_pos, t_bound, speed)
                else:
                    target = agent_map.shortest_path_step(current_pos, assigned_flag, speed)

    # --- HUNTING ---
    elif current_mode == "hunting":
        t_name = hunter_targets.get(agent_name)
        t_pos = next((p for n, p, _ in attackers if n == t_name), None)
        
        if not t_pos:
            current_mode = "blocking"
            target = current_pos
        else:
            R_a = get_reachable_set(t_pos, attacker_speed)
            if len(R_a & H_sense_c) > 0:
                target = agent_map.shortest_path_step(current_pos, t_pos, speed)
            else:
                # Block path
                c_flag = min(candidate_flags, key=lambda f: graph_distance(t_pos, f), default=None)
                if c_flag:
                    try:
                        path = nx.shortest_path(agent_map.graph, t_pos, c_flag)
                        intercept = None
                        min_d = float('inf')
                        my_node_data = global_map_sensor.nodes[current_pos]
                        for p_node in path:
                            pn_data = global_map_sensor.nodes[p_node]
                            euc = ((pn_data['x'] - my_node_data['x'])**2 + (pn_data['y'] - my_node_data['y'])**2)**0.5
                            if euc <= agent_sense_radius:
                                d = graph_distance(current_pos, p_node)
                                if d < min_d:
                                    min_d = d
                                    intercept = p_node
                        target = agent_map.shortest_path_step(current_pos, intercept if intercept else t_pos, speed)
                    except:
                        target = agent_map.shortest_path_step(current_pos, t_pos, speed)
                else:
                    target = agent_map.shortest_path_step(current_pos, t_pos, speed)

    # --- SEEKING ---
    elif current_mode == "seeking":
        if newly_discovered:
            # Switch to hunt new guy
            target_att = list(newly_discovered)[0]
            current_mode = "hunting"
            hunter_targets[agent_name] = target_att
            seekers.remove(agent_name)
            seeker_targets.pop(agent_name, None)  # Clean up our target
            
            att_pos = next((p for n, p, _ in attackers if n == target_att), None)
            target = agent_map.shortest_path_step(current_pos, att_pos, speed) if att_pos else current_pos
        else:
            # Clean up seeker_targets: only keep targets for current seekers
            seeker_targets = {name: node for name, node in seeker_targets.items() if name in seekers}
            
            # Explore diverse targets
            other_targets = set(seeker_targets.get(n) for n in seekers if n != agent_name and n in seeker_targets)
            best_node = None
            max_n = -1
            
            # Find node in H_sense_c with most neighbors in H_sense_c
            # Exclude nodes targeted by others
            candidates = [n for n in H_sense_c if n not in other_targets or n == seeker_targets.get(agent_name)]
            
            # print(f"[{agent_name}] SEEKING: H_sense_c size={len(H_sense_c)}, candidates={len(candidates)}, other_targets={other_targets}")
            
            if not candidates and H_sense_c:
                candidates = list(H_sense_c) # Fallback to shared targets
                # print(f"[{agent_name}] Fallback: using shared targets, candidates={len(candidates)}")
            
            # Score all candidates
            for node in candidates:
                n_count = sum(1 for n in agent_map.graph.neighbors(node) if n in H_sense_c)
                if n_count > max_n:
                    max_n = n_count
                    best_node = node
            
            # Print chosen node info
            if best_node:
                best_dist = graph_distance(current_pos, best_node)
                # print(f"[{agent_name}] CHOSEN: Node {best_node} with {max_n} H_sense_c neighbors, dist={best_dist}")
            
            if best_node:
                seeker_targets[agent_name] = best_node
                target = agent_map.shortest_path_step(current_pos, best_node, speed)
            else:
                # print(f"[{agent_name}] No valid seeking target found")
                target = current_pos

    # ===== UPDATE & RETURN =====
    defender_modes[agent_name] = current_mode
    
    # Print role assignment
    target_info = ""
    if current_mode == "hunting":
        target_att = hunter_targets.get(agent_name, "None")
        target_info = f" -> {target_att}"
    elif current_mode == "blocking":
        assigned_flag = flag_assignments.get(agent_name)
        target_info = f" -> Flag@{assigned_flag}" if assigned_flag else ""
    elif current_mode == "seeking":
        seek_node = seeker_targets.get(agent_name)
        target_info = f" -> Node {seek_node}" if seek_node else ""
    
    # print(f"[{agent_name}] Role: {current_mode.upper()}{target_info}")
    
    team_cache.set("defender_modes", defender_modes)
    team_cache.set("hunter_targets", hunter_targets)
    team_cache.set("flag_assignments", flag_assignments)
    team_cache.set("seekers", seekers)
    team_cache.set("seeker_targets", seeker_targets)
    
    state["action"] = target
    return set()

def map_strategy(agent_config):
    strategies = {}
    for name in agent_config.keys():
        strategies[name] = defender_strategy
    return strategies
