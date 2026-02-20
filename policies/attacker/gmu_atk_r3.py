def attacker_strategy(state):
    """
    GMU Attacker R3 - "Strike" Strategy

    Phases:
    1. Rendezvous: All attackers converge at a safe meeting point
    2. Search: Find a flag (skip if already found)
    2.5. Wait: Position at best boundary node, wait for optimal strike
    3. Strike: All rush the flag together
    """
    from typing import Dict, List, Set
    import networkx as nx
    from lib.core.apsp_cache import get_apsp_length_cache, get_cached_distance
    from lib.core.console import info, debug

    # ===== CONFIGURATION =====
    DEFAULT_DEFENDER_SPEED = 1
    DEFAULT_DEFENDER_TAG_RADIUS = 1
    DEFAULT_STATIONARY_SENSE_RADIUS = 450
    DEFAULT_DEFENDER_SENSE_RADIUS = 250
    STRICT_NO_ENTER_HOPS = 2
    N_WIN_WEIGHT_K = 1.0
    N_WIN_WEIGHT_K_LATE = 0.5
    N_WIN_WEIGHT_SWITCH_T = 100
    RENDEZVOUS_TIMEOUT_T = 40
    # RENDEZVOUS_TIMEOUT_T = 75

    # ===== INITIALIZATION =====
    agent_ctrl = state["agent_controller"]
    current_pos: int = state["curr_pos"]
    current_time: int = state["time"]
    team: str = agent_ctrl.team
    agent_name: str = agent_ctrl.name
    speed: int = int(agent_ctrl.speed)
    capture_radius: int = int(agent_ctrl.capture_radius)
    sense_radius = agent_ctrl.sensing_radius

    cache = agent_ctrl.cache
    team_cache = agent_ctrl.team_cache
    agent_map = agent_ctrl.map

    sensors = state["sensor"]
    global_map_payload = sensors["global_map"][1]
    global_map_sensor: nx.Graph = global_map_payload["graph"]
    global_map_apsp = global_map_payload.get("apsp")

    # ===== INITIALIZE MAP/APSP =====
    if agent_map.graph is None:
        nodes_data = {nid: global_map_sensor.nodes[nid] for nid in global_map_sensor.nodes()}
        edges_data = {}
        for idx, (u, v, data) in enumerate(global_map_sensor.edges(data=True)):
            edges_data[idx] = {"source": u, "target": v, **data}
        agent_map.attach_networkx_graph(
            nodes_data, edges_data,
            apsp_lookup=global_map_apsp if isinstance(global_map_apsp, dict) else None,
        )

    if agent_map.graph is not None and agent_map.apsp_lookup is None:
        agent_map.set_apsp_lookup(get_apsp_length_cache(agent_map.graph))

    apsp = agent_map.apsp_lookup if isinstance(agent_map.apsp_lookup, dict) else None
    agent_map.update_time(current_time)

    # ===== UPDATE POSITIONS =====
    enemy_team = "blue" if team == "red" else "red"
    if "agent" in sensors:
        sensor_agent_positions = sensors["agent"][1]
        seen_enemy_names = set()
        for name, pos in sensor_agent_positions.items():
            if name.startswith(team):
                agent_map.update_agent_position(team, name, pos, current_time)
            else:
                seen_enemy_names.add(name)
                agent_map.update_agent_position(enemy_team, name, pos, current_time)

        # Keep enemy cache synchronized with the latest team sensor frame.
        cached_enemy_names = [name for name, _, _ in agent_map.get_team_agents(enemy_team)]
        for cached_enemy in cached_enemy_names:
            if cached_enemy not in seen_enemy_names:
                agent_map.remove_agent(enemy_team, cached_enemy)
    else:
        agent_map.update_agent_position(team, agent_name, current_pos, current_time)

    # ===== PROCESS FLAGS =====
    candidate_flags: List[int] = sensors["candidate_flag"][1]["candidate_flags"]
    real_flags: Set[int] = set(team_cache.get("real_flags", []))
    fake_flags: Set[int] = set(team_cache.get("fake_flags", []))
    discovered_flags = set()

    if "egocentric_flag" in sensors:
        detected_flags: List[int] = sensors["egocentric_flag"][1]["detected_flags"]
        for flag in detected_flags:
            if flag not in real_flags:
                real_flags.add(flag)
        discovered_flags = set(detected_flags)

        for flag_pos in candidate_flags:
            if flag_pos not in real_flags and flag_pos not in fake_flags:
                flag_node = global_map_sensor.nodes[flag_pos]
                current_node = global_map_sensor.nodes[current_pos]
                eucl = ((flag_node['x'] - current_node['x'])**2 +
                        (flag_node['y'] - current_node['y'])**2)**0.5
                if eucl <= sense_radius:
                    if flag_pos not in detected_flags:
                        fake_flags.add(flag_pos)

    team_cache.set("real_flags", list(real_flags))
    team_cache.set("fake_flags", list(fake_flags))
    candidate_flag_set = set(candidate_flags)
    unresolved_candidate_flags = candidate_flag_set - real_flags - fake_flags
    all_flags_checked = not unresolved_candidate_flags

    # ===== HELPER FUNCTIONS =====

    def graph_distance(n1: int, n2: int) -> int:
        if apsp is not None:
            d = get_cached_distance(apsp, n1, n2)
            if d is not None:
                return d
        return agent_map.shortest_path_length(n1, n2)

    capture_region_by_flag: Dict[int, List[int]] = {}
    fallback_stationary_covered_nodes: List[int] = []

    def get_target_region_nodes(flags: Set[int]) -> Set[int]:
        target_nodes = set()
        for flag in flags:
            target_nodes.update(capture_region_by_flag.get(flag, ()))
        return target_nodes

    def defender_advantage(v: int, defender_pos: int, attacker_pos: int) -> float:
        """A_D(v | D^i, A^j) = d_g(v, A^j) - d_g(v, D^i) + R_tag
        Positive = defender advantage, Negative = attacker advantage"""
        return graph_distance(v, attacker_pos) - graph_distance(v, defender_pos) + defender_tag_radius

    def get_V_safe(defenders: List[int]) -> Set[int]:
        """V_safe: nodes not sensed by any defender or flag sensor"""
        sensing_set = set()
        # Defender egocentric sensing
        for d_pos in defenders:
            d_node = global_map_sensor.nodes[d_pos]
            for nid in global_map_sensor.nodes():
                n_data = global_map_sensor.nodes[nid]
                eucl = ((n_data['x'] - d_node['x'])**2 + (n_data['y'] - d_node['y'])**2)**0.5
                if eucl <= defender_sense_radius:
                    sensing_set.add(nid)

        # Stationary sensor coverage (flags)
        stationary_covered = set()
        for sname, spayload in sensors.items():
            if sname.startswith("stationary_"):
                covered = spayload[1].get("covered_nodes", [])
                stationary_covered.update(covered)

        if stationary_covered:
            sensing_set.update(stationary_covered)
        else:
            sensing_set.update(fallback_stationary_covered_nodes)

        return set(global_map_sensor.nodes()) - sensing_set

    def pick_safest_on_path_neighbor(
        pos: int,
        target: int,
        defenders: List[int],
        ignore_no_enter: bool = False,
    ) -> int:
        """Move one step toward target on shortest path, picking the neighbor
        that minimizes worst-case defender advantage."""
        if pos == target:
            return pos
        neighbors = list(agent_map.graph.neighbors(pos))
        d_pos_to_target = graph_distance(pos, target)
        # Filter to neighbors on a shortest path
        candidates = [n for n in neighbors
                      if graph_distance(pos, n) + graph_distance(n, target) == d_pos_to_target]
        if not ignore_no_enter:
            relaxed_hops = max(0, STRICT_NO_ENTER_HOPS - 1)
            filtered_candidates = []
            for n in candidates:
                if n not in no_enter_nodes:
                    filtered_candidates.append(n)
                    continue
                # Capture-region exception: allow one-step deeper no-enter violation
                # when this step is already in direct-capture region and still > relaxed_hops.
                if (
                    n in target_region_nodes
                    and all(graph_distance(n, d) > relaxed_hops for d in defenders)
                ):
                    filtered_candidates.append(n)
            candidates = filtered_candidates
        if not candidates:
            # Direct shortest-path step is blocked: try one safe-subgraph shortest-path step.
            if not ignore_no_enter:
                safe_sub = get_safe_subgraph()
                if pos in safe_sub and target in safe_sub:
                    try:
                        safe_path = nx.shortest_path(safe_sub, pos, target)
                        return safe_path[min(speed, len(safe_path) - 1)]
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass
            # No direct move and no safe-subgraph detour available: hold position.
            return pos
        if not defenders:
            return candidates[0]
        # Pick candidate minimizing max defender advantage
        def worst_adv(v):
            return max(defender_advantage(v, d, pos) for d in defenders)
        return min(candidates, key=worst_adv)

    def pick_strike_on_path_neighbor(pos: int, target: int, defenders: List[int]) -> int:
        """Strike movement toward a fixed target.
        Ignore no-enter, but when multiple progress moves exist,
        prefer the safer one by defender-advantage score."""
        if pos == target:
            return pos
        neighbors = list(agent_map.graph.neighbors(pos))
        if not neighbors:
            return pos

        d_pos_to_target = graph_distance(pos, target)
        shortest_candidates = [
            n for n in neighbors
            if graph_distance(pos, n) + graph_distance(n, target) == d_pos_to_target
        ]
        if shortest_candidates:
            best_progress = min(graph_distance(n, target) for n in shortest_candidates)
            best_progress_candidates = [n for n in shortest_candidates if graph_distance(n, target) == best_progress]
            if not defenders:
                return min(best_progress_candidates)
            return min(
                best_progress_candidates,
                key=lambda n: max(defender_advantage(n, d, pos) for d in defenders),
            )

        # Rare fallback for distance-cache/graph inconsistencies:
        # still take any neighbor that strictly decreases target distance.
        improving_neighbors = [n for n in neighbors if graph_distance(n, target) < d_pos_to_target]
        if improving_neighbors:
            best_progress = min(graph_distance(n, target) for n in improving_neighbors)
            best_progress_candidates = [n for n in improving_neighbors if graph_distance(n, target) == best_progress]
            if not defenders:
                return min(best_progress_candidates)
            return min(
                best_progress_candidates,
                key=lambda n: max(defender_advantage(n, d, pos) for d in defenders),
            )
        return pos

    # ===== GET DEFENDER/ATTACKER STATE =====
    defender_tag_radius = team_cache.get("defender_tag_radius", DEFAULT_DEFENDER_TAG_RADIUS)
    defender_sense_radius = team_cache.get("defender_sense_radius", DEFAULT_DEFENDER_SENSE_RADIUS)

    defender_positions: List[int] = []
    defender_states = []
    for name, pos, age in agent_map.get_team_agents(enemy_team):
        if pos is not None:
            defender_positions.append(pos)
            defender_states.append((name, pos))

    # Build attacker positions directly from sensor (includes all teammates)
    attacker_positions: Dict[str, int] = {}
    if "agent" in sensors:
        for name, pos in sensors["agent"][1].items():
            if name.startswith(team):
                attacker_positions[name] = pos
    else:
        attacker_positions[agent_name] = current_pos

    if agent_name not in attacker_positions:
        attacker_positions[agent_name] = current_pos

    alive_attackers = sorted(attacker_positions.keys())
    active_searcher = team_cache.get("active_searcher", None)
    search_paused = bool(team_cache.get("search_paused", False))

    if all_flags_checked:
        search_paused = False
        team_cache.set("search_paused", False)
    elif active_searcher not in attacker_positions:
        # If the searcher is gone (captured/removed), assign the next alive searcher.
        active_searcher = alive_attackers[0] if alive_attackers else agent_name
        team_cache.set("active_searcher", active_searcher)
        team_cache.set("search_target", None)
        search_paused = False
        team_cache.set("search_paused", False)
    is_active_searcher = agent_name == active_searcher

    all_nodes = set(global_map_sensor.nodes())
    all_nodes_list = list(all_nodes)
    candidate_flags_sorted = tuple(sorted(set(candidate_flags)))
    flag_signature = f"nodes={len(all_nodes_list)}|flags={','.join(str(f) for f in candidate_flags_sorted)}"

    # Precompute once per game/map: for each candidate flag, nodes in capture region.
    capture_region_signature = f"capture_radius={capture_radius}|{flag_signature}"
    cached_capture_regions = team_cache.get("_capture_region_by_flag", None)
    if (
        team_cache.get("_capture_region_signature", None) != capture_region_signature
        or not isinstance(cached_capture_regions, dict)
    ):
        rebuilt_capture_regions = {}
        for flag in candidate_flags_sorted:
            rebuilt_capture_regions[flag] = [
                nid for nid in all_nodes_list
                if graph_distance(flag, nid) <= capture_radius
            ]
        team_cache.set("_capture_region_by_flag", rebuilt_capture_regions)
        team_cache.set("_capture_region_signature", capture_region_signature)
        cached_capture_regions = rebuilt_capture_regions
    capture_region_by_flag = {
        int(flag): list(nodes) for flag, nodes in cached_capture_regions.items()
    }

    # Precompute once per game/map: fallback stationary sensor coverage from candidate flags.
    stationary_sense_radius = team_cache.get("stationary_sense_radius", DEFAULT_STATIONARY_SENSE_RADIUS)
    stationary_cover_signature = f"stationary_radius={stationary_sense_radius}|{flag_signature}"
    cached_stationary_cover = team_cache.get("_fallback_stationary_cover_nodes", None)
    if (
        team_cache.get("_fallback_stationary_cover_signature", None) != stationary_cover_signature
        or not isinstance(cached_stationary_cover, list)
    ):
        covered_nodes = set()
        for flag in candidate_flags_sorted:
            flag_node = global_map_sensor.nodes[flag]
            for nid in all_nodes_list:
                n_data = global_map_sensor.nodes[nid]
                eucl = ((n_data['x'] - flag_node['x'])**2 + (n_data['y'] - flag_node['y'])**2)**0.5
                if eucl <= stationary_sense_radius:
                    covered_nodes.add(nid)
        cached_stationary_cover = list(covered_nodes)
        team_cache.set("_fallback_stationary_cover_nodes", cached_stationary_cover)
        team_cache.set("_fallback_stationary_cover_signature", stationary_cover_signature)
    fallback_stationary_covered_nodes = cached_stationary_cover

    V_safe = get_V_safe(defender_positions)
    no_enter_nodes = set()
    no_enter_blockers: Dict[int, List[str]] = {}
    if defender_positions:
        for nid in all_nodes:
            blockers = []
            for dname, dpos in defender_states:
                if graph_distance(nid, dpos) <= STRICT_NO_ENTER_HOPS:
                    blockers.append(f"{dname}@{dpos}")
            if blockers:
                no_enter_nodes.add(nid)
                no_enter_blockers[nid] = blockers

    target_region_nodes = get_target_region_nodes(real_flags)

    # ===== DEBUG HELPER =====
    DEBUG_PHASE_ONLY = None  # Set to None to run all phases. "rendezvous"/"search"/"wait" to freeze after that phase

    # Only print once per tick (first agent handles team-level prints)
    _is_first_agent = not team_cache.get(f"_dbg_tick_{current_time}", False)
    if _is_first_agent:
        team_cache.set(f"_dbg_tick_{current_time}", True)

    def get_safe_subgraph() -> nx.Graph:
        """Return the subgraph excluding only no_enter_nodes (undirected).
        Agents may transit through sensed areas but never within STRICT_NO_ENTER_HOPS of a defender."""
        safe_nodes = all_nodes - no_enter_nodes
        if not safe_nodes:
            return nx.Graph()
        sub = agent_map.graph.subgraph(safe_nodes)
        return sub.to_undirected() if sub.is_directed() else sub

    def move_toward_wait_node_safe(wait_node: int):
        """Move toward wait_node using only V_safe paths.
        Returns the next node, or None if unreachable via safe subgraph."""
        if current_pos == wait_node:
            return current_pos
        safe_sub = get_safe_subgraph()
        if current_pos not in safe_sub:
            # Agent is outside V_safe — move to nearest safe neighbor
            safe_neighbors = [n for n in agent_map.graph.neighbors(current_pos) if n in safe_sub]
            if not safe_neighbors:
                return None  # no safe neighbor reachable
            # Pick the safe neighbor closest to wait_node
            return min(safe_neighbors, key=lambda n: graph_distance(n, wait_node))
        try:
            path = nx.shortest_path(safe_sub, current_pos, wait_node)
            return path[min(speed, len(path) - 1)]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None  # wait_node not reachable via safe subgraph

    def compute_personal_wait(target_flags: Set[int]):
        """Compute a personal best wait node from the agent's reachable safe component."""
        safe_sub = get_safe_subgraph()
        if current_pos not in safe_sub:
            # Find reachable safe nodes via neighbors
            safe_neighbors = [n for n in agent_map.graph.neighbors(current_pos) if n in safe_sub]
            if not safe_neighbors:
                return current_pos  # stuck, stay put
            reachable = set()
            for sn in safe_neighbors:
                reachable.update(nx.node_connected_component(safe_sub, sn))
        else:
            reachable = set(nx.node_connected_component(safe_sub, current_pos))

        if not reachable:
            return current_pos

        # Find boundary nodes in the reachable safe component
        boundary = set()
        for v in reachable:
            for nb in agent_map.graph.neighbors(v):
                if nb not in reachable:
                    boundary.add(v)
                    break
        if not boundary:
            boundary = reachable

        # Pick the boundary node closest to any target flag
        return min(boundary, key=lambda v: min(graph_distance(v, f) for f in target_flags))

    def describe_search_block(flag_target: int) -> Dict[int, List[str]]:
        if current_pos == flag_target:
            return {}
        neighbors = list(agent_map.graph.neighbors(current_pos))
        d_pos_to_target = graph_distance(current_pos, flag_target)
        shortest_candidates = [
            n for n in neighbors
            if graph_distance(current_pos, n) + graph_distance(n, flag_target) == d_pos_to_target
        ]
        blocked_info = {}
        for nid in shortest_candidates:
            if nid in no_enter_nodes:
                blocked_info[nid] = no_enter_blockers.get(nid, [])
        return blocked_info

    def compute_wait_plan(target_flags: Set[int]):
        if not target_flags:
            return None, None, -1, set()

        V_T = get_target_region_nodes(target_flags)

        if not V_T:
            return None, None, -1, set()

        boundary_V_safe = set()
        safe_for_wait = V_safe - no_enter_nodes
        if not safe_for_wait:
            safe_for_wait = V_safe
        for v in safe_for_wait:
            for nb in agent_map.graph.neighbors(v):
                if nb not in safe_for_wait:
                    boundary_V_safe.add(v)
                    break
        if not boundary_V_safe:
            boundary_V_safe = safe_for_wait if safe_for_wait else (all_nodes - no_enter_nodes)

        def compute_n_win(v_target: int, attacker_pos: int) -> int:
            return sum(1 for d in defender_positions
                       if defender_advantage(v_target, d, attacker_pos) < 0)

        def compute_best_vt_for_boundary(v_prime: int):
            best = None
            best_key = (-1, float('-inf'))
            for v in V_T:
                n_win = compute_n_win(v, v_prime)
                worst_adv = max(defender_advantage(v, d, v_prime) for d in defender_positions) if defender_positions else 0
                key = (n_win, -worst_adv)
                if key > best_key:
                    best_key = key
                    best = v
            return best, best_key[0], best_key[1]

        best_wait = None
        best_wait_nwin = -1
        best_wait_vt = None
        for v_prime in boundary_V_safe:
            if not defender_positions:
                best_wait = min(boundary_V_safe, key=lambda v: min(graph_distance(v, f) for f in target_flags))
                best_wait_vt = min(V_T, key=lambda v: graph_distance(best_wait, v))
                best_wait_nwin = 0
                break
            vt, nwin, _ = compute_best_vt_for_boundary(v_prime)
            if nwin > best_wait_nwin or (nwin == best_wait_nwin and vt is not None):
                best_wait_nwin = nwin
                best_wait = v_prime
                best_wait_vt = vt

        if best_wait is None:
            best_wait = current_pos
        return best_wait, best_wait_vt, best_wait_nwin, V_T

    def fallback_wait_move_from_strike() -> int:
        """When strike is blocked, recompute a wait node and move toward it."""
        if not real_flags:
            return current_pos

        group_members = set(team_cache.get("group_members", alive_attackers))
        my_wait = None
        if agent_name in group_members:
            best_wait, _, _, _ = compute_wait_plan(real_flags)
            my_wait = best_wait

        if my_wait is None:
            my_wait = compute_personal_wait(real_flags)

        target = move_toward_wait_node_safe(my_wait)
        if target is None:
            target = current_pos
        return target

    def print_strike_fallback(context: str, strike_target: int, fallback_target: int):
        # TEMP DEBUG: only log when strike fallback is actually triggered.
        debug(
            f"[t={current_time}][{agent_name}] STRIKE-FALLBACK: context={context}, "
            f"pos={current_pos}, strike_target={strike_target}, fallback_next={fallback_target}, "
            f"team_phase={team_cache.get('phase', None)}"
        )

    # ===== ESCAPE: if agent is inside no_enter_nodes, flee immediately =====
    phase = team_cache.get("phase", "rendezvous")
    solo_phase = cache.get("solo_phase", None)
    strike_active = phase == "strike" or solo_phase == "strike"
    if current_pos in no_enter_nodes and not strike_active:
        # Find neighbors outside no_enter_nodes
        escape_candidates = [n for n in agent_map.graph.neighbors(current_pos) if n not in no_enter_nodes]
        if escape_candidates:
            # Pick the escape neighbor that maximizes min distance to any defender
            def escape_score(n):
                if not defender_positions:
                    return 0
                return min(graph_distance(n, d) for d in defender_positions)
            escape_target = max(escape_candidates, key=escape_score)
        else:
            # All immediate neighbors are also in no_enter — try 2-hop escape via safe subgraph
            escape_target = None
            best_dist = -1
            for n1 in agent_map.graph.neighbors(current_pos):
                for n2 in agent_map.graph.neighbors(n1):
                    if n2 not in no_enter_nodes:
                        d = min(graph_distance(n2, d) for d in defender_positions) if defender_positions else 0
                        if d > best_dist:
                            best_dist = d
                            escape_target = n1  # move toward the intermediate node
            if escape_target is None:
                escape_target = current_pos  # truly stuck, hold position

        debug(
            f"[t={current_time}][{agent_name}] ESCAPE: pos={current_pos} is in no_enter_nodes, "
            f"fleeing to {escape_target}, blockers={no_enter_blockers.get(current_pos, [])}"
        )
        state["action"] = escape_target
        return discovered_flags

    # ===== PHASE STATE MACHINE =====
    if strike_active and current_pos in no_enter_nodes:
        debug(
            f"[t={current_time}][{agent_name}] ESCAPE-BYPASS: pos={current_pos}, "
            f"phase={phase}, solo_phase={solo_phase}"
        )

    # Transition: if flag found and still in rendezvous/search, advance
    if real_flags and phase in ("rendezvous", "search"):
        if phase == "rendezvous":
            meeting_point = team_cache.get("meeting_point", None)
            n_team = len(attacker_positions)
            all_at_meeting = (meeting_point is not None and n_team >= 1 and
                all(p == meeting_point for p in attacker_positions.values()))
            if all_at_meeting:
                phase = "wait"
                team_cache.set("phase", phase)
                if _is_first_agent:
                    info(f"[t={current_time}][{agent_name}] TRANSITION rendezvous -> wait (flag found)")
        else:
            phase = "wait"
            team_cache.set("phase", phase)
            if _is_first_agent:
                info(f"[t={current_time}][{agent_name}] TRANSITION search -> wait (flag found)")

    # ===== PHASE 1: RENDEZVOUS =====
    if phase == "rendezvous":
        meeting_point = team_cache.get("meeting_point", None)
        rendezvous_candidates = V_safe - no_enter_nodes
        if not rendezvous_candidates:
            rendezvous_candidates = V_safe

        # Only recompute meeting point ONCE per tick (first agent)
        # This prevents mid-tick desync where red_0 updates the meeting point
        # and red_1-4 read the new value, causing red_0 to lag behind.
        last_recompute_tick = team_cache.get("_meeting_recompute_tick", -1)
        if last_recompute_tick < current_time:
            team_cache.set("_meeting_recompute_tick", current_time)

            if meeting_point is None or meeting_point not in rendezvous_candidates:
                # v* = argmin_{v in V_safe} max_j d_g(A^j, v)
                best_v = None
                best_cost = float('inf')
                for v in rendezvous_candidates:
                    worst_dist = max(graph_distance(a_pos, v) for a_pos in attacker_positions.values())
                    if worst_dist < best_cost:
                        best_cost = worst_dist
                        best_v = v
                meeting_point = best_v
                team_cache.set("meeting_point", meeting_point)
            else:
                # Check: should we keep or update meeting point? (spec step 14-15)
                current_worst = max(graph_distance(a_pos, meeting_point)
                                   for a_pos in attacker_positions.values())
                needs_update = False
                for v in rendezvous_candidates:
                    worst_dist = max(graph_distance(a_pos, v) for a_pos in attacker_positions.values())
                    if worst_dist < current_worst:
                        needs_update = True
                        break
                if needs_update:
                    best_v = None
                    best_cost = float('inf')
                    for v in rendezvous_candidates:
                        worst_dist = max(graph_distance(a_pos, v) for a_pos in attacker_positions.values())
                        if worst_dist < best_cost:
                            best_cost = worst_dist
                            best_v = v
                    if best_v is not None:
                        meeting_point = best_v
                        team_cache.set("meeting_point", meeting_point)
        else:
            # Not the first agent this tick — just read the already-computed meeting point
            meeting_point = team_cache.get("meeting_point", None)

        if meeting_point is None:
            state["action"] = current_pos
            return discovered_flags

        # Print once per tick
        if _is_first_agent:
            debug(f"[t={current_time}][{agent_name}] RENDEZVOUS: target={meeting_point}")

        # Move toward meeting point
        if current_pos == meeting_point:
            target = current_pos
        else:
            target = pick_safest_on_path_neighbor(current_pos, meeting_point, defender_positions)

        # Check if all attackers reached meeting point
        all_arrived = (len(attacker_positions) >= 1 and
                       all(p == meeting_point for p in attacker_positions.values()))
        timed_out = (current_time >= RENDEZVOUS_TIMEOUT_T and not all_arrived)
        if all_arrived or timed_out:
            next_phase = "wait" if real_flags else "search"
            team_cache.set("phase", next_phase)
            phase = next_phase
            # Record group membership: agents at meeting point form the group
            group_members = [name for name, pos in attacker_positions.items() if pos == meeting_point]
            solo_members = [name for name, pos in attacker_positions.items() if pos != meeting_point]
            team_cache.set("group_members", group_members)
            team_cache.set("solo_members", solo_members)
            if _is_first_agent:
                if timed_out:
                    info(
                        f"[t={current_time}][{agent_name}] RENDEZVOUS TIMEOUT -> {next_phase} "
                        f"(group={group_members}, solo={solo_members})"
                    )
                else:
                    info(
                        f"[t={current_time}][{agent_name}] RENDEZVOUS COMPLETE -> {next_phase} "
                        f"(group={group_members})"
                    )

        # On timeout, switch to next phase immediately in this tick.
        if not timed_out:
            state["action"] = target
            return discovered_flags

    # ===== PHASE 2: SEARCH FOR FLAG =====
    # Team can be in wait while active_searcher keeps searching remaining candidates.
    run_search_logic = False
    if phase == "search":
        if real_flags:
            # First real flag found: switch team to wait immediately.
            team_cache.set("phase", "wait")
            phase = "wait"
            if is_active_searcher and not all_flags_checked and not search_paused:
                run_search_logic = True
                if _is_first_agent:
                    debug(
                        f"[t={current_time}][{agent_name}] SEARCH-CONTINUE: explorer={active_searcher}, "
                        f"real_flags={real_flags}, unresolved={len(unresolved_candidate_flags)}"
                    )
        else:
            run_search_logic = True
    elif phase == "wait" and is_active_searcher and real_flags and not all_flags_checked and not search_paused:
        run_search_logic = True

    if run_search_logic:
        if not is_active_searcher:
            meeting_point = team_cache.get("meeting_point", None)
            if meeting_point is None:
                target = current_pos
            else:
                target = pick_safest_on_path_neighbor(current_pos, meeting_point, defender_positions)
            if _is_first_agent:
                debug(
                    f"[t={current_time}][{agent_name}] SEARCH-HOLD: explorer={active_searcher}, "
                    f"holding_at_rendezvous={meeting_point}"
                )
            state["action"] = target
            return discovered_flags

        # Opportunistic strike while continuing search after real flag discovery.
        if real_flags and target_region_nodes:
            if defender_positions:
                def searcher_strike_key(v):
                    n_win = sum(1 for d in defender_positions
                                if defender_advantage(v, d, current_pos) < 0)
                    worst_adv = max(defender_advantage(v, d, current_pos) for d in defender_positions)
                    return (n_win, -worst_adv)

                strike_vt = max(target_region_nodes, key=searcher_strike_key)
                best_nwin = sum(1 for d in defender_positions
                                if defender_advantage(strike_vt, d, current_pos) < 0)
                n_loss = len(defender_positions) - best_nwin
                searcher_should_strike = (n_loss == 0)
            else:
                strike_vt = min(target_region_nodes, key=lambda v: min(graph_distance(v, f) for f in real_flags))
                searcher_should_strike = True

            if searcher_should_strike and strike_vt is not None and DEBUG_PHASE_ONLY != "wait":
                cache.set("solo_strike_target", strike_vt)
                cache.set("solo_phase", "strike")
                target = pick_strike_on_path_neighbor(current_pos, strike_vt, defender_positions)
                debug(f"[t={current_time}][{agent_name}] SEARCH-STRIKE: target={strike_vt}, next={target}")
                state["action"] = target
                return discovered_flags

        unsensed_candidate_flags = unresolved_candidate_flags
        if not unsensed_candidate_flags:
            team_cache.set("search_paused", False)
            team_cache.set("search_target", None)
            phase = "wait"
            info(
                f"[t={current_time}][{agent_name}] SEARCHER -> wait "
                f"(all candidate flags checked, real_flags={real_flags})"
            )
            # Fall through to wait phase below
        else:
            search_target = team_cache.get("search_target", None)
            chosen_target = None
            target = current_pos
            blocked_flags = []

            # Sticky search target: once selected, keep it until resolved or unreachable.
            if search_target in unsensed_candidate_flags:
                next_step = pick_safest_on_path_neighbor(current_pos, search_target, defender_positions)
                is_blocked = (current_pos != search_target and next_step == current_pos)
                if is_blocked:
                    block_detail = describe_search_block(search_target)
                    blocked_flags.append({"flag": search_target, "blocked_by": block_detail})
                    debug(
                        f"[t={current_time}][{agent_name}] SEARCH-BLOCKED: explorer={active_searcher}, "
                        f"flag={search_target}, block_detail={block_detail}"
                    )
                else:
                    chosen_target = search_target
                    target = next_step

            if chosen_target is None:
                remaining_flags = [v for v in unsensed_candidate_flags if v != search_target]
                if defender_positions:
                    ranked_flags = sorted(
                        remaining_flags,
                        key=lambda v: (max(defender_advantage(v, d, current_pos) for d in defender_positions), graph_distance(current_pos, v)),
                    )
                else:
                    ranked_flags = sorted(remaining_flags, key=lambda v: graph_distance(current_pos, v))

                for flag_target in ranked_flags:
                    next_step = pick_safest_on_path_neighbor(current_pos, flag_target, defender_positions)
                    is_blocked = (current_pos != flag_target and next_step == current_pos)
                    if is_blocked:
                        block_detail = describe_search_block(flag_target)
                        blocked_flags.append({"flag": flag_target, "blocked_by": block_detail})
                        debug(
                            f"[t={current_time}][{agent_name}] SEARCH-BLOCKED: explorer={active_searcher}, "
                            f"flag={flag_target}, block_detail={block_detail}"
                        )
                        continue
                    chosen_target = flag_target
                    target = next_step
                    break

            if chosen_target is None:
                team_cache.set("search_target", None)
                if real_flags:
                    # Pause searching for this searcher and switch it to wait.
                    team_cache.set("search_paused", True)
                    phase = "wait"
                    info(
                        f"[t={current_time}][{agent_name}] SEARCHER -> wait "
                        f"(all unresolved candidates blocked, blocked={blocked_flags})"
                    )
                    # Fall through to wait phase below
                else:
                    debug(
                        f"[t={current_time}][{agent_name}] SEARCH: explorer={active_searcher}, pos={current_pos}, "
                        f"all_paths_blocked={blocked_flags}"
                    )
                    state["action"] = current_pos
                    return discovered_flags
            else:
                team_cache.set("search_paused", False)
                search_target = chosen_target
                team_cache.set("search_target", search_target)
                debug(
                    f"[t={current_time}][{agent_name}] SEARCH: explorer={active_searcher}, pos={current_pos}, "
                    f"target={search_target}, next={target}, unsensed_flags={len(unsensed_candidate_flags)}, "
                    f"blocked_skipped={blocked_flags}"
                )
                state["action"] = target
                return discovered_flags

    # ===== PHASE 2.5: WAIT AT BEST ATTACKING POSITION =====
    if phase == "wait":
        if not real_flags:
            team_cache.set("phase", "search")
            state["action"] = current_pos
            return discovered_flags

        group_members = set(team_cache.get("group_members", alive_attackers))
        is_group = agent_name in group_members
        # Count alive group members for strike condition
        alive_group = [n for n in alive_attackers if n in group_members]
        n_group = len(alive_group)

        if is_group:
            # ---- GROUP: shared wait, group-level strike ----
            best_wait, best_wait_vt, best_wait_nwin, V_T = compute_wait_plan(real_flags)
            if best_wait is None or not V_T:
                state["action"] = current_pos
                return discovered_flags

            team_cache.set("wait_target", best_wait)

            # Strike condition based on group size vs defenders
            n_defenders = len(defender_positions)
            n_loss = n_defenders - best_wait_nwin
            current_k = N_WIN_WEIGHT_K_LATE if current_time >= N_WIN_WEIGHT_SWITCH_T else N_WIN_WEIGHT_K
            weighted_nloss = current_k * n_loss
            advantage_index = best_wait_nwin - weighted_nloss
            # Group needs enough members to absorb losses and still have survivors to capture
            should_strike = best_wait_nwin >= weighted_nloss and n_group > n_loss

            if _is_first_agent:
                debug(
                    f"[t={current_time}][{agent_name}] WAIT-GROUP: wait_node={best_wait}, strike_vt={best_wait_vt}, "
                    f"N_win={best_wait_nwin}, K={current_k}, N_loss={n_loss}, adv_idx={advantage_index}, "
                    f"group_size={n_group}, strike={'YES' if should_strike else 'no'}"
                )

            if should_strike and best_wait_vt is not None and DEBUG_PHASE_ONLY != "wait":
                team_cache.set("phase", "strike")
                team_cache.set("strike_target", best_wait_vt)
                phase = "strike"
                if _is_first_agent:
                    info(f"[t={current_time}][{agent_name}] WAIT-GROUP -> STRIKE: target={best_wait_vt}")
            else:
                target = move_toward_wait_node_safe(best_wait)
                my_wait = best_wait
                if target is None:
                    my_wait = compute_personal_wait(real_flags)
                    target = move_toward_wait_node_safe(my_wait)
                    if target is None:
                        target = current_pos
                flag_dists = {f: graph_distance(current_pos, f) for f in real_flags}
                debug(
                    f"[t={current_time}][{agent_name}] WAIT-GROUP-MOVE: pos={current_pos}, next={target}, "
                    f"wait_node={my_wait}{' (personal)' if my_wait != best_wait else ''}, flag_dists={flag_dists}"
                )
                state["action"] = target
                return discovered_flags
        else:
            # ---- SOLO: individual wait, individual strike ----
            my_wait = compute_personal_wait(real_flags)
            # Solo strike: can this agent reach a capture node without being tagged?
            solo_vt = None
            V_T = target_region_nodes
            if V_T and defender_positions:
                # Find best capture node for this solo agent
                def solo_strike_key(v):
                    n_win = sum(1 for d in defender_positions
                                if defender_advantage(v, d, current_pos) < 0)
                    worst_adv = max(defender_advantage(v, d, current_pos) for d in defender_positions)
                    return (n_win, -worst_adv)
                solo_vt = max(V_T, key=solo_strike_key)
                best_nwin = sum(1 for d in defender_positions
                                if defender_advantage(solo_vt, d, current_pos) < 0)
                n_loss = len(defender_positions) - best_nwin
                solo_should_strike = (n_loss == 0)  # solo only strikes if no defenders can reach in time
            elif V_T:
                solo_vt = min(V_T, key=lambda v: min(graph_distance(v, f) for f in real_flags))
                solo_should_strike = True
                n_loss = 0
                best_nwin = 0
            else:
                solo_should_strike = False
                n_loss = 0
                best_nwin = 0

            debug(
                f"[t={current_time}][{agent_name}] WAIT-SOLO: pos={current_pos}, wait_node={my_wait}, "
                f"solo_vt={solo_vt}, N_loss={n_loss}, strike={'YES' if solo_should_strike else 'no'}"
            )

            if solo_should_strike and solo_vt is not None and DEBUG_PHASE_ONLY != "wait":
                # Solo strike — go directly, don't set team phase
                cache.set("solo_strike_target", solo_vt)
                cache.set("solo_phase", "strike")
                target = pick_strike_on_path_neighbor(current_pos, solo_vt, defender_positions)
                debug(f"[t={current_time}][{agent_name}] SOLO-STRIKE: target={solo_vt}, next={target}")
                state["action"] = target
                return discovered_flags
            else:
                target = move_toward_wait_node_safe(my_wait)
                if target is None:
                    target = current_pos
                flag_dists = {f: graph_distance(current_pos, f) for f in real_flags}
                debug(
                    f"[t={current_time}][{agent_name}] WAIT-SOLO-MOVE: pos={current_pos}, next={target}, "
                    f"wait_node={my_wait}, flag_dists={flag_dists}"
                )
                state["action"] = target
                return discovered_flags

    # ===== SOLO STRIKE (individual agent, not team phase) =====
    solo_phase = cache.get("solo_phase", None)
    if solo_phase == "strike":
        solo_target = cache.get("solo_strike_target", None)
        if solo_target is not None:
            target = pick_strike_on_path_neighbor(current_pos, solo_target, defender_positions)
            debug(f"[t={current_time}][{agent_name}] SOLO-STRIKE: pos={current_pos}, target={solo_target}, next={target}")
            state["action"] = target
            return discovered_flags

    # ===== PHASE 3: STRIKE (group only — solo agents use solo strike above) =====
    group_members_strike = set(team_cache.get("group_members", alive_attackers))
    if phase == "strike" and agent_name in group_members_strike:
        strike_target = team_cache.get("strike_target", None)

        if strike_target is None:
            # Do not retarget during strike. If target was lost, exit strike cleanly.
            team_cache.set("phase", "wait")
            target = fallback_wait_move_from_strike()
            state["action"] = target
            return discovered_flags

        # Move toward fixed strike target.
        target = pick_strike_on_path_neighbor(current_pos, strike_target, defender_positions)
        debug(f"[t={current_time}][{agent_name}] STRIKE: pos={current_pos}, target={strike_target}, next={target}")
        state["action"] = target
        return discovered_flags

    # ===== FROZEN (debug) =====
    if phase == "frozen":
        state["action"] = current_pos
        return discovered_flags

    # Fallback
    state["action"] = current_pos
    return discovered_flags


def map_strategy(agent_config):
    strategies = {}
    for name in agent_config.keys():
        strategies[name] = attacker_strategy
    return strategies
