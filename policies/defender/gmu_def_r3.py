ATTACKER_CAPTURE_RADIUS = 2  # from game config (red capture_radius)

# Deployment weight k: only used in Phase 1 (static position selection).
# Coverage condition: DEPLOY_WEIGHT * dist(s, p) + dist(p, t) <= dist(b, t)
# Set to 0.0 for pure static coverage (no deployment penalty).
DEPLOY_WEIGHT = 0

# Phase 1 position search mode: "greedy" or "exhaustive"
# "exhaustive" runs brute-force up to MAX_EXHAUSTIVE_SECONDS then uses best found.
SEARCH_MODE = "greedy"
# SEARCH_MODE = "exhaustive"
MAX_EXHAUSTIVE_SECONDS = 500


def strategy(state):
    from typing import Dict, List, Tuple, Any
    import networkx as nx
    from lib.core.console import info, debug

    # ===== AGENT CONTROLLER =====
    agent_ctrl = state["agent_controller"]
    current_pos: int = state["curr_pos"]
    current_time: int = state["time"]
    team: str = agent_ctrl.team
    red_payoff: float = state["payoff"]["red"]
    blue_payoff: float = state["payoff"]["blue"]

    speed: float = agent_ctrl.speed
    tagging_radius: float = agent_ctrl.tagging_radius

    cache = agent_ctrl.cache
    team_cache = agent_ctrl.team_cache

    # ===== AGENT MAP =====
    agent_map = agent_ctrl.map
    global_map_payload: Dict[str, Any] = state["sensor"]["global_map"][1]
    global_map_sensor: nx.Graph = global_map_payload["graph"]
    global_map_apsp = global_map_payload.get("apsp")

    nodes_data: Dict[int, Dict[str, Any]] = {
        node_id: global_map_sensor.nodes[node_id]
        for node_id in global_map_sensor.nodes()
    }
    edges_data: Dict[int, Dict[str, Any]] = {}
    for idx, (u, v, data) in enumerate(global_map_sensor.edges(data=True)):
        edges_data[idx] = {"source": u, "target": v, **data}

    agent_map.attach_networkx_graph(
        nodes_data,
        edges_data,
        apsp_lookup=global_map_apsp if isinstance(global_map_apsp, dict) else None,
    )
    agent_map.update_time(current_time)
    agent_map.update_agent_position(team, agent_ctrl.name, current_pos, current_time)

    if "custom_team" in state["sensor"]:
        for teammate_name, teammate_pos in state["sensor"]["custom_team"][1].items():
            agent_map.update_agent_position(team, teammate_name, teammate_pos, current_time)

    if "egocentric_agent" in state["sensor"]:
        _enemy = "red" if team == "blue" else "blue"
        for agent_name, node_id in state["sensor"]["egocentric_agent"][1].items():
            if agent_name.startswith(_enemy):
                agent_map.update_agent_position(_enemy, agent_name, node_id, current_time)

    # ===== SENSORS =====
    sensors: Dict[str, Tuple[Any, Dict[str, Any]]] = state["sensor"]

    if "flag" in sensors:
        real_flags: List[int] = sensors["flag"][1]["real_flags"]
        fake_flags: List[int] = sensors["flag"][1]["fake_flags"]

    if "custom_team" in sensors:
        teammates_sensor: Dict[str, int] = sensors["custom_team"][1]

    if "egocentric_agent" in sensors:
        nearby_agents: Dict[str, int] = sensors["egocentric_agent"][1]

    if "egocentric_map" in sensors:
        visible_nodes: Dict[int, Any] = sensors["egocentric_map"][1]["nodes"]
        visible_edges: List[Any] = sensors["egocentric_map"][1]["edges"]

    # ===== DECISION LOGIC =====
    target: int = current_pos

    # ----- Build sensor region -----
    sensor_region = set()
    for sname, (_, sdata) in sensors.items():
        if sname.startswith("stationary_"):
            sensor_region.update(sdata.get("covered_nodes", []))

    # ----- Dynamic defender count -----
    if "custom_team" in sensors:
        n_defenders = 1 + len(sensors["custom_team"][1])
    else:
        n_defenders = 1

    defender_starts: Dict[str, int] = {agent_ctrl.name: current_pos}
    if "custom_team" in sensors:
        for name, pos in sensors["custom_team"][1].items():
            defender_starts[name] = pos

    if n_defenders != team_cache.get("n_defenders_cached", -1):
        if not team_cache.get("phase2_active", False):
            # Before Phase 2: redo Phase 1 precomputation with new defender count.
            team_cache.set("positions_computed", False)
        # In Phase 2: do nothing — defender_starts is rebuilt from sensors every turn,
        # so the computation naturally adapts to any change in active defenders.

    # ----- Detect visible attackers from stationary sensors -----
    enemy_team: str = "red" if team == "blue" else "blue"
    visible_attackers: Dict[str, int] = {}
    for sname, (_, sdata) in sensors.items():
        if sname.startswith("stationary_"):
            for agent_name, agent_info in sdata.get("detected_agents", {}).items():
                if agent_name.startswith(enemy_team):
                    visible_attackers[agent_name] = agent_info["node_id"]

    # =========================================================
    # PHASE 1: ONE-TIME PRECOMPUTATION
    # Finds optimal static positions for all defenders.
    # Caches distance tables for reuse in Phase 2.
    # Only the first agent to run this block does the work;
    # subsequent agents read from team_cache.
    # =========================================================
    if not team_cache.get("positions_computed", False):
        if "flag" in sensors and agent_map.graph is not None and sensor_region:
            graph = agent_map.graph

            info(f"[{agent_ctrl.name}] ===== PHASE 1 PRECOMPUTATION START =====")
            debug(f"[{agent_ctrl.name}] Defenders: {n_defenders}, DEPLOY_WEIGHT: {DEPLOY_WEIGHT}")
            debug(f"[{agent_ctrl.name}] Defender starts: { {k: v for k, v in sorted(defender_starts.items())} }")
            debug(f"[{agent_ctrl.name}] Sensor region: {len(sensor_region)} nodes")

            # Step 2: Boundary nodes — sensor region nodes with a neighbor outside
            boundary_nodes = [
                n for n in sensor_region
                if any(nb not in sensor_region for nb in graph.neighbors(n))
            ]
            debug(f"[{agent_ctrl.name}] Boundary nodes ({len(boundary_nodes)}): {sorted(boundary_nodes)}")

            # Step 3: Target nodes — within attacker capture radius of any flag
            target_nodes = [
                n for n in graph.nodes()
                if any(
                    agent_map.shortest_path_length(n, f) <= ATTACKER_CAPTURE_RADIUS
                    for f in real_flags
                )
            ]
            debug(f"[{agent_ctrl.name}] Real flags: {real_flags}")
            debug(f"[{agent_ctrl.name}] Target nodes ({len(target_nodes)}): {sorted(target_nodes)}")

            if boundary_nodes and target_nodes:
                total_pairs = len(boundary_nodes) * len(target_nodes)
                debug(f"[{agent_ctrl.name}] Total (b, t) pairs: {total_pairs}")

                # Step 4: Distance tables
                # b_threat_boundary: dist from each boundary node to each target node
                b_threat_boundary = {
                    b: {t: agent_map.shortest_path_length(b, t) for t in target_nodes}
                    for b in boundary_nodes
                }
                # p_reach_all: dist from EVERY graph node to each target node
                # (stored for Phase 2 — candidates may be outside sensor_region)
                p_reach_all = {
                    p: {t: agent_map.shortest_path_length(p, t) for t in target_nodes}
                    for p in graph.nodes()
                }
                # Deployment distance (Phase 1 only, multiplied by DEPLOY_WEIGHT)
                s_positions = list(defender_starts.values())
                min_deploy_dist = {
                    p: min(agent_map.shortest_path_length(s, p) for s in s_positions)
                    for p in sensor_region
                }
                debug(f"[{agent_ctrl.name}] Distance tables computed.")

                # Step 5: Deployment-aware coverage sets (sensor_region only)
                sensor_list = list(sensor_region)
                covers_boundary = {
                    p: {
                        (b, t)
                        for b in boundary_nodes
                        for t in target_nodes
                        if DEPLOY_WEIGHT * min_deploy_dist[p] + p_reach_all[p][t] <= b_threat_boundary[b][t]
                    }
                    for p in sensor_region
                }
                best_single = max(sensor_region, key=lambda p: len(covers_boundary[p]))
                debug(f"[{agent_ctrl.name}] Best single position: node {best_single} "
                      f"covers {len(covers_boundary[best_single])}/{total_pairs} pairs")

                # Step 6: Greedy max-min coverage
                all_pairs = {(b, t) for b in boundary_nodes for t in target_nodes}
                coverage_count = {pair: 0 for pair in all_pairs}
                chosen = []

                debug(f"[{agent_ctrl.name}] --- Greedy max-min ({n_defenders} defenders) ---")
                for round_i in range(n_defenders):
                    min_cov = min(coverage_count.values())
                    bottleneck = {pair for pair, cnt in coverage_count.items() if cnt == min_cov}

                    effective_bottleneck = bottleneck
                    if not any(covers_boundary[p] & effective_bottleneck for p in sensor_list):
                        for tier in sorted(set(coverage_count.values())):
                            candidate = {pair for pair, cnt in coverage_count.items() if cnt == tier}
                            if any(covers_boundary[p] & candidate for p in sensor_list):
                                effective_bottleneck = candidate
                                debug(f"[{agent_ctrl.name}]   Round {round_i + 1}: "
                                      f"bottleneck tier {min_cov} uncoverable, falling back to tier {tier}")
                                break

                    best_p = max(sensor_list, key=lambda p: len(covers_boundary[p] & effective_bottleneck))
                    newly_covered = len(covers_boundary[best_p] & effective_bottleneck)
                    chosen.append(best_p)

                    for pair in covers_boundary[best_p]:
                        coverage_count[pair] += 1

                    new_min = min(coverage_count.values())
                    debug(f"[{agent_ctrl.name}]   Round {round_i + 1}: chose node {best_p}, "
                          f"covered {newly_covered}/{len(effective_bottleneck)}, "
                          f"min coverage: {min_cov} -> {new_min}")

                final_min = min(coverage_count.values())
                final_max = max(coverage_count.values())
                uncovered = sum(1 for c in coverage_count.values() if c == 0)
                debug(f"[{agent_ctrl.name}] Greedy — min: {final_min}, max: {final_max}, "
                      f"uncovered: {uncovered}/{len(all_pairs)}")
                debug(f"[{agent_ctrl.name}] Greedy chosen: {chosen}")

                # ---- Optional exhaustive position search ----
                if SEARCH_MODE == "exhaustive":
                    import itertools
                    import math
                    import time
                    import numpy as np

                    pair_list = list(all_pairs)
                    pair_idx = {pair: i for i, pair in enumerate(pair_list)}
                    n_pairs = len(pair_list)
                    n_positions = len(sensor_list)

                    cover_matrix = np.zeros((n_positions, n_pairs), dtype=np.int8)
                    for i, p in enumerate(sensor_list):
                        for pair in covers_boundary[p]:
                            cover_matrix[i, pair_idx[pair]] = 1

                    total_combos = math.comb(n_positions, n_defenders)
                    debug(f"\n[{agent_ctrl.name}] === EXHAUSTIVE POSITION SEARCH ===")
                    debug(f"[{agent_ctrl.name}] {n_positions} positions, {n_defenders} defenders, "
                          f"{total_combos:,} combos, budget {MAX_EXHAUSTIVE_SECONDS}s")

                    best_min_ex, best_combo_ex = -1, None
                    start_ex = time.time()
                    timed_out = False
                    REPORT_EVERY = max(1, total_combos // 100)

                    for count, idx_combo in enumerate(
                        itertools.combinations(range(n_positions), n_defenders)
                    ):
                        cov = cover_matrix[list(idx_combo), :].sum(axis=0).min()
                        if cov > best_min_ex:
                            best_min_ex, best_combo_ex = cov, idx_combo
                        if count > 0 and count % REPORT_EVERY == 0:
                            elapsed = time.time() - start_ex
                            eta = (total_combos - count) / (count / elapsed)
                            debug(f"[{agent_ctrl.name}]   {count:,}/{total_combos:,} "
                                  f"({100 * count / total_combos:.1f}%) | "
                                  f"best: {best_min_ex} | {elapsed:.1f}s | ETA: {eta:.0f}s")
                        if time.time() - start_ex > MAX_EXHAUSTIVE_SECONDS:
                            timed_out = True
                            break

                    elapsed_ex = time.time() - start_ex
                    status = f"TIMED OUT ({elapsed_ex:.1f}s)" if timed_out else f"done ({elapsed_ex:.2f}s)"
                    debug(f"[{agent_ctrl.name}] {status} — optimal: {best_min_ex}, "
                          f"positions: {[sensor_list[i] for i in best_combo_ex]}")
                    if not timed_out:
                        debug(f"[{agent_ctrl.name}] Greedy gap: {best_min_ex - final_min} "
                              f"({'optimal' if best_min_ex == final_min else 'suboptimal'})")
                    chosen = [sensor_list[i] for i in best_combo_ex]
                    debug(f"[{agent_ctrl.name}] === END EXHAUSTIVE ===\n")
                else:
                    debug(f"[{agent_ctrl.name}] (SEARCH_MODE='exhaustive' for brute-force comparison)")

                # Step 7: Exhaustive assignment — n! permutations, min total travel
                import itertools as _it
                import math as _m

                sorted_defenders = sorted(defender_starts.keys())
                n_assign = len(sorted_defenders)
                debug(f"[{agent_ctrl.name}] --- Exhaustive assignment "
                      f"({n_assign}! = {_m.factorial(n_assign)} perms) ---")

                best_cost, best_perm = float("inf"), None
                for perm in _it.permutations(range(len(chosen))):
                    cost = sum(
                        agent_map.shortest_path_length(
                            defender_starts[sorted_defenders[i]], chosen[perm[i]]
                        )
                        for i in range(n_assign)
                    )
                    if cost < best_cost:
                        best_cost, best_perm = cost, perm

                static_assignment = {
                    sorted_defenders[i]: chosen[best_perm[i]]
                    for i in range(n_assign)
                }
                debug(f"[{agent_ctrl.name}] Best assignment (total dist={best_cost}):")
                for name in sorted_defenders:
                    d = agent_map.shortest_path_length(
                        defender_starts[name], static_assignment[name]
                    )
                    debug(f"[{agent_ctrl.name}]   {name} -> node {static_assignment[name]} (dist={d})")

                info(f"[{agent_ctrl.name}] ===== PHASE 1 PRECOMPUTATION END =====")

                # Cache everything needed for Phase 2
                team_cache.set("static_assignment", static_assignment)
                team_cache.set("boundary_nodes", boundary_nodes)
                team_cache.set("target_nodes", target_nodes)
                team_cache.set("b_threat_boundary", b_threat_boundary)
                team_cache.set("p_reach_all", p_reach_all)
                team_cache.set("n_defenders_cached", n_defenders)
                team_cache.set("positions_computed", True)

    # =========================================================
    # PHASE TRANSITION CHECK
    # Switch to Phase 2 permanently when all defenders have
    # reached their static positions OR any attacker appears.
    # =========================================================
    static_assignment: Dict[str, int] = team_cache.get("static_assignment", {})
    all_deployed = bool(static_assignment) and all(
        defender_starts.get(name) == static_assignment.get(name)
        for name in defender_starts
    )
    if not team_cache.get("phase2_active", False) and (all_deployed or bool(visible_attackers)):
        team_cache.set("phase2_active", True)
        reason = "all deployed" if all_deployed else "attacker detected"
        info(f"[{agent_ctrl.name}] *** PHASE 2 ACTIVE ({reason}) at t={current_time} ***")

    # =========================================================
    # PHASE 2: STEPWISE EXHAUSTIVE SEARCH (every turn)
    # Enumerates all combinations of 1-step moves for all
    # defenders and picks the joint move that maximises
    # min-coverage over all threat pairs.
    # Priority: real attacker (b,t) pairs > boundary (b,t) pairs.
    # Multiple attackers at the same node require coverage >= count.
    # Only the first agent per turn runs the computation;
    # all others read the cached result.
    # =========================================================
    if team_cache.get("phase2_active", False) and team_cache.get("positions_computed", False):

        if team_cache.get("dynamic_time") != current_time:
            import itertools as _it
            import numpy as np

            graph = agent_map.graph
            boundary_nodes: List[int] = team_cache.get("boundary_nodes", [])
            target_nodes: List[int] = team_cache.get("target_nodes", [])
            b_threat_boundary: Dict[int, Dict[int, int]] = team_cache.get("b_threat_boundary", {})
            p_reach_all: Dict[int, Dict[int, int]] = team_cache.get("p_reach_all", {})

            # Build threat map: attacker positions (stacked) + boundary nodes
            threat: Dict[int, int] = {}
            attacker_positions = set()
            for atk_pos in visible_attackers.values():
                threat[atk_pos] = threat.get(atk_pos, 0) + 1
                attacker_positions.add(atk_pos)
            for b in boundary_nodes:
                if b not in threat:
                    threat[b] = 1

            # Extend distance table for attacker positions not already in b_threat_boundary
            threat_dist: Dict[int, Dict[int, int]] = dict(b_threat_boundary)
            for atk_pos in attacker_positions:
                if atk_pos not in threat_dist:
                    threat_dist[atk_pos] = {
                        t: agent_map.shortest_path_length(atk_pos, t)
                        for t in target_nodes
                    }

            # All threat pairs and their required coverage
            all_threat_pairs = [(b, t) for b in threat for t in target_nodes]
            n_pairs = len(all_threat_pairs)
            pair_to_idx = {pair: j for j, pair in enumerate(all_threat_pairs)}
            required_arr = np.array([threat[b] for b, t in all_threat_pairs], dtype=np.int16)

            # Indices of attacker-specific pairs (for priority scoring)
            atk_pair_indices = [
                pair_to_idx[(b, t)]
                for b in attacker_positions
                for t in target_nodes
                if (b, t) in pair_to_idx
            ]

            # Candidates for each defender: current position + 1-hop neighbors
            sorted_defenders = sorted(defender_starts.keys())
            candidates_per_defender: List[List[int]] = []
            all_candidate_nodes: set = set()
            for name in sorted_defenders:
                pos = defender_starts[name]
                candidates = [pos] + list(graph.neighbors(pos))
                candidates_per_defender.append(candidates)
                all_candidate_nodes.update(candidates)

            # Coverage matrix for all candidate nodes
            # C[i, j] = 1 if unique_cands[i] covers threat pair j
            unique_cands = sorted(all_candidate_nodes)
            cand_idx_map = {p: i for i, p in enumerate(unique_cands)}
            n_cands = len(unique_cands)

            cov_matrix = np.zeros((n_cands, n_pairs), dtype=np.int8)
            for p in unique_cands:
                ci = cand_idx_map[p]
                p_reach_p = p_reach_all.get(p)
                if p_reach_p is None:
                    # Node not in precomputed table — compute on the fly
                    p_reach_p = {
                        t: agent_map.shortest_path_length(p, t) for t in target_nodes
                    }
                for j, (b, t) in enumerate(all_threat_pairs):
                    if b in attacker_positions:
                        # Strict: defender must beat the attacker by ≥1 step
                        # because the attacker can also move next turn.
                        if p_reach_p.get(t, float("inf")) < threat_dist[b][t]:
                            cov_matrix[ci, j] = 1
                    else:
                        # Non-strict: a tie (advantage = 0) is acceptable for
                        # hypothetical boundary attackers.
                        if p_reach_p.get(t, float("inf")) <= threat_dist[b][t]:
                            cov_matrix[ci, j] = 1

            # Exhaustive Cartesian product over 1-step moves (vectorised)
            cand_idx_lists = [
                [cand_idx_map[p] for p in candidates_per_defender[k]]
                for k in range(len(sorted_defenders))
            ]

            all_combos = np.array(list(_it.product(*cand_idx_lists)), dtype=np.int32)
            # all_combos shape: (n_combos, n_defenders)

            # Sum coverage rows for each combo
            cov_all = np.zeros((len(all_combos), n_pairs), dtype=np.int16)
            for k in range(len(sorted_defenders)):
                cov_all += cov_matrix[all_combos[:, k], :]

            # Margin = coverage count - required
            margins_all = cov_all - required_arr[np.newaxis, :]

            # Three-level score (lexicographic, all maximised):
            #   1. atk_margin     : min margin over attacker pairs (primary — full coverage)
            #   2. atk_sum_margin : sum of margins over attacker pairs (tiebreaker when
            #                       full coverage is impossible — maximises pairs covered
            #                       and gives directional guidance to prevent oscillation)
            #   3. bnd_margin     : min margin over all pairs (secondary — boundary coverage)
            if atk_pair_indices:
                atk_margins = margins_all[:, atk_pair_indices].min(axis=1)
                atk_sum_margins = margins_all[:, atk_pair_indices].sum(axis=1)
            else:
                atk_margins = np.zeros(len(all_combos), dtype=np.int16)
                atk_sum_margins = np.zeros(len(all_combos), dtype=np.int16)
            bnd_margins = margins_all.min(axis=1)

            best_idx = int(np.lexsort((bnd_margins, atk_sum_margins, atk_margins))[-1])
            best_combo_indices = tuple(int(x) for x in all_combos[best_idx])

            dynamic_assignment = {
                sorted_defenders[k]: unique_cands[best_combo_indices[k]]
                for k in range(len(sorted_defenders))
            }

            # atk_min  : min(cov-req) over real-attacker pairs  — ≥0 = all attacker pairs covered
            # atk_sum  : sum(cov-req) over real-attacker pairs  — tiebreaker, higher = more slack
            # all_min  : min(cov-req) over ALL pairs            — ≥0 = everything covered
            best_margins = margins_all[best_idx]
            bnd_pair_indices = [j for j in range(n_pairs) if j not in set(atk_pair_indices)]
            atk_covered = sum(1 for j in atk_pair_indices if best_margins[j] >= 0)
            bnd_covered = sum(1 for j in bnd_pair_indices if best_margins[j] >= 0)
            debug(f"[{agent_ctrl.name}] Phase2 t={current_time} | "
                  f"{len(all_combos)} combos | "
                  f"atk_worst_gap={int(atk_margins[best_idx])} "
                  f"atk_total_slack={int(atk_sum_margins[best_idx])} "
                  f"global_worst_gap={int(bnd_margins[best_idx])} | "
                  f"ATK:{atk_covered}/{len(atk_pair_indices)} BND:{bnd_covered}/{len(bnd_pair_indices)} | "
                  f"attackers={dict(sorted(visible_attackers.items()))}")

            team_cache.set("dynamic_assignment", dynamic_assignment)
            team_cache.set("dynamic_time", current_time)

        # Move to the assigned next position (already 1 step away by construction)
        dynamic_assignment: Dict[str, int] = team_cache.get("dynamic_assignment", {})
        assigned_pos = dynamic_assignment.get(agent_ctrl.name)
        if assigned_pos is not None:
            target = assigned_pos

    # =========================================================
    # PHASE 1 MOVEMENT: navigate toward static assigned position
    # =========================================================
    elif team_cache.get("positions_computed", False):
        assigned_pos = static_assignment.get(agent_ctrl.name)
        if assigned_pos is not None and current_pos != assigned_pos:
            target = agent_map.shortest_path_step(current_pos, assigned_pos, speed)

    # ===== OUTPUT =====
    state["action"] = target

    return set()


def map_strategy(agent_config):
    strategies = {}
    for name in agent_config.keys():
        strategies[name] = strategy
    return strategies
