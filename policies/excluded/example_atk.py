def strategy(state: dict) -> str:
    from typing import Dict, List, Tuple, Any
    import networkx as nx

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
    example_shared = agent_ctrl.get_team("example_key", 0)   # get a shared team value
    agent_ctrl.set_team("example_key", current_time)          # set a shared team value
    agent_ctrl.update_team(example_a=0, example_b="val")      # set multiple shared values at once

    # ===== SENSORS =====
    # All sensor data is read here once; variables are reused in map updates and decision logic below.
    sensors: Dict[str, Tuple[Any, Dict[str, Any]]] = state["sensor"]  # All sensor data

    candidates: List[int] = agent_ctrl.sensor_data(state, "candidate_flag")["candidate_flags"] if "candidate_flag" in sensors else []  # Possible flag locations

    # Defender visibility depends on the rule version. Up to v1.3 red carried the
    # whole-map "agent" sensor and saw every defender every tick; from v1.4 red
    # only sees defenders that are in line of sight, through "egocentric_agent"
    # (exactly how blue has always sensed red). Teammates stay whole-map either
    # way -- via "agent" before, via "custom_team" now. Reading both keeps this
    # example working under either rule file.
    if "agent" in sensors:  # v1.3 and earlier: omniscient
        enemies: Dict[str, int]   = agent_ctrl.sensor_data(state, "agent")["enemies"]    # Enemy agents in game {name: node_id}
        teammates: Dict[str, int] = agent_ctrl.sensor_data(state, "agent")["teammates"]  # Teammate agents in game {name: node_id}
    else:                   # v1.4+: occlusion-limited enemies, whole-map teammates
        enemies    = agent_ctrl.sensor_data(state, "egocentric_agent")["enemies"] if "egocentric_agent" in sensors else {}  # Defenders currently in line of sight {name: node_id}
        teammates  = agent_ctrl.sensor_data(state, "custom_team") if "custom_team" in sensors else {}                       # Teammate agents in game {name: node_id}

    detected_flags: List[int] = agent_ctrl.sensor_data(state, "egocentric_flag")["detected_flags"] if "egocentric_flag" in sensors else []  # Real flags within range; flags visible to agent
    flag_count: int           = agent_ctrl.sensor_data(state, "egocentric_flag").get("flag_count", len(detected_flags)) if "egocentric_flag" in sensors else 0   # Number of detected flags (region-sensor payloads don't carry this key; derive it)

    # ----- Sensor coverage -----
    # A ranged sensor reports what it can cover as well as what it found, so there is
    # no separate "..._region" sensor: ask the sensor itself. Both calls return None
    # when the sensor is absent or has no range (e.g. v1.3, where red has no
    # "egocentric_agent"), hence the `or frozenset()`.
    agent_coverage = agent_ctrl.coverage(state, "egocentric_agent") or frozenset()  # Nodes where this agent would spot a defender right now (line of sight, 400)
    flag_coverage  = agent_ctrl.coverage(state, "egocentric_flag")  or frozenset()  # Nodes where this agent would detect a real flag right now (line of sight, 400)
    flag_coverage_from_first_candidate = (agent_ctrl.coverage_from(state, "egocentric_flag", candidates[0]) or frozenset()) if candidates else frozenset()  # What the flag sensor WOULD cover standing on another node — any node works

    # ===== AGENT MAP (SHARED) =====
    agent_map = agent_ctrl.map  # Team-shared map with positions and graph
    global_map_payload: Dict[str, Any] = agent_ctrl.sensor_data(state, "global_map")
    global_map_sensor: nx.Graph = global_map_payload["graph"]  # Full graph topology from sensor
    global_map_apsp = global_map_payload.get("apsp")

    # The graph is static — attach it once on the first turn and reuse every turn after.
    # agent_map is shared across teammates, so only the first agent to run pays this cost.
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
    agent_map.update_time(current_time)  # Sync map time with game time for age tracking

    # Update own position in map (call this every turn to track your position)
    agent_map.update_agent_position(team, agent_ctrl.name, current_pos, current_time)

    # Update enemy positions from sensor data (if you have visibility)
    agent_map.update_team_agents(agent_ctrl.enemy_team, enemies, current_time)
    # You can update your teammates similarly from the `teammates` dict resolved above

    # How to get all positions of a team from agent map
    teammates_data: List[Tuple[str, int, int]] = agent_map.get_team_agents(team)            # [(name, pos, age)] of teammates

    # ===== DECISION LOGIC =====
    # Search the candidate flags as a team, capture the real ones, and keep out of defenders' reach.
    target: int = current_pos  # Default action is to stay at current position
    dist = agent_map.shortest_path_length

    # 1. Share flag knowledge through the team cache. A candidate inside the flag sensor's
    #    coverage is settled: real if the sensor reports a flag there, fake otherwise.
    real_known: set = agent_ctrl.get_team("real_flags") or set()         # Real flags any teammate has seen
    checked: set    = agent_ctrl.get_team("checked_candidates") or set()  # Candidates any teammate has settled
    real_known.update(detected_flags)
    checked.update(detected_flags)
    checked.update(c for c in candidates if c in flag_coverage)
    agent_ctrl.update_team(real_flags=real_known, checked_candidates=checked)

    # 2. Defenders come from the shared map, so one teammate's sighting warns the whole team.
    #    Sightings older than 3 ticks are dropped (the defender has moved on, or was tagged out).
    defenders = [pos for _, pos, age in agent_map.get_team_agents(agent_ctrl.enemy_team) if age <= 3]
    reach = opp_tagging_radius + rule_config["blue_global"]["speed"]  # A defender can step, then tag

    # 3. Pick a goal: the nearest known real flag or unsettled candidate. Candidates a living
    #    teammate is already heading for are skipped, so the team spreads out over the search,
    #    and so are goals a defender is standing guard over, as long as there is another choice.
    claims: Dict[str, Any] = agent_ctrl.get_team("claims") or {}  # {agent name: its current goal}
    claimed = {node for name, node in claims.items() if name != agent_ctrl.name and name in teammates}
    unchecked = [c for c in candidates if c not in checked]
    goals = list(real_known) + [c for c in unchecked if c not in claimed]
    if not goals:
        goals = unchecked  # Everything left is claimed — double up instead of idling
    goals = [g for g in goals if all(dist(g, d) > reach + capture_radius for d in defenders)] or goals
    goal = min(goals, key=lambda n: (dist(current_pos, n), n)) if goals else None
    claims[agent_ctrl.name] = goal
    agent_ctrl.set_team("claims", claims)

    # 4. Step toward the goal without entering a defender's reach. Tagging is resolved before
    #    capture, so a capturing step inside a defender's reach is not worth taking either.
    if goal is not None and agent_map.graph is not None:
        options: List[int] = [current_pos] + list(agent_map.graph.neighbors(current_pos))  # One hop is legal at any speed
        safe = [n for n in options if all(dist(n, d) > reach for d in defenders)]
        if safe:
            target = min(safe, key=lambda n: (dist(n, goal), n))
        else:
            # Cornered: get as far from the defenders as possible
            target = max(options, key=lambda n: (min(dist(n, d) for d in defenders), -n))

    # ===== OUTPUT =====
    state["action"] = target  # Required: set action for this turn
    return "moving" if target != current_pos else "holding position"  # Keep this a constant: an f-string is built on every call even when logging is off and will slow down mass eval


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
