from typing import Dict, List, Tuple, Any, Set, Union, Optional, Iterable, Hashable
import networkx as nx
import numpy as np
from collections import deque
import math
from collections import Counter


##############################################################
#                     Helper Constants....
##############################################################
# Dictionary to store the Team Vsense...
# _Vsense_global = {}
# Radius for stationary sensor observation...
R_GREEN = 450
# Radius for attacker observation of flag...
R_PROTECT_FLAG = 400
# Radius for defender sensing...
R_DEFENDER_SENSING = 250
# Radius for attacker to capture the flag...
R_BANNED = 2
# Radius for tag for defender....
R_TAG = 1
# Maximum waiting time for agent at node...
MAX_WAIT = 10


##############################################################
#                     Attacker Strategy....
##############################################################
def strategy(state):
    
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

    last_target: int = cache.get("last_target", None)  # Previously chosen target node
    visit_count: int = cache.get("visit_count", 0)  # Number of strategy calls for this agent

    cache.set("last_position", current_pos)  # Store current position for next turn
    cache.set("visit_count", visit_count + 1)  # Increment visit counter
    cache.update(last_time=current_time, patrol_index=cache.get("patrol_index", 0) + 1)  # Batch update multiple cache values
    
    if not cache.has('Wait_Counter'):
       cache.set("Wait_Counter",0)

    if not cache.has('Current_Target'):
       cache.set("Current_Target",int) 

    if not cache.has('Vgreen_Nearest_Node'):
       cache.set('Vgreen_Nearest_Node',int)

    if not cache.has('Vprotect_Nearest_Node'):
       cache.set('Vprotect_Nearest_Node',int)

    if not cache.has('Vbanned_Nearest_Node'):
       cache.set('Vbanned_Nearest_Node',int)

    

    # ===== TEAM CACHE (SHARED) =====
    team_cache = agent_ctrl.team_cache  # Shared storage across all teammates

    priority_targets: List[int] = agent_ctrl.get_team("priority_targets", [])  # How to get data from team cache

    agent_ctrl.set_team("last_update", current_time)  # Track when team cache was last modified
    agent_ctrl.update_team(total_captures=agent_ctrl.get_team("total_captures", 0), formation="spread")  # Update team-wide statistics

    if not team_cache.has('Vsense'):
       agent_ctrl.set_team("Vsense",{})

    if not team_cache.has('True-Flags'):
       agent_ctrl.set_team("True-Flags",set())

    if not team_cache.has('Fake-Flags'):
       agent_ctrl.set_team("Fake-Flags",set())

    if not team_cache.has('Vmost'):
       agent_ctrl.set_team("Vmost",int)

    if not team_cache.has('Vcand_nearest_list'):
       agent_ctrl.set_team("Vcand_nearest_list",[])

    if not team_cache.has('Vcand_best'):
       agent_ctrl.set_team("Vcand_best",int)


    

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
    # print(f'Nodes list of neigbours: {get_outgoing_neighbors_dict(global_map_sensor,list(nodes_data.keys()))}')
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
    enemy_pos, enemy_age = agent_map.get_agent_position(agent_ctrl.enemy_team, "blue_0")    # (position, age_in_timesteps) or (None, None)


    # ===== DECISION LOGIC =====
    target: int = current_pos  # Default action is to stay at current position

    if "egocentric_flag" in sensors:
        flags: List[int] = sensors["egocentric_flag"][1]["detected_flags"]  # Flags visible to agent
        if flags:
            # Move to the next
            target = agent_map.shortest_path_step(current_pos, flags[0], speed)# Move toward first visible flag
    elif agent_map.graph is not None:
        neighbors: List[int] = list(agent_map.graph.neighbors(current_pos))  # Adjacent nodes
        if neighbors:
            target = neighbors[0]  # Move to first neighbor

    # Getting the enemy agents name and current node ID:
    enemy_name = []
    enemy_pos = []
    for enemy in enemies.keys():
        enemy_name.append(enemy)          # Getting the enemy name...
        enemy_pos.append(enemies[enemy])  # Getting the enemy position...


    # Getting the Vcand: Candidate flag locations....
    Vcand: List[int] = candidates
    # Getting the observation form the condidate flags...
    for flag in Vcand:
        vgreen_flag_i = _get_nodes_in_range([flag],R_GREEN,global_map_sensor)
        # _update_Vsense(str(flag),vgreen_flag_i)
        _update_cache_Vsense(team_cache,str(flag),vgreen_flag_i)

    # Getting the observation from the enemy agents...
    for name in enemy_name:
        venemy_sensing = _get_nodes_in_range(enemy_pos,R_DEFENDER_SENSING,global_map_sensor)
        # _update_Vsense(name,venemy_sensing)
        _update_cache_Vsense(team_cache,name,venemy_sensing)

    # Getting nodes that are observed by stationary flags and enemy sensoing nodes...
    Vsense = _get_cache_Vsense(team_cache)
    # Getting the Vsense_boundary: Vsense frontier nodes...
    Vsense_boundary = _get_node_at_boundary(Vsense,global_map_sensor)
    # Getting the Vprotect:
    Vprotect = _get_nodes_in_range_dict(candidates,R_PROTECT_FLAG,global_map_sensor)
    # Getting the Vprotect_boundary:
    Vprotect_boundary = _get_node_at_boundary(Vprotect,global_map_sensor)
    # Getting Vbanned:
    Vbanned = _k_hop_neighbors_dict(global_map_sensor,candidates,2)
    # Getting Vbanned_boundary:
    Vbanned_boundary = _get_node_at_boundary(Vbanned,global_map_sensor)
    # Getting the set if Vtag nodes:
    Vtag = _k_hop_neighbors_dict(global_map_sensor,enemy_pos,2)
    Vtag_set = set().union(*Vtag.values())
   


    # Getting all nodes...
    all_nodes = set(nodes_data.keys())
    # Getting the set of the nodes that are outside Vgreen...
    V_not_green = set(all_nodes) - Vsense
    # Getting the dictionary of non Vgreen nodes and its outgoing neigbours...
    V_not_green_out_neighbours_dictionary = get_outgoing_neighbors_dict(global_map_sensor,list(V_not_green))
    # Getting the node with most outgoing neighbours...
    Vnmax = max(V_not_green_out_neighbours_dictionary,key=lambda k: len(V_not_green_out_neighbours_dictionary[k]))
    # Updating the Vnmax in team cache...
    agent_ctrl.set_team("Vmost",Vnmax)


    # Getting closest candidate to the defender/enemy observation list...
    nearest_cand, _, _ = closest_vcand_from_enemy(global_map_sensor,enemy_pos,candidates)
    nearest_cand_obs_list = agent_ctrl.get_team("Vcand_nearest_list")
    nearest_cand_obs_list.append(nearest_cand)
    

    # Getting the most observed Vcand location...
    most_repeated = Counter(agent_ctrl.get_team("Vcand_nearest_list")).most_common(1)[0][0]
    agent_ctrl.set_team("Vcand_best",most_repeated)
    

    # Setting the inital traget location...
    if cache.get("last_target") == None:
       cache.set("last_target",Vnmax)

    

    #*****************************************************************
    #**************** Strategy for attacker...************************
    #*****************************************************************
    
    # Getting the attacker teammate position...
    teammate_position = [info[1] for info in teammates_data]

    # For each of the flag get Vgreen nodes set...
    Vgreen_dic = {i: None for i in Vcand}
    for key in team_cache["Vsense"].keys():
        if not key.startswith("blue_"):
            Vgreen_dic[int(key)] = team_cache["Vsense"][key]

    # Take difference between the Vgreen and V protect for each agent...
    Vgreen_Vprotect_Area_dic = {i:None for i in Vcand}
    for i in Vcand:
        Vgreen_Vprotect_Area_dic[i] = Vgreen_dic[i] - Vprotect[i]
    # print(Vgreen_Vprotect_Area_dic)
    
    # Dictionary to maintian the attacker agent locaitons on the Vgreen_Vprotect_Area_dic region...
    Vgreen_Vprotect_Area_Attaacker_log = {i:None for i in Vcand}
    for i in Vcand:
        Vgreen_Vprotect_Area_Attaacker_log[i] = [x for x in teammate_position if x in Vgreen_Vprotect_Area_dic[i]]
    
    # Take difference between the Vprotext and Vtag for each agent...
    Vprotect_Vban_Area_dic = {i:None for i in Vcand}
    for i in Vcand:
        Vprotect_Vban_Area_dic[i] = Vprotect[i] - Vbanned[i]

    # Dictionary to maintian the attacker agent locaitons on the Vgreen_Vprotect_Area_dic region...
    Vprotect_Vban_Area_Attaacker_log = {i:None for i in Vcand}
    for i in Vcand:
       Vprotect_Vban_Area_Attaacker_log[i] = [x for x in teammate_position if x in  Vprotect_Vban_Area_dic[i]]
    
    # Initialig the path target varaible...
    path_to_min_node = []
    # Get shortest path to the the current set target node....
    path_to_min_node = nx.shortest_path(global_map_sensor, source=current_pos, target=cache.get("last_target"))
    
    # Checking if the current agent position is in the Vtag region...
    if state['curr_pos'] in Vtag_set:
       # Getting the nearest 1-hop nodes...
       current_position_1_hop_nbr = _k_hop_neighbors(global_map_sensor,current_pos,1)
       # Getting the set of 1-hop neighbours not in Vtag neighbours...
       current_position_non_tagable_1_hop_nbr = current_position_1_hop_nbr - Vtag_set
       # Checking if the non tagable nodes exist...
       if not current_position_non_tagable_1_hop_nbr:
          cache.set("last_target",Vnmax)
          path_to_min_node = nx.shortest_path(global_map_sensor, source=current_pos, target=cache.get("last_target"))
       else:
          # Getting nearest node to Vnmax...
          best_nbr, _ = nearest_node_by_hops(global_map_sensor,current_position_non_tagable_1_hop_nbr,Vnmax)
          path_to_min_node = [current_pos,best_nbr]

    else:
        # Checking if current position of agent is at Vnmax...
        if state['curr_pos'] == Vnmax:
            curr_wait_timer = cache.get("Wait_Counter")
            cache.set("Wait_Counter",curr_wait_timer+1)
            # print(f'Wait-Counter: {cache.get("Wait_Counter")}')
            if curr_wait_timer >= MAX_WAIT:
                # Resetting the current wait timer...
                cache.set("Wait_Counter",0)
                # Getting the best observed node...
                best_candidate_node = agent_ctrl.get_team("Vcand_best")
                # Getting the set of Vgreen nodes for the best Vcand...
                best_candidate_Vgreen = Vgreen_dic[best_candidate_node]
                # Getting the nearest hop Vgreen nodes for best Vcand...
                Vgreen_nearest_node = nearest_nhop_node(global_map_sensor,current_pos,best_candidate_Vgreen)
                # Updating the target in the cache to...
                cache.set("last_target",Vgreen_nearest_node)
                cache.set("Vgreen_Nearest_Node",Vgreen_nearest_node)
                # Getting path to the nearest Vgreen...
                path_to_min_node = nx.shortest_path(global_map_sensor, source=current_pos, target=cache.get("last_target"))
        
        # Checking if the current position of agent is on Vgreen Nearest...
        Vgreen_nearest_node = cache.get('Vgreen_Nearest_Node')
        if state['curr_pos'] == Vgreen_nearest_node:
            curr_wait_timer = cache.get("Wait_Counter")
            cache.set("Wait_Counter",curr_wait_timer+1)
            if curr_wait_timer >= MAX_WAIT-4:
                # Resetting the current wait timer...
                cache.set("Wait_Counter",0)
                # Getting the best observed node...
                best_candidate_node = agent_ctrl.get_team("Vcand_best")
                # Getting the set of Vprotect nodes for the best Vcand...
                best_candidate_Vprotect = Vprotect[best_candidate_node]
                # Getting the nearest hop Vprotect nodes for best Vcand...
                Vprotect_nearest_node = nearest_nhop_node(global_map_sensor,current_pos,best_candidate_Vprotect)
                # Updating the target in the cache to...
                cache.set("last_target",Vprotect_nearest_node)
                cache.set("Vprotect_Nearest_Node",Vprotect_nearest_node)
                # Getting path to the nearest Vgreen...
                path_to_min_node = nx.shortest_path(global_map_sensor, source=current_pos, target=cache.get("last_target"))

        # Checking if the current position of agent is on the Vprotect Nearest...
        Vprotect_nearest_node = cache.get('Vprotect_Nearest_Node')
        if state['curr_pos'] == Vprotect_nearest_node:
            curr_wait_timer = cache.get("Wait_Counter")
            cache.set("Wait_Counter",curr_wait_timer+1)
            if curr_wait_timer >= MAX_WAIT-6:
                # Resetting the current wait timer...
                cache.set("Wait_Counter",0)
                # Getting the best observed node...
                best_candidate_node = agent_ctrl.get_team("Vcand_best")
                # Getting the set of Vprotect nodes for the best Vcand...
                best_candidate_Vbanned = Vbanned[best_candidate_node]
                # Getting the nearest hop Vbanned nodes for best Vcand...
                Vbanned_nearest_node = nearest_nhop_node(global_map_sensor,current_pos,best_candidate_Vbanned)
                # Updating the target in the cache to...
                cache.set("last_target",Vbanned_nearest_node)
                cache.set("Vbanned_Nearest_Node",Vbanned_nearest_node)
                # Getting path to the nearest Vgreen...
                path_to_min_node = nx.shortest_path(global_map_sensor, source=current_pos, target=cache.get("last_target"))
    

    # If the path length is not zero...
    if len(path_to_min_node) != 0:
        # Getting the curent position index in the path list...
        curr_idx = path_to_min_node.index(state['curr_pos'])
        # If current position is goal/min_node don't move...
        if state['curr_pos'] == path_to_min_node[-1]:
            target: int = current_pos
        # Else selecct the next postion in path to go...
        else:
            next_node = path_to_min_node[curr_idx+1]
            if next_node in _k_hop_neighbors(global_map_sensor,state['curr_pos'],k=1):
                target: int = next_node
            else:
                # print('Error-Hopped-Multiple Nodes')
                target: int = current_pos
    # If path length is zero don't move...
    else:
        target: int = current_pos

    # ===== OUTPUT =====
    state["action"] = target  # Required: set target node for this turn
    return f"moving to {target}" if target != current_pos else "holding position"  # avoid f-strings here if possible — string construction runs every call even when logging is off and will slow down mass eval
    


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





##############################################################
#                  Helper Routines....
##############################################################

# Function to update the cache Vsense for given agent...
def _update_cache_Vsense(cache,agent,nodes):
    cache["Vsense"][agent] = nodes

# Function to get the all the Vsense nodes as set...
def _get_cache_Vsense(cache):
    return set().union(*cache["Vsense"].values())

# Function to get all the Vsense boundary points...
def _get_node_at_boundary(vsense,global_map):
    vsense_boundary = []
    for node in vsense:
        for neighbour in list(global_map.neighbors(node)):
            if neighbour not in vsense:
               vsense_boundary.append(node)
               break
    return vsense_boundary

# Function to get nodes within the radius range...
def _get_nodes_in_range(flag_node,radius,global_map):
    coords = {}
    for n, attrs in global_map.nodes(data=True):
        x = attrs.get('x')
        y = attrs.get('y')
        if x is not None and y is not None:
            coords[n] = (float(x), float(y))
    inrange_node: Set[int] = set()
    for src in flag_node:
        if src not in coords:
            continue
        x0, y0 = coords[src]
        for n, (x, y) in coords.items():
            if np.sqrt((x - x0) ** 2 + (y - y0) ** 2) <= radius:
                inrange_node.add(n)
    return inrange_node

# Function to get nodes within the radius range as dictionary...
def _get_nodes_in_range_dict(flag_nodes: Iterable[int], radius: float, global_map: nx.Graph) -> Dict[int, Set[int]]:
    # Precompute coordinates for all nodes once
    coords = {}
    for n, attrs in global_map.nodes(data=True):
        x = attrs.get("x")
        y = attrs.get("y")
        if x is not None and y is not None:
            coords[n] = (float(x), float(y))
    # Build per-flag sets
    flag_to_inrange: Dict[int, Set[int]] = {}
    r2 = radius * radius  # compare squared distances (faster)
    for flag in flag_nodes:
        if flag not in coords:
            flag_to_inrange[flag] = set()
            continue
        x0, y0 = coords[flag]
        inrange = set()
        for n, (x, y) in coords.items():
            dx = x - x0
            dy = y - y0
            if (dx * dx + dy * dy) <= r2:
                inrange.add(n)
        flag_to_inrange[flag] = inrange
    return flag_to_inrange

# Function to get the 'k-hop' neighbour nodes...
def _k_hop_neighbors(G: nx.Graph, start: Hashable, k: int) -> Set[Hashable]:
    """
    Returns the set of nodes within <= k hops of `start` in an unweighted graph.
    Includes `start` (0-hop).
    """
    if k < 0:
        return set()
    if start not in G:
        return set()
    visited = {start}
    q = deque([(start, 0)])  # (node, depth)
    while q:
        u, d = q.popleft()
        if d == k:
            continue
        for v in G.neighbors(u):
            if v not in visited:
                visited.add(v)
                q.append((v, d + 1))
    return visited

# Function to get the set of 'k-hop' neighbour nodes for list/set of nodes...
def _k_hop_neighbors_multi(G: nx.Graph, starts: Iterable[int], k: int) -> Set[int]:
    """
    Returns the union of nodes within <= k hops of ANY node in `starts`.
    """
    out: Set[int] = set()
    for s in starts:
        out |= _k_hop_neighbors(G, s, k)   # uses the single-source function you already have
    return out

# Function to get 'k-hop' neighours dictionary where root node is key and neighbours are values...
def _k_hop_neighbors_dict(G: nx.Graph, starts: Iterable[int], k: int) -> Dict[int, Set[int]]:
    """
    Returns:
      { start_node : set_of_nodes_within_<=k_hops_including_start }
    If a start node is not in G, it maps to an empty set.
    """
    out: Dict[int, Set[int]] = {}
    for s in starts:
        if s in G:
            out[s] = set(nx.single_source_shortest_path_length(G, s, cutoff=k).keys())
        else:
            out[s] = set()
    return out

# Function to get the number of hop from source/root to all the other targets and return it as dictionary...
def _hops_to_targets(
    G: nx.Graph,
    source: int,
    targets: Iterable[int],
    unreachable_value: Union[int, float] = float("inf"),
) -> Dict[int, Union[int, float]]:
    """
    Returns a dict: {target_node: hop_distance}
    Hop distance = number of edges in the shortest path (BFS).
    Unreachable targets get unreachable_value.
    """
    targets = list(targets)
    target_set = set(targets)
    # Single-source shortest path lengths (BFS for unweighted graphs)
    lengths = nx.single_source_shortest_path_length(G, source)
    out = {}
    for t in targets:
        out[t] = lengths.get(t, unreachable_value)
    return out

# Function to get the node with minimum hops from source to multiple target nodes...
def _min_hop_target(
    G: nx.Graph,
    source: int,
    targets: Iterable[int],
) -> Tuple[Optional[int], float]:
    """
    Returns (best_target, min_hops). If none reachable, returns (None, inf).
    """
    hop_dict = _hops_to_targets(G, source, targets, unreachable_value=float("inf"))
    best_t, best_h = min(hop_dict.items(), key=lambda kv: kv[1], default=(None, float("inf")))
    if best_h == float("inf"):
        return None, float("inf")
    return best_t, float(best_h)

# Function to get the n-th minimum hop to target from list of targets...
def _nth_min_hop_target(
    G: nx.Graph,
    source: int,
    targets: Iterable[int],
    n: int = 1,
) -> Tuple[Optional[int], float]:
    """
    Returns (target_node, hops) for the n-th smallest hop target.
    n=1 gives the minimum (same as your current function).
    If fewer than n reachable targets exist -> (None, inf).
    """
    if n <= 0:
        raise ValueError("n must be >= 1")
    hop_dict = _hops_to_targets(G, source, targets, unreachable_value=float("inf"))
    # keep only reachable
    reachable = [(t, h) for t, h in hop_dict.items() if math.isfinite(h)]
    if len(reachable) < n:
        return None, float("inf")
    # sort by (hop, target_id) for deterministic tie-breaking
    reachable.sort(key=lambda x: (x[1], x[0]))
    t, h = reachable[n - 1]
    return t, float(h)

# Function to get the dictionary for given node_id [int]: neighbours [list]...
def get_outgoing_neighbors_dict(
    G: nx.Graph,
    node_ids: List[Hashable],
) -> Dict[Hashable, List[Hashable]]:
    """
    Returns a dictionary:
        key   = node id
        value = list of outgoing neighbors / successors
    """

    outgoing_neighbors = {}

    for node in node_ids:
        if node in G:
            if G.is_directed():
                outgoing_neighbors[node] = list(G.successors(node))
            else:
                outgoing_neighbors[node] = list(G.neighbors(node))
        else:
            outgoing_neighbors[node] = []

    return outgoing_neighbors


# Function to get the closest vcand location from any given enemy...
def closest_vcand_from_enemy(
    G: nx.Graph,
    enemy: List[Hashable],
    Vcand: List[Hashable],
    directed: bool = True,
) -> Tuple[Optional[Hashable], Optional[int], Dict[Hashable, int]]:
    """
    Finds which node in Vcand has the minimum hop distance from any enemy node.

    For directed graphs:
        directed=True  -> follows outgoing/successor edges
        directed=False -> ignores edge direction

    Returns:
        best_vcand: closest node in Vcand
        best_dist: hop distance from nearest enemy node
        vcand_distances: distance for each reachable Vcand node
    """

    G_search = G if directed else G.to_undirected()

    sources = [node for node in enemy if node in G_search]

    if not sources:
        return None, None, {}

    Vcand_set = set(Vcand)

    # Multi-source BFS
    distances = {}
    queue = deque()

    for source in sources:
        distances[source] = 0
        queue.append(source)

    while queue:
        current = queue.popleft()

        for nbr in G_search.neighbors(current):
            if nbr not in distances:
                distances[nbr] = distances[current] + 1
                queue.append(nbr)

    # Keep only candidate nodes that are reachable
    vcand_distances = {
        node: distances[node]
        for node in Vcand
        if node in distances
    }

    if not vcand_distances:
        return None, None, {}

    best_vcand = min(vcand_distances, key=vcand_distances.get)
    best_dist = vcand_distances[best_vcand]

    return best_vcand, best_dist, vcand_distances


# Function to get the nearest nhop node...
def nearest_nhop_node(
    G: nx.Graph,
    location_node: Any,
    candidate_nodes: List[Any],
) -> Optional[Any]:
    """
    Find the nearest node from candidate_nodes to location_node
    using hop distance.

    Returns:
        nearest candidate node, or None if no candidate is reachable.
    """

    if location_node not in G:
        return None

    candidate_set = set(candidate_nodes)

    # shortest path lengths from location_node to all reachable nodes
    dist = nx.single_source_shortest_path_length(G, location_node)

    nearest_node = None
    nearest_dist = float("inf")

    for node in candidate_set:
        if node in dist and dist[node] < nearest_dist:
            nearest_node = node
            nearest_dist = dist[node]

    return nearest_node

# Another helper to get the nearest nodes from set/list...
def nearest_node_by_hops(G, candidate_nodes, source_node):
    """
    Find the candidate node with minimum hop distance from source_node.

    Parameters:
        G: networkx graph
        candidate_nodes: set/list of node IDs
        source_node: starting node ID

    Returns:
        (nearest_node, min_hops)
        or (None, None) if no candidate is reachable
    """

    if not candidate_nodes:
        return None, None

    # shortest hop distance from source_node to all reachable nodes
    hop_distances = nx.single_source_shortest_path_length(G, source_node)

    nearest_node = None
    min_hops = float("inf")

    for node in candidate_nodes:
        if node in hop_distances:
            if hop_distances[node] < min_hops:
                min_hops = hop_distances[node]
                nearest_node = node

    if nearest_node is None:
        return None, None

    return nearest_node, min_hops