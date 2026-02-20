from typing import Dict, List, Tuple, Any, Set, Union, Optional, Iterable, Hashable
import networkx as nx
import numpy as np
from collections import deque
import math


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

    nodes_data: Dict[int, Dict[str, Any]] = {node_id: global_map_sensor.nodes[node_id] for node_id in global_map_sensor.nodes()}  # Convert graph nodes to dict format

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
    for agent_name in all_agents.keys():
        if agent_name.startswith("blue_"):
           enemy_name.append(agent_name)
           enemy_pos.append(all_agents[agent_name])


    # Getting the Vcand: Candidate flag locations....
    Vcand: List[int] = candidates
    # Getting the observation form the condidate flags...
    for flag in Vcand:
        vgreen_flag_i = _get_nodes_in_range([flag],R_GREEN,global_map_sensor)
        _update_Vsense(str(flag),vgreen_flag_i)
    # Getting the observation from the ememy agents...
    for name in enemy_name:
        venemy_sensing = _get_nodes_in_range([flag],R_DEFENDER_SENSING,global_map_sensor)
        _update_Vsense(name,venemy_sensing)
    # Getting nodes that are observed by stationary flags and enemy sensoing nodes...
    Vsense = _get_Vsense()
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
    # print(Vprotect)
    Vtag = _k_hop_neighbors_dict(global_map_sensor,enemy_pos,2)
    Vtag_set = set().union(*Vtag.values())
    # print(set().union(*Vtag.values()))

    ##################################
    # Strategy for attacker...
    ##################################
    
    # Getting the attacker teammate position...
    teammate_position = [info[1] for info in teammates_data]

    # For each of the flag get Vgreen nodes set...
    Vgreen_dic = {i: None for i in Vcand}
    for key in _Vsense_global.keys():
        if not key.startswith("blue_"):
            Vgreen_dic[int(key)] = _Vsense_global[key]

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

    # If current postion of attacker is not on the defender sensing(ego + stationaty) area...
    if current_pos is not _get_Vsense():
        # Checking if attacker team is doing well...
        if red_payoff > blue_payoff:
            # Getting all the 1-hop neigbours of the attackers...
            attacter_1_hop = _k_hop_neighbors(global_map_sensor,current_pos,1)
            # Checking if any attacker 1-hop neighbour are in Vtag node set...
            if len(attacter_1_hop & Vtag_set) != 0:
               # Getting the set of attacker neighuours which are not in Vtag set...
               option_nodes = attacter_1_hop - Vtag_set
               # Among remaing attacker 1-hop neighbour we get node with most neighbour edges...
               best_node = max(option_nodes, key=lambda n: global_map_sensor.degree(n))
               min_node = best_node
               # Creating path for the next best 1-hop neighbour...
               path_to_min_node = [current_pos,best_node]
        else:
            # Getting the min hop node on the Vbanned boundary...
            min_node, _ = _min_hop_target(global_map_sensor,current_pos,Vbanned_boundary)
            # Getting shorest path to the min hope node...
            path_to_min_node = nx.shortest_path(global_map_sensor, source=current_pos, target=min_node)
            #  If path of attacker goes thru the Vtag area get the 1-hop neghours that are are not tagged....
            if any(x in set(path_to_min_node) for x in Vtag):
               attacter_1_hop = _k_hop_neighbors(global_map_sensor,current_pos,1)
               option_nodes = attacter_1_hop - Vtag_set
               if len(option_nodes) != 0:
                    # Getting the best nodes from the option (min hop) as next postion...
                    best_node = max(option_nodes, key=lambda n: global_map_sensor.degree(n))
                    min_node = best_node
                    # Creating path for the next best 1-hop neighbour...
                    path_to_min_node = [current_pos,best_node]
    
    for key in Vgreen_Vprotect_Area_Attaacker_log.keys():
        for attacker in Vgreen_Vprotect_Area_Attaacker_log[key]:
            if attacker == current_pos:
                min_node, min_dist = _min_hop_target(global_map_sensor,attacker,Vprotect[key])
                path_to_min_node = nx.shortest_path(global_map_sensor, source=current_pos, target=min_node)
                if any(x in set(path_to_min_node) for x in Vtag):
                    min_node, min_dist = _min_hop_target(global_map_sensor,attacker,Vgreen_dic[key])
                    path_to_min_node = nx.shortest_path(global_map_sensor, source=current_pos, target=min_node)

    # Iterating over region between the Vprotect and Vban area for each candidate flag...
    for key in Vprotect_Vban_Area_Attaacker_log.keys():
        # For given flag region between the Vprotect and Vban area checking all the attackers...
        for attacker in Vprotect_Vban_Area_Attaacker_log[key]:
            # Checking the attacker location is with current agent...
            if attacker == current_pos:
                # Checking the real flag condition...
                if detected:
                    # If real flag detected go to the nearest k-hop Vbanned region...
                    min_node, min_dist = _min_hop_target(global_map_sensor,attacker,Vbanned[key])
                    path_to_min_node = nx.shortest_path(global_map_sensor, source=current_pos, target=min_node)
                    # If path of attacker goes thru the Vtag area go back to nearst Vprotect...
                    if any(x in set(path_to_min_node) for x in Vtag):
                        min_node, min_dist = _min_hop_target(global_map_sensor,attacker,Vprotect[key])
                        path_to_min_node = nx.shortest_path(global_map_sensor, source=current_pos, target=min_node)
                else:
                    # This is the second minimum of the Vprotect target: When realizeing the flag is fake...
                    min_node, min_dist = _nth_min_hop_target(global_map_sensor,attacker,Vprotect[key],2)
                    path_to_min_node = nx.shortest_path(global_map_sensor, source=current_pos, target=min_node)

    # Getting the area that is not in stationary sensing area but defender sensing area...
    Vdef_not_Vgreen = Vsense - set().union(*Vgreen_dic.values())
    # Checking the attacker location is with current agent...
    if current_pos in Vdef_not_Vgreen:
        # Getting location that are one hope from every enemy agent...
        Vtag_1_hop =  set().union(*_k_hop_neighbors_dict(global_map_sensor,enemy_pos,1).values())
        # Getting the set of senssable node from 1-hop locations fo enemy Vtag locations...
        Vtag_1_hop_sense_nodes_dic = _get_nodes_in_range_dict(Vtag_1_hop,R_DEFENDER_SENSING,global_map_sensor)
        # Converting dictionary to the set...
        Vtag_1_hop_sense_nodes = set().union(* Vtag_1_hop_sense_nodes_dic.values())
        # Getting location that are one hope from every team agent...
        Vteam_1_hop =  set().union(*_k_hop_neighbors_dict(global_map_sensor,teammate_position,1).values())
        # Checking if any of the vteam one-hop nodes are outside of vtag one-hop sense node...
        option_node_for_attacker = Vteam_1_hop - Vtag_1_hop_sense_nodes
        if len(option_node_for_attacker) != 0:
            # Getting the best nodes from the option (min hop) as next postion...
            best_node = max(option_node_for_attacker, key=lambda n: global_map_sensor.degree(n))
            min_node = best_node
            # Creating path for the next best 1-hop neighbour...
            path_to_min_node = [current_pos,best_node]
        else:
            option_node_2 = Vteam_1_hop - Vtag
            if len(option_node_2) != 0:
                # Getting the best nodes from the option (min hop) as next postion...
                best_node = max(option_node_2, key=lambda n: global_map_sensor.degree(n))
                min_node = best_node
                # Creating path for the next best 1-hop neighbour...
                path_to_min_node = [current_pos,best_node]


    # If the path length is not zero...
    if len(path_to_min_node) != 0:
        # Getting the curent position index in the path list...
        curr_idx = path_to_min_node.index(state['curr_pos'])
        # If current position is goal/min_node don't move...
        if state['curr_pos'] == min_node:
            target: int = current_pos
        # Else selecct the next postion in path to go...
        else:
            next_node = path_to_min_node[curr_idx+1]
            target: int = next_node
    # If path length is zero don't move...
    else:
        target: int = current_pos

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


##############################################################
#                     Helper Constants....
##############################################################

# Dictionary to store the Team Vsense...
_Vsense_global = {}
# Radisu for stationary sensor observation...
R_GREEN = 450
# Radius for attacker observation of flag...
R_PROTECT_FLAG = 400
# Radius for defender sensing...
R_DEFENDER_SENSING = 250
# Radius for attacker to capture the flag...
R_BANNED = 2
# Radius for tag for defender....
R_TAG = 1


##############################################################
#                     Helper Routines....
##############################################################


# Getting the dictinary updated...
def _update_Vsense(agent,ego_nodes):
    _Vsense_global[agent] = ego_nodes

# Getting the all the Vsense values...
def _get_Vsense(dictionary_global=_Vsense_global):
    return set().union(*dictionary_global.values())

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

def _k_hop_neighbors_multi(G: nx.Graph, starts: Iterable[int], k: int) -> Set[int]:
    """
    Returns the union of nodes within <= k hops of ANY node in `starts`.
    """
    out: Set[int] = set()
    for s in starts:
        out |= _k_hop_neighbors(G, s, k)   # uses the single-source function you already have
    return out


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

