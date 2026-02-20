from typing import Dict, List, Tuple, Any, Set, Union, Optional, Iterable, Hashable
import networkx as nx
import numpy as np
from collections import deque
import math


def strategy(state):
    
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

    # ------- #
    nodes_data: Dict[int, Dict[str, Any]] = {node_id: global_map_sensor.nodes[node_id] for node_id in global_map_sensor.nodes()}

    edges_data: Dict[int, Dict[str, Any]] = {}
    for idx, (u, v, data) in enumerate(global_map_sensor.edges(data=True)):
        edges_data[idx] = {"source": u, "target": v, **data}

    agent_map.attach_networkx_graph(nodes_data, edges_data)
    # ------- #
    
    agent_map.update_time(current_time)
    agent_map.update_agent_position(team, agent_ctrl.name, current_pos, current_time)

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

    enemy_team = "red" if team == "blue" else "blue"
    attacker_nodes: List[int] = []

    # ===== SENSORS =====
    sensors: Dict[str, Tuple[Any, Dict[str, Any]]] = state["sensor"]

    if "flag" in sensors:
        real_flags: List[int] = sensors["flag"][1]["real_flags"]  # True flag locations
        fake_flags: List[int] = sensors["flag"][1]["fake_flags"]  # Fake flag locations
        # cand_flags: List[int] = sensors["candidate_flag"][1]["candidate_flags"]

    if "custom_team" in sensors:
        teammates_sensor: Dict[str, int] = sensors["custom_team"][1]  # Teammates only

    if "egocentric_agent" in sensors:
        nearby_agents: Dict[str, int] = sensors["egocentric_agent"][1]  # Agents within sensing radius
        for nm, node in nearby_agents.items():
            if nm.startswith(enemy_team):
                attacker_nodes.append(node)

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
            for nm, info in detected_agents.items():
                if nm.startswith(enemy_team):
                    attacker_nodes.append(info.get("node_id", info.get("node", None)))

    attacker_nodes = {x for x in attacker_nodes if x is not None}
    

    # print(stationary_detections)  # For debugging
    # Getting the Vflag: True flag locations...
    Vflags: List[int] = real_flags
    # Getting the Vcand: Candidate flag locations....
    Vcand: List[int] = fake_flags + real_flags
    # vsense_stationary = _get_nodes_in_range()
    for flag in Vcand:
        vgreen_flag_i = _get_nodes_in_range([flag],R_GREEN,global_map_sensor)
        _update_Vsense(str(flag),vgreen_flag_i)
    # Getting the Vsense: Nodes where defenders can see attackers...
    _update_Vsense(agent_ctrl.name,list(visible_nodes.keys()))
    # Getting nodes that are observed by stationary flags...
    Vsense = _get_Vsense()
    # Getting the Vsense_boundary: Vsense frontier nodes...
    Vsense_boundary = _get_node_at_boundary(Vsense,global_map_sensor)
    # Getting the Vprotect:
    Vprotect = _get_nodes_in_range_dict(Vflags,R_PROTECT_FLAG,global_map_sensor)
    # Getting Green circlke nodes...
    Vgreen = _get_nodes_in_range(Vflags,R_GREEN,global_map_sensor)
    Vgreen_boundary = _get_node_at_boundary(Vgreen,global_map_sensor)
    # Getting the Vprotect_boundary:
    Vprotect_boundary = _get_node_at_boundary(Vprotect,global_map_sensor)
    # Getting the Vsecondary_protect:
    Vprotect_secondary = _get_nodes_in_range(fake_flags,R_PROTECT_FLAG,global_map_sensor)
    # Getting the Vsecondary_protect_boundary:
    Vsecondary_protect_boundary = _get_node_at_boundary(Vprotect_secondary,global_map_sensor)
    # Getting Vbanned:
    Vbanned = _k_hop_neighbors_dict(global_map_sensor,Vcand,2)
    # Getting Vbanned_boundary:
    Vbanned_boundary = _get_node_at_boundary(Vbanned,global_map_sensor)
    teammate_nodes = [node_id for (_, node_id, _) in teammates_data]
    Vtag = _k_hop_neighbors_dict(global_map_sensor,teammate_nodes,2)
    Vtag_set = set().union(*Vtag.values())
    Vattacker = attacker_nodes

    ##################################
    # Strategy for defender...
    ##################################


    # Getting the attacker teammate position...
    teammate_position = teammate_nodes

    # For each of the flag get Vgreen nodes set...
    Vgreen_dic = {i: None for i in Vcand}
    for key in _Vsense_global.keys():
        if not key.startswith("blue_"):
            Vgreen_dic[int(key)] = _Vsense_global[key]

    # Getting the Vprotect dictionary...
    Vprotect_dic = _get_nodes_in_range_dict(Vcand,R_PROTECT_FLAG,global_map_sensor)
    
    # Take difference between the Vgreen and V protect for each agent...
    Vgreen_Vprotect_Area_dic = {i:None for i in Vcand}
    for i in Vcand:
        Vgreen_Vprotect_Area_dic[i] = Vgreen_dic[i] - Vprotect_dic[i]
    # print(Vgreen_Vprotect_Area_dic)
    
    # Dictionary to maintian the attacker agent locaitons on the Vgreen_Vprotect_Area_dic region...
    Vgreen_Vprotect_Area_Attaacker_log = {i:None for i in Vcand}
    for i in Vcand:
        Vgreen_Vprotect_Area_Attaacker_log[i] = [x for x in attacker_nodes if x in Vgreen_Vprotect_Area_dic[i]]
    
    # Take difference between the Vprotext and Vtag for each agent...
    Vprotect_Vban_Area_dic = {i:None for i in Vcand}
    for i in Vcand:
        Vprotect_Vban_Area_dic[i] = Vprotect_dic[i] - Vbanned[i]

    # Dictionary to maintian the attacker agent locaitons on the Vgreen_Vprotect_Area_dic region...
    Vprotect_Vban_Area_Attacker_log = {i:None for i in Vcand}
    for i in Vcand:
       Vprotect_Vban_Area_Attacker_log[i] = [x for x in attacker_nodes if x in  Vprotect_Vban_Area_dic[i]]

    # Sorted List of attackers in Vprotect and Vban...
    high_ranked_Vattacker, _ = _sort_nodes_by_min_hops_to_targets(global_map_sensor,set().union(*Vprotect_Vban_Area_Attacker_log.values()),Vbanned_boundary)
    # Sorted List of attackers in Vprotect and Vgreen...
    medium_ranked_Vattacker, _ = _sort_nodes_by_min_hops_to_targets(global_map_sensor,set().union(*Vgreen_Vprotect_Area_Attaacker_log.values()),Vprotect_boundary)
    # Sorted List of attackers outiside Vgreen...
    low_ranked_Vattacker, _ = _sort_nodes_by_min_hops_to_targets(global_map_sensor,[atk for atk in attacker_nodes if atk not in Vgreen],_get_node_at_boundary(Vgreen,global_map_sensor))
    

    # ===== DECISION LOGIC ====
    path_to_min_node = []
    global_assigned_defender = []
    global_unassigned_defenders = []

    # Getting the list of attacker nodes that are defender sensing zone...
    if len(high_ranked_Vattacker) != 0:
       assigned_defender_r1 = []
       # Getting each attacker in high ranked Vattacker...
       for attacker in high_ranked_Vattacker:
           # Getting agent defender for given attacker with minimum hops...
           best_defender,_ = _min_hop_target(global_map_sensor,attacker,[x for x in teammate_nodes if x not in set(assigned_defender_r1)])
           if current_pos == best_defender: 
              min_node = attacker
              path_to_min_node = nx.shortest_path(global_map_sensor, source=current_pos, target=attacker)
           # Adding the best assigned defender to assinged list... 
           assigned_defender_r1.append(best_defender)
       # Getting the list of unassinged defenders...
       unassinged_defenders_r1 = [x for x in teammate_nodes if x not in set(assigned_defender_r1)]
       # Getting the list of flag allocated defender list...
       allocated_defenders_r1 = []
       # Getting the unassinged list of defenders alloted to flags...
       for flag in Vflags:
           # Getting agent defender for given attacker with minimum hops...
           best_defender,_ = _min_hop_target(global_map_sensor,flag,[x for x in unassinged_defenders_r1 if x not in set(allocated_defenders_r1)])
           if current_pos == best_defender: 
              min_node = attacker
              path_to_min_node = nx.shortest_path(global_map_sensor, source=current_pos, target=attacker)
           # Adding the best assigned defender to assinged list... 
           allocated_defenders_r1.append(best_defender)
       # Updaitng the global list values...
       global_assigned_defender.extend(assigned_defender_r1)
       global_assigned_defender.extend(allocated_defenders_r1)
       global_unassigned_defenders.extend(unassinged_defenders_r1)

    
    # Getting the list of medium rank list and check if any of the defenders are unassinged...
    if (len(medium_ranked_Vattacker) !=0) and (len(global_unassigned_defenders) != 0):
       assigned_defender_r2 = []
       # Getting each attacker in high ranked Vattacker...
       for attacker in medium_ranked_Vattacker:
           # Getting agent defender for given attacker with minimum hops...
           best_defender,_ = _min_hop_target(global_map_sensor,attacker,[x for x in teammate_nodes if x not in set(assigned_defender_r2)])
           if current_pos == best_defender: 
              min_node = attacker
              path_to_min_node = nx.shortest_path(global_map_sensor, source=current_pos, target=attacker)
           # Adding the best assigned defender to assinged list... 
           assigned_defender_r2.append(best_defender)
       # Getting the list of unassinged defenders...
       unassinged_defenders_r2 = [x for x in teammate_nodes if x not in set(assigned_defender_r2)]
       # Getting the list of flag allocated defender list...
       allocated_defenders_r2 = []
       # Getting the unassinged list of defenders alloted to flags...
       for flag in Vflags:
           # Getting agent defender for given attacker with minimum hops...
           best_defender,_ = _min_hop_target(global_map_sensor,flag,[x for x in unassinged_defenders_r2 if x not in set(allocated_defenders_r2)])
           if current_pos == best_defender: 
              min_node = attacker
              path_to_min_node = nx.shortest_path(global_map_sensor, source=current_pos, target=attacker)
           # Adding the best assigned defender to assinged list... 
           allocated_defenders_r2.append(best_defender)
       # Updaitng the global list values...
       global_assigned_defender.extend(assigned_defender_r2)
       global_assigned_defender.extend(allocated_defenders_r2)
       global_unassigned_defenders.extend(unassinged_defenders_r2)



    if (len(low_ranked_Vattacker) !=0) and (len(global_unassigned_defenders) != 0):
       assigned_defender_r3 = []
       # Getting each attacker in high ranked Vattacker...
       for attacker in low_ranked_Vattacker:
           # Getting agent defender for given attacker with minimum hops...
           best_defender,_ = _min_hop_target(global_map_sensor,attacker,[x for x in teammate_nodes if x not in set(assigned_defender_r3)])
           if current_pos == best_defender: 
              min_node = attacker
              path_to_min_node = nx.shortest_path(global_map_sensor, source=current_pos, target=attacker)
           # Adding the best assigned defender to assinged list... 
           assigned_defender_r3.append(best_defender)
       # Getting the list of unassinged defenders...
       unassinged_defenders_r3 = [x for x in teammate_nodes if x not in set(assigned_defender_r3)]
       # Getting the list of flag allocated defender list...
       allocated_defenders_r3 = []
       # Getting the unassinged list of defenders alloted to flags...
       for flag in Vflags:
           # Getting agent defender for given attacker with minimum hops...
           best_defender,_ = _min_hop_target(global_map_sensor,flag,[x for x in unassinged_defenders_r3 if x not in set(allocated_defenders_r3)])
           if current_pos == best_defender: 
              min_node = attacker
              path_to_min_node = nx.shortest_path(global_map_sensor, source=current_pos, target=attacker)
           # Adding the best assigned defender to assinged list... 
           allocated_defenders_r3.append(best_defender)
       # Updaitng the global list values...
       global_assigned_defender.extend(assigned_defender_r3)
       global_assigned_defender.extend(allocated_defenders_r3)
       global_unassigned_defenders.extend(unassinged_defenders_r3)




    if not(len(high_ranked_Vattacker) != 0 or len(medium_ranked_Vattacker) != 0 or len(low_ranked_Vattacker) != 0):
        if blue_payoff >= red_payoff:
            assigned_defender_flags = []
            # Getting each attacker in high ranked Vattacker...
            for flag in Vflags:
                # Getting agent defender for given attacker with minimum hops...
                best_defender,_ = _min_hop_target(global_map_sensor,flag,[x for x in teammate_nodes if x not in set(assigned_defender_flags)])
                if current_pos == best_defender: 
                    min_node = flag
                    path_to_min_node = nx.shortest_path(global_map_sensor, source=current_pos, target=flag)
                # Adding the best assigned defender to assinged list... 
                assigned_defender_flags.append(best_defender)
            # Getting the list of unassinged defenders...
            unassinged_defenders_flag = [x for x in teammate_nodes if x not in set(assigned_defender_flags)]
            # Getting the list of flag allocated defender list...
            allocated_defenders_Vgreen = []
            # Getting the unassinged list of defenders alloted to Vgreen boundary...
            for defender in unassinged_defenders_flag:
                if defender not in Vgreen:
                    hop_1_neighbours = _k_hop_neighbors(global_map_sensor,defender,1)
                    hop_1_neighbours_non_green = _k_hop_neighbors(global_map_sensor,defender,1) - Vgreen
                    if len(hop_1_neighbours_non_green) !=0:
                        best_node = max(hop_1_neighbours_non_green, key=lambda n: global_map_sensor.degree(n))
                        min_node = best_node
                        # Creating path for the next best 1-hop neighbour...
                        path_to_min_node = [current_pos,best_node]
                    else:
                        best_node = max(hop_1_neighbours, key=lambda n: global_map_sensor.degree(n))
                        min_node = best_node
                        # Creating path for the next best 1-hop neighbour...
                        path_to_min_node = [current_pos,best_node]
                else:
                    # Getting agent defender for given attacker with minimum hops...
                    best_Vgreen,_ = _min_hop_target(global_map_sensor,defender,global_map_sensor)
                    if current_pos == defender: 
                        min_node = best_Vgreen
                        path_to_min_node = nx.shortest_path(global_map_sensor, source=current_pos, target=best_Vgreen)
                    # Adding the best assigned defender to assinged list... 
                    allocated_defenders_Vgreen.append(best_defender)
                    # Updaitng the global list values...
                    global_assigned_defender.extend(assigned_defender_flags)
                    global_assigned_defender.extend(allocated_defenders_Vgreen)
                    global_unassigned_defenders.extend(unassinged_defenders_flag)
        else:
            # Getting the list of flag allocated defender list...
            allocated_defenders_Vgreen = []
            # Getting the unassinged list of defenders alloted to Vgreen boundary...
            for defender in global_unassigned_defenders:
                if defender not in Vgreen:
                    hop_1_neighbours = _k_hop_neighbors(global_map_sensor,defender,1)
                    hop_1_neighbours_non_green = _k_hop_neighbors(global_map_sensor,defender,1) - Vgreen
                    if len(hop_1_neighbours_non_green) !=0:
                        best_node = max(hop_1_neighbours_non_green, key=lambda n: global_map_sensor.degree(n))
                        min_node = best_node
                        # Creating path for the next best 1-hop neighbour...
                        path_to_min_node = [current_pos,best_node]
                    else:
                        best_node = max(hop_1_neighbours, key=lambda n: global_map_sensor.degree(n))
                        min_node = best_node
                        # Creating path for the next best 1-hop neighbour...
                        path_to_min_node = [current_pos,best_node]
                else:
                    # Getting agent defender for given attacker with minimum hops...
                    best_Vgreen,_ = _min_hop_target(global_map_sensor,defender,global_map_sensor)
                    if current_pos == defender: 
                        min_node = best_Vgreen
                        path_to_min_node = nx.shortest_path(global_map_sensor, source=current_pos, target=best_Vgreen)
                    # Adding the best assigned defender to assinged list... 
                    allocated_defenders_Vgreen.append(best_defender)
                    # Updaitng the global list values...
                    global_assigned_defender.extend(assigned_defender_flags)
                    global_assigned_defender.extend(allocated_defenders_Vgreen)
                    global_unassigned_defenders.extend(unassinged_defenders_flag)


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

##############################################################
#                     Helper Constants....
##############################################################

# Dictionary to store the Team Vsense...
_Vsense_global = {}
# Radius for attacker observation of flag...
R_GREEN = 450
# Radisu for stationary sensor observation...
R_PROTECT_FLAG = 400
# Radius for the defender.
R_DEFEND = 250

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

# Function to get the pairwise distance between nodes...
def _pairwise_euclidean_distances(node_set1, node_set2, global_map):
    A_ids = np.array(list(node_set1), dtype=int)
    B_ids = np.array(list(node_set2), dtype=int)
    A_xy = np.array([(global_map.nodes[u]['x'], global_map.nodes[u]['y']) for u in A_ids], dtype=float)
    B_xy = np.array([(global_map.nodes[v]['x'], global_map.nodes[v]['y']) for v in B_ids], dtype=float) 
    diff = A_xy[:, None, :] - B_xy[None, :, :]
    D = np.linalg.norm(diff, axis=2)
    return A_ids, B_ids, D 

# Function to ge the hop matrix...
def _hop_matrix(
    G: nx.Graph,
    listA: List[int],
    listB: List[int],
    unreachable_value: Union[int, float] = np.inf,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      A_ids: (m,) int array of nodes from listA (filtered to nodes that exist in G)
      B_ids: (n,) int array of nodes from listB (filtered to nodes that exist in G)
      H:     (m,n) array, H[i,j] = min hops from A_ids[i] to B_ids[j]
             (unreachable -> unreachable_value)

    Notes:
      - Uses BFS (unweighted shortest paths).
      - If you want unreachable as -1, pass unreachable_value=-1.
    """
    # Keep only nodes that exist in G (avoid NetworkX errors)
    A_ids = np.array([u for u in listA if u in G], dtype=int)
    B_ids = np.array([v for v in listB if v in G], dtype=int)

    m, n = len(A_ids), len(B_ids)
    H = np.full((m, n), unreachable_value, dtype=float if unreachable_value is np.inf else type(unreachable_value))

    if m == 0 or n == 0:
        return A_ids, B_ids, H

    # For fast lookup of which B nodes we still need distances to
    B_set = set(B_ids)
    B_index: Dict[int, int] = {node: j for j, node in enumerate(B_ids)}

    # BFS from each A node, but stop early once all B nodes found
    for i, src in enumerate(A_ids):
        q = deque([src])
        dist = {src: 0}
        remaining = set(B_set)

        # If src itself is in B, record 0 hop
        if src in remaining:
            H[i, B_index[src]] = 0
            remaining.remove(src)
            if not remaining:
                continue

        while q and remaining:
            u = q.popleft()
            du = dist[u]
            for w in G.neighbors(u):
                if w not in dist:
                    dist[w] = du + 1
                    # If this node is one we care about, record it
                    if w in remaining:
                        H[i, B_index[w]] = dist[w]
                        remaining.remove(w)
                        if not remaining:
                            break
                    q.append(w)

    return A_ids, B_ids, H


# Fucntion to get the flag assignment based on hopping distance...
def _assign_flags_by_hops_iterative(
    global_map_sensor,
    teammate_nodes: List[int],
    Vflags: List[int],
    agent_ids: Optional[List[int]] = None,
    hop_matrix_fn=None,   # pass your _hop_matrix here if name differs
) -> Tuple[Dict[int, Optional[int]], List[int], List[int]]:
    """
    Iteratively assigns flags to agents using your 'agent picks nearest flag,
    then flag picks best agent' rule, repeating on leftover agents/flags.

    Returns:
      assignment: dict {flag_node: agent_id or None}
      unassigned_agents: list of agent_ids not used
      unassigned_flags: list of flag nodes that couldn't be assigned
    """
    if hop_matrix_fn is None:
        hop_matrix_fn = _hop_matrix  # expects your function exists

    if agent_ids is None:
        agent_ids = list(range(len(teammate_nodes)))

    # Final assignment over ORIGINAL flags
    assignment: Dict[int, Optional[int]] = {flag: None for flag in Vflags}

    # Working sets (we shrink these each round)
    remaining_flags = list(Vflags)
    remaining_agents = list(agent_ids)
    remaining_nodes = list(teammate_nodes)

    while remaining_flags and remaining_agents:
        # Compute hop matrix for current remaining agents -> remaining flags
        _, _, hop_mat = hop_matrix_fn(global_map_sensor, remaining_nodes, remaining_flags)
        # hop_mat shape: (num_agents, num_flags)

        # 1) Each agent chooses its nearest reachable flag (finite hop)
        #    Store tuples: (agent_id, chosen_flag, min_hop)
        agent_choices: List[Tuple[int, int, float]] = []
        for i in range(hop_mat.shape[0]):
            row = hop_mat[i, :]

            # If all are inf/unreachable, skip this agent this round
            finite_mask = np.isfinite(row)
            if not np.any(finite_mask):
                continue

            # choose min among finite only
            finite_indices = np.where(finite_mask)[0]
            j_best = finite_indices[np.argmin(row[finite_indices])]
            min_hop = row[j_best]
            chosen_flag = remaining_flags[j_best]

            agent_choices.append((remaining_agents[i], chosen_flag, float(min_hop)))

        # 2) For each flag, pick the best (min hop) agent among those who chose it
        # flag -> list of (agent_id, hop)
        flag_to_candidates: Dict[int, List[Tuple[int, float]]] = {f: [] for f in remaining_flags}
        for agent_id, flag, hop in agent_choices:
            flag_to_candidates[flag].append((agent_id, hop))

        # Proposed assignments this round
        round_assignment: Dict[int, int] = {}
        used_agents_this_round = set()

        for flag in remaining_flags:
            cands = flag_to_candidates[flag]
            if not cands:
                continue
            best_agent, best_hop = min(cands, key=lambda t: t[1])
            # agent can only be used once (it is anyway, but keep safe)
            if best_agent not in used_agents_this_round:
                round_assignment[flag] = best_agent
                used_agents_this_round.add(best_agent)

        # Write into final assignment dict (only if not already assigned)
        for flag, agent_id in round_assignment.items():
            if assignment[flag] is None:
                assignment[flag] = agent_id

        # 3) Update remaining flags and remaining agents/nodes for next iteration
        newly_assigned_flags = set(round_assignment.keys())
        remaining_flags = [f for f in remaining_flags if f not in newly_assigned_flags]

        remaining_agents_next = []
        remaining_nodes_next = []
        for a_id, node in zip(remaining_agents, remaining_nodes):
            if a_id not in used_agents_this_round:
                remaining_agents_next.append(a_id)
                remaining_nodes_next.append(node)

        remaining_agents = remaining_agents_next
        remaining_nodes = remaining_nodes_next

        # If nothing got assigned this round, break to avoid infinite loop
        if not newly_assigned_flags:
            break

    # Anything still None is unassigned
    unassigned_flags = [f for f, a in assignment.items() if a is None]

    assigned_agents = {a for a in assignment.values() if a is not None}
    unassigned_agents = [a for a in agent_ids if a not in assigned_agents]

    return assignment, unassigned_agents, unassigned_flags


def assign_flags_by_hops_iterative_nodekey(
    global_map_sensor,
    teammate_nodes: List[int],   # agent current nodes (these become dict keys)
    Vflags: List[int],           # goal/flag nodes (dict values)
    hop_matrix_fn,               # your _hop_matrix(...)
) -> Tuple[Dict[int, Optional[int]], List[int], List[int]]:
    """
    Returns:
      node_to_flag: {teammate_node: flag_node or None}
      unassigned_nodes: list of teammate_node keys that remain unassigned
      unassigned_flags: list of flags not assigned
    """

    # Final mapping over original teammate_nodes
    node_to_flag: Dict[int, Optional[int]] = {n: None for n in teammate_nodes}

    remaining_nodes = list(teammate_nodes)
    remaining_flags = list(Vflags)

    while remaining_nodes and remaining_flags:
        # hop_mat rows correspond to A_ids, columns correspond to B_ids
        A_ids, B_ids, hop_mat = hop_matrix_fn(global_map_sensor, remaining_nodes, remaining_flags)

        active_nodes = list(A_ids)
        active_flags = list(B_ids)

        # If filtering removed everything or matrix empty, stop
        if hop_mat.size == 0 or len(active_nodes) == 0 or len(active_flags) == 0:
            break

        # 1) Each node (agent position) chooses its nearest reachable flag
        # store: (node, chosen_flag, hop)
        node_choices = []
        for i, node in enumerate(active_nodes):
            row = hop_mat[i, :]

            finite_mask = np.isfinite(row)
            if not np.any(finite_mask):
                continue

            finite_idxs = np.where(finite_mask)[0]
            j_best = finite_idxs[np.argmin(row[finite_idxs])]
            chosen_flag = active_flags[j_best]
            chosen_hop = float(row[j_best])
            node_choices.append((node, chosen_flag, chosen_hop))

        # If no node can reach any flag, stop
        if not node_choices:
            break

        # 2) For each flag, pick the best (min hop) node among those who chose it
        flag_to_candidates = {f: [] for f in active_flags}
        for node, flag, hop in node_choices:
            flag_to_candidates[flag].append((node, hop))

        # Round assignments: flag -> node (ensure each node used once)
        round_flag_to_node = {}
        used_nodes = set()

        for flag, cands in flag_to_candidates.items():
            if not cands:
                continue
            best_node, best_hop = min(cands, key=lambda t: t[1])
            if best_node not in used_nodes:
                round_flag_to_node[flag] = best_node
                used_nodes.add(best_node)

        if not round_flag_to_node:
            break

        # 3) Commit into final mapping node -> flag
        for flag, node in round_flag_to_node.items():
            if node_to_flag[node] is None:
                node_to_flag[node] = flag

        # 4) Remove assigned flags and assigned nodes for next iteration
        assigned_flags = set(round_flag_to_node.keys())
        assigned_nodes = set(round_flag_to_node.values())

        remaining_flags = [f for f in remaining_flags if f not in assigned_flags]
        remaining_nodes = [n for n in remaining_nodes if n not in assigned_nodes]

    unassigned_nodes = [n for n, f in node_to_flag.items() if f is None]
    unassigned_flags = list(remaining_flags)

    return node_to_flag, unassigned_nodes, unassigned_flags





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

# Function to get agent sorted based on min hop distance to boundary...
def _sort_nodes_by_min_hops_to_targets(
    G: nx.Graph,
    sources: List[int],   # list1
    targets: List[int],   # list2
    unreachable_value=float("inf"),
) -> Tuple[List[int], Dict[int, float]]:
    """
    Returns:
      sorted_sources: sources sorted by min hop distance to ANY target
      src_to_min_hops: dict {source_node: min_hops_to_any_target} (inf if unreachable)

    Notes:
      - Uses multi-source BFS from targets (unweighted).
      - If a node in sources/targets is not in G, it is treated as unreachable.
    """
    # Keep only targets that exist in the graph
    target_set = [t for t in targets if t in G]
    src_to_min_hops: Dict[int, float] = {s: unreachable_value for s in sources}

    # Edge cases
    if not sources:
        return [], src_to_min_hops
    if not target_set:
        return list(sources), src_to_min_hops  # all unreachable

    # Multi-source BFS: dist_to_target[u] = min hops from u to any target
    dist_to_target: Dict[int, int] = {}
    q = deque()

    for t in target_set:
        dist_to_target[t] = 0
        q.append(t)

    while q:
        u = q.popleft()
        du = dist_to_target[u]
        for v in G.neighbors(u):
            if v not in dist_to_target:
                dist_to_target[v] = du + 1
                q.append(v)

    # Fill min-hop values for each source
    for s in sources:
        if s in dist_to_target:
            src_to_min_hops[s] = float(dist_to_target[s])

    # Sort sources by (min_hops, node_id) for deterministic ordering
    sorted_sources = sorted(sources, key=lambda s: (src_to_min_hops[s], s))

    return sorted_sources, src_to_min_hops




