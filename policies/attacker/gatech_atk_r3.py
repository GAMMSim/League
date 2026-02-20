from Game_Generation import TADGamePrimitives
import networkx as nx

from solver import NashSolver
from nash_utils import linprog_solve
import pickle
import os
import numpy as np
import itertools
import multiprocessing as mp
from scipy.optimize import linear_sum_assignment

att_policy_1v1 = None
def_policy_1v1 = None
V_1v1 = None
mamcts = None
last_time = 0
real_flags_discovered = None

class MCTSNode:
    def __init__(self, state, depth,parent=None, action=None):
        self.state = state  # (attacker_dict, defender_dict)
        self.depth = depth  # Remaining lookahead depth
        self.parent = parent  # Parent MCTSNode
        self.action = action  # (att_joint_action, def_joint_action)
        self.children = []  # List of MCTSNode children
        self.value = None  # Final backed-up value

class MultiAgentMCTS:
    def __init__(self,graph,candidate_flag_nodes,k = 1.0,tag_radius = 1,capture_radius = 2,gamma = 0.95):
        self.graph = graph
        self.candidate_flag_nodes = set(candidate_flag_nodes)
        self.k = k
        self.gamma = gamma
        self.n_nodes = len(graph.nodes)
        self.n_workers = 4
        self.sp_cache = dict(nx.all_pairs_shortest_path_length(graph))
        self.fixed_defender = None

        self.tag_radius = int(tag_radius)
        self.capture_radius = int(capture_radius)

        # Pre-compute and cache neighbor information
        self.neighbors_cache = {}#np.empty(self.n_nodes,dtype = object)
        self._precompute_neighbors()

    def _precompute_neighbors(self):
        """Pre-compute all neighbor relationships to avoid repeated calculations"""
        for node in self.graph.nodes:
            sorted_neighbors = sorted(list(self.graph.neighbors(node)))
            sorted_neighbors.remove(node) if node in sorted_neighbors else None
            neighbors = sorted_neighbors+[node]
            self.neighbors_cache[node] = neighbors


    def get_neighbors(self, pos):
        return self.neighbors_cache[pos]

def att_action(jointstate):
    return att_policy_1v1[jointstate]

def def_action(jointstate):
    return def_policy_1v1[jointstate]

def rollout_policy_combined_better(state, n_nodes, neighbours_cache,sp_cache ,real_flags,min_actions_a = 1, min_actions_d = 1,n_closest_defenders = 3):
    attacker_dict, defender_dict = state
    attacker_names = sorted(attacker_dict.keys())
    defender_names = sorted(defender_dict.keys())
    n_attackers = len(attacker_names)
    n_defenders = len(defender_names)

    attacker_actions = [set() for _ in range(n_attackers)]
    defender_actions = [set() for _ in range(n_defenders)]

    for i in range(n_attackers):
        atk_pos = attacker_dict[attacker_names[i]]
        atk_neighbors = neighbours_cache[atk_pos]

        if n_defenders <= n_closest_defenders:
            closest_defenders_name = defender_names
        else:
            defender_distance = {defender_name: sp_cache[atk_pos].get(defender_pos, float('inf')) for
                                 defender_name, defender_pos in defender_dict.items()}
            sorted_defenders = sorted(defender_names, key=lambda d: defender_distance[d])
            closest_defenders_name = sorted_defenders[:n_closest_defenders]
        #print(closest_defenders_name)

        for j, def_name in enumerate(closest_defenders_name):
            def_pos = defender_dict[def_name]
            def_neighbors = neighbours_cache[def_pos]

            joint_state = def_pos * n_nodes + atk_pos

            atk_vector = att_action(joint_state)
            def_vector = def_action(joint_state)

            atk_sample = atk_neighbors[np.random.choice(len(atk_vector), p=atk_vector)]
            def_sample = def_neighbors[np.random.choice(len(def_vector), p=def_vector)]

            attacker_actions[i].add(atk_sample)
            def_idx = defender_names.index(def_name)
            defender_actions[def_idx].add(def_sample)

    # Add perturbation for sparse agents
    for i in range(n_attackers):
        if len(attacker_actions[i]) < min_actions_a :
            pos = attacker_dict[attacker_names[i]]
            neighbors = neighbours_cache[pos]
            candidates = list(set(neighbors) - attacker_actions[i])
            min_flag = min(real_flags, key=lambda f: sp_cache[pos].get(f, float('inf')))
            best_moves = []
            min_dist = float('inf')
            for n in candidates:
                d = sp_cache[n].get(min_flag, float('inf'))
                if d < min_dist:
                    min_dist = d
                    best_moves = [n]
                # elif d == min_dist:
                #     best_moves.append(n)
            attacker_actions[i].add(best_moves[0])


    for j in range(n_defenders):
        if len(defender_actions[j]) < min_actions_d :
            pos = defender_dict[defender_names[j]]
            neighbors = neighbours_cache[pos]
            candidates = list(set(neighbors) - defender_actions[j])
            np.random.shuffle(candidates)
            defender_actions[j].update(candidates[:1])

    # print(attacker_actions)
    # print(defender_actions)
    att_joint_actions = list(itertools.product(*[list(a) for a in attacker_actions]))
    def_joint_actions = list(itertools.product(*[list(d) for d in defender_actions]))
    return attacker_names, defender_names, att_joint_actions, def_joint_actions

def leaf_value_estimate(state,n_nodes,for_attacker :bool,gamma = 0.95):
    attacker_dict, defender_dict = state
    attacker_names = sorted(attacker_dict.keys())
    defender_names = sorted(defender_dict.keys())

    atk_positions = np.array([attacker_dict[name] for name in attacker_names])  # (A,)
    def_positions = np.array([defender_dict[name] for name in defender_names])  # (D,)

    A = len(atk_positions)
    D = len(def_positions)
    # Broadcast: joint_state_matrix[d, a] = def_pos[d] * n_nodes + atk_pos[a]
    joint_states = def_positions[:, None] * n_nodes + atk_positions[None, :]  # shape (D, A)
    #print(V_1v1[joint_states])
    value_matrix = V_1v1[joint_states]  # shape (D, A)
    #print(value_matrix)
    if for_attacker:
        row_indices, col_indices = linear_sum_assignment(value_matrix, maximize=False)
        total_value = np.sum(value_matrix[row_indices, col_indices])
        value = total_value/D
    else:
        row_indices, col_indices = linear_sum_assignment(value_matrix, maximize=True)
        total_value = np.sum(value_matrix[row_indices, col_indices])
        value = total_value/A
    return value

def filter_active_agents(attacker_dict, defender_dict,real_flag_nodes,candidate_flag_nodes,sp_cache, tag_radius = 1, capture_radius = 2):
    surviving_att,surviving_def = {},{}
    captured_attackers = set()
    capturing_defenders = set()
    flag_reached = set()
    used_defenders = set()
    candidate_flag_reached = set()
    # print(attacker_dict)
    for a_name, a_pos in attacker_dict.items():
        captured = False
        for d_name, d_pos in defender_dict.items():
            if d_name in used_defenders:
                continue
            dist_ad = sp_cache[a_pos].get(d_pos, float('inf'))
            if dist_ad <= tag_radius:
                captured = True
                capturing_defenders.add(d_name)
                captured_attackers.add(a_name)
                used_defenders.add(d_name)
                break
        if not captured:
            for f in real_flag_nodes:
                if sp_cache[a_pos].get(f, float('inf')) <= capture_radius:
                    flag_reached.add(a_name)
                    break
                else:
                    surviving_att[a_name] = a_pos
            for f in candidate_flag_nodes:
                if sp_cache[a_pos].get(f, float('inf')) <= capture_radius:
                    candidate_flag_reached.add(a_name)

    for d_name, d_pos in defender_dict.items():
        if d_name not in capturing_defenders:
            surviving_def[d_name] = d_pos
    return surviving_att, surviving_def, captured_attackers, capturing_defenders, flag_reached,candidate_flag_reached

_G = {}
def _init_worker(neighbors_cache,candidate_flags,real_flags,n_nodes, k, gamma,sp_cache,capture_radius,atk_names, def_names, att_act, def_act):

    _G['neighbors'] = neighbors_cache
    _G['real_flags_discovered'] = set(real_flags)
    _G['candidate_flags'] = set(candidate_flags)
    _G['n_nodes'] = n_nodes
    _G['k'] = k
    _G['gamma'] = gamma
    _G['sp_cache'] = sp_cache
    _G['atk_names'] = atk_names
    _G['def_names'] = def_names
    _G['att_acts'] = att_act
    _G['def_acts'] = def_act
    _G['capture_radius'] = capture_radius

def _eval_subtree(args):
    state, depth,att_idx,def_idx,for_attacker = args
    real_flags = _G['real_flags_discovered'];candidate_flags = _G['candidate_flags'] ;neighbors_cache = _G['neighbors']
    n_nodes = _G['n_nodes'];k = _G['k'];gamma = _G['gamma'];sp_cache = _G['sp_cache'];capture_rad = _G['capture_radius']
    att_dict,def_dict = state
    surviving_att,surviving_def,captured,_,flags_reached,candidate_flags_reached = filter_active_agents(att_dict,def_dict,real_flags,candidate_flags,sp_cache,capture_radius = capture_rad)
    running = -k*len(captured)+len(flags_reached) + 0.01*len(candidate_flags_reached)

    if not surviving_def or not surviving_att:
        return att_idx,def_idx,running

    if depth == 0:
        return att_idx, def_idx,running +gamma*leaf_value_estimate((surviving_att,surviving_def),n_nodes,for_attacker)

    atk_names, def_names, att_acts, def_acts = rollout_policy_combined_better((surviving_att, surviving_def), n_nodes,neighbors_cache, sp_cache, real_flags,2, 2, 3)
    #print('no of actions at depth d ', depth, 'att_actions',len(att_acts),'def_actions' ,len(def_acts))

    q = np.zeros((len(att_acts), len(def_acts)))
    for atk_idx, att_move in enumerate(att_acts):
        new_att = {name:pos for name, pos in zip(atk_names, att_move)}
        for deff_idx, def_move in enumerate(def_acts):
            new_def = {name:pos for name, pos in zip(def_names, def_move)}
            _,_,value = _eval_subtree(((new_att,new_def),depth-1,atk_idx,deff_idx,for_attacker))
            q[atk_idx, deff_idx] = value
    #print(q)
    if for_attacker:
        game_val = np.max(np.min(q, axis=1))
    else:
        game_val = np.min(np.max(q, axis=0))

    value = running + gamma * game_val
    return att_idx,def_idx,value


def run_mcts(attacker, defender,mamcts,real_flags,capture_rad,depth=2,for_attacker = True ,n_workers=4):

    #print('real_flags in the tree search',real_flags)
    root = MCTSNode(state=(attacker.copy(), defender.copy()), depth=depth)
    if len(attacker) <= 4:
        min_actions_a = 2
    else:
        min_actions_a = 1
    if len(defender) <= 4:
        min_actions_d = 2
    else:
        min_actions_d = 1
    atk_names, def_names, att_acts, def_acts = rollout_policy_combined_better((attacker, defender), mamcts.n_nodes,mamcts.neighbors_cache,mamcts.sp_cache,real_flags ,min_actions_a, min_actions_d, 3)
    #print(len(att_acts),len(def_acts))

    joint_actions = len(att_acts) * len(def_acts)
    if joint_actions < 300:
        n_workers = 1

    tasks = []
    for att_idx, att_act in enumerate(att_acts):
        new_att = {name: pos for name, pos in zip(atk_names, att_act)}
        for def_idx, def_act in enumerate(def_acts):
            new_def = {name: pos for name, pos in zip(def_names, def_act)}
            tasks.append(((new_att, new_def), depth - 1, att_idx, def_idx, for_attacker))

    q_matrix = np.zeros((len(att_acts), len(def_acts)))

    initargs = (mamcts.neighbors_cache,mamcts.candidate_flag_nodes,real_flags,
                      mamcts.n_nodes, mamcts.k, mamcts.gamma,mamcts.sp_cache,capture_rad,atk_names,def_names,att_acts,def_acts)

    ctx = mp.get_context('fork')

    with ctx.Pool(processes=n_workers,initializer = _init_worker,
                 initargs = initargs) as pool:
        results = pool.map(_eval_subtree, tasks)
    for att_idx, def_idx, child_value in results:
        q_matrix[att_idx, def_idx] = child_value

    game_value,att_policy,def_policy = linprog_solve(q_matrix)

    root.value = game_value
    att_idx = np.random.choice(len(att_policy), p=att_policy)
    def_idx = np.random.choice(len(def_policy), p=def_policy)
    root.value = game_value

    best_att = {name: pos for name, pos in zip(atk_names, att_acts[att_idx])}
    best_def = {name: pos for name, pos in zip(def_names, def_acts[def_idx])}

    return best_att,best_def,root,att_policy,def_policy,q_matrix


def extract_attackers_and_defenders(agent_data):
    attackers = {}
    defenders = {}
    for agent_name in agent_data:
        if 'red' in agent_name:
            attackers[agent_name] = agent_data[agent_name]
        else:
            defenders[agent_name] = agent_data[agent_name]
    return attackers, defenders


POLICY_DIR = "policy_files"
os.makedirs(POLICY_DIR, exist_ok=True)

def get_or_solve_policy(graph, flags,n_defenders,capture_radius):
    """
    Check if policy exists for this graph+flags, else solve and save.
    """
    flag = sorted(flags)
    flags = [int(x) for x in flag]
    fname = f"attackers_{graph}_F{flags}_{n_defenders}.pkl"
    fpath = os.path.join(POLICY_DIR, fname)

    if os.path.exists(fpath):
        print(" Primitive Policy exists for this graph+flags! loading them")
        with open(fpath, "rb") as f:
            atk_policy, def_policy, value = pickle.load(f)
        return atk_policy, def_policy, value
    else:
        print("Primitive Policy doesn't exist for this graph+flags solving the game")
        game = TADGamePrimitives(graph=graph, goal_positions=flags,num_attackers = 1, num_defenders=n_defenders,capture_radius=1)
        game.generate_compressed_transitions()
        solver = NashSolver(game=game)
        solver.solve(eps=1, n_policy_eval=7, n_workers=6,save_path=None, save_checkpoint=False, verbose=True)
        att_policy_1v1 = solver.policy_1
        def_policy_1v1 = solver.policy_2
        value = solver.V

        with open(fpath, "wb") as f:
            pickle.dump([att_policy_1v1,def_policy_1v1, value], f)

        return att_policy_1v1, def_policy_1v1, value

def strategy(state):
    global att_policy_1v1, def_policy_1v1,V_1v1,mamcts,last_time,real_flags_discovered
    from typing import Dict, List, Tuple, Any
    import networkx as nx

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
    cache.update(last_time=current_time,
                 patrol_index=cache.get("patrol_index", 0) + 1)  # Batch update multiple cache values

    # ===== TEAM CACHE (SHARED) =====
    team_cache = agent_ctrl.team_cache  # Shared storage across all teammates

    priority_targets: List[int] = team_cache.get("priority_targets", [])  # How to get data from team cache

    team_cache.set("last_update", current_time)  # Track when team cache was last modified
    team_cache.update(total_captures=team_cache.get("total_captures", 0),
                      formation="spread")  # Update team-wide statistics

    # ===== AGENT MAP (SHARED) =====
    agent_map = agent_ctrl.map  # Team-shared map with positions and graph
    global_map_sensor: nx.Graph = state["sensor"]["global_map"][1]["graph"]  # Full graph topology from sensor

    nodes_data: Dict[int, Dict[str, Any]] = {node_id: global_map_sensor.nodes[node_id] for node_id in
                                             global_map_sensor.nodes()}  # Convert graph nodes to dict format

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
    #print('teammates_data', teammates_data)
    # How to get a specific agent's position from agent map
    enemy_pos, enemy_age = agent_map.get_agent_position("blue","blue_0")  # (position, age_in_timesteps) or (None, None)

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

    prev = team_cache.get('detected_flags')
    if prev is None or count > prev[1]:
        team_cache['detected_flags'] = (detected, count)
        #print('Updated detected_flags:', detected, count)

    if team_cache.get('detected_flags')[0]:
        # We’ve already detected some real flags — use them
        real_flags_discovered = team_cache['detected_flags'][0]
    else:
        # No real detections yet — fall back to candidates
        real_flags_discovered = candidates


    if "egocentric_flag" in sensors:
        flags: List[int] = sensors["egocentric_flag"][1]["detected_flags"]  # Flags visible to agent

    time = state["time"]
    current_position = state['curr_pos']

    if time == 1:
        ap, dp, value = get_or_solve_policy(global_map_sensor, candidates, 1,capture_radius)
        att_policy_1v1 = ap
        def_policy_1v1 = dp
        V_1v1 = value
        mamcts = MultiAgentMCTS(graph=global_map_sensor, candidate_flag_nodes=candidates,k=0.5)


    #print('all agents visible to the particular attacker',agent_ctrl.name,all_agents)
    if agent_ctrl.name not in all_agents:
        all_agents[agent_ctrl.name] = current_position
    attackers, defenders = extract_attackers_and_defenders(all_agents)
    #print('all agents visible to the particular attacker',agent_ctrl.name,all_agents)
    #print(attackers, defenders)
    # ===== OUTPUT =====
    try:
        if team_cache.get('att_action') is None or last_time != time:
            atk_action, *_ = run_mcts(attacker=attackers, defender=defenders, mamcts=mamcts, real_flags=real_flags_discovered, depth=2, capture_rad=capture_radius
                                      , for_attacker=True, n_workers=4)

            team_cache['att_action'] = atk_action
            team_cache['time'] = time
            last_time = time
            #print('attackers action at time',time,atk_action)

        state['action'] = team_cache['att_action'].get(agent_ctrl.name, None)
        #print(state['action'])
    except Exception as e:
        print(f"MCTS failed: {e}, using current position")
        state['action'] = current_position


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
