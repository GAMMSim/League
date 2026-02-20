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

att_policy_1v1_defender = None
def_policy_1v1_defender = None
V_1v1_defender = None
mamcts_defender = None

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

        self.tag_radius = int(tag_radius)
        self.capture_radius = int(capture_radius)

        # Pre-compute and cache neighbor information
        self.neighbors_cache = {}
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

def att_action_defender(jointstate):
    return att_policy_1v1_defender[jointstate]

def def_action_defender(jointstate):
    return def_policy_1v1_defender[jointstate]

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

            atk_vector = att_action_defender(joint_state)
            def_vector = def_action_defender(joint_state)

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
    value_matrix = V_1v1_defender[joint_states]  # shape (D, A)
    #print(value_matrix)
    if for_attacker:
        row_indices, col_indices = linear_sum_assignment(value_matrix, maximize=False)
        total_value = np.sum(value_matrix[row_indices, col_indices])
        value = total_value/D
    else:
        row_indices, col_indices = linear_sum_assignment(value_matrix, maximize=True)
        total_value = np.sum(value_matrix[row_indices, col_indices])
        value = total_value/D
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
    _G['real_flags'] = set(real_flags)
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
    real_flags = _G['real_flags'];candidate_flags = _G['candidate_flags'] ;neighbors_cache = _G['neighbors']
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
    #print('attacker_actions and defender',att_acts,def_acts)

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


POLICY_DIR = "policy_files"
os.makedirs(POLICY_DIR, exist_ok=True)

def get_or_solve_policy(graph, flags,n_defenders,capture_radius):
    """
    Check if policy exists for this graph+flags, else solve and save.
    """
    flag = sorted(flags)
    flags = [int(x) for x in flag]
    fname = f"defenders_{graph}_F{flags}_{n_defenders}.pkl"
    fpath = os.path.join(POLICY_DIR, fname)

    if os.path.exists(fpath):
        print(" Primitive Policy exists for this graph+flags! loading them")
        with open(fpath, "rb") as f:
            atk_policy, def_policy, value = pickle.load(f)
        return atk_policy, def_policy, value
    else:
        print("Primitive Policy doesn't exist for this graph+flags solving the game")
        game = TADGamePrimitives(graph=graph, goal_positions=flags,num_attackers = 1, num_defenders=n_defenders,capture_radius=capture_radius)
        game.generate_compressed_transitions()
        solver = NashSolver(game=game)
        solver.solve(eps=1, n_policy_eval=7, n_workers=6,save_path=None, save_checkpoint=False, verbose=True)
        att_policy_1v1_defender = solver.policy_1
        def_policy_1v1_defender = solver.policy_2
        value = solver.V

        with open(fpath, "wb") as f:
            pickle.dump([att_policy_1v1_defender,def_policy_1v1_defender, value], f)

        return att_policy_1v1_defender, def_policy_1v1_defender, value

def strategy(state):
    global att_policy_1v1_defender, def_policy_1v1_defender, V_1v1_defender, mamcts_defender, last_time
    from typing import Dict, List, Tuple, Any
    import networkx as nx
    import random

    # ============================ HELPERS (NEW) ============================

    def assign_defenders_to_flags(defenders: Dict[str, int],
                                  flags: List[int],
                                  sp_cache: Dict[int, Dict[int, float]],
                                  load_penalty: float = 3.0) -> Dict[str, int]:
        """
        Greedy, load-balanced assignment.
        cost(defender->flag) = dist(def_pos, flag) + load_penalty * (#defenders_already_assigned_to_flag)
        Returns: {defender_name: assigned_flag}
        """
        if not flags:
            return {name: None for name in defenders}

        load = {f: 0 for f in flags}
        assignment: Dict[str, int] = {}

        def min_dist_to_any_flag(pos: int) -> float:
            return min(sp_cache[pos].get(f, float("inf")) for f in flags)

        # Assign farthest defenders first (they have fewer good options)
        ordered = sorted(defenders.items(), key=lambda kv: -min_dist_to_any_flag(kv[1]))

        for name, pos in ordered:
            best_f, best_cost = None, float("inf")
            for f in flags:
                d = sp_cache[pos].get(f, float("inf"))
                cost = d + load_penalty * load[f]
                if cost < best_cost:
                    best_cost = cost
                    best_f = f
            assignment[name] = best_f
            load[best_f] += 1

        return assignment

    def step_toward_target(current_pos: int,
                           target: int,
                           neighbors: List[int],
                           sp_cache: Dict[int, Dict[int, float]]) -> int:
        """Pick a neighbor that decreases distance to target; random tie-break."""
        if target is None or not neighbors:
            return current_pos
        best_moves, best = [], float("inf")
        for n in neighbors:
            d = sp_cache[n].get(target, float("inf"))
            if d < best:
                best = d
                best_moves = [n]
            elif d == best:
                best_moves.append(n)
        return random.choice(best_moves) if best_moves else current_pos

    def choose_chase_team(defenders: Dict[str, int],
                        attacker_nodes: List[int],
                        sp_cache: Dict[int, Dict[int, float]],
                        chase_count: int) -> set:
      """Pick chase_count defenders closest (in shortest-path hops) to any detected attacker."""
      if chase_count <= 0 or not attacker_nodes: return set()

      def dist_to_attackers(dpos: int) -> float:
          return min(sp_cache[dpos].get(a, float("inf")) for a in attacker_nodes)

      ordered = sorted(defenders.items(), key=lambda kv: dist_to_attackers(kv[1]))
      return set([name for name, _ in ordered[:min(chase_count, len(ordered))]])

    # ============================ AGENT CONTROLLER ============================
    agent_ctrl = state["agent_controller"]
    current_pos: int = state["curr_pos"]
    current_time: int = state["time"]
    team: str = agent_ctrl.team
    red_payoff: float = state["payoff"]["red"]
    blue_payoff: float = state["payoff"]["blue"]

    speed: float = agent_ctrl.speed
    tagging_radius: float = agent_ctrl.tagging_radius

    # ============================ INDIVIDUAL CACHE ============================
    cache = agent_ctrl.cache
    last_target: int = cache.get("last_target", None)
    visit_count: int = cache.get("visit_count", 0)

    cache.set("last_position", current_pos)
    cache.set("visit_count", visit_count + 1)
    cache.update(last_time=current_time, patrol_index=cache.get("patrol_index", 0) + 1)

    # ============================ TEAM CACHE (SHARED) ============================
    team_cache = agent_ctrl.team_cache
    priority_targets: List[int] = team_cache.get("priority_targets", [])

    team_cache.set("last_update", current_time)
    team_cache.update(total_tags=team_cache.get("total_tags", 0), formation="defensive")

    # -------- NEW: captured attackers bookkeeping --------
    # We treat "captured" as "inactive": if an attacker isn't seen for a while, move it here.
    if "captured_attackers" not in team_cache:
        team_cache["captured_attackers"] = set()
    if "attacker_last_seen" not in team_cache:
        team_cache["attacker_last_seen"] = {}  # {attacker_name: last_seen_time}

    # ============================ AGENT MAP (SHARED) ============================
    agent_map = agent_ctrl.map
    global_map_sensor: nx.Graph = state["sensor"]["global_map"][1]["graph"]

    nodes_data: Dict[int, Dict[str, Any]] = {node_id: global_map_sensor.nodes[node_id] for node_id in global_map_sensor.nodes()}
    edges_data: Dict[int, Dict[str, Any]] = {}
    for idx, (u, v, data) in enumerate(global_map_sensor.edges(data=True)):
        edges_data[idx] = {"source": u, "target": v, **data}

    agent_map.attach_networkx_graph(nodes_data, edges_data)
    agent_map.update_time(current_time)
    agent_map.update_agent_position(team, agent_ctrl.name, current_pos, current_time)

    # Update teammate positions from custom_team sensor
    teammates_sensor: Dict[str, int] = {}
    if "custom_team" in state["sensor"]:
        teammates_sensor = state["sensor"]["custom_team"][1]
        for teammate_name, teammate_pos in teammates_sensor.items():
            agent_map.update_agent_position(team, teammate_name, teammate_pos, current_time)

    # Update enemy positions from egocentric_agent sensor
    nearby_agents: Dict[str, int] = {}
    if "egocentric_agent" in state["sensor"]:
        nearby_agents = state["sensor"]["egocentric_agent"][1]
        enemy_team: str = "red" if team == "blue" else "blue"
        for agent_name, node_id in nearby_agents.items():
            if agent_name.startswith(enemy_team):
                agent_map.update_agent_position(enemy_team, agent_name, node_id, current_time)

    teammates_data: List[Tuple[str, int, int]] = agent_map.get_team_agents(team)
    enemy_pos, enemy_age = agent_map.get_agent_position("red", "red_0")

    # ============================ SENSORS ============================
    sensors: Dict[str, Tuple[Any, Dict[str, Any]]] = state["sensor"]

    real_flags: List[int] = []
    fake_flags: List[int] = []
    if "flag" in sensors:
        real_flags = sensors["flag"][1]["real_flags"]
        fake_flags = sensors["flag"][1]["fake_flags"]

    # Stationary sensors (detect attackers near flags)
    stationary_detections: List[Dict[str, Any]] = []
    attackers_detected_near_flags: Dict[str, int] = {}
    for sensor_name in sensors:
        if sensor_name.startswith("stationary_"):
            sensor_data: Dict[str, Any] = sensors[sensor_name][1]
            fixed_pos: int = sensor_data["fixed_position"]
            detected_agents: Dict[str, Dict[str, Any]] = sensor_data["detected_agents"]
            if detected_agents:
                for agent_name, agent_data in detected_agents.items():
                    if agent_name.startswith("red"):
                        attackers_detected_near_flags[agent_name] = agent_data["node_id"]

            agent_count: int = sensor_data["agent_count"]
            stationary_detections.append({"position": fixed_pos, "detected": detected_agents, "count": agent_count})

    # Also include attackers detected near the agent (egocentric)
    attackers_near_agent: Dict[str, int] = {}
    for agent_name, node_id in nearby_agents.items():
        if agent_name.startswith("red"):
            attackers_near_agent[agent_name] = node_id
    attackers_detected_near_flags.update(attackers_near_agent)

    # =============================== DETECTION LOGGING (TEAM CACHE) ===============================
    if "detected_attackers" not in team_cache:
        team_cache["detected_attackers"] = {}

    for agent_name, node_id in attackers_detected_near_flags.items():
        if agent_name not in team_cache["detected_attackers"]:
            team_cache["detected_attackers"][agent_name] = {}
        team_cache["detected_attackers"][agent_name][current_time] = node_id

    # ================================== INIT (t==1) ===============================================
    time = current_time
    current_position = current_pos

    if time == 1:
        ap, dp, value = get_or_solve_policy(global_map_sensor, real_flags, 1, 2)
        att_policy_1v1_defender = ap
        def_policy_1v1_defender = dp
        V_1v1_defender = value
        mamcts_defender = MultiAgentMCTS(graph=global_map_sensor, candidate_flag_nodes=real_flags + fake_flags, k=0.5)


    # ============================ BUILD attackers_detected_at_time_t SAFELY ============================
    attackers_detected_at_time_t: Dict[str, int] = {}
    if team_cache.get("detected_attackers"):
        attackers_detected_at_time_t = {
            a: team_cache["detected_attackers"][a][current_time]
            for a in team_cache["detected_attackers"]
            if current_time in team_cache["detected_attackers"][a]
        }

    # ================================ AGGREGATE DETECTIONS THIS TIMESTEP ===============================
    if team_cache.get("atk_detected_cache_time") != current_time:
        team_cache["atk_detected_per_agent_at_time_t"] = {}
        team_cache["atk_detected_cache_time"] = current_time

    team_cache["atk_detected_per_agent_at_time_t"][agent_ctrl.name] = (
        attackers_detected_at_time_t.copy(), current_time, agent_ctrl.name
    )

    all_attackers_seen_now = set()
    for atk_data, t, _ in team_cache.get("atk_detected_per_agent_at_time_t", {}).values():
        if t == current_time:
            all_attackers_seen_now.update(atk_data.keys())

    previous_attackers_used = team_cache.get("attackers_used_in_last_mcts", set())
    new_attackers_detected = not all_attackers_seen_now.issubset(previous_attackers_used)

    # =================================== DECISION LOGIC ==========================================

    defenders: Dict[str, int] = dict(teammates_sensor) if teammates_sensor else {}
    defenders.update({agent_ctrl.name: current_position})
    num_def = len(defenders)
    num_atk = len(attackers_detected_at_time_t)

    # NEW Added in the new round : threshold for "few attackers" -> hybrid chase + coverage
    FEW_ATK_RATIO = 0.5
    few_attackers = (num_atk > 0) and (num_atk < FEW_ATK_RATIO * max(1, num_def))

    # 1) If enough attackers are detected => full MCTS (your original behavior)
    if num_atk > 0 and not few_attackers:
        try:
            if team_cache.get("def_action") is None or last_time != current_time or new_attackers_detected:
                _, def_action, *_ = run_mcts(
                    attacker=attackers_detected_at_time_t,
                    defender=defenders,
                    mamcts=mamcts_defender,
                    real_flags=real_flags,
                    depth=2,
                    capture_rad=2,
                    for_attacker=False,
                    n_workers=4,
                )
                team_cache["def_action"] = def_action
                team_cache["time"] = current_time
                team_cache["attackers_used_in_last_mcts"] = all_attackers_seen_now.copy()
                last_time = current_time

            state["action"] = team_cache["def_action"].get(agent_ctrl.name, None)
            return set()
        except Exception as e:
            print(f"MCTS failed: {e}, using coverage fallback")

    # 2) Coverage mode if (a) no attackers detected OR (b) few attackers detected
    #    -> spread defenders across flags (NEW)
    available_actions = list(agent_map.graph.neighbors(current_pos))
    sp = mamcts_defender.sp_cache

    # Load-balanced assignment of ALL defenders to flags
    flag_assignment = assign_defenders_to_flags(defenders, real_flags, sp_cache=sp, load_penalty=3.0)
    my_flag = flag_assignment.get(agent_ctrl.name, None)

    if few_attackers:
        # Hybrid: choose a small chase-team, rest cover flags
        atk_nodes = list(attackers_detected_at_time_t.values())
        chase_team = choose_chase_team(defenders, atk_nodes, sp_cache=sp, chase_count=min(num_def, 2 * num_atk))

        if agent_ctrl.name in chase_team:
            # Chase via shallow MCTS (only detected attackers), but keep everyone else covering
            chase_defenders = {name: defenders[name] for name in chase_team}
            try:
                _, def_action, *_ = run_mcts(
                    attacker=attackers_detected_at_time_t,
                    defender=chase_defenders,
                    mamcts=mamcts_defender,
                    real_flags=real_flags,
                    depth=2,
                    capture_rad=2,
                    for_attacker=False,
                    n_workers=4,
                )
                state["action"] = def_action.get(agent_ctrl.name, None)
                return set()
            except Exception as e:
                # If chase MCTS fails, still move toward assigned flag
                state["action"] = step_toward_target(current_pos, my_flag, available_actions, sp)
                return set()

        # Not in chase team -> cover assigned flag
        state["action"] = step_toward_target(current_pos, my_flag, available_actions, sp)
        return set()

    if my_flag is None:
        state["action"] = random.choice(available_actions) if available_actions else current_pos
    else:
        state["action"] = step_toward_target(current_pos, my_flag, available_actions, sp)

    return set()

def map_strategy(agent_config):
    strategies = {}
    for name in agent_config.keys():
        strategies[name] = strategy
    return strategies