import networkx as nx
import pathlib
import pickle
import os
import numpy as np
from typing import List, Tuple, Union
from tqdm import tqdm
from copy import deepcopy
import time
import itertools
from multiprocessing import Pool
import multiprocessing as mp
from scipy.optimize import linear_sum_assignment
from scipy.optimize import linprog

#######################################################################################
################################ TREE SEARCH HELPERS ###################################
#######################################################################################

att_policy_1v1_defender = None
def_policy_1v1_defender = None
V_1v1_defender = None
mamcts_defender = None
last_time_defender = None

class MCTSNode:
    def __init__(self, state, depth,parent=None, action=None):
        self.state = state  # (attacker_dict, defender_dict)
        self.depth = depth  # Remaining lookahead depth
        self.parent = parent  # Parent MCTSNode
        self.action = action  # (att_joint_action, def_joint_action)
        self.children = []  # List of MCTSNode children
        self.value = None  # Final backed-up value

class MultiAgentMCTS:
    def __init__(self,graph,flag_nodes,candidate_flag_nodes,k = 1.0,tag_radius = 1,capture_radius = 2,gamma = 0.95):
        self.graph = graph
        self.candidate_flag_nodes = set(candidate_flag_nodes)
        self.flag_nodes = set(flag_nodes)
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

    # Add shortest path action for sparse agents
    for i in range(n_attackers):
        if len(attacker_actions[i]) < min_actions_a:
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
            attacker_actions[i].add(best_moves[0])
   
    for j in range(n_defenders):
        if len(defender_actions[j]) < min_actions_d:
            pos = defender_dict[defender_names[j]]
            neighbors = neighbours_cache[pos]
            candidates = list(set(neighbors) - defender_actions[j])
            if candidates:
                # Step toward nearest attacker; fall back to nearest flag if no attackers
                if attacker_dict:
                    target = min(attacker_dict.values(),key=lambda a: sp_cache[pos].get(a, float('inf')))
                else:
                    target = min(real_flags,key=lambda f: sp_cache[pos].get(f, float('inf')))
                best = min(candidates, key=lambda n: sp_cache[n].get(target, float('inf')))
                defender_actions[j].add(best)

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


def leaf_value_estimate_batch(att_pos: np.ndarray, def_pos: np.ndarray,
                               n_nodes: int) -> np.ndarray:
    """
    Vectorized leaf value estimate over all (attacker_action, defender_action) pairs.

    att_pos : (A, Na)  — A joint attacker actions, Na attacker positions each
    def_pos : (D, Nd)  — D joint defender actions, Nd defender positions each
    Returns : (A, D)   — value[a, d] = optimal 1v1-assignment value for
                         Na attackers at att_pos[a] vs Nd defenders at def_pos[d].

    Builds the (Nd × Na) 1v1-primitive matrix for each (a, d) pair, solves
    linear_sum_assignment (defender maximises), and normalises by Nd — identical
    convention to leaf_value_estimate.
    """
    A, Na = att_pos.shape
    D, Nd = def_pos.shape

    # joint_states[a, d, nd, na] = def_pos[d, nd] * n_nodes + att_pos[a, na]
    joint_states = (def_pos[None, :, :, None].astype(np.int64) * n_nodes
                    + att_pos[:, None, None, :].astype(np.int64))  # (A, D, Nd, Na)

    value_cube = V_1v1_defender[joint_states]  # (A, D, Nd, Na)

    values = np.zeros((A, D))
    for a in range(A):
        for d in range(D):
            mat = value_cube[a, d]  # (Nd, Na) — row=defender, col=attacker
            r_idx, c_idx = linear_sum_assignment(mat, maximize=False)
            # print('length of the matching for the attacker',len(r_idx))
            # print("no of defenders and attackers",Nd,Na)
            values[a, d] = np.sum(mat[r_idx, c_idx]) / Nd

    return values

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
            reached = any(sp_cache[a_pos].get(f, float('inf')) <= capture_radius for f in real_flag_nodes)
            if reached:
                flag_reached.add(a_name)
            else:
                surviving_att[a_name] = a_pos

            for f in candidate_flag_nodes:
                if sp_cache[a_pos].get(f, float('inf')) <= capture_radius:
                    candidate_flag_reached.add(a_name)

    for d_name, d_pos in defender_dict.items():
        if d_name not in capturing_defenders:
            surviving_def[d_name] = d_pos
    return surviving_att, surviving_def, captured_attackers, capturing_defenders, flag_reached,candidate_flag_reached


POLICY_DIR = "policy_files"
os.makedirs(POLICY_DIR, exist_ok=True)


def get_or_solve_policy(graph, flags, n_defenders, capture_radius):
    """
    Check if policy exists for this graph+flags, else solve and save.
    """
    flag = sorted(flags)
    flags = [int(x) for x in flag]
    fname = f"defenders_{graph}_F{flags}_{n_defenders}.pkl"
    fpath = os.path.join(POLICY_DIR, fname)

    if os.path.exists(fpath):
        # print(" Primitive Policy exists for this graph+flags! loading them")
        with open(fpath, "rb") as f:
            atk_policy, def_policy, value = pickle.load(f)
        return atk_policy, def_policy, value
    else:
        print("Primitive Policy doesn't exist for this graph+flags solving the game")
        game = TADGamePrimitives(graph=graph, goal_positions=flags, num_attackers=1, num_defenders=n_defenders,
                                 capture_radius=capture_radius)
        game.generate_compressed_transitions()
        solver = NashSolver(game=game)
        solver.solve(eps=1, n_policy_eval=7, n_workers=6, save_path=None, save_checkpoint=False, verbose=True)
        att_policy_1v1_defender = solver.policy_1
        def_policy_1v1_defender = solver.policy_2
        value = solver.V

        with open(fpath, "wb") as f:
            pickle.dump([att_policy_1v1_defender, def_policy_1v1_defender, value], f)

        return att_policy_1v1_defender, def_policy_1v1_defender, value

#######################################################################################
################################ REGRET MATCHING  ###################################
#######################################################################################
# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

class RMNode:
    """
    Node for Regret-Matching simultaneous-move MCTS.

    Stores cumulative regret and average-strategy tables for both players.
    Child means are cached in `child_value_sum` / `child_visit_count` and
    are used to estimate the expected value of un-sampled joint actions.
    """
    __slots__ = (
        "state", "depth", "parent",
        "att_names", "def_names",
        "att_actions", "def_actions",
        "n_att", "n_def",
        # Regret tables  r_i[a]
        "att_regret",   # shape (n_att,)
        "def_regret",   # shape (n_def,)
        # Average strategy tables  σ̄_i[a]
        "att_avg",      # shape (n_att,)
        "def_avg",      # shape (n_def,)
        # Child value estimates (full matrix, used in regret update)
        "child_value_sum",    # shape (n_att, n_def)
        "child_visit_count",  # shape (n_att, n_def)
        "node_visits",
        "children",
        "is_expanded",
    )

    def __init__(self, state, depth, parent=None):
        self.state  = state
        self.depth  = depth
        self.parent = parent

        self.att_names   = None
        self.def_names   = None
        self.att_actions = None
        self.def_actions = None
        self.n_att = 0
        self.n_def = 0

        self.att_regret = None
        self.def_regret = None
        self.att_avg    = None
        self.def_avg    = None

        self.child_value_sum   = None
        self.child_visit_count = None

        self.node_visits = 0
        self.children    = {}
        self.is_expanded = False

    def expand(self, n_nodes, neighbors_cache, sp_cache, min_actions_a,min_actions_d,closest_opponents,
               gamma_discount,
               flag_nodes, tag_radius, capture_radius, k,  # needed for immediate reward
               prior_weight: float = 1.0,
               use_sp_actions: bool = False):
        anames, dnames, att_acts, def_acts = rollout_policy_combined_better(
            self.state, n_nodes, neighbors_cache, sp_cache,flag_nodes,
            n_closest_defenders=closest_opponents,
            min_actions_a= min_actions_a,
            min_actions_d=min_actions_d)

        self.att_names = anames
        self.def_names = dnames
        self.att_actions = att_acts
        self.def_actions = def_acts
        self.n_att = len(att_acts)
        self.n_def = len(def_acts)

        self.att_regret = np.zeros(self.n_att)
        self.def_regret = np.zeros(self.n_def)
        self.att_avg = np.zeros(self.n_att)
        self.def_avg = np.zeros(self.n_def)

        self.child_value_sum = np.zeros((self.n_att, self.n_def))
        self.child_visit_count = np.zeros((self.n_att, self.n_def))

        # --- Convert to arrays (critical for speed) ---
        att_pos = np.array(att_acts, dtype=np.int32)   # (A, Na)
        def_pos = np.array(def_acts, dtype=np.int32)   # (D, Nd)
        flag_arr = np.array(list(flag_nodes), dtype=np.int32)
        A, Na = att_pos.shape
        D, Nd = def_pos.shape

        # --- Tag mask: any attacker within tag_radius of any defender ---
        # dist_ad[a, na, d, nd] via sp_cache lookup, then any() over (na, nd)
        att_flat = att_pos.reshape(-1)    # (A*Na,)
        def_flat = def_pos.reshape(-1)    # (D*Nd,)
        dist_ad_flat = np.array([[sp_cache[int(a)].get(int(d), np.inf) for d in def_flat] for a in att_flat],dtype=float)                  # (A*Na, D*Nd)
        dist_ad = dist_ad_flat.reshape(A, Na, D, Nd)
        captured = (dist_ad <= tag_radius).any(axis=(1, 3))  # (A, D)

        # --- Flag mask: any attacker within capture_radius of any flag ---
        dist_af_flat = np.array([[sp_cache[int(a)].get(int(f), np.inf) for f in flag_arr] for a in att_flat],dtype=float)                  # (A*Na, F)
        dist_af = dist_af_flat.reshape(A, Na, len(flag_arr))
        at_flag = (dist_af <= capture_radius).any(axis=(1, 2))[:, None]  # (A, 1)

        # --- Leaf values for ALL cells, zero out terminal ones after ---
        est_all = leaf_value_estimate_batch(att_pos, def_pos, n_nodes)  # (A, D)
        # est = np.where(captured | at_flag, 0.0, est_all)  # zero terminal cells

        # --- Immediate rewards: capture → -k, flag → +1, else → 0 ---
        imm = np.where(captured, float(-k),np.where(at_flag, 1.0, 0.0))               # (A, D)

        values = imm + gamma_discount * est_all
        self.child_value_sum = values * prior_weight
        self.child_visit_count = np.full((self.n_att, self.n_def), prior_weight)
        self.is_expanded = True

    # ------------------------------------------------------------------
    # Strategy derivation from cumulative regret  (Equation 4)
    # ------------------------------------------------------------------

    def _regret_strategy(self, regret: np.ndarray) -> np.ndarray:
        """σ_i(a) ∝ r_i(a)^+;  uniform if all regrets ≤ 0."""
        pos = np.maximum(0.0, regret)
        R   = pos.sum()
        if R > 0:
            return pos / R
        return np.ones(len(regret)) / len(regret)

    def _on_policy_mix(self, strategy: np.ndarray, gamma: float) -> np.ndarray:
        """Mix with uniform to ensure exploration (γ/|A| + (1-γ)σ_i(a))."""
        n    = len(strategy)
        dist = gamma / n + (1.0 - gamma) * strategy
        return dist / dist.sum()

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(self, gamma: float):
        """
        Sample (att_idx, def_idx) from the on-policy mixed strategies.
        Returns indices and the mixed probabilities (for regret update).
        """
        att_strat = self._regret_strategy(self.att_regret)
        def_strat = self._regret_strategy(self.def_regret)

        att_dist = self._on_policy_mix(att_strat, gamma)
        def_dist = self._on_policy_mix(def_strat, gamma)

        att_idx = int(np.random.choice(self.n_att, p=att_dist))
        def_idx = int(np.random.choice(self.n_def, p=def_dist))

        return att_idx, def_idx, att_strat, def_strat

    # ------------------------------------------------------------------
    # Child-mean estimate helper
    # ------------------------------------------------------------------

    def child_mean(self, ai: int, di: int) -> float:
        """
        Expected value estimate for joint action (ai, di).
        Falls back to 0 if the child has not been visited.
        """
        n = self.child_visit_count[ai, di]
        if n == 0:
            return 0.0
        return self.child_value_sum[ai, di] / n

    # ------------------------------------------------------------------
    # Update  (regret and average strategy)
    # ------------------------------------------------------------------

    def update(self, att_idx: int, def_idx: int,immediate:float,
               u1: float,gamma:float, att_strat: np.ndarray, def_strat: np.ndarray):
        """
        Update cumulative regret for both players and the average strategy.
        u2 = -u1 in zero-sum convention.
        """
        # --- Attacker regret update ---
        # Regret of not playing a'_1: reward(a'_1, a2) - u1
        for ai in range(self.n_att):
            if ai == att_idx:
                continue
            self.att_regret[ai] += immediate + gamma*self.child_mean(ai, def_idx) - u1
            # self.att_regret[ai] += u1 - self.child_mean(ai, def_idx)
        # Playing the chosen action (regret of switching away = 0 by definition)

        # --- Defender regret update  (defender minimises; u2 = -u1) ---
        for di in range(self.n_def):
            if di == def_idx:
                continue
            # Defender's regret: reward_for_def(a1, d') - u2
            #   = (-child_mean(att_idx, d')) - (-u1)
            #   = u1 - child_mean(att_idx, d')
            self.def_regret[di] += u1 - (immediate + gamma*self.child_mean(att_idx, di))
            # self.def_regret[di] += self.child_mean(att_idx, di) -u1

        # --- Average strategy accumulation ---
        self.att_avg += att_strat
        self.def_avg += def_strat

        self.node_visits += 1

    def record_child(self, ai: int, di: int, value: float):
        """Record child visit for the child-mean estimator."""
        self.child_value_sum[ai, di]   += value
        self.child_visit_count[ai, di] += 1

    # ------------------------------------------------------------------
    # Final move selection
    # ------------------------------------------------------------------

    def final_att_action(self) -> int:
        """Sample from the normalised average strategy σ̄_att."""
        s = self.att_avg.sum()
        if s == 0:
            return int(np.argmax(self.child_value_sum.max(axis=1)))
        dist = self.att_avg / s
        self.att_avg = dist
        return int(np.random.choice(self.n_att, p=dist))

    def final_def_action(self) -> int:
        """Sample from the normalised average strategy σ̄_def."""
        s = self.def_avg.sum()
        if s == 0:
            return int(np.argmin(self.child_value_sum.min(axis=0)))
        dist = self.def_avg / s
        self.def_avg = dist
        return int(np.random.choice(self.n_def, p=dist))


# ---------------------------------------------------------------------------
# Single simulation
# ---------------------------------------------------------------------------

def _simulate(node: RMNode, mamcts, gamma_discount: float,
              gamma_rm: float,
              closest_opponents: int, min_actions_a: int,min_actions_d: int) -> float:
    attacker_dict, defender_dict = node.state
    flag_nodes = mamcts.flag_nodes
    sp_cache   = mamcts.sp_cache
    n_nodes    = mamcts.n_nodes
    k          = mamcts.k

    surv_att, surv_def, captured, _,flags_reached,candidate_flags_reached = filter_active_agents(
        attacker_dict, defender_dict, flag_nodes, mamcts.candidate_flag_nodes, sp_cache,
        mamcts.tag_radius, mamcts.capture_radius)
    immediate = len(flags_reached)-k*len(captured) +0.1*len(candidate_flags_reached)

    if (not surv_att) or (not surv_def):
        return immediate

    node.state = (surv_att, surv_def)

    if not node.is_expanded:
        # --- CHANGED: pass gamma_discount, for_attacker, use_2v1 --------
        node.expand(
            n_nodes, mamcts.neighbors_cache, sp_cache, min_actions_a,min_actions_d,closest_opponents,
            gamma_discount=gamma_discount,
            flag_nodes=flag_nodes,
            tag_radius=mamcts.tag_radius,
            capture_radius=mamcts.capture_radius,
            k=k,
            prior_weight=1.0)  # <-- tune: higher = trust prior mo,)
    # After expansion M is fully populated — selection is immediately valid
    # -----------------------------------------------------------------

    att_idx, def_idx, att_strat, def_strat = node.select(gamma_rm)

    if node.depth == 1:
        prior_val = node.child_mean(att_idx, def_idx)
        discounted = immediate + gamma_discount * prior_val

        # node.record_child(att_idx, def_idx,prior_val)
        node.update(att_idx, def_idx,immediate,discounted,gamma_discount ,att_strat, def_strat)
        return discounted

    att_move = node.att_actions[att_idx]
    def_move = node.def_actions[def_idx]

    new_att   = {n: p for n, p in zip(node.att_names, att_move)}
    new_def   = {n: p for n, p in zip(node.def_names, def_move)}
    new_state = (new_att, new_def)

    child_key = (att_idx, def_idx)
    if child_key not in node.children:
        child = RMNode(new_state, depth=node.depth - 1, parent=node)
        node.children[child_key] = child
    else:
        child = node.children[child_key]
        child.state = new_state

    value = _simulate(child, mamcts, gamma_discount, gamma_rm, closest_opponents, min_actions_a, min_actions_d)
    discounted = immediate + gamma_discount * value

    # record_child now *adds* to the leaf prior, diluting it over time
    #node.record_child(att_idx, def_idx, discounted)
    node.record_child(att_idx, def_idx, value)
    node.update(att_idx, def_idx,immediate,discounted,gamma_discount ,att_strat, def_strat)

    return discounted

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_regret_matching(attacker: dict, defender: dict, mamcts,
                         n_simulations: int = 200,
                         depth: int = 2,
                         gamma_rm: float = 0.2,
                         gamma_discount: float = 0.95,
                         closest_opponents: int = 3,
                         min_actions_a: int = 1,
                         min_actions_d: int =1) -> tuple:
    """
    Run Regret Matching simultaneous-move MCTS.

    Parameters
    ----------
    attacker, defender : dict  {name: graph_node_id}
    mamcts             : MultiAgentMCTS
    n_simulations      : number of SM-MCTS iterations
    depth              : lookahead depth
    gamma_rm           : exploration / on-policy mixing coefficient γ
                         (recommended range 0.1 – 0.5; tune per setting)
    gamma_discount     : MDP discount factor
    closest_opponents  : nearest opponents considered per agent in rollout
    min_actions        : minimum candidate moves per agent

    Returns
    -------
    best_att  : dict  recommended attacker positions
    best_def  : dict  recommended defender positions
    root      : RMNode  root node (full regret / average-strategy tables)
    """
    root_state = (attacker.copy(), defender.copy())
    root = RMNode(root_state, depth=depth)
    #print("root-state",root_state)
    #print(root.__slots__)
    if len(attacker) + len(defender) >=10:
        min_actions_d = 1

    for _ in range(n_simulations):
        _simulate(root, mamcts, gamma_discount, gamma_rm, closest_opponents, min_actions_a,min_actions_d)

    if not root.is_expanded:
        return attacker.copy(), defender.copy(), root

    att_idx = root.final_att_action()
    def_idx = root.final_def_action()

    best_att = {n: p for n, p in zip(root.att_names, root.att_actions[att_idx])}
    best_def = {n: p for n, p in zip(root.def_names, root.def_actions[def_idx])}
    return best_att, best_def, root

#######################################################################################
################################ STRATEGY FUNCTION ###################################
#######################################################################################

def strategy(state: dict) -> str:
    global att_policy_1v1_defender, def_policy_1v1_defender, V_1v1_defender, mamcts_defender, last_time_defender
    from typing import Dict, List, Tuple, Any
    import networkx as nx
    import random

    # ============================ HELPERS ============================

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

    # ===== AGENT CONTROLLER =====
    # DO NOT directly call agent_ctrl methods unless you understand the library
    agent_ctrl = state["agent_controller"]  # Wrapper containing agent state and methods
    current_pos: int = state["curr_pos"]  # Current node ID where agent is located
    current_time: int = state["time"]  # Current game timestep
    team: str = agent_ctrl.team  # Team identifier ('red' or 'blue')
    red_payoff: float = state["payoff"]["red"]  # Red team accumulated score
    blue_payoff: float = state["payoff"]["blue"]  # Blue team accumulated score

    # Agent parameters
    speed: float = agent_ctrl.speed  # Movement speed (max nodes per turn)
    tagging_radius: float = agent_ctrl.tagging_radius  # Distance to tag attackers
    # print('tagging radius',tagging_radius)
    # ===== RULE CONFIG =====
    rule_config = state["rule_config"]  # Read-only view of red_global, blue_global, environment
    # Opponent (red/attacker) parameters
    opp_capture_radius: float = rule_config["red_global"]["capture_radius"]  # flag capture range
    opp_sensing_radius: float = rule_config["red_global"]["sensing_radius"]  # red vision radius
    # Environment (stationary sensor network)
    stationary_radius: float = rule_config["environment"]["blue_stationary_sensor_radius"]
    stationary_positions: list = rule_config["environment"]["blue_static_sensor_positions"]

    # ===== INDIVIDUAL CACHE =====
    cache = agent_ctrl.cache  # Per-agent storage, not shared with teammates

    example_val = cache.get("example_key", None)  # get a value (returns default if missing)
    cache.set("example_key", example_val)  # set a value
    cache.update(example_a=0, example_b=1)  # set multiple values at once

    # ===== TEAM CACHE (SHARED) =====
    example_shared = agent_ctrl.get_team("example_key", 0)  # get a shared team value
    agent_ctrl.set_team("example_key", current_time)  # set a shared team value
    agent_ctrl.update_team(example_a=0, example_b="val")  # set multiple shared values at once

    # ===== SENSORS =====
    # All sensor data is read here once; variables are reused in map updates and decision logic below.
    sensors: Dict[str, Tuple[Any, Dict[str, Any]]] = state["sensor"]  # All sensor data

    real_flags: List[int] = agent_ctrl.sensor_data(state, "flag")[
        "real_flags"] if "flag" in sensors else []  # True flag locations
    fake_flags: List[int] = agent_ctrl.sensor_data(state, "flag")[
        "fake_flags"] if "flag" in sensors else []  # Fake flag locations

    teammates_sensor: Dict[str, int] = agent_ctrl.sensor_data(state,
                                                              "custom_team") if "custom_team" in sensors else {}  # Teammates only {name: node_id}

    nearby_enemies: Dict[str, int] = agent_ctrl.sensor_data(state, "egocentric_agent")[
        "enemies"] if "egocentric_agent" in sensors else {}  # Enemy agents within sensing radius
    nearby_teammates: Dict[str, int] = agent_ctrl.sensor_data(state, "egocentric_agent")[
        "teammates"] if "egocentric_agent" in sensors else {}  # Teammates within sensing radius

    visible_nodes: Dict[int, Any] = agent_ctrl.sensor_data(state, "egocentric_agent_region")[
        "nodes"] if "egocentric_agent_region" in sensors else {}  # Nodes within sensing radius
    visible_edges: List[Any] = agent_ctrl.sensor_data(state, "egocentric_agent_region")[
        "edges"] if "egocentric_agent_region" in sensors else []  # Edges within sensing radius

    # Stationary sensors — pre-consolidated by game engine and pre-filtered by team
    # {"enemies": {name: node_id}, "teammates": {name: node_id}, "detections": [per-sensor entries]}
    # Each per-sensor entry: {"fixed_position": int, "detected_agents": {name: {node_id, distance}}, "agent_count": int, "covered_nodes": [...]}
    stationary: Dict[str, Any] = agent_ctrl.sensor_data(state, "stationary") or {"enemies": {}, "teammates": {},
                                                                                 "detections": []}
    stationary_enemies: Dict[str, int] = stationary[
        "enemies"]  # {name: node_id} of attackers seen by any stationary sensor
    stationary_detections: List[Dict[str, Any]] = stationary["detections"]  # raw per-sensor entries

    # ===== AGENT MAP (SHARED) =====
    agent_map = agent_ctrl.map  # Team-shared map with positions and graph
    global_map_payload: Dict[str, Any] = agent_ctrl.sensor_data(state, "global_map")
    global_map_sensor: nx.Graph = global_map_payload["graph"]  # Full graph topology from sensor
    global_map_apsp = global_map_payload.get("apsp")

    # The graph is static — attach it once on the first turn and reuse every turn after.
    # agent_map is shared across teammates, so only the first agent to run pays this cost.
    if agent_map.graph is None:
        nodes_data: Dict[int, Dict[str, Any]] = {node_id: global_map_sensor.nodes[node_id] for node_id in
                                                 global_map_sensor.nodes()}
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

    # Update teammate positions from custom_team sensor
    agent_map.update_team_agents(team, teammates_sensor, current_time)

    # Update enemy positions from egocentric_agent sensor
    agent_map.update_team_agents(agent_ctrl.enemy_team, nearby_enemies, current_time)

    # How to get all positions of a team from agent map
    teammates_data: List[Tuple[str, int, int]] = agent_map.get_team_agents(team)  # [(name, pos, age)] of teammates
    # How to get a specific agent's position from agent map
    enemy_pos, enemy_age = agent_map.get_agent_position(agent_ctrl.enemy_team,
                                                        "red_0")  # (position, age_in_timesteps) or (None, None)

    #  ================================== INIT (t==1) ===============================================
    time = current_time
    current_position = current_pos

    if time == 1:
        ap, dp, value = get_or_solve_policy(global_map_sensor, real_flags, 1, 2)
        att_policy_1v1_defender = ap
        def_policy_1v1_defender = dp
        V_1v1_defender = value
        mamcts_defender = MultiAgentMCTS(graph=global_map_sensor,flag_nodes=real_flags, candidate_flag_nodes=real_flags + fake_flags, k=0.5,tag_radius=1,capture_radius=2)

    # ======================================= grouping all the detected attackers from the stationary sensors ===================================
    agent_ctrl.set_team("enemy_detected", stationary_enemies)

    # ===== DECISION LOGIC =====
    # Merge both detection sources: per-agent egocentric sensor + stationary sensor network
    attackers_detected_at_time_t = {**nearby_enemies, **stationary_enemies}  # {name: node_id}

    # ================================ AGGREGATE DETECTIONS THIS TIMESTEP ===============================
    if agent_ctrl.get_team("atk_detected_cache_time", -1) != current_time:
        agent_ctrl.set_team("atk_detected_per_agent_at_time_t", {})
        agent_ctrl.set_team("atk_detected_cache_time", current_time)

    per_agent = agent_ctrl.get_team("atk_detected_per_agent_at_time_t", {})
    per_agent[agent_ctrl.name] = (attackers_detected_at_time_t.copy(), current_time, agent_ctrl.name)
    agent_ctrl.set_team("atk_detected_per_agent_at_time_t", per_agent)

    all_attackers_detected_now: Dict[str, int] = {}
    for atk_data, t, _ in agent_ctrl.get_team("atk_detected_per_agent_at_time_t", {}).values():
        if t == current_time:
            all_attackers_detected_now.update(atk_data)

    previous_attackers_used = agent_ctrl.get_team("attackers_used_in_last_mcts", set())
    new_attackers_detected = not set(all_attackers_detected_now.keys()).issubset(previous_attackers_used)

    defenders: Dict[str, int] = dict(teammates_sensor) if teammates_sensor else {}
    defenders.update({agent_ctrl.name: current_position})
    num_def = len(defenders)
    num_atk = len(all_attackers_detected_now)

    FEW_ATK_RATIO = 0.5
    few_attackers = (num_atk > 0) and (num_atk < FEW_ATK_RATIO * max(1, num_def))

    # 1) Enough attackers detected => full MCTS
    if num_atk > 0 and not few_attackers:
        try:
            if agent_ctrl.get_team("def_action", None) is None or last_time_defender != current_time or new_attackers_detected:
                _,def_action,_= run_regret_matching(
                    attacker=all_attackers_detected_now,
                    defender = defenders,
                    mamcts=mamcts_defender,
                    n_simulations = 2000,
                    depth= 2,
                    gamma_rm = 0.4,
                    gamma_discount = 0.95,
                    closest_opponents = 3,
                    min_actions_a= 2,
                    min_actions_d =2)
                agent_ctrl.set_team("def_action", def_action)
                agent_ctrl.set_team("attackers_used_in_last_mcts", set(all_attackers_detected_now.keys()))
                last_time_defender = current_time

            state["action"] = agent_ctrl.get_team("def_action", {}).get(agent_ctrl.name, None)
            return 'goodbye'
        except Exception as e:
            print(f"MCTS failed: {e}, using coverage fallback")

    available_actions = list(agent_map.graph.neighbors(current_pos))
    sp = mamcts_defender.sp_cache

    # 2) Few attackers => hybrid chase + coverage
    if few_attackers:
        atk_nodes = list(all_attackers_detected_now.values())
        chase_team = choose_chase_team(defenders, atk_nodes, sp_cache=sp, chase_count=min(num_def, 2 * num_atk))
        chase_defenders = {name: defenders[name] for name in chase_team}
        left_defenders = {name: defenders[name] for name in defenders if name not in chase_team}
        flag_assignment = assign_defenders_to_flags(left_defenders, real_flags, sp_cache=sp, load_penalty=3.0)

        if agent_ctrl.name in chase_team:
            try:
                _, def_action, _ = run_regret_matching(
                    attacker=all_attackers_detected_now,
                    defender=chase_defenders,
                    mamcts=mamcts_defender,
                    n_simulations=2000,
                    depth=2,
                    gamma_rm=0.4,
                    gamma_discount=0.95,
                    closest_opponents=3,
                    min_actions_a=2,
                    min_actions_d=2)
                state["action"] = def_action.get(agent_ctrl.name, None)
                return 'goodbye'
            except Exception as e:
                state["action"] = random.choice(available_actions) if available_actions else current_pos
                return 'goodbye'

        my_flag = flag_assignment.get(agent_ctrl.name, None)
        state["action"] = step_toward_target(current_pos, my_flag, available_actions, sp)
        return 'goodbye'

    # 3) No attackers detected => coverage mode
    flag_assignment = assign_defenders_to_flags(defenders, real_flags, sp_cache=sp, load_penalty=3.0)
    my_flag = flag_assignment.get(agent_ctrl.name, None)
    if my_flag is None:
        state["action"] = random.choice(available_actions) if available_actions else current_pos
    else:
        state["action"] = step_toward_target(current_pos, my_flag, available_actions, sp)

    return 'goodbye'


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


#######################################################################################
################################ BASEGAME TEMPLATE ###################################
#######################################################################################

class BaseGame:
    def __init__(self, n_states: Union[int],
                 n_actions_1: Union[List[int], int], n_actions_2: Union[List[int], int], gamma: float = 0.95):
        """
        The base game class used by the Nash solver.
        :param n_states: int, number of (joint) states of the game
        :param n_actions_1: int or List[int], number of actions for player 1 (in each state)
        :param n_actions_2: int or List[int], number of actions for player 2 (in each state)
        :param gamma: float, discount factor in (0,1)
        """
        self._n_states = n_states
        self._n_actions_1 = n_actions_1
        self._n_actions_2 = n_actions_2
        self.gamma = gamma

        self._rewards = None
        self._terminal_states = None
        self._transitions = None
        self._transitions_to_state, self._transitions_prob = None, None
        self._padded_transitions_to_state, self._padded_transitions_prob = None, None

    def set_rewards(self, rewards: np.ndarray):
        """
        Set the rewards of the game. Either r(s) or r(s, a1, a2)
        :param rewards: np.ndarray, either 1d or 3d.
               if 1d, rewards[s] gives the reward of state s,
               if 3d, rewards[s, a1, a2] gives the reward of state s with action a1 and a2
        :return: None
        """
        assert len(rewards.shape) == 1 or len(rewards.shape) == 3, "rewards should be 1d or 3d"
        if len(rewards.shape) == 1:
            rewards = rewards.reshape((self._n_states, 1, 1))
        self._rewards = rewards

    def set_terminal_states(self, terminal_states: List):
        """
        Set the terminal states of the game
        :param terminal_states: List[int], list of terminal states
        :return: None
        """
        self._terminal_states = deepcopy(terminal_states)

    def set_transitions(self, transitions: List[List[np.ndarray]]):
        """
        Set the transitions of the game.
        If action space is state dependent, one should use max number of actions as the size of the lists.
        :param transitions: List[List[np.ndarray]],
               transitions[a1][a2] gives the n_states x n_states transition matrix with row sum to 1
        :return: None
        """
        for a1 in range(self.get_max_n_action1()):
            for a2 in range(self.get_max_n_action2()):
                assert transitions[a1][a2].shape == (self._n_states, self._n_states), "transitions shape mismatch"
                assert transitions[a1][a2].sum(axis=1).all() == 1, "transitions should sum to 1"
        self._transitions = transitions

    def set_compressed_transitions(self,
                                   transition_to_state: List[List[List[List[int]]]],
                                   transitions_prob: List[List[List[List[int]]]]):
        """
        Set the compressed transitions of the game
        :param transition_to_state: transition_to_state[s][a1][a2] gives the state indices that the joint state will transition to
        :param transitions_prob: transitions_prob[s][a1][a2] gives the probability of transitioning to the corresponding state
        :return: None
        """
        assert len(transition_to_state) == self._n_states and len(transitions_prob) == self._n_states, \
            ("Compressed transitions size error: "
             "use transitions_prob[s][a1][a2] to store the probabilities of transitioning to the corresponding state.")
        self._transitions_to_state = deepcopy(transition_to_state)
        self._transitions_prob = deepcopy(transitions_prob)

        self._generate_padded_compressed_transitions()

    @property
    def has_compressed_transition(self):
        return self._transitions_to_state is not None and self._transitions_prob is not None

    def get_transitions_s(self, s: int, a1: int, a2: int) -> np.ndarray:
        """
        Get the transition probability from state s with action a1 and a2
        :param s: state
        :param a1: action 1
        :param a2: action 2
        :return: p_s_prime[s_prime] = P(s_prime | s, a1, a2)
        """
        return self._transitions[a1][a2][s, :]

    def get_compressed_transitions_s(self, s: int, a1: int, a2: int) -> Tuple[List[int], List[np.ndarray]]:
        """
        Get the compressed representation of the transition matrix from state s with action a1 and a2
        :param s: state
        :param a1: action 1
        :param a2: action 2
        :return:
            to_states: a list of state indices that the joint state will transition to
            to_states_prob: a 1d array of probabilities of transitioning to the corresponding state
        """
        assert self.has_compressed_transition, "Compressed transitions not set for the game!"
        to_states = self._transitions_to_state[s][a1][a2]
        to_states_prob = self._transitions_prob[s][a1][a2]
        return to_states, to_states_prob

    def get_all_compressed_transitions_s(self, s: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get compressed transitions from state s for all actions
        :param s: state
        :return:
            to_states: ndarray with to_states[a1, a2] gives the state indices that the joint state will transition to
            to_states_prob: ndarray with to_states_prob[a1, a2] gives the probability of transitioning
        """
        assert self.has_compressed_transition, "Compressed transitions not set for the game!"
        to_states = np.array(self._transitions_to_state[s]).astype(int)
        to_states_prob = np.array(self._transitions_prob[s])
        return to_states, to_states_prob

    def get_all_transitions_s(self, s: int) -> np.ndarray:
        """
        Get transitions from state s for all actions
        :param s: state
        :return: transitions [a1, a2] gives the transition vector that sum to 1
        """
        trans_prob = np.array(self._transitions)[:, :, s, :]
        return trans_prob

    def get_all_transitions(self) -> np.ndarray:
        """
        Get transitions for all states
        :return: transitions [s, a1, a2] gives the transition vector that sum to 1
        """
        return np.array(self._transitions).transpose((3, 0, 1, 2))

    def get_padded_transitions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the padded transitions for all states size with (S, MAX_A1, MAX_A2, MAX_NEXT_STATE)
        :return:
            to_states: ndarray with to_states[s, a1, a2] gives the state indices that the joint state will transition, padded with -1
            to_states_prob: ndarray with to_states_prob[s, a1, a2] gives the probability of transitioning, padded with -1
        """
        return self._padded_transitions_to_state, self._padded_transitions_prob

    def get_rewards(self, s, a1=None, a2=None) -> float:
        """
        Get the reward of the game
        :param s: state
        :param a1: action 1
        :param a2: action 2
        :return: reward, either state dependent or state-action dependent
        """
        if len(self._rewards.shape) == 3:
            assert a1 is not None and a2 is not None
            return float(self._rewards[s, a1, a2])
        else:
            return float(self._rewards[s])

    def get_all_rewards_s(self, s) -> Union[np.ndarray, float]:
        if len(self._rewards.shape) == 3:
            return self._rewards[s, :, :]
        else:
            return self._rewards[s]

    def get_all_rewards(self) -> Union[np.ndarray, float]:
        """
        Get rewards at state s for all actions
        :return: SXA1XA2 ndarray with rewards R[s, a1, a2] gives the reward of state s with action a1 and a2
            or   S ndarray if reward is not action dependent
        """
        return self._rewards


    def is_terminal(self, state: int) -> bool:
        if self._terminal_states is not None:
            return state in self._terminal_states
        else:
            return False

    def get_n_states(self) -> int:
        return self._n_states

    def get_n_action1(self, state) -> int:
        if isinstance(self._n_actions_1, int):
            return self._n_actions_1
        else:  # state dependent action set
            return self._n_actions_1[state]

    def get_n_action2(self, state) -> int:
        if isinstance(self._n_actions_2, int):
            return self._n_actions_2
        else:  # state dependent action set
            return self._n_actions_2[state]

    def get_max_n_action1(self) -> int:
        if isinstance(self._n_actions_1, int):
            return self._n_actions_1
        else:
            return max(self._n_actions_1)

    def get_max_n_action2(self) -> int:
        if isinstance(self._n_actions_2, int):
            return self._n_actions_2
        else:
            return max(self._n_actions_2)

    def _generate_padded_compressed_transitions(self):
        n_states = self._n_states
        n_a1_max = self.get_max_n_action1()
        n_a2_max = self.get_max_n_action2()
        #n_next_state_max = max([len(self._transitions_to_state[s][a1][a2]) for s in range(n_states)for a1 in range(self.get_n_action1(s)) for a2 in range(self.get_n_action2(s))])
        n_next_state_max = 1
        self._padded_transitions_to_state = np.full((n_states, n_a1_max, n_a2_max, n_next_state_max), 0, dtype=int)
        self._padded_transitions_prob = np.full((n_states, n_a1_max, n_a2_max, n_next_state_max), 0, dtype=float)
        for s in range(n_states):
            n_a1, n_a2 = self.get_n_action1(s), self.get_n_action2(s)
            for a1 in range(n_a1):
                for a2 in range(n_a2):
                    to_states, to_states_prob = self.get_compressed_transitions_s(s, a1, a2)
                    #n_next = len(to_states)
                    n_next = 1
                    self._padded_transitions_to_state[s, a1, a2, :n_next] = to_states
                    self._padded_transitions_prob[s, a1, a2, :n_next] = to_states_prob

################################################################################
############################# GAME GENERATION FOR THE TAD GAME #################
################################################################################

class TADGamePrimitives(BaseGame):
    """
    Unified game class.
    Supported primitives: 1v1, 1v2, 2v1, 2v2

    """

    def __init__(self, graph, goal_positions,
                 num_attackers=2, num_defenders=2,
                 capture_radius=2, tag_radius=1, k=0.5):
        if (num_attackers, num_defenders) not in {(1, 1), (1, 2), (2, 1), (2, 2)}:
            raise ValueError("Supported primitives are 1v1, 1v2, 2v1, and 2v2.")

        # ---- config ----
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.num_nodes = len(self.nodes)
        self.goal_positions = sorted(map(int, goal_positions))
        self.num_attackers = int(num_attackers)
        self.num_defenders = int(num_defenders)
        self.capture_radius = int(capture_radius)
        self.tag_radius = int(tag_radius)
        self.k = k

        # INACTIVE placeholder (index = num_nodes)
        self.INACTIVE = self.num_nodes

        # ---- actions (neighbors + stay); INACTIVE has only stay ----
        self.actions_set = {}
        for i in range(self.num_nodes):
            nbrs = sorted(self.graph.neighbors(i))
            if i in nbrs:  # defensive
                nbrs.remove(i)
            self.actions_set[i] = nbrs + [i]
        # INACTIVE can only stay at INACTIVE
        self.actions_set[self.INACTIVE] = [self.INACTIVE]

        # ---- distances on physical graph (no INACTIVE in metric) ----
        self._dist = np.full((self.num_nodes, self.num_nodes), np.inf, dtype=float)
        for i in range(self.num_nodes):
            self._dist[i, i] = 0.0
        for src, dmap in nx.all_pairs_shortest_path_length(self.graph):
            for dst, d in dmap.items():
                self._dist[src, dst] = float(d)

        # min distance to any goal; INACTIVE gets +inf
        if len(self.goal_positions) == 0:
            base_min_goal = np.full(self.num_nodes, np.inf, dtype=float)
        else:
            cols = np.array(self.goal_positions, dtype=int)
            base_min_goal = np.min(self._dist[:, cols], axis=1)
        # Add INACTIVE distance (always +inf)
        self.min_dist_to_goal = np.concatenate([base_min_goal, [np.inf]])

        # ---- state space size (base + absorbing terminal) ----
        base_states = self._calc_base_state_space()
        self.TERMINAL_STATE = base_states
        self._n_states = base_states + 1  # +1 absorbing state

        # per-state action counts (attackers = player 1, defenders = player 2)
        self._n_actions_1, self._n_actions_2 = self._calc_action_spaces()
        self.max_attacker_actions = max(self._n_actions_1)
        self.max_defender_actions = max(self._n_actions_2)

        # init BaseGame and allocate tensors
        super().__init__(self._n_states, self._n_actions_1, self._n_actions_2)
        self._transitions_to_state = np.zeros(
            (self._n_states, self.max_attacker_actions, self.max_defender_actions),
            dtype=np.int64)
        self._transitions_prob = np.zeros_like(self._transitions_to_state, dtype=float)
        self._transitions_reward = np.zeros_like(self._transitions_to_state, dtype=float)

        print(f"Initialized primitive {self.num_attackers}v{self.num_defenders} "
              f"with {self._n_states} states (including absorbing).")

    # =========================
    #   State encoding utils
    # =========================
    @staticmethod
    def _tri(n: int) -> int:
        """Triangular number counting unordered pairs (i<=j) from 0..n-1."""
        return n * (n + 1) // 2

    @staticmethod
    def _encode_pair(i: int, j: int) -> int:
        if j < i:
            i, j = j, i
        return (j * (j + 1)) // 2 + (j - i)

    @staticmethod
    def _decode_pair(c: int) -> tuple:
        n = int((np.sqrt(1 + 8 * c) - 1) // 2)
        r = c - (n * (n + 1)) // 2
        i1 = n - r
        i2 = n
        return i1, i2

    def _calc_base_state_space(self) -> int:
        if self.num_defenders == 1:
            n_def = self.num_nodes
        else:
            # 2 defenders → include INACTIVE in the triangular encoding only when we have 2 attackers
            n_def = self._tri(self.num_nodes if self.num_attackers == 1 else self.num_nodes + 1)

        if self.num_attackers == 1:
            n_att = self.num_nodes
        else:
            # 2 attackers → include INACTIVE in the triangular encoding
            n_att = self._tri(self.num_nodes + 1)

        return n_def * n_att

    def state_to_positions(self, s: int) -> tuple:
        """Decode global state index to (defenders..., attackers...)"""
        if s == self.TERMINAL_STATE:
            # Return harmless zeros of right arity (not used for transitions anyway).
            if (self.num_attackers, self.num_defenders) == (1, 1):
                return (0, 0)
            if (self.num_attackers, self.num_defenders) == (1, 2):
                return (0, 0, 0)
            if (self.num_attackers, self.num_defenders) == (2, 1):
                return (0, 0, 0)
            # 2v2
            return (0, 0, 0, 0)

        # split into defender/attacker components with correct radix
        if self.num_attackers == 1:
            nA = self.num_nodes  # linear radix
        else:
            nA = self._tri(self.num_nodes + 1)  # triangular radix with INACTIVE

        a_comp = s % nA
        d_comp = s // nA

        # defenders
        if self.num_defenders == 1:
            defenders = [d_comp]
        else:
            defenders = list(self._decode_pair(d_comp))

        # attackers
        if self.num_attackers == 1:
            attackers = [a_comp]
        else:
            attackers = list(self._decode_pair(a_comp))  # elements in 0..N (N==INACTIVE)

        return tuple(defenders + attackers)

    def positions_to_state(self, *positions: int) -> int:
        """Encode (defenders..., attackers...) → state index (non-absorbing)."""
        # defenders
        if self.num_defenders == 1:
            d_comp = positions[0]
        else:
            d_comp = self._encode_pair(positions[0], positions[1])

        # attackers
        if self.num_attackers == 1:
            a_comp = positions[-1]
        else:
            a_comp = self._encode_pair(positions[-2], positions[-1])  # values can be INACTIVE

        # attacker radix
        if self.num_attackers == 1:
            nA = self.num_nodes
        else:
            nA = self._tri(self.num_nodes + 1)

        return d_comp * nA + a_comp

    # =========================
    #   Action spaces
    # =========================
    def _calc_action_spaces(self):
        nA_list, nD_list = [], []
        for s in range(self._n_states):
            if s == self.TERMINAL_STATE:
                nA_list.append(1)
                nD_list.append(1)
                continue

            pos = self.state_to_positions(s)

            # attackers (player 1)
            if self.num_attackers == 1:
                a = pos[-1]
                nA_list.append(len(self.actions_set[a]))
            else:
                a1, a2 = pos[-2], pos[-1]
                nA_list.append(len(self.actions_set[a1]) * len(self.actions_set[a2]))

            # defenders (player 2)
            if self.num_defenders == 1:
                d = pos[0]
                nD_list.append(len(self.actions_set[d]))
            else:
                d1, d2 = pos[0], pos[1]
                nD_list.append(len(self.actions_set[d1]) * len(self.actions_set[d2]))

        return nA_list, nD_list

    # =========================
    #   Helpers
    # =========================
    def _within_radius(self, i: int, j: int, radius: int) -> bool:
        """Distance check (returns False if either index is INACTIVE)."""
        if i == self.INACTIVE or j == self.INACTIVE:
            return False
        return self._dist[i, j] <= radius

    def _attacker_captures_goal(self, idx: int) -> bool:
        """Goal capture (False for INACTIVE)."""
        if idx == self.INACTIVE:
            return False
        return self.min_dist_to_goal[idx] <= self.capture_radius

    def _valid_super_defender_actions(self, s: int):
        """Return list of valid flat defender action indices."""
        if s == self.TERMINAL_STATE:
            return [0]
        return list(range(self._n_actions_2[s]))

    def _valid_super_attacker_actions(self, s: int):
        """Return list of valid flat attacker action indices."""
        if s == self.TERMINAL_STATE:
            return [0]
        return list(range(self._n_actions_1[s]))

    def _decompose_defender_action(self, s: int, d_idx: int):
        """Convert flat defender action index to individual actions."""
        if self.num_defenders == 1:
            return d_idx
        d1, d2 = self.state_to_positions(s)[:2]
        n1 = len(self.actions_set[d1])
        n2 = len(self.actions_set[d2])
        # flat = i*n2 + j  ->  (i, j)
        return (d_idx // n2, d_idx % n2)

    def _decompose_attacker_action(self, s: int, a_idx: int):
        """Convert flat attacker action index to individual actions."""
        if self.num_attackers == 1:
            return a_idx
        a1, a2 = self.state_to_positions(s)[-2:]
        n1 = len(self.actions_set[a1])
        n2 = len(self.actions_set[a2])
        # flat = i*n2 + j  ->  (i, j)
        return (a_idx // n2, a_idx % n2)

    # =========================
    #   Transitions: fill T,P,R
    # =========================
    def generate_compressed_transitions(self):
        """Populate _transitions_to_state, _transitions_prob, _transitions_reward."""
        # Absorbing terminal: self-loop, reward 0
        sT = self.TERMINAL_STATE
        self._transitions_to_state[sT, :, :] = sT
        self._transitions_prob[sT, :, :] = 1.0
        self._transitions_reward[sT, :, :] = 0.0

        # Physical states
        for s in tqdm(range(self._n_states - 1)):
            pos = self.state_to_positions(s)
            defenders = list(pos[:self.num_defenders])
            attackers = list(pos[self.num_defenders:])

            n_def_actions = self._n_actions_2[s]
            n_att_actions = self._n_actions_1[s]

            for a_idx in range(n_att_actions):
                for d_idx in range(n_def_actions):
                    # Decompose flat action indices
                    a_act = self._decompose_attacker_action(s, a_idx)
                    d_act = self._decompose_defender_action(s, d_idx)

                    ns, r = self._one_step_transition(defenders, attackers, d_act, a_act)
                    self._transitions_to_state[s, a_idx, d_idx] = ns
                    self._transitions_prob[s, a_idx, d_idx] = 1.0
                    self._transitions_reward[s, a_idx, d_idx] = r

        # hand off transitions (BaseGame expects to_state + prob)
        self.set_compressed_transitions(self._transitions_to_state, self._transitions_prob)
        self.set_rewards(self._transitions_reward)
        # optional: store a per-state average immediate reward (not required if solver uses R(s,a1,a2))
        # self._rewards = np.sum(self._transitions_reward, axis=(1, 2))
        return self._transitions_to_state, self._transitions_reward

    def _one_step_transition(self, defenders, attackers, d_act, a_act):
        """Deterministic next state + immediate reward given joint actions.

        d_act: int (for 1 defender) or tuple(int, int) (for 2 defenders)
        a_act: int (for 1 attacker) or tuple(int, int) (for 2 attackers)
        """
        # 1) apply actions
        # defenders
        if self.num_defenders == 1:
            next_def = [self.actions_set[defenders[0]][d_act]]
        else:
            next_def = [self.actions_set[defenders[0]][d_act[0]],
                        self.actions_set[defenders[1]][d_act[1]]]

        # attackers
        if self.num_attackers == 1:
            next_att = [self.actions_set[attackers[0]][a_act]]
        else:
            next_att = [self.actions_set[attackers[0]][a_act[0]],
                        self.actions_set[attackers[1]][a_act[1]]]

        # 2) event checks + rewards
        reward = 0.0

        # --- 1v1: single attacker, single defender ---
        if self.num_attackers == 1 and self.num_defenders == 1:
            a = next_att[0]
            # Tagging is preferred over capturing
            if self._within_radius(a, next_def[0], self.tag_radius):
                return self.TERMINAL_STATE, -self.k
            if self._attacker_captures_goal(a):
                return self.TERMINAL_STATE, +1.0
            return self.positions_to_state(*(next_def + next_att)), reward

        # --- 1v2: single attacker, two defenders ---
        if self.num_attackers == 1 and self.num_defenders == 2:
            a = next_att[0]
            # Tagging is preferred over capturing (any defender can tag)
            if any(self._within_radius(a, d, self.tag_radius) for d in next_def):
                return self.TERMINAL_STATE, -self.k
            if self._attacker_captures_goal(a):
                return self.TERMINAL_STATE, +1.0
            # Sort defenders for canonical encoding
            next_def.sort()
            return self.positions_to_state(*(next_def + next_att)), reward

        # --- 2v1 or 2v2: two attackers ---
        attacker_alive = [True, True]
        defender_alive = [True] * self.num_defenders

        # Process each attacker
        for att_idx in range(2):
            a_i = next_att[att_idx]
            # Skip INACTIVE attackers - they were already eliminated in previous step
            if a_i == self.INACTIVE:
                attacker_alive[att_idx] = False
                continue

            # Check tagging FIRST (tagging is preferred over capturing)
            # Only active (non-INACTIVE) defenders who are still alive can tag
            tagged = False
            for def_idx in range(self.num_defenders):
                # Skip INACTIVE defenders or defenders already eliminated this timestep
                if next_def[def_idx] == self.INACTIVE or not defender_alive[def_idx]:
                    continue

                if self._within_radius(a_i, next_def[def_idx], self.tag_radius):
                    reward -= self.k  # Tagged (trade)
                    attacker_alive[att_idx] = False
                    defender_alive[def_idx] = False  # Defender trades their life
                    tagged = True
                    break  # This attacker is tagged, move to next attacker

            # Check capture only if not tagged
            if not tagged and self._attacker_captures_goal(a_i):
                reward += 1.0  # Successful capture
                attacker_alive[att_idx] = False

        # Determine next state based on who's alive

        # 1) If defenders are exhausted this step -> terminal +1 (attackers win)
        if self.num_defenders == 1 and not defender_alive[0]:
            return self.TERMINAL_STATE, +1.0 - self.k
        if self.num_defenders == 2 and not any(defender_alive):
            return self.TERMINAL_STATE, -2 * self.k

        # 2) If all attackers are out -> terminal with the summed reward
        if not any(attacker_alive):
            return self.TERMINAL_STATE, reward

        # 3) Mark eliminated attackers as INACTIVE (safe in 2v1/2v2; attackers' encoding includes INACTIVE)
        for i in range(2):
            if not attacker_alive[i]:
                next_att[i] = self.INACTIVE

        # 4) Mark eliminated defenders as INACTIVE **only if there are two defenders**
        if self.num_defenders == 2:
            for i in range(2):
                if not defender_alive[i]:
                    next_def[i] = self.INACTIVE

        # 5) Canonicalize and encode
        next_att.sort()
        if self.num_defenders == 2:
            next_def.sort()

        ns = self.positions_to_state(*(next_def + next_att))
        # (Optional) safety guard
        # assert 0 <= ns < self._n_states, f"next_state {ns} out of bounds in state with pos {defenders+attackers}"
        return ns, reward

################################################################################
############################# SOLVER helper for the LP #########################
################################################################################

def linprog_solve(reward_matrix: np.ndarray,
                  precision: int = 4,
                  rng: np.random.Generator | None = None) -> (float, np.ndarray, np.ndarray):
    """
    :param reward_matrix: M x N matrix. The entry of the jointly selected row and column
                          represents the winnings of the row player and the loss of the
                          column player (row maximizes, column minimizes).
    :param precision: precision of the matrix solver
    :param rng: optional numpy random generator; if None, uses default_rng()
    :return: (value of the game, optimal row policy, optimal column policy)
    """

    if rng is None:
        rng = np.random.default_rng()

    reward_matrix = -np.nan_to_num(np.round(reward_matrix, precision))
    m, n = reward_matrix.shape

    # Early-out if zero matrix: just play uniform
    if np.allclose(reward_matrix, 0):
        return 0.0, np.ones(m) / m, np.ones(n) / n

    # ========== Build LP for column player ==========
    # decision vars: y_1,...,y_n, v
    # minimize -v  (C = [0,...,0,-1])
    C = [0.0] * n + [-1.0]

    # constraints: reward_matrix[i_row, :] @ y >= v  for all rows
    # => -reward_matrix[i_row, :] @ y + v <= 0
    A = []
    B = []
    for i_row in range(m):
        col = reward_matrix[i_row, :]
        constraint_row = [-float(item) for item in col] + [1.0]
        A.append(constraint_row)
        B.append(0.0)

    # equality: sum_j y_j = 1
    A_eq_row = [1.0] * n + [0.0]
    A_eq = [A_eq_row]
    B_eq = [1.0]

    # bounds: 0 <= y_j <= 1, v free
    bounds = [(0.0, 1.0)] * n + [(None, None)]

    res = linprog(C, A_ub=A, b_ub=B, A_eq=A_eq, b_eq=B_eq,
                  bounds=bounds, method='highs')

    # ========== Handle failure: random fallback ==========
    if res is None or not res.success:
        print("WARNING: LP failed. Returning random policies. Status:", res)
        policy_row = rng.dirichlet(np.ones(m))
        policy_col = rng.dirichlet(np.ones(n))
        game_value = 0.0
        return game_value, policy_row, policy_col

    # ========== Column policy from primal ==========
    policy_col = np.array(res.x[:-1], dtype=float)
    policy_col[policy_col < 0] = 0.0
    sum_col = policy_col.sum()

    if sum_col <= 0 or not np.isfinite(sum_col):
        # fallback: random policy
        policy_col = rng.dirichlet(np.ones(n))
    else:
        policy_col /= sum_col

    # ========== Row policy from dual ==========
    # res.ineqlin.marginals are duals for A_ub * x <= b_ub
    if not hasattr(res, "ineqlin") or res.ineqlin is None:
        # fallback if duals not available
        policy_row = rng.dirichlet(np.ones(m))
    else:
        policy_row = -np.array(res.ineqlin.marginals, dtype=float)
        policy_row[policy_row < 0] = 0.0
        sum_row = policy_row.sum()

        if sum_row <= 0 or not np.isfinite(sum_row):
            # fallback: random policy
            policy_row = rng.dirichlet(np.ones(m))
        else:
            policy_row /= sum_row

    # NOTE: depending on your sign convention, you might want -res.fun here.
    game_value = res.fun

    return game_value, policy_row, policy_col

################################################################################
############################# SOLVER ###########################################
################################################################################


class NashSolver:
    def __init__(self, game: BaseGame):
        """
        Nash equilibrium solver using value iteration and linear programming.
        Player 1 (row) maximizes while Player 2 (column) minimizes.
        :param game: BaseGame object that provides the necessary information of the game
        """
        self.game = game

        # V stores the old value function and V_ stores the updated value function
        self.V, self.V_ = np.zeros(self.game.get_n_states()), np.zeros(self.game.get_n_states())
        # Q stores the Q matrix for each state. For games with different number of actions, the Q matrix is padded
        self.Q = np.zeros((self.game.get_n_states(), self.game.get_max_n_action1(), self.game.get_max_n_action2()))

        # Pre-load the n_actions for faster access
        self.n_actions_1 = [self.game.get_n_action1(s) for s in range(self.game.get_n_states())]
        self.n_actions_2 = [self.game.get_n_action2(s) for s in range(self.game.get_n_states())]

        # store the policy as a list of numpy arrays, such that policy[s] gives the action distribution
        self.policy_1 = [None for _ in range(self.game.get_n_states())]
        self.policy_2 = [None for _ in range(self.game.get_n_states())]

        # initialize logging variables
        self.iter_counter = 0
        self.error = []
        self.time = []

        # saving path
        self.save_path = None

    def solve(self,
              eps: float = 1e-3,
              n_policy_eval: int = 0,
              n_workers: int = 1,
              save_path: pathlib.Path = None,
              save_checkpoint: bool = False,
              verbose: bool = False) -> None:
        """
        Solve the Nash equilibrium of the zero-sum stochastic game using value iteration and linear programming
        :param eps: float, the convergence threshold of the l2-norm difference between old and new value function. `
        :param n_policy_eval: int, the number of policy evaluation step before the LP step. Default 0 to skip.
        :param n_workers: int, the number of workers to use for parallel computation. Default 1 for no parallelization.
        :param save_path: Path to save the model and log. If None, no saving is performed.
        :param save_checkpoint: Bool, whether to save the model and log at each iteration. Default False.
                                Recommended for large games in case solver crashed.
        :param verbose: Bool, whether to print the log. Default False.
        :return: Policy for player 1, Policy for player 2, Value function, Q function
        """

        self._initialize_saving(save_path, save_checkpoint)

        tic = time.time()
        diff = 1000
        self._print("Solving Nash equilibrium of a game with {} states".format(self.game.get_n_states()), verbose)
        self._print("Using {} workers for parallel computation".format(n_workers), verbose)
        self._print(
            f"{'Iter' : <10} {'Total Time': <10} {'Difference': <15} {'VI Time': <10} {'LP Time': <10} {'Policy Eval Time': <10}",
            verbose)

        if n_workers > 1:
            ctx = mp.get_context('spawn')
            pool = ctx.Pool(n_workers)
            self.n_workers = n_workers
        else:
            pool = None

        while diff > eps:

            if self.iter_counter >5:
                eps = 0.1*max(self.error)
                #print(eps)
            # update value function with the current policies
            tic_ = time.time()
            self._policy_eval(n_policy_eval=n_policy_eval)
            toc_ = time.time()
            time_policy_eval = toc_ - tic_

            # update the q function with the current value function
            tic_ = time.time()
            self._update_q()
            toc_ = time.time()
            time_q_update = toc_ - tic_

            # compute the new value function with the nash matrix game LP solver
            tic_ = time.time()
            self._update_v_(pool=pool)
            toc_ = time.time()
            time_v_update = toc_ - tic_

            toc = time.time()

            diff = np.round(np.linalg.norm(self.V_ - self.V), 4)
            self.error.append(diff)
            self.time.append(toc - tic)

            # copy the new value function to the old value function
            self._update_v()

            self._print(
                f"{self.iter_counter : <10} {np.round(toc - tic, 4): <10} {diff: <15} {np.round(time_q_update, 4): <10} {np.round(time_v_update, 4): <10} {np.round(time_policy_eval, 4): <10}",
                verbose)
            self.iter_counter += 1

            if self.iter_counter % 5 == 0 and self.iter_counter > 0 and self.save_checkpoint:
                self.save(check_point=True)

        self._print("Value iterations converged!", verbose)
        # self.policy_1, self.policy_2 = self._generate_policy_parallel()

        if pool is not None:
            pool.terminate()
            pool.join()

        n_matrix_game_solver_called = int(self.game.get_n_states() * self.iter_counter)
        self._print("Matrix Game solver called {} times".format(n_matrix_game_solver_called), verbose)

    def _policy_eval(self, n_policy_eval: int, pool: Pool = None):
        # evaluate the value function under the current policy to help speed up the convergence.
        # if the n_policy_eval is 0, then this step is skipped.
        for _ in range(n_policy_eval):
            #t1 = time.time()
            self._update_q()
            #t2 = time.time()
            self._eval_v()
            #t3 = time.time()
            #print("update q eval took {} seconds".format(t2 - t1))
            #print("update v eval took {} seconds".format(t3 - t2))

    def _update_q(self):
        if self.game.has_compressed_transition:
            trans_to_state, trans_prob = self.game.get_padded_transitions()
            rewards = self.game.get_all_rewards()
            #self.Q = rewards.reshape(-1,1,1) + self.game.gamma * np.sum(trans_prob * self.V[trans_to_state], axis=3)
            self.Q = rewards + self.game.gamma * np.sum(trans_prob * self.V[trans_to_state], axis=3)
        else:
            trans = self.game.get_all_transitions()
            rewards = self.game.get_all_rewards()
            self.Q = rewards.reshape(-1,1,1) + self.game.gamma * np.sum(trans * self.V, axis=3)

    def _eval_v(self):
        if self.policy_1[0] is not None or self.policy_2[0] is not None:
            v = [self.policy_1[s] @ self.Q[s, :self.n_actions_1[s], :self.n_actions_2[s]]
                 @ self.policy_2[s] for s in range(self.game.get_n_states())]
            self.V = np.array(v)

    def _update_v_(self, pool: Pool = None):
        if pool is not None:
            # Q matrix are passed with their actual size rather than the padded one to save time for LP.
            results = pool.map(linprog_solve, [self.Q[s, : self.n_actions_1[s], : self.n_actions_2[s]]
                                                for s in range(self.game.get_n_states())])
            self.V_ = np.array([res[0] for res in results])
            self.policy_1 = [res[1] for res in results]
            self.policy_2 = [res[2] for res in results]
        else:
            for s in range(self.game.get_n_states()):
                self.V_[s], self.policy_1[s], self.policy_2[s] = linprog_solve(self.Q[s])

    def _update_v(self):
        self.V = deepcopy(self.V_)

    def get_policy(self):
        return self.policy_1, self.policy_2

    def get_v(self):
        return self.V

    def get_q(self):
        return self.Q

    def save(self, check_point=False):
        save_path = self.save_path
        self.save_policy(save_path, check_point)
        self.save_model(save_path, check_point)
        self.save_log(save_path, check_point)

    def save_policy(self, save_path: pathlib.Path, check_point):
        if not check_point:
            with open(save_path / "policy.pkl", "wb") as f:
                data = [self.policy_1, self.policy_2]
                pickle.dump(data, f)

    def save_model(self, save_path: pathlib.Path, check_point):
        if check_point:
            file_name = save_path / "model_check_point.pkl"
        else:
            file_name = save_path / "model.pkl"
        with open(file_name, "wb") as f:
            data = [self.V, self.Q]
            pickle.dump(data, f)

    def save_log(self, save_path: pathlib.Path, check_point):
        if check_point:
            file_name = save_path / "log_check_point.pkl"
        else:
            file_name = save_path / "log.pkl"
        with open(file_name, "wb") as f:
            data = [self.error, self.time]
            pickle.dump(data, f)

    def load_checkpoint(self, save_path_model: pathlib.Path, save_path_log: pathlib.Path):
        with open(save_path_model, "rb") as f_model:
            self.V, self.Q = deepcopy(pickle.load(f_model))
        with open(save_path_log, "rb") as f_log:
            self.error, self.time = deepcopy(pickle.load(f_log))

        self.V_ = deepcopy(self.V)
        self.iter_counter = len(self.error)

    def _print(self, text: str, verbose: bool = True):
        if verbose:
            print(text)

    def _initialize_saving(self, save_path: pathlib.Path, save_checkpoint: bool):
        self.save_checkpoint = save_checkpoint
        if self.save_checkpoint:
            assert save_path is not None, "Requested to save checkpoints, but no save path provided!"
        self.save_path = save_path

        if save_path is not None and not save_path.exists():
            save_path.mkdir(parents=True)

if __name__ == "__main__":
    import yaml
    from pathlib import Path

    # --- Resolve paths relative to this file ---
    LEAGUE_ROOT = Path(__file__).resolve().parent.parent.parent  # .../League/
    GRAPH_DIR   = LEAGUE_ROOT / "graphs"
    # CONFIG_PATH = LEAGUE_ROOT / "example" / "example_config.yml"
    CONFIG_PATH = "/Users/mukeshgollen/PycharmProjects/League_may7_25/League/config/example_configs/osm_a/R5B5F3-5/R5B5F3-5_run4.yml"
    # --- Load config ---
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    graph_name  = cfg["environment"]["graph_name"]                  # "osm_200_a.pkl"
    real_flags  = cfg["flags"]["real_positions"]                    # [24, 25, 9]
    cand_flags  = cfg["flags"]["candidate_positions"]               # [24, 25, 9, 8, 15]
    tag_radius  = cfg["agents"]["blue_global"]["tagging_radius"]    # 1
    capture_rad = cfg["agents"]["red_global"]["capture_radius"]     # 2

    atk_positions = {name: info["start_node_id"]
                     for name, info in cfg["agents"]["red_config"].items()}
    def_positions = {name: info["start_node_id"]
                     for name, info in cfg["agents"]["blue_config"].items()}

    # --- Load graph and convert to undirected simple graph for MCTS ---
    with open(GRAPH_DIR / graph_name, "rb") as f:
        raw_graph = pickle.load(f)
    graph_simple = nx.Graph(raw_graph)

    # --- Build MCTS object ---
    mamcts_defender = MultiAgentMCTS(
        graph=graph_simple,
        flag_nodes=real_flags,
        candidate_flag_nodes=cand_flags,
        k=0.5,
        tag_radius=tag_radius,
        capture_radius=capture_rad,
    )

    # --- Load / solve the 1v1 primitive policy ---
    att_policy_1v1_defender, def_policy_1v1_defender, V_1v1_defender = get_or_solve_policy(
        graph_simple, real_flags, n_defenders=1, capture_radius=capture_rad
    )

    # --- Timing run ---
    print("Attacker positions:", atk_positions)
    print("Defender positions:", def_positions)

    N_RUNS = 5
    run_times = []
    for i in range(N_RUNS):
        t0 = time.perf_counter()
        best_att, best_def, root = run_regret_matching(
            attacker=atk_positions,
            defender=def_positions,
            mamcts=mamcts_defender,
            n_simulations=2000,
            depth=2,
            gamma_rm=0.4,
            min_actions_a=2,
            min_actions_d=1,
        )
        elapsed = time.perf_counter() - t0
        run_times.append(elapsed)
        print(f"  run {i + 1}: {elapsed:.3f}s | att={best_att} | def={best_def}")

    print(f"\nAvg over {N_RUNS} runs: {sum(run_times) / N_RUNS:.3f}s")

    