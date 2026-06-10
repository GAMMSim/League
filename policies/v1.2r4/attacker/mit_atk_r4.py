"""
MIT GNN Attacker r3 — single-file submission.

Blue MAPPO attacker using a 3-agent checkpoint with zero-shot transfer to 5v5.
All GNN policy classes are inlined; no external graph_policy.py or customCTF.py required.

Zero-shot transfer: agent_id % policy.n_agents gives round-robin role assignment,
so the 3-agent checkpoint runs unchanged for any team size.

Submission package
------------------
    mit_atk_r3.py                this file (self-contained)
    iter1_blue_br_best.zip       3-agent MAPPO checkpoint  (3v3 and 5v5 zero-shot)
    requirements.txt

Runtime dependencies (see requirements.txt):
    torch>=2.7, torch-geometric>=2.7, stable-baselines3>=2.6,
    networkx>=3.4, numpy>=2.2, gymnasium>=1.1
"""

import os
import sys
import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.data import Data, Batch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy, BaseFeaturesExtractor
from gymnasium.spaces import Box, Dict, Discrete

# ── Path setup ────────────────────────────────────────────────────────────────
_POLICY_DIR = os.path.dirname(os.path.abspath(__file__))

if _POLICY_DIR not in sys.path:
    sys.path.insert(0, _POLICY_DIR)

# ── Checkpoint path (single 3-agent checkpoint; zero-shot to any team size) ───
_CKPT_PATH = os.path.join(_POLICY_DIR, 'iter1_blue_br_best.zip')
_CKPT_PATHS = {3: _CKPT_PATH, 5: _CKPT_PATH}
_DEVICE = 'cpu'

_GRU_H            = 128
_MAX_DEGREE       = 9
_NUM_ACTIONS      = 11
_EGO_MAX_HOPS     = 6
_FLAG_SENSE_UTM_M = 450.0

# ── Per-map flag hypothesis override ──────────────────────────────────────────
# Normally not needed — flag hypotheses come from sensors["candidate_flag"].
# Key   = frozenset of graph node IDs  (unique map fingerprint)
# Value = (flag_hyp_node_ids, blue_flag_node_id, sensing_radius_utm)
_FLAG_HYP_CONFIG = {}

# ── Checkpoint obs/action spaces (exact dims from training; bypasses cloudpickle) ──
_MAX_N, _MAX_E, _F_NODE = 104, 260, 17
def _build_obs_space():
    _N, _E, _FN = _MAX_N, _MAX_E, _F_NODE
    ego = {
        'x':                    Box(-np.inf, np.inf, (_N, _FN), np.float32),
        'edge_index':           Box(0, _N-1,  (2, _E),     np.int64),
        'edge_attr':            Box(-np.inf, np.inf, (_E, 1), np.float32),
        'node_visibility_mask': Box(0, 1,    (_N,),         np.int8),
        'edge_visibility_mask': Box(0.0, 1.0,(_E,),         np.float32),
        'neighbor_local_idx':   Box(0, _N-1, (9,),          np.int64),
        'neighbor_mask':        Box(0.0, 1.0,(9,),           np.float32),
        'agent_node_local_idx': Box(0, _N-1, (1,),          np.int64),
        'agent_node_mask':      Box(0.0, 1.0,(200,),         np.float32),
        'action_mask':          Box(0, 1,    (11,),          np.int8),
        'num_visible_nodes':    Box(0, _N,   (1,),           np.int64),
        'num_visible_edges':    Box(0, _E,   (1,),           np.int64),
        'agent_id':             Box(0, 2,    (1,),           np.int64),
    }
    for tm in ('teammate_0', 'teammate_1'):
        ego.update({
            f'{tm}_x':                    Box(-np.inf, np.inf, (_N, _FN), np.float32),
            f'{tm}_edge_index':           Box(0, _N-1,  (2, _E),     np.int64),
            f'{tm}_edge_attr':            Box(-np.inf, np.inf, (_E, 1), np.float32),
            f'{tm}_node_visibility_mask': Box(0, 1,    (_N,),         np.int8),
            f'{tm}_edge_visibility_mask': Box(0.0, 1.0,(_E,),         np.float32),
            f'{tm}_agent_node_local_idx': Box(0, _N-1, (1,),          np.int64),
            f'{tm}_agent_id':             Box(0, 2,    (1,),           np.int64),
        })
    return Dict(ego)
_OBS_SPACE    = _build_obs_space()
_ACTION_SPACE = Discrete(11)

# ── Lazy globals ──────────────────────────────────────────────────────────────
_MODELS      = {}   # n_blue → PPO model
_POLICIES    = {}   # n_blue → model.policy
_GRAPH_CACHE = {}   # (frozenset(nodes), frozenset(hyps)) → (GRAPH dict, EGO dict)
_GRAPH            = None
_EGO              = None
_FLAG_HYP_IDXS    = None
_DEFAULT_NODE_IDX = 0

_TM_KEYS = ('x', 'node_visibility_mask', 'edge_visibility_mask',
            'edge_index', 'edge_attr', 'agent_node_local_idx')


# ═════════════════════════════════════════════════════════════════════════════
# GNN Architecture (inlined from graph_policy.py)
# ═════════════════════════════════════════════════════════════════════════════

class MPNNLayer_SubgraphCompatible(MessagePassing):
    def __init__(self, node_in_dim, edge_in_dim, out_dim, use_attention=False):
        super().__init__(aggr='add')
        self.use_attention = use_attention
        self.mlp = nn.Sequential(
            nn.Linear(2 * node_in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        if use_attention:
            self.attn = nn.Linear(2 * node_in_dim, 1, bias=False)

    def forward(self, x, edge_index, edge_attr, edge_mask=None):
        x = x.float()
        if edge_attr is not None:
            edge_attr = edge_attr.float()
        else:
            edge_attr = torch.zeros((edge_index.size(1), 0), device=x.device)

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        if edge_attr.size(1) == 0:
            loop_attr = edge_attr.new_zeros((x.size(0), 0))
            edge_attr = torch.cat([edge_attr, loop_attr], dim=0)

        if edge_mask is not None:
            loop_mask = edge_mask.new_ones(x.size(0))
            edge_mask = torch.cat([edge_mask, loop_mask], dim=0)
        else:
            edge_mask = None

        return self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_mask=None)

    def message(self, x_i, x_j, edge_attr, edge_index_i, edge_mask=None):
        m   = torch.cat([x_i, x_j], dim=-1)
        msg = self.mlp(m)
        if self.use_attention:
            alpha = self.attn(m)
            alpha = softmax(alpha, edge_index_i)
            msg   = msg * alpha
        if edge_mask is not None:
            msg = msg * edge_mask.unsqueeze(-1)
        return msg


def build_pyg_batch(obs):
    B         = obs["x"].shape[0]
    data_list = []
    for b in range(B):
        node_mask  = obs["node_visibility_mask"][b].bool()
        edge_mask  = obs["edge_visibility_mask"][b].bool()
        x          = obs["x"][b][node_mask]
        edge_index = obs["edge_index"][b][:, edge_mask]
        edge_attr  = obs["edge_attr"][b][edge_mask]
        data = Data(
            x          = torch.as_tensor(x,          dtype=torch.float32),
            edge_index = torch.as_tensor(edge_index, dtype=torch.long),
            edge_attr  = torch.as_tensor(edge_attr,  dtype=torch.float32)
                         if edge_attr is not None else None,
        )
        data_list.append(data)
    return Batch.from_data_list(data_list)


class GraphCTFMPNNExtractor_SubgraphCompatible(BaseFeaturesExtractor):
    def __init__(self, observation_space, embedding_dim=64, use_attention=False):
        super().__init__(observation_space, features_dim=embedding_dim)
        node_feat_dim = observation_space['x'].shape[-1]
        edge_feat_dim = observation_space.get(
            'edge_attr', Box(low=-1, high=1, shape=(1, 1))
        ).shape[-1]
        self.mpnn1 = MPNNLayer_SubgraphCompatible(node_feat_dim, edge_feat_dim, 64,
                                                   use_attention=use_attention)
        self.mpnn2 = MPNNLayer_SubgraphCompatible(64, edge_feat_dim, 64,
                                                   use_attention=use_attention)
        self.mpnn3 = MPNNLayer_SubgraphCompatible(64, edge_feat_dim, embedding_dim,
                                                   use_attention=use_attention)

    def forward(self, obs):
        pyg_batch = build_pyg_batch(obs)
        if pyg_batch.edge_index.dtype != torch.long:
            pyg_batch.edge_index = pyg_batch.edge_index.long()
        h = pyg_batch.x
        h = F.relu(self.mpnn1(h, pyg_batch.edge_index, pyg_batch.edge_attr))
        h = F.relu(self.mpnn2(h, pyg_batch.edge_index, pyg_batch.edge_attr))
        h = self.mpnn3(h, pyg_batch.edge_index, pyg_batch.edge_attr)
        return h, pyg_batch



class GraphCTFNeighborBatchPolicy_SubgraphCompatible_v4_hybrid(ActorCriticPolicy):
    """
    Hybrid policy: v2 actor (independent neighbor scoring) + v4 critic (global pooling).

    Actor : node_mlp(h_neighbor) — h_agent NOT used in movement logits, shielding
            the backbone from policy-loss gradient through neighbor scores.
    Critic: value_head(cat[h_agent, mean_pool, max_pool]) — v4 global context.
    """

    def __init__(self, *args, use_attention: bool = False, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=GraphCTFMPNNExtractor_SubgraphCompatible,
            features_extractor_kwargs=dict(embedding_dim=64, use_attention=use_attention),
        )
        hidden_dim = 64
        self.hidden_dim       = hidden_dim
        self.extra_action_dim = 2

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
        )
        self.extra_action_head = nn.Linear(hidden_dim, self.extra_action_dim)
        self.value_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
        )

    def _get_embeddings(self, obs):
        device = obs['x'].device
        node_h, pyg_batch = self.features_extractor(obs)
        ptr = pyg_batch.ptr
        B   = ptr.size(0) - 1

        agent_local_idx = obs['agent_node_local_idx'].long().to(device)
        if agent_local_idx.dim() > 1:
            agent_local_idx = agent_local_idx.squeeze(-1)
        agent_global_idx    = ptr[:-1] + agent_local_idx
        h_agent             = node_h[agent_global_idx]

        neighbor_local_idx  = obs['neighbor_local_idx'].long().to(device)
        neighbor_mask       = obs['neighbor_mask'].to(device)
        neighbor_global_idx = (ptr[:-1].unsqueeze(1) + neighbor_local_idx).clamp(0, node_h.size(0) - 1)
        neighbor_embeddings = node_h[neighbor_global_idx]

        return node_h, pyg_batch, B, device, h_agent, neighbor_embeddings, neighbor_mask

    def _compute_logits(self, neighbor_embeddings, neighbor_mask, h_agent, obs, B, device):
        neighbor_logits = self.node_mlp(neighbor_embeddings).squeeze(-1)
        neighbor_logits = neighbor_logits.masked_fill(neighbor_mask == 0, float('-inf'))
        extra_logits    = self.extra_action_head(h_agent)
        full_logits     = torch.cat([neighbor_logits, extra_logits], dim=-1)
        if 'action_mask' in obs and obs['action_mask'] is not None:
            mask = obs['action_mask'].to(torch.bool).to(device)
            A = full_logits.shape[1]
            if mask.shape[1] > A:
                mask = mask[:, :A]
            elif mask.shape[1] < A:
                pad = torch.zeros(B, A - mask.shape[1], dtype=torch.bool, device=device)
                mask = torch.cat([mask, pad], dim=-1)
            full_logits = full_logits.masked_fill(~mask, float('-inf'))
        return full_logits

    def _compute_values(self, node_h, pyg_batch, h_agent):
        mean_g = global_mean_pool(node_h, pyg_batch.batch)
        max_g  = global_max_pool(node_h, pyg_batch.batch)
        return self.value_head(torch.cat([h_agent, mean_g, max_g], dim=-1)).squeeze(-1)

    def forward(self, obs, deterministic=False):
        node_h, pyg_batch, B, device, h_agent, neighbor_embeddings, neighbor_mask = \
            self._get_embeddings(obs)
        full_logits = self._compute_logits(
            neighbor_embeddings, neighbor_mask, h_agent, obs, B, device)
        dist      = Categorical(logits=full_logits)
        actions   = dist.sample() if not deterministic else torch.argmax(full_logits, dim=-1)
        log_probs = dist.log_prob(actions)
        values    = self._compute_values(node_h, pyg_batch, h_agent)
        return actions, values, log_probs

    def predict_values(self, obs):
        node_h, pyg_batch, B, device, h_agent, _, _ = self._get_embeddings(obs)
        return self._compute_values(node_h, pyg_batch, h_agent)

    def evaluate_actions(self, obs, actions):
        node_h, pyg_batch, B, device, h_agent, neighbor_embeddings, neighbor_mask = \
            self._get_embeddings(obs)
        full_logits = self._compute_logits(
            neighbor_embeddings, neighbor_mask, h_agent, obs, B, device)
        dist     = Categorical(logits=full_logits)
        log_prob = dist.log_prob(actions)
        entropy  = dist.entropy()
        values   = self._compute_values(node_h, pyg_batch, h_agent)
        return values, log_prob, entropy


class CommBlock(nn.Module):
    """Single-head cross-attention communication block for MARL."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.W_q   = nn.Linear(dim, dim, bias=False)
        self.W_k   = nn.Linear(dim, dim, bias=False)
        self.W_v   = nn.Linear(dim, dim, bias=False)
        self.W_o   = nn.Linear(dim, dim, bias=False)
        self.norm  = nn.LayerNorm(dim)
        self.scale = dim ** -0.5

    def forward(
        self,
        z_i:   torch.Tensor,
        z_j:   torch.Tensor,
        alive: torch.Tensor | None,
    ) -> torch.Tensor:
        q      = self.W_q(z_i).unsqueeze(1)
        k      = self.W_k(z_j)
        v      = self.W_v(z_j)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        if alive is not None:
            scores = scores.masked_fill(~alive.unsqueeze(1), float('-inf'))
        attn = torch.softmax(scores, dim=-1).nan_to_num(0.0)
        out  = (attn @ v).squeeze(1)
        return self.norm(z_i + self.W_o(out))


class _MAPPOIndependentActorMixin:
    """
    Shared MAPPO logic for v4_hybrid backbone.

    Actor : node_mlp_i(h_nbr) — independent per-agent scoring, no h_actor dependency.
    Critic: DeepSets V = rho(phi_ego + sum phi_tm) — permutation invariant.
    GRU   : optional per-agent GRUCells (use_gru=True).

    MRO must place this mixin BEFORE the backbone class.
    Concrete subclasses call self._mappo_independent_setup(D, use_gru, n_agents)
    at the end of their __init__.
    """

    _GRU_H = 128

    def _mappo_independent_setup(self, D: int, use_gru: bool, n_agents: int = 2) -> None:
        self.use_gru       = use_gru
        self._h_buf        = None
        self._h_buf_critic = None
        self.n_agents      = n_agents
        A = self._GRU_H if use_gru else D

        if use_gru:
            self.gru_cells = nn.ModuleList([
                nn.GRUCell(input_size=D, hidden_size=self._GRU_H) for _ in range(n_agents)
            ])
            self.critic_gru_cells = nn.ModuleList([
                nn.GRUCell(input_size=D, hidden_size=self._GRU_H) for _ in range(n_agents)
            ])
            self.comm_block = CommBlock(self._GRU_H)

        self.actor_node_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, 1))
            for _ in range(n_agents)
        ])
        self.extra_action_heads = nn.ModuleList([
            nn.Linear(A, self.extra_action_dim) for _ in range(n_agents)
        ])
        self.phi_net = nn.Sequential(nn.Linear(A + 2*D, A), nn.ReLU(), nn.Linear(A, A))
        self.rho_net = nn.Sequential(nn.Linear(A, A), nn.ReLU(), nn.Linear(A, 1))

    # GRU helpers -------------------------------------------------------

    def _apply_gru(self, h: torch.Tensor, agent_id: torch.Tensor) -> torch.Tensor:
        B, device = h.size(0), h.device
        if not self.training:
            if self._h_buf is None or self._h_buf.shape[0] != B or self._h_buf.device != device:
                self._h_buf = torch.zeros(B, self._GRU_H, device=device)
            h0 = self._h_buf.detach()
        else:
            h0 = h.new_zeros(B, self._GRU_H)
        h_cells = torch.stack([cell(h, h0) for cell in self.gru_cells], dim=1)
        idx     = agent_id.clamp(0, self.n_agents - 1)
        h_new   = h_cells[torch.arange(B, device=device), idx]
        if not self.training:
            self._h_buf = h_new.detach()
        return h_new

    def reset_hidden(self, indices=None) -> None:
        for buf_attr in ('_h_buf', '_h_buf_critic'):
            buf = getattr(self, buf_attr, None)
            if buf is None:
                continue
            if indices is None:
                buf.zero_()
            else:
                buf[indices] = 0.0

    def _apply_gru_stateless(self, h: torch.Tensor, agent_id: torch.Tensor) -> torch.Tensor:
        B, device = h.size(0), h.device
        h0      = h.new_zeros(B, self._GRU_H)
        h_cells = torch.stack([cell(h, h0) for cell in self.gru_cells], dim=1)
        return h_cells[torch.arange(B, device=device), agent_id.clamp(0, self.n_agents-1)]

    def _apply_gru_critic(self, h: torch.Tensor, agent_id: torch.Tensor) -> torch.Tensor:
        B, device = h.size(0), h.device
        if not self.training:
            if (self._h_buf_critic is None or self._h_buf_critic.shape[0] != B
                    or self._h_buf_critic.device != device):
                self._h_buf_critic = torch.zeros(B, self._GRU_H, device=device)
            h0 = self._h_buf_critic.detach()
        else:
            h0 = h.new_zeros(B, self._GRU_H)
        h_cells = torch.stack([cell(h, h0) for cell in self.critic_gru_cells], dim=1)
        idx     = agent_id.clamp(0, self.n_agents - 1)
        h_new   = h_cells[torch.arange(B, device=device), idx]
        if not self.training:
            self._h_buf_critic = h_new.detach()
        return h_new

    def _apply_gru_critic_stateless(self, h: torch.Tensor, agent_id: torch.Tensor) -> torch.Tensor:
        B, device = h.size(0), h.device
        h0      = h.new_zeros(B, self._GRU_H)
        h_cells = torch.stack([cell(h, h0) for cell in self.critic_gru_cells], dim=1)
        return h_cells[torch.arange(B, device=device), agent_id.clamp(0, self.n_agents-1)]

    def _get_tm_h_actor(self, B, device, agent_id):
        # CommBlock weights were never trained; always return None.
        return None

    def _get_tm_alive(self, obs, B, device):
        alive_list = []
        for t in range(self.n_agents - 1):
            if f'teammate_{t}_x' not in obs:
                return None
            tm_x     = obs[f'teammate_{t}_x']
            tm_local = obs[f'teammate_{t}_agent_node_local_idx'].long()
            if tm_local.dim() > 1:
                tm_local = tm_local.squeeze(-1)
            alive_list.append(tm_x[torch.arange(B, device=device), tm_local, -1].bool())
        return torch.stack(alive_list, dim=1)

    def _get_tm_alive_flat(self, obs, BT, device):
        alive_list = []
        for t in range(self.n_agents - 1):
            if f'teammate_{t}_x' not in obs:
                return None
            tm_x     = obs[f'teammate_{t}_x']
            tm_local = obs[f'teammate_{t}_agent_node_local_idx'].long()
            if tm_local.dim() > 1:
                tm_local = tm_local.squeeze(-1)
            alive_list.append(tm_x[torch.arange(BT, device=device), tm_local, -1].bool())
        return torch.stack(alive_list, dim=1)

    def _compute_tm_z_actor_stateless(self, obs, agent_id_flat, BT, device):
        z_tm_list = []
        for t in range(self.n_agents - 1):
            if f'teammate_{t}_x' not in obs:
                return None
            node_h_tm, pyg_batch_tm = self._get_teammate_embeddings(obs, t)
            tm_local = obs[f'teammate_{t}_agent_node_local_idx'].long()
            if tm_local.dim() > 1:
                tm_local = tm_local.squeeze(-1)
            h_tm  = node_h_tm[pyg_batch_tm.ptr[:-1] + tm_local]
            tm_id_key = f'teammate_{t}_agent_id'
            tm_id = (obs[tm_id_key].long().squeeze(-1) if tm_id_key in obs
                     else (agent_id_flat + t + 1) % self.n_agents)
            z_tm_list.append(self._apply_gru_stateless(h_tm, tm_id))
        return torch.stack(z_tm_list, dim=1)

    def load_state_dict(self, state_dict, strict=True):
        """Remap old per-name keys to ModuleList keys for backward compat."""
        remapped = {}
        for k, v in state_dict.items():
            new_k = k
            for i in range(8):
                new_k = new_k.replace(f'node_mlp_{i}.',       f'actor_node_mlps.{i}.')
                new_k = new_k.replace(f'extra_action_head_{i}.', f'extra_action_heads.{i}.')
                new_k = new_k.replace(f'gru_{i}.',            f'gru_cells.{i}.')
            remapped[new_k] = v
        return super().load_state_dict(remapped, strict=strict)

    # Actor: independent scoring ----------------------------------------

    def _compute_actor_logits(self, h_actor, neighbor_embeddings, neighbor_mask, agent_id):
        B, device = h_actor.size(0), h_actor.device
        nbr_all = torch.stack(
            [mlp(neighbor_embeddings).squeeze(-1) for mlp in self.actor_node_mlps], dim=1
        )
        ext_all = torch.stack(
            [head(h_actor) for head in self.extra_action_heads], dim=1
        )
        idx          = agent_id.clamp(0, self.n_agents - 1)
        arange       = torch.arange(B, device=device)
        nbr_logits   = nbr_all[arange, idx].masked_fill(neighbor_mask == 0, float('-inf'))
        extra_logits = ext_all[arange, idx]
        return nbr_logits, extra_logits

    # Critic: DeepSets --------------------------------------------------

    def _get_teammate_embeddings(self, obs, t):
        tm_obs = {
            'x':                    obs[f'teammate_{t}_x'],
            'node_visibility_mask': obs[f'teammate_{t}_node_visibility_mask'],
            'edge_index':           obs[f'teammate_{t}_edge_index'],
            'edge_attr':            obs[f'teammate_{t}_edge_attr'],
            'edge_visibility_mask': obs[f'teammate_{t}_edge_visibility_mask'],
        }
        return self.features_extractor(tm_obs)

    def _compute_values(self, node_h, pyg_batch, h_critic, obs=None, agent_id=None):
        mean_s  = global_mean_pool(node_h, pyg_batch.batch)
        max_s   = global_max_pool(node_h, pyg_batch.batch)
        phi_ego = self.phi_net(torch.cat([h_critic, mean_s, max_s], dim=-1))
        team_phi = phi_ego
        for t in range(self.n_agents - 1):
            if obs is None or f'teammate_{t}_x' not in obs:
                continue
            node_h_tm, pyg_batch_tm = self._get_teammate_embeddings(obs, t)
            tm_local = obs[f'teammate_{t}_agent_node_local_idx'].long()
            if tm_local.dim() > 1:
                tm_local = tm_local.squeeze(-1)
            h_tm    = node_h_tm[pyg_batch_tm.ptr[:-1] + tm_local]
            mean_tm = global_mean_pool(node_h_tm, pyg_batch_tm.batch)
            max_tm  = global_max_pool(node_h_tm, pyg_batch_tm.batch)
            if self.use_gru and agent_id is not None:
                tm_id_key = f'teammate_{t}_agent_id'
                tm_id = (obs[tm_id_key].long().squeeze(-1) if tm_id_key in obs
                         else (agent_id + t + 1) % self.n_agents)
                h_tm = self._apply_gru_critic_stateless(h_tm, tm_id)
            team_phi = team_phi + self.phi_net(torch.cat([h_tm, mean_tm, max_tm], dim=-1))
        return self.rho_net(team_phi).squeeze(-1)

    def _apply_mask(self, full_logits, obs, B, device):
        if 'action_mask' not in obs or obs['action_mask'] is None:
            return full_logits
        mask = obs['action_mask'].to(torch.bool).to(device)
        A = full_logits.shape[1]
        if mask.shape[1] > A:
            mask = mask[:, :A]
        elif mask.shape[1] < A:
            pad = torch.zeros(B, A - mask.shape[1], dtype=torch.bool, device=device)
            mask = torch.cat([mask, pad], dim=-1)
        return full_logits.masked_fill(~mask, float('-inf'))

    def _get_agent_id(self, obs, B, device):
        if 'agent_id' in obs:
            return obs['agent_id'].long().squeeze(-1)
        return torch.zeros(B, dtype=torch.long, device=device)

    # ActorCriticPolicy interface ----------------------------------------

    def forward(self, obs, deterministic=False):
        node_h, pyg_batch, B, device, h_agent, neighbor_embeddings, neighbor_mask = \
            self._get_embeddings(obs)
        agent_id = self._get_agent_id(obs, B, device)
        if self.use_gru:
            z_actor  = self._apply_gru(h_agent, agent_id)
            z_tm     = self._get_tm_h_actor(B, device, agent_id)
            z_comm   = self.comm_block(z_actor, z_tm, self._get_tm_alive(obs, B, device)) \
                       if z_tm is not None else z_actor
            z_critic = self._apply_gru_critic(h_agent, agent_id)
        else:
            z_comm = z_critic = h_agent
        nbr_logits, extra_logits = self._compute_actor_logits(
            z_comm, neighbor_embeddings, neighbor_mask, agent_id)
        full_logits = self._apply_mask(
            torch.cat([nbr_logits, extra_logits], dim=-1), obs, B, device)
        dist      = Categorical(logits=full_logits)
        actions   = dist.sample() if not deterministic else torch.argmax(full_logits, dim=-1)
        log_probs = dist.log_prob(actions)
        values    = self._compute_values(node_h, pyg_batch, z_critic, obs, agent_id)
        return actions, values, log_probs

    def predict_values(self, obs):
        node_h, pyg_batch, B, device, h_agent, _, _ = self._get_embeddings(obs)
        agent_id = self._get_agent_id(obs, B, device)
        z_critic = self._apply_gru_critic(h_agent, agent_id) if self.use_gru else h_agent
        return self._compute_values(node_h, pyg_batch, z_critic, obs, agent_id)

    def evaluate_actions(self, obs, actions):
        node_h, pyg_batch, B, device, h_agent, neighbor_embeddings, neighbor_mask = \
            self._get_embeddings(obs)
        agent_id = self._get_agent_id(obs, B, device)
        if self.use_gru:
            z_actor  = self._apply_gru(h_agent, agent_id)
            z_tm     = self._get_tm_h_actor(B, device, agent_id)
            z_comm   = self.comm_block(z_actor, z_tm, self._get_tm_alive(obs, B, device)) \
                       if z_tm is not None else z_actor
            z_critic = self._apply_gru_critic(h_agent, agent_id)
        else:
            z_comm = z_critic = h_agent
        nbr_logits, extra_logits = self._compute_actor_logits(
            z_comm, neighbor_embeddings, neighbor_mask, agent_id)
        full_logits = self._apply_mask(
            torch.cat([nbr_logits, extra_logits], dim=-1), obs, B, device)
        dist     = Categorical(logits=full_logits)
        values   = self._compute_values(node_h, pyg_batch, z_critic, obs, agent_id)
        return values, dist.log_prob(actions), dist.entropy()

    def evaluate_actions_sequence(self, obs, actions, h0, episode_starts,
                                   seq_len_critic=None):
        BT = actions.shape[0]
        B, T = episode_starts.shape
        device = h0.device
        node_h, pyg_batch, _, _, h_agent_flat, neighbor_embeddings, neighbor_mask = \
            self._get_embeddings(obs)
        agent_id_flat = self._get_agent_id(obs, BT, device)

        if self.use_gru:
            h0_actor, h0_critic = h0[:, :self._GRU_H], h0[:, self._GRU_H:]
            h_agent_seq  = h_agent_flat.view(B, T, -1)
            agent_id_seq = agent_id_flat.view(B, T)

            z_tm_flat  = self._compute_tm_z_actor_stateless(obs, agent_id_flat, BT, device)
            alive_flat = self._get_tm_alive_flat(obs, BT, device) if z_tm_flat is not None else None

            h_a = h0_actor
            h_gru_actor = []
            for t in range(T):
                if t > 0:
                    h_a = h_a * (1.0 - episode_starts[:, t].unsqueeze(-1))
                aid_t   = agent_id_seq[:, t]
                h_cells = torch.stack(
                    [cell(h_agent_seq[:, t], h_a) for cell in self.gru_cells], dim=1)
                h_a = h_cells[torch.arange(B, device=device), aid_t.clamp(0, self.n_agents-1)]
                h_gru_actor.append(h_a)
            z_actor_flat = torch.stack(h_gru_actor, dim=1).reshape(BT, -1)
            z_comm_flat  = (self.comm_block(z_actor_flat, z_tm_flat, alive_flat)
                            if z_tm_flat is not None else z_actor_flat)

            sc = seq_len_critic if seq_len_critic is not None else T
            if T % sc != 0:
                raise ValueError(f"seq_len_critic={sc} must evenly divide T={T}")
            h_c = h0_critic
            h_gru_critic = []
            for t in range(T):
                if t > 0 and t % sc == 0:
                    h_c = h_c.detach()
                if t > 0:
                    h_c = h_c * (1.0 - episode_starts[:, t].unsqueeze(-1))
                aid_t   = agent_id_seq[:, t]
                h_cells = torch.stack(
                    [cell(h_agent_seq[:, t], h_c) for cell in self.critic_gru_cells], dim=1)
                h_c = h_cells[torch.arange(B, device=device), aid_t.clamp(0, self.n_agents-1)]
                h_gru_critic.append(h_c)
            z_critic_flat = torch.stack(h_gru_critic, dim=1).reshape(BT, -1)
        else:
            z_comm_flat = z_critic_flat = h_agent_flat

        nbr_logits, extra_logits = self._compute_actor_logits(
            z_comm_flat, neighbor_embeddings, neighbor_mask, agent_id_flat)
        full_logits = self._apply_mask(
            torch.cat([nbr_logits, extra_logits], dim=-1), obs, BT, device)
        dist     = Categorical(logits=full_logits)
        values   = self._compute_values(node_h, pyg_batch, z_critic_flat, obs, agent_id_flat)
        return values, dist.log_prob(actions), dist.entropy()


class GraphCTFMAPPOPolicy_v4_hybrid(
    _MAPPOIndependentActorMixin,
    GraphCTFNeighborBatchPolicy_SubgraphCompatible_v4_hybrid,
):
    """
    MAPPO on the v4_hybrid backbone — independent actor + DeepSets centralized critic.
    This is the class used by the iter1_blue_br_best.zip checkpoint.
    """

    def __init__(self, *args, use_attention: bool = False, use_gru: bool = False,
                 n_agents: int = 2, **kwargs):
        super().__init__(*args, use_attention=use_attention, **kwargs)
        D = self.hidden_dim
        del self.node_mlp
        del self.extra_action_head
        del self.value_head
        self._mappo_independent_setup(D, use_gru, n_agents)


def load_policy_safe(path: str, device=None):
    """Load a MAPPO checkpoint safely for inference on any machine.

    SB3 pickles schedule callables (lr_schedule, clip_range, clip_range_vf) with
    cloudpickle; if they were custom closures on the training machine, unpickling
    fails on a different machine.  We override all three with safe constants.

    policy_class is forced explicitly to avoid loading a stale class from a prior
    submission's graph_policy.py that might be on the organizer's machine.
    """
    lp  = path[:-4] if path.endswith('.zip') else path
    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    _customs = {
        "lr_schedule"      : lambda _: 3e-4,
        "learning_rate"    : lambda _: 3e-4,
        "clip_range"       : lambda _: 0.2,
        "clip_range_vf"    : lambda _: 0.2,
        "policy_class"     : GraphCTFMAPPOPolicy_v4_hybrid,
        "observation_space": _OBS_SPACE,
        "action_space"     : _ACTION_SPACE,
    }
    try:
        return PPO.load(lp, device=dev, custom_objects=_customs)
    except Exception as e:
        raise RuntimeError(
            f"load_policy_safe: could not load '{lp}'.\n"
            f"  error: {e}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# Graph feature computation (identical to r2)
# ═════════════════════════════════════════════════════════════════════════════

def _compute_graph_features(nx_graph, flag_hyp_node_ids, blue_flag_node_id,
                             sensing_radius_utm=None):
    G = nx.Graph(nx_graph)
    G.remove_edges_from(nx.selfloop_edges(G))

    nodes_sorted = sorted(G.nodes())
    N            = len(nodes_sorted)
    node_to_idx  = {n: i for i, n in enumerate(nodes_sorted)}
    idx_to_node  = np.array(nodes_sorted, dtype=np.int32)

    flag_hyp_ids = list(flag_hyp_node_ids)
    n_hyp        = len(flag_hyp_ids)

    max_deg = max(G.degree(n) for n in nodes_sorted)
    MAX_DEG = max(max_deg, _MAX_DEGREE)

    nbr_matrix   = np.zeros((N, MAX_DEG), dtype=np.int32)
    nbr_mask_mat = np.zeros((N, MAX_DEG), dtype=np.float32)
    for i, n in enumerate(nodes_sorted):
        nbrs = sorted(node_to_idx[nb] for nb in G.neighbors(n))
        k = len(nbrs)
        nbr_matrix[i, :k]   = nbrs
        nbr_mask_mat[i, :k] = 1.0

    path_lengths   = dict(nx.all_pairs_shortest_path_length(G))
    graph_diameter = max(max(d.values()) for d in path_lengths.values())

    apsp = np.zeros((N, N), dtype=np.float32)
    for i, ni in enumerate(nodes_sorted):
        for j, nj in enumerate(nodes_sorted):
            apsp[i, j] = path_lengths[ni].get(nj, graph_diameter) / graph_diameter

    flag_dists = np.zeros((n_hyp, N), dtype=np.float32)
    for fi, fn in enumerate(flag_hyp_ids):
        spl = nx.single_source_shortest_path_length(G, fn)
        for j, nj in enumerate(nodes_sorted):
            flag_dists[fi, j] = spl.get(nj, graph_diameter) / graph_diameter

    blue_spl = nx.single_source_shortest_path_length(G, blue_flag_node_id)
    dist_to_blue_flag = np.array(
        [blue_spl.get(n, graph_diameter) / graph_diameter for n in nodes_sorted],
        dtype=np.float32)

    has_utm = all('x' in G.nodes[n] and 'y' in G.nodes[n] for n in nodes_sorted)
    is_frontier       = np.zeros(N, dtype=np.float32)
    frontier_res_mass = np.zeros(N, dtype=np.float32)

    if has_utm and sensing_radius_utm is not None:
        xs      = np.array([G.nodes[n]['x'] for n in nodes_sorted], dtype=np.float64)
        ys      = np.array([G.nodes[n]['y'] for n in nodes_sorted], dtype=np.float64)
        flag_xy = np.array([[G.nodes[fn]['x'], G.nodes[fn]['y']]
                            for fn in flag_hyp_ids], dtype=np.float64)
        for i in range(N):
            dists  = np.sqrt((xs[i] - flag_xy[:, 0])**2 + (ys[i] - flag_xy[:, 1])**2)
            within = (dists <= sensing_radius_utm).sum()
            if within > 0:
                is_frontier[i]       = 1.0
                frontier_res_mass[i] = within / n_hyp
    else:
        for fi, fn in enumerate(flag_hyp_ids):
            spl = nx.single_source_shortest_path_length(G, fn, cutoff=3)
            for nd in spl:
                ni = node_to_idx[nd]
                is_frontier[ni]       = 1.0
                frontier_res_mass[ni] = min(frontier_res_mass[ni] + 1.0 / n_hyp, 1.0)

    frontier_node_ids         = [nodes_sorted[i] for i in range(N) if is_frontier[i] > 0]
    dist_to_nearest_frontier  = np.full(N, float(graph_diameter), dtype=np.float32)
    for fn in frontier_node_ids:
        spl = nx.single_source_shortest_path_length(G, fn)
        for j, nj in enumerate(nodes_sorted):
            d = spl.get(nj, graph_diameter)
            if d < dist_to_nearest_frontier[j]:
                dist_to_nearest_frontier[j] = d
    dist_to_nearest_frontier /= graph_diameter

    structural_degree = np.array(
        [G.degree(n) / MAX_DEG for n in nodes_sorted], dtype=np.float32)

    max_ego_nodes, max_ego_edges = 0, 0
    for n in nodes_sorted:
        ego = nx.ego_graph(G, n, radius=_EGO_MAX_HOPS)
        max_ego_nodes = max(max_ego_nodes, ego.number_of_nodes())
        max_ego_edges = max(max_ego_edges, 2 * ego.number_of_edges())

    ego_edge_index      = np.zeros((N, 2, max_ego_edges), dtype=np.int32)
    ego_num_vis_nodes   = np.zeros(N, dtype=np.int32)
    ego_num_vis_edges   = np.zeros(N, dtype=np.int32)
    ego_agent_local_idx = np.zeros(N, dtype=np.int32)
    ego_neighbor_local  = np.zeros((N, MAX_DEG), dtype=np.int32)
    ego_neighbor_mask   = np.zeros((N, MAX_DEG), dtype=np.float32)
    ego_hops_to_ego     = np.zeros((N, max_ego_nodes), dtype=np.float32)
    ego_local_to_global = np.zeros((N, max_ego_nodes), dtype=np.int32)

    for gi, gn in enumerate(nodes_sorted):
        ego       = nx.ego_graph(G, gn, radius=_EGO_MAX_HOPS)
        ego_nodes = sorted(ego.nodes())
        l2g = {li: node_to_idx[en] for li, en in enumerate(ego_nodes)}
        g2l = {node_to_idx[en]: li for li, en in enumerate(ego_nodes)}
        nv  = len(ego_nodes)

        ego_num_vis_nodes[gi]   = nv
        ego_agent_local_idx[gi] = g2l[gi]

        hop_dict = nx.single_source_shortest_path_length(ego, gn)
        for li, en in enumerate(ego_nodes):
            ego_hops_to_ego[gi, li]     = hop_dict.get(en, _EGO_MAX_HOPS)
            ego_local_to_global[gi, li] = node_to_idx[en]

        edges    = list(ego.edges())
        bi_edges = [(g2l[node_to_idx[u]], g2l[node_to_idx[v]])
                    for u, v in edges
                    if node_to_idx[u] in g2l and node_to_idx[v] in g2l]
        bi_edges += [(v, u) for u, v in bi_edges]
        ne = len(bi_edges)
        ego_num_vis_edges[gi] = ne
        if ne > 0:
            src, dst = zip(*bi_edges)
            ego_edge_index[gi, 0, :ne] = src
            ego_edge_index[gi, 1, :ne] = dst

        nbrs_local = sorted(g2l[node_to_idx[nb]]
                            for nb in G.neighbors(gn)
                            if node_to_idx[nb] in g2l)
        k = len(nbrs_local)
        ego_neighbor_local[gi, :k] = nbrs_local
        ego_neighbor_mask[gi, :k]  = 1.0

    GRAPH = {
        'idx_to_node':          idx_to_node,
        'node_to_idx':          node_to_idx,
        'nbr_matrix':           nbr_matrix,
        'nbr_mask':             nbr_mask_mat,
        'apsp':                 apsp,
        'flag_dists':           flag_dists,
        'dist_to_blue_flag':    dist_to_blue_flag,
        'is_frontier':          is_frontier,
        'frontier_res_mass':    frontier_res_mass,
        'dist_to_nearest_fron': dist_to_nearest_frontier,
        'structural_degree':    structural_degree,
        'graph_diameter':       float(graph_diameter),
        'max_degree':           int(MAX_DEG),
        'max_ego_nodes':        int(max_ego_nodes),
        'max_ego_edges':        int(max_ego_edges),
        'blue_flag_node_idx':   node_to_idx[blue_flag_node_id],
        'flag_hyp_idxs':        np.array([node_to_idx[fn] for fn in flag_hyp_ids],
                                         dtype=np.int32),
    }
    EGO = {
        'edge_index':      ego_edge_index,
        'num_vis_nodes':   ego_num_vis_nodes,
        'num_vis_edges':   ego_num_vis_edges,
        'agent_local_idx': ego_agent_local_idx,
        'neighbor_local':  ego_neighbor_local,
        'neighbor_mask':   ego_neighbor_mask,
        'hops_to_ego':     ego_hops_to_ego,
        'local_to_global': ego_local_to_global,
    }
    return GRAPH, EGO


def _activate_graph(nx_graph, flag_hyp_node_ids=None, blue_flag_node_id=None,
                    sensing_radius_utm=None):
    global _GRAPH, _EGO, _FLAG_HYP_IDXS, _DEFAULT_NODE_IDX

    node_key  = frozenset(nx_graph.nodes())
    hyp_key   = frozenset(flag_hyp_node_ids) if flag_hyp_node_ids else frozenset()
    cache_key = (node_key, hyp_key)

    if cache_key in _GRAPH_CACHE:
        _GRAPH, _EGO      = _GRAPH_CACHE[cache_key]
        _FLAG_HYP_IDXS    = _GRAPH['flag_hyp_idxs']
        _DEFAULT_NODE_IDX = list(_GRAPH['node_to_idx'].values())[0]
        return

    cfg = _FLAG_HYP_CONFIG.get(node_key)
    if cfg is not None:
        flag_hyp_node_ids, blue_flag_node_id, sensing_radius_utm = cfg
    else:
        G_und = nx.Graph(nx_graph)
        G_und.remove_edges_from(nx.selfloop_edges(G_und))
        if flag_hyp_node_ids is None:
            cc = nx.closeness_centrality(G_und)
            flag_hyp_node_ids = tuple(sorted(cc, key=cc.get, reverse=True)[:3])
            print(f"[mit_atk_r3] WARNING: unknown map, using heuristic flag hypotheses "
                  f"{flag_hyp_node_ids}.")
        if blue_flag_node_id is None:
            # Blue's home base is far from Red's flag hypotheses.
            # Pick the node with maximum mean hop-distance from all flag hypotheses.
            all_nodes = list(G_und.nodes())
            flag_spls = [nx.single_source_shortest_path_length(G_und, fn)
                         for fn in flag_hyp_node_ids]
            blue_flag_node_id = max(
                all_nodes,
                key=lambda n: sum(spl.get(n, 0) for spl in flag_spls)
            )
            print(f"[mit_atk_r3] WARNING: unknown map, using heuristic blue flag node "
                  f"{blue_flag_node_id} (furthest from flag hypotheses).")
        if sensing_radius_utm is None:
            sensing_radius_utm = _FLAG_SENSE_UTM_M

    print(f"[mit_atk_r3] Computing graph features for new map "
          f"({len(node_key)} nodes) — this runs once per map …")
    G_feat, E_feat = _compute_graph_features(
        nx_graph, flag_hyp_node_ids, blue_flag_node_id, sensing_radius_utm)
    _GRAPH_CACHE[cache_key] = (G_feat, E_feat)
    _GRAPH, _EGO      = G_feat, E_feat
    _FLAG_HYP_IDXS    = _GRAPH['flag_hyp_idxs']
    _DEFAULT_NODE_IDX = list(_GRAPH['node_to_idx'].values())[0]
    print(f"[mit_atk_r3] Graph features computed and cached.")


def _ensure_loaded(n_blue):
    """Load the checkpoint for n_blue agents if not already done."""
    if n_blue in _POLICIES:
        return

    ckpt = _CKPT_PATHS.get(n_blue)
    if ckpt is None or not os.path.exists(ckpt):
        available = [k for k, v in _CKPT_PATHS.items() if os.path.exists(v)]
        if not available:
            raise FileNotFoundError(
                f"[mit_atk_r3] No checkpoint found. Expected: {list(_CKPT_PATHS.values())}")
        fallback = min(available, key=lambda k: abs(k - n_blue))
        print(f"[mit_atk_r3] WARNING: no {n_blue}-agent checkpoint found, "
              f"falling back to {fallback}-agent checkpoint.")
        _ensure_loaded(fallback)
        _MODELS[n_blue]   = _MODELS[fallback]
        _POLICIES[n_blue] = _POLICIES[fallback]
        return

    model  = load_policy_safe(_CKPT_PATHS[n_blue], device=_DEVICE)
    policy = model.policy
    policy.eval()
    _MODELS[n_blue]   = model
    _POLICIES[n_blue] = policy
    print(f"[mit_atk_r3] Loaded {n_blue}-agent checkpoint "
          f"(policy.n_agents={getattr(policy, 'n_agents', '?')}).")


# ═════════════════════════════════════════════════════════════════════════════
# Obs builder (identical to r2, with dim-5 and dim-16 fixes)
# ═════════════════════════════════════════════════════════════════════════════

def _build_obs(agent_node_id, agent_idx,
               own_positions, opp_positions,
               flag_known, confirmed_flag_node_id,
               active_flag_dist_idx):
    G, E = _GRAPH, _EGO
    node_to_idx = G['node_to_idx']

    ni = node_to_idx.get(agent_node_id, _DEFAULT_NODE_IDX)

    num_vis_n = int(E['num_vis_nodes'][ni])
    num_vis_e = int(E['num_vis_edges'][ni])
    max_en    = G['max_ego_nodes']
    max_ee    = G['max_ego_edges']
    agent_li  = int(E['agent_local_idx'][ni])
    l2g       = E['local_to_global'][ni]

    opp_global_idxs = set(node_to_idx.get(p, _DEFAULT_NODE_IDX)
                          for p in opp_positions.values())
    tm_global_idxs  = set(node_to_idx.get(p, _DEFAULT_NODE_IDX)
                          for idx, p in own_positions.items()
                          if idx != agent_idx)

    flag_gi = (node_to_idx.get(confirmed_flag_node_id, _DEFAULT_NODE_IDX)
               if confirmed_flag_node_id is not None else None)

    away_flag_dist_row = G['flag_dists'][active_flag_dist_idx]
    apsp = G['apsp']

    min_opp_d = 1.0
    for r_gi in opp_global_idxs:
        d = float(apsp[ni, r_gi])
        if d < min_opp_d:
            min_opp_d = d

    min_tm_d = 1.0
    for tm_gi in tm_global_idxs:
        d = float(apsp[ni, tm_gi])
        if d < min_tm_d:
            min_tm_d = d

    x = np.zeros((max_en, 17), dtype=np.float32)
    for li in range(num_vis_n):
        gi = int(l2g[li])
        x[li, 0]  = float(li == agent_li)
        x[li, 1]  = float(G['is_frontier'][gi])
        x[li, 2]  = float(G['frontier_res_mass'][gi])
        x[li, 3]  = float(not (x[li, 0] or x[li, 1]))
        x[li, 4]  = float(flag_known and flag_gi is not None and gi == flag_gi)
        x[li, 5]  = float(E['hops_to_ego'][ni, li]) / G['graph_diameter']  # normalized
        x[li, 6]  = float(G['dist_to_blue_flag'][gi])
        x[li, 7]  = float(away_flag_dist_row[gi]) if flag_known else 0.0
        x[li, 9]  = float(G['dist_to_nearest_fron'][gi])
        x[li, 10] = float(gi in opp_global_idxs)
        x[li, 11] = float(gi in tm_global_idxs)
        x[li, 14] = float(G['structural_degree'][gi])
        x[li, 15] = 1.0

    x[agent_li, 8]  = float(flag_known)
    x[agent_li, 12] = min_opp_d
    x[agent_li, 13] = min_tm_d
    x[agent_li, 16] = 1.0   # is_alive: ego node only

    edge_index = E['edge_index'][ni].copy().astype(np.int64)
    edge_attr  = np.zeros((max_ee, 1), dtype=np.float32)

    node_vis_mask = np.zeros(max_en, dtype=np.float32)
    node_vis_mask[:num_vis_n] = 1.0
    edge_vis_mask = np.zeros(max_ee, dtype=np.float32)
    edge_vis_mask[:num_vis_e] = 1.0

    nbr_mask    = E['neighbor_mask'][ni]
    action_mask = np.zeros(_NUM_ACTIONS, dtype=np.float32)
    action_mask[:_MAX_DEGREE] = nbr_mask
    action_mask[_MAX_DEGREE]  = 1.0

    return {
        'x':                    x,
        'edge_index':           edge_index,
        'edge_attr':            edge_attr,
        'node_visibility_mask': node_vis_mask,
        'edge_visibility_mask': edge_vis_mask,
        'agent_node_local_idx': agent_li,
        'neighbor_local_idx':   E['neighbor_local'][ni].copy(),
        'neighbor_mask':        nbr_mask.copy(),
        'action_mask':          action_mask,
        'num_visible_nodes':    num_vis_n,
        'num_visible_edges':    num_vis_e,
    }


# ═════════════════════════════════════════════════════════════════════════════
# MAPPO obs injection  (n_blue is a parameter)
# ═════════════════════════════════════════════════════════════════════════════

def _inject_mappo_obs(obs, agent_idx, prev_obs, n_blue):
    aug = dict(obs)
    aug['agent_id'] = np.array([agent_idx], dtype=np.int64)

    all_tm     = [j for j in range(n_blue) if j != agent_idx]
    policy_n   = getattr(_POLICIES.get(n_blue), 'n_agents', n_blue)
    n_tm_slots = policy_n - 1
    for t in range(n_tm_slots):
        tm_idx = all_tm[t % len(all_tm)] if all_tm else agent_idx
        tm_obs = prev_obs.get(tm_idx, obs)
        for key in _TM_KEYS:
            if key in tm_obs:
                aug[f'teammate_{t}_{key}'] = tm_obs[key]
        aug[f'teammate_{t}_agent_id'] = np.array([tm_idx], dtype=np.int64)

    return aug


# ═════════════════════════════════════════════════════════════════════════════
# PyTorch forward pass  (n_blue selects the right policy)
# ═════════════════════════════════════════════════════════════════════════════

def _obs_to_tensors(obs_dict):
    out = {}
    for key, val in obs_dict.items():
        if isinstance(val, np.ndarray):
            out[key] = torch.as_tensor(val, device=_DEVICE).unsqueeze(0)
        elif isinstance(val, torch.Tensor):
            out[key] = val.unsqueeze(0).to(_DEVICE)
        else:
            out[key] = torch.tensor([val], device=_DEVICE)
    return out


def _policy_forward(obs, agent_idx, h_actor, h_critic, n_blue):
    p = _POLICIES[n_blue]

    p._h_buf = (torch.as_tensor(h_actor, dtype=torch.float32, device=_DEVICE).unsqueeze(0)
                if h_actor is not None else None)
    p._h_buf_critic = (torch.as_tensor(h_critic, dtype=torch.float32, device=_DEVICE).unsqueeze(0)
                       if h_critic is not None else None)

    with torch.no_grad():
        actions, _, _ = p.forward(_obs_to_tensors(obs), deterministic=True)

    action_idx   = int(actions.cpu().item())
    h_actor_new  = p._h_buf[0].cpu().numpy()
    h_critic_new = (p._h_buf_critic[0].cpu().numpy()
                    if p._h_buf_critic is not None
                    else np.zeros(_GRU_H, dtype=np.float32))

    return action_idx, h_actor_new, h_critic_new


# ═════════════════════════════════════════════════════════════════════════════
# Action decoding
# ═════════════════════════════════════════════════════════════════════════════

def _action_to_node(action_idx, agent_node_id):
    G  = _GRAPH
    ni = G['node_to_idx'].get(agent_node_id, _DEFAULT_NODE_IDX)
    nbr = G['nbr_matrix'][ni]
    msk = G['nbr_mask'][ni]
    if action_idx < _MAX_DEGREE and msk[action_idx] > 0:
        return int(G['idx_to_node'][nbr[action_idx]])
    return agent_node_id


def _agent_idx(agent_name):
    return int(agent_name.rsplit('_', 1)[1])


# ═════════════════════════════════════════════════════════════════════════════
# Contest API
# ═════════════════════════════════════════════════════════════════════════════

def attacker_strategy(state):
    agent_ctrl   = state['agent_controller']
    agent_name   = agent_ctrl.name
    current_pos  = int(state['curr_pos'])
    current_time = int(state['time'])
    team_cache   = agent_ctrl.team_cache
    sensors      = state['sensor']
    own_team     = agent_ctrl.team.lower()

    agent_idx = _agent_idx(agent_name)

    # ── Episode reset ──────────────────────────────────────────────────────
    if current_time == 0 or 'init_done' not in team_cache:
        agent_sensor = sensors.get('agent', (None, {}))[1]
        n_blue = sum(1 for nm in agent_sensor if nm.lower().startswith(own_team))
        if n_blue == 0:
            n_blue = 3  # fallback if sensor absent in test context

        _ensure_loaded(n_blue)

        team_cache.clear()
        team_cache['init_done']  = True
        team_cache['n_blue']     = n_blue
        team_cache['own_pos']    = {}
        team_cache['conf_flags'] = set()
        team_cache['flag_known'] = False
        team_cache['prev_obs']   = {}
        team_cache['h_actor']    = {}
        team_cache['h_critic']   = {}

        nx_graph = sensors['global_map'][1]['graph']
        candidate_hyps = None
        if 'candidate_flag' in sensors:
            cands = sensors['candidate_flag'][1].get('candidate_flags', [])
            if cands:
                candidate_hyps = [int(n) for n in cands]
        _activate_graph(nx_graph, flag_hyp_node_ids=candidate_hyps)

    n_blue = team_cache['n_blue']

    # ── Parse sensors ─────────────────────────────────────────────────────
    discovered_flags = set()
    opp_positions    = {}

    if 'agent' in sensors:
        for nm, pos in sensors['agent'][1].items():
            if not nm.lower().startswith(own_team):
                try:
                    ri = int(nm.rsplit('_', 1)[1])
                    opp_positions[ri] = int(pos)
                except (ValueError, IndexError):
                    pass

    if 'egocentric_flag' in sensors:
        detected = sensors['egocentric_flag'][1].get('detected_flags', [])
        discovered_flags = set(int(n) for n in detected)

    # ── Update shared knowledge ────────────────────────────────────────────
    team_cache['own_pos'][agent_idx] = current_pos
    team_cache['conf_flags'].update(discovered_flags)
    if discovered_flags:
        team_cache['flag_known'] = True

    flag_known = team_cache['flag_known']
    conf       = team_cache['conf_flags']
    conf_flag  = next(iter(conf)) if conf else None

    if conf_flag is not None and conf_flag in _GRAPH['node_to_idx']:
        gi       = _GRAPH['node_to_idx'][conf_flag]
        hyp_idxs = [int(x) for x in _FLAG_HYP_IDXS]
        act_fi   = hyp_idxs.index(gi) if gi in hyp_idxs else 0
    else:
        act_fi = 0

    # ── Build obs ─────────────────────────────────────────────────────────
    obs = _build_obs(
        agent_node_id          = current_pos,
        agent_idx              = agent_idx,
        own_positions          = team_cache['own_pos'],
        opp_positions          = opp_positions,
        flag_known             = flag_known,
        confirmed_flag_node_id = conf_flag,
        active_flag_dist_idx   = act_fi,
    )

    # ── Inject teammate obs + agent_id for MAPPO critic ───────────────────
    aug_obs = _inject_mappo_obs(obs, agent_idx, team_cache['prev_obs'], n_blue)

    # ── Forward pass ──────────────────────────────────────────────────────
    h_actor  = team_cache['h_actor'].get(agent_idx)
    h_critic = team_cache['h_critic'].get(agent_idx)

    action_idx, h_actor_new, h_critic_new = _policy_forward(
        aug_obs, agent_idx, h_actor, h_critic, n_blue)

    # ── Save state ────────────────────────────────────────────────────────
    team_cache['h_actor'][agent_idx]  = h_actor_new
    team_cache['h_critic'][agent_idx] = h_critic_new
    team_cache['prev_obs'][agent_idx] = obs

    # ── Decode action → node ──────────────────────────────────────────────
    target_node = _action_to_node(action_idx, current_pos)
    state['action'] = target_node        # old engine: reads state['action']
    return str(target_node)              # new engine: requires str return value


def map_strategy(agent_config):
    return {name: attacker_strategy for name in agent_config}
