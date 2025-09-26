from typing import Any, Dict, List, Tuple, Set
from typeguard import typechecked
import networkx as nx

from lib.core.console import info, warning, error, debug
from lib.game.agent_engine import AgentEngine


@typechecked
class InteractionEngine:
    """
    Per-tick interaction resolution for red vs blue teams.
    Operates on an AgentEngine instance and a full config dict (from ConfigLoader).
    - Capture: agent vs opponent flags (graph distance <= agent.capture_radius)
    - Combat: red vs blue (graph distance <= min(tagging_radii))
    - Order: capture-first or combat-first (read from config, safe fallbacks)
    """

    def __init__(self, agent_engine: AgentEngine, G: nx.Graph, config: Dict[str, Any]) -> None:
        debug("Initializing InteractionEngine")
        warning("Current interaction engine only acceptps red-vs-blue CTF-style games.")
        
        self.engine = agent_engine
        self.ctx = agent_engine.ctx
        self.G = G
        self.config = config

        # per-step scratch
        self._processed: Set[str] = set()
        self._red_killed = 0
        self._blue_killed = 0
        self._capture_details: List[Tuple[str, str, Any]] = []  # (agent_name, team, flag_node)
        self._tagging_details: List[Tuple[str, str, str]] = []  # (red_name, blue_name, outcome)
        debug("InteractionEngine initialized")

    # ------------------------------ Public API ------------------------------

    def step(self, time: int) -> Tuple[int, int, int, int, int, int, List[Tuple[str, str, Any]], List[Tuple[str, str, str]]]:
        """
        Resolve one timestep of interactions.

        Returns:
            (red_captures, blue_captures,
             red_killed, blue_killed,
             remaining_red, remaining_blue,
             capture_details, tagging_details)
        """
        debug(f"Processing interactions for timestep {time}")
        self._processed.clear()
        self._red_killed = 0
        self._blue_killed = 0
        self._capture_details.clear()
        self._tagging_details.clear()

        order = self._prioritize_mode()
        debug(f"Interaction order: {order}")
        if order == "capture":
            red_caps, blue_caps = self._process_flag_captures(time)
            self._process_combat_interactions()
        else:
            self._process_combat_interactions()
            red_caps, blue_caps = self._process_flag_captures(time)

        remaining_red = sum(1 for a in self.ctx.agent.create_iter() if getattr(a, "team", "") == "red")
        remaining_blue = sum(1 for a in self.ctx.agent.create_iter() if getattr(a, "team", "") == "blue")

        debug(f"Timestep {time} complete: {red_caps} red captures, {blue_caps} blue captures, {self._red_killed} red killed, {self._blue_killed} blue killed")
        return (red_caps, blue_caps, self._red_killed, self._blue_killed, remaining_red, remaining_blue, self._capture_details, self._tagging_details)

    # --------------------------- Capture resolution --------------------------

    def _process_flag_captures(self, time: int) -> Tuple[int, int]:
        debug("Processing flag captures")
        red_flags, blue_flags = self._get_flags()
        red_captures = 0
        blue_captures = 0

        # red captures blue flags
        if blue_flags is not None:
            for r in self._iter_team("red"):
                got, captured = self._try_capture(r, blue_flags, "red", "blue", time)
                red_captures += got
                if captured:
                    break

        # blue captures red flags
        if red_flags is not None:
            for b in self._iter_team("blue"):
                got, captured = self._try_capture(b, red_flags, "blue", "red", time)
                blue_captures += got
                if captured:
                    break

        debug(f"Flag captures complete: {red_captures} red, {blue_captures} blue")
        return red_captures, blue_captures

    def _try_capture(self, agent: Any, opponent_flags: List[Any], agent_team: str, flag_team: str, time: int) -> Tuple[int, bool]:
        """Attempt to capture any opponent flag; return (captures, did_capture_any)."""
        debug(f"Checking capture attempts for {agent_team} agent {agent.name}")
        for flag_node in opponent_flags:
            try:
                dist = nx.shortest_path_length(self.G, agent.current_node_id, flag_node)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            ctrl = self.engine.get_validated_agent(agent.name)
            if not ctrl:
                warning(f"Agent '{agent.name}' failed validation during capture attempt")
                continue
            cr = getattr(ctrl, "capture_radius", 0)
            if dist <= cr:
                info(f"{agent_team.title()} {agent.name} captured {flag_team} flag {flag_node} at t={time}")
                if self._handle_interaction(agent, self._capture_action()):
                    self._capture_details.append((agent.name, agent_team, flag_node))
                    return 1, True
        return 0, False

    # ---------------------------- Combat resolution -------------------------

    def _process_combat_interactions(self) -> None:
        """Pairwise combat between red and blue agents (unprocessed only)."""
        debug("Processing combat interactions")
        reds = [a for a in self.ctx.agent.create_iter() if getattr(a, "team", "") == "red" and a.name not in self._processed]
        blues = [b for b in self.ctx.agent.create_iter() if getattr(b, "team", "") == "blue" and b.name not in self._processed]

        debug(f"Combat candidates: {len(reds)} red agents, {len(blues)} blue agents")
        combat_count = 0
        for r in reds:
            for b in blues:
                if r.name in self._processed:
                    break
                if b.name in self._processed:
                    continue
                if self._agents_in_tagging_range(r, b):
                    debug(f"Combat between {r.name} and {b.name}")
                    outcome = self._resolve_original_combat(r, b)
                    self._tagging_details.append((r.name, b.name, outcome))
                    combat_count += 1
                    # after a resolution, one/both may be processed; move to next red
                    if r.name in self._processed:
                        break
                    if b.name in self._processed:
                        continue

        debug(f"Combat interactions complete: {combat_count} combats resolved")

    def _agents_in_tagging_range(self, r: Any, b: Any) -> bool:
        """Check if red r and blue b are within tagging range (min of radii)."""
        try:
            if r.name in self._processed or b.name in self._processed:
                return False
            
            r_ctrl = self.engine.get_validated_agent(r.name)
            b_ctrl = self.engine.get_validated_agent(b.name)
            
            if not r_ctrl or not b_ctrl:
                return False
            rr = getattr(r_ctrl, "tagging_radius", 0)
            br = getattr(b_ctrl, "tagging_radius", 0)
            rng = min(rr, br)
            dist = nx.shortest_path_length(self.G, r.current_node_id, b.current_node_id)
            return dist <= rng
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return False

    def _resolve_original_combat(self, r: Any, b: Any) -> str:
        """
        Resolve combat based on a tagging action from config.
        Supported:
          - 'both_kill'     → kill both
          - 'both_respawn'  → respawn both
          - 'kill' (or other single action) → apply to both; kills counted if action == 'kill'
          - list[float]     → probabilistic [P(red dies), P(blue dies), P(both die)]
        """
        action = self._tagging_action()
        debug(f"Resolving combat between {r.name} and {b.name} with action: {action}")

        if isinstance(action, list):
            return self._resolve_probabilistic(r, b, action)

        if action == "both_kill":
            red_dead = self._handle_interaction(r, "kill")
            blue_dead = self._handle_interaction(b, "kill")
            if red_dead: self._red_killed += 1
            if blue_dead: self._blue_killed += 1
            return "both_killed"

        if action == "both_respawn":
            self._handle_interaction(r, "respawn")
            self._handle_interaction(b, "respawn")
            return "both_respawned"

        # Single action applied to both (e.g., "kill", "respawn", or custom)
        red_success = self._handle_interaction(r, action)
        blue_success = self._handle_interaction(b, action)
        if action == "kill":
            self._red_killed += 1 if red_success else 0
            self._blue_killed += 1 if blue_success else 0
        return f"both_{action}"

    def _resolve_probabilistic(self, r: Any, b: Any, probs: List[float]) -> str:
        """Probabilistic combat outcome with probs = [P(red dies), P(blue dies), P(both die)]."""
        import random

        debug(f"Resolving probabilistic combat between {r.name} and {b.name} with probabilities: {probs}")
        s = float(sum(probs))
        p = [x / s for x in probs] if s > 0 else [0.0, 0.0, 0.0]
        u = random.random()

        if u < p[2]:  # both die
            red_dead = self._handle_interaction(r, "kill")
            blue_dead = self._handle_interaction(b, "kill")
            self._red_killed += 1 if red_dead else 0
            self._blue_killed += 1 if blue_dead else 0
            return "both_killed"
        elif u < p[2] + p[0]:  # red dies
            red_dead = self._handle_interaction(r, "kill")
            self._red_killed += 1 if red_dead else 0
            return "red_killed"
        elif u < p[2] + p[0] + p[1]:  # blue dies
            blue_dead = self._handle_interaction(b, "kill")
            self._blue_killed += 1 if blue_dead else 0
            return "blue_killed"
        else:
            return "no_casualties"

    # ------------------------------ Utilities -------------------------------

    def _iter_team(self, team: str):
        for ctrl in self.engine.get_active_agents_by_team(team):
            if ctrl.name not in self._processed:
                # Need to get the gamms agent from the controller
                yield ctrl.gamms_agent

    def _handle_interaction(self, agent: Any, action: str) -> bool:
        """
        Apply an action to an agent and mark it processed.
        Supported actions: "kill", "respawn", or any custom no-op string.
        """
        debug(f"Handling interaction: {action} for agent {agent.name}")
        self._processed.add(agent.name)

        if action == "kill":
            # Delegate to AgentEngine.kill_agent so it handles ctx cleanup + engine state
            ok = bool(self.engine.kill_agent(agent.name))
            if ok:
                debug(f"Agent '{agent.name}' killed via AgentEngine.kill_agent()")
            else:
                warning(f"kill_agent('{agent.name}') returned False (already dead or not found)")
            return ok

        elif action == "respawn":
            ctrl = self.engine.get_validated_agent(agent.name)
            if not ctrl:
                warning(f"Respawn failed for '{agent.name}': agent validation failed")
                return False
                
            start_node = getattr(ctrl, "start_node_id", None)
            if start_node is None:
                warning(f"Respawn skipped for '{agent.name}': no start_node_id on controller")
                return False
            
            # Validate start node exists in graph
            if start_node not in self.G:
                error(f"Respawn failed for '{agent.name}': start_node {start_node} not found in graph")
                return False
            
            # Use controller's update_position method for proper synchronization
            success = ctrl.update_position(start_node)
            if success:
                debug(f"Agent '{agent.name}' respawned at node {start_node}")
                return True
            else:
                warning(f"Respawn failed for '{agent.name}': update_position returned False")
                return False

        # Unknown action => treat as no-op success
        debug(f"Unknown action '{action}' for agent {agent.name}, treating as no-op")
        return True

    # ------------------------------ Config I/O -------------------------------

    def _get_flags(self) -> Tuple[List[Any], List[Any]]:
        """
        Read red/blue flag node lists from the full config.
        Tries modern keys first; also supports legacy alpha/beta keys as fallback.
        """
        game_rule = self.config.get("game", {}).get("game_rule", {})
        debug(f"Current game rule: {game_rule}")
        flags = self.config.get("flags", {})
        debug(f"Current flags: {flags}")
        if game_rule == "v2":
            red_flags = flags.get("red_flag_positions", [])
            blue_flags = flags.get("blue_flag_positions", [])
        elif game_rule == "v1.2":
            debug(f"Using game rule v1.2; interpreting 'real_positions' as blue flags and no red flags")
            red_flags = []
            blue_flags = flags.get("real_positions", [])
        return red_flags, blue_flags

    def _prioritize_mode(self) -> str:
        """Return 'capture' or 'combat' (default: 'capture')."""
        game = self.config.get("game", {})
        return game.get("interaction", {}).get("prioritize") or game.get("interaction_model", {}).get("prioritize") or game.get("prioritize") or "capture"

    def _capture_action(self) -> str:
        """Capture action (default: 'kill')."""
        game = self.config.get("game", {})
        return game.get("interaction", {}).get("capture") or game.get("interaction_model", {}).get("capture") or game.get("capture") or "kill"

    def _tagging_action(self):
        """
        Tagging action (default: 'both_kill').
        May be a string or a list[float] for probabilistic outcomes:
        [P(red dies), P(blue dies), P(both die)]
        """
        game = self.config.get("game", {})
        return game.get("interaction", {}).get("tagging") or game.get("interaction_model", {}).get("tagging") or game.get("tagging") or "both_kill"
