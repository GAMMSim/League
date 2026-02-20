from typing import Any, Dict, List, Optional, Tuple, Set, Union
from typeguard import typechecked
import networkx as nx
import traceback
import importlib
import os
import time

try:
    from lib.core.console import *
    from lib.core.apsp_cache import get_apsp_length_cache, get_cached_distance
    from lib.game.agent_engine import AgentEngine
    from lib.game.sensor_engine import SensorEngine
    from lib.game.interaction_engine import InteractionEngine
    from lib.game.visualization_engine_new import VisEngine
    from lib.core.logger import Logger
except ImportError:
    from ..core.console import *
    from ..core.apsp_cache import get_apsp_length_cache, get_cached_distance
    from ..game.agent_engine import AgentEngine
    from ..game.sensor_engine import SensorEngine
    from ..game.interaction_engine import InteractionEngine
    from .visualization_engine_new import VisEngine
    from ..core.logger import Logger


@typechecked
class GameEngine:
    """
    Main game engine that coordinates the entire game loop.
    Handles movement validation, strategy execution, interactions, payoffs, and termination.
    """

    def __init__(
        self,
        ctx: Any,
        graph: nx.Graph,
        config: Dict[str, Any],
        agent_engine: AgentEngine,
        interaction_engine: InteractionEngine,
        vis_engine: Optional[VisEngine] = None,
        logger: Optional[Logger] = None,
    ):
        self.ctx = ctx
        self.graph = graph
        self.config = config
        self.agent_engine = agent_engine
        self.interaction_engine = interaction_engine
        self.vis_engine = vis_engine
        self.logger = logger

        # Game state
        self.time_counter = 0
        self.is_running = False
        self.game_terminated = False

        # Game configuration
        self.game_config = config.get("game", {})
        self.max_time = self.game_config.get("max_time", 1000)

        # Payoff tracking
        self.red_payoff_accum = 0.0
        self.blue_payoff_accum = 0.0
        self.red_captures = 0
        self.blue_captures = 0
        self.red_killed = 0
        self.blue_killed = 0
        self.discovered_flags: Set[int] = set()
        self.discovered_flags_count = 0
        # Share the discovered_flags set with the interaction engine so it can
        # enforce discovery-before-capture (flags must be sensed before captured).
        self.interaction_engine._discovered_flags = self.discovered_flags

        # Payoff breakdown (for HUD)
        self.red_tag_penalty = 0.0
        self.red_capture_reward = 0.0
        self.red_discover_reward = 0.0
        self.step_payoff_breakdown = self._empty_step_payoff_breakdown()

        # Flag positions from config
        self.flag_config = self.config.get("flags", {})

        # Strategy storage
        self.strategies: Dict[str, Any] = {}
        self._agent_death_time: Dict[str, int] = {}
        self._first_tag_time: Optional[int] = None
        self._first_capture_time: Optional[int] = None
        self._first_discover_time: Optional[int] = None

    def _empty_step_payoff_breakdown(self) -> Dict[str, Dict[str, float]]:
        return {
            "red": {"capture": 0.0, "tag": 0.0, "discover": 0.0, "total": 0.0},
            "blue": {"capture": 0.0, "tag": 0.0, "discover": 0.0, "total": 0.0},
        }

    # ---------- Convenience / Builder APIs (new) ----------

    @classmethod
    def from_runtime(
        cls,
        ctx: Any,
        graph: nx.Graph,
        config: Dict[str, Any],
        agent_engine: AgentEngine,
        interaction_engine: InteractionEngine,
        vis_engine: Optional[VisEngine],
        logger: Optional[Logger],
    ) -> "GameEngine":
        """Construct directly from prebuilt subsystems."""
        return cls(ctx, graph, config, agent_engine, interaction_engine, vis_engine, logger)

    @staticmethod
    def load_strategies(
        red_strategy: Union[str, Any, None],
        blue_strategy: Union[str, Any, None],
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Accept module objects or import strings. Returns (red_module, blue_module).
        """

        def _ensure_module(maybe_mod: Union[str, Any, None]) -> Optional[Any]:
            if maybe_mod is None:
                return None
            if isinstance(maybe_mod, str):
                return importlib.import_module(maybe_mod)
            return maybe_mod

        return _ensure_module(red_strategy), _ensure_module(blue_strategy)

    @staticmethod
    def build_runtime(
        config: Dict[str, Any],
        graph: Optional[nx.Graph] = None,
        *,
        ctx: Optional[Any] = None,
        root_path: Optional[str] = None,
        log_name: Optional[str] = "result",
        vis_engine_kind: Optional[Any] = None,
        record: bool = False,
        vis: bool = True,
    ) -> Tuple[Any, nx.Graph, VisEngine, SensorEngine, AgentEngine, InteractionEngine, Optional[Logger]]:
        """
        Build the full runtime (ctx, graph, vis, sensors, agents, interaction, logger).
        If ctx/graph are missing, create them from config using repo utilities.
        If log_name is None, no logger will be created and no logging will occur.
        Recording is independent of logging.
        """
        # Lazy imports to avoid module import cycles when this file is library-imported
        from lib.config.config_loader import ConfigLoader  # not used here but kept to mirror original imports
        from lib.utils.file_utils import export_graph_config, get_directories
        from lib.core.logger import Logger as _Logger
        from lib.game.sensor_engine import SensorEngine

        # 1) Context & graph
        if ctx is None:
            import gamms  # local import to avoid hard dependency at import-time

            vis_cfg = config.get("visualization", {}) or {}
            w, h = vis_cfg.get("window_size", [1200, 800]), vis_cfg.get("window_size", [1200, 800])
            vis_kwargs = {
                "width": w[0] if isinstance(w, list) and len(w) > 0 else 1200,
                "height": w[1] if isinstance(w, list) and len(w) > 1 else 800,
            }
            if vis:
                engine_kind = vis_engine_kind or getattr(gamms.visual.Engine, "PYGAME", None)
            else:
                engine_kind = getattr(gamms.visual.Engine, "NO_VIS", None)
            ctx = gamms.create_context(vis_engine=engine_kind, vis_kwargs=vis_kwargs)
            
        # Create logger only if log_name is provided
        logger = _Logger(log_name) if log_name is not None else None
        
        # Recording is independent of logging
        if record:
            # Generate a recording filename with timestamp
            timestamp = int(time.time())
            recording_filename = f"recording_{timestamp}.ggr"
            ctx.record.start(path=recording_filename.replace(".ggr", ""))
            success(f"Recording enabled: {recording_filename}")

        if graph is None:
            import pathlib

            current_path = pathlib.Path(__file__).resolve()
            root = current_path.parent.parent.parent if root_path is None else pathlib.Path(root_path)
            dirs = get_directories(str(root))
            graph = export_graph_config(config, dirs)
        # Pre-warm shared APSP cache once for simulator internals.
        try:
            get_apsp_length_cache(graph)
        except Exception as e:
            warning(f"Failed to prebuild APSP distance cache: {e}")
        # Attach the graph to ctx if available
        try:
            ctx.graph.attach_networkx_graph(graph)  # type: ignore[attr-defined]
        except Exception:
            pass  # Some contexts may not expose this; safe to ignore

        # 2) Visualization
        vis_engine = VisEngine(ctx, config)

        # 3) Sensors
        sensor_engine = SensorEngine(ctx, config, graph)

        # 4) Agents
        agent_engine = AgentEngine(ctx, ["red", "blue"], vis_engine=vis_engine, sensor_engine=sensor_engine)
        agent_engine.create_agents_from_config(config)

        # 5) Interactions
        interaction_engine = InteractionEngine(agent_engine, graph, config)

        return ctx, graph, vis_engine, sensor_engine, agent_engine, interaction_engine, logger

    @classmethod
    def launch_from_files(
        cls,
        *,
        config_main: str = "output.yaml",
        extra_defs: Optional[str] = "config/game_config.yml",
        red_strategy: Union[str, Any, None] = "test_atk",
        blue_strategy: Union[str, Any, None] = "test_def",
        log_name: Optional[str] = "test_result",
        set_level: Optional["LogLevel"] = LogLevel.WARNING,
        record: bool = False,
        vis: bool = True,
    ) -> "GameEngine":
        """
        One-liner runner that mirrors the old __main__ behavior but keeps the call-site clean.
        Loads config, builds runtime, assigns strategies, sets up visuals, runs, and returns the engine.
        
        Args:
            log_name: Name for the log file. If None, no logging will occur.
            record: If True, record the game session to a .ggr file. Recording is independent of logging.
            vis: If True, use PYGAME visualization. If False, use NO_VIS mode.
        """
        # Lazily import loader/utilities here
        from lib.config.config_loader import ConfigLoader

        if set_level is not None:
            set_log_level(set_level)

        # Load config
        loader = ConfigLoader(config_main)
        if extra_defs:
            loader.load_extra_definitions(extra_defs, force=True)
        config = loader.config_data

        # Build runtime subsystems
        ctx, graph, vis_engine, sensor_engine, agent_engine, interaction_engine, logger = cls.build_runtime(
            config=config,
            graph=None,
            ctx=None,
            root_path=None,
            log_name=log_name,
            vis_engine_kind=None,
            record=record,
            vis=vis,
        )

        # Create the unified engine
        engine = cls.from_runtime(ctx, graph, config, agent_engine, interaction_engine, vis_engine, logger)

        # Visuals
        engine.setup_game_visuals(config.get("agents", {}))

        # Strategies
        red_mod, blue_mod = cls.load_strategies(red_strategy, blue_strategy)
        if red_mod or blue_mod:
            engine.assign_strategies(red_mod, blue_mod)

        if engine.logger:
            def _strategy_label(strategy_obj: Union[str, Any, None]) -> Optional[str]:
                if strategy_obj is None:
                    return None
                if isinstance(strategy_obj, str):
                    return strategy_obj
                return getattr(strategy_obj, "__name__", strategy_obj.__class__.__name__)

            try:
                metadata = engine.logger.get_metadata()
                metadata.update(
                    {
                        "config_main": str(config_main),
                        "extra_defs": str(extra_defs) if extra_defs else None,
                        "red_strategy": _strategy_label(red_strategy),
                        "blue_strategy": _strategy_label(blue_strategy),
                        "max_time": config.get("game", {}).get("max_time"),
                        "game_rule": config.get("game", {}).get("game_rule"),
                    }
                )
                engine.logger.set_metadata(metadata)
            except Exception as e:
                warning(f"Failed to set logger metadata: {e}")

        # Run
        engine.run_game()
        success(str(engine))  # prints nice summary via __str__

        if engine.logger:
            # Make sure the target directory exists; avoids write errors in the logger
            if hasattr(engine.logger, "path"):
                try:
                    os.makedirs(engine.logger.path, exist_ok=True)
                except Exception:
                    pass  # non-fatal; logger will throw if it truly cannot write

            # Write a JSON log file (auto-generated name)
            try:
                fname = engine.logger.write_to_file(format="json")  # JSON ensures broad compatibility
                if hasattr(engine.logger, "path"):
                    success(f"Log written to: {os.path.join(engine.logger.path, fname)}")
                else:
                    success(f"Log file: {fname}")
            except Exception as e:
                warning(f"Failed to write log file: {e}")
        return engine

    # ---------- Original functionality (unchanged logic) ----------

    def setup_game_visuals(self, agent_config: Dict[str, Dict[str, Any]]) -> None:
        if not self.vis_engine:
            return
        try:
            self.vis_engine.setup_agent_visuals(agent_config)
            self.vis_engine.create_flags(self.flag_config)
            self.vis_engine.create_agent_labels(agent_config)
            self.vis_engine.create_agent_sensor_circles()
            self.vis_engine.setup_hud()
            success("Game visuals initialized")
        except Exception as e:
            error(f"Failed to setup game visuals: {e}")

    def assign_strategies(self, red_strategy_module: Any, blue_strategy_module: Any) -> None:
        try:
            red_configs: Dict[str, Dict[str, Any]] = {}
            blue_configs: Dict[str, Dict[str, Any]] = {}

            for agent_name, agent_controller in self.agent_engine.agents.items():
                if agent_controller.team == "red":
                    red_configs[agent_name] = {"team": agent_controller.team, **agent_controller.to_dict()}
                elif agent_controller.team == "blue":
                    blue_configs[agent_name] = {"team": agent_controller.team, **agent_controller.to_dict()}

            if red_configs and red_strategy_module is not None:
                red_strategies = red_strategy_module.map_strategy(red_configs)
                self.strategies.update(red_strategies)

            if blue_configs and blue_strategy_module is not None:
                blue_strategies = blue_strategy_module.map_strategy(blue_configs)
                self.strategies.update(blue_strategies)

            for agent_name, strategy in self.strategies.items():
                if agent_name in self.agent_engine.agents:
                    agent_controller = self.agent_engine.agents[agent_name]
                    agent_controller.strategy = strategy
                    try:
                        if hasattr(agent_controller.gamms_agent, "register_strategy"):
                            agent_controller.gamms_agent.register_strategy(strategy)
                    except Exception:
                        pass

            success(f"Assigned strategies to {len(self.strategies)} agents")
        except Exception as e:
            error(f"Failed to assign strategies: {e}")
            raise

    def validate_and_execute_movement(self, agent_name: str, target_node: Optional[int]) -> bool:
        agent = self.agent_engine.agents.get(agent_name)
        if not agent or not agent.is_alive():
            return False

        current_node = agent.current_position

        if target_node is None:
            # Explicit "stay"
            return True

        if target_node not in self.graph.nodes():
            warning(f"Agent {agent_name}: Target node {target_node} does not exist")
            return False

        distance_lookup = get_apsp_length_cache(self.graph)
        distance = get_cached_distance(distance_lookup, current_node, target_node)
        if distance is None:
            warning(f"Agent {agent_name}: No path from {current_node} to {target_node}")
            return False
        agent_speed = getattr(agent, "speed", 1)
        if distance > agent_speed:
            warning(f"Agent {agent_name}: Cannot reach node {target_node} from {current_node} " f"(distance: {distance}, speed: {agent_speed})")
            return False

        try:
            agent.update_position(target_node)
            return True
        except Exception as e:
            error(f"Failed to execute movement for {agent_name}: {e}")
            return False

    def execute_agent_strategies(self) -> Dict[str, int]:
        actions: Dict[str, int] = {}
        for agent_controller in self.agent_engine.active_agents:
            agent_name = agent_controller.name
            try:
                gamms_agent = agent_controller.gamms_agent
                state = gamms_agent.get_state()
                state.update(
                    {
                        "time": self.time_counter,
                        "payoff": {"red": self.red_payoff_accum, "blue": self.blue_payoff_accum},
                        "name": agent_name,
                        "agent_controller": agent_controller,
                        "team_cache": agent_controller.cache,
                    }
                )

                if hasattr(agent_controller, "strategy") and agent_controller.strategy is not None:
                    try:
                        agent_controller.strategy(state)
                        action = state.get("action")
                        actions[agent_name] = action
                    except Exception as e:
                        error(f"Strategy execution failed for {agent_name}: {e}")
                        traceback.print_exc()
                        actions[agent_name] = agent_controller.current_position
                else:
                    if self.vis_engine and hasattr(self.vis_engine, "get_human_input"):
                        action = self.vis_engine.get_human_input(agent_name, state)
                        actions[agent_name] = action or agent_controller.current_position
                    else:
                        actions[agent_name] = agent_controller.current_position
            except Exception as e:
                error(f"Error processing strategy for {agent_name}: {e}")
                actions[agent_name] = agent_controller.current_position
        return actions

    def process_movements(self, actions: Dict[str, int]) -> None:
        for agent_name, target_node in actions.items():
            if agent_name in self.agent_engine.agents:
                self.validate_and_execute_movement(agent_name, target_node)

    def compute_payoff(self, red_captures: int, blue_captures: int, red_killed: int, blue_killed: int) -> Tuple[float, float]:
        payoff_config = self.game_config.get("payoff_model", {})
        model = payoff_config.get("model", "zero_sum")

        default_constants = {"red_capture": 1, "blue_capture": 1, "red_killed": 0.5, "blue_killed": 0.5}
        constants = {**default_constants, **payoff_config.get("constants", {})}

        red_capture_reward = constants["red_capture"]
        blue_capture_reward = constants["blue_capture"]
        red_kill_penalty = constants["red_killed"]
        blue_kill_penalty = constants["blue_killed"]

        step_red_capture = 0.0
        step_red_tag = 0.0
        step_red_discover = 0.0

        if model == "zero_sum" or model == "zero_sum_reward":
            step_red_capture = red_capture_reward * red_captures
            step_red_tag = -red_kill_penalty * red_killed
            if len(self.discovered_flags) > self.discovered_flags_count:
                newly_discovered = len(self.discovered_flags) - self.discovered_flags_count
                step_red_discover = newly_discovered * 0.1
                self.discovered_flags_count = len(self.discovered_flags)

            red_payoff = step_red_capture + step_red_tag + step_red_discover
            blue_payoff = -red_payoff
        elif model == "non_zero_sum":
            step_red_capture = red_capture_reward * red_captures
            step_red_tag = -red_kill_penalty * red_killed
            red_payoff = step_red_capture + step_red_tag
            blue_payoff = blue_capture_reward * blue_captures - blue_kill_penalty * blue_killed
        else:
            warning(f"Unknown payoff model: {model}. Using zero_sum.")
            step_red_capture = red_capture_reward * red_captures - blue_capture_reward * blue_captures
            step_red_tag = -red_kill_penalty * red_killed + blue_kill_penalty * blue_killed
            red_payoff = step_red_capture + step_red_tag
            blue_payoff = -red_payoff

        self.red_capture_reward += step_red_capture
        self.red_tag_penalty += step_red_tag
        self.red_discover_reward += step_red_discover

        self.step_payoff_breakdown = {
            "red": {
                "capture": step_red_capture,
                "tag": step_red_tag,
                "discover": step_red_discover,
                "total": red_payoff,
            },
            "blue": {"total": blue_payoff},
        }

        return red_payoff, blue_payoff

    def check_termination(self) -> Optional[str]:
        reason = self._termination_reason()
        if reason == "time":
            info(f"Game terminated: Maximum time ({self.max_time}) reached")
            return reason
        if reason == "red_eliminated":
            info("Game terminated: All red team agents eliminated")
            return reason
        if reason == "blue_eliminated":
            info("Game terminated: All blue team agents eliminated")
            return reason
        return None

    def _termination_reason(self) -> Optional[str]:
        if self.time_counter >= self.max_time:
            return "time"

        remaining_red = len(self.agent_engine.get_active_agents_by_team("red"))
        remaining_blue = len(self.agent_engine.get_active_agents_by_team("blue"))

        if remaining_red == 0:
            return "red_eliminated"

        if remaining_blue == 0:
            return "blue_eliminated"

        return None

    def _maybe_render_dry_run_frame(self, reason: Optional[str]) -> None:
        if reason not in ("red_eliminated", "blue_eliminated"):
            return
        if not self.vis_engine:
            return
        try:
            self.vis_engine.update_display()
        except Exception as e:
            warning(f"Dry-run frame render failed: {e}")

    def _build_agent_survival_records(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        end_time = self.time_counter
        for ctrl in self.agent_engine.all_agents:
            name = ctrl.name
            death_time = self._agent_death_time.get(name)
            observed = death_time is not None
            survival_time = max(0, (death_time if observed else end_time))

            records.append(
                {
                    "agent_name": name,
                    "team": ctrl.team,
                    "start_node": ctrl.start_node_id,
                    "death_time": death_time,
                    "censored": not observed,
                    "survival_time": survival_time,
                }
            )
        return records

    def _update_first_event_times(self, t: int, capture_details, tagging_details, new_discoveries) -> None:
        if self._first_capture_time is None and capture_details:
            self._first_capture_time = t
        if self._first_tag_time is None and tagging_details:
            self._first_tag_time = t
        if self._first_discover_time is None and new_discoveries:
            self._first_discover_time = t

    def _record_deaths(self, time_value: int, alive_before: Set[str]) -> None:
        alive_after = {agent.name for agent in self.agent_engine.get_active_agents()}
        for name in alive_before - alive_after:
            self._agent_death_time.setdefault(name, time_value)

    def log_game_state(
        self,
        step_red_caps: int = 0,
        step_blue_caps: int = 0,
        step_red_killed: int = 0,
        step_blue_killed: int = 0,
        capture_details: Optional[List] = None,
        tagging_details: Optional[List] = None,
        discovered_flags: Optional[Set[int]] = None,
    ) -> None:
        if not self.logger:
            return

        agent_positions: Dict[str, int] = {}
        for agent_controller in self.agent_engine.all_agents:
            if agent_controller.is_alive():
                agent_positions[agent_controller.name] = agent_controller.current_position

        log_data = {
            "agents": agent_positions,
            "payoff": {"red": self.red_payoff_accum, "blue": self.blue_payoff_accum},
            "payoff_step_components": self.step_payoff_breakdown,
            "payoff_cumulative_components": {
                "red": {
                    "capture": self.red_capture_reward,
                    "tag": self.red_tag_penalty,
                    "discover": self.red_discover_reward,
                    "total": self.red_payoff_accum,
                },
            },
            "red_captures": self.red_captures,
            "blue_captures": self.blue_captures,
            "red_agent_killed": step_red_killed,
            "blue_agent_killed": step_blue_killed,
            "total_tags": len(tagging_details or []),
            "tagging_details": tagging_details or [],
            "capture_details": capture_details or [],
            "discovered_flags_this_step": sorted(list(discovered_flags or set())),
            "discovered_flags_cumulative": sorted(list(self.discovered_flags)),
            "time": self.time_counter,
        }

        self.logger.log_data(log_data, self.time_counter)

    def _feed_hud_events(self, capture_details, tagging_details, discovered_flags) -> None:
        """Send game events and payoff breakdown to the HUD overlay."""
        if not self.vis_engine:
            return
        RED = (200, 50, 50)
        BLUE = (50, 50, 200)
        GREEN = (50, 180, 50)

        for agent_name, _agent_team, flag_node in (capture_details or []):
            self.vis_engine.add_hud_event(f"{agent_name} captured Flag {flag_node}", GREEN)

        for red_name, blue_name, outcome in (tagging_details or []):
            if "both" in outcome:
                self.vis_engine.add_hud_event(f"{blue_name} tagged {red_name} ({outcome})", BLUE)
            elif "red" in outcome:
                self.vis_engine.add_hud_event(f"{blue_name} tagged {red_name}", BLUE)
            else:
                self.vis_engine.add_hud_event(f"{red_name} tagged {blue_name}", RED)

        for flag_id in (discovered_flags or set()):
            self.vis_engine.add_hud_event(f"Flag {flag_id} discovered", GREEN)

        self.vis_engine.update_hud_payoff(
            total=self.red_payoff_accum,
            tag=self.red_tag_penalty,
            capture=self.red_capture_reward,
            discover=self.red_discover_reward,
        )

    def run_single_step(self) -> bool:
        try:
            actions = self.execute_agent_strategies()
            self.process_movements(actions)

            if self.vis_engine:
                self.vis_engine.update_display()

            alive_before = {a.name for a in self.agent_engine.get_active_agents()}

            (
                step_red_caps,
                step_blue_caps,
                step_red_killed,
                step_blue_killed,
                remaining_red,
                remaining_blue,
                capture_details,
                tagging_details,
                step_discovered_flags,
            ) = self.interaction_engine.step(self.time_counter)
            self._record_deaths(self.time_counter, alive_before)

            self.red_captures += step_red_caps
            self.blue_captures += step_blue_caps
            self.red_killed += step_red_killed
            self.blue_killed += step_blue_killed
            new_discoveries = step_discovered_flags - self.discovered_flags
            self.discovered_flags.update(step_discovered_flags)
            self._update_first_event_times(self.time_counter, capture_details, tagging_details, new_discoveries)

            # Visualize flag captures
            if self.vis_engine and capture_details:
                for agent_name, agent_team, flag_node in capture_details:
                    self.vis_engine.mark_flag_captured(agent_name, agent_team, flag_node)

            red_payoff, blue_payoff = self.compute_payoff(step_red_caps, step_blue_caps, step_red_killed, step_blue_killed)
            self.red_payoff_accum += red_payoff
            self.blue_payoff_accum += blue_payoff

            self._feed_hud_events(capture_details, tagging_details, new_discoveries)

            self.log_game_state(
                step_red_caps,
                step_blue_caps,
                step_red_killed,
                step_blue_killed,
                capture_details,
                tagging_details,
                discovered_flags=new_discoveries,
            )

            reason = self.check_termination()
            if reason:
                self._maybe_render_dry_run_frame(reason)
                return False

            return True
        except Exception as e:
            error(f"Error in game step {self.time_counter}: {e}")
            traceback.print_exc()
            return False

    def run_game(self) -> Tuple[float, float, int, int, int, int, int]:
        try:
            self.is_running = True
            self.game_terminated = False
            self.step_payoff_breakdown = self._empty_step_payoff_breakdown()
            self._agent_death_time = {}
            self._first_tag_time = None
            self._first_capture_time = None
            self._first_discover_time = None

            info(f"Starting game with max time: {self.max_time}")

            alive_before = {a.name for a in self.agent_engine.get_active_agents()}

            (
                init_red_caps,
                init_blue_caps,
                init_red_killed,
                init_blue_killed,
                remaining_red,
                remaining_blue,
                capture_details,
                tagging_details,
                init_discovered_flags,
            ) = self.interaction_engine.step(self.time_counter)
            self._record_deaths(self.time_counter, alive_before)

            self.red_captures += init_red_caps
            self.blue_captures += init_blue_caps
            self.red_killed += init_red_killed
            self.blue_killed += init_blue_killed
            init_new_discoveries = init_discovered_flags - self.discovered_flags
            self.discovered_flags.update(init_discovered_flags)
            self._update_first_event_times(self.time_counter, capture_details, tagging_details, init_new_discoveries)

            # Visualize flag captures at t=0
            if self.vis_engine and capture_details:
                for agent_name, agent_team, flag_node in capture_details:
                    self.vis_engine.mark_flag_captured(agent_name, agent_team, flag_node)

            red_payoff, blue_payoff = self.compute_payoff(init_red_caps, init_blue_caps, init_red_killed, init_blue_killed)
            self.red_payoff_accum += red_payoff
            self.blue_payoff_accum += blue_payoff

            self._feed_hud_events(capture_details, tagging_details, init_new_discoveries)

            self.log_game_state(
                init_red_caps,
                init_blue_caps,
                init_red_killed,
                init_blue_killed,
                capture_details,
                tagging_details,
                discovered_flags=init_new_discoveries,
            )

            reason = self.check_termination()
            if reason:
                self._maybe_render_dry_run_frame(reason)
                self.game_terminated = True
                return (
                    self.red_payoff_accum,
                    self.blue_payoff_accum,
                    self.time_counter,
                    self.red_captures,
                    self.blue_captures,
                    self.red_killed,
                    self.blue_killed,
                )

            while self.is_running and not self.game_terminated:
                self.time_counter += 1
                if not self.run_single_step():
                    break

            self.game_terminated = True
            success(f"Game completed at time {self.time_counter}")

            return (
                self.red_payoff_accum,
                self.blue_payoff_accum,
                self.time_counter,
                self.red_captures,
                self.blue_captures,
                self.red_killed,
                self.blue_killed,
            )

        except KeyboardInterrupt:
            warning("Game interrupted by user")
            self.is_running = False
            return (
                self.red_payoff_accum,
                self.blue_payoff_accum,
                self.time_counter,
                self.red_captures,
                self.blue_captures,
                self.red_killed,
                self.blue_killed,
            )
        except Exception as e:
            error(f"Fatal error in game loop: {e}")
            traceback.print_exc()
            raise
        finally:
            if self.logger:
                self.logger.finalize(
                    red_payoff=self.red_payoff_accum,
                    blue_payoff=self.blue_payoff_accum,
                    time_value=self.time_counter,
                    red_captures=self.red_captures,
                    blue_captures=self.blue_captures,
                    red_killed=self.red_killed,
                    blue_killed=self.blue_killed,
                    survival_records=self._build_agent_survival_records(),
                    payoff_components={
                        "capture": self.red_capture_reward,
                        "tag": self.red_tag_penalty,
                        "discover": self.red_discover_reward,
                    },
                )

    def stop_game(self) -> None:
        self.is_running = False

    def get_game_state(self) -> Dict[str, Any]:
        return {
            "time": self.time_counter,
            "running": self.is_running,
            "terminated": self.game_terminated,
            "payoffs": {"red": self.red_payoff_accum, "blue": self.blue_payoff_accum},
            "captures": {"red": self.red_captures, "blue": self.blue_captures},
            "killed": {"red": self.red_killed, "blue": self.blue_killed},
            "remaining_agents": {
                "red": len(self.agent_engine.get_active_agents_by_team("red")),
                "blue": len(self.agent_engine.get_active_agents_by_team("blue")),
            },
        }

    def __str__(self) -> str:
        state = "running" if self.is_running else "stopped"
        return (
            f"GameEngine(time={self.time_counter}, state={state}, "
            f"red_payoff={self.red_payoff_accum:.2f}, blue_payoff={self.blue_payoff_accum:.2f}, "
            f"breakdown={{capture:{self.red_capture_reward:.2f}, "
            f"tag:{self.red_tag_penalty:.2f}, discover:{self.red_discover_reward:.2f}}})"
        )


# ----------------------- Clean __main__ -----------------------

if __name__ == "__main__":
    # Old long block replaced by a single high-level call.
    # You can override modules/paths via CLI later if desired.
    result = GameEngine.launch_from_files(
        config_main="output.yaml",
        extra_defs="config/game_config.yml",
        red_strategy="test_atk",
        blue_strategy="test_def",
        log_name="test_result",
        set_level=LogLevel.WARNING,
    )
    print(result)
