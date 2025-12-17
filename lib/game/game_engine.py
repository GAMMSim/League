from typing import Any, Dict, List, Optional, Tuple, Set, Union
from typeguard import typechecked
import networkx as nx
import traceback
import importlib

try:
    from lib.core.console import *
    from lib.game.agent_engine import AgentEngine
    from lib.game.sensor_engine import SensorEngine
    from lib.game.interaction_engine import InteractionEngine
    from lib.game.visualization_engine import VisEngine
    from lib.core.logger import Logger
except ImportError:
    from ..core.console import *
    from ..game.agent_engine import AgentEngine
    from ..game.sensor_engine import SensorEngine
    from ..game.interaction_engine import InteractionEngine
    from ..game.visualization_engine import VisEngine
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

        # Flag positions from config
        self.flag_config = self.config.get("flags", {})

        # Strategy storage
        self.strategies: Dict[str, Any] = {}

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
        log_name: str = "result",
        vis_engine_kind: Optional[Any] = None,
    ) -> Tuple[Any, nx.Graph, VisEngine, SensorEngine, AgentEngine, InteractionEngine, Logger]:
        """
        Build the full runtime (ctx, graph, vis, sensors, agents, interaction, logger).
        If ctx/graph are missing, create them from config using repo utilities.
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
            engine_kind = vis_engine_kind or getattr(gamms.visual.Engine, "PYGAME", None)
            ctx = gamms.create_context(vis_engine=engine_kind, vis_kwargs=vis_kwargs)

        if graph is None:
            import pathlib

            current_path = pathlib.Path(__file__).resolve()
            root = current_path.parent.parent.parent if root_path is None else pathlib.Path(root_path)
            dirs = get_directories(str(root))
            graph = export_graph_config(config, dirs)
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

        # 6) Logger
        logger = _Logger(log_name)

        return ctx, graph, vis_engine, sensor_engine, agent_engine, interaction_engine, logger

    @classmethod
    def launch_from_files(
        cls,
        *,
        config_main: str = "output.yaml",
        extra_defs: Optional[str] = "config/game_config.yml",
        red_strategy: Union[str, Any, None] = "test_atk",
        blue_strategy: Union[str, Any, None] = "test_def",
        log_name: str = "test_result",
        set_level: Optional["LogLevel"] = LogLevel.WARNING,
    ) -> "GameEngine":
        """
        One-liner runner that mirrors the old __main__ behavior but keeps the call-site clean.
        Loads config, builds runtime, assigns strategies, sets up visuals, runs, and returns the engine.
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
        )

        # Create the unified engine
        engine = cls.from_runtime(ctx, graph, config, agent_engine, interaction_engine, vis_engine, logger)

        # Visuals
        engine.setup_game_visuals(config.get("agents", {}))

        # Strategies
        red_mod, blue_mod = cls.load_strategies(red_strategy, blue_strategy)
        if red_mod or blue_mod:
            engine.assign_strategies(red_mod, blue_mod)

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

        try:
            distance = nx.shortest_path_length(self.graph, current_node, target_node)
            agent_speed = getattr(agent, "speed", 1)
            if distance > agent_speed:
                warning(f"Agent {agent_name}: Cannot reach node {target_node} from {current_node} " f"(distance: {distance}, speed: {agent_speed})")
                return False
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            warning(f"Agent {agent_name}: No path from {current_node} to {target_node}")
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
                        flag_discovered = agent_controller.strategy(state)
                        self.discovered_flags.update(flag_discovered)
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

        default_constants = {"red_capture": 1, "blue_capture": 1, "red_killed": 1, "blue_killed": 1}
        constants = {**default_constants, **payoff_config.get("constants", {})}

        red_capture_reward = constants["red_capture"]
        blue_capture_reward = constants["blue_capture"]
        red_kill_penalty = constants["red_killed"]
        blue_kill_penalty = constants["blue_killed"]

        if model == "zero_sum" or model == "zero_sum_reward":
            red_payoff = red_capture_reward * red_captures - red_kill_penalty * red_killed
            real_flags = set(self.flag_config.get("real_positions", []))
            self.discovered_flags.intersection_update(set(real_flags))
            if len(self.discovered_flags) > self.discovered_flags_count:
                newly_discovered = len(self.discovered_flags) - self.discovered_flags_count
                red_payoff += newly_discovered * 0.1  # Reward for discovering new flags
                self.discovered_flags_count = len(self.discovered_flags)
            
            blue_payoff = -red_payoff
        elif model == "non_zero_sum":
            red_payoff = red_capture_reward * red_captures - red_kill_penalty * red_killed
            blue_payoff = blue_capture_reward * blue_captures - blue_kill_penalty * blue_killed
        else:
            warning(f"Unknown payoff model: {model}. Using zero_sum.")
            red_payoff = red_capture_reward * red_captures - blue_capture_reward * blue_captures - red_kill_penalty * red_killed + blue_kill_penalty * blue_killed
            blue_payoff = -red_payoff

        return red_payoff, blue_payoff

    def check_termination(self) -> bool:
        if self.time_counter >= self.max_time:
            info(f"Game terminated: Maximum time ({self.max_time}) reached")
            return True

        remaining_red = len(self.agent_engine.get_active_agents_by_team("red"))
        remaining_blue = len(self.agent_engine.get_active_agents_by_team("blue"))

        if remaining_red == 0:
            info("Game terminated: All red team agents eliminated")
            return True

        if remaining_blue == 0:
            info("Game terminated: All blue team agents eliminated")
            return True

        return False

    def log_game_state(
        self,
        step_red_caps: int = 0,
        step_blue_caps: int = 0,
        step_red_killed: int = 0,
        step_blue_killed: int = 0,
        capture_details: Optional[List] = None,
        tagging_details: Optional[List] = None,
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
            "red_captures": self.red_captures,
            "blue_captures": self.blue_captures,
            "red_agent_killed": step_red_killed,
            "blue_agent_killed": step_blue_killed,
            "total_tags": len(tagging_details or []),
            "tagging_details": tagging_details or [],
            "capture_details": capture_details or [],
            "time": self.time_counter,
        }

        self.logger.log_data(log_data, self.time_counter)

    def run_single_step(self) -> bool:
        try:
            actions = self.execute_agent_strategies()
            self.process_movements(actions)

            if self.vis_engine:
                self.vis_engine.update_all_agent_labels()
                self.vis_engine.update_display()

            (
                step_red_caps,
                step_blue_caps,
                step_red_killed,
                step_blue_killed,
                remaining_red,
                remaining_blue,
                capture_details,
                tagging_details,
            ) = self.interaction_engine.step(self.time_counter)

            self.red_captures += step_red_caps
            self.blue_captures += step_blue_caps
            self.red_killed += step_red_killed
            self.blue_killed += step_blue_killed

            red_payoff, blue_payoff = self.compute_payoff(step_red_caps, step_blue_caps, step_red_killed, step_blue_killed)
            self.red_payoff_accum += red_payoff
            self.blue_payoff_accum += blue_payoff

            self.log_game_state(step_red_caps, step_blue_caps, step_red_killed, step_blue_killed, capture_details, tagging_details)

            if self.check_termination():
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

            info(f"Starting game with max time: {self.max_time}")

            (
                init_red_caps,
                init_blue_caps,
                init_red_killed,
                init_blue_killed,
                remaining_red,
                remaining_blue,
                capture_details,
                tagging_details,
            ) = self.interaction_engine.step(self.time_counter)

            self.red_captures += init_red_caps
            self.blue_captures += init_blue_caps
            self.red_killed += init_red_killed
            self.blue_killed += init_blue_killed

            red_payoff, blue_payoff = self.compute_payoff(init_red_caps, init_blue_caps, init_red_killed, init_blue_killed)
            self.red_payoff_accum += red_payoff
            self.blue_payoff_accum += blue_payoff

            self.log_game_state(init_red_caps, init_blue_caps, init_red_killed, init_blue_killed, capture_details, tagging_details)

            if self.check_termination():
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
        return f"GameEngine(time={self.time_counter}, state={state}, red_payoff={self.red_payoff_accum:.2f}, blue_payoff={self.blue_payoff_accum:.2f})"


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
