from typing import Any, Dict, List, Optional, Tuple, Set
from typeguard import typechecked
import networkx as nx
import traceback

try:
    from lib.core.console import *
    from lib.game.agent_engine import AgentEngine
    from lib.game.interaction_engine import InteractionEngine
    from lib.game.visualization_engine import VisEngine
    from lib.core.logger import Logger
except ImportError:
    from ..core.console import *
    from ..game.agent_engine import AgentEngine
    from ..game.interaction_engine import InteractionEngine
    from ..game.visualization_engine import VisEngine
    from ..core.logger import Logger


@typechecked
class GameEngine:
    """
    Main game engine that coordinates the entire game loop.
    Handles movement validation, strategy execution, interactions, payoffs, and termination.
    """

    def __init__(self, ctx: Any, graph: nx.Graph, config: Dict[str, Any], agent_engine: AgentEngine, interaction_engine: InteractionEngine, vis_engine: VisEngine = None, logger: Optional[Logger] = None):
        """
        Initialize the Game Engine.

        Args:
            ctx: Game context object
            graph: NetworkX graph for the game environment
            config: Complete configuration dictionary
            agent_engine: Agent management engine
            interaction_engine: Interaction processing engine
            vis_engine: Visualization engine (optional)
            logger: Logger for game events (optional)
        """
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

        # Flag positions from config
        self.flag_config = self.config.get("flags", {})

        # Strategy storage
        self.strategies = {}

    def setup_game_visuals(self, agent_config: Dict[str, Dict[str, Any]]) -> None:
        """
        Setup game visualization including agent visuals and flags.

        Args:
            agent_config: Dictionary mapping agent names to their configurations
        """
        if not self.vis_engine:
            return

        try:
            # Setup agent visuals
            self.vis_engine.setup_agent_visuals(agent_config)

            # Create flags
            self.vis_engine.create_flags(self.flag_config)
            
            # Create agent labels
            self.vis_engine.create_agent_labels(agent_config)

            success("Game visuals initialized")

        except Exception as e:
            error(f"Failed to setup game visuals: {e}")

    def assign_strategies(self, red_strategy_module: Any, blue_strategy_module: Any) -> None:
        """
        Assign strategies to agents using strategy modules.

        Args:
            red_strategy_module: Module with map_strategy function for red team
            blue_strategy_module: Module with map_strategy function for blue team
        """
        try:
            # Get agent configurations for strategy mapping
            red_configs = {}
            blue_configs = {}

            for agent_name, agent_controller in self.agent_engine.agents.items():
                if agent_controller.team == "red":
                    red_configs[agent_name] = {"team": agent_controller.team, **agent_controller.to_dict()}
                elif agent_controller.team == "blue":
                    blue_configs[agent_name] = {"team": agent_controller.team, **agent_controller.to_dict()}

            # Get strategies from modules
            if red_configs:
                red_strategies = red_strategy_module.map_strategy(red_configs)
                self.strategies.update(red_strategies)

            if blue_configs:
                blue_strategies = blue_strategy_module.map_strategy(blue_configs)
                self.strategies.update(blue_strategies)

            # Assign strategies to agent controllers
            for agent_name, strategy in self.strategies.items():
                if agent_name in self.agent_engine.agents:
                    agent_controller = self.agent_engine.agents[agent_name]
                    agent_controller.strategy = strategy

                    # Also register with underlying GAMMS agent if possible
                    try:
                        if hasattr(agent_controller.gamms_agent, "register_strategy"):
                            agent_controller.gamms_agent.register_strategy(strategy)
                    except Exception:
                        pass  # Not critical if GAMMS registration fails

            success(f"Assigned strategies to {len(self.strategies)} agents")

        except Exception as e:
            error(f"Failed to assign strategies: {e}")
            raise

    def validate_and_execute_movement(self, agent_name: str, target_node: int) -> bool:
        """
        Validate agent movement and execute if valid.

        Args:
            agent_name: Name of the agent
            target_node: Desired target node

        Returns:
            True if movement was valid and executed, False otherwise
        """
        agent = self.agent_engine.agents.get(agent_name)
        if not agent or not agent.is_alive():
            return False

        current_node = agent.current_position

        # If target is None, stay at current position
        if target_node is None:
            target_node = current_node
            return True

        # Check if target node exists
        if target_node not in self.graph.nodes():
            warning(f"Agent {agent_name}: Target node {target_node} does not exist")
            return False

        # Check if movement is within speed limit
        try:
            distance = nx.shortest_path_length(self.graph, current_node, target_node)
            agent_speed = getattr(agent, "speed", 1)

            if distance > agent_speed:
                warning(f"Agent {agent_name}: Cannot reach node {target_node} from {current_node} " f"(distance: {distance}, speed: {agent_speed})")
                return False

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            warning(f"Agent {agent_name}: No path from {current_node} to {target_node}")
            return False

        # Valid movement - execute it
        try:
            agent.update_position(target_node)
            return True

        except Exception as e:
            error(f"Failed to execute movement for {agent_name}: {e}")
            return False

    def execute_agent_strategies(self) -> Dict[str, int]:
        """
        Execute strategies for all living agents and collect their actions.

        Returns:
            Dictionary mapping agent names to their chosen actions (node IDs)
        """
        actions = {}

        for agent_controller in self.agent_engine.active_agents:
            agent_name = agent_controller.name

            try:
                # Get the underlying GAMMS agent for state management
                gamms_agent = agent_controller.gamms_agent

                # Prepare state for strategy
                state = gamms_agent.get_state()
                state.update(
                    {
                        "time": self.time_counter,
                        "payoff": {
                            "red": self.red_payoff_accum,
                            "blue": self.blue_payoff_accum,
                        },
                        "name": agent_name,
                        "agent_params": agent_controller.to_dict(),
                    }
                )

                # Execute strategy
                if hasattr(agent_controller, "strategy") and agent_controller.strategy is not None:
                    try:
                        agent_controller.strategy(state)
                        action = state.get("action")
                        actions[agent_name] = action

                    except Exception as e:
                        error(f"Strategy execution failed for {agent_name}: {e}")
                        traceback.print_exc()
                        # Keep agent at current position
                        actions[agent_name] = agent_controller.current_position

                else:
                    # No strategy - use human input if available
                    if self.vis_engine and hasattr(self.vis_engine, "get_human_input"):
                        action = self.vis_engine.get_human_input(agent_name, state)
                        actions[agent_name] = action or agent_controller.current_position
                    else:
                        # No strategy and no human input - stay put
                        actions[agent_name] = agent_controller.current_position

            except Exception as e:
                error(f"Error processing strategy for {agent_name}: {e}")
                actions[agent_name] = agent_controller.current_position

        return actions

    def process_movements(self, actions: Dict[str, int]) -> None:
        """
        Process and validate all agent movements.

        Args:
            actions: Dictionary mapping agent names to desired target nodes
        """
        for agent_name, target_node in actions.items():
            if agent_name in self.agent_engine.agents:
                self.validate_and_execute_movement(agent_name, target_node)

    def compute_payoff(self, red_captures: int, blue_captures: int, red_killed: int, blue_killed: int) -> Tuple[float, float]:
        """
        Compute payoffs for both teams based on captures and kills.

        Args:
            red_captures: Number of captures by red team this step
            blue_captures: Number of captures by blue team this step
            red_killed: Number of red agents killed this step
            blue_killed: Number of blue agents killed this step

        Returns:
            Tuple of (red_payoff, blue_payoff) for this step
        """
        payoff_config = self.game_config.get("payoff_model", {})
        model = payoff_config.get("model", "zero_sum")

        # Default constants
        default_constants = {"red_capture": 1, "blue_capture": 1, "red_killed": 1, "blue_killed": 1}

        constants = {**default_constants, **payoff_config.get("constants", {})}

        red_capture_reward = constants["red_capture"]
        blue_capture_reward = constants["blue_capture"]
        red_kill_penalty = constants["red_killed"]
        blue_kill_penalty = constants["blue_killed"]

        if model == "zero_sum":
            red_payoff = red_capture_reward * red_captures - blue_capture_reward * blue_captures - red_kill_penalty * red_killed + blue_kill_penalty * blue_killed
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
        """
        Check if the game should terminate.

        Returns:
            True if game should end, False otherwise
        """
        # Check time limit
        if self.time_counter >= self.max_time:
            info(f"Game terminated: Maximum time ({self.max_time}) reached")
            return True

        # Count remaining agents
        remaining_red = len(self.agent_engine.get_active_agents_by_team("red"))
        remaining_blue = len(self.agent_engine.get_active_agents_by_team("blue"))

        # Check team elimination
        if remaining_red == 0:
            info("Game terminated: All red team agents eliminated")
            return True

        if remaining_blue == 0:
            info("Game terminated: All blue team agents eliminated")
            return True

        return False

    def log_game_state(self, step_red_caps: int = 0, step_blue_caps: int = 0, step_red_killed: int = 0, step_blue_killed: int = 0, capture_details: List = None, tagging_details: List = None) -> None:
        """
        Log the current game state.

        Args:
            step_red_caps: Red captures this step
            step_blue_caps: Blue captures this step
            step_red_killed: Red agents killed this step
            step_blue_killed: Blue agents killed this step
            capture_details: Details of capture events
            tagging_details: Details of tagging events
        """
        if not self.logger:
            return

        # Get agent positions
        agent_positions = {}
        for agent_controller in self.agent_engine.all_agents:
            if agent_controller.is_alive():
                agent_positions[agent_controller.name] = agent_controller.current_position

        # Create log entry
        log_data = {
            "agents": agent_positions,
            "payoff": {
                "red": self.red_payoff_accum,
                "blue": self.blue_payoff_accum,
            },
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
        """
        Execute a single game step.

        Returns:
            True if game should continue, False if terminated
        """
        try:
            # 1. Execute agent strategies
            actions = self.execute_agent_strategies()

            # 2. Process movements
            self.process_movements(actions)

            # 3. Update visualization
            if self.vis_engine:
                self.vis_engine.update_display()

            # 4. Process interactions
            (step_red_caps, step_blue_caps, step_red_killed, step_blue_killed, remaining_red, remaining_blue, capture_details, tagging_details) = self.interaction_engine.step(self.time_counter)

            # 5. Update counters
            self.red_captures += step_red_caps
            self.blue_captures += step_blue_caps
            self.red_killed += step_red_killed
            self.blue_killed += step_blue_killed

            # 6. Compute payoffs
            red_payoff, blue_payoff = self.compute_payoff(step_red_caps, step_blue_caps, step_red_killed, step_blue_killed)
            self.red_payoff_accum += red_payoff
            self.blue_payoff_accum += blue_payoff

            # 7. Log state
            self.log_game_state(step_red_caps, step_blue_caps, step_red_killed, step_blue_killed, capture_details, tagging_details)

            # 8. Check termination
            if self.check_termination():
                return False

            return True

        except Exception as e:
            error(f"Error in game step {self.time_counter}: {e}")
            traceback.print_exc()
            return False

    def run_game(self) -> Tuple[float, float, int, int, int, int, int]:
        """
        Run the complete game loop.

        Returns:
            Tuple of (red_payoff, blue_payoff, game_time, red_captures,
                     blue_captures, red_killed, blue_killed)
        """
        try:
            self.is_running = True
            self.game_terminated = False

            info(f"Starting game with max time: {self.max_time}")

            # Check initial interactions (before any moves)
            (init_red_caps, init_blue_caps, init_red_killed, init_blue_killed, remaining_red, remaining_blue, capture_details, tagging_details) = self.interaction_engine.step(self.time_counter)

            # Update initial counters
            self.red_captures += init_red_caps
            self.blue_captures += init_blue_caps
            self.red_killed += init_red_killed
            self.blue_killed += init_blue_killed

            # Compute initial payoffs
            red_payoff, blue_payoff = self.compute_payoff(init_red_caps, init_blue_caps, init_red_killed, init_blue_killed)
            self.red_payoff_accum += red_payoff
            self.blue_payoff_accum += blue_payoff

            # Log initial state
            self.log_game_state(init_red_caps, init_blue_caps, init_red_killed, init_blue_killed, capture_details, tagging_details)

            # Check if game should terminate immediately
            if self.check_termination():
                self.game_terminated = True
                return (self.red_payoff_accum, self.blue_payoff_accum, self.time_counter, self.red_captures, self.blue_captures, self.red_killed, self.blue_killed)

            # Main game loop
            while self.is_running and not self.game_terminated:
                self.time_counter += 1

                # Run single step
                continue_game = self.run_single_step()
                if not continue_game:
                    break

            self.game_terminated = True
            success(f"Game completed at time {self.time_counter}")

            return (self.red_payoff_accum, self.blue_payoff_accum, self.time_counter, self.red_captures, self.blue_captures, self.red_killed, self.blue_killed)

        except KeyboardInterrupt:
            warning("Game interrupted by user")
            self.is_running = False
            return (self.red_payoff_accum, self.blue_payoff_accum, self.time_counter, self.red_captures, self.blue_captures, self.red_killed, self.blue_killed)

        except Exception as e:
            error(f"Fatal error in game loop: {e}")
            traceback.print_exc()
            raise

        finally:
            # Finalize logger
            if self.logger:
                self.logger.finalize(
                    red_payoff=self.red_payoff_accum, blue_payoff=self.blue_payoff_accum, time_value=self.time_counter, red_captures=self.red_captures, blue_captures=self.blue_captures, red_killed=self.red_killed, blue_killed=self.blue_killed
                )

    def stop_game(self) -> None:
        """Stop the game loop."""
        self.is_running = False

    def get_game_state(self) -> Dict[str, Any]:
        """
        Get current game state information.

        Returns:
            Dictionary containing current game state
        """
        return {
            "time": self.time_counter,
            "running": self.is_running,
            "terminated": self.game_terminated,
            "payoffs": {"red": self.red_payoff_accum, "blue": self.blue_payoff_accum},
            "captures": {"red": self.red_captures, "blue": self.blue_captures},
            "killed": {"red": self.red_killed, "blue": self.blue_killed},
            "remaining_agents": {"red": len(self.agent_engine.get_active_agents_by_team("red")), "blue": len(self.agent_engine.get_active_agents_by_team("blue"))},
        }

    def __str__(self) -> str:
        """String representation of the GameEngine."""
        state = "running" if self.is_running else "stopped"
        return f"GameEngine(time={self.time_counter}, state={state}, red_payoff={self.red_payoff_accum:.2f}, blue_payoff={self.blue_payoff_accum:.2f})"
    

if __name__ == "__main__":
    import pathlib
    import gamms
    current_path = pathlib.Path(__file__).resolve()
    root_path = current_path.parent.parent.parent
    print("Current path:", current_path)
    print("Root path:", root_path)
    print("=====Test GameEngine Module=====")
    from lib.config.config_loader import ConfigLoader
    from lib.utils.file_utils import export_graph_config, get_directories
    from lib.core.logger import Logger
    dir = get_directories(str(root_path))
    print("Directories:", dir)
    
    loader = ConfigLoader("output.yaml")
    loader.load_extra_definitions("config/game_config.yml", force=True)
    config = loader.config_data
    
    graph = export_graph_config(config, dir)
    info(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    # Get window size from visualization config
    vis_config = config.get("visualization", {})
    window_size = vis_config.get("window_size", [1200, 800])
    
    # Create context with window size using vis_kwargs
    vis_kwargs = {
        "width": window_size[0] if isinstance(window_size, list) and len(window_size) > 0 else 1200,
        "height": window_size[1] if isinstance(window_size, list) and len(window_size) > 1 else 800,
    }
    ctx = gamms.create_context(vis_engine=gamms.visual.Engine.PYGAME, vis_kwargs=vis_kwargs)
    ctx.graph.attach_networkx_graph(graph)
    
    testagent_engine = AgentEngine(ctx, ["red", "blue"])
    testagent_engine.create_agents_from_config(config)
    
    testinteraction_engine = InteractionEngine(testagent_engine, graph, config)
    
    testvis_engine = VisEngine(ctx, config)
    
    testlogger = Logger("test_result")

    testgameengine = GameEngine(ctx, graph, config, testagent_engine, testinteraction_engine, testvis_engine, testlogger)
    testgameengine.setup_game_visuals(config.get("agents", {}))
    
    import test_atk as red_strategy
    import test_def as blue_strategy
    testgameengine.assign_strategies(red_strategy, blue_strategy)
    
    testgameengine.run_game()
    print(testgameengine)