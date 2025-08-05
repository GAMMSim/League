import os
import pathlib
import traceback
import networkx as nx
import gamms

from lib.core.console import *
from lib.utils.file_utils import get_directories, export_graph_config
from lib.utils.config_utils import load_configuration, load_config_metadata, create_context_with_sensors
from lib.utils.sensor_utils import create_static_sensors
from lib.utils.game_utils import *
import lib.core.logger as TLOG
from typing import Optional, Tuple


# --- Main Game Runner ---
def run_game(config_name: str, root_dir: str, alpha_strategy, beta_strategy, logger: Optional[TLOG.Logger] = None, visualization: bool = False, debug: bool = False) -> Tuple[float, float, int, int, int, int, int]:
    """
    Runs the multi-agent symmetric team game using the provided configuration.

    Args:
        config_name: Name of the configuration file or path to the file (relative or absolute)
        root_dir: Root directory of the project
        alpha_strategy: Strategy module for alpha team
        beta_strategy: Strategy module for beta team
        logger: Optional TimeLogger instance for logging game events
        visualization: Whether to use visualization
        debug: Whether to print debug messages

    Returns:
        tuple: (alpha_payoff, beta_payoff, game_time, alpha_captures, beta_captures, alpha_killed, beta_killed)
    """
    # Initialize variables outside the try to ensure they exist in finally block
    time_counter = 0
    alpha_payoff_accum = 0.0
    beta_payoff_accum = 0.0
    alpha_agent_killed = 0
    beta_agent_killed = 0
    alpha_captures = 0
    beta_captures = 0
    ctx = None
    error_message = None

    try:
        dirs = get_directories(root_dir)
        config = load_configuration(config_name, dirs, debug)
        if debug:
            success("Loaded configuration successfully")

        G = export_graph_config(config, dirs, debug)

        # Create static sensor definitions once
        static_sensors = create_static_sensors()

        # Create a new context with sensors for this run
        ctx = create_context_with_sensors(config, G, visualization, static_sensors, debug)

        # Initialize agents and assign strategies
        agent_config, agent_params_dict = initialize_agents(ctx, config)
        assign_strategies(ctx, agent_config, alpha_strategy, beta_strategy)
        if debug:
            success("Assigned strategies successfully")

        # Configure visualization and initialize flags
        configure_visualization(ctx, agent_config, config)
        team_flags = initialize_flags(ctx, config, debug)

        # Retrieve game parameters
        max_time = config.get("game", {}).get("max_time", 1000)
        alpha_flag_positions = team_flags.get("alpha", [])
        beta_flag_positions = team_flags.get("beta", [])
        flag_weights = config.get("game", {}).get("flags", {}).get("weights")
        interaction_config = config.get("game", {}).get("interaction", {})
        payoff_config = config.get("game", {}).get("payoff", {})

        # Initialize the time logger
        metadata = load_config_metadata(config)
        if logger is not None:
            current_metadata = logger.get_metadata()
            merged_metadata = {**current_metadata, **metadata}
            logger.set_metadata(merged_metadata)
        if debug:
            success("Initialized time logger")

        if debug:
            success(f"Starting game with max time: {max_time}")

        # Check initial interactions before any moves
        init_alpha_caps, init_beta_caps, init_alpha_agent_killed, init_beta_agent_killed, remaining_alpha, remaining_beta, capture_details, tagging_details = check_agent_interaction(ctx, G, agent_params_dict, team_flags, interaction_config, time_counter, debug)
        alpha_captures += init_alpha_caps
        beta_captures += init_beta_caps
        alpha_agent_killed += init_alpha_agent_killed
        beta_agent_killed += init_beta_agent_killed
        alpha_payoff, beta_payoff = compute_payoff(payoff_config, init_alpha_caps, init_beta_caps, init_alpha_agent_killed, init_beta_agent_killed)
        alpha_payoff_accum += alpha_payoff
        beta_payoff_accum += beta_payoff

        # Log initial state
        initial_step_log = {
            "agents": {agent.name: agent.get_state().get("curr_pos") for agent in ctx.agent.create_iter()},
            "alpha_flag_positions": alpha_flag_positions,
            "beta_flag_positions": beta_flag_positions,
            "payoff": {
                "alpha": alpha_payoff_accum,
                "beta": beta_payoff_accum,
            },
            "alpha_captures": init_alpha_caps,
            "beta_captures": init_beta_caps,
            "alpha_agent_killed": init_alpha_agent_killed,
            "beta_agent_killed": init_beta_agent_killed,
            "total_tags": len(tagging_details),
            "tagging_details": tagging_details,
            "capture_details": capture_details,
        }
        if logger is not None:
            logger.log_data(initial_step_log, time_counter)

        # Check if game should terminate after initial state
        if check_termination(time_counter, max_time, remaining_alpha, remaining_beta):
            if debug:
                success(f"Game terminated at time {time_counter}")
            return alpha_payoff_accum, beta_payoff_accum, time_counter, alpha_captures, beta_captures, alpha_agent_killed, beta_agent_killed

        # Main game loop
        while not ctx.is_terminated():
            time_counter += 1
            next_actions = {}

            try:
                for agent in ctx.agent.create_iter():
                    state = agent.get_state()
                    state.update(
                        {
                            "alpha_flag_pos": alpha_flag_positions,
                            "beta_flag_pos": beta_flag_positions,
                            "flag_weight": flag_weights,
                            "agent_params": agent_params_dict.get(agent.name, {}),
                            "time": time_counter,
                            "payoff": {
                                "alpha": alpha_payoff_accum,
                                "beta": beta_payoff_accum,
                            },
                            "name": agent.name,
                            "agent_params_dict": agent_params_dict,
                            "team_flags": team_flags,
                        }
                    )
                    if hasattr(agent, "strategy") and agent.strategy is not None:
                        try:
                            agent.strategy(state)
                        except Exception as e:
                            error(f"Error executing strategy for {agent.name}: {e}")
                            traceback.print_exc()
                            raise RuntimeError(f"Error in config={config_name}, alpha={alpha_strategy.__name__}, " f"beta={beta_strategy.__name__}, agent={agent.name}") from e
                    else:
                        node = ctx.visual.human_input(agent.name, state)
                        state["action"] = node
                    next_actions[agent.name] = state["action"]
            except Exception as e:
                error(f"An error occurred during agent turn: {e}")
                raise e

            # Update agents with their actions
            for agent in ctx.agent.create_iter():
                state = agent.get_state()
                state["action"] = next_actions.get(agent.name, state.get("action", None))
                agent.set_state()

            # Update visualization display (simulate handles pygame events internally)
            ctx.visual.simulate()

            # Log current agent positions (using "curr_pos" from state)
            agent_positions = {agent.name: agent.get_state().get("curr_pos") for agent in ctx.agent.create_iter()}

            # Check interactions (captures, tags, etc.)
            step_alpha_caps, step_beta_caps, step_alpha_agent_killed, step_beta_agent_killed, remaining_alpha, remaining_beta, capture_details, tag_details = check_agent_interaction(
                ctx, G, agent_params_dict, team_flags, interaction_config, time_counter, debug
            )
            alpha_captures += step_alpha_caps
            beta_captures += step_beta_caps
            alpha_agent_killed += step_alpha_agent_killed
            beta_agent_killed += step_beta_agent_killed
            alpha_payoff, beta_payoff = compute_payoff(
                payoff_config, step_alpha_caps, step_beta_caps, step_alpha_agent_killed, step_beta_agent_killed
            )
            alpha_payoff_accum += alpha_payoff
            beta_payoff_accum += beta_payoff

            step_log = {
                "agents": agent_positions,
                "alpha_flag_positions": alpha_flag_positions,
                "beta_flag_positions": beta_flag_positions,
                "payoff": {
                    "alpha": alpha_payoff_accum,
                    "beta": beta_payoff_accum,
                },
                "alpha_captures": alpha_captures,
                "beta_captures": beta_captures,
                "alpha_agent_killed": step_alpha_agent_killed,
                "beta_agent_killed": step_beta_agent_killed,
                "total_tags": len(tag_details),
                "tagging_details": tag_details,
                "capture_details": capture_details,
                "time": time_counter,
            }
            if logger is not None:
                logger.log_data(step_log, time_counter)

            if check_termination(time_counter, max_time, remaining_alpha, remaining_beta):
                if debug:
                    success(f"Game terminated at time {time_counter}")
                break

        return alpha_payoff_accum, beta_payoff_accum, time_counter, alpha_captures, beta_captures, alpha_agent_killed, beta_agent_killed

    except KeyboardInterrupt:
        warning("Game interrupted by user.")
        error_message = "Game interrupted by user"
        return alpha_payoff_accum, beta_payoff_accum, time_counter, alpha_captures, beta_captures, alpha_agent_killed, beta_agent_killed
    except Exception as e:
        err_header = f"Error in config '{config_name}' after {time_counter:.2f}s: {e}"
        full_tb = traceback.format_exc()
        error(err_header)  # your logger
        traceback.print_exc()  # to stderr
        raise RuntimeError(f"{err_header}\n{full_tb}") from e
    finally:
        # Always finalize the logger if it exists
        if logger is not None:
            # Include error information in the finalization if an error occurred
            if error_message:
                logger.finalize(alpha_payoff=alpha_payoff_accum, beta_payoff=beta_payoff_accum, time_value=time_counter, alpha_captures=alpha_captures, beta_captures=beta_captures, error=error_message)
            else:
                logger.finalize(alpha_payoff=alpha_payoff_accum, beta_payoff=beta_payoff_accum, time_value=time_counter, alpha_captures=alpha_captures, beta_captures=beta_captures)
        if debug:
            success("Game completed")


if __name__ == "__main__":
    set_log_level(LogLevel.SUCCESS)
    
    current_path = pathlib.Path(__file__).resolve()
    root_path = current_path.parent
    print("Current path:", current_path)
    print("Root path:", root_path)

    import example_strategy as alpha_team
    import example_strategy as beta_team

    RESULT_PATH = os.path.join(root_path, "data/result")
    alpha_name = alpha_team.__name__.split(".")[-1]
    beta_name = beta_team.__name__.split(".")[-1]
    print(alpha_name, beta_name)
    logger = TLOG.Logger("test")
    logger.set_metadata(
        {
            "alpha_team": alpha_name,
            "beta_team": beta_name,
        }
    )

    config_path = "AF3BF3A5B5_68fb24_r01.yml"  # Example path to a config file
    # Example of using the updated runner with the new file structure
    # You can now specify just the filename and it will be found automatically
    alpha_payoff, beta_payoff, game_time, alpha_caps, beta_caps, alpha_killed, beta_killed = run_game(config_path, root_dir=str(root_path), alpha_strategy=alpha_team, beta_strategy=beta_team, logger=logger, visualization=True, debug=False)

    # logger.write_to_file("test.json")
    print("Final payoff (Alpha):", alpha_payoff)
    print("Final payoff (Beta):", beta_payoff)
    print("Game time:", game_time)
    print("Alpha captures:", alpha_caps)
    print("Beta captures:", beta_caps)
    print("Alpha killed:", alpha_killed)
    print("Beta killed:", beta_killed)