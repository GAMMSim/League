"""
Main Game Script
This script implements a multi-agent game system with attackers and defenders.
It handles initialization, configuration, and the main game loop.
"""

# ------------------------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------------------------
import os
import sys
import pickle
from gamms.VisualizationEngine import Color

# Add the games directory to the Python path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if not root_dir.endswith("games"):
    root_dir = os.path.join(root_dir, "games")
sys.path.append(root_dir)

# Local imports
import gamms
import attacker_strategy
import defender_strategy
import interaction_model
from config import *
from utilities import initialize_game_context, configure_agents, assign_strategies, configure_visualization, initialize_flags

# ------------------------------------------------------------------------------
# Main Game Script
# ------------------------------------------------------------------------------


def main():
    """
    Main function that initializes and runs the game simulation.
    Handles setup of the game environment, agents, and main game loop.
    """
    # Initialize game context and graph
    # This sets up the basic game environment and loads the graph structure
    ctx, G = initialize_game_context(VISUALIZATION_ENGINE, GRAPH_PATH, LOCATION, RESOLUTION)

    # Create sensor systems for environment perception
    # These sensors allow agents to gather information about their environment
    ctx.sensor.create_sensor("map", MAP_SENSOR)
    ctx.sensor.create_sensor("agent", AGENT_SENSOR)
    ctx.sensor.create_sensor("neighbor", NEIGHBOR_SENSOR)

    # Configure agents with their respective parameters
    # Sets up both attacker and defender agents with their specific attributes
    agent_config, agent_params_map = configure_agents(
        ctx,
        ATTACKER_CONFIG,
        DEFENDER_CONFIG,
        {
            "attacker_sensors": ATTACKER_GLOBAL_SENSORS,
            "attacker_color": ATTACKER_GLOBAL_COLOR,
            "attacker_speed": ATTACKER_GLOBAL_SPEED,
            "attacker_capture_radius": ATTACKER_GLOBAL_CAPTURE_RADIUS,
            "defender_sensors": DEFENDER_GLOBAL_SENSORS,
            "defender_color": DEFENDER_GLOBAL_COLOR,
            "defender_speed": DEFENDER_GLOBAL_SPEED,
            "defender_capture_radius": DEFENDER_GLOBAL_CAPTURE_RADIUS,
        },
    )

    # Assign strategies to agents
    # Links the strategy implementation to each agent type
    assign_strategies(ctx, agent_config, attacker_strategy, defender_strategy)

    # Configure visualization settings
    # Sets up the game's visual representation parameters
    configure_visualization(
        ctx, agent_config, {"width": 1980, "height": 1080, "draw_node_id": DRAW_NODE_ID, "node_color": Color.Black, "edge_color": Color.Gray, "game_speed": GAME_SPEED, "default_color": Color.White, "default_size": GLOBAL_AGENT_SIZE}
    )

    # Initialize flags in the game environment
    # Sets up objective points for the agents
    initialize_flags(ctx, FLAG_POSITIONS, FLAG_SIZE, FLAG_COLOR)

    # Main game loop
    while not ctx.is_terminated():
        # Process each agent's turn
        for agent in ctx.agent.create_iter():
            # Get current state and update with flag information
            state = agent.get_state()
            state.update({"flag_pos": FLAG_POSITIONS, "flag_weight": FLAG_WEIGHTS, "agent_params": agent_params_map[agent.name]})

            # Execute agent strategy or handle human input
            if agent.strategy is not None:
                agent.strategy(state)
            else:
                # Handle human-controlled agents
                node = ctx.visual.human_input(agent.name, state)
                state["action"] = node

            agent.set_state()

        # Check victory condition: all attackers captured
        still_has_attackers = any(agent.team == "attacker" for agent in ctx.agent.create_iter())
        if not still_has_attackers:
            print("All attackers have been captured")
            break

        # Update visualization and check agent interactions
        ctx.visual.simulate()
        interaction_model.check_agent_interaction(ctx, G, agent_params_map, INTERACTION_MODEL)

    print("Game terminated.")


if __name__ == "__main__":
    main()
