from typing import Any, Dict, List, Optional, Tuple

try:
    from lib.core.core import *
    from lib.utils.distribution import *
    from lib.agent.agent_memory import AgentMemory
    from lib.agent.agent_graph import AgentGraph
except ImportError:
    from ..core.core import *
    from ..utils.distribution import *
    from ..agent.agent_memory import AgentMemory
    from ..agent.agent_graph import AgentGraph


def initialize_agents(ctx: Any, config: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, AgentMemory]]:
    """
    Configure and create agents in the game context based on a structured config dict.
    Individual agent parameters override global defaults when specified.

    Args:
        ctx: Game context object with agent creation capabilities
        config: Configuration dictionary containing agent settings

    Returns:
        Tuple containing:
            - agent_config: Dictionary mapping agent names to their configuration settings
            - agent_params_dict: Dictionary mapping agent names to their AgentMemory objects
    """
    # Extract agent configurations
    attacker_config = config.get("agents", {}).get("attacker_config", {})
    defender_config = config.get("agents", {}).get("defender_config", {})

    # Extract global defaults
    attacker_global = config.get("agents", {}).get("attacker_global", {})
    defender_global = config.get("agents", {}).get("defender_global", {})

    # Extract visualization settings
    vis_settings = config.get("visualization", {})
    colors = vis_settings.get("colors", {})
    sizes = vis_settings.get("sizes", {})

    # Set default values if not provided
    global_agent_size = sizes.get("global_agent_size", 10)

    def get_agent_param(agent_config: Dict[str, Any], param_name: str, global_config: Dict[str, Any]) -> Any:
        """Get parameter with priority: individual config > global params"""
        return agent_config.get(param_name, global_config.get(param_name))

    def create_agent_entries(configs: Dict[str, Dict[str, Any]], team: str, global_config: Dict[str, Any], team_color: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, AgentMemory]]:
        """
        Create agent entries and memory objects for a team

        Args:
            configs: Dictionary of agent configurations
            team: Team name (attacker/defender)
            global_config: Global configuration for the team
            team_color: Default color for the team

        Returns:
            Tuple of (agent_entries, agent_memories)
        """
        entries: Dict[str, Dict[str, Any]] = {}
        memories: Dict[str, AgentMemory] = {}

        for name, config in configs.items():
            # Ensure config is a dictionary (handle empty configs)
            if config is None:
                config = {}

            start_node_id = config.get("start_node_id")
            if start_node_id is None:
                warning(f"{name} has no start_node_id. Skipping.")
                continue

            # Get parameters with fallback to global defaults
            speed = get_agent_param(config, "speed", global_config)
            capture_radius = get_agent_param(config, "capture_radius", global_config)
            sensors = get_agent_param(config, "sensors", global_config)
            color = colors.get(f"{team}_global", team_color)

            # Create agent entry for the context
            entries[name] = {"team": team, "sensors": sensors, "color": color, "current_node_id": start_node_id, "start_node_id": start_node_id, "size": global_agent_size}

            # Extract known parameters
            known_params = ["speed", "capture_radius", "sensors", "start_node_id"]

            # Get any extra parameters as kwargs
            extra_params = {k: v for k, v in config.items() if k not in known_params}

            # Create parameter object with both required and extra parameters
            memories[name] = AgentMemory(speed=speed, capture_radius=capture_radius, map=AgentGraph(), start_node_id=start_node_id, **extra_params)

        # success(f"Created {len(entries)} {team} agents.")
        return entries, memories

    # Default colors if not specified
    default_attacker_color = "red"
    default_defender_color = "blue"

    # Create entries for both teams
    attacker_entries, attacker_memories = create_agent_entries(attacker_config, "attacker", attacker_global, default_attacker_color)

    defender_entries, defender_memories = create_agent_entries(defender_config, "defender", defender_global, default_defender_color)

    # Combine configurations
    agent_config = {**attacker_entries, **defender_entries}
    agent_params_dict = {**attacker_memories, **defender_memories}

    # Create agents in context
    for name, config in agent_config.items():
        ctx.agent.create_agent(name, **config)

    success(f"Created {len(attacker_entries)} attackers and {len(defender_memories)} defenders.")
    ctx.sensor.get_sensor("agent").set_owner(None)
    return agent_config, agent_params_dict


def assign_strategies(ctx: Any, agent_config: Dict[str, Dict[str, Any]], attacker_strategy_module: Any, defender_strategy_module: Any) -> None:
    """
    Assign strategies to agents based on their team.

    This function maps agent configurations to strategies using separate strategy modules
    for attackers and defenders. Then, it registers the strategy with each agent in the context.

    Args:
        ctx (Any): The initialized game context with agent management.
        agent_config (Dict[str, Dict[str, Any]]): Dictionary of agent configurations keyed by agent name.
        attacker_strategy_module (Any): Module providing strategies for attackers via a `map_strategy` function.
        defender_strategy_module (Any): Module providing strategies for defenders via a `map_strategy` function.

    Returns:
        None
    """
    try:
        strategies: Dict[str, Any] = {}
        # Build strategy mappings for attackers and defenders.
        attacker_configs = {name: config for name, config in agent_config.items() if config.get("team") == "attacker"}
        defender_configs = {name: config for name, config in agent_config.items() if config.get("team") == "defender"}

        strategies.update(attacker_strategy_module.map_strategy(attacker_configs))
        strategies.update(defender_strategy_module.map_strategy(defender_configs))

        # Register each agent's strategy if available.
        for agent in ctx.agent.create_iter():
            agent.register_strategy(strategies.get(agent.name))

        # success("Strategies assigned to agents.")
    except Exception as e:
        error(f"Error assigning strategies: {e}")


def configure_visualization(ctx: Any, agent_config: Dict[str, Dict[str, Any]], config: Dict[str, Any]) -> None:
    """
    Configure visualization settings for the game graph and agents.

    This function extracts visualization parameters from the config dictionary,
    sets up the global visualization parameters for the graph, and configures
    individual visualization parameters for each agent.

    Args:
        ctx (Any): The initialized game context that contains visualization methods.
        agent_config (Dict[str, Dict[str, Any]]): Dictionary of agent configurations keyed by agent name.
        config (Dict[str, Any]): Complete configuration dictionary containing visualization settings.

    Returns:
        None
    """
    # Extract visualization settings from config
    vis_config = config.get("visualization", {})

    # Extract window size with defaults
    window_size = vis_config.get("window_size", [1980, 1080])
    width = window_size[0] if isinstance(window_size, list) and len(window_size) > 0 else 1980
    height = window_size[1] if isinstance(window_size, list) and len(window_size) > 1 else 1080

    # Extract other visualization parameters with defaults
    draw_node_id = vis_config.get("draw_node_id", False)
    game_speed = vis_config.get("game_speed", 1)

    # Get color settings
    colors = vis_config.get("colors", {})

    # Import color constants if they're being used
    try:
        from gamms.VisualizationEngine import Color

        node_color = Color.Black
        edge_color = Color.Gray
        default_color = Color.White
    except ImportError:
        # Fallback to string colors if Color class isn't available
        node_color = "black"
        edge_color = "gray"
        default_color = "white"

    # Get size settings
    sizes = vis_config.get("sizes", {})
    default_size = sizes.get("global_agent_size", 10)

    # Set global graph visualization parameters
    ctx.visual.set_graph_visual(width=width, height=height, draw_id=draw_node_id, node_color=node_color, edge_color=edge_color)
    # Set game speed
    ctx.visual._sim_time_constant = game_speed

    # Set individual agent visualization parameters
    for name, agent_cfg in agent_config.items():
        # Determine agent's team to get the right color
        team = agent_cfg.get("team", "")
        team_color = colors.get(f"{team}_global", default_color)

        # Get color and size with appropriate defaults
        color = agent_cfg.get("color", team_color)
        size = agent_cfg.get("size", default_size)

        # Apply visual settings to the agent
        ctx.visual.set_agent_visual(name, color=color, size=size)

    success("Visualization configured.")


def initialize_flags(ctx: Any, config: Dict[str, Any], debug: Optional[bool] = False) -> None:
    """
    Initialize flags in the game context based on the configuration.

    Args:
        ctx: The game context
        config: Configuration dictionary containing flag settings
        debug: If True, debug messages will be printed during the process

    Returns:
        None

    Raises:
        Exception: If flag positions are not found in the config
    """
    # Extract flag positions
    flag_positions = config.get("game", {}).get("flag", {}).get("positions", [])
    if not flag_positions:
        warning("No flag positions found in config.")
        return

    # Extract visualization settings
    vis_config = config.get("visualization", {})
    colors = vis_config.get("colors", {})
    sizes = vis_config.get("sizes", {})
    flag_color = colors.get("flag", (0, 255, 0))  # default green
    flag_size = sizes.get("flag_size", 10)

    # Try mapping string to Color enum if available
    try:
        from gamms.VisualizationEngine import Color

        color_map = {"green": Color.Green, "red": Color.Red, "blue": Color.Blue, "yellow": Color.Yellow, "white": Color.White, "black": Color.Black, "gray": Color.Gray}

        if isinstance(flag_color, str) and flag_color.lower() in color_map:
            flag_color = color_map[flag_color.lower()]
    except ImportError:
        if debug:
            info("Color enum not available, using provided color values", debug)

    # Create a flag for each position
    for idx, node_id in enumerate(flag_positions):
        try:
            node = ctx.graph.graph.get_node(node_id)

            # Create artist using the Artist class API
            try:
                from gamms.VisualizationEngine.artist import Artist
                from gamms.VisualizationEngine import Shape

                # Create artist with circle shape
                artist = Artist(ctx, Shape.Circle, layer=20)
                artist.data.update({"x": node.x, "y": node.y, "radius": flag_size, "color": flag_color})
                ctx.visual.add_artist(f"flag_{idx}", artist)

                info(f"Flag {idx} created at node {node_id} using Artist API", debug)

            # Fallback for older versions or if Artist import fails
            except (ImportError, AttributeError):
                # Simple dictionary-based artist creation
                data = {"x": node.x, "y": node.y, "radius": flag_size, "color": flag_color, "layer": 20}  # Use radius instead of scale for better compatibility  # Set layer explicitly
                ctx.visual.add_artist(f"flag_{idx}", data)

                info(f"Flag {idx} created at node {node_id} using dictionary API", debug)

        except Exception as e:
            error(f"Failed to create flag {idx} at node {node_id}: {str(e)}")

    success(f"Successfully initialized {len(flag_positions)} flags", debug)


def handle_interaction(ctx: Any, agent: Any, action: str, processed: Set[str], agent_params: Dict[str, Any], debug: Optional[bool] = False) -> bool:
    """
    Handle the result of an interaction.

    Args:
        ctx: The game context
        agent: The agent involved in the interaction
        action: The action to perform ("kill", "respawn", etc.)
        processed: Set of agent names that have been processed
        agent_params: Dictionary of agent parameters
        debug: If True, debug messages will be printed during the process

    Returns:
        bool: True if the interaction was successful, False otherwise
    """
    processed.add(agent.name)  # Mark this agent as processed

    if action == "kill":
        try:
            # 1. Deregister all sensors
            for sensor_name in list(agent._sensor_list):
                agent.deregister_sensor(sensor_name)  # Logs AGENT_SENSOR_DEREGISTER :contentReference[oaicite:8]{index=8}

            # 2. Remove main agent artist
            if hasattr(ctx.visual, "remove_artist"):
                try:
                    ctx.visual.remove_artist(agent.name)
                except Exception:
                    pass  # Delegates to RenderManager.remove_artist :contentReference[oaicite:9]{index=9}

                # 3. Remove sensor artists
                for sensor_name in list(agent._sensor_list):
                    try:
                        ctx.visual.remove_artist(f"sensor_{sensor_name}")
                    except Exception:
                        warning(f"Could not remove sensor artist 'sensor_{sensor_name}'")

            # 4. Remove any auxiliary artists (prefix: "{agent.name}_")
            if hasattr(ctx.visual, "_render_manager"):
                rm = ctx.visual._render_manager
                for artist_name in list(rm._artists.keys()):
                    if artist_name.startswith(f"{agent.name}_"):
                        try:
                            ctx.visual.remove_artist(artist_name)
                        except Exception:
                            warning(f"Could not remove artist '{artist_name}'")

            # 5. Delete agent from engine
            ctx.agent.delete_agent(agent.name)  # Writes AGENT_DELETE and pops agent :contentReference[oaicite:10]{index=10}
            info(f"Agent '{agent.name}' fully removed", debug)
            return True

        except Exception as e:
            error(f"Cleanup failed for '{agent.name}': {e}")
            # Fallback: force delete agent
            try:
                ctx.agent.delete_agent(agent.name)
                warning(f"Agent '{agent.name}' deleted with partial cleanup")
                return True
            except Exception as e2:
                error(f"Critical: Could not delete agent '{agent.name}': {e2}")
                return False

    elif action == "respawn":
        start_node = agent_params[agent.name].start_node
        agent.current_node_id = start_node  # Reset position
        agent.prev_node_id = start_node
        return True

    return False


def check_agent_interaction(
    ctx: Any, G: nx.Graph, agent_params: Dict[str, Any], flag_positions: List[Any], interaction_config: Dict[str, Any], time: float, debug: Optional[bool] = False
) -> Tuple[int, int, int, int, List[Tuple[str, Any]], List[Tuple[str, str]]]:
    """
    Main interaction checking function between agents and flags.
    """
    captures = tags = 0
    processed: Set[str] = set()  # Agents that have been deleted or otherwise processed
    capture_details: List[Tuple[str, Any]] = []
    tagging_details: List[Tuple[str, str]] = []

    # Process interactions based on priority
    if interaction_config["prioritize"] == "capture":
        for attacker in list(ctx.agent.create_iter()):
            if attacker.team != "attacker" or attacker.name in processed:
                continue
            for flag in flag_positions:
                try:
                    shortest_distance = nx.shortest_path_length(G, attacker.current_node_id, flag)
                    attacker_capture_radius = getattr(agent_params[attacker.name], "capture_radius", 0)
                    if shortest_distance <= attacker_capture_radius:
                        info(f"Attacker {attacker.name} captured flag {flag} at time {time}")
                        # Store attacker name for capture details before potential deletion
                        attacker_name = attacker.name
                        if handle_interaction(ctx, attacker, interaction_config["capture"], processed, agent_params, debug):
                            captures += 1
                            # Use stored name in case attacker was deleted
                            capture_details.append((attacker_name, flag))
                            break  # Break out of the flag loop
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
    # Check combat interactions - get fresh list for each type
    defenders = [d for d in ctx.agent.create_iter() if d.team == "defender" and d.name not in processed]

    for defender in defenders:
        attackers = [a for a in ctx.agent.create_iter() if a.team == "attacker" and a.name not in processed]
        for attacker in attackers:
            # Double check neither has been processed
            if attacker.name in processed or defender.name in processed:
                continue

            try:
                defender_capture_radius = getattr(agent_params[defender.name], "capture_radius", 0)
                shortest_distance = nx.shortest_path_length(G, attacker.current_node_id, defender.current_node_id)

                if shortest_distance <= defender_capture_radius:
                    info(f"Defender {defender.name} tagged attacker {attacker.name} at time {time}")
                    defender_name = defender.name
                    attacker_name = attacker.name

                    if interaction_config["tagging"] == "both_kill":
                        handle_interaction(ctx, attacker, "kill", processed, agent_params, debug)
                        handle_interaction(ctx, defender, "kill", processed, agent_params, debug)
                    elif interaction_config["tagging"] == "both_respawn":
                        handle_interaction(ctx, attacker, "respawn", processed, agent_params, debug)
                        handle_interaction(ctx, defender, "respawn", processed, agent_params, debug)
                    else:
                        handle_interaction(ctx, attacker, interaction_config["tagging"], processed, agent_params, debug)

                    tags += 1
                    tagging_details.append((defender_name, attacker_name))

                    # If the defender was processed, break out of the inner loop
                    if defender.name in processed:
                        break
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

    # If tags processed first, check captures second
    if interaction_config["prioritize"] != "capture":
        for attacker in list(ctx.agent.create_iter()):
            # Skip non-attackers or already processed agents
            if attacker.team != "attacker" or attacker.name in processed:
                continue

            for flag in flag_positions:
                try:
                    shortest_distance = nx.shortest_path_length(G, attacker.current_node_id, flag)
                    attacker_capture_radius = getattr(agent_params[attacker.name], "capture_radius", 0)
                    if shortest_distance <= attacker_capture_radius:
                        info(f"Attacker {attacker.name} captured flag {flag} at time {time}")
                        # Store attacker name before potential deletion
                        attacker_name = attacker.name
                        if handle_interaction(ctx, attacker, interaction_config["capture"], processed, agent_params, debug):
                            captures += 1
                            # Use stored name in case attacker was deleted
                            capture_details.append((attacker_name, flag))
                            break  # Break out of the flag loop
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue

    # Count remaining agents - gets fresh accurate counts
    remaining_attackers = sum(1 for a in ctx.agent.create_iter() if a.team == "attacker")
    remaining_defenders = sum(1 for d in ctx.agent.create_iter() if d.team == "defender")

    return captures, tags, remaining_attackers, remaining_defenders, capture_details, tagging_details


def check_termination(time: int, MAX_TIME: int, remaining_attackers: int, remaining_defenders: int) -> bool:
    """
    Check if the game should be terminated based on time or if one team is eliminated.

    Args:
        time (int): The current time step.
        MAX_TIME (int): The maximum allowed time steps.
        remaining_attackers (int): The number of remaining attackers.
        remaining_defenders (int): The number of remaining defenders.

    Returns:
        bool: True if termination condition is met, False otherwise.
    """
    if time >= MAX_TIME:
        success("Maximum time reached.")
        return True
    if remaining_attackers == 0:
        success("All attackers have been eliminated.")
        return True
    if remaining_defenders == 0:
        success("All defenders have been eliminated.")
        return True
    return False


def check_agent_dynamics(state: Dict[str, Any], agent_params: Any, G: nx.Graph) -> None:
    """
    Checks and adjusts the next node for an agent based on its speed and connectivity.

    Args:
        state (Dict[str, Any]): A dictionary containing the agent's current state with keys 'action', 'curr_pos', and 'name'.
        agent_params (Any): The agent's parameters including speed.
        G (nx.Graph): The graph representing the game environment.
    """
    agent_next_node = state["action"]
    agent_speed = agent_params.speed
    agent_prev_node = state["curr_pos"]
    if agent_next_node is None:
        agent_next_node = agent_prev_node
        warning(f"Agent {state['name']} has no next node, staying at {agent_prev_node}")
    try:
        shortest_path_length = nx.shortest_path_length(G, source=agent_prev_node, target=agent_next_node)
        if shortest_path_length > agent_speed:
            warning(f"Agent {state['name']} cannot reach {agent_next_node} from {agent_prev_node} within speed limit of {agent_speed}. Staying at {agent_prev_node}")
            state["action"] = agent_prev_node
    except nx.NetworkXNoPath:
        warning(f"No path from {agent_prev_node} to {agent_next_node}. Staying at {agent_prev_node}")
        state["action"] = agent_prev_node


def compute_payoff(payoff_config: Dict[str, Any], captures: int, tags: int) -> float:
    """
    Computes the payoff based on the specified model in the config.

    Args:
        payoff_config (Dict[str, Any]): Payoff configuration containing model name and constants
        captures (int): Number of attackers captured by defenders
        tags (int): Number of successful flag tags by attackers

    Returns:
        float: Calculated payoff value
    """
    # Check which payoff model to use
    model = payoff_config.get("model", "V1")

    if model == "V1":
        return V1(payoff_config, captures, tags)
    else:
        # Fallback to V1 if model not recognized
        return V1(payoff_config, captures, tags)


def V1(payoff_config: Dict[str, Any], captures: int, tags: int) -> float:
    """
    Original V1 payoff function: captures - k * tags

    Args:
        payoff_config (Dict[str, Any]): Payoff configuration containing constants
        captures (int): Number of attackers captured by defenders
        tags (int): Number of successful flag tags by attackers

    Returns:
        float: Calculated payoff value
    """
    # Extract k value from constants, default to 1.0 if not found
    constants = payoff_config.get("constants", {})
    k = constants.get("k", 1.0)

    # Calculate payoff
    payoff = captures - k * tags
    return payoff


def check_and_install_dependencies() -> bool:
    """
    Check if required packages are installed and install them if they're missing.

    Returns:
        bool: True if all dependencies are satisfied, False if installation failed.
    """
    import subprocess
    import sys

    # Required packages mapping: import_name -> pip package name
    required_packages = {
        "yaml": "pyyaml",
        "osmnx": "osmnx",
        "networkx": "networkx",
    }

    missing_packages: List[str] = []

    for import_name, pip_name in required_packages.items():
        try:
            __import__(import_name)
            success(f"✓ {import_name} is already installed")
        except ImportError:
            warning(f"✗ {import_name} is not installed")
            missing_packages.append(pip_name)

    if missing_packages:
        info("Installing missing packages...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            for package in missing_packages:
                info(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                success(f"✓ Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            error(f"Failed to install packages: {e}")
            warning("Please try installing the packages manually:\n" + "\n".join([f"pip install {pkg}" for pkg in missing_packages]))
            return False
        except Exception as e:
            error(f"An unexpected error occurred: {e}")
            return False

    success("All required dependencies are satisfied!")
    return True
