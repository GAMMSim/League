from typing import Any, Dict, List, Optional, Tuple, Set
import networkx as nx

try:
    from lib.core.console import *
    from lib.agent.agent_memory import AgentMemory
    from lib.agent.agent_graph import AgentGraph
except ImportError:
    from ..core.console import *
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
    # Extract agent configurations for both teams
    alpha_config = config.get("agents", {}).get("alpha_config", {})
    beta_config = config.get("agents", {}).get("beta_config", {})

    # Extract global defaults for both teams
    alpha_global = config.get("agents", {}).get("alpha_global", {})
    beta_global = config.get("agents", {}).get("beta_global", {})

    # Extract visualization settings
    vis_settings = config.get("visualization", {})
    colors = vis_settings.get("colors", {})
    sizes = vis_settings.get("sizes", {})

    # Set default values if not provided
    global_agent_size = sizes.get("global_agent_size", 10)

    def get_agent_param(agent_config: Dict[str, Any], param_name: str, global_config: Dict[str, Any]) -> Any:
        """Get parameter with priority: individual config > global params"""
        return agent_config.get(param_name, global_config.get(param_name))

    def create_agent_entries(configs: Dict[str, Dict[str, Any]], team: str, global_config: Dict[str, Any], team_color: str, partition: Optional[str] = None) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, AgentMemory]]:
        """
        Create agent entries and memory objects for a team

        Args:
            configs: Dictionary of agent configurations
            team: Team name (alpha/beta)
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
            tagging_radius = get_agent_param(config, "tagging_radius", global_config)  # ← Add this
            sensors = get_agent_param(config, "sensors", global_config)
            color = colors.get(f"{team}_global", team_color)

            # Create agent entry for the context
            entries[name] = {"team": team, "sensors": sensors, "color": color, "current_node_id": start_node_id, "start_node_id": start_node_id, "size": global_agent_size}

            # Extract known parameters
            known_params = ["speed", "capture_radius", "tagging_radius", "sensors", "start_node_id"]

            # Get any extra parameters as kwargs
            extra_params = {k: v for k, v in config.items() if k not in known_params}
            extra_params["team"] = team
            extra_params["territory"] = partition  # Add partition info if available

            # Create parameter object with both required and extra parameters
            memories[name] = AgentMemory(speed=speed, capture_radius=capture_radius, tagging_radius=tagging_radius, map=AgentGraph(), start_node_id=start_node_id, **extra_params)

        return entries, memories

    # Default colors for teams if not specified
    default_alpha_color = "blue"
    default_beta_color = "red"

    partition = config.get("environment", {}).get("assignment", None)
    # Create entries for both teams
    alpha_entries, alpha_memories = create_agent_entries(alpha_config, "alpha", alpha_global, default_alpha_color, partition)
    beta_entries, beta_memories = create_agent_entries(beta_config, "beta", beta_global, default_beta_color, partition)

    # Combine configurations
    agent_config = {**alpha_entries, **beta_entries}
    agent_params_dict = {**alpha_memories, **beta_memories}

    # Create agents in context
    for name, config in agent_config.items():
        ctx.agent.create_agent(name, **config)

    success(f"Created {len(alpha_entries)} alpha team agents and {len(beta_entries)} beta team agents.")
    ctx.sensor.get_sensor("agent").set_owner(None)
    return agent_config, agent_params_dict


def assign_strategies(ctx: Any, agent_config: Dict[str, Dict[str, Any]], alpha_strategy_module: Any, beta_strategy_module: Any) -> None:
    """
    Assign strategies to agents based on their team.

    This function maps agent configurations to strategies using separate strategy modules
    for alpha and beta teams. Then, it registers the strategy with each agent in the context.

    Args:
        ctx (Any): The initialized game context with agent management.
        agent_config (Dict[str, Dict[str, Any]]): Dictionary of agent configurations keyed by agent name.
        alpha_strategy_module (Any): Module providing strategies for alpha team via a `map_strategy` function.
        beta_strategy_module (Any): Module providing strategies for beta team via a `map_strategy` function.

    Returns:
        None
    """
    try:
        strategies: Dict[str, Any] = {}
        # Build strategy mappings for alpha and beta teams.
        alpha_configs = {name: config for name, config in agent_config.items() if config.get("team") == "alpha"}
        beta_configs = {name: config for name, config in agent_config.items() if config.get("team") == "beta"}

        strategies.update(alpha_strategy_module.map_strategy(alpha_configs))
        strategies.update(beta_strategy_module.map_strategy(beta_configs))

        # Register each agent's strategy if available.
        for agent in ctx.agent.create_iter():
            agent.register_strategy(strategies.get(agent.name))

        success("Strategies assigned to agents.")
    except Exception as e:
        error(f"Error assigning strategies: {e}")


def legacy_configure_visualization(ctx: Any, agent_config: Dict[str, Dict[str, Any]], config: Dict[str, Any]) -> None:
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

    # Extract other visualization parameters with defaults
    draw_node_id = vis_config.get("draw_node_id", False)
    game_speed = vis_config.get("game_speed", 1)
    game_speed = 1

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
    ctx.visual.set_graph_visual(draw_id=draw_node_id, node_color=node_color, edge_color=edge_color, node_size=6)

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
    ctx.visual.set_node_color(5, Color.Purple)  # This method likely doesn't exist

    success("Visualization configured.")


def render_partition_node(ctx: Any, data: Dict[str, Any]):
    """
    Render a colored circle overlay for partition visualization.
    Similar to render_flag but for individual nodes.
    """
    x = data.get("x", 0)
    y = data.get("y", 0)
    color = data.get("color", (255, 255, 0))  # Default yellow
    size = data.get("size", 6)

    # Render colored circle on top of existing node
    ctx.visual.render_circle(x, y, size, color)


def configure_visualization(ctx: Any, agent_config: Dict[str, Dict[str, Any]], config: Dict[str, Any]) -> None:
    """
    Configure visualization settings for the game graph and agents.
    Overlay approach: Creates one normal graph + colored overlays for partitions.

    Args:
        ctx (Any): The initialized game context that contains visualization methods.
        agent_config (Dict[str, Dict[str, Any]]): Dictionary of agent configurations keyed by agent name.
        config (Dict[str, Any]): Complete configuration dictionary containing visualization settings.

    Returns:
        None
    """
    # Extract visualization settings from config
    vis_config = config.get("visualization", {})
    env_config = config.get("environment", {})

    # Extract other visualization parameters with defaults
    draw_node_id = vis_config.get("draw_node_id", False)
    game_speed = vis_config.get("game_speed", 1)
    game_speed = 1

    # Get color settings
    colors = vis_config.get("colors", {})

    # Import color constants if they're being used
    try:
        from gamms.VisualizationEngine import Color

        # Add custom light colors to the Color class
        Color.LightBlue = (130, 130, 255)  # Light blue
        Color.LightRed = (255, 130, 130)  # Light red/pink

        # Default graph colors
        default_node_color = Color.Black
        edge_color = Color.Gray
        default_color = Color.White

        # Partition colors - using the new light colors
        partition_0_color = Color.LightRed  # For '0' bits
        partition_1_color = Color.LightBlue  # For '1' bits

    except ImportError:
        # Fallback to string colors if Color class isn't available
        default_node_color = "black"
        edge_color = "gray"
        default_color = "white"
        partition_0_color = "yellow"
        partition_1_color = "purple"

    # Get size settings
    sizes = vis_config.get("sizes", {})
    default_size = sizes.get("global_agent_size", 10)
    node_size = 6

    # Create the base graph with default colors (this stays as-is)
    ctx.visual.set_graph_visual(draw_id=draw_node_id, node_color=default_node_color, edge_color=edge_color, node_size=node_size)

    # Check if we have assignment data for partition-based coloring
    assignment = env_config.get("assignment", None)

    if assignment and len(assignment) == 200:
        print("Setting up partition-based node color overlays...")

        # Create colored overlays for each node based on partition
        for node_id in range(200):
            try:
                # Get the node position from the graph
                node = ctx.graph.graph.get_node(node_id)

                # Determine color based on assignment bit
                color = partition_0_color if assignment[node_id] == "0" else partition_1_color

                # Create overlay artist (similar to flag pattern)
                try:
                    from gamms.VisualizationEngine import Artist

                    artist = Artist(ctx, render_partition_node, layer=15)  # Layer 15 = on top of graph (layer 10)
                    artist.data.update({"x": node.x, "y": node.y, "color": color, "size": node_size})

                    ctx.visual.add_artist(f"partition_node_{node_id}", artist)

                except (ImportError, AttributeError):
                    # Fallback to dictionary API (like flags do)
                    data = {"x": node.x, "y": node.y, "color": color, "size": node_size, "layer": 15, "drawer": render_partition_node}
                    ctx.visual.add_artist(f"partition_node_{node_id}", data)

            except Exception as e:
                print(f"Warning: Could not create overlay for node {node_id}: {e}")

        # Count nodes in each partition for debugging
        partition_0_count = assignment.count("0")
        partition_1_count = assignment.count("1")
        print(f"Created overlays: {partition_0_count} yellow nodes, {partition_1_count} purple nodes")

    else:
        print("No assignment data found, using default graph coloring...")

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

    print("Visualization configured with partition-based node color overlays.")


def render_flag(ctx: Any, data: Dict[str, Any]):
    """
    Render a flag shape using basic primitives.
    Node position (x, y) represents the bottom of the flag pole.
    """
    x = data.get("x", 0)
    y = data.get("y", 0)
    flag_color = data.get("color", (255, 0, 0))
    pole_color = data.get("pole_color", (0, 0, 0))  # black

    # Flag dimensions
    flag_width = data.get("flag_width", 15)
    flag_height = data.get("flag_height", 10)
    pole_height = data.get("pole_height", 20)
    pole_width = 2

    # Draw the flag pole (vertical rectangle) - extends upward from node position
    ctx.visual.render_rectangle(x - pole_width // 2, y + pole_height * 1.8, pole_width, pole_height, pole_color)

    # Draw the flag (rectangle attached to top of pole)
    ctx.visual.render_rectangle(x, y + pole_height * 1.8, flag_width, flag_height, flag_color)


def initialize_flags(ctx: Any, config: Dict[str, Any], debug: Optional[bool] = False) -> Dict[str, List[Any]]:
    """
    Initialize flags for both teams in the game context based on the configuration.

    Args:
        ctx: The game context
        config: Configuration dictionary containing flag settings
        debug: If True, debug messages will be printed during the process

    Returns:
        Dict[str, List[Any]]: Dictionary mapping team names to their flag positions

    Raises:
        Exception: If flag positions are not found in the config
    """
    # Extract configuration
    game_config = config.get("game", {}).get("flags", {})
    alpha_positions = game_config.get("alpha_positions", [])
    beta_positions = game_config.get("beta_positions", [])

    if not alpha_positions and not beta_positions:
        warning("No flag positions found in config for either team.")
        return {"alpha": [], "beta": []}

    # Get visualization settings
    vis_config = config.get("visualization", {})
    colors = vis_config.get("colors", {})
    sizes = vis_config.get("sizes", {})

    alpha_color = colors.get("alpha_flag", (0, 0, 255))  # default blue
    beta_color = colors.get("beta_flag", (255, 0, 0))  # default red
    flag_size = sizes.get("flag_size", 10)

    # Map string colors to Color enum if available
    try:
        from gamms.VisualizationEngine import Color

        color_map = {"green": Color.Green, "red": Color.Red, "blue": Color.Blue, "yellow": Color.Yellow, "white": Color.White, "black": Color.Black, "gray": Color.Gray}
        alpha_color = color_map.get(str(alpha_color).lower(), alpha_color) if isinstance(alpha_color, str) else alpha_color
        beta_color = color_map.get(str(beta_color).lower(), beta_color) if isinstance(beta_color, str) else beta_color
    except ImportError:
        if debug:
            info("Color enum not available, using provided color values")

    # Helper function to create a flag
    def create_flag(team, positions, color):
        for idx, node_id in enumerate(positions):
            try:
                node = ctx.graph.graph.get_node(node_id)

                # Try Artist API first
                try:
                    from gamms.VisualizationEngine.artist import Artist

                    artist = Artist(ctx, render_flag, layer=20)
                    artist.data.update({"x": node.x, "y": node.y, "color": color, "flag_width": flag_size, "flag_height": flag_size * 0.7, "pole_height": flag_size * 1.5})  # Make height proportional
                    ctx.visual.add_artist(f"{team}_flag_{idx}", artist)

                    if debug:
                        info(f"{team.capitalize()} flag {idx} created at node {node_id} using custom flag drawer")

                # Fallback to dictionary API
                except (ImportError, AttributeError):
                    data = {"x": node.x, "y": node.y, "color": color, "layer": 20, "flag_width": flag_size, "flag_height": flag_size * 0.7, "pole_height": flag_size * 1.5}
                    ctx.visual.add_artist(f"{team}_flag_{idx}", data)

                    if debug:
                        info(f"{team.capitalize()} flag {idx} created at node {node_id} using dictionary API")

            except Exception as e:
                error(f"Failed to create {team} flag {idx} at node {node_id}: {str(e)}")

    # Create flags for both teams
    create_flag("alpha", alpha_positions, alpha_color)
    create_flag("beta", beta_positions, beta_color)

    if debug:
        success(f"Successfully initialized {len(alpha_positions)} alpha flags and {len(beta_positions)} beta flags")

    return {"alpha": alpha_positions, "beta": beta_positions}


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
                agent.deregister_sensor(sensor_name)

            # 2. Remove main agent artist
            if hasattr(ctx.visual, "remove_artist"):
                try:
                    ctx.visual.remove_artist(agent.name)
                except Exception:
                    pass

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
            ctx.agent.delete_agent(agent.name)
            if debug:
                info(f"Agent '{agent.name}' fully removed")
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
    ctx: Any, G: nx.Graph, agent_params: Dict[str, Any], team_flags: Dict[str, List[Any]], interaction_config: Dict[str, Any], time: float, debug: Optional[bool] = False
) -> Tuple[int, int, int, int, int, int, List[Tuple[str, str, Any]], List[Tuple[str, str, str]]]:
    """
    Main interaction checking function between agents and flags in symmetric team game.

    Returns:
        Tuple containing:
        - alpha_captures: Number of flags captured by alpha team
        - beta_captures: Number of flags captured by beta team
        - alpha_agent_killed: Number of alpha agents killed
        - beta_agent_killed: Number of beta agents killed
        - remaining_alpha: Number of remaining alpha team agents
        - remaining_beta: Number of remaining beta team agents
        - capture_details: List of (agent_name, team, flag_node) tuples
        - tagging_details: List of (agent1_name, agent2_name, outcome) tuples
    """
    processed: Set[str] = set()
    capture_details: List[Tuple[str, str, Any]] = []
    tagging_details: List[Tuple[str, str, str]] = []
    alpha_agent_killed = beta_agent_killed = 0

    def process_flag_captures() -> Tuple[int, int]:
        """Process flag captures for both teams."""
        alpha_captures = beta_captures = 0

        # Alpha capturing beta flags
        for alpha_agent in ctx.agent.create_iter():
            if alpha_agent.team != "alpha" or alpha_agent.name in processed:
                continue
            captures, captured = _check_team_captures(alpha_agent, team_flags["beta"], "alpha", "beta", time)
            alpha_captures += captures
            if captured:
                break

        # Beta capturing alpha flags
        for beta_agent in ctx.agent.create_iter():
            if beta_agent.team != "beta" or beta_agent.name in processed:
                continue
            captures, captured = _check_team_captures(beta_agent, team_flags["alpha"], "beta", "alpha", time)
            beta_captures += captures
            if captured:
                break

        return alpha_captures, beta_captures

    def _check_team_captures(agent, opponent_flags, agent_team, flag_team, time) -> Tuple[int, bool]:
        """Check if an agent can capture any opponent flags."""
        for flag_node in opponent_flags:
            try:
                distance = nx.shortest_path_length(G, agent.current_node_id, flag_node)
                capture_radius = getattr(agent_params[agent.name], "capture_radius", 0)

                if distance <= capture_radius:
                    if debug:
                        info(f"{agent_team.capitalize()} agent {agent.name} captured {flag_team} flag {flag_node} at time {time}")

                    agent_name = agent.name
                    if handle_interaction(ctx, agent, interaction_config["capture"], processed, agent_params, debug):
                        capture_details.append((agent_name, agent_team, flag_node))
                        return 1, True
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        return 0, False

    def process_combat_interactions() -> None:
        """Process combat between opposing team agents."""
        alpha_agents = [a for a in ctx.agent.create_iter() if a.team == "alpha" and a.name not in processed]
        beta_agents = [a for a in ctx.agent.create_iter() if a.team == "beta" and a.name not in processed]

        for alpha_agent in alpha_agents:
            for beta_agent in beta_agents:
                if alpha_agent.name in processed or beta_agent.name in processed:
                    continue

                if _agents_in_combat_range(alpha_agent, beta_agent):
                    outcome = _resolve_combat(alpha_agent, beta_agent)
                    tagging_details.append((alpha_agent.name, beta_agent.name, outcome))

                    if alpha_agent.name in processed or beta_agent.name in processed:
                        break

    def _agents_in_combat_range(alpha_agent, beta_agent) -> bool:
        """Check if two agents are within combat range."""
        try:
            alpha_radius = getattr(agent_params[alpha_agent.name], "tagging_radius", 0)  # ← Change to tagging_radius
            beta_radius = getattr(agent_params[beta_agent.name], "tagging_radius", 0)  # ← Change to tagging_radius
            tagging_range = min(alpha_radius, beta_radius)

            distance = nx.shortest_path_length(G, alpha_agent.current_node_id, beta_agent.current_node_id)
            return distance <= tagging_range
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return False

    def _resolve_combat(alpha_agent, beta_agent) -> str:
        """Resolve combat between two agents based on territory control and configuration."""
        nonlocal alpha_agent_killed, beta_agent_killed
        
        # Get partition string from agent parameters
        partition = None
        for agent_name, agent_param in agent_params.items():
            if hasattr(agent_param, 'territory') and agent_param.territory:
                partition = agent_param.territory
                break
        
        # If no partition data, use original combat rules
        if not partition or len(partition) < 200:
            return _resolve_original_combat(alpha_agent, beta_agent)
        
        # Determine which team controls which partition value by checking flag locations
        alpha_flags = team_flags.get("alpha", [])
        beta_flags = team_flags.get("beta", [])
        
        alpha_partition_value = None
        beta_partition_value = None
        
        # Check which partition value the team flags are in
        if alpha_flags and len(alpha_flags) > 0 and alpha_flags[0] < len(partition):
            alpha_partition_value = partition[alpha_flags[0]]  # '0' or '1'
            beta_partition_value = '1' if alpha_partition_value == '0' else '0'
        elif beta_flags and len(beta_flags) > 0 and beta_flags[0] < len(partition):
            beta_partition_value = partition[beta_flags[0]]  # '0' or '1' 
            alpha_partition_value = '1' if beta_partition_value == '0' else '0'
        else:
            # Can't determine partition mapping, use original combat rules
            return _resolve_original_combat(alpha_agent, beta_agent)
        
        # Get both agent positions
        alpha_position = alpha_agent.current_node_id
        beta_position = beta_agent.current_node_id
        
        # Check if positions are valid
        if alpha_position >= len(partition) or beta_position >= len(partition):
            return _resolve_original_combat(alpha_agent, beta_agent)
        
        # Get territory values for both positions
        alpha_territory = partition[alpha_position]
        beta_territory = partition[beta_position]
        
        if debug:
            info(f"Combat: Alpha at {alpha_position} (territory: {alpha_territory}), Beta at {beta_position} (territory: {beta_territory})")
            info(f"Alpha territory: {alpha_partition_value}, Beta territory: {beta_partition_value}")
        
        # Apply territory-based combat rules
        if alpha_territory == beta_territory:
            # Both agents in same territory
            if alpha_territory == '0':
                # Both in red territory - red team wins
                if alpha_partition_value == '0':
                    # Alpha is red team, alpha wins
                    handle_interaction(ctx, beta_agent, "kill", processed, agent_params, debug)
                    beta_agent_killed += 1
                    if debug:
                        info(f"Red agent {alpha_agent.name} tags Blue agent {beta_agent.name} in Red territory")
                    return "red_wins"
                else:
                    # Beta is red team, beta wins
                    handle_interaction(ctx, alpha_agent, "kill", processed, agent_params, debug)
                    alpha_agent_killed += 1
                    if debug:
                        info(f"Red agent {beta_agent.name} tags Blue agent {alpha_agent.name} in Red territory")
                    return "red_wins"
            else:  # alpha_territory == '1'
                # Both in blue territory - blue team wins
                if alpha_partition_value == '1':
                    # Alpha is blue team, alpha wins
                    handle_interaction(ctx, beta_agent, "kill", processed, agent_params, debug)
                    beta_agent_killed += 1
                    if debug:
                        info(f"Blue agent {alpha_agent.name} tags Red agent {beta_agent.name} in Blue territory")
                    return "blue_wins"
                else:
                    # Beta is blue team, beta wins
                    handle_interaction(ctx, alpha_agent, "kill", processed, agent_params, debug)
                    alpha_agent_killed += 1
                    if debug:
                        info(f"Blue agent {beta_agent.name} tags Red agent {alpha_agent.name} in Blue territory")
                    return "blue_wins"
        else:
            # Agents in different territories (boundary case) - both die
            handle_interaction(ctx, alpha_agent, "kill", processed, agent_params, debug)
            handle_interaction(ctx, beta_agent, "kill", processed, agent_params, debug)
            alpha_agent_killed += 1
            beta_agent_killed += 1
            if debug:
                info(f"Boundary combat: Both agents {alpha_agent.name} and {beta_agent.name} killed (different territories)")
            return "both_killed_boundary"


    def _resolve_original_combat(alpha_agent, beta_agent) -> str:
        """Original combat resolution for fallback when no partition data available."""
        nonlocal alpha_agent_killed, beta_agent_killed
        tagging_action = interaction_config["tagging"]

        if tagging_action == "both_kill":
            handle_interaction(ctx, alpha_agent, "kill", processed, agent_params, debug)
            handle_interaction(ctx, beta_agent, "kill", processed, agent_params, debug)
            alpha_agent_killed += 1
            beta_agent_killed += 1
            return "both_killed"

        elif tagging_action == "both_respawn":
            handle_interaction(ctx, alpha_agent, "respawn", processed, agent_params, debug)
            handle_interaction(ctx, beta_agent, "respawn", processed, agent_params, debug)
            return "both_respawned"

        elif isinstance(tagging_action, list):
            return _resolve_probabilistic_combat(alpha_agent, beta_agent, tagging_action)
        else:
            # Custom action applied to both - only count kills if action is "kill"
            handle_interaction(ctx, alpha_agent, tagging_action, processed, agent_params, debug)
            handle_interaction(ctx, beta_agent, tagging_action, processed, agent_params, debug)
            if tagging_action == "kill":
                alpha_agent_killed += 1
                beta_agent_killed += 1
            return f"both_{tagging_action}"


    def _resolve_probabilistic_combat(alpha_agent, beta_agent, probabilities) -> str:
        """Resolve combat using probabilistic outcomes."""
        nonlocal alpha_agent_killed, beta_agent_killed
        import random

        def normalize_probabilities(values):
            total = sum(values)
            return [x / total for x in values] if total > 0 else [0] * len(values)

        probs = normalize_probabilities(probabilities)
        rand = random.random()

        if rand < probs[2]:  # Both die
            handle_interaction(ctx, alpha_agent, "kill", processed, agent_params, debug)
            handle_interaction(ctx, beta_agent, "kill", processed, agent_params, debug)
            alpha_agent_killed += 1
            beta_agent_killed += 1
            return "both_killed"
        elif rand < probs[2] + probs[0]:  # Alpha dies
            handle_interaction(ctx, alpha_agent, "kill", processed, agent_params, debug)
            alpha_agent_killed += 1
            return "alpha_killed"
        elif rand < probs[2] + probs[0] + probs[1]:  # Beta dies
            handle_interaction(ctx, beta_agent, "kill", processed, agent_params, debug)
            beta_agent_killed += 1
            return "beta_killed"
        else:
            return "no_casualties"

    # Main execution logic
    if interaction_config["prioritize"] == "capture":
        alpha_captures, beta_captures = process_flag_captures()
        process_combat_interactions()
    else:
        process_combat_interactions()
        alpha_captures, beta_captures = process_flag_captures()

    # Count remaining agents
    remaining_alpha = sum(1 for a in ctx.agent.create_iter() if a.team == "alpha")
    remaining_beta = sum(1 for a in ctx.agent.create_iter() if a.team == "beta")

    return alpha_captures, beta_captures, alpha_agent_killed, beta_agent_killed, remaining_alpha, remaining_beta, capture_details, tagging_details


def check_termination(time: int, MAX_TIME: int, remaining_alpha: int, remaining_beta: int) -> bool:
    """
    Check if the game should be terminated based on time or if one team is eliminated.

    Args:
        time (int): The current time step.
        MAX_TIME (int): The maximum allowed time steps.
        remaining_alpha (int): The number of remaining alpha team agents.
        remaining_beta (int): The number of remaining beta team agents.

    Returns:
        bool: True if termination condition is met, False otherwise.
    """
    if time >= MAX_TIME:
        success("Maximum time reached.")
        return True
    if remaining_alpha == 0:
        success("All alpha team agents have been eliminated.")
        return True
    if remaining_beta == 0:
        success("All beta team agents have been eliminated.")
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


def compute_payoff(payoff_config: Dict[str, Any], alpha_captures: int, beta_captures: int, alpha_agent_killed: int = 0, beta_agent_killed: int = 0) -> Tuple[float, float]:
    """
    Computes the payoff based on the specified model in the config for symmetric team game.

    Args:
        payoff_config (Dict[str, Any]): Payoff configuration containing model name and constants
        alpha_captures (int): Number of flags captured by alpha team
        beta_captures (int): Number of flags captured by beta team
        alpha_agent_killed (int): Number of alpha team agents killed
        beta_agent_killed (int): Number of beta team agents killed

    Returns:
        Tuple[float, float]: (alpha_payoff, beta_payoff)
    """
    # Check which payoff model to use
    model = payoff_config.get("model", "V2_zero_sum")

    # Set up default constants (all 1)
    default_constants = {"alpha_capture": 1, "beta_capture": 1, "alpha_killed": 1, "beta_killed": 1}

    # Get constants from config, using defaults for missing keys
    constants = {**default_constants, **payoff_config.get("constants", {})}

    # Extract constants (no defaults needed since we handled that above)
    alpha_capture_reward = constants["alpha_capture"]
    beta_capture_reward = constants["beta_capture"]
    alpha_kill_penalty = constants["alpha_killed"]
    beta_kill_penalty = constants["beta_killed"]

    if model == "V2_zero_sum":
        # Zero-sum: one team's gain is exactly the other's loss
        alpha_payoff = alpha_capture_reward * alpha_captures - beta_capture_reward * beta_captures - alpha_kill_penalty * alpha_agent_killed + beta_kill_penalty * beta_agent_killed
        beta_payoff = -alpha_payoff

    elif model == "V2_non_zero_sum":
        # Non-zero-sum: each team's payoff calculated independently
        alpha_payoff = alpha_capture_reward * alpha_captures - alpha_kill_penalty * alpha_agent_killed
        beta_payoff = beta_capture_reward * beta_captures - beta_kill_penalty * beta_agent_killed

    else:
        print(f"Warning: Unknown payoff model: {model}. Defaulting to V2_zero_sum model.")
        # Default to zero-sum calculation
        alpha_payoff = alpha_capture_reward * alpha_captures - beta_capture_reward * beta_captures - alpha_kill_penalty * alpha_agent_killed + beta_kill_penalty * beta_agent_killed
        beta_payoff = -alpha_payoff

    return alpha_payoff, beta_payoff


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
