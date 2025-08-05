from typeguard import typechecked
import networkx as nx
import gamms

try:
    from lib.core.console import *
except ImportError:
    from ..core.console import *


@typechecked
def create_static_sensors() -> dict:
    """
    Create static sensor definitions that can be reused across contexts.
    """
    # Include additional configuration parameters for each sensor type
    sensors = {
        "map": {"type": gamms.sensor.SensorType.MAP}, 
        "agent": {"type": gamms.sensor.SensorType.AGENT, "sensor_range": float("inf")}, 
        "neighbor": {"type": gamms.sensor.SensorType.NEIGHBOR}
    }
    debug(f"Created static sensors: {list(sensors.keys())}")
    return sensors


def extract_map_sensor_data(state):
    """
    Extract map sensor data from agent state.
    
    Args:
        state (dict): The agent state dictionary.
        
    Returns:
        tuple: (nodes_data, edges_data) containing map information.
        
    Raises:
        ValueError: If map sensor data is not found.
    """
    sensor_data = state.get("sensor", {})
    debug(f"Available sensor keys: {list(sensor_data.keys())}")

    map_data = None
    for key, value in sensor_data.items():
        if key == "map" or (isinstance(key, str) and key.startswith("map")):
            _, map_data = value
            debug(f"Found map sensor data with key: {key}")
            break
    
    if map_data is None:
        error("No map sensor data found in state")
        raise ValueError("No map sensor data found in state.")

    nodes_data = map_data["nodes"]
    edges_data = map_data["edges"]
    edges_data = {edge.id: edge for edge in edges_data}

    info(f"Extracted map data: {len(nodes_data)} nodes, {len(edges_data)} edges")
    return nodes_data, edges_data


def extract_neighbor_sensor_data(state):
    """
    Extract neighbor sensor data from agent state.

    Args:
        state (dict): The agent state dictionary.

    Returns:
        list: List of neighboring node IDs.

    Raises:
        ValueError: If sensor data is missing or in unexpected format.
    """
    sensor_data = state.get("sensor", {})
    debug(f"Searching for neighbor sensor in keys: {list(sensor_data.keys())}")

    # Try to find neighbor sensor with exact match or prefix
    neighbor_sensor = None
    for key, value in sensor_data.items():
        if key == "neighbor" or (isinstance(key, str) and key.startswith("neigh")):
            neighbor_sensor = value
            debug(f"Found neighbor sensor with key: {key}")
            break

    if neighbor_sensor is None:
        error("No neighbor sensor data found in state")
        raise ValueError("No neighbor sensor data found in state.")

    # Unpack the tuple (sensor_type, data)
    sensor_type, neighbor_data = neighbor_sensor
    info(f"Extracted neighbor data: {len(neighbor_data) if neighbor_data else 0} neighbors")
    return neighbor_data


def extract_agent_sensor_data(state):
    """
    Extract agent sensor data from agent state.

    Args:
        state (dict): The agent state dictionary.

    Returns:
        dict: Dictionary mapping agent names to their current positions.

    Raises:
        ValueError: If sensor data is missing or in unexpected format.
    """
    sensor_data = state.get("sensor", {})
    debug(f"Searching for agent sensor in keys: {list(sensor_data.keys())}")

    # Try to find agent sensor with exact match or prefix
    agent_sensor = None
    for key, value in sensor_data.items():
        if key == "agent" or (isinstance(key, str) and key.startswith("agent")):
            agent_sensor = value
            debug(f"Found agent sensor with key: {key}")
            break

    if agent_sensor is None:
        error("No agent sensor data found in state")
        raise ValueError("No agent sensor data found in state.")

    # Unpack the tuple (sensor_type, data)
    sensor_type, agent_info = agent_sensor
    info(f"Extracted agent data: {len(agent_info) if agent_info else 0} agents")
    return agent_info


def extract_sensor_data(state, flag_pos, flag_weight, agent_params):
    """
    Extract and process all sensor data from agent state.
    Updated for alpha/beta team system.

    Args:
        state (dict): The agent state dictionary.
        flag_pos (list): List of flag position node IDs.
        flag_weight (dict): Dictionary mapping flag IDs to their weights.
        agent_params (object): Object with map and other agent parameters.

    Returns:
        tuple: (alpha_positions, beta_positions) containing team positions.
    """
    debug("Starting sensor data extraction")
    
    try:
        nodes_data, edges_data = extract_map_sensor_data(state)
        agent_info = extract_agent_sensor_data(state)
        agent_info = state.get("agent_info", agent_info)

        # Add the current agent to agent_info if not already present
        current_agent_name = state.get("name")
        current_agent_pos = state.get("curr_pos")

        if current_agent_name and current_agent_pos is not None:
            if current_agent_name not in agent_info:
                info(f"Adding current agent {current_agent_name} at position {current_agent_pos} to agent info")
                agent_info[current_agent_name] = current_agent_pos
            else:
                debug(f"Current agent {current_agent_name} already in agent info")

        # Update agent parameters
        debug("Updating agent parameters with extracted data")
        agent_params.map.update_networkx_graph(nodes_data, edges_data)
        agent_params.map.set_agent_dict(agent_info)
        agent_params.map.set_flag_positions(flag_pos)
        agent_params.map.set_flag_weights(flag_weight)

        # Get team positions using alpha/beta instead of attacker/defender
        try:
            alpha_positions = agent_params.map.get_team_positions("alpha")
            beta_positions = agent_params.map.get_team_positions("beta")
        except (AttributeError, KeyError):
            # Fallback: manually determine team positions from agent_params_dict
            debug("Fallback: manually determining team positions")
            alpha_positions = []
            beta_positions = []
            
            agent_params_dict = state.get("agent_params_dict", {})
            for agent_name, position in agent_info.items():
                if agent_name in agent_params_dict:
                    agent_param = agent_params_dict[agent_name]
                    if hasattr(agent_param, 'team'):
                        if agent_param.team == "alpha":
                            alpha_positions.append((agent_name, position))
                        elif agent_param.team == "beta":
                            beta_positions.append((agent_name, position))
            
            # Also store the separated team data in the map for strategy access
            alpha_dict = {name: pos for name, pos in alpha_positions}
            beta_dict = {name: pos for name, pos in beta_positions}
            
            # Store team-specific data if map supports it
            if hasattr(agent_params.map, 'set_team_positions'):
                agent_params.map.set_team_positions("alpha", alpha_dict)
                agent_params.map.set_team_positions("beta", beta_dict)
        
        info(f"Successfully extracted sensor data - Alpha: {len(alpha_positions)}, Beta: {len(beta_positions)}")
        return alpha_positions, beta_positions
        
    except Exception as e:
        error(f"Error in extract_sensor_data: {str(e)}")
        
        # For debugging, log the types of data
        if "nodes_data" in locals() and "edges_data" in locals():
            debug(f"nodes_data type: {type(nodes_data)}, edges_data type: {type(edges_data)}")
        
        # Log additional context for debugging
        debug(f"State keys: {list(state.keys()) if isinstance(state, dict) else 'Not a dict'}")
        debug(f"Flag positions: {flag_pos}")
        debug(f"Flag weights: {flag_weight}")
        
        raise


def extract_agent_info(state: Dict[str, Any]) -> Tuple[str, int, int, int, int, int, str]:
    """
    Extract basic agent information from the game state.
    
    Args:
        state: The game state dictionary
        
    Returns:
        Tuple containing:
        - name: Agent's name
        - capture_radius: Distance needed to capture flags
        - tagging_radius: Distance needed to tag enemies
        - speed: Movement speed per turn
        - current_position: Current node ID
        - time: Current game timestep
        - team: Team membership ("alpha" or "beta")
    """
    # Basic agent info
    name = state["name"]
    current_position = state["curr_pos"]
    time = state["time"]
    
    # Agent parameters from AgentMemory object
    agent_params = state["agent_params"]
    capture_radius = agent_params.capture_radius
    tagging_radius = agent_params.tagging_radius
    speed = agent_params.speed
    
    # Team info from agent parameters dict
    my_agent_params = state["agent_params_dict"][name]
    team = my_agent_params.team
    
    return name, capture_radius, tagging_radius, speed, current_position, time, team


def extract_team_and_enemy_info(state: Dict[str, Any]) -> Tuple[Dict[str, int], List[int], Dict[str, int], List[int]]:
    """
    Extract team and enemy information from the game state.
    Integrates sensor processing to update agent map with latest data.
    
    Args:
        state: The game state dictionary
        
    Returns:
        Tuple containing:
        - teammate_positions: Dict mapping teammate names to their positions
        - team_flags: List of flag positions this team should defend
        - enemy_positions: Dict mapping enemy names to their positions  
        - enemy_flags: List of enemy flag positions to attack
    """
    debug("Starting team/enemy info extraction with sensor processing")
    
    # Get basic team info first
    my_name = state["name"]
    my_team = state["agent_params_dict"][my_name].team
    
    if my_team == "alpha":
        team_flags = state["alpha_flag_pos"]
        enemy_flags = state["beta_flag_pos"]
    else:
        team_flags = state["beta_flag_pos"] 
        enemy_flags = state["alpha_flag_pos"]
    
    # Process sensor data to update agent map
    try:
        # Extract map sensor data
        sensor_data = state.get("sensor", {})
        map_data = None
        
        for key, value in sensor_data.items():
            if key == "map" or (isinstance(key, str) and key.startswith("map")):
                _, map_data = value
                debug(f"Found map sensor data with key: {key}")
                break
        
        if map_data is not None:
            nodes_data = map_data["nodes"]
            edges_data = map_data["edges"]
            edges_data = {edge.id: edge for edge in edges_data}
            info(f"Extracted map data: {len(nodes_data)} nodes, {len(edges_data)} edges")
        else:
            warning("No map sensor data found")
            nodes_data, edges_data = {}, {}
        
        # Extract agent sensor data
        agent_info = {}
        for key, value in sensor_data.items():
            if key == "agent" or (isinstance(key, str) and key.startswith("agent")):
                _, agent_info = value
                debug(f"Found agent sensor with key: {key}")
                break
        
        # Add current agent if missing
        current_agent_pos = state.get("curr_pos")
        if my_name and current_agent_pos is not None:
            if my_name not in agent_info:
                info(f"Adding current agent {my_name} at position {current_agent_pos}")
                agent_info[my_name] = current_agent_pos
        
        # Update agent parameters map
        agent_params = state["agent_params"]
        if nodes_data or edges_data:
            agent_params.map.update_networkx_graph(nodes_data, edges_data)
        agent_params.map.set_agent_dict(agent_info)
        agent_params.map.set_flag_positions(team_flags + enemy_flags)
        agent_params.map.set_flag_weights(state.get("flag_weight"))
        
        # Get team positions using updated map
        try:
            alpha_positions = agent_params.map.get_team_positions("alpha")
            beta_positions = agent_params.map.get_team_positions("beta")
            
            # Convert to dictionaries for easier access
            alpha_dict = {name: pos for name, pos in alpha_positions}
            beta_dict = {name: pos for name, pos in beta_positions}
            
        except (AttributeError, KeyError):
            # Fallback: manually determine team positions
            debug("Fallback: manually determining team positions")
            alpha_dict = {}
            beta_dict = {}
            
            agent_params_dict = state.get("agent_params_dict", {})
            for agent_name, position in agent_info.items():
                if agent_name in agent_params_dict:
                    agent_param = agent_params_dict[agent_name]
                    if hasattr(agent_param, 'team'):
                        if agent_param.team == "alpha":
                            alpha_dict[agent_name] = position
                        elif agent_param.team == "beta":
                            beta_dict[agent_name] = position
        
        # Return based on my team
        if my_team == "alpha":
            teammate_positions = alpha_dict
            enemy_positions = beta_dict
        else:
            teammate_positions = beta_dict
            enemy_positions = alpha_dict
            
        info(f"Team info extracted - Teammates: {len(teammate_positions)}, Enemies: {len(enemy_positions)}")
        
    except Exception as e:
        error(f"Error in sensor processing: {str(e)}")
        # Fallback: use existing map data
        agent_map = state["agent_params"].map
        all_agents = state["agent_params_dict"]
        teammate_positions = {}
        enemy_positions = {}
        
        for agent_name, agent_params in all_agents.items():
            current_pos = agent_map.agent_dict.get(agent_name)
            if current_pos is not None:
                if agent_params.team == my_team:
                    teammate_positions[agent_name] = current_pos
                else:
                    enemy_positions[agent_name] = current_pos
    
    return teammate_positions, team_flags, enemy_positions, enemy_flags


def extract_map_and_territory_info(state: Dict[str, Any]) -> Tuple[nx.MultiDiGraph, List[int], List[int], List[int]]:
    """
    Extract map and territory information from the game state.
    Integrates neighbor sensor processing and uses partition assignment for territory.
    
    Args:
        state: The game state dictionary
        
    Returns:
        Tuple containing:
        - graph: NetworkX MultiDiGraph representing the game map
        - team_territory: List of nodes controlled by this team (based on partition)
        - enemy_territory: List of nodes controlled by enemy team (based on partition)
        - neighbor_nodes: List of neighboring node IDs from neighbor sensor
    """
    debug("Starting map/territory extraction with neighbor sensor processing")
    
    # Get the graph from agent parameters
    graph = state["agent_params"].map.graph
    
    # Extract neighbor sensor data
    neighbor_nodes = []
    try:
        sensor_data = state.get("sensor", {})
        for key, value in sensor_data.items():
            if key == "neighbor" or (isinstance(key, str) and key.startswith("neigh")):
                _, neighbor_data = value
                neighbor_nodes = neighbor_data if neighbor_data else []
                debug(f"Found neighbor sensor with key: {key}")
                info(f"Extracted neighbor data: {len(neighbor_nodes)} neighbors")
                break
        
        if not neighbor_nodes:
            warning("No neighbor sensor data found")
            
    except Exception as e:
        error(f"Error extracting neighbor sensor data: {str(e)}")
        neighbor_nodes = []
    
    # Get team info and partition assignment for territory calculation
    my_name = state["name"]
    my_team = state["agent_params_dict"][my_name].team
    
    # Get partition assignment from agent parameters or state
    partition = None
    if hasattr(state["agent_params_dict"][my_name], 'territory'):
        partition = state["agent_params_dict"][my_name].territory
    
    # Fallback: try to get from state
    if partition is None:
        partition = (
            state.get("assignment") or
            state.get("partition") or
            state.get("environment", {}).get("assignment")
        )
    
    # Calculate territory based on partition assignment
    team_territory = []
    enemy_territory = []
    
    if partition and len(partition) >= 200:
        debug("Using partition assignment for territory calculation")
        
        # Get flag positions to determine team partition mapping
        alpha_flags = state.get("alpha_flag_pos", [])
        beta_flags = state.get("beta_flag_pos", [])
        
        # Check which partition value the team flags are in
        alpha_partition_value = None
        beta_partition_value = None
        
        if alpha_flags and len(alpha_flags) > 0 and alpha_flags[0] < len(partition):
            alpha_partition_value = partition[alpha_flags[0]]
            beta_partition_value = '1' if alpha_partition_value == '0' else '0'
        
        # Determine my team's partition value
        if my_team == "alpha":
            my_partition_value = alpha_partition_value
            enemy_partition_value = beta_partition_value
        else:
            my_partition_value = beta_partition_value  
            enemy_partition_value = alpha_partition_value
        
        # Build territory lists based on partition assignment
        if my_partition_value is not None:
            for node_id in range(min(len(partition), 200)):
                if partition[node_id] == my_partition_value:
                    team_territory.append(node_id)
                elif partition[node_id] == enemy_partition_value:
                    enemy_territory.append(node_id)
        
        info(f"Territory from partition - Team: {len(team_territory)}, Enemy: {len(enemy_territory)}")
        
    else:
        # Fallback: calculate territory as nodes within 2 steps of flags
        warning("No partition data found, using flag-based territory calculation")
        
        if my_team == "alpha":
            team_flags = state["alpha_flag_pos"]
            enemy_flags = state["beta_flag_pos"]
        else:
            team_flags = state["beta_flag_pos"]
            enemy_flags = state["alpha_flag_pos"]
        
        try:
            # Team territory: nodes within 2 steps of team flags
            for flag_pos in team_flags:
                for node in graph.nodes():
                    try:
                        if nx.shortest_path_length(graph, flag_pos, node) <= 2:
                            if node not in team_territory:
                                team_territory.append(node)
                    except nx.NetworkXNoPath:
                        continue
            
            # Enemy territory: nodes within 2 steps of enemy flags  
            for flag_pos in enemy_flags:
                for node in graph.nodes():
                    try:
                        if nx.shortest_path_length(graph, flag_pos, node) <= 2:
                            if node not in enemy_territory:
                                enemy_territory.append(node)
                    except nx.NetworkXNoPath:
                        continue
                        
            info(f"Territory from flags - Team: {len(team_territory)}, Enemy: {len(enemy_territory)}")
            
        except Exception as e:
            error(f"Could not calculate flag-based territories: {e}")
            team_territory = team_flags.copy() if 'team_flags' in locals() else []
            enemy_territory = enemy_flags.copy() if 'enemy_flags' in locals() else []
    
    return graph, team_territory, enemy_territory, neighbor_nodes