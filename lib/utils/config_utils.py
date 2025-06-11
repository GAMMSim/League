from typing import Any, Dict, List, Optional, Tuple
from typeguard import typechecked
import datetime
import hashlib
import gamms
import os

try:
    from lib.core.core import *
    from lib.utils.distribution import *
    from lib.utils.file_utils import read_yml_file, write_yaml_config
except ImportError:
    from ..core.core import *
    from ..utils.distribution import *
    from ..utils.file_utils import read_yml_file, write_yaml_config


@typechecked
def generate_position_with_distribution(graph: nx.Graph, num_nodes: int, dist_type: str, param, center_node: Optional[int] = None, debug: Optional[bool] = False) -> Tuple[list, Optional[int]]:
    """
    Picks a center node (provided or randomly selected) from the graph, then generates positions using the given distribution.

    Parameters:
    -----------
    graph : nx.Graph
        The input graph.
    num_nodes : int
        The number of nodes to select.
    dist_type : str
        The distribution type to use. Options include:
          - "uniform": Uses distribute_uniform_random (param is max_distance)
          - "normal": Uses distribute_normal (param should be a tuple (mean_distance, std_dev))
          - "exponential": Uses distribute_exponential (param is scale)
          - "power_law": Uses distribute_power_law (param is exponent)
          - "beta": Uses distribute_beta (param should be a tuple (alpha, beta))
          - "high_degree": Uses distribute_degree_weighted with favor_high_degree=True
          - "low_degree": Uses distribute_degree_weighted with favor_high_degree=False
    param : varies
        Parameter(s) required for the selected distribution.
    center_node : Optional[int]
        The center node id to use. If None, a random center node is selected.
    debug : Optional[bool]
        If True, debug messages will be printed during the process.

    Returns:
    --------
    tuple
        (positions, center_node), where positions is a list of selected node ids.
        Returns (None, None) in case of an error.
    """
    # If no center_node provided, choose one randomly
    if center_node is None:
        try:
            center_node = random.choice([n for n in graph.nodes() if isinstance(n, int)])
        except Exception as e:
            error(f"Error selecting random center node: {e}")
            return None, None
    else:
        # Verify that the provided center_node is in the graph
        if center_node not in graph.nodes():
            warning(f"Provided center_node {center_node} is not in the graph.")
            try:
                center_node = random.choice([n for n in graph.nodes() if isinstance(n, int)])
                info(f"Using center node: {center_node}", debug)
            except Exception as e:
                error(f"Error selecting random center node: {e}")
                return None, None

    if dist_type == "uniform":
        positions = distribute_uniform_random(graph, center_node, num_nodes, max_distance=param)
    elif dist_type == "normal":
        try:
            mean_d, std = param
        except Exception as e:
            error(f"Invalid parameter for normal distribution: {param}. Error: {e}")
            return None, center_node
        positions = distribute_normal(graph, center_node, num_nodes, mean_distance=mean_d, std_dev=std)
    elif dist_type == "exponential":
        positions = distribute_exponential(graph, center_node, num_nodes, scale=param)
    elif dist_type == "power_law":
        positions = distribute_power_law(graph, center_node, num_nodes, exponent=param)
    elif dist_type == "beta":
        try:
            alpha, beta_param = param
        except Exception as e:
            error(f"Invalid parameter for beta distribution: {param}. Error: {e}")
            return None, center_node
        positions = distribute_beta(graph, center_node, num_nodes, alpha=alpha, beta=beta_param)
    elif dist_type == "high_degree":
        positions = distribute_degree_weighted(graph, center_node, num_nodes, favor_high_degree=True)
    elif dist_type == "low_degree":
        positions = distribute_degree_weighted(graph, center_node, num_nodes, favor_high_degree=False)
    else:
        warning(f"Distribution type '{dist_type}' not recognized. Using default center positions.")
        positions = [center_node] * num_nodes
    info(f"Generated {num_nodes} positions using distribution: {dist_type}", debug)
    return positions, center_node


@typechecked
def recursive_update(default: Dict, override: Dict, force: bool, debug: Optional[bool] = False) -> Dict:
    """
    Recursively updates the 'default' dictionary with the 'override' dictionary.

    For each key in the override dictionary:
      - If force is True:
          - If the key exists in default, override the value and print a warning.
          - If the key does not exist in default, add the key with the override value and print a debug message.
      - If force is False:
          - If the key exists in default and its value is None or "Error", override and print a debug message.
          - If the key does not exist in default, add it with the override value and print a debug message.

    If both values are dictionaries, the function updates them recursively.

    Parameters:
    -----------
    default : Dict
        The original configuration dictionary.
    override : Dict
        The extra (override) dictionary.
    force : bool
        Whether to force overriding keys that already have a valid value.

    Returns:
    --------
    Dict
        The updated dictionary.
    """
    for key, value in override.items():
        # If both default and override values are dictionaries, update recursively.
        if key in default and isinstance(default[key], dict) and isinstance(value, dict):
            default[key] = recursive_update(default[key], value, force)
        else:
            # Force is True: Always override or add.
            if force:
                if key in default:
                    # Check if the keys are the same
                    if default[key] != value:
                        warning(f"Overriding key '{key}': {default[key]} -> {value}", debug)
                        default[key] = value
                else:
                    info(f"Key '{key}' not found in original config. Adding with value: {value}", debug)
                    default[key] = value
            else:
                # Force is False: Only override if key is missing or its value is None or "Error".
                if key in default:
                    current = default.get(key)
                    if current is None or current == "Error":
                        info(f"Key '{key}' is missing or invalid (current: {current}). Setting to: {value}", debug)
                        default[key] = value
                else:
                    info(f"Key '{key}' not found in original config. Adding with value: {value}", debug)
                    default[key] = value
    return default


@typechecked
def load_config_metadata(config: Dict) -> Dict[str, Any]:
    metadata = {}
    # Graph file name
    metadata["graph_file"] = config["environment"]["graph_name"]

    # Flag parameters
    flag_config = config["extra_prameters"]["parameters"]["flag"]
    metadata["flag_num"] = flag_config["number"]
    metadata["flag_dist_type"] = flag_config["distribution"]["type"]
    metadata["flag_param"] = flag_config["distribution"]["param"]

    # Attacker parameters
    attacker_config = config["extra_prameters"]["parameters"]["attacker"]
    metadata["attacker_num"] = attacker_config["number"]
    metadata["attacker_dist_type"] = attacker_config["distribution"]["type"]
    metadata["attacker_param"] = attacker_config["distribution"]["param"]

    # Defender parameters
    defender_config = config["extra_prameters"]["parameters"]["defender"]
    metadata["defender_num"] = defender_config["number"]
    metadata["defender_dist_type"] = defender_config["distribution"]["type"]
    metadata["defender_param"] = defender_config["distribution"]["param"]

    return metadata


@typechecked
def generate_config_parameters(
    graph_file: str,
    game_rule: str,
    flag_num: int,
    flag_dist_type: str,
    flag_param: Any,
    center_node_flag: Any,
    flag_positions: Any,
    attacker_num: int,
    attacker_dist_type: str,
    attacker_param: Any,
    center_node_attacker: Any,
    attacker_positions: Any,
    defender_num: int,
    defender_dist_type: str,
    defender_param: Any,
    center_node_defender: Any,
    defender_positions: Any,
) -> Tuple[Dict, str]:
    # Build individual attacker and defender configurations
    ATTACKER_CONFIG = {f"attacker_{i}": {"start_node_id": attacker_positions[i]} for i in range(len(attacker_positions))}
    DEFENDER_CONFIG = {f"defender_{i}": {"start_node_id": defender_positions[i]} for i in range(len(defender_positions))}

    # Build the parameters information to be stored under extra_prameters
    parameters = {
        "flag": {
            "center_node": center_node_flag,
            "number": flag_num,
            "distribution": {"type": flag_dist_type, "param": flag_param},
        },
        "attacker": {
            "center_node": center_node_attacker,
            "number": attacker_num,
            "distribution": {"type": attacker_dist_type, "param": attacker_param},
        },
        "defender": {
            "center_node": center_node_defender,
            "number": defender_num,
            "distribution": {"type": defender_dist_type, "param": defender_param},
        },
    }

    # Get current date and time up to minutes
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")[:-3]

    # Create a unique hash based on the PARAMETERS information, other key parts, and the timestamp.
    config_str = str(parameters) + graph_file + game_rule + timestamp
    hash_key = hashlib.sha256(config_str.encode()).hexdigest()[:10]

    # Build the generated configuration (partial) from the parameters
    generated_config = {
        "game": {
            "rule": game_rule,
            # Note: Other game settings (max_time, interaction, payoff, etc.) will be filled by the default config.
            "flag": {
                "positions": flag_positions,
            },
        },
        "environment": {
            "graph_name": graph_file,
        },
        "agents": {
            "attacker_config": ATTACKER_CONFIG,
            "defender_config": DEFENDER_CONFIG,
        },
        # Store the original PARAMETERS info, the generated CONFIG_ID, and timestamp in extra_prameters
        "extra_prameters": {
            "parameters": parameters,
            "CONFIG_ID": hash_key,
            "timestamp": timestamp,
        },
    }
    return generated_config, hash_key


@typechecked
def generate_single_config(
    graph: nx.Graph,
    graph_file: str,
    flag_num: int,
    flag_dist_type: str,
    flag_param: Any,
    attacker_num: int,
    defender_num: int,
    attacker_dist_type: str,
    attacker_param: Any,
    defender_dist_type: str,
    defender_param: Any,
    game_rule: str,
    output_dir: str,
    default_config_path: str,  # New parameter for the default config file
    debug: Optional[bool] = False,
    center_node_flag: Optional[int] = None,
    center_node_attacker: Optional[int] = None,
    center_node_defender: Optional[int] = None,
    custom_flag_positions: Optional[List[int]] = None,
    custom_attacker_positions: Optional[List[int]] = None,
    custom_defender_positions: Optional[List[int]] = None,
) -> Tuple[bool, str]:
    """
    Generates a single configuration file based on the given parameters.
    It loads a default configuration from 'default_config_path' and fills in missing keys.

    Parameters:
    -----------
    graph : nx.Graph
        The graph object.
    graph_file : str
        The name of the graph file.
    flag_num : int
        Number of flags.
    flag_dist_type : str
        Distribution type for flag positions.
    flag_param : Any
        Parameter(s) for the flag distribution.
    attacker_num : int
        Number of attacker agents.
    defender_num : int
        Number of defender agents.
    attacker_dist_type : str
        Distribution type for attacker positions.
    attacker_param : Any
        Parameter(s) for the attacker distribution.
    defender_dist_type : str
        Distribution type for defender positions.
    defender_param : Any
        Parameter(s) for the defender distribution.
    game_rule : str
        The game rule to include in the configuration.
    output_dir : str
        Directory where the configuration file will be saved.
    default_config_path : str
        Path to the default configuration YAML file.
    debug : Optional[bool]
        If True, debug messages will be printed during the process.
    center_node_flag : Optional[int]
        The center node for flag positions. If None, a random node will be selected.
    center_node_attacker : Optional[int]
        The center node for attacker positions. If None, a random node will be selected.
    center_node_defender : Optional[int]
        The center node for defender positions. If None, a random node will be selected.
    custom_flag_positions : Optional[List[int]]
        Custom flag positions. If provided, this will override the generated positions.
    custom_attacker_positions : Optional[List[int]]
        Custom attacker positions. If provided, this will override the generated positions.
    custom_defender_positions : Optional[List[int]]
        Custom defender positions. If provided, this will override the generated positions.

    Returns:
    --------
    bool
        True if the configuration was generated successfully, False otherwise.
    """
    # Generate positions for flag, attacker, and defender using provided functions.
    # These functions should return positions and a center node.
    if custom_flag_positions is None:
        flag_positions, center_node_flag = generate_position_with_distribution(graph, flag_num, flag_dist_type, flag_param, center_node=center_node_flag)
        if flag_positions is None:
            error(f"Flag position generation failed for graph {graph_file} with parameters: flag_num={flag_num}, distribution={flag_dist_type}, param={flag_param}")
            return False, ""
    else:
        # Use custom flag positions if provided
        flag_positions = custom_flag_positions
        center_node_flag = None  # Indicate that positions were manually set
        flag_num = len(flag_positions)  # Update flag_num based on custom positions
        flag_dist_type = "handpicked"  
        flag_param = None  # Indicate that no distribution was used

    if custom_attacker_positions is None:
        attacker_positions, center_node_attacker = generate_position_with_distribution(graph, attacker_num, attacker_dist_type, attacker_param, center_node=center_node_attacker)
        if attacker_positions is None:
            error(f"Attacker position generation failed for graph {graph_file} with parameters: attacker_num={attacker_num}, distribution={attacker_dist_type}, param={attacker_param}")
            return False, ""
    else:
        # Use custom attacker positions if provided
        attacker_positions = custom_attacker_positions
        center_node_attacker = None
        attacker_num = len(attacker_positions)  # Update flag_num based on custom positions
        attacker_dist_type = "handpicked"
        attacker_param = None  # Indicate that no distribution was used
    
    if custom_defender_positions is None:
        defender_positions, center_node_defender = generate_position_with_distribution(graph, defender_num, defender_dist_type, defender_param, center_node=center_node_defender)
        if defender_positions is None:
            error(f"Defender position generation failed for graph {graph_file} with parameters: defender_num={defender_num}, distribution={defender_dist_type}, param={defender_param}")
            return False, ""
    else:
        # Use custom defender positions if provided
        defender_positions = custom_defender_positions
        center_node_defender = None
        defender_num = len(defender_positions)  # Update flag_num based on custom positions
        defender_dist_type = "handpicked"
        defender_param = None  # Indicate that no distribution was used

    # Build the generated configuration (partial) and compute CONFIG_ID
    generated_config, hash_key = generate_config_parameters(
        graph_file=graph_file,
        game_rule=game_rule,
        flag_num=flag_num,
        flag_dist_type=flag_dist_type,
        flag_param=flag_param,
        center_node_flag=center_node_flag,
        flag_positions=flag_positions,
        attacker_num=attacker_num,
        attacker_dist_type=attacker_dist_type,
        attacker_param=attacker_param,
        center_node_attacker=center_node_attacker,
        attacker_positions=attacker_positions,
        defender_num=defender_num,
        defender_dist_type=defender_dist_type,
        defender_param=defender_param,
        center_node_defender=center_node_defender,
        defender_positions=defender_positions,
    )

    # Load the default configuration
    try:
        default_config = read_yml_file(default_config_path, debug)
    except Exception as e:
        error(e)
        return False, ""

    # Merge the generated configuration into the default config (generated values override defaults)
    merged_config = recursive_update(generated_config, default_config, debug=debug, force=False)

    # Write the merged configuration to a YAML file
    filename = f"config_{hash_key}.yml"
    if not write_yaml_config(merged_config, output_dir, filename):
        return False, filename

    success(f"Generated configuration: {filename}", debug)
    return True, filename


@typechecked
def extract_positions_from_config(config: Dict[str, Any]) -> Tuple[List[int], List[int], List[int], Optional[str]]:
    """
    Extract the attacker and defender start node IDs, flag positions, and graph name from a configuration dictionary.

    Args:
        config (Dict[str, Any]): The configuration dictionary created by generate_config_parameters

    Returns:
        Tuple[List[int], List[int], List[int], Optional[str]]: A tuple containing:
            - List of attacker start node IDs
            - List of defender start node IDs
            - List of flag positions
            - Graph name (or None if not found)
    """
    # Extract attacker start node IDs
    attacker_config = config.get("agents", {}).get("attacker_config", {})
    attacker_positions = []
    for i in range(len(attacker_config)):
        # Try with both string and integer keys
        key = f"attacker_{i}"
        if key in attacker_config and "start_node_id" in attacker_config[key]:
            attacker_positions.append(attacker_config[key]["start_node_id"])

    # Extract defender start node IDs
    defender_config = config.get("agents", {}).get("defender_config", {})
    defender_positions = []
    for i in range(len(defender_config)):
        # Try with both string and integer keys
        key = f"defender_{i}"
        if key in defender_config and "start_node_id" in defender_config[key]:
            defender_positions.append(defender_config[key]["start_node_id"])

    # Extract flag positions
    flag_positions = config.get("game", {}).get("flag", {}).get("positions", [])

    # Extract graph name
    graph_name = config.get("environment", {}).get("graph_name")

    return attacker_positions, defender_positions, flag_positions, graph_name


@typechecked
def apply_game_rule_overrides(config: Dict, game_rule_path: str, debug: Optional[bool] = False) -> Dict:

    # Check if game rule is in the config.
    if "game" not in config or "rule" not in config["game"]:
        warning("No game rule found in the configuration. Skipping game rule overrides.")
        return config
    game_rule_name = config["game"]["rule"]
    game_rule_file = os.path.join(game_rule_path, f"{game_rule_name}.yml")
    try:
        gr = read_yml_file(game_rule_file, debug).pop("gamerule", {})
    except Exception as e:
        error(f"Error reading game rule file {game_rule_file}: {e}")
        return config
    if not gr:
        warning(f"No gamerule found in {game_rule_file}. Skipping game rule overrides.")
        return config

    # Process non-agent keys first.
    for key, value in gr.items():
        if key != "agents":
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                # Use force=True to override all keys.
                config[key] = recursive_update(config[key], value, force=True)
            else:
                if key in config:
                    warning(f"Overriding key '{key}': {config[key]} -> {value}", debug)
                else:
                    info(f"Key '{key}' not found in original config. Adding with value: {value}", debug)
                config[key] = value

    # --- Override the agents section ---
    if "agents" in gr:
        agents_overrides = gr["agents"]

        # Process attacker overrides.
        if "attacker_global" in agents_overrides:
            attacker_override = agents_overrides["attacker_global"]
            if "agents" in config:
                # Override global attacker settings.
                if "attacker_global" in config["agents"]:
                    old_value = config["agents"]["attacker_global"]
                    new_value = attacker_override.copy()
                    if old_value != new_value:
                        warning(f"Overriding agents.attacker_global: {old_value} -> {new_value}", debug)
                        config["agents"]["attacker_global"] = new_value
                # Override each individual attacker.
                if "attacker_config" in config["agents"]:
                    for key, a_conf in config["agents"]["attacker_config"].items():
                        old_value = a_conf.copy()
                        start_node = a_conf.get("start_node_id")
                        new_conf = attacker_override.copy()
                        if start_node is not None:
                            new_conf["start_node_id"] = start_node
                        if old_value != new_conf:
                            warning(f"Overriding agents.attacker_config.{key}: {old_value} -> {new_conf}", debug)
                            config["agents"]["attacker_config"][key] = new_conf

        # Process defender overrides.
        if "defender_global" in agents_overrides:
            defender_override = agents_overrides["defender_global"]
            if "agents" in config:
                # Override global defender settings.
                if "defender_global" in config["agents"]:
                    old_value = config["agents"]["defender_global"]
                    new_value = defender_override.copy()
                    if old_value != new_value:
                        warning(f"Overriding agents.defender_global: {old_value} -> {new_value}", debug)
                        config["agents"]["defender_global"] = new_value
                # Override each individual defender.
                if "defender_config" in config["agents"]:
                    for key, d_conf in config["agents"]["defender_config"].items():
                        old_value = d_conf.copy()
                        start_node = d_conf.get("start_node_id")
                        new_conf = defender_override.copy()
                        if start_node is not None:
                            new_conf["start_node_id"] = start_node
                        if old_value != new_conf:
                            warning(f"Overriding agents.defender_config.{key}: {old_value} -> {new_conf}", debug)
                            config["agents"]["defender_config"][key] = new_conf
    return config


@typechecked
def load_configuration(config_name: str, dirs: dict, debug: bool = False) -> dict:
    """
    Load and process the configuration file, searching nested folders if needed,
    and apply any game‑rule overrides.
    """
    # Resolve absolute vs. relative paths
    if os.path.isabs(config_name):
        config_path = config_name
    else:
        config_path = os.path.join(dirs["config"], config_name)

    # Read YAML (will search under dirs["config"] if not found at config_path)
    original_config = read_yml_file(
        config_path,
        search_if_not_found=True,
        config_dir=dirs["config"],
        debug=debug,
    )
    success("Read original config file successfully", debug)

    # Apply overrides from your rules directory
    config = apply_game_rule_overrides(
        original_config,
        dirs["rules"],
        debug=debug,
    )
    success("Applied game rule overrides", debug)

    return config


@typechecked
def create_context_with_sensors(config: dict, G: nx.MultiDiGraph, visualization: bool, static_sensors: dict, debug: bool = False):
    """
    Create a new game context, attach the graph, and create the sensors using pre-initialized definitions.
    """
    # Choose visualization engine
    if not visualization:
        VIS_ENGINE = gamms.visual.Engine.NO_VIS
    else:
        if config["visualization"]["visualization_engine"] == "PYGAME":
            VIS_ENGINE = gamms.visual.Engine.PYGAME
        else:
            VIS_ENGINE = gamms.visual.Engine.NO_VIS
    # success(f"Visualization Engine: {VIS_ENGINE}", debug)

    # Create a new context
    ctx = gamms.create_context(vis_engine=VIS_ENGINE)
    ctx.graph.attach_networkx_graph(G)

    # Create sensors using static definitions with their configuration
    for sensor_name, sensor_config in static_sensors.items():
        sensor_type = sensor_config["type"]
        # Extract other parameters (excluding type)
        params = {k: v for k, v in sensor_config.items() if k != "type"}
        sensor = ctx.sensor.create_sensor(sensor_name, sensor_type, **params)

    ctx.sensor.get_sensor("agent").set_owner(None)

    # Check the ownership of the agent sensor
    print(f"Agent sensor ownership: {ctx.sensor.get_sensor('agent')._owner}")
    print(f"Map sensor ownership: {ctx.sensor.get_sensor('map')._owner}")
    print(f"Neighbor sensor ownership: {ctx.sensor.get_sensor('neighbor')._owner}")

    return ctx
