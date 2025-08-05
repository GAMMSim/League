from typing import Any, Dict, List, Optional, Tuple
from typeguard import typechecked
import datetime
import hashlib
import gamms
import os

try:
    from lib.core.console import *
    from lib.utils.distribution import *
    from lib.utils.file_utils import read_yml_file, write_yaml_config
except ImportError:
    from ..core.console import *
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
                if debug:
                    info(f"Using center node: {center_node}")
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
    
    if debug:
        info(f"Generated {num_nodes} positions using distribution: {dist_type}")
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
                        if debug:
                            warning(f"Overriding key '{key}': {default[key]} -> {value}")
                        default[key] = value
                else:
                    if debug:
                        info(f"Key '{key}' not found in original config. Adding with value: {value}")
                    default[key] = value
            else:
                # Force is False: Only override if key is missing or its value is None or "Error".
                if key in default:
                    current = default.get(key)
                    if current is None or current == "Error":
                        if debug:
                            info(f"Key '{key}' is missing or invalid (current: {current}). Setting to: {value}")
                        default[key] = value
                else:
                    if debug:
                        info(f"Key '{key}' not found in original config. Adding with value: {value}")
                    default[key] = value
    return default


@typechecked
def load_config_metadata(config: Dict) -> Dict[str, Any]:
    metadata = {}
    # Graph file name
    metadata["graph_file"] = config["environment"]["graph_name"]
    metadata["territory"] = config["environment"].get("assignment", None)

    # Alpha flag parameters
    alpha_flag_config = config["extra_parameters"]["parameters"]["alpha_flags"]
    metadata["alpha_flag_num"] = alpha_flag_config["number"]
    metadata["alpha_flag_dist_type"] = alpha_flag_config["distribution"]["type"]
    metadata["alpha_flag_param"] = alpha_flag_config["distribution"]["param"]

    # Beta flag parameters
    beta_flag_config = config["extra_parameters"]["parameters"]["beta_flags"]
    metadata["beta_flag_num"] = beta_flag_config["number"]
    metadata["beta_flag_dist_type"] = beta_flag_config["distribution"]["type"]
    metadata["beta_flag_param"] = beta_flag_config["distribution"]["param"]

    # Alpha team parameters
    alpha_config = config["extra_parameters"]["parameters"]["alpha"]
    metadata["alpha_num"] = alpha_config["number"]
    metadata["alpha_dist_type"] = alpha_config["distribution"]["type"]
    metadata["alpha_param"] = alpha_config["distribution"]["param"]

    # Beta team parameters
    beta_config = config["extra_parameters"]["parameters"]["beta"]
    metadata["beta_num"] = beta_config["number"]
    metadata["beta_dist_type"] = beta_config["distribution"]["type"]
    metadata["beta_param"] = beta_config["distribution"]["param"]

    return metadata


@typechecked
def generate_config_parameters(
    graph_file: str,
    game_rule: str,
    alpha_flag_num: int,
    alpha_flag_dist_type: str,
    alpha_flag_param: Any,
    center_node_alpha_flag: Any,
    alpha_flag_positions: Any,
    beta_flag_num: int,
    beta_flag_dist_type: str,
    beta_flag_param: Any,
    center_node_beta_flag: Any,
    beta_flag_positions: Any,
    alpha_num: int,
    alpha_dist_type: str,
    alpha_param: Any,
    center_node_alpha: Any,
    alpha_positions: Any,
    beta_num: int,
    beta_dist_type: str,
    beta_param: Any,
    center_node_beta: Any,
    beta_positions: Any,
) -> Tuple[Dict, str]:
    # Build individual alpha and beta team configurations
    ALPHA_CONFIG = {f"alpha_{i}": {"start_node_id": alpha_positions[i]} for i in range(len(alpha_positions))}
    BETA_CONFIG = {f"beta_{i}": {"start_node_id": beta_positions[i]} for i in range(len(beta_positions))}

    # Build the parameters information to be stored under extra_parameters
    parameters = {
        "alpha_flags": {
            "center_node": center_node_alpha_flag,
            "number": alpha_flag_num,
            "distribution": {"type": alpha_flag_dist_type, "param": alpha_flag_param},
        },
        "beta_flags": {
            "center_node": center_node_beta_flag,
            "number": beta_flag_num,
            "distribution": {"type": beta_flag_dist_type, "param": beta_flag_param},
        },
        "alpha": {
            "center_node": center_node_alpha,
            "number": alpha_num,
            "distribution": {"type": alpha_dist_type, "param": alpha_param},
        },
        "beta": {
            "center_node": center_node_beta,
            "number": beta_num,
            "distribution": {"type": beta_dist_type, "param": beta_param},
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
            "flags": {
                "alpha_positions": alpha_flag_positions,
                "beta_positions": beta_flag_positions,
            },
        },
        "environment": {
            "graph_name": graph_file,
        },
        "agents": {
            "alpha_config": ALPHA_CONFIG,
            "beta_config": BETA_CONFIG,
        },
        # Store the original PARAMETERS info, the generated CONFIG_ID, and timestamp in extra_parameters
        "extra_parameters": {
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
    alpha_flag_num: int,
    alpha_flag_dist_type: str,
    alpha_flag_param: Any,
    beta_flag_num: int,
    beta_flag_dist_type: str,
    beta_flag_param: Any,
    alpha_num: int,
    beta_num: int,
    alpha_dist_type: str,
    alpha_param: Any,
    beta_dist_type: str,
    beta_param: Any,
    game_rule: str,
    output_dir: str,
    default_config_path: str,  # New parameter for the default config file
    debug: Optional[bool] = False,
    center_node_alpha_flag: Optional[int] = None,
    center_node_beta_flag: Optional[int] = None,
    center_node_alpha: Optional[int] = None,
    center_node_beta: Optional[int] = None,
    custom_alpha_flag_positions: Optional[List[int]] = None,
    custom_beta_flag_positions: Optional[List[int]] = None,
    custom_alpha_positions: Optional[List[int]] = None,
    custom_beta_positions: Optional[List[int]] = None,
) -> Tuple[bool, str]:
    """
    Generates a single configuration file based on the given parameters for symmetric alpha-beta team game.
    It loads a default configuration from 'default_config_path' and fills in missing keys.

    Parameters:
    -----------
    graph : nx.Graph
        The graph object.
    graph_file : str
        The name of the graph file.
    alpha_flag_num : int
        Number of alpha team flags.
    alpha_flag_dist_type : str
        Distribution type for alpha flag positions.
    alpha_flag_param : Any
        Parameter(s) for the alpha flag distribution.
    beta_flag_num : int
        Number of beta team flags.
    beta_flag_dist_type : str
        Distribution type for beta flag positions.
    beta_flag_param : Any
        Parameter(s) for the beta flag distribution.
    alpha_num : int
        Number of alpha team agents.
    beta_num : int
        Number of beta team agents.
    alpha_dist_type : str
        Distribution type for alpha team positions.
    alpha_param : Any
        Parameter(s) for the alpha team distribution.
    beta_dist_type : str
        Distribution type for beta team positions.
    beta_param : Any
        Parameter(s) for the beta team distribution.
    game_rule : str
        The game rule to include in the configuration.
    output_dir : str
        Directory where the configuration file will be saved.
    default_config_path : str
        Path to the default configuration YAML file.
    debug : Optional[bool]
        If True, debug messages will be printed during the process.
    center_node_alpha_flag : Optional[int]
        The center node for alpha flag positions. If None, a random node will be selected.
    center_node_beta_flag : Optional[int]
        The center node for beta flag positions. If None, a random node will be selected.
    center_node_alpha : Optional[int]
        The center node for alpha team positions. If None, a random node will be selected.
    center_node_beta : Optional[int]
        The center node for beta team positions. If None, a random node will be selected.
    custom_alpha_flag_positions : Optional[List[int]]
        Custom alpha flag positions. If provided, this will override the generated positions.
    custom_beta_flag_positions : Optional[List[int]]
        Custom beta flag positions. If provided, this will override the generated positions.
    custom_alpha_positions : Optional[List[int]]
        Custom alpha team positions. If provided, this will override the generated positions.
    custom_beta_positions : Optional[List[int]]
        Custom beta team positions. If provided, this will override the generated positions.

    Returns:
    --------
    bool
        True if the configuration was generated successfully, False otherwise.
    """
    # Generate positions for alpha flags
    if custom_alpha_flag_positions is None:
        alpha_flag_positions, center_node_alpha_flag = generate_position_with_distribution(
            graph, alpha_flag_num, alpha_flag_dist_type, alpha_flag_param, center_node=center_node_alpha_flag, debug=debug
        )
        if alpha_flag_positions is None:
            error(f"Alpha flag position generation failed for graph {graph_file} with parameters: alpha_flag_num={alpha_flag_num}, distribution={alpha_flag_dist_type}, param={alpha_flag_param}")
            return False, ""
    else:
        alpha_flag_positions = custom_alpha_flag_positions
        center_node_alpha_flag = None
        alpha_flag_num = len(alpha_flag_positions)
        alpha_flag_dist_type = "handpicked"
        alpha_flag_param = None

    # Generate positions for beta flags
    if custom_beta_flag_positions is None:
        beta_flag_positions, center_node_beta_flag = generate_position_with_distribution(
            graph, beta_flag_num, beta_flag_dist_type, beta_flag_param, center_node=center_node_beta_flag, debug=debug
        )
        if beta_flag_positions is None:
            error(f"Beta flag position generation failed for graph {graph_file} with parameters: beta_flag_num={beta_flag_num}, distribution={beta_flag_dist_type}, param={beta_flag_param}")
            return False, ""
    else:
        beta_flag_positions = custom_beta_flag_positions
        center_node_beta_flag = None
        beta_flag_num = len(beta_flag_positions)
        beta_flag_dist_type = "handpicked"
        beta_flag_param = None

    # Generate positions for alpha team
    if custom_alpha_positions is None:
        alpha_positions, center_node_alpha = generate_position_with_distribution(
            graph, alpha_num, alpha_dist_type, alpha_param, center_node=center_node_alpha, debug=debug
        )
        if alpha_positions is None:
            error(f"Alpha team position generation failed for graph {graph_file} with parameters: alpha_num={alpha_num}, distribution={alpha_dist_type}, param={alpha_param}")
            return False, ""
    else:
        alpha_positions = custom_alpha_positions
        center_node_alpha = None
        alpha_num = len(alpha_positions)
        alpha_dist_type = "handpicked"
        alpha_param = None

    # Generate positions for beta team
    if custom_beta_positions is None:
        beta_positions, center_node_beta = generate_position_with_distribution(
            graph, beta_num, beta_dist_type, beta_param, center_node=center_node_beta, debug=debug
        )
        if beta_positions is None:
            error(f"Beta team position generation failed for graph {graph_file} with parameters: beta_num={beta_num}, distribution={beta_dist_type}, param={beta_param}")
            return False, ""
    else:
        beta_positions = custom_beta_positions
        center_node_beta = None
        beta_num = len(beta_positions)
        beta_dist_type = "handpicked"
        beta_param = None

    # Build the generated configuration (partial) and compute CONFIG_ID
    generated_config, hash_key = generate_config_parameters(
        graph_file=graph_file,
        game_rule=game_rule,
        alpha_flag_num=alpha_flag_num,
        alpha_flag_dist_type=alpha_flag_dist_type,
        alpha_flag_param=alpha_flag_param,
        center_node_alpha_flag=center_node_alpha_flag,
        alpha_flag_positions=alpha_flag_positions,
        beta_flag_num=beta_flag_num,
        beta_flag_dist_type=beta_flag_dist_type,
        beta_flag_param=beta_flag_param,
        center_node_beta_flag=center_node_beta_flag,
        beta_flag_positions=beta_flag_positions,
        alpha_num=alpha_num,
        alpha_dist_type=alpha_dist_type,
        alpha_param=alpha_param,
        center_node_alpha=center_node_alpha,
        alpha_positions=alpha_positions,
        beta_num=beta_num,
        beta_dist_type=beta_dist_type,
        beta_param=beta_param,
        center_node_beta=center_node_beta,
        beta_positions=beta_positions,
    )

    # Load the default configuration
    try:
        default_config = read_yml_file(default_config_path, debug=debug)
    except Exception as e:
        error(str(e))
        return False, ""

    # Merge the generated configuration into the default config (generated values override defaults)
    merged_config = recursive_update(generated_config, default_config, debug=debug, force=False)

    # Write the merged configuration to a YAML file
    filename = f"config_{hash_key}.yml"
    if not write_yaml_config(merged_config, output_dir, filename):
        return False, filename

    if debug:
        success(f"Generated configuration: {filename}")
    return True, filename


@typechecked
def extract_positions_from_config(config: Dict[str, Any]) -> Tuple[List[int], List[int], List[int], List[int], Optional[str]]:
    """
    Extract the alpha and beta team start node IDs, flag positions for both teams, and graph name from a configuration dictionary.

    Args:
        config (Dict[str, Any]): The configuration dictionary created by generate_config_parameters

    Returns:
        Tuple[List[int], List[int], List[int], List[int], Optional[str]]: A tuple containing:
            - List of alpha team start node IDs
            - List of beta team start node IDs
            - List of alpha flag positions
            - List of beta flag positions
            - Graph name (or None if not found)
    """
    # Extract alpha team start node IDs
    alpha_config = config.get("agents", {}).get("alpha_config", {})
    alpha_positions = []
    for i in range(len(alpha_config)):
        key = f"alpha_{i}"
        if key in alpha_config and "start_node_id" in alpha_config[key]:
            alpha_positions.append(alpha_config[key]["start_node_id"])

    # Extract beta team start node IDs
    beta_config = config.get("agents", {}).get("beta_config", {})
    beta_positions = []
    for i in range(len(beta_config)):
        key = f"beta_{i}"
        if key in beta_config and "start_node_id" in beta_config[key]:
            beta_positions.append(beta_config[key]["start_node_id"])

    # Extract flag positions for both teams
    alpha_flag_positions = config.get("game", {}).get("flags", {}).get("alpha_positions", [])
    beta_flag_positions = config.get("game", {}).get("flags", {}).get("beta_positions", [])

    # Extract graph name
    graph_name = config.get("environment", {}).get("graph_name")

    return alpha_positions, beta_positions, alpha_flag_positions, beta_flag_positions, graph_name


@typechecked
def apply_game_rule_overrides(config: Dict, game_rule_path: str, debug: Optional[bool] = False) -> Dict:

    # Check if game rule is in the config.
    if "game" not in config or "rule" not in config["game"]:
        warning("No game rule found in the configuration. Skipping game rule overrides.")
        return config
    game_rule_name = config["game"]["rule"]
    game_rule_file = os.path.join(game_rule_path, f"{game_rule_name}.yml")
    try:
        gr = read_yml_file(game_rule_file, debug=debug).pop("gamerule", {})
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
                config[key] = recursive_update(config[key], value, force=True, debug=debug)
            else:
                if key in config:
                    if debug:
                        warning(f"Overriding key '{key}': {config[key]} -> {value}")
                else:
                    if debug:
                        info(f"Key '{key}' not found in original config. Adding with value: {value}")
                config[key] = value

    # --- Override the agents section ---
    if "agents" in gr:
        agents_overrides = gr["agents"]

        # Process alpha team overrides.
        if "alpha_global" in agents_overrides:
            alpha_override = agents_overrides["alpha_global"]
            if "agents" in config:
                # Override global alpha settings.
                if "alpha_global" in config["agents"]:
                    old_value = config["agents"]["alpha_global"]
                    new_value = alpha_override.copy()
                    if old_value != new_value:
                        if debug:
                            warning(f"Overriding agents.alpha_global: {old_value} -> {new_value}")
                        config["agents"]["alpha_global"] = new_value
                # Override each individual alpha agent.
                if "alpha_config" in config["agents"]:
                    for key, a_conf in config["agents"]["alpha_config"].items():
                        old_value = a_conf.copy()
                        start_node = a_conf.get("start_node_id")
                        # Preserve any existing radii settings
                        capture_radius = a_conf.get("capture_radius")
                        tagging_radius = a_conf.get("tagging_radius")
                        
                        new_conf = alpha_override.copy()
                        if start_node is not None:
                            new_conf["start_node_id"] = start_node
                        if capture_radius is not None:
                            new_conf["capture_radius"] = capture_radius
                        if tagging_radius is not None:
                            new_conf["tagging_radius"] = tagging_radius
                        if old_value != new_conf:
                            if debug:
                                warning(f"Overriding agents.alpha_config.{key}: {old_value} -> {new_conf}")
                            config["agents"]["alpha_config"][key] = new_conf

        # Process beta team overrides.
        if "beta_global" in agents_overrides:
            beta_override = agents_overrides["beta_global"]
            if "agents" in config:
                # Override global beta settings.
                if "beta_global" in config["agents"]:
                    old_value = config["agents"]["beta_global"]
                    new_value = beta_override.copy()
                    if old_value != new_value:
                        if debug:
                            warning(f"Overriding agents.beta_global: {old_value} -> {new_value}")
                        config["agents"]["beta_global"] = new_value
                # Override each individual beta agent.
                if "beta_config" in config["agents"]:
                    for key, d_conf in config["agents"]["beta_config"].items():
                        old_value = d_conf.copy()
                        start_node = d_conf.get("start_node_id")
                        # Preserve any existing radii settings
                        capture_radius = d_conf.get("capture_radius")
                        tagging_radius = d_conf.get("tagging_radius")
                        
                        new_conf = beta_override.copy()
                        if start_node is not None:
                            new_conf["start_node_id"] = start_node
                        if capture_radius is not None:
                            new_conf["capture_radius"] = capture_radius
                        if tagging_radius is not None:
                            new_conf["tagging_radius"] = tagging_radius
                        if old_value != new_conf:
                            if debug:
                                warning(f"Overriding agents.beta_config.{key}: {old_value} -> {new_conf}")
                            config["agents"]["beta_config"][key] = new_conf
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
    if debug:
        success("Read original config file successfully")

    # Apply overrides from your rules directory
    config = apply_game_rule_overrides(
        original_config,
        dirs["rules"],
        debug=debug,
    )
    if debug:
        success("Applied game rule overrides")

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
    # if debug:
    #     success(f"Visualization Engine: {VIS_ENGINE}")

    # Extract window size from config
    vis_config = config.get("visualization", {})
    window_size = vis_config.get("window_size", [1280, 720])

    # Create a new context with window size
    vis_kwargs = {
        "width": window_size[0] if isinstance(window_size, list) and len(window_size) > 0 else 1280,
        "height": window_size[1] if isinstance(window_size, list) and len(window_size) > 1 else 720,
    }
    ctx = gamms.create_context(vis_engine=VIS_ENGINE, vis_kwargs=vis_kwargs)
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