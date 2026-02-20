from typing import Dict, Any, Optional, Union, List, Set, Tuple
from typeguard import typechecked
from pathlib import Path


try:
    from ..core.console import *
    from distribution import *
except ImportError:
    from lib.core.console import *
    from lib.config.distribution import *


@typechecked
def recursive_update(default: Dict, override: Dict, force: bool) -> Dict:
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
                        if default[key] is not None:
                            warning(f"Overriding key '{key}': {default[key]} -> {value}")
                        default[key] = value
                else:
                    info(f"Key '{key}' not found in original config. Adding with value: {value}")
                    default[key] = value
            else:
                # Force is False: Only override if key is missing or its value is None or "Error".
                if key in default:
                    current = default.get(key)
                    if current is None or current == "Error":
                        info(f"Key '{key}' is missing or invalid (current: {current}). Setting to: {value}")
                        default[key] = value
                else:
                    info(f"Key '{key}' not found in original config. Adding with value: {value}")
                    default[key] = value
    return default


@typechecked
def generate_position_with_distribution(graph: nx.Graph, num_nodes: int, dist_type: str, param, center_node: Optional[int] = None) -> Tuple[list, Optional[int]]:
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
    elif dist_type == "normal_t":
        try:
            mean_d, std, trunc = param
        except Exception as e:
            error(f"Invalid parameter for normal_t distribution: {param}. Error: {e}")
            return None, center_node
        positions = distribute_normal_truncated(graph, center_node, num_nodes, mean_distance=mean_d, std_dev=std, n_std_cutoff=trunc)
        print(positions)
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

    info(f"Generated {num_nodes} positions using distribution: {dist_type}")
    return positions, center_node


@typechecked
def generate_single_component(
    graph: nx.Graph,
    component_type: str,
    num: int,
    distribution: Tuple[str, Union[int, List]],
    center_node: int,
) -> Tuple[Tuple[str, List], List[int]]:
    """
    Generates positions for a single component using specified distribution parameters.
    Returns both a compact dictionary representation of the generation parameters and
    the actual list of generated node positions.

    Args:
        graph: NetworkX graph object on which to generate positions
        component_type: Component identifier in format "team_component" (e.g., "red_flags", "blue_agents", "green_towers")
        num: Number of positions to generate
        distribution: Tuple of (distribution_type, parameters) e.g., ("uniform", 2) or ("normal", [6, 2])
        center_node: Center node ID from which distribution distances are calculated

    Returns:
        Tuple of:
        - (key, data): Tuple of component key and list [center, num, dist_type, ...params]
        - positions: List of generated node IDs
    """
    # Generate positions using the specified distribution
    dist_type, params = distribution
    positions, _ = generate_position_with_distribution(
        graph=graph,
        num_nodes=num,
        dist_type=dist_type,
        param=params,
        center_node=center_node,
    )

    # Format params compactly
    if not isinstance(params, list):
        params = [params]

    # Parse component_type
    team, component = component_type.split("_")

    # Use first two characters if available, otherwise fall back to first character
    team_key = team[:2] if len(team) > 1 else team[0]
    comp_key = component[:2] if len(component) > 1 else component[0]

    # Return key and data separately
    key = f"{team_key}{comp_key}"
    data = [center_node, num, dist_type] + params

    return (key, data), positions


@typechecked
def generate_config_label(identifier: str = "") -> Dict[str, str]:
    """
    Generates timestamp and unique hash ID for configuration tracking.

    Args:
        identifier: String to combine with timestamp for unique ID generation

    Returns:
        Dict with "id" (10-character hash) and "ts" (timestamp string)
    """
    import hashlib
    import time
    import datetime

    # Combine identifier with time for uniqueness
    unique_string = f"{identifier}-{time.time()}"

    # Generate unique ID from combined string
    config_id = hashlib.md5(unique_string.encode()).hexdigest()[:10]

    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {"id": config_id, "ts": timestamp}


@typechecked
def binary_to_compact_hex(binary_str: str) -> str:
    """
    Convert binary string to hex with length prefix to preserve leading zeros.
    Format: "LENGTH:HEXVALUE"

    Example: "00000000101011001" (17 bits) -> "17:159"
    """
    bit_length = len(binary_str)
    hex_value = hex(int(binary_str, 2))[2:]  # Remove '0x' prefix
    return f"{bit_length}:{hex_value}"


@typechecked
def compact_hex_to_binary(compact_hex: str) -> str:
    """
    Convert compact hex back to original binary string with leading zeros.

    Example: "17:159" -> "00000000101011001"
    """
    bit_length, hex_value = compact_hex.split(":")
    bit_length = int(bit_length)
    if hex_value == "0":  # Special case for all zeros
        return "0" * bit_length
    binary = bin(int(hex_value, 16))[2:]  # Convert hex to binary, remove '0b'
    return binary.zfill(bit_length)  # Pad with leading zeros
