from typing import Any, Dict, List, Optional, Tuple
from typeguard import typechecked
from pathlib import Path
import networkx as nx
import pickle
import yaml
import os

try:
    from lib.core.core import *
    from lib.utils.graph_utils import cast_to_multidigraph
except ImportError:
    from ..core.core import *
    from ..utils.graph_utils import cast_to_multidigraph


@typechecked
def read_yml_file(file_path: str, search_if_not_found: bool = True, config_dir: Optional[str] = None, debug: Optional[bool] = False) -> Dict:
    """
    Reads a YAML file from the given file path and returns its contents as a dictionary.
    If the file is not found directly, it can optionally search for it recursively.

    Parameters:
    -----------
    file_path : str
        The path to the YAML file.
    search_if_not_found : bool
        If True and the file is not found at the provided path, search for it recursively.
    config_dir : Optional[str]
        The root directory to start searching from if needed. If None, uses directory of file_path.
    debug : Optional[bool]
        If True, debug messages will be printed during the process.

    Returns:
    --------
    Dict
        The configuration dictionary loaded from the YAML file.
    """
    try:
        # Try to open the file directly first
        try:
            with open(file_path, "r") as f:
                config = yaml.safe_load(f)
            success(f"Successfully loaded config from {file_path}")
            return config
        except FileNotFoundError:
            # If file not found and search is enabled
            if search_if_not_found:
                dbg(f"File not found at {file_path}, searching recursively...", debug)

                # Get just the filename in case a path was provided
                file_name = os.path.basename(file_path)

                # Determine search root directory
                if config_dir is not None:
                    search_root = config_dir
                else:
                    # If no config_dir provided, use parent directory of file_path
                    search_root = os.path.dirname(file_path)
                    if not search_root:  # If file_path was just a filename
                        search_root = os.getcwd()

                # Verify search directory exists
                if not os.path.exists(search_root):
                    error(f"Search root directory {search_root} does not exist.")
                    raise FileNotFoundError(f"Search root directory {search_root} does not exist.")

                info(f"Searching for {file_name} in {search_root} and subdirectories...", debug)

                # Search recursively for the file, including all subdirectories
                found_path = None
                for root, _, files in os.walk(search_root):
                    if file_name in files:
                        found_path = os.path.join(root, file_name)
                        success(f"Found config file at: {found_path}")
                        break

                if found_path:
                    # Read the found file
                    with open(found_path, "r") as f:
                        config = yaml.safe_load(f)
                    success(f"Successfully loaded config from {found_path}")
                    return config

                # If we get here, we didn't find the file
                error(f"Could not find {file_name} in {search_root} or its subdirectories.")
                raise FileNotFoundError(f"Could not find {file_name} in {search_root} or its subdirectories.")
            else:
                # If search is disabled, just raise the original error
                raise
    except Exception as e:
        if isinstance(e, FileNotFoundError):
            error(f"File not found: {file_path}")
        else:
            error(f"Error reading YAML file at {file_path}: {e}")
        raise Exception(f"Error reading YAML file: {e}")


@typechecked
def write_yaml_config(config: Dict, output_dir: str, filename: str) -> bool:
    filepath = os.path.join(output_dir, filename)
    if os.path.exists(filepath):
        warning(f"File {filepath} already exists. Skipping write.")
        return False
    try:
        with open(filepath, "w") as file:
            yaml.dump(config, file, default_flow_style=False)
        return True
    except Exception as e:
        error(f"Error writing config to file {filepath}: {e}")
        return False


@typechecked
def get_directories(root_dir: str) -> dict:
    """
    Build and return a dictionary of common directories.
    """
    return {
        "config": os.path.join(root_dir, "data/config"),
        "graph": os.path.join(root_dir, "data/graphs"),
        "rules": os.path.join(root_dir, "data/rules"),
        "result": os.path.join(root_dir, "data/result"),
    }


@typechecked
def export_graph_pkl_config(config: dict, dirs: dict, debug: bool = False) -> nx.MultiDiGraph:
    """
    Load the graph from a pickle file specified in the configuration.
    """
    graph_name = config["environment"]["graph_name"]
    graph_path = os.path.join(dirs["graph"], graph_name)
    G = export_graph_pkl(graph_path)
    if not isinstance(G, nx.MultiDiGraph):
        warning(f"Graph {graph_name} is not a MultiDiGraph!")
    success(f"Loaded graph: {graph_name}", debug)
    return G


@typechecked
def export_graph_pkl(filename: str, debug: bool = False) -> nx.MultiDiGraph:
    """
    Load and return a NetworkX MultiDiGraph from a pickled file.

    This function verifies that the specified file exists, then attempts to load
    a pickled graph from the file. If the file is missing or the unpickling fails,
    an appropriate exception is raised and an error is logged.

    Parameters:
        filename (str): The path to the pickle file containing the graph.
        debug (bool): Flag indicating whether to print debug messages. Defaults to False.

    Returns:
        nx.MultiDiGraph: The loaded directed multigraph.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: For errors during unpickling the graph.
    """
    if not os.path.exists(filename):
        error(f"Graph file does not exist at {filename}.")
        raise FileNotFoundError(f"Graph file does not exist at {filename}.")
    with open(filename, "rb") as f:
        try:
            G = pickle.load(f)
        except Exception as e:
            error(f"Error loading graph from {filename}: {e}")
            raise Exception(f"Error loading graph from {filename}: {e}")
    # Ensure the graph is a MultiDiGraph
    if not isinstance(G, nx.MultiDiGraph):
        G = cast_to_multidigraph(G, debug)
    success(f"Graph loaded from {filename} with {len(G.nodes)} nodes.", debug)
    return G


@typechecked
def export_all_graphs_pkl(path: str, debug: Optional[bool] = False) -> Union[bool, Dict[str, nx.Graph]]:
    """
    Exports graph objects from pickle (.pkl) files.

    If the provided path is a directory, the function will load all .pkl files in that directory.
    If the provided path is a file, the function will load that .pkl file (if it has a .pkl extension).

    Parameters:
    -----------
    path : str
        The directory or file path from which to load graph objects.

    Returns:
    --------
    dict or bool
        A dictionary mapping file names to loaded graph objects if successful.
        If the provided path does not exist or is invalid, returns False.
    """
    # Check if the path exists
    if not os.path.exists(path):
        error(f"Path does not exist: {path}. Aborting process.")
        return False

    loaded_graphs = {}

    # If path is a file, process that file only
    if os.path.isfile(path):
        if not path.endswith(".pkl"):
            error(f"Provided file is not a .pkl file: {path}. Aborting process.")
            return False

        try:
            with open(path, "rb") as f:
                graph = pickle.load(f)
            # Assuming the graph has an attribute 'nodes' that returns a list or set of nodes.
            num_nodes = len(getattr(graph, "nodes", lambda: [])())
            success(f"Graph loaded from {path} with {num_nodes} nodes.", debug)
            if not isinstance(graph, nx.MultiDiGraph):
                warning(f"Graph {path} is not a MultiDiGraph.")
                graph = cast_to_multidigraph(graph, debug)
            loaded_graphs[os.path.basename(path)] = graph
        except Exception as e:
            error(f"Error loading graph from {path}: {e}")
        return loaded_graphs

    # If path is a directory, iterate over files in the directory
    elif os.path.isdir(path):
        for filename in os.listdir(path):
            if filename.endswith(".pkl"):
                file_path = os.path.join(path, filename)
                if not os.path.exists(file_path):
                    error(f"Graph file does not exist at {file_path}. Skipping.")
                    continue
                try:
                    with open(file_path, "rb") as f:
                        graph = pickle.load(f)
                    num_nodes = len(getattr(graph, "nodes", lambda: [])())
                    success(f"Graph loaded from {file_path} with {num_nodes} nodes.", debug)
                    if not isinstance(graph, nx.MultiDiGraph):
                        warning(f"Graph {file_path} is not a MultiGraph.")
                        graph = cast_to_multidigraph(graph, debug)
                    loaded_graphs[filename] = graph
                except Exception as e:
                    error(f"Error loading graph from {file_path}: {e}")
        return loaded_graphs

    else:
        error(f"Provided path is neither a file nor a directory: {path}.")
        return False


@typechecked
def export_graph_dsg(path: Union[str, Path], debug: bool = False) -> nx.MultiDiGraph:
    """
    Load a Dynamic Scene Graph (DSG) file and extract its Places layer as a NetworkX MultiDiGraph.

    This function checks that the DSG file exists, attempts to load it, and then converts the
    Places layer (typically layer 3) into a NetworkX MultiDiGraph—computing 3D Euclidean lengths
    for edges. If any step fails, an appropriate exception is raised and an error is logged.

    Parameters:
        path (str | Path): Path to the DSG file on disk.
        debug (bool): If True, print debug or success messages. Defaults to False.

    Returns:
        nx.MultiDiGraph: The Places subgraph, with node attributes ('id', 'x', 'y', 'z') and
                         edge attributes ('id', 'length').

    Raises:
        FileNotFoundError: If the file does not exist at `path`.
        Exception: If loading the DSG or converting to a MultiDiGraph fails.
    """
    from math import sqrt
    try:
        import spark_dsg as dsg
    except ImportError:
        error("spark_dsg module is not installed. Please install it to use this function.")
        raise ImportError("spark_dsg module is required for DSG operations.")
    
    warning("Calling export_graph_dsg() function, this function is not tested yet, please use with caution.", True)
    # Normalize path and verify existence
    dsg_path = Path(path)
    if not dsg_path.exists():
        error(f"DSG file not found at {dsg_path}")  # replace with your logging function
        raise FileNotFoundError(f"DSG file not found at {dsg_path}")

    # Load the DSG from file
    try:
        scene_graph = dsg.DynamicSceneGraph.load(str(dsg_path))
    except Exception as e:
        error(f"Failed to load DSG from {dsg_path}: {e}")
        raise Exception(f"Failed to load DSG from {dsg_path}: {e}")

    # Extract the Places layer (DSG layer enum)
    try:
        places_layer = scene_graph.get_layer(dsg.DsgLayers.PLACES)
    except Exception as e:
        error(f"Could not retrieve Places layer: {e}")
        raise Exception(f"Could not retrieve Places layer: {e}")

    # Build a directed graph of places
    nx_places = nx.DiGraph()

    # Add nodes with (id, x, y, z) attributes
    for node in places_layer.nodes:
        attrs = node.attributes
        pos = attrs.position  # expected as a 3-element sequence
        nx_places.add_node(node.id.value, id=str(node.id), x=pos[0], y=pos[1], z=pos[2])

    # Add directed edges, computing Euclidean length from node coordinates
    for edge in places_layer.edges:
        source_data = nx_places.nodes[edge.source]
        target_data = nx_places.nodes[edge.target]
        dx = source_data["x"] - target_data["x"]
        dy = source_data["y"] - target_data["y"]
        dz = source_data["z"] - target_data["z"]
        distance = sqrt(dx * dx + dy * dy + dz * dz)

        nx_places.add_edge(edge.source, edge.target, id=(source_data["id"], target_data["id"]), length=distance)

    # If the resulting graph is not a MultiDiGraph, cast it
    if not isinstance(nx_places, nx.MultiDiGraph):
        try:
            nx_places = cast_to_multidigraph(nx_places, debug)
        except Exception as e:
            error(f"Failed to cast Places graph to MultiDiGraph: {e}")
            raise Exception(f"Failed to cast Places graph to MultiDiGraph: {e}")

    # Log success
    success(f"Exported DSG Places subgraph: {nx_places.number_of_nodes()} nodes, " f"{nx_places.number_of_edges()} edges.", debug)

    return nx_places
