from typing import Any, Dict, List, Optional, Tuple, Union
from typeguard import typechecked
from pathlib import Path
import networkx as nx
import pickle
import yaml
import os

try:
    from lib.core.console import *
    from lib.utils.graph_utils import cast_to_multidigraph, convert_gml_to_multidigraph
except ImportError:
    from ..core.console import *
    from ..utils.graph_utils import cast_to_multidigraph, convert_gml_to_multidigraph

# Optional dependency for DSG support
try:
    # import spark_dsg as dsg
    HAS_SPARK_DSG = False
except ImportError:
    dsg = None
    HAS_SPARK_DSG = False


@typechecked
def get_directories(root_dir: str) -> dict:
    """
    Build and return a dictionary of common directories.
    """
    return {
        "config": os.path.join(root_dir, "config"),
        "graph": os.path.join(root_dir, "graphs"),
        "rules": os.path.join(root_dir, "config/rules"),
        "result": os.path.join(root_dir, "data/result"),
    }


@typechecked
def export_graph_config(config: dict, dirs: dict) -> nx.MultiDiGraph:
    """
    Load the graph from a pickle file specified in the configuration.
    """
    graph_name = config["environment"]["graph_name"]
    graph_path = os.path.join(dirs["graph"], graph_name)
    G = export_graph_generic(graph_path)
    if not isinstance(G, nx.MultiDiGraph):
        warning(f"Graph {graph_name} is not a MultiDiGraph!")
    debug(f"Loaded graph from config: {graph_name}")
    return G


@typechecked
def _annotate_graph_source(graph: nx.MultiDiGraph, filename: str) -> None:
    """
    Attach source-file metadata to graph.graph so runtime caches can reuse results
    across newly loaded graph objects from the same file.
    """
    try:
        source_path = str(Path(filename).resolve())
        stat = os.stat(source_path)
        source_token = f"{source_path}:{stat.st_mtime_ns}:{stat.st_size}"
    except Exception:
        source_path = str(filename)
        source_token = source_path

    try:
        graph.graph["__graph_source_path"] = source_path
        graph.graph["__graph_source_token"] = source_token
    except Exception:
        # Non-fatal; graph-level cache will still work per-object
        pass


@typechecked
def export_graph_pkl(filename: str) -> nx.MultiDiGraph:
    """
    Load and return a NetworkX MultiDiGraph from a pickled file.

    This function verifies that the specified file exists, then attempts to load
    a pickled graph from the file. If the file is missing or the unpickling fails,
    an appropriate exception is raised and an error is logged.

    Parameters:
        filename (str): The path to the pickle file containing the graph.

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
        G = cast_to_multidigraph(G)
    debug(f"Graph loaded from {filename} with {len(G.nodes)} nodes.")
    return G


@typechecked
def export_all_graphs_pkl(path: str) -> Union[bool, Dict[str, nx.Graph]]:
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
            num_nodes = len(getattr(graph, "nodes", lambda: [])())
            debug(f"Graph loaded from {path} with {num_nodes} nodes.")
            if not isinstance(graph, nx.MultiDiGraph):
                warning(f"Graph {path} is not a MultiDiGraph.")
                graph = cast_to_multidigraph(graph)
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
                    debug(f"Graph loaded from {file_path} with {num_nodes} nodes.")
                    if not isinstance(graph, nx.MultiDiGraph):
                        warning(f"Graph {file_path} is not a MultiGraph.")
                        graph = cast_to_multidigraph(graph)
                    loaded_graphs[filename] = graph
                except Exception as e:
                    error(f"Error loading graph from {file_path}: {e}")
        return loaded_graphs

    else:
        error(f"Provided path is neither a file nor a directory: {path}.")
        return False


@typechecked
def export_graph_dsg(path: Union[str, Path]) -> nx.MultiDiGraph:
    """
    Load a Dynamic Scene Graph (DSG) file and extract its Places layer as a NetworkX MultiDiGraph.

    This function checks that the DSG file exists, attempts to load it, and then converts the
    Places layer (typically layer 3) into a NetworkX MultiDiGraph—computing 3D Euclidean lengths
    for edges. If any step fails, an appropriate exception is raised and an error is logged.

    Parameters:
        path (str | Path): Path to the DSG file on disk.

    Returns:
        nx.MultiDiGraph: The Places subgraph, with node attributes ('id', 'x', 'y', 'z') and
                         edge attributes ('id', 'length').

    Raises:
        FileNotFoundError: If the file does not exist at `path`.
        Exception: If loading the DSG or converting to a MultiDiGraph fails.
    """
    from math import sqrt

    # Check if spark_dsg is available
    if not HAS_SPARK_DSG:
        error("spark_dsg module is not installed. Please install it to use this function.")
        raise ImportError("spark_dsg module is required for DSG operations.")

    warning("Calling export_graph_dsg() function, this function is not tested yet, please use with caution.")
    # Normalize path and verify existence
    dsg_path = Path(path)
    if not dsg_path.exists():
        error(f"DSG file not found at {dsg_path}")
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
            nx_places = cast_to_multidigraph(nx_places)
        except Exception as e:
            error(f"Failed to cast Places graph to MultiDiGraph: {e}")
            raise Exception(f"Failed to cast Places graph to MultiDiGraph: {e}")

    debug(f"Exported DSG Places subgraph: {nx_places.number_of_nodes()} nodes, {nx_places.number_of_edges()} edges.")

    return nx_places


@typechecked
def export_graph_gml(filename: str, scale_factor: float = 1, offset_x: float = 0, offset_y: float = 0) -> nx.MultiDiGraph:
    """
    Load a GML file and convert it to a MultiDiGraph with spatial attributes.

    This function verifies that the specified GML file exists, loads it as a NetworkX graph,
    then converts it to a MultiDiGraph with 2D spatial coordinates and LineString edges
    matching the format expected by the simulator.

    Parameters:
        filename (str): The path to the GML file containing the graph.
        scale_factor (float): Scaling factor for coordinates. Defaults to 600000.
        offset_x (float): X coordinate offset. Defaults to 586000.
        offset_y (float): Y coordinate offset. Defaults to 4582000.

    Returns:
        nx.MultiDiGraph: The converted spatial multigraph.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: For errors during loading or converting the graph.
    """
    if not os.path.exists(filename):
        error(f"GML file does not exist at {filename}.")
        raise FileNotFoundError(f"GML file does not exist at {filename}.")

    try:
        G = nx.read_gml(filename)
        debug(f"GML file loaded from {filename} with {len(G.nodes)} nodes.")
    except Exception as e:
        error(f"Error loading GML file from {filename}: {e}")
        raise Exception(f"Error loading GML file from {filename}: {e}")

    # Convert to MultiDiGraph with spatial attributes
    return convert_gml_to_multidigraph(G, scale_factor, offset_x, offset_y)


@typechecked
def export_graph_generic(filename: str) -> nx.MultiDiGraph:
    """
    Load a generic graph file and convert it to a MultiDiGraph.

    This function verifies that the specified file exists, attempts to load it as a NetworkX graph,
    and then converts it to a MultiDiGraph if necessary.

    Parameters:
        filename (str): The path to the graph file.

    Returns:
        nx.MultiDiGraph: The loaded and converted graph.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: For errors during loading or converting the graph.
    """
    if not os.path.exists(filename):
        error(f"Graph file does not exist at {filename}.")
        raise FileNotFoundError(f"Graph file does not exist at {filename}.")

    # Check the file extension
    file_extension = os.path.splitext(filename)[1].lower()
    if file_extension == ".gml":
        graph = export_graph_gml(filename)
        _annotate_graph_source(graph, filename)
        return graph
    elif file_extension == ".pkl":
        graph = export_graph_pkl(filename)
        _annotate_graph_source(graph, filename)
        return graph
    elif file_extension == ".json":
        graph = export_graph_dsg(filename)
        _annotate_graph_source(graph, filename)
        return graph
    else:
        error(f"Unsupported graph file format: {file_extension}. Supported formats are .gml, .pkl, and .json.")
        raise Exception(f"Unsupported graph file format: {file_extension}. Supported formats are .gml, .pkl, and .json.")
