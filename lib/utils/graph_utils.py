import os
import pickle
import random
import numpy as np
import networkx as nx
from copy import deepcopy
from typeguard import typechecked
from scipy.spatial import Delaunay
from shapely.geometry import LineString


try:
    from lib.core.core import *
    import lib.visual.graph_visualizer as gfvis
except ModuleNotFoundError:
    from ..core.core import *
    from visual import graph_visualizer as gfvis


@typechecked
def export_graph(filename: str, debug: bool = False) -> nx.MultiDiGraph:
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
            if filename.endswith(".json"):
                G = export_places_graph_from_dsg(filename, debug)
            else:
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
def export_places_graph_from_dsg(filename: str, debug: bool = False) -> nx.MultiDiGraph:
    from math import sqrt
    import spark_dsg as dsg
    # Check if the file name is a json file
    if not filename.endswith(".json"):
        error(f"Provided file is not a .json file: {filename}. Aborting process.")
        return False
    G = dsg.DynamicSceneGraph.load(str(filename))

    # Get the places layer (usually layer 3)
    places_layer = G.get_layer(dsg.DsgLayers.PLACES)

    nx_places = nx.DiGraph()  # Use DiGraph

    # Add nodes
    for node in places_layer.nodes:
        attrs = node.attributes
        pos = attrs.position
        nx_places.add_node(node.id.value, id=str(node.id), x=pos[0],y=pos[1],z=pos[2])

    # Add edges
    for edge in places_layer.edges:
        ns = nx_places.nodes[edge.source]
        nt = nx_places.nodes[edge.target]
        dist = sqrt((ns.get('x') - nt.get('x'))**2 + (ns.get('y') - nt.get('y'))**2 + (ns.get('z') - nt.get('z'))**2)
        nx_places.add_edge(edge.source, edge.target, id = (ns.get('id'),nt.get('id')), length = dist)

    if not isinstance(nx_places, nx.MultiDiGraph):
        nx_places = cast_to_multidigraph(nx_places)
    # Print basic info about the places graph
    success(f"DSG Places subgraph has {nx_places.number_of_nodes()} nodes and {nx_places.number_of_edges()} edges", debug)

    return nx_places


@typechecked
def export_all_graphs(path: str, debug: Optional[bool] = False) -> Union[bool, Dict[str, nx.Graph]]:
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
def cast_to_multidigraph(G: nx.DiGraph, debug: Optional[bool] = False) -> nx.MultiDiGraph:
    """
    Convert a DiGraph to a MultiDiGraph ensuring that if an edge (u, v)
    exists, a reverse edge (v, u) also exists. If the reverse edge is missing,
    it is added. If the edge data contains a 'linestring', its coordinates
    are reversed for the reverse edge.

    Also prints the number of reverse edges that were added.

    Args:
        G (nx.DiGraph): Input directed graph.

    Returns:
        nx.MultiDiGraph: A MultiDiGraph with ensured bidirectional edges.
    """
    MD = nx.MultiDiGraph()

    # Add all nodes with attributes
    MD.add_nodes_from(G.nodes(data=True))

    # Counter for reverse edges added
    added_count = 0

    # Process each edge in the original graph
    for u, v, data in G.edges(data=True):
        # Add original edge
        MD.add_edge(u, v, **data)

        # Check if the reverse edge (v, u) exists in the original graph
        if not G.has_edge(v, u):
            # Create a deep copy of the data for the reverse edge
            rev_data = deepcopy(data)
            # If a 'linestring' is present, reverse its coordinates
            if "linestring" in rev_data and isinstance(rev_data["linestring"], LineString):
                rev_data["linestring"] = LineString(rev_data["linestring"].coords[::-1])
            MD.add_edge(v, u, **rev_data)
            added_count += 1

    info(f"Added {added_count} reverse edge(s) to ensure bidirectionality.", debug)
    success(f"Converted DiGraph to MultiDiGraph with {MD.number_of_nodes()} nodes and {MD.number_of_edges()} edges.", debug)
    return MD


@typechecked
def generate_simple_grid(rows: int = 10, cols: int = 10, debug: bool = False) -> nx.MultiDiGraph:
    """
    Generate a grid graph with the specified number of rows and columns as a MultiDiGraph.

    The graph is first created with nodes identified by tuple coordinates.
    Then, the nodes are relabeled as integers (from 0 to n-1), and each node is
    assigned positional attributes: 'x' for the column index and 'y' for the row index.
    Finally, the graph is converted to a MultiDiGraph.

    Parameters:
        rows (int): The number of rows in the grid.
        cols (int): The number of columns in the grid.
        debug (bool): If True, prints debug information about the generated graph.

    Returns:
        nx.MultiDiGraph: A grid graph with integer node labels, positional attributes,
                         and represented as a MultiDiGraph.
    """
    # Create a grid graph with nodes as tuples.
    G = nx.grid_2d_graph(rows, cols)
    # Convert node labels to integers.
    G = nx.convert_node_labels_to_integers(G)

    # Assign positional attributes for visualization or spatial reference.
    for node in G.nodes():
        row = node // cols
        col = node % cols
        G.nodes[node]["x"] = col
        G.nodes[node]["y"] = row

    # Convert the undirected grid graph to a directed multigraph.
    G_multi = nx.MultiDiGraph(G)
    success(f"Generated grid with {G_multi.number_of_nodes()} nodes and {G_multi.number_of_edges()} edges.", debug)
    return G_multi


@typechecked
def generate_lattice_grid(rows: int = 10, cols: int = 10, debug: bool = False) -> nx.MultiDiGraph:
    """
    Generate a lattice grid graph (including diagonals) with the specified
    number of rows and columns as a MultiDiGraph.

    Parameters:
        rows (int): The number of rows in the grid.
        cols (int): The number of columns in the grid.
        debug (bool): If True, prints debug information about the generated graph.

    Returns:
        nx.MultiDiGraph: A lattice graph with integer node labels, positional
                         attributes, and represented as a MultiDiGraph.
    """
    # Start with an undirected 2D grid
    G = nx.grid_2d_graph(rows, cols)

    # Add diagonal edges: for each cell, connect (r,c) to (r+1,c+1) and (r+1,c-1)
    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows and c + 1 < cols:
                G.add_edge((r, c), (r + 1, c + 1))
            if r + 1 < rows and c - 1 >= 0:
                G.add_edge((r, c), (r + 1, c - 1))

    # Relabel nodes to integers 0..n-1
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")

    # Assign positional attributes
    for node in G.nodes():
        row = node // cols
        col = node % cols
        G.nodes[node]["x"] = col
        G.nodes[node]["y"] = row

    # Convert to a directed multigraph
    G_multi = nx.MultiDiGraph(G)

    success(f"Generated lattice grid with {G_multi.number_of_nodes()} nodes " f"and {G_multi.number_of_edges()} edges.", debug)

    return G_multi


@typechecked
def generate_triangular_lattice_graph(rows: int = 10, cols: int = 10, debug: bool = False) -> nx.MultiDiGraph:
    """
    Generate a triangular-lattice graph with the specified number of rows and columns
    as a MultiDiGraph.  Each node (r,c) is connected to its right neighbor, down neighbor,
    and down-left neighbor, producing a mesh of triangles.

    Parameters:
        rows (int): Number of rows.
        cols (int): Number of columns.
        debug (bool): If True, prints debug info.

    Returns:
        nx.MultiDiGraph: Triangular-lattice as a directed multigraph with integer labels
                         and 'x','y' positional attributes.
    """
    # Start from an empty undirected graph
    G = nx.Graph()

    # Add nodes
    for r in range(rows):
        for c in range(cols):
            G.add_node((r, c))

    # Add edges for triangular tiling
    for r in range(rows):
        for c in range(cols):
            # right neighbor
            if c + 1 < cols:
                G.add_edge((r, c), (r, c + 1))
            # down neighbor
            if r + 1 < rows:
                G.add_edge((r, c), (r + 1, c))
            # down-left neighbor
            if r + 1 < rows and c - 1 >= 0:
                G.add_edge((r, c), (r + 1, c - 1))

    # Relabel to integers
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")

    # Assign x, y attributes
    for node in G.nodes():
        row = node // cols
        col = node % cols
        G.nodes[node]["x"] = col
        G.nodes[node]["y"] = row

    # Convert to MultiDiGraph
    G_multi = nx.MultiDiGraph(G)

    success(f"Generated triangular lattice with {G_multi.number_of_nodes()} nodes " f"and {G_multi.number_of_edges()} edges.", debug)

    return G_multi


@typechecked
def generate_random_delaunay_graph(n_points: int = 100, side: float = 1.0, seed: int = 0, debug: bool = False) -> nx.MultiDiGraph:
    """
    Generate n_points uniformly in the square [0, side] × [0, side],
    compute their Delaunay triangulation, and return a graph connecting every
    pair of points that share a triangle edge.

    Parameters:
        n_points (int): Number of random 2D points.
        side (float): Length of the square’s side (origin is fixed at (0,0)).
        seed (int): RNG seed for reproducibility.
        debug (bool): If True, prints node/edge counts.

    Returns:
        nx.MultiDiGraph: nodes 0..n_points-1 with 'x','y' attrs and triangulation edges.
    """
    # 1) Sample points in [0, side]^2
    rng = np.random.default_rng(seed)
    points = rng.random((n_points, 2)) * side

    # 2) Delaunay triangulation
    tri = Delaunay(points)

    # 3) Build undirected graph
    G = nx.Graph()
    for idx, (x, y) in enumerate(points):
        G.add_node(int(idx), x=float(x), y=float(y))

    # 4) Add triangle edges
    for simplex in tri.simplices:
        i, j, k = [int(val) for val in simplex]  # Convert NumPy integers to Python integers
        G.add_edge(i, j)
        G.add_edge(j, k)
        G.add_edge(k, i)

    # 5) Convert to MultiDiGraph
    G_multi = nx.MultiDiGraph(G)

    success(f"Generated Delaunay triangulation with {G_multi.number_of_nodes()} nodes, " f"{G_multi.number_of_edges()} edges.", debug)

    return G_multi


@typechecked
def renumber_graph(G: nx.MultiDiGraph, debug: bool = False) -> nx.MultiDiGraph:
    """
    Renumber the nodes of a graph to have consecutive integer IDs starting from 0.

    This function creates a new graph with node IDs renumbered, while preserving the
    original node attributes and edge data. If an edge has an 'id' attribute, it is updated
    to a new sequential ID.

    Parameters:
        G (nx.MultiDiGraph): The input multigraph with arbitrary node IDs.
        debug (bool): If True, prints debug information about the renumbered graph.

    Returns:
        nx.MultiDiGraph: A new multigraph with nodes renumbered from 0 to n-1.
    """
    try:
        H = nx.MultiDiGraph()
        # Map old node IDs to new node IDs.
        mapping = {old_id: new_id for new_id, old_id in enumerate(G.nodes())}

        # Add nodes with their corresponding attributes.
        for old_id in G.nodes():
            new_id = mapping[old_id]
            H.add_node(new_id, **G.nodes[old_id])

        # Add edges with remapped node IDs and update edge 'id' if present.
        edge_id = 0
        for u, v, data in G.edges(data=True):
            new_u = mapping[u]
            new_v = mapping[v]
            edge_data = data.copy()
            if "id" in edge_data:
                edge_data["id"] = edge_id
                edge_id += 1
            H.add_edge(new_u, new_v, **edge_data)
        success(f"Graph renumbered with {len(H.nodes)} nodes.", debug)
        return H

    except Exception as e:
        error(f"Error in renumber_graph: {e}")
        raise Exception(f"Error in renumber_graph: {e}")


@typechecked
def reduce_graph_to_size(G: nx.MultiDiGraph, node_limit: int, debug: bool = False) -> nx.MultiDiGraph:
    """
    Reduce a MultiDiGraph to at most a given number of nodes while preserving connectivity.

    This function first checks if the graph already meets the node limit. If not, it identifies
    the largest weakly connected component of the MultiDiGraph. If that component is still too large,
    it uses a breadth-first search (BFS) starting from a random node within the component to extract
    a subgraph of the desired size. Finally, the subgraph is renumbered to have consecutive node IDs.

    Parameters:
        G (nx.MultiDiGraph): The original directed multigraph.
        node_limit (int): The maximum number of nodes desired in the reduced graph.
        debug (bool): If True, prints debug information.

    Returns:
        nx.MultiDiGraph: A reduced and renumbered MultiDiGraph with at most node_limit nodes.

    Raises:
        Exception: Propagates any errors encountered during graph reduction.
    """
    try:
        # If the graph is already small enough, renumber and return it.
        if G.number_of_nodes() <= node_limit:
            info(f"Graph has {G.number_of_nodes()} nodes which is within the limit.", debug)
            return renumber_graph(G)

        # Identify the largest weakly connected component in the MultiDiGraph.
        largest_cc = max(nx.weakly_connected_components(G), key=len)
        info(f"Largest weakly connected component has {len(largest_cc)} nodes.", debug)

        if len(largest_cc) <= node_limit:
            sub_G = G.subgraph(largest_cc).copy()
            info(f"Using largest component as it is within the node limit.", debug)
            return renumber_graph(sub_G)

        # Otherwise, perform a BFS from a random node in the largest component.
        start_node = random.choice(list(largest_cc))
        subgraph_nodes: Set[Any] = {start_node}
        frontier: List[Any] = [start_node]

        while len(subgraph_nodes) < node_limit and frontier:
            current = frontier.pop(0)
            # For MultiDiGraph, consider both successors and predecessors.
            neighbors = list(G.successors(current)) + list(G.predecessors(current))
            for neighbor in neighbors:
                if neighbor not in subgraph_nodes:
                    subgraph_nodes.add(neighbor)
                    frontier.append(neighbor)
                    info(f"Added node {neighbor}. Current subgraph size: {len(subgraph_nodes)}", debug)
                    if len(subgraph_nodes) >= node_limit:
                        break

        if not frontier and len(subgraph_nodes) < node_limit:
            warning("Frontier exhausted before reaching node limit; resulting subgraph may be smaller than desired.")

        sub_G = G.subgraph(subgraph_nodes).copy()
        reduced_graph = renumber_graph(sub_G)
        success(f"Graph reduced and renumbered to {reduced_graph.number_of_nodes()} nodes.", debug)
        return reduced_graph

    except Exception as e:
        error(f"Error reducing graph: {e}")
        raise Exception(f"Error reducing graph: {e}")


@typechecked
def compute_x_neighbors(G: nx.MultiDiGraph, nodes: Union[List[Any], Set[Any]], distance: int) -> Set[Any]:
    """
    Compute all nodes in MultiDiGraph G that are within a given distance from a set or list of nodes.

    For each node in the input, a breadth-first search (BFS) is performed up to the specified
    cutoff distance. The union of all nodes found (including the original nodes) is returned.

    Parameters:
        G (nx.MultiDiGraph): The input directed multigraph.
        nodes (Union[List[Any], Set[Any]]): A set or list of starting node IDs.
        distance (int): The maximum distance (number of hops) to search.
                        A distance of 0 returns only the input nodes.

    Returns:
        Set[Any]: A set of nodes that are within the given distance from any of the input nodes.
    """
    node_set = set(nodes)  # Convert input to a set if it's not already
    result: Set[Any] = set(node_set)

    for node in node_set:
        # Compute shortest path lengths up to the given cutoff.
        neighbors = nx.single_source_shortest_path_length(G, node, cutoff=distance)
        result.update(neighbors.keys())

    return result


@typechecked
def compute_node_dominance_region(node1: int, node2: int, speed1: int, speed2: int, G: nx.MultiDiGraph, debug: bool = False) -> Tuple[List[int], List[int], List[int], Dict[int, float]]:
    """
    Compute the dominance regions of two nodes based on their speeds on a MultiDiGraph.

    This function calculates which nodes in the graph are reached faster by either node1 or node2,
    given their respective speeds. It uses the shortest path lengths from each node to every other node
    in the graph (assuming uniform edge weights). A node is assigned to the region of the node that reaches
    it first (dominance region), or marked as contested if both reach it at the same time. An advantage index
    is computed as the difference in distances from the two source nodes.

    Parameters:
        node1 (int): The first source node.
        node2 (int): The second source node.
        speed1 (int): The speed associated with node1 (higher means faster).
        speed2 (int): The speed associated with node2.
        G (nx.MultiDiGraph): The directed multigraph where distances are computed (assumes unit weight for edges).
        debug (bool): A flag for enabling debug mode. No debug prints are included by default.

    Returns:
        Tuple containing:
          - List[int]: Nodes reached faster by node2.
          - List[int]: Nodes reached faster by node1.
          - List[int]: Nodes reached simultaneously (contested).
          - Dict[int, int]: A mapping from node ID to the advantage index (difference in path lengths).
    """
    # warning("This version of compute_node_dominance_region have reversed output and only handles speed of 1, please consider fixing before proceed.", True)
    dist1 = nx.single_source_shortest_path_length(G, source=node1)
    dist2 = nx.single_source_shortest_path_length(G, source=node2)

    region1: List[int] = []  # Nodes where node2 is faster (dominance region for node2)
    region2: List[int] = []  # Nodes where node1 is faster (dominance region for node1)
    contested: List[int] = []  # Nodes reached at the same time
    advantage_index: Dict[int, float] = {}  # Advantage index for each node

    for node in G.nodes():
        d1 = dist1.get(node, float("inf"))
        d2 = dist2.get(node, float("inf"))

        # Calculate travel times based on speeds.
        time1 = d1 / speed1
        time2 = d2 / speed2

        if time2 < time1:
            region1.append(node)
            advantage_index[node] = time1 - time2
            # Positive advantage means node2 can reach the node faster
        elif time1 < time2:
            region2.append(node)
            advantage_index[node] = time1 - time2
            # Negative advantage means node1 can reach the node faster
        else:
            contested.append(node)
            advantage_index[node] = 0

    return region1, region2, contested, advantage_index


# ==================================================================================
# Perimeter Computation
# ==================================================================================
@typechecked
def compute_attraction_distances(G: nx.Graph, A: Optional[Set[Any]] = None, method: str = "sum", debug: bool = False) -> Dict[Any, float]:
    """
    Precompute the distances from each node to attraction nodes using specified method.

    Parameters:
        G (nx.Graph): The graph
        A (Set[Any]): Set of attraction nodes
        method (str): Method to compute distances - "min" for minimum distance to any attraction node
                     or "sum" for sum of distances to all attraction nodes
        debug (bool): Whether to print debug information

    Returns:
        Dict[Any, float]: Dictionary mapping each node to its attraction distance
    """
    if not A:
        warning("No attraction nodes provided. Returning empty dictionary.")
        return {}

    attraction_distances_dict = {}

    if method == "min":
        for attraction_node in A:
            length_dict = nx.single_source_shortest_path_length(G, attraction_node)
            for node, dist in length_dict.items():
                if node not in attraction_distances_dict or dist < attraction_distances_dict[node]:
                    attraction_distances_dict[node] = dist
    elif method == "sum":
        attraction_distances_dict = {node: 0 for node in G.nodes()}
        for attraction_node in A:
            length_dict = nx.single_source_shortest_path_length(G, attraction_node)
            for node, dist in length_dict.items():
                attraction_distances_dict[node] += dist
    else:
        error(f"Unknown method: {method}. Use 'min' or 'sum'.")
        raise ValueError(f"Unknown method: {method}. Use 'min' or 'sum'.")
    success(f"Computed attraction distances using {method} method.", debug)
    return attraction_distances_dict


@typechecked
def _lagacy1_patch_boundary_for_connectivity(G: nx.Graph, P: Set[Any], H: Set[Any], penalty: Optional[int] = None, debug: bool = False) -> Set[Any]:
    """
    (Legacy) Ensure that the boundary set P is connected by patching in additional nodes from H.

    The function checks the induced subgraph of P for connectivity. If it is disconnected,
    it searches for a shortest path between components in the subgraph of H, using an edge weight
    function that penalizes stepping outside the current boundary P. Nodes along the shortest path
    are then added to P to connect the disconnected components.

    Parameters:
        G (nx.Graph): The full graph.
        P (Set[Any]): The initial set of boundary nodes.
        H (Set[Any]): The superset of nodes representing the current hull.
        penalty (Optional[int]): The extra cost for stepping outside P. Defaults to the number of nodes in G.
        debug (bool): If True, debug messages are printed.

    Returns:
        Set[Any]: The updated set of boundary nodes, patched to form a connected subgraph.
    """
    warning("You are using the legacy version of patch_boundary_for_connectivity.", debug)
    if penalty is None:
        penalty = G.number_of_nodes()  # Default penalty

    P_new = set(P)

    def edge_weight(a: Any, b: Any, d: Any) -> int:
        # Lower cost if both nodes are already in P_new.
        return 1 if (a in P_new and b in P_new) else 1 + penalty

    while True:
        subP = G.subgraph(P_new)
        components = list(nx.connected_components(subP))
        if len(components) <= 1:
            break  # Already connected

        best_path = None
        best_cost = float("inf")
        found_patch = False

        # Iterate over pairs of disconnected components.
        for i in range(len(components)):
            if found_patch:
                break
            for j in range(i + 1, len(components)):
                if found_patch:
                    break
                comp1 = components[i]
                comp2 = components[j]
                for u in comp1:
                    if found_patch:
                        break
                    for v in comp2:
                        try:
                            cost = nx.dijkstra_path_length(G.subgraph(H), u, v, weight=edge_weight)
                            if cost < best_cost:
                                best_cost = cost
                                best_path = nx.dijkstra_path(G.subgraph(H), u, v, weight=edge_weight)
                                found_patch = True
                                break
                        except nx.NetworkXNoPath:
                            continue
                    if found_patch:
                        break
                if found_patch:
                    break

        if not found_patch:
            info("Unable to patch any further disconnected components in H.", debug)
            break

        # Add nodes along the best path to P_new.
        for node in best_path:
            if node not in P_new:
                P_new.add(node)
                info(f"Added node {node} to boundary from path connecting components.", debug)
        continue

    return P_new


@typechecked
def _legacy2_patch_boundary_for_connectivity(G: nx.Graph, P: Set[Any], H: Set[Any], penalty: Optional[int] = None, debug: bool = False) -> Set[Any]:
    """
    Ensure that the boundary set P is connected by patching in additional nodes from H.

    The function checks the induced subgraph of P for connectivity.
    If it is disconnected, it searches for a shortest path between components in the subgraph of H,
    using an edge weight function that penalizes stepping outside the current boundary P. Once a path is
    found connecting any two disconnected components, the nodes along that path are added to P.
    This process repeats until the boundary becomes connected.

    Parameters:
        G (nx.Graph): The full undirected graph.
        P (Set[Any]): The initial set of boundary nodes.
        H (Set[Any]): The superset of nodes representing the current hull.
        penalty (Optional[int]): The extra cost for stepping outside P. Defaults to the number of nodes in G.
        debug (bool): If True, debug messages are printed.

    Returns:
        Set[Any]: The updated set of boundary nodes, patched to form a connected subgraph.
    """
    if penalty is None:
        penalty = G.number_of_nodes()  # Default penalty

    P_new = set(P)

    # DIFFERENT: Define edge weight function once, outside the loop
    def edge_weight(a: Any, b: Any, d: Any) -> int:
        # Lower cost if both nodes are already in P_new
        return 1 if (a in P_new and b in P_new) else 1 + penalty

    while True:
        subP = G.subgraph(P_new)
        components = list(nx.connected_components(subP))
        if nx.is_connected(subP) and all(subP.degree(n) == 2 for n in subP.nodes()):
            info("Boundary is already connected.", debug)
            break  # The boundary forms a cycle.

        found_path = None
        found_patch = False

        for i in range(len(components)):
            if found_patch:
                break
            for j in range(i + 1, len(components)):
                if found_patch:
                    break
                comp1 = components[i]
                comp2 = components[j]

                H_subgraph = G.subgraph(H)

                for u in comp1:
                    if found_patch:
                        break
                    for v in comp2:
                        try:
                            # DIFFERENT: More efficient path computation by checking length first
                            cost = nx.dijkstra_path_length(H_subgraph, u, v, weight=edge_weight)
                            found_path = nx.dijkstra_path(H_subgraph, u, v, weight=edge_weight)
                            found_patch = True
                            break
                        except nx.NetworkXNoPath:
                            continue
                    if found_patch:
                        break
                if found_patch:
                    break

        if not found_patch:
            info("Unable to patch any further disconnected components in H.", debug)
            break

        # Add nodes along the best path to P_new.
        for node in found_path:
            if node not in P_new:
                P_new.add(node)
                info(f"Added node {node} to boundary from path connecting components.", debug)

    return P_new


@typechecked
def patch_boundary_for_connectivity(G: nx.Graph, P: Set[Any], H: Set[Any], attraction_distances_dict: Optional[Dict[Any, float]] = None, penalty: Optional[int] = None, debug: bool = False) -> Set[Any]:
    """
    Ensure that the boundary set P is connected by patching in additional nodes.

    This is a two-step process:
    1. First try to patch using only nodes from H
    2. If P remains disconnected, use nodes from outside H with a penalty

    Parameters:
        G (nx.Graph): The full undirected graph.
        P (Set[Any]): The initial set of boundary nodes.
        H (Set[Any]): The superset of nodes representing the current hull.
        attraction_distances_dict (Optional[Dict[Any, float]]): Precomputed distances to attraction nodes.
            Can be created using compute_attraction_distances().
        penalty (Optional[int]): The extra cost for stepping outside H. Defaults to the number of nodes in G.
        debug (bool): If True, debug messages are printed.

    Returns:
        Set[Any]: The updated set of boundary nodes, patched to form a connected subgraph.
    """
    if penalty is None:
        penalty = G.number_of_nodes()  # Default penalty

    P_new = set(P)

    # Store attraction distances in graph for edge_weight functions to access
    G.A_distances = attraction_distances_dict or {}

    # Step 1: Try to patch using only nodes from H
    P_new = patch_with_hull(G, P_new, H, debug)
    H.update(P_new)

    # Step 2: If P is still disconnected, patch using nodes from outside H with penalty
    if not nx.is_connected(G.subgraph(P_new)):
        info("Could not fully connect boundary using only nodes from H. Trying external nodes.", debug)
        P_new = patch_with_external(G, P_new, H, penalty, debug)
        H.update(P_new)

    # Here P should already be connected, if not, raise a warning
    if not nx.is_connected(G.subgraph(P_new)):
        warning("Boundary is still disconnected after patching with external nodes.")

    # P_new = ensure_boundary_min_degree(G, P_new, H, debug)
    # H.update(P_new)

    return P_new


@typechecked
def patch_with_hull(G: nx.Graph, P: Set[Any], H: Set[Any], debug: bool) -> Set[Any]:
    """
    Patch the boundary set P using only nodes from H.

    Parameters:
        G (nx.Graph): The full undirected graph.
        P (Set[Any]): The initial set of boundary nodes.
        H (Set[Any]): The superset of nodes representing the current hull.
        debug (bool): If True, debug messages are printed.

    Returns:
        Set[Any]: The updated set of boundary nodes, patched with nodes from H.
    """
    P_new = set(P)
    H_subgraph = G.subgraph(H)

    while True:
        # Check connectivity of current boundary
        subP = G.subgraph(P_new)

        # If P is connected, we're done with this step
        if nx.is_connected(subP):
            info("Boundary is now connected using nodes from H.", debug)
            break

        # Get the connected components
        components = list(nx.connected_components(subP))

        # Define edge weight function
        def edge_weight(a: Any, b: Any, d: Any) -> int:
            if a in P_new and b in P_new:
                return 0  # Zero cost for traveling within P
            elif a in H and b in H:
                return 1  # Cost of 1 for nodes in H but not in P
            else:
                return float("inf")  # Infinite cost for nodes outside H (effectively preventing their use)

        # Find the best path between component pairs
        best_path = None
        best_path_length = float("inf")
        best_pair = None

        # Iterate through component pairs, prioritizing smaller components
        for i in range(len(components) - 1):
            comp1 = components[i]  # Smaller component

            for j in range(i + 1, len(components)):
                comp2 = components[j]

                # Select representative nodes from each component
                u = next(iter(comp1))
                v = next(iter(comp2))

                try:
                    # First check if a path exists and get its length
                    path_length = nx.dijkstra_path_length(H_subgraph, u, v, weight=edge_weight)

                    # If this is better than our current best path, update it
                    if path_length < best_path_length:
                        best_path = nx.dijkstra_path(H_subgraph, u, v, weight=edge_weight)
                        best_path_length = path_length
                        best_pair = (u, v)

                except (nx.NetworkXNoPath, nx.NetworkXError):
                    continue

        # If no path was found between any components, break
        if best_path is None:
            info("Unable to patch any further disconnected components using nodes from H.", debug)
            break

        # Add nodes along the best path to P_new
        for node in best_path:
            if node not in P_new:
                P_new.add(node)
                info(f"Added node {node} from H to boundary when connecting node {best_pair[0]} to {best_pair[1]}.", debug)

    return P_new


@typechecked
def patch_with_external(G: nx.Graph, P: Set[Any], H: Set[Any], penalty: int, debug: bool) -> Set[Any]:
    """
    Patch the boundary set P using nodes from both H and outside H, with a penalty for using external nodes.

    Parameters:
        G (nx.Graph): The full undirected graph.
        P (Set[Any]): The initial set of boundary nodes.
        H (Set[Any]): The superset of nodes representing the current hull.
        penalty (int): The extra cost for stepping outside H.
        debug (bool): If True, debug messages are printed.

    Returns:
        Set[Any]: The updated set of boundary nodes, patched with nodes from both H and outside.
    """
    P_new = set(P)
    attraction_distances = getattr(G, "A_distances", {})
    print(attraction_distances)
    print("!!!!!!!!!!!!!")

    while True:
        # Check connectivity of current boundary
        subP = G.subgraph(P_new)

        if nx.is_connected(subP):
            info("Boundary is already connected without external nodes.", debug)
            break

        # Get the connected components
        components = list(nx.connected_components(subP))

        # Define edge weight function for external patching
        def edge_weight(a: Any, b: Any, d: Any) -> int:
            if a in P_new and b in P_new:
                return 0  # Zero cost for traveling within P
            elif a in H and b in H:
                return 1  # Cost of 1 for nodes in H but not in P
            else:
                if attraction_distances:
                    a_value = attraction_distances.get(a, 0)
                    b_value = attraction_distances.get(b, 0)
                    adjustment = (a_value + b_value) / 2
                    print(f"Attraction distances: {a_value}, {b_value}, adjustment: {adjustment}")
                    return penalty + 1 + adjustment * 0.5
                else:
                    return penalty + 1

        # Find the first path between any two components
        found_path = None

        for i, comp1 in enumerate(components):
            if found_path:
                break

            for j in range(i + 1, len(components)):
                comp2 = components[j]

                # Check one pair of nodes from each component
                u = next(iter(comp1))
                v = next(iter(comp2))

                try:
                    found_path = nx.dijkstra_path(G, u, v, weight=edge_weight)
                    break
                except nx.NetworkXNoPath:
                    continue

        # If no path was found between any components, break
        if found_path is None:
            info("Unable to patch any further disconnected components even with external nodes.", debug)
            break

        # Add nodes along the found path to P_new
        for node in found_path:
            if node not in P_new:
                P_new.add(node)
                # Check if the node is in H or outside
                if node not in H:
                    info(f"Added external node {node} to boundary.", debug)
                else:
                    warning(f"Added node {node} to boundary from H inside patch with external. Potentially algorithm flaw.")
    return P_new


@typechecked
def _lagacy1_compute_graph_convex_hull(G: nx.Graph, S: Set[Any], visualize_steps: bool = False, debug: bool = False) -> Set[Any]:
    """
    (Legacy) Compute the convex hull of a set S on graph G based on graph distances.

    Starting with the initial set S, the convex hull H is iteratively expanded.
    For every pair of boundary nodes, if there is a shorter path outside the current boundary,
    the nodes along that path are added to H. Optionally, the steps of the computation can be visualized.

    Parameters:
        G (nx.Graph): The input graph.
        S (Set[Any]): The initial set of nodes (seed points).
        visualize_steps (bool): If True, intermediate steps of the hull computation are visualized.
        debug (bool): If True, prints debug messages.

    Returns:
        Set[Any]: The computed convex hull as a set of nodes.
    """
    warning("You are using the legacy version of compute_graph_convex_hull.", debug)
    H: Set[Any] = set(S)

    if visualize_steps:
        try:
            initial_P = {u for u in H if any(v not in H for v in G.neighbors(u))}
            gv = gfvis.GraphVisualizer(G=G, mode="static", extra_info={"Step": "Initial convex hull"}, node_size=300)
            gv.color_nodes(list(S), color="green", mode="solid", name="Seed")
            gv.color_nodes(list(H), color="green", mode="transparent", name="Hull")
            gv.color_nodes(list(initial_P), color="orange", mode="transparent", name="Boundary")
            gv.visualize()
        except Exception as e:
            warning(f"Error visualizing initial state: {e}")

    while True:
        P = {u for u in H if any(v not in H for v in G.neighbors(u))}
        P_subgraph = G.subgraph(P)
        changed = False
        P_list = list(P)

        for i in range(len(P_list)):
            for j in range(i + 1, len(P_list)):
                u, v = P_list[i], P_list[j]
                try:
                    boundary_distance = nx.shortest_path_length(P_subgraph, u, v)
                except nx.NetworkXNoPath:
                    inside_path_penalty = G.number_of_nodes()

                    def edge_weight(a: Any, b: Any, d: Any) -> int:
                        return 1 if (a in P and b in P) else 1 + inside_path_penalty

                    try:
                        dpath = nx.dijkstra_path(G.subgraph(H), u, v, weight=edge_weight)
                        boundary_distance = len(dpath) - 1
                        for node in dpath:
                            if node not in P:
                                P.add(node)
                                info(f"Added node {node} to boundary from path connecting {u} and {v}", debug)
                    except nx.NetworkXNoPath:
                        boundary_distance = float("inf")

                # Check for a shortcut outside H.
                outside_nodes = (set(G.nodes()) - H) | P
                outside_nodes.update({u, v})
                outside_graph = G.subgraph(outside_nodes)

                try:
                    outside_distance = nx.shortest_path_length(outside_graph, u, v)
                    if outside_distance < boundary_distance:
                        info(f"Shortcut found between {u} and {v} outside the current hull.", debug)
                        outside_path = nx.shortest_path(outside_graph, u, v)
                        H.update(outside_path)

                        if visualize_steps:
                            try:
                                P = {u for u in H if any(v not in H for v in G.neighbors(u))}
                                P_subgraph = G.subgraph(P)
                                P_list = list(P)
                                gv = gfvis.GraphVisualizer(G=G, mode="static", extra_info={"Step": f"Updated hull with shortcut between {u} and {v}"}, node_size=300)
                                gv.color_nodes(list(S), color="green", mode="solid", name="Seed")
                                gv.color_nodes(list(H), color="green", mode="transparent", name="Hull")
                                gv.color_nodes(list(P), color="orange", mode="transparent", name="Boundary")
                                gv.visualize()
                            except Exception as e:
                                warning(f"Error visualizing updated state: {e}")

                        changed = True
                        break
                except nx.NetworkXNoPath:
                    pass
            if changed:
                break
        if not changed:
            break
    success("Convex hull computation completed.", debug)
    return H


@typechecked
def _lagacy2_compute_graph_convex_hull(G: nx.Graph, S: Set[Any], visualize_steps: bool = False, debug: bool = False) -> Set[Any]:
    """
    Compute the convex hull of a set S on graph G based on graph distances.

    Starting with the initial set S, the convex hull H is iteratively expanded.
    For every pair of boundary nodes, if there is a shorter path outside the current boundary,
    the nodes along that path are added to H. Optionally, the steps of the computation can be visualized.

    Parameters:
        G (nx.Graph): The input graph.
        S (Set[Any]): The initial set of nodes (seed points).
        visualize_steps (bool): If True, intermediate steps of the hull computation are visualized.
        debug (bool): If True, prints debug messages.

    Returns:
        Set[Any]: The computed convex hull as a set of nodes.
    """
    H: Set[Any] = set(S)

    while True:
        info(f"Current hull at start of it: {H}", debug)
        P = {u for u in H if any(v not in H for v in G.neighbors(u))}
        P = patch_boundary_for_connectivity(G, P, H, debug=debug)
        info(f"Boundary nodes: {P}", debug)
        # Create a subgraph of the boundary nodes
        changed = False
        P_list = list(P)
        if visualize_steps:
            try:
                viz_G = nx.MultiDiGraph(G) if G is not None else None
                gv = gfvis.GraphVisualizer(G=viz_G, mode="static", node_size=300)
                gv.color_nodes(list(S), color="green", mode="solid", name="Seed")
                gv.color_nodes(list(H), color="green", mode="transparent", name="Hull")
                gv.color_nodes(list(P), color="orange", mode="transparent", name="Boundary")
                gv.visualize()
            except Exception as e:
                warning(f"Error visualizing initial state: {e}")

        # Create pairs of all boundary nodes
        boundary_pairs = []
        for i in range(len(P_list)):
            for j in range(i + 1, len(P_list)):
                boundary_pairs.append((P_list[i], P_list[j]))

        for u, v in boundary_pairs:
            try:
                boundary_distance = nx.shortest_path_length(G.subgraph(P), u, v)
            except nx.NetworkXNoPath:
                inside_path_penalty = G.number_of_nodes()

                def edge_weight(a: Any, b: Any, d: Any) -> int:
                    return 1 if (a in P and b in P) else 1 + inside_path_penalty

                try:
                    weighted_shortest_path = nx.dijkstra_path(G.subgraph(H), u, v, weight=edge_weight)
                    boundary_distance = len(weighted_shortest_path) - 1
                    for node in weighted_shortest_path:
                        if node not in P:
                            P.add(node)
                            H.add(node)
                            info(f"Added node {node} to boundary from path connecting {u} and {v}", debug)
                except nx.NetworkXNoPath:
                    boundary_distance = float("inf")

            outside_nodes = (set(G.nodes()) - H) | P
            outside_graph = G.subgraph(outside_nodes)

            try:
                outside_distance = nx.shortest_path_length(outside_graph, u, v)
                if outside_distance < boundary_distance:
                    info(f"Outside distance: {outside_distance}", debug)
                    info(f"Boundary distance: {boundary_distance}", debug)
                    info(f"Shortcut found between {u} and {v} outside the current hull.", debug)
                    outside_path = nx.shortest_path(outside_graph, u, v)
                    info(f"Adding nodes {outside_path} to the hull.", debug)
                    H.update(outside_path)
                    info(f"The new hull is {H}", debug)

                    changed = True
                    break
            except nx.NetworkXNoPath:
                warning(f"No path found between {u} and {v} outside the current hull, convex hull might be the entire graph.")
                break
        if not changed:
            break
    success("Convex hull computation completed.", debug)
    return H


@typechecked
def compute_graph_convex_hull(G: nx.Graph, S: Set[Any], visualize_steps: bool = False, debug: bool = False, attraction_distances_dict: Optional[Dict[Any, float]] = None) -> Set[Any]:
    """
    Compute the convex hull of a set S on graph G based on graph distances.

    Starting with the initial set S, the convex hull H is iteratively expanded.
    For every pair of boundary nodes, if there is a shorter path outside the current boundary,
    the nodes along that path are added to H. Optionally, the steps of the computation can be visualized.

    Parameters:
        G (nx.Graph): The input graph.
        S (Set[Any]): The initial set of nodes (seed points).
        visualize_steps (bool): If True, intermediate steps of the hull computation are visualized.
        debug (bool): If True, prints debug messages.
        attraction_distances_dict (Optional[Dict[Any, float]]): Pre-computed dictionary mapping nodes to attraction values.

    Returns:
        Set[Any]: The computed convex hull as a set of nodes.
    """
    H: Set[Any] = set(S)
    iteration_counter = 0
    while True:
        info(f"=================================================== Iteration {iteration_counter} ===================================================", debug)
        info(f"Current hull at start of iteration {iteration_counter}: {sorted(H)}", debug)
        P = {u for u in H if any(v not in H for v in G.neighbors(u))}
        P = patch_boundary_for_connectivity(G, P, H, attraction_distances_dict=attraction_distances_dict, debug=debug)
        info(f"Updated hull after patching: {sorted(H)}", debug)
        info(f"Boundary nodes at iteration {iteration_counter}: {sorted(P)}", debug)
        if visualize_steps:
            try:
                viz_G = nx.MultiDiGraph(G) if G is not None else None
                gv = gfvis.GraphVisualizer(G=viz_G, mode="static", node_size=300, extra_info={"Step": f"Iteration {iteration_counter}"})
                gv.color_nodes(list(S), color="green", mode="solid", name="Seed")
                gv.color_nodes(list(H), color="green", mode="transparent", name="Hull")
                gv.color_nodes(list(P), color="orange", mode="solid", name="Boundary")
                gv.visualize()
            except Exception as e:
                warning(f"Error visualizing initial state: {e}")
        P_list = list(P)
        boundary_pairs = []
        for i in range(len(P_list)):
            for j in range(i + 1, len(P_list)):
                boundary_pairs.append((P_list[i], P_list[j]))

        changed = False
        for u, v in boundary_pairs:
            # Calculate the distance through the boundary
            try:
                # Create a subgraph of just the boundary nodes to find the path length within the boundary
                boundary_graph = G.subgraph(P)
                if nx.is_connected(boundary_graph):
                    boundary_distance = nx.shortest_path_length(boundary_graph, u, v)
                else:
                    warning(f"Boundary graph is not connected, this should not happen after patching.")
                    boundary_distance = float("inf")
            except nx.NetworkXNoPath:
                boundary_distance = float("inf")
            # Check for shortcuts outside the hull
            outside_nodes = (set(G.nodes()) - H) | P
            outside_graph = G.subgraph(outside_nodes)

            try:
                # Define edge weighting function to prefer paths near attraction points
                def outside_edge_weight(a: Any, b: Any, d: Any) -> int:
                    base_cost = 1
                    p_reward = 0
                    attraction_reward = 0
                    if a in P and b in P:
                        p_reward = 1 / G.number_of_nodes()
                    if attraction_distances_dict:
                        a_value = attraction_distances_dict.get(a, 0)
                        b_value = attraction_distances_dict.get(b, 0)
                        adjustment = (a_value + b_value) / 2
                        attraction_reward = adjustment / G.number_of_nodes() / G.number_of_nodes()
                    return base_cost - p_reward + attraction_reward

                outside_path = nx.shortest_path(outside_graph, u, v, weight=outside_edge_weight)
                outside_distance = len(outside_path) - 1

                if outside_distance < boundary_distance:
                    info(f"Outside distance: {outside_distance}", debug)
                    info(f"Boundary distance: {boundary_distance}", debug)
                    info(f"Shortcut found between {u} and {v} outside the current hull.", debug)
                    info(f"Adding nodes {outside_path} to the hull.", debug)

                    H.update(outside_path)
                    info(f"The new hull is {sorted(H)}", debug)

                    changed = True
                    break
            except nx.NetworkXNoPath:
                warning(f"No path found between {u} and {v} outside the current hull, convex hull might be the entire graph.")
                continue
        iteration_counter += 1
        if not changed:
            break

    success("Convex hull computation completed.", debug)
    return H


@typechecked
def compute_convex_hull_and_perimeter(G: nx.MultiDiGraph, S: Union[Set[Any], List[Any]], visualize_steps: bool = False, attraction_distances_dict: Optional[Dict[Any, float]] = None, debug: Optional[bool] = False) -> Tuple[Set[Any], List[Any]]:
    """
    Compute the convex hull of a set S on graph G and identify its boundary (perimeter) nodes.

    The convex hull is computed using graph distances, and afterwards the boundary nodes are
    determined as those nodes within the hull that have at least one neighbor outside the hull.
    The boundary set is then patched to ensure connectivity.

    Parameters:
        G (nx.MultiDiGraph): The input graph, which can be a MultiDiGraph.
        S (Union[Set[Any], List[Any]]): The initial set of nodes forming the seed points.
        visualize_steps (bool): If True, intermediate steps are visualized.
        attraction_distances_dict (Optional[Dict[Any, float]]): Pre-computed dictionary mapping nodes to
            attraction values. Influences how paths are selected during hull computation and boundary patching.
        debug (Optional[bool]): If True, debug messages are printed.

    Returns:
        Tuple[Set[Any], List[Any]]:
            - H: The set of nodes forming the convex hull.
            - P: The list of boundary node IDs that define the perimeter of the hull.
    """
    # Convert MultiDiGraph to undirected graph to ensure proper hull computation
    undirected_G = nx.Graph()

    # Add all nodes from the original graph
    undirected_G.add_nodes_from(G.nodes(data=True))

    # Add all edges (converting directed multi-edges to single undirected edges)
    for u, v, _ in G.edges:
        if not undirected_G.has_edge(u, v):
            # Add the edge if it doesn't exist yet
            undirected_G.add_edge(u, v)

    # Compute the convex hull using the undirected graph, with attraction distances if provided
    H = compute_graph_convex_hull(undirected_G, set(S), visualize_steps=visualize_steps, debug=debug, attraction_distances_dict=attraction_distances_dict)

    # Find boundary nodes (nodes with at least one neighbor outside the hull)
    P = [node for node in H if any(neighbor not in H for neighbor in undirected_G.neighbors(node))]

    # Ensure boundary connectivity, using attraction distances if provided
    P = list(patch_boundary_for_connectivity(undirected_G, set(P), H, attraction_distances_dict=attraction_distances_dict, debug=debug))

    return H, P


@typechecked
def compute_shortest_path_step(graph: Union[nx.Graph, nx.MultiDiGraph], source_node: Any, target: Union[Any, Iterable[Any]], step: Optional[int] = 1) -> Optional[Union[Any, List[Any]]]:
    """
    Compute the shortest path from a source node to one or more target nodes and return a specific step along that path.

    The function calculates the shortest path(s) using Dijkstra's algorithm. If a valid path is found,
    it returns the node at the specified step index along the shortest path. If `step` is None, the entire
    shortest path is returned. If no path is found or the source node does not exist, a warning is logged
    and None is returned.

    Parameters:
        graph (Union[nx.Graph, nx.MultiDiGraph]): The input graph.
        source_node (Any): The starting node.
        target (Union[Any, Iterable[Any]]): The target node or an iterable of target nodes.
        step (Optional[int]): The step index along the shortest path to return.
                              If None, the entire path is returned. Defaults to 1.

    Returns:
        Optional[Union[Any, List[Any]]]:
            - The node at the specified step along the shortest path if step is provided.
            - The full path as a list if step is None.
            - None if no valid path exists.
    """
    # Ensure target is iterable.
    if not isinstance(target, (list, tuple, set)):
        target_nodes = [target]
    else:
        target_nodes = list(target)

    best_path: Optional[List[Any]] = None
    best_length: float = float("inf")

    if source_node not in graph.nodes:
        error(f"Source node {source_node} not found in the graph.")
        return None

    for t in target_nodes:
        try:
            length, path = nx.single_source_dijkstra(graph, source=source_node, target=t, weight="length")
            if length < best_length:
                best_length = length
                best_path = path
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            if isinstance(e, nx.NodeNotFound):
                warning(f"Node {t} not found in the graph.")
            continue

    if best_path is None:
        warning(f"No path found from {source_node} to any of the target nodes: {target_nodes}")
        return None

    if step is None:
        return best_path

    index = step if step < len(best_path) else len(best_path) - 1
    return best_path[index]


# Example usage:
if __name__ == "__main__":
    # G = generate_random_delaunay_graph(n_points=400, side=10, seed=42, debug=True)
    # G = generate_simple_grid(rows=20, cols=20, debug=True)
    # G = generate_lattice_grid(rows=20, cols=20, debug=True)

    root_folder = add_root_folder_to_sys_path()

    config_file_path = os.path.join(root_folder, "data/config/config_04151259_ga/F5A10D10_0ac5d2/F5A10D10_0ac5d2_r38.yml")
    graph_file_path = os.path.join(root_folder, "data/graphs/graph_200_200_a.pkl")  # Example path to your graph file

    G = export_graph(graph_file_path)
    visualizer = gfvis.GraphVisualizer(file_path=config_file_path, mode="interactive", simple_layout=False, debug=True, node_size=100)

    # attackers_positions = [23, 171, 178, 176, 181]
    # visualizer.color_nodes(attackers_positions, color="red", mode="solid", name="Attacker", size_multiplier=1.3)

    # defenders_positions = [167, 166, 70, 148, 195]
    # visualizer.color_nodes(defenders_positions, color="blue", mode="solid", name="Defender", size_multiplier=1.3)

    flag_positions = [30, 31, 13, 5, 182]
    # # visualizer.color_nodes(flag_positions, color="green", mode="solid", name="Flag")
    target_nodes = compute_x_neighbors(G, set(flag_positions), 2)
    # attraction_distances_dict = compute_attraction_distances(G, set(flag_positions), debug=False)

    H, P = compute_convex_hull_and_perimeter(G, target_nodes, visualize_steps=False, attraction_distances_dict=None)
    print(target_nodes)
    print(f"Convex Hull: {H}")
    print(f"Perimeter: {P}")

    visualizer.color_nodes(list(H), color="green", mode="transparent", name="Convex Hull")
    visualizer.color_nodes(P, color="orange", mode="solid", name="Perimeter")
    # visualizer.color_nodes(flag_positions, color="green", mode="solid", name="Flag")
    # visualizer.color_nodes(defenders_positions, color="blue", mode="solid", name="Defender")

    visualizer.visualize()

    # root_folder = add_root_folder_to_sys_path()
    # graph_file_path = os.path.join(root_folder, "data", "graphs", "graph_200_200.pkl")
    # G = export_graph(graph_file_path)

    # flag_nodes = [251, 67]  # Example positions in the grid
    # defender_nodes = [152, 153]  # Example positions in the grid
    # attacker_nodes = [3, 4]  # Example positions in the grid

    # target_nodes = compute_x_neighbors(G, set(flag_nodes), 2)

    # attraction_distances_dict = compute_attraction_distances(G, set(defender_nodes), debug=True)
    # # attraction_distances_dict = None

    # H, P = compute_convex_hull_and_perimeter(G, target_nodes, visualize_steps=False, attraction_distances_dict=attraction_distances_dict)
    # print(f"Convex Hull: {H}")
    # print(f"Perimeter: {P}")

    # # Create a GraphVisualizer instance in interactive mode
    # gv = gfvis.GraphVisualizer(G=G, mode="interactive", extra_info=None, node_size=100, node_color="lightgray", transparent_alpha=0.3)

    # # Color the flag nodes with solid green
    # gv.color_nodes(flag_nodes, color="green", name="Flag Nodes")
    # hull_without_flags = [node for node in H if node not in flag_nodes]
    # gv.color_nodes(hull_without_flags, color="green", mode="transparent", name="Convex Hull")

    # gv.color_nodes(attacker_nodes, color="red", name="Attacker")

    # perimeter_without_flags = [node for node in P if node not in flag_nodes]
    # # perimeter_without_flags = [node for node in perimeter_without_flags if node not in defender_capture_radius]
    # gv.color_nodes(perimeter_without_flags, color="yellow", mode="solid", name="Perimeter")
    # gv.color_nodes(defender_nodes, color="blue", name="Defender")
    # # gv.color_nodes(defender_capture_radius, color="lightblue", mode="solid", name="Defender Capture Radius")

    # # # Visualize the graph using the unified visualize() method.
    # gv.visualize("test.png")  # You can also pass a save path if desired, e.g., gv.visualize("output.png")
