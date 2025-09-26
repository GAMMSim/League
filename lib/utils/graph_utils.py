import random
import numpy as np
import networkx as nx
from copy import deepcopy
from typeguard import typechecked
from scipy.spatial import Delaunay
from shapely.geometry import LineString
from typing import Any, List, Set, Union, Optional


try:
    from lib.core.console import *
except ImportError:
    try:
        from ..core.console import *
    except ImportError:
        # Fallback logging functions if core module not available
        def debug(msg: str):
            print(f"DEBUG: {msg}")

        def info(msg: str):
            print(f"INFO: {msg}")

        def warning(msg: str):
            print(f"WARNING: {msg}")

        def error(msg: str):
            print(f"ERROR: {msg}")

        def success(msg: str):
            print(f"SUCCESS: {msg}")


@typechecked
def cast_to_multidigraph(G: nx.DiGraph) -> nx.MultiDiGraph:
    """
    Convert a DiGraph to a MultiDiGraph ensuring bidirectionality and unique edge IDs.

    For each directed edge (u, v) in G:
      - Assign a fresh integer 'id' to (u, v).
      - If the reverse edge (v, u) does not exist in G, add it with its own new 'id'.
        If the edge data contains a 'linestring', reverse its coordinates for the reverse edge.

    Args:
        G (nx.DiGraph): Input directed graph. May have node and edge attributes.

    Returns:
        nx.MultiDiGraph: A MultiDiGraph with bidirectional edges and unique 'id' for every edge.
    """
    debug(f"Converting DiGraph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges to MultiDiGraph")

    MD = nx.MultiDiGraph()
    MD.add_nodes_from(G.nodes(data=True))

    next_edge_id = 0
    added_reverse_count = 0

    for u, v, data in G.edges(data=True):
        original_data = deepcopy(data) if data is not None else {}
        original_data["id"] = next_edge_id
        MD.add_edge(u, v, **original_data)
        next_edge_id += 1

        if not G.has_edge(v, u):
            rev_data = deepcopy(data) if data is not None else {}
            if "linestring" in rev_data and isinstance(rev_data["linestring"], LineString):
                rev_data["linestring"] = LineString(rev_data["linestring"].coords[::-1])
            rev_data["id"] = next_edge_id
            MD.add_edge(v, u, **rev_data)
            next_edge_id += 1
            added_reverse_count += 1
        else:
            existing_data = deepcopy(G[v][u]) if G[v][u] is not None else {}
            existing_data["id"] = next_edge_id
            MD.add_edge(v, u, **existing_data)
            next_edge_id += 1

    info(f"Added {added_reverse_count} reverse edge(s) to ensure bidirectionality")
    success(f"Converted DiGraph to MultiDiGraph with {MD.number_of_nodes()} nodes and {MD.number_of_edges()} edges")
    return MD


def convert_gml_to_multidigraph(G: nx.Graph, scale_factor: float = 1, offset_x: float = 0, offset_y: float = 0) -> nx.MultiDiGraph:
    """
    Convert a normalized 3D graph to a MultiDiGraph with 2D coordinates and LineString edges.

    Args:
        G: Input graph with normalized coordinates
        scale_factor: Scaling factor for coordinates
        offset_x: X coordinate offset
        offset_y: Y coordinate offset

    Returns:
        nx.MultiDiGraph: Spatial MultiDiGraph with LineString edges
    """
    debug(f"Converting GML graph with scale_factor={scale_factor}, offset_x={offset_x}, offset_y={offset_y}")

    try:
        # Create new DiGraph
        spatial_graph = nx.DiGraph()

        # Transform node coordinates, casting the ID to int
        for node_str, data in G.nodes(data=True):
            node_int = int(node_str)  # convert "5764607…" → 5764607 (or however your IDs parse)
            new_x = data["x"] * scale_factor + offset_x
            new_y = data["y"] * scale_factor + offset_y
            spatial_graph.add_node(node_int, x=new_x, y=new_y)

        debug(f"Transformed {spatial_graph.number_of_nodes()} nodes with new coordinates")

        # Add edges with LineString geometry and length
        edge_id = 0
        for u_str, v_str, data in G.edges(data=True):
            u = int(u_str)
            v = int(v_str)

            u_coords = spatial_graph.nodes[u]
            v_coords = spatial_graph.nodes[v]

            linestring = LineString([(u_coords["x"], u_coords["y"]), (v_coords["x"], v_coords["y"])])
            length = ((v_coords["x"] - u_coords["x"]) ** 2 + (v_coords["y"] - u_coords["y"]) ** 2) ** 0.5

            spatial_graph.add_edge(u, v, id=edge_id, linestring=linestring, length=length)
            edge_id += 1

        debug(f"Added {spatial_graph.number_of_edges()} edges with LineString geometry")

    except Exception as e:
        error(f"Error converting gml to spatial DiGraph: {e}")
        raise Exception(f"Error converting to gml spatial DiGraph: {e}")

    success(f"Converted to spatial DiGraph with {spatial_graph.number_of_nodes()} nodes and {spatial_graph.number_of_edges()} edges")

    # Convert to MultiDiGraph with bidirectional edges
    return cast_to_multidigraph(spatial_graph)


@typechecked
def generate_simple_grid(rows: int = 10, cols: int = 10) -> nx.MultiDiGraph:
    """
    Generate a grid graph with the specified number of rows and columns as a MultiDiGraph.

    The graph is first created with nodes identified by tuple coordinates.
    Then, the nodes are relabeled as integers (from 0 to n-1), and each node is
    assigned positional attributes: 'x' for the column index and 'y' for the row index.
    Finally, the graph is converted to a MultiDiGraph.

    Parameters:
        rows (int): The number of rows in the grid.
        cols (int): The number of columns in the grid.

    Returns:
        nx.MultiDiGraph: A grid graph with integer node labels, positional attributes,
                         and represented as a MultiDiGraph.
    """
    debug(f"Generating simple grid with {rows} rows and {cols} columns")

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
    success(f"Generated grid with {G_multi.number_of_nodes()} nodes and {G_multi.number_of_edges()} edges")
    return G_multi


@typechecked
def generate_lattice_grid(rows: int = 10, cols: int = 10) -> nx.MultiDiGraph:
    """
    Generate a lattice grid graph (including diagonals) with the specified
    number of rows and columns as a MultiDiGraph.

    Parameters:
        rows (int): The number of rows in the grid.
        cols (int): The number of columns in the grid.

    Returns:
        nx.MultiDiGraph: A lattice graph with integer node labels, positional
                         attributes, and represented as a MultiDiGraph.
    """
    debug(f"Generating lattice grid with {rows} rows and {cols} columns (including diagonals)")

    # Start with an undirected 2D grid
    G = nx.grid_2d_graph(rows, cols)

    # Add diagonal edges: for each cell, connect (r,c) to (r+1,c+1) and (r+1,c-1)
    diagonal_count = 0
    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows and c + 1 < cols:
                G.add_edge((r, c), (r + 1, c + 1))
                diagonal_count += 1
            if r + 1 < rows and c - 1 >= 0:
                G.add_edge((r, c), (r + 1, c - 1))
                diagonal_count += 1

    debug(f"Added {diagonal_count} diagonal edges")

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

    success(f"Generated lattice grid with {G_multi.number_of_nodes()} nodes and {G_multi.number_of_edges()} edges")
    return G_multi


@typechecked
def generate_triangular_lattice_graph(rows: int = 10, cols: int = 10) -> nx.MultiDiGraph:
    """
    Generate a triangular-lattice graph with the specified number of rows and columns
    as a MultiDiGraph.  Each node (r,c) is connected to its right neighbor, down neighbor,
    and down-left neighbor, producing a mesh of triangles.

    Parameters:
        rows (int): Number of rows.
        cols (int): Number of columns.

    Returns:
        nx.MultiDiGraph: Triangular-lattice as a directed multigraph with integer labels
                         and 'x','y' positional attributes.
    """
    debug(f"Generating triangular lattice graph with {rows} rows and {cols} columns")

    # Start from an empty undirected graph
    G = nx.Graph()

    # Add nodes
    for r in range(rows):
        for c in range(cols):
            G.add_node((r, c))

    # Add edges for triangular tiling
    edge_count = 0
    for r in range(rows):
        for c in range(cols):
            # right neighbor
            if c + 1 < cols:
                G.add_edge((r, c), (r, c + 1))
                edge_count += 1
            # down neighbor
            if r + 1 < rows:
                G.add_edge((r, c), (r + 1, c))
                edge_count += 1
            # down-left neighbor
            if r + 1 < rows and c - 1 >= 0:
                G.add_edge((r, c), (r + 1, c - 1))
                edge_count += 1

    debug(f"Added {edge_count} triangular lattice edges")

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

    success(f"Generated triangular lattice with {G_multi.number_of_nodes()} nodes and {G_multi.number_of_edges()} edges")
    return G_multi


@typechecked
def generate_random_delaunay_graph(n_points: int = 100, side: float = 1.0, seed: int = 0) -> nx.MultiDiGraph:
    """
    Generate n_points uniformly in the square [0, side] × [0, side],
    compute their Delaunay triangulation, and return a graph connecting every
    pair of points that share a triangle edge.

    Parameters:
        n_points (int): Number of random 2D points.
        side (float): Length of the square's side (origin is fixed at (0,0)).
        seed (int): RNG seed for reproducibility.

    Returns:
        nx.MultiDiGraph: nodes 0..n_points-1 with 'x','y' attrs and triangulation edges.
    """
    debug(f"Generating random Delaunay graph with {n_points} points, side={side}, seed={seed}")

    # 1) Sample points in [0, side]^2
    rng = np.random.default_rng(seed)
    points = rng.random((n_points, 2)) * side

    # 2) Delaunay triangulation
    tri = Delaunay(points)
    debug(f"Computed Delaunay triangulation with {len(tri.simplices)} triangles")

    # 3) Build undirected graph
    G = nx.Graph()
    for idx, (x, y) in enumerate(points):
        G.add_node(int(idx), x=float(x), y=float(y))

    # 4) Add triangle edges
    edges_added = 0
    for simplex in tri.simplices:
        i, j, k = [int(val) for val in simplex]  # Convert NumPy integers to Python integers
        if not G.has_edge(i, j):
            G.add_edge(i, j)
            edges_added += 1
        if not G.has_edge(j, k):
            G.add_edge(j, k)
            edges_added += 1
        if not G.has_edge(k, i):
            G.add_edge(k, i)
            edges_added += 1

    debug(f"Added {edges_added} unique edges from triangulation")

    # 5) Convert to MultiDiGraph
    G_multi = nx.MultiDiGraph(G)

    success(f"Generated Delaunay triangulation with {G_multi.number_of_nodes()} nodes, {G_multi.number_of_edges()} edges")
    return G_multi


@typechecked
def renumber_graph(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Renumber the nodes of a graph to have consecutive integer IDs starting from 0.

    This function creates a new graph with node IDs renumbered, while preserving the
    original node attributes and edge data. If an edge has an 'id' attribute, it is updated
    to a new sequential ID.

    Parameters:
        G (nx.MultiDiGraph): The input multigraph with arbitrary node IDs.

    Returns:
        nx.MultiDiGraph: A new multigraph with nodes renumbered from 0 to n-1.
    """
    debug(f"Renumbering graph with {G.number_of_nodes()} nodes")

    try:
        H = nx.MultiDiGraph()
        # Map old node IDs to new node IDs.
        mapping = {old_id: new_id for new_id, old_id in enumerate(G.nodes())}
        debug(f"Created node mapping for {len(mapping)} nodes")

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

        success(f"Graph renumbered with {len(H.nodes)} nodes and {H.number_of_edges()} edges")
        return H

    except Exception as e:
        error(f"Error in renumber_graph: {e}")
        raise Exception(f"Error in renumber_graph: {e}")


@typechecked
def reduce_graph_to_size(G: nx.MultiDiGraph, node_limit: int) -> nx.MultiDiGraph:
    """
    Reduce a MultiDiGraph to at most a given number of nodes while preserving connectivity.

    This function first checks if the graph already meets the node limit. If not, it identifies
    the largest weakly connected component of the MultiDiGraph. If that component is still too large,
    it uses a breadth-first search (BFS) starting from a random node within the component to extract
    a subgraph of the desired size. Finally, the subgraph is renumbered to have consecutive node IDs.

    Parameters:
        G (nx.MultiDiGraph): The original directed multigraph.
        node_limit (int): The maximum number of nodes desired in the reduced graph.

    Returns:
        nx.MultiDiGraph: A reduced and renumbered MultiDiGraph with at most node_limit nodes.

    Raises:
        Exception: Propagates any errors encountered during graph reduction.
    """
    debug(f"Reducing graph from {G.number_of_nodes()} nodes to at most {node_limit} nodes")

    try:
        # If the graph is already small enough, renumber and return it.
        if G.number_of_nodes() <= node_limit:
            info(f"Graph has {G.number_of_nodes()} nodes which is within the limit")
            return renumber_graph(G)

        # Identify the largest weakly connected component in the MultiDiGraph.
        largest_cc = max(nx.weakly_connected_components(G), key=len)
        info(f"Largest weakly connected component has {len(largest_cc)} nodes")

        if len(largest_cc) <= node_limit:
            sub_G = G.subgraph(largest_cc).copy()
            info("Using largest component as it is within the node limit")
            return renumber_graph(sub_G)

        # Otherwise, perform a BFS from a random node in the largest component.
        start_node = random.choice(list(largest_cc))
        debug(f"Starting BFS from node {start_node}")

        subgraph_nodes: Set[Any] = {start_node}
        frontier: List[Any] = [start_node]
        nodes_added_in_iteration = 0

        while len(subgraph_nodes) < node_limit and frontier:
            current = frontier.pop(0)
            # For MultiDiGraph, consider both successors and predecessors.
            neighbors = list(G.successors(current)) + list(G.predecessors(current))

            for neighbor in neighbors:
                if neighbor not in subgraph_nodes:
                    subgraph_nodes.add(neighbor)
                    frontier.append(neighbor)
                    nodes_added_in_iteration += 1

                    if len(subgraph_nodes) >= node_limit:
                        break

            # Log progress periodically
            if nodes_added_in_iteration >= 100:
                debug(f"BFS progress: {len(subgraph_nodes)} nodes selected")
                nodes_added_in_iteration = 0

        if not frontier and len(subgraph_nodes) < node_limit:
            warning(f"Frontier exhausted before reaching node limit; resulting subgraph has {len(subgraph_nodes)} nodes instead of {node_limit}")

        sub_G = G.subgraph(subgraph_nodes).copy()
        reduced_graph = renumber_graph(sub_G)
        success(f"Graph reduced and renumbered to {reduced_graph.number_of_nodes()} nodes")
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
    debug(f"Computing {distance}-hop neighbors for {len(node_set)} starting nodes")

    result: Set[Any] = set(node_set)
    initial_count = len(result)

    for node in node_set:
        # Compute shortest path lengths up to the given cutoff.
        neighbors = nx.single_source_shortest_path_length(G, node, cutoff=distance)
        result.update(neighbors.keys())

    neighbors_found = len(result) - initial_count
    debug(f"Found {neighbors_found} additional neighbors within distance {distance}")

    return result


@typechecked
def compute_territory_distbase(G: nx.MultiDiGraph, set0_nodes: List[int], set1_nodes: List[int]) -> str:
    """
    Compute the territory based on distance to the two sets of nodes.

    Args:
        G: NetworkX MultiDiGraph
        set0_nodes: List of node IDs belonging to set 0
        set1_nodes: List of node IDs belonging to set 1

    Returns:
        Binary string where position i represents node i's assignment (0 or 1).
        Leading zeros are preserved. Equal distances are randomly assigned.
    """
    num_nodes = G.number_of_nodes()
    assignment = []

    for node_id in range(num_nodes):
        if node_id not in G.nodes():
            # If node doesn't exist, default to 0
            assignment.append("0")
            continue

        # Compute shortest distance to any node in set 0
        min_dist_to_set0 = float("inf")
        for set0_node in set0_nodes:
            if set0_node in G.nodes():
                try:
                    dist = nx.shortest_path_length(G, node_id, set0_node)
                    min_dist_to_set0 = min(min_dist_to_set0, dist)
                except nx.NetworkXNoPath:
                    continue

        # Compute shortest distance to any node in set 1
        min_dist_to_set1 = float("inf")
        for set1_node in set1_nodes:
            if set1_node in G.nodes():
                try:
                    dist = nx.shortest_path_length(G, node_id, set1_node)
                    min_dist_to_set1 = min(min_dist_to_set1, dist)
                except nx.NetworkXNoPath:
                    continue

        # Assign to closer set, randomly choose if equal
        if min_dist_to_set0 < min_dist_to_set1:
            assignment.append("0")
        elif min_dist_to_set1 < min_dist_to_set0:
            assignment.append("1")
        else:
            # Equal distances (including both inf) - randomly assign
            assignment.append(random.choice(["0", "1"]))

    return "".join(assignment)
