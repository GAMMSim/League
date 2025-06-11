import random
import numpy as np
import networkx as nx
from copy import deepcopy
from typeguard import typechecked
from scipy.spatial import Delaunay
from shapely.geometry import LineString


try:
    from lib.core.core import *
except ModuleNotFoundError:
    from ..core.core import *


@typechecked
def cast_to_multidigraph(G: nx.DiGraph, debug: Optional[bool] = False) -> nx.MultiDiGraph:
    """
    Convert a DiGraph to a MultiDiGraph ensuring bidirectionality and unique edge IDs.

    For each directed edge (u, v) in G:
      - Assign a fresh integer 'id' to (u, v).
      - If the reverse edge (v, u) does not exist in G, add it with its own new 'id'.
        If the edge data contains a 'linestring', reverse its coordinates for the reverse edge.

    Args:
        G (nx.DiGraph): Input directed graph. May have node and edge attributes.
        debug (Optional[bool]): If True, print debug/info messages.

    Returns:
        nx.MultiDiGraph: A MultiDiGraph with bidirectional edges and unique 'id' for every edge.
    """
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
            if "linestring" in existing_data and isinstance(existing_data["linestring"], LineString):
                existing_data["linestring"] = LineString(existing_data["linestring"].coords[::-1])
            existing_data["id"] = next_edge_id
            MD.add_edge(v, u, **existing_data)
            next_edge_id += 1

    info(f"Added {added_reverse_count} reverse edge(s) to ensure bidirectionality.", debug)
    success(
        f"Converted DiGraph to MultiDiGraph with {MD.number_of_nodes()} nodes and {MD.number_of_edges()} edges.",
        debug
    )
    return MD


def convert_gml_to_multidigraph(G: nx.Graph, scale_factor: float = 1, offset_x: float = 0, offset_y: float = 0, debug: Optional[bool] = False) -> nx.MultiDiGraph:
    """
    Convert a normalized 3D graph to a MultiDiGraph with 2D coordinates and LineString edges.

    Args:
        G: Input graph with normalized coordinates
        scale_factor: Scaling factor for coordinates
        offset_x: X coordinate offset
        offset_y: Y coordinate offset
        debug: Debug flag

    Returns:
        nx.MultiDiGraph: Spatial MultiDiGraph with LineString edges
    """
    try: 
        # Create new DiGraph
        spatial_graph = nx.DiGraph()

        # Transform node coordinates, casting the ID to int
        for node_str, data in G.nodes(data=True):
            node_int = int(node_str)  # convert "5764607…" → 5764607 (or however your IDs parse)
            new_x = data["x"] * scale_factor + offset_x
            new_y = data["y"] * scale_factor + offset_y
            spatial_graph.add_node(node_int, x=new_x, y=new_y)

        # Add edges with LineString geometry and length
        edge_id = 0
        for u_str, v_str, data in G.edges(data=True):
            u = int(u_str)
            v = int(v_str)

            u_coords = spatial_graph.nodes[u]
            v_coords = spatial_graph.nodes[v]

            linestring = LineString([
                (u_coords["x"], u_coords["y"]),
                (v_coords["x"], v_coords["y"])
            ])
            length = ((v_coords["x"] - u_coords["x"])**2 + (v_coords["y"] - u_coords["y"])**2)**0.5

            spatial_graph.add_edge(u, v, id=edge_id, linestring=linestring, length=length)
            edge_id += 1
    except Exception as e:
        error(f"Error converting gml to spatial DiGraph: {e}")
        raise Exception(f"Error converting to gml spatial DiGraph: {e}")

    success(f"Converted to spatial DiGraph with {spatial_graph.number_of_nodes()} nodes and {spatial_graph.number_of_edges()} edges", debug)

    # Convert to MultiDiGraph with bidirectional edges
    return cast_to_multidigraph(spatial_graph, debug)


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


# Example usage:
# if __name__ == "__main__":
#     try:
#         from lib.core.core import *
#         import lib.visual.graph_visualizer as gfvis
#         from lib.utils.file_utils import export_graph_pkl, add_root_folder_to_sys_path
#         from lib.utils.strategy_utils import compute_convex_hull_and_perimeter, compute_attraction_distances
#     except ModuleNotFoundError:
#         from ..core.core import *
#         from visual import graph_visualizer as gfvis

#     # G = generate_random_delaunay_graph(n_points=400, side=10, seed=42, debug=True)
#     # G = generate_simple_grid(rows=20, cols=20, debug=True)
#     # G = generate_lattice_grid(rows=20, cols=20, debug=True)

#     root_folder = add_root_folder_to_sys_path()

#     config_file_path = os.path.join(root_folder, "data/config/config_04151259_ga/F5A10D10_0ac5d2/F5A10D10_0ac5d2_r38.yml")
#     graph_file_path = os.path.join(root_folder, "data/graphs/graph_200_200_a.pkl")  # Example path to your graph file

#     G = export_graph_pkl(graph_file_path)
#     visualizer = gfvis.GraphVisualizer(file_path=config_file_path, mode="interactive", simple_layout=False, debug=True, node_size=100)

#     # attackers_positions = [23, 171, 178, 176, 181]
#     # visualizer.color_nodes(attackers_positions, color="red", mode="solid", name="Attacker", size_multiplier=1.3)

#     # defenders_positions = [167, 166, 70, 148, 195]
#     # visualizer.color_nodes(defenders_positions, color="blue", mode="solid", name="Defender", size_multiplier=1.3)

#     flag_positions = [30, 31, 13, 5, 182]
#     # # visualizer.color_nodes(flag_positions, color="green", mode="solid", name="Flag")
#     target_nodes = compute_x_neighbors(G, set(flag_positions), 2)
#     # attraction_distances_dict = compute_attraction_distances(G, set(flag_positions), debug=False)

#     H, P = compute_convex_hull_and_perimeter(G, target_nodes, visualize_steps=False, attraction_distances_dict=None)
#     print(target_nodes)
#     print(f"Convex Hull: {H}")
#     print(f"Perimeter: {P}")

#     visualizer.color_nodes(list(H), color="green", mode="transparent", name="Convex Hull")
#     visualizer.color_nodes(P, color="orange", mode="solid", name="Perimeter")
#     # visualizer.color_nodes(flag_positions, color="green", mode="solid", name="Flag")
#     # visualizer.color_nodes(defenders_positions, color="blue", mode="solid", name="Defender")

#     visualizer.visualize()

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
