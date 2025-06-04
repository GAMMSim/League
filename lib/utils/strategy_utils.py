import networkx as nx
from typeguard import typechecked


try:
    from lib.core.core import *
    import lib.visual.graph_visualizer as gfvis
except ModuleNotFoundError:
    from ..core.core import *
    from visual import graph_visualizer as gfvis


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
def _V1R11_patch_boundary_for_connectivity(G: nx.Graph, P: Set[Any], H: Set[Any], penalty: Optional[int] = None, debug: bool = False) -> Set[Any]:
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
def _V1R12_patch_boundary_for_connectivity(G: nx.Graph, P: Set[Any], H: Set[Any], penalty: Optional[int] = None, debug: bool = False) -> Set[Any]:
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
def _V1R11_compute_graph_convex_hull(G: nx.Graph, S: Set[Any], visualize_steps: bool = False, debug: bool = False) -> Set[Any]:
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
def _V1R12_compute_graph_convex_hull(G: nx.Graph, S: Set[Any], visualize_steps: bool = False, debug: bool = False) -> Set[Any]:
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
