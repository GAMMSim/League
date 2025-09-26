from collections import defaultdict
import networkx as nx
import numpy as np
import random


def get_nodes_by_distance(G: nx.Graph, center_node: int) -> dict:
    """
    Groups nodes by their distance from the center node.

    Parameters:
    -----------
    G : nx.Graph
        The input graph
    center_node : int
        The center node ID to measure distances from

    Returns:
    --------
    dict
        A dictionary where keys are distances and values are lists of nodes at that distance
    """
    distances = nx.single_source_shortest_path_length(G, center_node)
    nodes_by_distance = defaultdict(list)

    for node, distance in distances.items():
        nodes_by_distance[distance].append(node)

    return nodes_by_distance


def distribute_uniform_random(G: nx.Graph, center_node: int, num_nodes: int, max_distance: int = None) -> list:
    """
    Selects nodes uniformly at random from the graph, optionally within a max distance.

    Parameters:
    -----------
    G : nx.Graph
        The input graph
    center_node : int
        The center node
    num_nodes : int
        Number of nodes to select
    max_distance : int, optional
        Maximum distance from center node. If None, all nodes are considered.

    Returns:
    --------
    list
        Selected nodes
    """
    nodes_by_distance = get_nodes_by_distance(G, center_node)

    # Filter nodes by max_distance if specified
    eligible_nodes = []
    if max_distance is not None:
        for dist, nodes in nodes_by_distance.items():
            if dist <= max_distance:
                eligible_nodes.extend(nodes)
    else:
        # All nodes
        eligible_nodes = list(G.nodes())

    # If we need more nodes than available, return all eligible nodes
    if num_nodes >= len(eligible_nodes):
        return eligible_nodes

    return random.sample(eligible_nodes, num_nodes)


def distribute_normal(G: nx.Graph, center_node: int, num_nodes: int, mean_distance: float = 1, std_dev: float = 0.5) -> list:
    """
    Selects nodes following a normal distribution based on distance from center.

    Parameters:
    -----------
    G : nx.Graph
        The input graph
    center_node : int
        The center node
    num_nodes : int
        Number of nodes to select
    mean_distance : float
        Mean distance from center node
    std_dev : float
        Standard deviation of the distance

    Returns:
    --------
    list
        Selected nodes
    """
    nodes_by_distance = get_nodes_by_distance(G, center_node)
    if not nodes_by_distance:
        return []

    # Calculate probability weights for each distance
    max_dist = max(nodes_by_distance.keys())
    distance_weights = {}

    for dist in range(0, max_dist + 1):
        if dist in nodes_by_distance:
            # Normal PDF centered at mean_distance
            weight = np.exp(-0.5 * ((dist - mean_distance) / std_dev) ** 2)
            distance_weights[dist] = weight

    # Normalize weights
    total_weight = sum(distance_weights.values())
    if total_weight == 0:
        # Fallback to uniform if all weights are zero
        return distribute_uniform_random(G, center_node, num_nodes)

    for dist in distance_weights:
        distance_weights[dist] /= total_weight

    # Select distances based on weights
    selected_nodes = []
    remaining_nodes = num_nodes

    while remaining_nodes > 0 and distance_weights:
        # Sample distances with replacement according to their weights
        distances = list(distance_weights.keys())
        probs = [distance_weights[d] for d in distances]

        sampled_distance = np.random.choice(distances, p=probs)

        # Choose a random node from the sampled distance
        if nodes_by_distance[sampled_distance]:
            node = random.choice(nodes_by_distance[sampled_distance])
            selected_nodes.append(node)

            # Remove the selected node to avoid duplicates
            nodes_by_distance[sampled_distance].remove(node)
            if not nodes_by_distance[sampled_distance]:
                del nodes_by_distance[sampled_distance]
                del distance_weights[sampled_distance]

                # Renormalize weights if needed
                if distance_weights:
                    total_weight = sum(distance_weights.values())
                    for dist in distance_weights:
                        distance_weights[dist] /= total_weight

            remaining_nodes -= 1
        else:
            # If no nodes at this distance, remove it
            del nodes_by_distance[sampled_distance]
            del distance_weights[sampled_distance]

            # Renormalize weights if needed
            if distance_weights:
                total_weight = sum(distance_weights.values())
                for dist in distance_weights:
                    distance_weights[dist] /= total_weight

    return selected_nodes


def distribute_exponential(G: nx.Graph, center_node: int, num_nodes: int, scale: float = 1.0) -> list:
    """
    Selects nodes following an exponential distribution based on distance.
    Nodes closer to the center have higher probability.

    Parameters:
    -----------
    G : nx.Graph
        The input graph
    center_node : int
        The center node
    num_nodes : int
        Number of nodes to select
    scale : float
        Scale parameter of the exponential distribution (1/lambda)

    Returns:
    --------
    list
        Selected nodes
    """
    nodes_by_distance = get_nodes_by_distance(G, center_node)
    if not nodes_by_distance:
        return []

    # Calculate probability weights for each distance
    distance_weights = {}

    for dist in nodes_by_distance:
        # Exponential PDF
        weight = np.exp(-dist / scale)
        distance_weights[dist] = weight

    # Same selection logic as in normal distribution
    total_weight = sum(distance_weights.values())
    for dist in distance_weights:
        distance_weights[dist] /= total_weight

    selected_nodes = []
    remaining_nodes = num_nodes

    while remaining_nodes > 0 and distance_weights:
        distances = list(distance_weights.keys())
        probs = [distance_weights[d] for d in distances]

        sampled_distance = np.random.choice(distances, p=probs)

        if nodes_by_distance[sampled_distance]:
            node = random.choice(nodes_by_distance[sampled_distance])
            selected_nodes.append(node)

            nodes_by_distance[sampled_distance].remove(node)
            if not nodes_by_distance[sampled_distance]:
                del nodes_by_distance[sampled_distance]
                del distance_weights[sampled_distance]

                if distance_weights:
                    total_weight = sum(distance_weights.values())
                    for dist in distance_weights:
                        distance_weights[dist] /= total_weight

            remaining_nodes -= 1
        else:
            del nodes_by_distance[sampled_distance]
            del distance_weights[sampled_distance]

            if distance_weights:
                total_weight = sum(distance_weights.values())
                for dist in distance_weights:
                    distance_weights[dist] /= total_weight

    return selected_nodes


def distribute_power_law(G: nx.Graph, center_node: int, num_nodes: int, exponent: float = 2.0) -> list:
    """
    Selects nodes following a power law distribution based on distance.

    Parameters:
    -----------
    G : nx.Graph
        The input graph
    center_node : int
        The center node
    num_nodes : int
        Number of nodes to select
    exponent : float
        Exponent of the power law distribution

    Returns:
    --------
    list
        Selected nodes
    """
    nodes_by_distance = get_nodes_by_distance(G, center_node)
    if not nodes_by_distance:
        return []

    # Calculate probability weights for each distance
    distance_weights = {}

    for dist in nodes_by_distance:
        # Power law: p(x) ∝ x^(-exponent)
        weight = 1 / (dist**exponent) if dist > 0 else 0
        distance_weights[dist] = weight

    # Same selection logic as before
    total_weight = sum(distance_weights.values())
    for dist in distance_weights:
        distance_weights[dist] /= total_weight

    selected_nodes = []
    remaining_nodes = num_nodes

    while remaining_nodes > 0 and distance_weights:
        distances = list(distance_weights.keys())
        probs = [distance_weights[d] for d in distances]

        sampled_distance = np.random.choice(distances, p=probs)

        if nodes_by_distance[sampled_distance]:
            node = random.choice(nodes_by_distance[sampled_distance])
            selected_nodes.append(node)

            nodes_by_distance[sampled_distance].remove(node)
            if not nodes_by_distance[sampled_distance]:
                del nodes_by_distance[sampled_distance]
                del distance_weights[sampled_distance]

                if distance_weights:
                    total_weight = sum(distance_weights.values())
                    for dist in distance_weights:
                        distance_weights[dist] /= total_weight

            remaining_nodes -= 1
        else:
            del nodes_by_distance[sampled_distance]
            del distance_weights[sampled_distance]

            if distance_weights:
                total_weight = sum(distance_weights.values())
                for dist in distance_weights:
                    distance_weights[dist] /= total_weight

    return selected_nodes


def distribute_beta(G: nx.Graph, center_node: int, num_nodes: int, alpha: float = 2.0, beta: float = 2.0, max_distance: int = None) -> list:
    """
    Selects nodes following a beta distribution based on normalized distance.

    Parameters:
    -----------
    G : nx.Graph
        The input graph
    center_node : int
        The center node
    num_nodes : int
        Number of nodes to select
    alpha : float
        First shape parameter of the beta distribution
    beta : float
        Second shape parameter of the beta distribution
    max_distance : int, optional
        Maximum distance to consider

    Returns:
    --------
    list
        Selected nodes
    """
    nodes_by_distance = get_nodes_by_distance(G, center_node)
    if not nodes_by_distance:
        return []

    # Determine max distance
    max_dist = max(nodes_by_distance.keys())
    if max_distance is not None:
        max_dist = min(max_dist, max_distance)
        # Filter nodes beyond max_distance
        nodes_by_distance = {d: nodes for d, nodes in nodes_by_distance.items() if d <= max_distance}

    # Calculate beta distribution weights
    distance_weights = {}
    for dist in nodes_by_distance:
        # Normalize distance to [0,1] for beta distribution
        x = dist / max_dist if max_dist > 0 else 0
        # Apply beta PDF - avoiding division by zero
        if 0 < x < 1:  # Beta PDF is defined on (0,1)
            weight = x ** (alpha - 1) * (1 - x) ** (beta - 1)
        else:
            weight = 0
        distance_weights[dist] = weight

    # Normalize weights
    total_weight = sum(distance_weights.values())
    if total_weight > 0:
        for dist in distance_weights:
            distance_weights[dist] /= total_weight
    else:
        # Fallback to uniform if all weights are zero
        return distribute_uniform_random(G, center_node, num_nodes, max_distance)

    # Select nodes using the same mechanism as other distributions
    selected_nodes = []
    remaining_nodes = num_nodes

    while remaining_nodes > 0 and distance_weights:
        distances = list(distance_weights.keys())
        probs = [distance_weights[d] for d in distances]

        sampled_distance = np.random.choice(distances, p=probs)

        if nodes_by_distance[sampled_distance]:
            node = random.choice(nodes_by_distance[sampled_distance])
            selected_nodes.append(node)

            nodes_by_distance[sampled_distance].remove(node)
            if not nodes_by_distance[sampled_distance]:
                del nodes_by_distance[sampled_distance]
                del distance_weights[sampled_distance]

                if distance_weights:
                    total_weight = sum(distance_weights.values())
                    for dist in distance_weights:
                        distance_weights[dist] /= total_weight

            remaining_nodes -= 1
        else:
            del nodes_by_distance[sampled_distance]
            del distance_weights[sampled_distance]

            if distance_weights:
                total_weight = sum(distance_weights.values())
                for dist in distance_weights:
                    distance_weights[dist] /= total_weight

    return selected_nodes


def distribute_degree_weighted(G: nx.Graph, center_node: int, num_nodes: int, favor_high_degree: bool = True, max_distance: int = None) -> list:
    """
    Selects nodes weighted by their degree (can favor high or low degree nodes).

    Parameters:
    -----------
    G : nx.Graph
        The input graph
    center_node : int
        The center node
    num_nodes : int
        Number of nodes to select
    favor_high_degree : bool
        If True, favor nodes with higher degrees; if False, favor lower degrees
    max_distance : int, optional
        Maximum distance to consider

    Returns:
    --------
    list
        Selected nodes
    """
    nodes_by_distance = get_nodes_by_distance(G, center_node)
    if not nodes_by_distance:
        return []

    # Filter by max_distance if needed
    if max_distance is not None:
        nodes_by_distance = {d: nodes for d, nodes in nodes_by_distance.items() if d <= max_distance}

    # Collect all eligible nodes with their distances and degrees
    eligible_nodes = []
    for dist, nodes in nodes_by_distance.items():
        for node in nodes:
            degree = G.degree(node)
            eligible_nodes.append((node, degree))

    if not eligible_nodes:
        return []

    # Calculate weights based on degree
    nodes = [n for n, _ in eligible_nodes]
    degrees = [d for _, d in eligible_nodes]

    if favor_high_degree:
        weights = np.array(degrees)
    else:
        # For low degree preference, invert the weights (add small constant to avoid division by zero)
        weights = 1.0 / (np.array(degrees) + 0.1)

    # Normalize weights
    total_weight = sum(weights)
    if total_weight > 0:
        weights = weights / total_weight
    else:
        weights = np.ones(len(nodes)) / len(nodes)

    # Sample nodes without replacement
    selected_indices = np.random.choice(len(nodes), size=min(num_nodes, len(nodes)), replace=False, p=weights)

    return [nodes[i] for i in selected_indices]


def test_distributions():
    """
    Test function that demonstrates the usage of different node distribution methods.
    """
    # Create different types of test graphs
    graphs = {
        "Small Random": nx.gnp_random_graph(20, 0.2, seed=42),
        "Grid": nx.grid_2d_graph(5, 5),  # Convert to integers for center node
        "Scale-Free": nx.barabasi_albert_graph(50, 2, seed=42),
        "Small World": nx.watts_strogatz_graph(30, 4, 0.3, seed=42),
    }

    # Convert grid graph nodes to integers for consistency
    if "Grid" in graphs:
        grid = graphs["Grid"]
        mapping = {node: i for i, node in enumerate(grid.nodes())}
        graphs["Grid"] = nx.relabel_nodes(grid, mapping)

    # Test parameters
    num_nodes_to_select = 10
    center_nodes = {"Small Random": 0, "Grid": 12, "Scale-Free": 0, "Small World": 0}  # Center of 5x5 grid after relabeling  # Hub in Barabasi-Albert is often node 0

    # Distribution methods to test
    distributions = [
        ("Uniform Random", lambda g, c, n: distribute_uniform_random(g, c, n)),
        ("Normal (μ=2, σ=1)", lambda g, c, n: distribute_normal(g, c, n, mean_distance=2, std_dev=1)),
        ("Exponential (scale=1.5)", lambda g, c, n: distribute_exponential(g, c, n, scale=1.5)),
        ("Power Law (exponent=2)", lambda g, c, n: distribute_power_law(g, c, n, exponent=2)),
        ("Beta (α=0.5, β=0.5)", lambda g, c, n: distribute_beta(g, c, n, alpha=0.5, beta=0.5)),
        ("Beta (α=2, β=5)", lambda g, c, n: distribute_beta(g, c, n, alpha=2, beta=5)),
        ("High Degree Weighted", lambda g, c, n: distribute_degree_weighted(g, c, n, favor_high_degree=True)),
        ("Low Degree Weighted", lambda g, c, n: distribute_degree_weighted(g, c, n, favor_high_degree=False)),
    ]

    # Run tests and print results
    print("=== Node Distribution Tests ===")

    for graph_name, graph in graphs.items():
        center = center_nodes[graph_name]
        print(f"\nGraph: {graph_name} (Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()})")
        print(f"Center Node: {center}")

        # Get distance distribution for reference
        distances = nx.single_source_shortest_path_length(graph, center)
        distance_counts = {}
        for node, dist in distances.items():
            distance_counts[dist] = distance_counts.get(dist, 0) + 1

        print("Distance distribution from center:")
        for dist in sorted(distance_counts.keys()):
            print(f"  Distance {dist}: {distance_counts[dist]} nodes")

        print("\nTesting distributions:")
        for dist_name, dist_func in distributions:
            selected = dist_func(graph, center, num_nodes_to_select)

            # Count distances of selected nodes
            selected_distances = [nx.shortest_path_length(graph, center, node) for node in selected]
            dist_summary = {}
            for d in selected_distances:
                dist_summary[d] = dist_summary.get(d, 0) + 1

            # Format as readable string
            dist_str = ", ".join([f"d={d}:{count}" for d, count in sorted(dist_summary.items())])

            print(f"  {dist_name}: Selected {len(selected)} nodes - {dist_str}")

    print("\nTests completed.")


if __name__ == "__main__":
    test_distributions()
