import pickle

# Load the graph from the pickle file
with open("graph.pkl", "rb") as f:
    G = pickle.load(f)

# Open a text file for writing the node and edge information
with open("matlab_graph.txt", "w") as f:
    # Write node information header
    f.write("Node information: index, pos_x, pos_y;\n")
    # Assuming node attributes 'x' and 'y' contain the coordinates.
    # Sorting nodes by index for consistent ordering.
    for node, data in sorted(G.nodes(data=True)):
        # Retrieve coordinates; if not available, use "N/A"
        x = data.get("x", "N/A")
        y = data.get("y", "N/A")
        f.write(f"{node}, {x}, {y};\n")
    
    f.write("\nEdge information: node1_index, node2_index, length;\n")
    for u, v, data in G.edges(data=True):
        # Prefer the stored 'length' attribute if available,
        # otherwise calculate from the 'linestring' geometry if it exists.
        length = data.get("length")
        if length is None:
            if "linestring" in data:
                length = data["linestring"].length
            else:
                length = "N/A"
        # Format the length as a float with three decimals if it is a number.
        if isinstance(length, (float, int)):
            length_str = f"{length:.3f}"
        else:
            length_str = str(length)
        f.write(f"{u}, {v}, {length_str};\n")

print("Graph data saved in graph.txt")
