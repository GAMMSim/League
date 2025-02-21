import pickle

# Load the graph from the pickle file
with open("graph.pkl", "rb") as f:
    G = pickle.load(f)

# Open a text file for writing the graph details
with open("graph.txt", "w") as f:
    # Write nodes and their attributes
    f.write("Nodes:\n")
    for node, data in G.nodes(data=True):
        f.write(f"{node}: {data}\n")
    
    f.write("\nEdges:\n")
    # Write edges with id and computed length
    for u, v, data in G.edges(data=True):
        edge_id = data.get('id', 'N/A')
        # Compute the length from the LINESTRING geometry if available
        if 'linestring' in data:
            length = data['linestring'].length
        else:
            length = "N/A"
        f.write(f"{u} -> {v}: id: {edge_id}, length: {length}\n")

print("Graph successfully converted to graph.txt with edge lengths.")
