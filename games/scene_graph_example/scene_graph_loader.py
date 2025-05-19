import spark_dsg as dsg
import networkx as nx
from math import sqrt


def get_places_graph_from_dsg(path):
    G = dsg.DynamicSceneGraph.load(str(path))

    # Get the places layer (usually layer 3)
    places_layer = G.get_layer(dsg.DsgLayers.PLACES)

    # Manual conversion to NetworkX graph
    nx_places = nx.DiGraph()  # Use DiGraph
    # nx_places = nx.Graph() # Use undirected graph

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
        # nx_places.add_edge(edge.target, edge.source, id = (nt.get('id'),ns.get('id')), length = dist)

    # Print basic info about the places graph
    print(f"DSG Places subgraph has {nx_places.number_of_nodes()} nodes and {nx_places.number_of_edges()} edges")

    return nx_places