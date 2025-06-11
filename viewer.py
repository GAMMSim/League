from lib.visual.graph_visualizer import GraphVisualizer as Visualizer

GRAPH_FILE_PATH = "data/graphs/west_point_downsampled_N_ds3k.gml"

visualizer = Visualizer(file_path=GRAPH_FILE_PATH, mode="interactive", simple_layout=False, debug=True)

visualizer.visualize()
# visualizer.visualize(save_path="graph_visualization.png")