from networkx.algorithms.planarity import check_planarity
from matplotlib.colors import to_rgba
from typeguard import typechecked
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle
import pygame
import math
import sys
import os

try:
    from ..core.core import *
    from ..utils.file_utils import read_yml_file, export_graph_dsg, export_graph_gml, export_graph_generic
    from ..utils.config_utils import extract_positions_from_config
except ImportError:
    from lib.core.core import *
    from lib.utils.file_utils import read_yml_file, export_graph_dsg, export_graph_gml, export_graph_generic
    from lib.utils.config_utils import extract_positions_from_config


@typechecked
class GraphVisualizer:
    """
    A class for visualizing graphs using NetworkX.
    """

    def __init__(
        self,
        file_path: Optional[str] = None,
        G: Optional[nx.MultiDiGraph] = None,
        simple_layout: bool = False,
        node_color: str = "gray",
        node_size: int = 60,
        edge_color: str = "black",
        mode: str = "static",  # "static" or "interactive" or "quick"
        extra_info: Optional[Dict] = None,  # Extra info to be printed on the graph
        scale_factor: float = 100,  # Scale factor for dynamic drawing transformation
        transparent_alpha: float = 0.4,  # Transparency for static drawing
        dynamic_screen_size: Tuple[int, int] = (1200, 1200),
        desired_screen_margin: float = 0.95,  # fraction of screen to use for graph fit
        static_edge_width: float = 1.2,
        static_edge_alpha: float = 0.5,
        dynamic_edge_width: int = 2,
        dynamic_edge_alpha: float = 0.5,
        dynamic_node_size_multiplier: float = 1.0,  # multiplier for instance node_size
        dynamic_node_size_factor: float = 0.2,  # factor for dynamic node drawing size
        dynamic_node_outline_thickness: int = 4,
        dynamic_selection_outline_thickness: int = 3,
        dynamic_selection_color: Tuple[int, int, int] = (128, 0, 128),  # purple
        debug: bool = False,
    ) -> None:
        """
        Initialize the graph visualizer.
        """
        self.debug = debug

        attackers_positions = []
        defenders_positions = []
        flag_positions = []
        if file_path:
            if not os.path.exists(file_path):
                error(f"File does not exist at {file_path}.")
                raise Exception(f"File does not exist at {file_path}.")
            with open(file_path, "rb") as gf:
                try:
                    # Check if file_path is an yml file
                    if file_path.endswith(".yml"):
                        warning("Loading graph from a config file.")
                        config = read_yml_file(file_path)
                        attackers_positions, defenders_positions, flag_positions, graph_name = extract_positions_from_config(config)
                        current_dir = os.path.dirname(file_path)
                        root_dir = None
                        while current_dir != '/':
                            # Check if this could be the project root (where lib exists)
                            if os.path.exists(os.path.join(current_dir, "lib")):
                                root_dir = current_dir
                                break
                            # Move up one directory
                            parent_dir = os.path.dirname(current_dir)
                            if parent_dir == current_dir:  # We've reached the root of the filesystem
                                break
                            current_dir = parent_dir
                        
                        if root_dir is None:
                            error("Could not find project root directory containing 'lib'.")
                            raise Exception("Could not find project root directory.")
                                        
                        graph_file_path = os.path.join(root_dir, "data", "graphs", graph_name)

                        if not os.path.exists(graph_file_path):
                            error(f"Graph file does not exist at {graph_file_path}.")
                            raise Exception(f"Graph file does not exist at {graph_file_path}.")

                        # Load the graph from the specified path
                        G = export_graph_generic(graph_file_path, debug=self.debug)
                        if G is None:
                            error(f"Failed to load graph from {graph_file_path}.")
                            raise Exception(f"Failed to load graph from {graph_file_path}.")
                        self.graph = G
                    elif file_path.endswith(".json"):
                        G = export_graph_dsg(file_path, debug=self.debug)
                        if G is None:
                            error(f"Failed to load graph from {file_path}.")
                            raise Exception(f"Failed to load graph from {file_path}.")
                        self.graph = G
                    elif file_path.endswith(".gml"):
                        G = export_graph_gml(file_path, debug=self.debug)
                        if G is None:
                            error(f"Failed to load graph from {file_path}.")
                            raise Exception(f"Failed to load graph from {file_path}.")
                        self.graph = G
                    else:
                        self.graph = pickle.load(gf)
                except Exception as e:
                    error(f"Error loading graph from file: {e}")
                    raise Exception(f"Error loading graph from file: {e}")
        elif G:
            self.graph = G
        else:
            error("No graph provided.")
            raise Exception("No graph provided.")

        if type(self.graph) != nx.MultiDiGraph:
            warning("Graph is not a MultiDiGraph.")

        if simple_layout:
            self._simple_layout()

        self.node_color = node_color
        self.node_size = node_size  # treated as pixel diameter in dynamic mode
        self.edge_color = edge_color
        self.scale_factor = scale_factor
        self.transparent_alpha = transparent_alpha

        if mode not in ["static", "interactive", "quick"]:
            warning("Invalid visualization mode. Defaulting to static mode.")
            mode = "static"
        self.mode = mode

        self.extra_info = extra_info or {}

        # Static visualization settings:
        self.static_edge_width = static_edge_width
        self.static_edge_alpha = static_edge_alpha

        # Dynamic visualization settings:
        self.dynamic_screen_size = dynamic_screen_size
        self.desired_screen_margin = desired_screen_margin
        self.dynamic_edge_width = dynamic_edge_width
        self.dynamic_edge_alpha = dynamic_edge_alpha
        self.dynamic_node_size_multiplier = dynamic_node_size_multiplier
        self.dynamic_node_size_factor = dynamic_node_size_factor
        self.dynamic_node_outline_thickness = dynamic_node_outline_thickness
        self.dynamic_selection_outline_thickness = dynamic_selection_outline_thickness
        self.dynamic_selection_color = dynamic_selection_color

        # Dictionaries for node coloring.
        self.node_color_mapping = {}  # {node: [color, ...]}
        self.node_transparency = {}  # {node: [alpha, ...]}
        self.node_color_groups = {}  # {group_name: (node_list, color, mode)}
        self.node_size_multiplier = {}  # {node: size}

        if attackers_positions:
            self.color_nodes(attackers_positions, "red", "solid", "Attackers", 1.5)
        if defenders_positions:
            self.color_nodes(defenders_positions, "blue", "solid", "Defenders", 1.5)
        if flag_positions:
            self.color_nodes(flag_positions, "green", "solid", "Flags", 1.5)

        success("Graph visualizer initialized successfully.", self.debug)

    def color_nodes(self, node_list: list[int], color: str, mode: str = "solid", name: Optional[str] = None, size_multiplier: float = 1.0) -> None:
        """Color a list of nodes with the specified color and transparency mode."""
        if name:
            self.node_color_groups[name] = (node_list, color, mode)
        for node in node_list:
            if node in self.graph.nodes():
                self.node_color_mapping.setdefault(node, []).append(color)
                self.node_transparency.setdefault(node, []).append(self.transparent_alpha if mode == "transparent" else 1.0)
                self.node_size_multiplier[node] = size_multiplier
            else:
                warning(f"Node {node} not found in the graph")
        success(f"Colored {len(node_list)} nodes with {color} ({mode})", self.debug)

    def remove_node_coloring(self, nodes: Optional[list[int]] = None, name: Optional[str] = None) -> None:
        warning("This function has not been tested yet.")
        """Remove coloring from specific nodes or a named group."""
        if name and name in self.node_color_groups:
            nodes_to_remove, _, _ = self.node_color_groups[name]
            for node in nodes_to_remove:
                self.node_color_mapping.pop(node, None)
                self.node_transparency.pop(node, None)
                self.node_size_multiplier.pop(node, None)
            del self.node_color_groups[name]
        elif nodes:
            for node in nodes:
                self.node_color_mapping.pop(node, None)
                self.node_transparency.pop(node, None)
                self.node_size_multiplier.pop(node, None)
        success("Node coloring removed successfully.", self.debug)

    def _simple_layout(self) -> None:
        """Rearrange the graph into a balanced simple planar layout with straight edges."""
        is_planar, _ = check_planarity(self.graph)
        if not is_planar:
            error("The provided graph is not planar.", self.debug)
            raise ValueError("The provided graph is not planar.")

        # Compute planar layout
        pos = nx.spring_layout(self.graph, seed=42)

        # Normalize positions to improve spacing
        positions = np.array(list(pos.values()))
        if len(positions) > 0:
            # Find current bounds
            min_x, min_y = positions.min(axis=0)
            max_x, max_y = positions.max(axis=0)

            # Normalize to [0.1, 0.9] range (leaving some margin)
            scale_x = max_x - min_x if max_x > min_x else 1
            scale_y = max_y - min_y if max_y > min_y else 1

            for node in pos:
                x, y = pos[node]
                x_norm = 0.1 + 0.8 * (x - min_x) / scale_x  # Scaled to [0.1, 0.9]
                y_norm = 0.1 + 0.8 * (y - min_y) / scale_y  # Scaled to [0.1, 0.9]
                pos[node] = (x_norm, y_norm)

        # Explicitly update node positions with x and y attributes
        for node in self.graph.nodes:
            self.graph.nodes[node]["x"] = pos[node][0]
            self.graph.nodes[node]["y"] = pos[node][1]

        # Remove any existing linestring attributes to ensure straight edges
        for u, v, key in self.graph.edges:
            if "linestring" in self.graph.edges[u, v, key]:
                del self.graph.edges[u, v, key]["linestring"]

        success("Graph relayouted to simple planar successfully with improved spacing.", self.debug)

    def _compute_node_positions(self) -> Dict:
        """Compute and return node positions; use provided x,y attributes or fall back to spring layout."""
        node_positions = {}
        for node, data in self.graph.nodes(data=True):
            if "x" in data and "y" in data:
                node_positions[node] = (data["x"], data["y"])
        if not node_positions:
            warning("No node positions found. Using spring layout.")
            node_positions = nx.spring_layout(self.graph)
        success(f"Computed positions for {len(node_positions)} nodes.", self.debug)
        return node_positions

    def _compute_graph_bounds(self, positions: Dict) -> Tuple:
        """Compute the bounding box of the given positions."""
        xs = [pos[0] for pos in positions.values()]
        ys = [pos[1] for pos in positions.values()]
        return min(xs), max(xs), min(ys), max(ys)

    def _transform(self, pos: Tuple, zoom: float, pan_offset: List[float], screen_size: Tuple) -> Tuple:
        """Transform a graph coordinate to screen coordinate."""
        x, y = pos
        screen_x = screen_size[0] / 2 + (x * zoom * self.scale_factor) + pan_offset[0]
        screen_y = screen_size[1] / 2 - (y * zoom * self.scale_factor) + pan_offset[1]
        return int(screen_x), int(screen_y)

    # --- Static Visualization ---

    def _visualize_static(self, save_path: Optional[str] = None, transparent_background: bool = False, quick: Optional[bool] = False, show_ids: Optional[bool] = True) -> None:
        """Internal static visualization using Matplotlib."""
        if quick:
            fig = plt.figure(figsize=(4, 4))
        else:
            fig = plt.figure(figsize=(12, 12))
        ax = plt.gca()

        # Set background transparency if requested
        if transparent_background:
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")

        node_positions = self._compute_node_positions()

        for u, v, data in self.graph.edges(data=True):
            try:
                source_pos = node_positions[u]
                target_pos = node_positions[v]
                if "linestring" in data and data["linestring"] is not None:
                    linestring = data["linestring"]
                    coords = [(x, y) for x, y in linestring.coords]
                    line_points = [source_pos] + coords + [target_pos]
                    xs, ys = zip(*line_points)
                    plt.plot(xs, ys, "k-", lw=self.static_edge_width, alpha=self.static_edge_alpha)
                else:
                    plt.plot([source_pos[0], target_pos[0]], [source_pos[1], target_pos[1]], "k-", lw=self.static_edge_width, alpha=self.static_edge_alpha)
            except Exception as e:
                warning(f"Exception {e} while drawing edge {u} -> {v}. Skipping.")
                continue
        success("Edges drawn successfully.", self.debug)

        node_colors = []
        node_sizes = []
        for node in self.graph.nodes():
            if node in self.node_color_mapping:
                colors = self.node_color_mapping[node]
                alphas = self.node_transparency[node]
                node_sizes.append(self.node_size * self.node_size_multiplier.get(node, 1.0))
                final_color = None
                for c, a in zip(colors, alphas):
                    if a == 1.0:
                        final_color = to_rgba(c, alpha=1.0)
                if final_color is None:
                    rgba_values = [to_rgba(c, alpha=a) for c, a in zip(colors, alphas)]
                    r = sum(rgba[0] for rgba in rgba_values) / len(rgba_values)
                    g = sum(rgba[1] for rgba in rgba_values) / len(rgba_values)
                    b = sum(rgba[2] for rgba in rgba_values) / len(rgba_values)
                    a = sum(rgba[3] for rgba in rgba_values) / len(rgba_values)
                    final_color = (r, g, b, a)
                node_colors.append(final_color)
            else:
                node_colors.append(to_rgba(self.node_color, alpha=1.0))
                node_sizes.append(self.node_size)

        if quick:
            node_sizes = [s * 0.3 for s in node_sizes]

        nx.draw_networkx_nodes(self.graph, pos=node_positions, node_size=node_sizes, node_color=node_colors)
        if not quick and show_ids:
            for node, pos in node_positions.items():
                plt.text(pos[0], pos[1], str(node), fontsize=8, ha="center", va="center")
        plt.title(f"Graph Visualization: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

        # Only add extra info text and legend if not in transparent mode
        if not transparent_background and not quick:
            if self.extra_info:
                info_text = "\n".join(f"{k}: {v}" for k, v in self.extra_info.items())
                plt.text(0.01, 0.99, info_text, transform=ax.transAxes, verticalalignment="top", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
            if self.node_color_groups:
                import matplotlib.lines as mlines

                legend_handles = []
                legend_labels = []
                for group, (_, color, mode) in self.node_color_groups.items():
                    if mode == "transparent":
                        group_face_color = to_rgba(color, alpha=0.5)
                        group_edge_color = to_rgba(color, alpha=1.0)
                        handle = mlines.Line2D([], [], marker="o", color=group_edge_color, markerfacecolor=group_face_color, markeredgecolor=group_edge_color, linestyle="None")
                    else:
                        group_color = to_rgba(color, alpha=1.0)
                        handle = mlines.Line2D([], [], marker="o", color="w", markerfacecolor=group_color, markeredgecolor=group_color, linestyle="None")
                    legend_handles.append(handle)
                    legend_labels.append(group)
                if legend_handles:
                    plt.legend(handles=legend_handles, labels=legend_labels)

        plt.axis("off")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=transparent_background)
            success(f"Static graph saved to {save_path}")
        plt.show()
        plt.close()

    # --- Dynamic Visualization ---

    def _display_info(self, screen: pygame.Surface, info_font: pygame.font.Font) -> None:
        """Display extra information in the top left."""
        if self.extra_info:
            extra_text = "\n".join(f"{k}: {v}" for k, v in self.extra_info.items())
            for i, line in enumerate(extra_text.split("\n")):
                text = info_font.render(line, True, (0, 0, 0))
                screen.blit(text, (10, 10 + i * 20))

    def _display_selected(self, screen: pygame.Surface, info_font: pygame.font.Font, screen_size: Tuple[int, int], selected_nodes: List[int]) -> None:
        """Display selected node IDs in the top right."""
        if selected_nodes:
            selected_text = "Selected: " + ", ".join(str(n) for n in selected_nodes)
            text = info_font.render(selected_text, True, (0, 0, 0))
            screen.blit(text, (screen_size[0] - text.get_width() - 10, 10))

    def _display_legend(self, screen: pygame.Surface, legend_font: pygame.font.Font, screen_size: Tuple[int, int]) -> None:
        """Display a legend of node colors in the bottom right corner."""
        if not self.node_color_groups:
            return  # No groups to display

        # Create a list of legend items (group name, color, mode)
        legend_items = []
        for group_name, (_, color, mode) in self.node_color_groups.items():
            legend_items.append((group_name, color, mode))

        if not legend_items:
            return

        # Legend settings
        legend_item_height: int = 25
        legend_margin: int = 20
        legend_circle_radius: int = 8

        # Calculate legend dimensions
        max_text_width: int = 0
        for group_name, _, _ in legend_items:
            text = legend_font.render(group_name, True, (0, 0, 0))
            max_text_width = max(max_text_width, text.get_width())

        legend_width: int = max_text_width + 60  # Circle + spacing + text
        legend_height: int = (len(legend_items) * legend_item_height) + (2 * legend_margin)

        # Create legend background
        legend_rect = pygame.Rect(screen_size[0] - legend_width - legend_margin, screen_size[1] - legend_height - legend_margin, legend_width, legend_height)

        # Draw semi-transparent background
        legend_surface = pygame.Surface((legend_rect.width, legend_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(legend_surface, (255, 255, 255, 220), pygame.Rect(0, 0, legend_rect.width, legend_rect.height), border_radius=10)
        pygame.draw.rect(legend_surface, (0, 0, 0, 100), pygame.Rect(0, 0, legend_rect.width, legend_rect.height), width=1, border_radius=10)

        # Draw legend title
        title_text = legend_font.render("Legend", True, (0, 0, 0))
        legend_surface.blit(title_text, (legend_rect.width // 2 - title_text.get_width() // 2, 10))

        # Draw legend items
        for i, (group_name, color, mode) in enumerate(legend_items):
            # Calculate y position
            y_pos: int = legend_margin + (i * legend_item_height) + 10

            # Draw color circle
            r, g, b, a = to_rgba(color, alpha=1.0 if mode == "solid" else self.transparent_alpha)
            circle_color = (int(r * 255), int(g * 255), int(b * 255), int(a * 255))

            # Draw filled circle
            pygame.draw.circle(legend_surface, circle_color, (20, y_pos + legend_item_height // 2), legend_circle_radius)
            # Draw circle outline
            pygame.draw.circle(legend_surface, (0, 0, 0), (20, y_pos + legend_item_height // 2), legend_circle_radius, width=1)
            # Draw text
            text = legend_font.render(group_name, True, (0, 0, 0))
            legend_surface.blit(text, (40, y_pos + (legend_item_height - text.get_height()) // 2))

        # Blit legend surface to screen
        screen.blit(legend_surface, (legend_rect.left, legend_rect.top))

    def _display_instructions(self, screen: pygame.Surface, info_font: pygame.font.Font, screen_size: Tuple[int, int]) -> None:
        """Display instructions in the bottom left."""
        instructions = ["W/A/S/D: Pan (adaptive)", "Mouse Drag: Pan", "Mouse Wheel: Zoom", "R: Recenter", "Click on node: Toggle selection", "Backspace: Clear selection", "Hover on node: Show node number", "Esc: Quit"]
        for i, line in enumerate(instructions):
            text = info_font.render(line, True, (0, 0, 0))
            screen.blit(text, (10, screen_size[1] - (len(instructions) - i) * 20 - 10))

    def _draw_edges(self, overlay: pygame.Surface, node_positions: Dict[Any, Tuple[float, float]], zoom: float, pan_offset: List[float], screen_size: Tuple[int, int]) -> None:
        """Draw edges on the overlay."""
        # Convert edge_color to RGBA (0-255) based on dynamic_edge_alpha.
        edge_rgba = tuple(int(c * 255) for c in to_rgba(self.edge_color, alpha=self.dynamic_edge_alpha))
        for u, v, data in self.graph.edges(data=True):
            pos_u = self._transform(node_positions[u], zoom, pan_offset, screen_size)
            pos_v = self._transform(node_positions[v], zoom, pan_offset, screen_size)
            if "linestring" in data and data["linestring"] is not None:
                coords = [(x, y) for x, y in data["linestring"].coords]
                points = [self._transform(node_positions[u], zoom, pan_offset, screen_size)] + [self._transform(pt, zoom, pan_offset, screen_size) for pt in coords] + [self._transform(node_positions[v], zoom, pan_offset, screen_size)]
                pygame.draw.aalines(overlay, edge_rgba, False, points, self.dynamic_edge_width)
            else:
                pygame.draw.aaline(overlay, edge_rgba, pos_u, pos_v, self.dynamic_edge_width)

    def _draw_nodes(self, overlay: pygame.Surface, node_positions: Dict[Any, Tuple[float, float]], zoom: float, pan_offset: List[float], screen_size: Tuple[int, int], selected_nodes: List[int], mouse_pos: Tuple[int, int]) -> None:
        """Draw nodes on the overlay, including hover and selection indicators with a glow effect."""
        dynamic_node_size = self.node_size * self.dynamic_node_size_multiplier
        for node in self.graph.nodes():
            pos = self._transform(node_positions[node], zoom, pan_offset, screen_size)
            current_node_size = dynamic_node_size
            # Compute final node color.
            if node in self.node_color_mapping:
                colors = self.node_color_mapping[node]
                alphas = self.node_transparency[node]
                current_node_size = dynamic_node_size * self.node_size_multiplier.get(node, 1.0)
                final_color = None
                for c, a in zip(colors, alphas):
                    if a == 1.0:
                        final_color = to_rgba(c, alpha=1.0)
                if final_color is None:
                    rgba_values = [to_rgba(c, alpha=a) for c, a in zip(colors, alphas)]
                    r = sum(rgba[0] for rgba in rgba_values) / len(rgba_values)
                    g = sum(rgba[1] for rgba in rgba_values) / len(rgba_values)
                    b = sum(rgba[2] for rgba in rgba_values) / len(rgba_values)
                    a = sum(rgba[3] for rgba in rgba_values) / len(rgba_values)
                    final_color = (r, g, b, a)
            else:
                final_color = to_rgba(self.node_color, alpha=1.0)

            # Calculate node drawing parameters.
            node_radius = int(max(current_node_size * self.dynamic_node_size_factor, 4) // 2)
            # To avoid clipping the glow/outline, create a larger surface.
            extra_margin = 16  # extra space for glow and outline
            surface_size = node_radius * 2 + extra_margin
            center = surface_size // 2

            node_surface = pygame.Surface((surface_size, surface_size), pygame.SRCALPHA)
            r, g, b, a = final_color
            node_draw_color = (int(r * 255), int(g * 255), int(b * 255), int(a * 255))
            # Draw the filled circle centered in the surface.
            pygame.draw.circle(node_surface, node_draw_color, (center, center), node_radius)
            # Draw a thin black outline.
            pygame.draw.circle(node_surface, (0, 0, 0), (center, center), node_radius, 1)

            # Determine if the node should be highlighted.
            dx = pos[0] - mouse_pos[0]
            dy = pos[1] - mouse_pos[1]
            is_highlight = math.hypot(dx, dy) <= node_radius or (node in selected_nodes)
            if is_highlight:
                # Draw an outer glow (semi-transparent purple) and a thicker purple outline.
                glow_color = (self.dynamic_selection_color[0], self.dynamic_selection_color[1], self.dynamic_selection_color[2], 100)  # semi-transparent glow
                pygame.draw.circle(node_surface, glow_color, (center, center), node_radius + 8, 4)
                pygame.draw.circle(node_surface, self.dynamic_selection_color, (center, center), node_radius + 4, self.dynamic_selection_outline_thickness)
            overlay.blit(node_surface, (pos[0] - center, pos[1] - center))

    def _display_info(self, screen: pygame.Surface, info_font: pygame.font.Font) -> None:
        """Display extra information in the top left."""
        if self.extra_info:
            extra_text: str = "\n".join(f"{k}: {v}" for k, v in self.extra_info.items())
            for i, line in enumerate(extra_text.split("\n")):
                text = info_font.render(line, True, (0, 0, 0))
                screen.blit(text, (10, 10 + i * 20))

    def _display_selected(self, screen: pygame.Surface, info_font: pygame.font.Font, screen_size: Tuple[int, int], selected_nodes: List[int]) -> None:
        """Display selected node IDs in the top right."""
        if selected_nodes:
            selected_text: str = "Selected: " + ", ".join(str(n) for n in selected_nodes)
            text = info_font.render(selected_text, True, (0, 0, 0))
            screen.blit(text, (screen_size[0] - text.get_width() - 10, 10))

    def _display_instructions(self, screen: pygame.Surface, info_font: pygame.font.Font, screen_size: Tuple[int, int]) -> None:
        """Display instructions in the bottom left."""
        instructions: List[str] = ["W/A/S/D: Pan (adaptive)", "Mouse Drag: Pan", "Mouse Wheel: Zoom", "R: Recenter", "Click on node: Toggle selection", "Backspace: Clear selection", "Hover on node: Show node number", "Esc: Quit"]
        for i, line in enumerate(instructions):
            text = info_font.render(line, True, (0, 0, 0))
            screen.blit(text, (10, screen_size[1] - (len(instructions) - i) * 20 - 10))

    def _draw_tooltip(
        self,
        screen: pygame.Surface,
        pos: Tuple[int, int],
        node_text: str,
        group_info: List[Tuple[str, Any, str]],
        font: pygame.font.Font,
        circle_radius: int = 6,
        padding: int = 4,
        extra_space: int = 20,
        line_spacing: int = 2,
        bg_color: Tuple[int, int, int, int] = (255, 255, 255, 200),
        border_color: Tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        """
        Draws a semi-transparent tooltip at the given screen position.

        Parameters:
            screen: The pygame surface.
            pos (tuple): (x,y) position to draw the tooltip.
            node_text (str): The first line of text (e.g., "Node 55").
            group_info (list of tuples): Each tuple is (group_name, color, mode).
            font: The pygame font to render text.
            circle_radius (int): Radius of the colored circle for group lines.
            padding (int): Padding around the text.
            extra_space (int): Additional horizontal space reserved for the circle in group lines.
            line_spacing (int): Vertical spacing between lines.
            bg_color (tuple): Background color with alpha (RGBA) for the tooltip.
            border_color (tuple): Border color.
        """
        # Prepare text lines: first line is node_text, then one line per group.
        lines: List[str] = [node_text] + [group for group, _, _ in group_info]
        surfaces: List[pygame.Surface] = [font.render(line, True, (0, 0, 0)) for line in lines]

        # Compute maximum width. For group lines, reserve extra_space for the colored circle.
        max_width: int = surfaces[0].get_width()
        if len(surfaces) > 1:
            group_width = max(surface.get_width() for surface in surfaces[1:]) + extra_space
            max_width = max(max_width, group_width)
        width: int = max_width + 2 * padding
        total_height: int = sum(surface.get_height() for surface in surfaces) + (len(surfaces) - 1) * line_spacing + 2 * padding

        # Create a surface with per-pixel alpha.
        tooltip_surface: pygame.Surface = pygame.Surface((width, total_height), pygame.SRCALPHA)
        tooltip_surface.fill(bg_color)
        pygame.draw.rect(tooltip_surface, border_color, tooltip_surface.get_rect(), 1)

        current_y: int = padding
        # Draw the first line (node text)
        tooltip_surface.blit(surfaces[0], (padding, current_y))
        current_y += surfaces[0].get_height() + line_spacing

        # For each group line, draw the colored circle based on the group's mode.
        for i, (group, color, mode) in enumerate(group_info):
            desired_alpha: float = 0.5 if mode == "transparent" else 1.0
            r, g, b, _ = to_rgba(color)
            circle_color: Tuple[int, int, int, int] = (int(r * 255), int(g * 255), int(b * 255), int(desired_alpha * 255))
            circle_center: Tuple[int, int] = (padding + circle_radius, current_y + circle_radius)
            pygame.draw.circle(tooltip_surface, circle_color, circle_center, circle_radius)
            tooltip_surface.blit(surfaces[i + 1], (padding + 2 * circle_radius + 4, current_y))
            current_y += surfaces[i + 1].get_height() + line_spacing

        screen.blit(tooltip_surface, pos)

    def visualize(self, save_path: Optional[str] = None, transparent_background: Optional[bool] = False, show_ids: Optional[bool] = True) -> None:
        """
        Visualize the graph based on the mode.
        If mode is "static", use Matplotlib.
        If mode is "interactive", use Pygame.
        If save_path is provided, save the output.
        """
        if self.mode == "static":
            self._visualize_static(save_path, transparent_background, show_ids=show_ids)
        elif self.mode == "quick":
            self._visualize_static(save_path, transparent_background=False, quick=True)
        else:
            pygame.init()
            font: pygame.font.Font = pygame.font.SysFont("Arial", 18)
            info_font: pygame.font.Font = pygame.font.SysFont("Arial", 16)
            legend_font: pygame.font.Font = pygame.font.SysFont("Arial", 16)
            screen_size: Tuple[int, int] = self.dynamic_screen_size
            screen: pygame.Surface = pygame.display.set_mode(screen_size, pygame.SCALED)
            pygame.display.set_caption(f"Graph Visualization: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

            node_positions: Dict[Any, Tuple[float, float]] = self._compute_node_positions()
            if not node_positions:
                error("No node positions available. Exiting dynamic visualization.", self.debug)
                pygame.quit()
                sys.exit()

            # Compute bounds and initial zoom.
            min_x, max_x, min_y, max_y = self._compute_graph_bounds(node_positions)
            graph_width: float = max_x - min_x
            graph_height: float = max_y - min_y
            desired_width: float = screen_size[0] * self.desired_screen_margin
            desired_height: float = screen_size[1] * self.desired_screen_margin
            zoom_x: float = desired_width / (graph_width * self.scale_factor)
            zoom_y: float = desired_height / (graph_height * self.scale_factor)
            zoom: float = min(zoom_x, zoom_y) * 0.9  # 90% of the minimum zoom
            pan_offset: List[float] = [0, 0]
            base_pan_speed: float = max(graph_width, graph_height) / 600

            dragging: bool = False
            drag_start: Tuple[int, int] = (0, 0)
            pan_start: List[float] = [0, 0]

            selected_nodes = set()

            def recenter() -> None:
                xs = [pos[0] for pos in node_positions.values()]
                ys = [pos[1] for pos in node_positions.values()]
                center_x: float = sum(xs) / len(xs)
                center_y: float = sum(ys) / len(ys)
                pan_offset[0] = -center_x * zoom * self.scale_factor
                pan_offset[1] = center_y * zoom * self.scale_factor

            recenter()
            clock = pygame.time.Clock()
            running: bool = True

            initial_frame_captured: bool = False
            while running:
                dt: float = clock.tick(60) / 1000
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                    elif event.type == pygame.MOUSEWHEEL:
                        old_zoom: float = zoom
                        zoom *= 1.03 if event.y > 0 else 0.97
                        mouse_pos = pygame.mouse.get_pos()
                        inv_x: float = (mouse_pos[0] - screen_size[0] / 2 - pan_offset[0]) / (old_zoom * self.scale_factor)
                        inv_y: float = (screen_size[1] / 2 + pan_offset[1] - mouse_pos[1]) / (old_zoom * self.scale_factor)
                        new_screen_x: float = screen_size[0] / 2 + (inv_x * zoom * self.scale_factor) + pan_offset[0]
                        new_screen_y: float = screen_size[1] / 2 - (inv_y * zoom * self.scale_factor) + pan_offset[1]
                        pan_offset[0] += mouse_pos[0] - new_screen_x
                        pan_offset[1] += mouse_pos[1] - new_screen_y
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        if event.key == pygame.K_r:
                            recenter()
                        if event.key == pygame.K_BACKSPACE:
                            selected_nodes.clear()
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:
                            click_pos = pygame.mouse.get_pos()
                            for node in self.graph.nodes():
                                pos = self._transform(node_positions[node], zoom, pan_offset, screen_size)
                                node_radius: int = int(max(self.node_size * self.dynamic_node_size_factor, 4) // 2)
                                dx: float = pos[0] - click_pos[0]
                                dy: float = pos[1] - click_pos[1]
                                if math.hypot(dx, dy) <= node_radius:
                                    if node in selected_nodes:
                                        selected_nodes.remove(node)
                                    else:
                                        selected_nodes.add(node)
                                    break
                            dragging = True
                            drag_start = pygame.mouse.get_pos()
                            pan_start = pan_offset.copy()
                    elif event.type == pygame.MOUSEBUTTONUP:
                        if event.button == 1:
                            dragging = False

                keys = pygame.key.get_pressed()
                if keys[pygame.K_w]:
                    pan_offset[1] += base_pan_speed * dt / zoom
                if keys[pygame.K_s]:
                    pan_offset[1] -= base_pan_speed * dt / zoom
                if keys[pygame.K_a]:
                    pan_offset[0] += base_pan_speed * dt / zoom
                if keys[pygame.K_d]:
                    pan_offset[0] -= base_pan_speed * dt / zoom

                if dragging:
                    mouse_now = pygame.mouse.get_pos()
                    dx = mouse_now[0] - drag_start[0]
                    dy = mouse_now[1] - drag_start[1]
                    pan_offset[0] = pan_start[0] + dx
                    pan_offset[1] = pan_start[1] + dy

                if transparent_background:
                    screen.fill((255, 255, 255, 0))
                else:
                    screen.fill((255, 255, 255))
                overlay = pygame.Surface(screen_size, pygame.SRCALPHA).convert_alpha()

                def transform(pos: Tuple[float, float]) -> Tuple[float, float]:
                    return self._transform(pos, zoom, pan_offset, screen_size)

                self._draw_edges(overlay, node_positions, zoom, pan_offset, screen_size)
                mouse_pos = pygame.mouse.get_pos()
                self._draw_nodes(overlay, node_positions, zoom, pan_offset, screen_size, list(selected_nodes), mouse_pos)
                screen.blit(overlay, (0, 0))

                self._display_info(screen, info_font)
                self._display_selected(screen, info_font, screen_size, list(selected_nodes))
                self._display_legend(screen, legend_font, screen_size)
                hovered_node = None
                for node in self.graph.nodes():
                    pos = transform(node_positions[node])
                    node_radius = int(max(self.node_size * self.dynamic_node_size_factor, 4) // 2)
                    dx = pos[0] - mouse_pos[0]
                    dy = pos[1] - mouse_pos[1]
                    if math.hypot(dx, dy) <= node_radius:
                        hovered_node = node
                        break
                if hovered_node is not None:
                    group_info: List[Tuple[str, Any, str]] = []
                    for group, (node_list, color, mode) in self.node_color_groups.items():
                        if hovered_node in node_list:
                            group_info.append((group, color, mode))
                    node_text: str = f"Node {hovered_node}"
                    tooltip_pos: Tuple[int, int] = (mouse_pos[0] + 10, mouse_pos[1] + 10)
                    self._draw_tooltip(screen, tooltip_pos, node_text, group_info, font)
                self._display_instructions(screen, info_font, screen_size)

                pygame.display.flip()

                if save_path and not initial_frame_captured:
                    if transparent_background:
                        # For transparent background in Pygame, need to use a different approach
                        # Create a new surface with alpha channel
                        transparent_surface = pygame.Surface(screen_size, pygame.SRCALPHA)

                        # Draw edges on transparent surface
                        self._draw_edges(transparent_surface, node_positions, zoom, pan_offset, screen_size)

                        # Draw nodes on transparent surface
                        self._draw_nodes(transparent_surface, node_positions, zoom, pan_offset, screen_size, list(selected_nodes), mouse_pos)

                        # Add legend if enabled
                        if self.node_color_groups:
                            self._display_legend(transparent_surface, legend_font, screen_size)

                        # Add node tooltip for hovered node if any
                        hovered_node = None
                        for node in self.graph.nodes():
                            pos = transform(node_positions[node])
                            node_radius = int(max(self.node_size * self.dynamic_node_size_factor, 4) // 2)
                            dx = pos[0] - mouse_pos[0]
                            dy = pos[1] - mouse_pos[1]
                            if math.hypot(dx, dy) <= node_radius:
                                hovered_node = node
                                break

                        if hovered_node is not None:
                            group_info = []
                            for group, (node_list, color, mode) in self.node_color_groups.items():
                                if hovered_node in node_list:
                                    group_info.append((group, color, mode))
                            node_text = f"Node {hovered_node}"
                            tooltip_pos = (mouse_pos[0] + 10, mouse_pos[1] + 10)
                            self._draw_tooltip(transparent_surface, tooltip_pos, node_text, group_info, font)

                        # Save the transparent surface
                        pygame.image.save(transparent_surface, save_path)
                    else:
                        # Regular save with background
                        pygame.image.save(screen, save_path)

                    success(f"Dynamic graph screenshot saved to {save_path}" + (" with transparent background" if transparent_background else ""), self.debug)
                    initial_frame_captured = True

            pygame.quit()
            sys.exit()


if __name__ == "__main__":
    root_folder = add_root_folder_to_sys_path()

    # config_file_path = os.path.join(root_folder, "data", "config", "config_gatech.yml")
    graph_file_path = os.path.join(root_folder, "data", "graphs", "graph_200_200_a.pkl")

    visualizer = GraphVisualizer(file_path=graph_file_path, mode="interactive", simple_layout=False, debug=True)

    attackers_positions = [160, 157, 113, 69]
    # visualizer.color_nodes(attackers_positions, color="red", mode="solid", name="Attacker", size_multiplier=1)

    defenders_positions = [149, 165, 70, 153]
    # visualizer.color_nodes(defenders_positions, color="blue", mode="solid", name="Defender")

    flag_positions = [29, 35, 30]
    # visualizer.color_nodes(flag_positions, color="green", mode="solid", name="Flag")

    visualizer.visualize()
    # visualizer.visualize(save_path=os.path.join(root_folder, "data", "image", "gatech.png"), transparent_background=True)
    # visualizer.visualize(save_path="graph_visualization.png")
