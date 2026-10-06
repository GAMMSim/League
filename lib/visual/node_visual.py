from typing import Any, Dict, Optional
from typeguard import typechecked

try:
    from lib.core.console import *
    from lib.core.visibility_polygon import VisibilityPolygonIndex
except ImportError:
    from ..core.console import *
    from ..core.visibility_polygon import VisibilityPolygonIndex

# Transparency (alpha) for all sensor region node-halo circles
SENSOR_ALPHA = 0.11
# Outline width (pixels) for all sensor region node-halo circles
SENSOR_EDGE_WIDTH = 1
# Outline color (RGB) and alpha for all sensor region node-halo circles
SENSOR_EDGE_COLOR = (200, 200, 200)
SENSOR_EDGE_ALPHA = 150
# World-unit radius of the small halo drawn around each node inside a
# sensing region (not the sensing radius itself — a fixed marker size so
# adjacent in-region nodes visually blend into a covered area).
SENSOR_NODE_HALO_RADIUS = 30
# Screen-pixel width of the line connecting two in-region nodes that share a
# graph edge (see sensor_region_edges in visualization config).
SENSOR_EDGE_LINE_WIDTH = 12

# World-unit radius of the small soft-glow marker drawn at each node the
# real per-node table (not the rendering polygon) considers visible — much
# smaller than SENSOR_NODE_HALO_RADIUS, no hard border, only used for
# line_of_sight-backed sensors (alongside the visibility polygon fill). Drawn
# on a layer below the graph's own node markers (see layer=9 at the artist
# creation site), so it reads as a glow the node sits on top of, not a
# recoloring of the node itself.
GLOW_NODE_RADIUS = 8
# Concentric rings used to fake a soft radial-gradient falloff (outer =
# faintest, inner = brightest) — cheap approximation, no per-pixel gradient.
GLOW_RINGS = 5
# Alpha floor/ceiling for the glow gradient (outermost/innermost ring). Both
# are set well above SENSOR_ALPHA*255 (~38, the region-fill's own alpha) —
# a glow marking a *specific* visible node should always read as more solid
# than the diffuse region wash behind it, at every ring, not just at its
# brightest point.
GLOW_MIN_ALPHA = 65
GLOW_MAX_ALPHA = 150


@typechecked
class NodeVisual:
    """
    Base for visuals anchored to graph nodes that draw a sensing region
    around them (agents, flags/towers). Owns the static graph-geometry
    caches and the line_of_sight region drawing both share.
    """

    def __init__(self, ctx: Any, config: Dict[str, Any]):
        """
        Args:
            ctx: Game context object with visualization capabilities
            config: Complete configuration dictionary
        """
        self.ctx = ctx
        self.config = config
        self.vis_config = config.get("visualization", {})

        # Color and size settings
        self.colors = self.vis_config.get("colors", {})
        self.sizes = self.vis_config.get("sizes", {})

        # Shared label settings for moving, dead, and flag-capture labels
        self._label_spacing = self.vis_config.get("label_spacing", 15)
        self._label_y_offset = self.vis_config.get("label_y_offset", -self._label_spacing)

        # Lazily-populated {node_id: (x, y)} cache — graph nodes are static
        # for the whole game, so this is built once and reused every frame.
        self._node_coords: Dict[int, Any] = {}
        # Lazily-populated {node_id: {neighbor_id, ...}} cache, symmetric
        # (both directions of an edge) — same lifetime as _node_coords.
        self._node_neighbors: Dict[int, set] = {}
        # Lazily-populated {frozenset({u, v}): [(x, y), ...]} cache of each
        # edge's real linestring geometry, so region-connector lines follow
        # the actual road shape instead of a straight segment.
        self._edge_geometries: Dict[frozenset, list] = {}
        # Whether to connect two in-region nodes with a line when they share
        # a graph edge (config: visualization.sensor_region_edges).
        self._draw_region_edges = self.vis_config.get("sensor_region_edges", True)

        # Building index for line_of_sight region polygons — built by the
        # subclass (only when vis is on and it carries such a sensor), then
        # queried per-origin at render time.
        self._vis_polygon_index: Optional[VisibilityPolygonIndex] = None

    def _get_node_coords(self, ctx: Any) -> Dict[int, Any]:
        """Lazily cache every graph node's (x, y) — nodes are static for the game."""
        if not self._node_coords:
            try:
                for node_id in ctx.graph.graph.get_nodes():
                    node = ctx.graph.graph.get_node(node_id)
                    self._node_coords[node_id] = (node.x, node.y)
            except Exception as e:
                warning(f"Failed to cache node coordinates for sensor halos: {e}")
        return self._node_coords

    def _get_node_neighbors(self, ctx: Any) -> Dict[int, set]:
        """Lazily cache a symmetric adjacency set for every node — nodes and
        edges are static for the game. Symmetric because the underlying graph
        may only store one direction of a two-way street; for "are these two
        nodes connected" purposes either direction counts."""
        if not self._node_neighbors:
            try:
                for node_id in ctx.graph.graph.get_nodes():
                    self._node_neighbors.setdefault(node_id, set())
                    for neighbor_id in ctx.graph.graph.get_neighbors(node_id):
                        self._node_neighbors[node_id].add(neighbor_id)
                        self._node_neighbors.setdefault(neighbor_id, set()).add(node_id)
            except Exception as e:
                warning(f"Failed to cache node adjacency for sensor region edges: {e}")
        return self._node_neighbors

    def _get_edge_geometries(self, ctx: Any) -> Dict[frozenset, list]:
        """Lazily cache each edge's real linestring geometry (a list of
        (x, y) world points), keyed by the unordered {source, target} node
        pair — nodes/edges are static for the game. Falls back to a straight
        2-point segment per edge if no linestring is set (matches gamms'
        own add_edge fallback), so region-connector lines follow the actual
        road shape instead of a straight line between node centers."""
        if not self._edge_geometries:
            try:
                for edge_id in ctx.graph.graph.get_edges():
                    edge = ctx.graph.graph.get_edge(edge_id)
                    key = frozenset((edge.source, edge.target))
                    if key in self._edge_geometries:
                        continue
                    points = list(edge.linestring.coords) if edge.linestring is not None else None
                    if not points or len(points) < 2:
                        node_coords = self._get_node_coords(ctx)
                        points = [node_coords[edge.source], node_coords[edge.target]]
                    self._edge_geometries[key] = points
            except Exception as e:
                warning(f"Failed to cache edge geometries for sensor region edges: {e}")
        return self._edge_geometries

    def _draw_visibility_polygon(self, ctx: Any, points, color: tuple) -> None:
        """Fill a precomputed visibility-polygon boundary (world coords, in
        angular order — see lib/core/visibility_polygon.py) as one polygon.
        The boundary is already shaped by real building edges, so no gap
        heuristic is needed here: it's just a fill."""
        if len(points) < 3:
            return
        if len(color) == 4:
            r, g, b, a = color
        else:
            r, g, b, a = color[0], color[1], color[2], 100
        fill_color = (r, g, b, a)

        render_manager = ctx.visual._render_manager
        surface = ctx.visual._get_target_surface()
        try:
            import pygame

            screen_points = [render_manager.world_to_screen(px, py) for px, py in points]

            pad = 2
            xs = [p[0] for p in screen_points]
            ys = [p[1] for p in screen_points]
            min_x, max_x = min(xs) - pad, max(xs) + pad
            min_y, max_y = min(ys) - pad, max(ys) + pad
            w, h = int(max_x - min_x), int(max_y - min_y)
            if w <= 0 or h <= 0:
                return

            temp_surface = pygame.Surface((w, h), pygame.SRCALPHA)
            local_points = [(px - min_x, py - min_y) for px, py in screen_points]
            pygame.draw.polygon(temp_surface, fill_color, local_points)
            surface.blit(temp_surface, (int(min_x), int(min_y)))
        except Exception:
            pass  # Silent fail

    def _draw_node_glows(self, ctx: Any, points, color: tuple) -> None:
        """Small soft glow marker at each node the real per-node table (not
        the rendering polygon) actually considers visible — a few
        concentric circles of increasing alpha toward the center fake a
        radial-gradient falloff, smaller and airier than the old hard-edged
        halo, without a border."""
        if not points:
            return
        r, g, b = color[0], color[1], color[2]

        render_manager = ctx.visual._render_manager
        screen_radius = render_manager.world_to_screen_scale(GLOW_NODE_RADIUS)
        if screen_radius < 1:
            return
        surface = ctx.visual._get_target_surface()
        try:
            import pygame

            surf_size = int(screen_radius * 2 + 4)
            center = int(screen_radius + 2)
            glow_sprite = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)
            for ring in range(GLOW_RINGS, 0, -1):
                ring_radius = max(1, int(screen_radius * ring / GLOW_RINGS))
                # ring=GLOW_RINGS (outermost) -> GLOW_MIN_ALPHA; ring=1 (innermost) -> GLOW_MAX_ALPHA.
                t = (GLOW_RINGS - ring) / (GLOW_RINGS - 1)
                ring_alpha = int(GLOW_MIN_ALPHA + (GLOW_MAX_ALPHA - GLOW_MIN_ALPHA) * t)
                pygame.draw.circle(glow_sprite, (r, g, b, ring_alpha), (center, center), ring_radius)

            for x, y in points:
                screen_x, screen_y = render_manager.world_to_screen(x, y)
                blit_pos = (int(screen_x - screen_radius - 2), int(screen_y - screen_radius - 2))
                surface.blit(glow_sprite, blit_pos)
        except Exception:
            pass  # Silent fail

