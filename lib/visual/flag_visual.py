from typing import Any, Dict, List, Optional, Tuple
from typeguard import typechecked
import networkx as nx

try:
    from lib.core.console import *
    from lib.core.visibility_lookup import resolve_line_of_sight_ranges, resolve_line_of_sight_tables, resolve_building_polygons
    from lib.core.visibility_polygon import VisibilityPolygonIndex
except ImportError:
    from ..core.console import *
    from ..core.visibility_lookup import resolve_line_of_sight_ranges, resolve_line_of_sight_tables, resolve_building_polygons
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
class FlagVisual:
    """
    Flag Visual Handler for managing all flag-related visualization.
    Handles flag creation, rendering, and sensor circles for flags.
    """

    def __init__(self, ctx: Any, config: Dict[str, Any], graph: Optional[nx.Graph] = None, vis: bool = True):
        """
        Initialize the Flag Visual Handler.

        Args:
            ctx: Game context object with visualization capabilities
            config: Complete configuration dictionary
            graph: The game's networkx graph, if available — used to look up
                any line_of_sight model's real range for static (tower)
                sensors, so they draw a true building-occluded visibility
                polygon instead of a euclidean-radius halo. None disables this.
            vis: If False (headless/no-vis run), skip resolving buildings and
                line_of_sight ranges entirely — never drawn without a display.
        """
        debug("Initializing FlagVisual")
        self.ctx = ctx
        self.config = config
        self.vis_config = config.get("visualization", {})
        self.env_config = config.get("environment", {})
        
        # Color and size settings
        self.colors = self.vis_config.get("colors", {})
        self.sizes = self.vis_config.get("sizes", {})
        self._label_spacing = self.vis_config.get("label_spacing", 15)
        self._label_y_offset = self.vis_config.get("label_y_offset", -self._label_spacing)
        
        # Flag tracking
        self.flags_created = False
        # Maps flag node_id -> (x, y) world position for later lookup
        self.flag_positions: Dict[int, Tuple[float, float]] = {}
        # Maps flag node_id -> list of (agent_name, agent_team) captures
        self.flag_captures: Dict[int, List[Tuple[str, str]]] = {}

        # Lazily-populated {node_id: (x, y)} cache — graph nodes are static
        # for the whole game, so this is built once and reused.
        self._node_coords: Dict[int, Any] = {}
        # Lazily-populated {node_id: {neighbor_id, ...}} cache, symmetric.
        self._node_neighbors: Dict[int, set] = {}
        # Lazily-populated {frozenset({u, v}): [(x, y), ...]} cache of each
        # edge's real linestring geometry.
        self._edge_geometries: Dict[frozenset, list] = {}
        # Whether to connect two in-region nodes with a line when they share
        # a graph edge (config: visualization.sensor_region_edges).
        self._draw_region_edges = self.vis_config.get("sensor_region_edges", True)

        # {node_id: max_range} for any line_of_sight model carried by a
        # static (tower, `at:`-expanded) sensor — see
        # resolve_line_of_sight_ranges. {node_id: table} is the real per-node
        # table (cheap — already built/cached for gameplay) for marking
        # exactly which nodes the actual sensor considers visible, alongside
        # the polygon. All stay empty when vis=False so a headless run never
        # fetches/indexes buildings.
        self._static_los_ranges: Dict[int, float] = {}
        self._static_los_tables: Dict[int, Any] = {}
        self._vis_polygon_index: Optional[VisibilityPolygonIndex] = None
        if vis and graph is not None:
            _, self._static_los_ranges = resolve_line_of_sight_ranges(config)
            if self._static_los_ranges:
                polygons = resolve_building_polygons(config, graph)
                self._vis_polygon_index = VisibilityPolygonIndex(polygons)
                _, self._static_los_tables = resolve_line_of_sight_tables(config, graph)

        success("FlagVisual initialized")

    def create_flags(self, flags_config: Dict[str, List[int]]) -> None:
        """
        Create flag visualizations based on the game rule version.

        Args:
            flags_config: Dictionary from the config file's 'flags' section.
        """
        if self.flags_created:
            debug("Flags already created, skipping")
            return
        debug(f"This is flags_config: {flags_config}")

        try:
            debug("Creating flags")
            game_rule = self.config.get("game", {}).get("game_rule")
            flag_size = self.sizes.get("flag_size", 15)

            # Logic for v1.2: Real (green) vs. Fake (gray) flags
            if game_rule in ("v1.2", "test"):
                # Define colors for real and fake flags
                try:
                    from gamms.VisualizationEngine import Color
                    real_flag_color = Color.Green
                    fake_flag_color = Color.Gray
                except ImportError:
                    real_flag_color = (0, 255, 0)
                    fake_flag_color = (128, 128, 128)

                # Get flag positions from the config
                real_positions = flags_config.get("real_positions", [])
                debug(f"Real flag positions: {real_positions}")
                candidate_positions = flags_config.get("candidate_positions", [])
                debug(f"Candidate flag positions: {candidate_positions}")

                # Determine which candidate flags are fake
                real_set = set(real_positions)
                candidate_set = set(candidate_positions)
                fake_positions = list(candidate_set - real_set)

                # Create the flag visuals
                self._create_team_flags("real", real_positions, real_flag_color, flag_size)
                self._create_team_flags("fake", fake_positions, fake_flag_color, flag_size)

                success(f"Created {len(real_positions)} real (green) flags and {len(fake_positions)} fake (gray) flags")

            # Default/fallback logic for v2: Red vs. Blue flags
            elif game_rule in ["v2", "v2.1"]:
                # Default colors
                try:
                    from gamms.VisualizationEngine import Color
                    red_color = self.colors.get("red_flag", Color.Red)
                    blue_color = self.colors.get("blue_flag", Color.Blue)
                except ImportError:
                    red_color = self.colors.get("red_flag", (255, 0, 0))
                    blue_color = self.colors.get("blue_flag", (0, 0, 255))

                # Create flags for each team
                red_flags = flags_config.get("red", [])
                blue_flags = flags_config.get("blue", [])

                self._create_team_flags("red", red_flags, red_color, flag_size)
                self._create_team_flags("blue", blue_flags, blue_color, flag_size)

                success(f"Created {len(red_flags)} red flags and {len(blue_flags)} blue flags")
            else:
                warning(f"Unknown game rule '{game_rule}'; skipping flag creation")
            self.flags_created = True

        except Exception as e:
            error(f"Failed to create flags: {e}")

    def _create_team_flags(self, flag_type: str, positions: List[int], color: Any, size: int) -> None:
        """
        Creates visual artists for a list of flags.
        
        Args:
            flag_type: A string identifier (e.g., "red", "blue", "real", "fake").
            positions: A list of node IDs where flags should be placed.
            color: The color of the flags.
            size: The size of the flags.
        """
        debug(f"Creating {len(positions)} {flag_type} flags")
        
        # Get stationary sensor radius from environment config
        stationary_sensor_radius = self.env_config.get("blue_stationary_sensor_radius", None)
        if stationary_sensor_radius is not None:
            debug(f"Stationary sensor radius: {stationary_sensor_radius}")
        else:
            debug("No stationary sensor radius configured")
        
        for idx, node_id in enumerate(positions):
            try:
                # Get node position
                node = self.ctx.graph.graph.get_node(node_id)
                # Store position for capture visualization lookup
                self.flag_positions[node_id] = (node.x, node.y)

                # Create sensor region node-halos if stationary_sensor_radius exists.
                # The tower's origin is static for the whole game, so the
                # qualifying node set (and its edges) — or, for a
                # line_of_sight tower (see self._static_los_ranges), the true
                # building-occluded visibility polygon — is computed once
                # here, not per frame.
                visibility_polygon = None
                glow_points = []
                los_range = self._static_los_ranges.get(node_id)
                if los_range is not None and self._vis_polygon_index is not None:
                    visibility_polygon = self._vis_polygon_index.polygon_at((node.x, node.y), los_range)
                    table = self._static_los_tables.get(node_id)
                    if table is not None:
                        node_coords = self._get_node_coords(self.ctx)
                        region = table.get(node_id, frozenset((node_id,)))
                        glow_points = [node_coords[nid] for nid in region if nid in node_coords]
                    in_region_points = []
                    edge_pairs = []
                elif stationary_sensor_radius is not None:
                    node_coords = self._get_node_coords(self.ctx)
                    r2 = stationary_sensor_radius * stationary_sensor_radius
                    in_region_ids = [
                        nid for nid, (nx_, ny_) in node_coords.items()
                        if (nx_ - node.x) ** 2 + (ny_ - node.y) ** 2 <= r2
                    ]
                    in_region_points = [node_coords[nid] for nid in in_region_ids]

                    edge_pairs = []  # each element: [(x,y), ...] along that edge's real geometry
                    if self._draw_region_edges:
                        neighbors = self._get_node_neighbors(self.ctx)
                        edge_geometries = self._get_edge_geometries(self.ctx)
                        region_set = set(in_region_ids)
                        seen = set()
                        for u in in_region_ids:
                            for v in neighbors.get(u, ()):
                                if v == u or v not in region_set:
                                    continue
                                edge_key = frozenset((u, v))
                                if edge_key in seen:
                                    continue
                                seen.add(edge_key)
                                edge_pairs.append(
                                    edge_geometries.get(edge_key) or [node_coords[u], node_coords[v]]
                                )
                else:
                    in_region_points = []
                    edge_pairs = []

                if visibility_polygon is not None or stationary_sensor_radius is not None:
                    try:
                        from gamms.VisualizationEngine import Artist

                        debug(f"Creating sensor region halos for flag at node {node_id}, position ({node.x}, {node.y}), radius {stationary_sensor_radius} ({len(in_region_points)} nodes, {len(edge_pairs)} edges)")

                        # layer=9: gamms draws the base graph (node dots +
                        # edges) at its own fixed layer 10; anything above
                        # that paints OVER the node markers instead of
                        # glowing around them. Layer 9 keeps this region
                        # fill/glow a backdrop the graph draws cleanly on top of.
                        circle_artist = Artist(self.ctx, self._render_sensor_circle, layer=9)
                        circle_artist.data.update({
                            "points": in_region_points,
                            "edges": edge_pairs,
                            "color": (144, 238, 144, int(SENSOR_ALPHA * 255)),  # Light green RGBA
                            "visibility_polygon": visibility_polygon,
                            "glow_points": glow_points,
                        })
                        self.ctx.visual.add_artist(f"{flag_type}_flag_{idx}_sensor_circle", circle_artist)
                        debug(f"Added sensor circle artist: {flag_type}_flag_{idx}_sensor_circle")

                    except (ImportError, AttributeError):
                        # Fallback to dictionary API
                        circle_data = {
                            "points": in_region_points,
                            "edges": edge_pairs,
                            "color": (144, 238, 144, int(SENSOR_ALPHA * 255)),  # Light green RGBA
                            "visibility_polygon": visibility_polygon,
                            "glow_points": glow_points,
                            "layer": 9,
                            "drawer": self._render_sensor_circle
                        }
                        self.ctx.visual.add_artist(f"{flag_type}_flag_{idx}_sensor_circle", circle_data)

                # Try Artist API first for the flag
                try:
                    from gamms.VisualizationEngine import Artist

                    artist = Artist(self.ctx, self._render_flag, layer=20)
                    artist.data.update({"x": node.x, "y": node.y, "color": color, "flag_width": size, "flag_height": int(size * 0.7), "pole_height": int(size * 1.5)})
                    self.ctx.visual.add_artist(f"{flag_type}_flag_{idx}", artist)

                except (ImportError, AttributeError):
                    # Fallback to dictionary API
                    data = {"x": node.x, "y": node.y, "color": color, "flag_width": size, "flag_height": int(size * 0.7), "pole_height": int(size * 1.5), "layer": 20, "drawer": self._render_flag}
                    self.ctx.visual.add_artist(f"{flag_type}_flag_{idx}", data)

            except Exception as e:
                warning(f"Failed to create {flag_type} flag {idx} at node {node_id}: {e}")

    def _render_flag(self, ctx: Any, data: Dict[str, Any]) -> None:
        """Render function for flag visualization."""
        x = data.get("x", 0)
        y = data.get("y", 0)
        flag_color = data.get("color", (255, 0, 0))
        pole_color = data.get("pole_color", (0, 0, 0))

        # All dimensions are in screen pixels
        flag_width  = data.get("flag_width",  15)
        flag_height = data.get("flag_height", 10)
        pole_height = data.get("pole_height", 20)
        pole_width  = 2

        # Convert node world position to screen coords once
        render_manager = ctx.visual._render_manager
        sx, sy = render_manager.world_to_screen(x, y)

        try:
            import pygame
            surface = ctx.visual._get_target_surface()

            # Pole: centered on sx, rising upward from the node (sy - pole_height to sy)
            pole_rect = pygame.Rect(sx - pole_width // 2, sy - pole_height, pole_width, pole_height)
            pygame.draw.rect(surface, pole_color, pole_rect)

            # Flag banner: attached to the top of the pole, extending right
            flag_rect = pygame.Rect(sx, sy - pole_height, flag_width, flag_height)
            pygame.draw.rect(surface, flag_color, flag_rect)

        except Exception as e:
            # Fallback: just draw a circle at the node position
            ctx.visual.render_circle(x, y, 2, flag_color)

    def mark_flag_captured(self, agent_name: str, agent_team: str, flag_node: int) -> None:
        """
        Mark a flag as captured by a specific attacker.
        Adds a colored label near the flag showing the attacker's abbreviated name.
        Multiple captures on the same flag stack vertically.

        Args:
            agent_name: Name of the capturing agent (e.g., "red_0")
            agent_team: Team of the capturing agent ("red" or "blue")
            flag_node: Node ID of the captured flag
        """
        # Look up the flag's world position
        pos = self.flag_positions.get(flag_node)
        if pos is None:
            warning(f"Cannot mark capture: flag at node {flag_node} not found in flag_positions")
            return

        # Track the capture
        if flag_node not in self.flag_captures:
            self.flag_captures[flag_node] = []
        capture_index = len(self.flag_captures[flag_node])
        self.flag_captures[flag_node].append((agent_name, agent_team))

        # Abbreviated name: "red_0" -> "r0", "blue_12" -> "b12"
        if "_" in agent_name:
            parts = agent_name.split("_")
            display_text = parts[0][0] + parts[1]
        else:
            display_text = agent_name

        label_color = (0, 0, 0)

        flag_x, flag_y = pos
        flag_size = self.sizes.get("flag_size", 15)
        # Stack labels below the flag pole; each subsequent capture shifts down
        label_y = flag_y + flag_size * 1.5 + self._label_y_offset + capture_index * self._label_spacing

        try:
            from gamms.VisualizationEngine import Artist

            artist_id = f"flag_capture_{flag_node}_{capture_index}"
            artist = Artist(self.ctx, self._render_capture_label, layer=30)
            artist.data.update({
                "x": flag_x,
                "y": label_y,
                "text": display_text,
                "color": label_color,
            })
            self.ctx.visual.add_artist(artist_id, artist)
            success(f"Marked flag {flag_node} as captured by {agent_name} ({agent_team})")

        except (ImportError, AttributeError) as e:
            warning(f"Failed to create capture label for flag {flag_node}: {e}")

    def _render_capture_label(self, ctx: Any, data: Dict[str, Any]) -> None:
        """Render a static label showing which attacker captured a flag."""
        x = data.get("x", 0)
        y = data.get("y", 0)
        text = data.get("text", "")
        color = data.get("color", (0, 0, 0))
        font_size = self.sizes.get("label_font_size")
        ctx.visual.render_text(text, x=x, y=y, color=color, font_size=font_size)

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
        edges are static for the game."""
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
        2-point segment if no linestring is set."""
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

    def _draw_edges(self, ctx: Any, edge_geometries, color: tuple) -> None:
        """Draw a polyline for each edge's [(x,y), ...] world-point list —
        follows the real road shape instead of a straight segment, so the
        highlighted region reads as a connected patch of the road network."""
        if not edge_geometries:
            return
        if len(color) == 4:
            r, g, b, a = color
        else:
            r, g, b, a = color[0], color[1], color[2], 100
        line_color = (r, g, b, min(255, int(a * 1.6)))

        render_manager = ctx.visual._render_manager
        surface = ctx.visual._get_target_surface()
        try:
            import pygame

            for world_points in edge_geometries:
                screen_points = [render_manager.world_to_screen(px, py) for px, py in world_points]
                if len(screen_points) < 2:
                    continue

                pad = SENSOR_EDGE_LINE_WIDTH + 2
                xs = [p[0] for p in screen_points]
                ys = [p[1] for p in screen_points]
                min_x, max_x = min(xs) - pad, max(xs) + pad
                min_y, max_y = min(ys) - pad, max(ys) + pad
                w, h = int(max_x - min_x), int(max_y - min_y)
                if w <= 0 or h <= 0:
                    continue

                temp_surface = pygame.Surface((w, h), pygame.SRCALPHA)
                temp_surface.fill((0, 0, 0, 0))
                local_points = [(px - min_x, py - min_y) for px, py in screen_points]
                pygame.draw.lines(temp_surface, line_color, False, local_points, SENSOR_EDGE_LINE_WIDTH)
                surface.blit(temp_surface, (int(min_x), int(min_y)))
        except Exception:
            pass  # Silent fail

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

    def _render_sensor_circle(self, ctx: Any, data: Dict[str, Any]) -> None:
        """Highlight a static (tower) sensing region — as a true
        building-occluded visibility polygon if this tower carries a
        line_of_sight model (see visibility_polygon), else a small halo
        circle per in-radius node instead of one big disk. Both are
        precomputed once at creation time (see _create_team_flags), since the
        tower's origin never moves."""
        points = data.get("points", [])
        edges = data.get("edges", [])
        color = data.get("color", (144, 238, 144, 100))  # RGBA
        visibility_polygon = data.get("visibility_polygon")

        if visibility_polygon is not None:
            self._draw_visibility_polygon(ctx, visibility_polygon, color)
            self._draw_node_glows(ctx, data.get("glow_points", []), color)
            return

        if not points:
            return

        self._draw_edges(ctx, edges, color)

        if isinstance(color, tuple) and len(color) == 4:
            fill_r, fill_g, fill_b, alpha_value = color
        else:
            fill_r, fill_g, fill_b = color if len(color) >= 3 else (144, 238, 144)
            alpha_value = 100

        render_manager = ctx.visual._render_manager
        screen_halo_radius = render_manager.world_to_screen_scale(SENSOR_NODE_HALO_RADIUS)
        if screen_halo_radius < 1:
            return

        surface = ctx.visual._get_target_surface()
        try:
            import pygame

            surf_size = int(screen_halo_radius * 2 + 4)
            center = int(screen_halo_radius + 2)
            temp_surface = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)

            for x, y in points:
                screen_x, screen_y = render_manager.world_to_screen(x, y)
                temp_surface.fill((0, 0, 0, 0))
                pygame.draw.circle(temp_surface, (fill_r, fill_g, fill_b, alpha_value), (center, center), int(screen_halo_radius))
                pygame.draw.circle(
                    temp_surface,
                    (SENSOR_EDGE_COLOR[0], SENSOR_EDGE_COLOR[1], SENSOR_EDGE_COLOR[2], SENSOR_EDGE_ALPHA),
                    (center, center),
                    int(screen_halo_radius),
                    SENSOR_EDGE_WIDTH,
                )
                blit_pos = (int(screen_x - screen_halo_radius - 2), int(screen_y - screen_halo_radius - 2))
                surface.blit(temp_surface, blit_pos)
        except Exception:
            pass  # Silent fail
