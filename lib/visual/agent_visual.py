from typing import Any, Dict, Optional, Set
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
class AgentVisual:
    """
    Agent Visual Handler for managing all agent-related visualization.
    Handles agent visuals, labels, and sensor circles.
    """

    def __init__(
        self,
        ctx: Any,
        config: Dict[str, Any],
        agent_properties: Dict[str, Dict[str, Any]],
        team_properties: Dict[str, Dict[str, Any]],
        graph: Optional[nx.Graph] = None,
        vis: bool = True,
    ):
        """
        Initialize the Agent Visual Handler.

        Args:
            ctx: Game context object with visualization capabilities
            config: Complete configuration dictionary
            agent_properties: Dictionary of agent properties extracted by VisEngine
            team_properties: Dictionary of team properties extracted by VisEngine
            graph: The game's networkx graph, if available — used to look up
                any line_of_sight model's real range, so a team's sensor
                circle draws its true building-occluded visibility polygon
                instead of a euclidean-radius halo. None disables this.
            vis: If False (headless/no-vis run), skip building the visibility
                polygon index entirely — it's only ever drawn by a render
                callback that never fires without a display, so resolving it
                would just fetch/index buildings for nothing.
        """
        debug("Initializing AgentVisual")
        self.ctx = ctx
        self.config = config
        self.vis_config = config.get("visualization", {})
        
        # Color and size settings
        self.colors = self.vis_config.get("colors", {})
        self.sizes = self.vis_config.get("sizes", {})

        # Shared label settings for moving, dead, and flag-capture labels
        self._label_spacing = self.vis_config.get("label_spacing", 15)
        self._label_y_offset = self.vis_config.get("label_y_offset", -self._label_spacing)
        
        # Agent properties reference
        self.agent_properties = agent_properties
        self.team_properties = team_properties
        
        # Agent name label tracking
        self.agent_labels_created: Set[str] = set()

        # This will store the actual artist objects, mapped by agent name
        self.agent_label_artists: Dict[str, Any] = {}

        # Track dead label positions for anti-overlap: list of (x, y) tuples
        self._dead_label_positions: list = []

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

        # {team_name: max_range} for any line_of_sight model carried by that
        # team's agents (carrier=agent) — see resolve_line_of_sight_ranges.
        # Building index built once and queried per-origin at render time.
        # {team_name: table}, the real per-node table (cheap — sensor_engine
        # already built/cached it for gameplay) for marking exactly which
        # nodes the actual sensor considers visible, alongside the polygon.
        # All stay empty when vis=False so a headless run never fetches
        # buildings/tables or indexes them.
        self._dynamic_los_ranges: Dict[str, float] = {}
        self._dynamic_los_tables: Dict[str, Any] = {}
        self._vis_polygon_index: Optional[VisibilityPolygonIndex] = None
        # {(agent_name, node_id): [(x,y), ...]} — the visibility polygon only
        # changes when the agent's *logical* node changes, not every rendered
        # frame (gamms interpolates smoothly between two nodes), so caching
        # per node avoids recomputing it dozens of times per tick.
        self._fan_cache: Dict[Any, list] = {}
        if vis and graph is not None:
            self._dynamic_los_ranges, _ = resolve_line_of_sight_ranges(config)
            if self._dynamic_los_ranges:
                polygons = resolve_building_polygons(config, graph)
                self._vis_polygon_index = VisibilityPolygonIndex(polygons)
                self._dynamic_los_tables, _ = resolve_line_of_sight_tables(config, graph)

        success("AgentVisual initialized")

    def setup_agent_visuals(self, agents_config: Dict[str, Any]) -> None:
        """
        Configure visual settings for all agents.

        Args:
            agents_config: Complete agents configuration from config file
                          Contains red_global, blue_global, red_config, blue_config, etc.
        """
        try:
            debug("Setting up agent visuals from configuration")
            
            # Default values
            default_size = self.sizes.get("global_agent_size", 10)

            # Import default color
            try:
                from gamms.VisualizationEngine import Color
                default_color = Color.White
            except ImportError:
                default_color = "white"

            agents_configured = 0

            # Process each team configuration
            for key, config_data in agents_config.items():
                if key.endswith("_config"):  # red_config, blue_config, etc.
                    team_name = key.replace("_config", "")  # extract "red" from "red_config"
                    debug(f"Processing {key} for team '{team_name}'")

                    # Get global config for this team
                    global_key = f"{team_name}_global"
                    global_config = agents_config.get(global_key, {})

                    # Get team color from global config; convert list to tuple if loaded from YAML
                    raw_color = global_config.get("color", default_color)
                    team_color = tuple(raw_color) if isinstance(raw_color, list) else raw_color
                    global_size = global_config.get("size", default_size)
                    
                    debug(f"Team '{team_name}' using color: {team_color}, size: {global_size}")

                    # Configure each individual agent
                    for agent_name, agent_config in config_data.items():
                        try:
                            # Agent-specific settings override global settings
                            color = agent_config.get("color", team_color)
                            size = agent_config.get("size", global_size)

                            # Apply visual settings
                            self.ctx.visual.set_agent_visual(agent_name, color=color, size=size)
                            debug(f"Configured visual for agent '{agent_name}': color={color}, size={size}")
                            agents_configured += 1

                        except Exception as e:
                            warning(f"Failed to configure visual for agent '{agent_name}': {e}")

            success(f"Configured visuals for {agents_configured} agents")

        except Exception as e:
            error(f"Failed to setup agent visuals: {e}")
            raise

    def create_agent_labels(self, agents_config: Dict[str, Any]) -> None:
        """
        Create name labels for all agents.

        Args:
            agents_config: Complete agents configuration from config file
        """
        try:
            debug("Creating agent name labels")
            
            # Label settings
            label_offset_y = self._label_y_offset
            
            # Default label color
            try:
                from gamms.VisualizationEngine import Color
                default_label_color = Color.Black
            except ImportError:
                default_label_color = (0, 0, 0)

            labels_created = 0

            # Process each team configuration
            for key, config_data in agents_config.items():
                if key.endswith("_config"):
                    team_name = key.replace("_config", "")
                    debug(f"Creating labels for {team_name} team agents")
                    
                    # Create label for each agent
                    for agent_name, agent_config in config_data.items():
                        try:
                            # Agent-specific label color or default
                            label_color = agent_config.get("label_color", default_label_color)
                            
                            self._create_agent_label(agent_name, label_color, label_offset_y)
                            self.agent_labels_created.add(agent_name)
                            labels_created += 1
                            debug(f"Created label for '{agent_name}'")
                                
                        except Exception as e:
                            warning(f"Failed to create label for agent '{agent_name}': {e}")

            success(f"Created name labels for {labels_created} agents")

        except Exception as e:
            error(f"Failed to create agent labels: {e}")

    def create_agent_sensor_circles(self) -> None:
        """
        Automatically create sensor circles for all agents that have a sensing_radius defined.
        Red team agents get transparent red circles, blue team agents get transparent blue circles.
        """
        try:
            debug("Creating agent sensor circles")
            
            circles_created = 0
            
            # Team colors for sensor circles (RGB)
            team_sensor_colors = {
                "red": (255, 0, 0),    # Red
                "blue": (0, 0, 255)    # Blue
            }
            
            # Iterate through all agents with properties
            for agent_name, properties in self.agent_properties.items():
                sensing_radius = properties.get("sensing_radius")
                team = properties.get("team")
                
                # Only create circle if sensing_radius exists
                if sensing_radius is not None and sensing_radius > 0:
                    # Get team color
                    sensor_color_rgb = team_sensor_colors.get(team, (128, 128, 128))  # Default gray if team unknown
                    
                    # Create RGBA color with transparency
                    sensor_color_rgba = (sensor_color_rgb[0], sensor_color_rgb[1], sensor_color_rgb[2], int(SENSOR_ALPHA * 255))
                    
                    # Create the sensor circle
                    self.create_agent_sensor_circle(agent_name, sensing_radius, sensor_color_rgba, team)
                    circles_created += 1
                    debug(f"Created {team} sensor circle for '{agent_name}' with radius {sensing_radius}")
            
            success(f"Created sensor circles for {circles_created} agents")
            
        except Exception as e:
            error(f"Failed to create agent sensor circles: {e}")

    def _create_agent_label(self, agent_name: str, color: Any, offset_y: float) -> None:
        """Create a text label for an agent that follows it automatically."""
        try:
            label_id = f"{agent_name}_label"
            
            from gamms.VisualizationEngine import Artist
            from gamms.typing import ArtistType
            
            artist = Artist(self.ctx, self._render_agent_label, layer=30)
            artist.set_artist_type(ArtistType.DYNAMIC)
            
            artist.data.update({
                "agent_name": agent_name,
                "offset_y": offset_y,
                "color": color,
                "_alpha": 1.0
            })
            
            returned_artist = self.ctx.visual.add_artist(label_id, artist)
            self.agent_label_artists[agent_name] = returned_artist
            
        except Exception as e:
            warning(f"Failed to create label artist for agent '{agent_name}': {e}")

    def _render_agent_label(self, ctx: Any, data: Dict[str, Any]) -> None:
        """Render label by following the agent's animated position.
        Automatically separates overlapping labels on the y-axis when
        multiple agents share the same graph node.
        """
        agent_name = data.get("agent_name", "")
        offset_y = data.get("offset_y", -50)
        color = data.get("color", (0, 0, 0))

        try:
            # Get the agent
            agent = ctx.agent.get_agent(agent_name)
            current_node_id = agent.current_node_id

            # Follow the agent dot's already-computed position (includes edge linestring)
            agent_artist = ctx.visual._dynamic_artists.get(agent_name)
            if (agent_artist is not None
                    and 'agent_data' in agent_artist.data
                    and agent_artist.data['agent_data'].current_position is not None):
                x, y = agent_artist.data['agent_data'].current_position
            else:
                node = ctx.graph.graph.get_node(current_node_id)
                x, y = node.x, node.y

            # Find co-located agents at the same node and compute y-offset
            colocated = sorted(
                a.name for a in ctx.agent.create_iter()
                if a.current_node_id == current_node_id and a.name in self.agent_labels_created
            )
            if len(colocated) > 1:
                idx = colocated.index(agent_name) if agent_name in colocated else 0
                label_spacing = self._label_spacing
                # Center the group around the base offset
                group_offset = (idx - (len(colocated) - 1) / 2) * label_spacing
                y += offset_y + group_offset
            else:
                y += offset_y

            # Simplify text
            text = agent_name.split('_')[0][0] + agent_name.split('_')[1] if '_' in agent_name else agent_name

            font_size = self.sizes.get("label_font_size")
            ctx.visual.render_text(text, x=x, y=y, color=color, font_size=font_size)
        except Exception:
            pass  # Silent fail if agent doesn't exist

    def create_agent_sensor_circle(self, agent_name: str, radius: float, color: tuple = (0, 255, 255, 100), team: Optional[str] = None) -> None:
        """
        Create a transparent circle that follows an agent.

        Args:
            agent_name: Name of the agent to follow
            radius: Circle radius in world coordinates
            color: RGBA tuple (R, G, B, Alpha)
            team: Agent's team — if it has a line_of_sight model (see
                self._dynamic_los_ranges), a true building-occluded visibility
                polygon is drawn instead of this euclidean radius.
        """
        try:
            from gamms.VisualizationEngine import Artist
            from gamms.typing import ArtistType

            circle_id = f"{agent_name}_sensor_circle"

            # layer=9: gamms draws the base graph (node dots + edges) at its
            # own fixed layer 10 (gamms/VisualizationEngine/pygame_engine.py);
            # anything above that paints OVER the node markers, tinting them
            # instead of glowing around them. Layer 9 keeps this region
            # fill/glow a backdrop the graph then draws cleanly on top of.
            artist = Artist(self.ctx, self._render_agent_circle, layer=9)
            artist.set_artist_type(ArtistType.DYNAMIC)

            artist.data.update({
                "agent_name": agent_name,
                "radius": radius,
                "color": color,
                "team": team,
                "_alpha": 1.0  # Required for animation
            })
            
            self.ctx.visual.add_artist(circle_id, artist)
            debug(f"Created animated circle for agent '{agent_name}'")
            
        except Exception as e:
            warning(f"Failed to create circle for agent '{agent_name}': {e}")

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

    def _draw_region_edges_between(self, ctx: Any, in_region_ids, node_coords: Dict[int, Any], color: tuple) -> None:
        """Draw each in-region edge along its real road shape (not a straight
        segment) — makes the highlighted region read as a connected patch of
        the road network instead of disconnected dots."""
        if len(in_region_ids) < 2:
            return
        neighbors = self._get_node_neighbors(ctx)
        edge_geometries = self._get_edge_geometries(ctx)
        region_set = set(in_region_ids)
        if len(color) == 4:
            r, g, b, a = color
        else:
            r, g, b, a = color[0], color[1], color[2], 100
        line_color = (r, g, b, min(255, int(a * 1.6)))

        render_manager = ctx.visual._render_manager
        surface = ctx.visual._get_target_surface()
        try:
            import pygame

            seen = set()
            for u in in_region_ids:
                for v in neighbors.get(u, ()):
                    if v == u or v not in region_set:
                        continue
                    edge_key = frozenset((u, v))
                    if edge_key in seen:
                        continue
                    seen.add(edge_key)

                    world_points = edge_geometries.get(edge_key) or [node_coords[u], node_coords[v]]
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

    def _draw_node_halos(self, ctx: Any, points, color: tuple) -> None:
        """Draw one small semi-transparent circle per (x, y) world point —
        the actual sensing-region node set, not a single geometric disk."""
        if not points:
            return
        if len(color) == 4:
            fill_r, fill_g, fill_b, alpha_value = color
        else:
            fill_r, fill_g, fill_b, alpha_value = color[0], color[1], color[2], 100

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

    def _render_agent_circle(self, ctx: Any, data: Dict[str, Any]) -> None:
        """Highlight the agent's sensing region — as a true building-occluded
        visibility polygon if a line_of_sight model backs its sensor (see
        self._dynamic_los_ranges), else a small halo circle per in-radius
        node (recomputed live from its current position), instead of one big
        disk — reflects the graph's actual node set, not a euclidean
        approximation."""
        agent_name = data.get("agent_name", "")
        radius = data.get("radius", 50)
        color = data.get("color", (0, 255, 255, 100))
        team = data.get("team")

        try:
            # Get the agent
            agent = ctx.agent.get_agent(agent_name)

            # Follow the agent dot's already-computed position (includes edge linestring)
            agent_artist = ctx.visual._dynamic_artists.get(agent_name)
            if (agent_artist is not None
                    and 'agent_data' in agent_artist.data
                    and agent_artist.data['agent_data'].current_position is not None):
                x, y = agent_artist.data['agent_data'].current_position
            else:
                node = ctx.graph.graph.get_node(agent.current_node_id)
                x, y = node.x, node.y

            los_range = self._dynamic_los_ranges.get(team) if team else None
            if los_range is not None and self._vis_polygon_index is not None:
                origin_node = agent.current_node_id
                # Cached per (agent, node): the polygon is only meaningful at
                # the granularity the underlying sensor updates it (once per
                # tick, keyed by node) — recomputing it for every interpolated
                # animation frame between two ticks would be pure waste.
                cache_key = (agent_name, origin_node)
                polygon = self._fan_cache.get(cache_key)
                if polygon is None:
                    node_coords = self._get_node_coords(ctx)
                    origin_xy = node_coords.get(origin_node, (x, y))
                    polygon = self._vis_polygon_index.polygon_at(origin_xy, los_range)
                    self._fan_cache[cache_key] = polygon
                self._draw_visibility_polygon(ctx, polygon, color)

                table = self._dynamic_los_tables.get(team)
                if table is not None:
                    node_coords = self._get_node_coords(ctx)
                    region = table.get(origin_node, frozenset((origin_node,)))
                    glow_points = [node_coords[nid] for nid in region if nid in node_coords]
                    self._draw_node_glows(ctx, glow_points, color)
                return

            node_coords = self._get_node_coords(ctx)
            r2 = radius * radius
            in_region_ids = [
                nid for nid, (nx_, ny_) in node_coords.items()
                if (nx_ - x) ** 2 + (ny_ - y) ** 2 <= r2
            ]

            if self._draw_region_edges:
                self._draw_region_edges_between(ctx, in_region_ids, node_coords, color)

            points = [node_coords[nid] for nid in in_region_ids]
            self._draw_node_halos(ctx, points, color)

        except Exception:
            pass  # Silent fail if agent doesn't exist

    def mark_agent_tagged(self, agent_name: str) -> None:
        """
        Mark agent as tagged/dead with static visuals.
        Must be called BEFORE removing agent from ctx!
        """
        # 1. Capture final position while agent still exists in ctx
        try:
            agent = self.ctx.agent.get_agent(agent_name)
            node = self.ctx.graph.graph.get_node(agent.current_node_id)
            final_x, final_y = node.x, node.y
            debug(f"Agent '{agent_name}' tagged at position: ({final_x}, {final_y})")
        except Exception as e:
            warning(f"Could not get position for '{agent_name}': {e}")
            return
        
        # 2. Get agent properties (size, team)
        properties = self.agent_properties.get(agent_name, {})
        team = properties.get("team", "red")
        agent_size = properties.get("size", 10)
        
        # Desaturated team colors
        if team == "red":
            dead_color = (180, 100, 100)  # Grayish red
        else:
            dead_color = (100, 100, 180)  # Grayish blue
        
        # 3. Create static dead agent body (square at death position)
        self._create_static_dead_body(agent_name, final_x, final_y, agent_size, dead_color)
        
        # 4. Convert label from animated to static (at death position)
        self._convert_label_to_static(agent_name, final_x, final_y)
        
        # 5. Add death cross at static position
        self._create_static_death_cross(agent_name, final_x, final_y)
        
        # 6. Remove sensor circle (as requested)
        self.remove_agent_sensor_circle(agent_name)
        
        # 7. Remove the original agent visual artist (CRITICAL - prevents "Agent not found" errors)
        try:
            self.ctx.visual.remove_artist(agent_name)
            debug(f"Removed original agent visual artist for '{agent_name}'")
        except Exception as e:
            warning(f"Failed to remove original agent visual: {e}")
        
        success(f"Agent '{agent_name}' marked as tagged with static visuals")

    def _create_static_dead_body(self, agent_name: str, x: float, y: float, size: float, color: tuple) -> None:
        """
        Create a static dead body square at death position.

        Args:
            agent_name: Agent name (for artist ID)
            x, y: Position to draw the dead body
            size: Size of the agent
            color: RGB color tuple for the dead body
        """
        try:
            from gamms.VisualizationEngine import Artist

            body_id = f"{agent_name}_dead_body"

            # Create static artist (NO ArtistType.AGENT - won't follow agent)
            artist = Artist(self.ctx, self._render_static_dead_body, layer=20)  # Below label
            artist.data.update({
                "x": x,
                "y": y,
                "size": size,
                "color": color
            })

            self.ctx.visual.add_artist(body_id, artist)
            debug(f"Created dead body for '{agent_name}' at ({x}, {y})")

        except Exception as e:
            warning(f"Failed to create dead body for '{agent_name}': {e}")

    def _render_static_dead_body(self, ctx: Any, data: Dict[str, Any]) -> None:
        """
        Render a static dead agent body as a grayed-out circle.
        Does NOT follow agent movement.
        """
        x = data.get("x", 0)
        y = data.get("y", 0)
        size = data.get("size", 10)
        color = data.get("color", (128, 128, 128))

        try:
            # Get render manager for coordinate transformation
            render_manager = ctx.visual._render_manager

            # Transform to screen coordinates
            screen_x, screen_y = render_manager.world_to_screen(x, y)
            screen_radius = render_manager.world_to_screen_scale(size)

            if screen_radius < 1:
                return

            surface = ctx.visual._get_target_surface()

            import pygame

            # Draw filled circle (dead body)
            pygame.draw.circle(
                surface,
                color,
                (int(screen_x), int(screen_y)),
                int(screen_radius)
            )

            # Draw darker outline for better visibility
            outline_color = tuple(max(0, c - 60) for c in color)
            pygame.draw.circle(
                surface,
                outline_color,
                (int(screen_x), int(screen_y)),
                int(screen_radius),
                2  # outline width
            )

        except Exception:
            pass  # Silent fail

    def _convert_label_to_static(self, agent_name: str, x: float, y: float) -> None:
        """
        Replace animated label (that follows agent) with static label at death position.
        
        Args:
            agent_name: Agent whose label to convert
            x, y: Final death position
        """
        label_id = f"{agent_name}_label"
        
        # Get label properties from old artist before removing
        label_color = (0, 0, 0)  # Default black

        if agent_name in self.agent_label_artists:
            old_artist = self.agent_label_artists[agent_name]
            label_color = old_artist.data.get("color", (0, 0, 0))
        
        # Remove old animated label
        try:
            self.ctx.visual.remove_artist(label_id)
            if agent_name in self.agent_label_artists:
                del self.agent_label_artists[agent_name]
            self.agent_labels_created.discard(agent_name)
            debug(f"Removed animated label for '{agent_name}'")
        except Exception as e:
            warning(f"Failed to remove old label: {e}")
        
        # Create new STATIC label at fixed position
        try:
            from gamms.VisualizationEngine import Artist
            
            # Create static artist (NO ArtistType.AGENT - won't follow agent)
            artist = Artist(self.ctx, self._render_static_label, layer=30)
            
            # Simplify agent name for display
            display_text = agent_name.split('_')[0][0] + agent_name.split('_')[1] if '_' in agent_name else agent_name
            
            dead_label_y_offset = self._label_y_offset
            dead_label_spacing = self._label_spacing
            y += dead_label_y_offset

            # Anti-overlap: offset y if another dead label is at a nearby position
            for px, py in self._dead_label_positions:
                if abs(px - x) < 5 and abs(py - y) < dead_label_spacing:
                    y = py - dead_label_spacing
            self._dead_label_positions.append((x, y))

            artist.data.update({
                "x": x,
                "y": y,
                "text": display_text,
                "color": label_color,
            })

            self.ctx.visual.add_artist(label_id, artist)
            debug(f"Created static label for '{agent_name}' at ({x}, {y})")
            
        except Exception as e:
            warning(f"Failed to create static label for '{agent_name}': {e}")

    def _render_static_label(self, ctx: Any, data: Dict[str, Any]) -> None:
        """
        Render a static text label at fixed position.
        Does NOT follow agent movement.
        """
        x = data.get("x", 0)
        y = data.get("y", 0)
        text = data.get("text", "")
        color = data.get("color", (0, 0, 0))

        font_size = self.sizes.get("label_font_size")
        ctx.visual.render_text(text, x=x, y=y, color=color, font_size=font_size)

    def _create_static_death_cross(self, agent_name: str, x: float, y: float) -> None:
        """
        Create a static X mark at death position.

        Args:
            agent_name: Agent name (for artist ID)
            x, y: Position to draw the cross
        """
        try:
            from gamms.VisualizationEngine import Artist

            cross_id = f"{agent_name}_death_cross"

            # Get agent size for proportional cross
            properties = self.agent_properties.get(agent_name, {})
            agent_size = properties.get("size", 10)
            cross_ratio = self.vis_config.get("death_cross_size_ratio", 1.0)
            cross_size = agent_size * cross_ratio

            # Create static artist (NO ArtistType.AGENT)
            artist = Artist(self.ctx, self._render_static_cross, layer=35)  # Above label
            artist.data.update({
                "x": x,
                "y": y,
                "size": cross_size,
                "color": (0, 0, 0),  # Black cross
                "width": 2
            })

            self.ctx.visual.add_artist(cross_id, artist)
            debug(f"Created death cross for '{agent_name}' at ({x}, {y})")

        except Exception as e:
            warning(f"Failed to create death cross for '{agent_name}': {e}")

    def _render_static_cross(self, ctx: Any, data: Dict[str, Any]) -> None:
        """
        Render an X mark at fixed position.
        Does NOT follow agent movement.
        """
        x = data.get("x", 0)
        y = data.get("y", 0)
        size = data.get("size", 10)
        color = data.get("color", (0, 0, 0))
        width = data.get("width", 2)
        
        # Draw X as two diagonal lines
        ctx.visual.render_line(x - size, y - size, x + size, y + size, color, width=width)
        ctx.visual.render_line(x - size, y + size, x + size, y - size, color, width=width)

    def remove_agent_label(self, agent_name: str) -> None:
        """
        Remove the name label for a specific agent.

        Args:
            agent_name: Name of the agent whose label to remove
        """
        try:
            label_id = f"{agent_name}_label"
            
            if hasattr(self.ctx.visual, "remove_artist"):
                try:
                    self.ctx.visual.remove_artist(label_id)
                    debug(f"Removed label for agent '{agent_name}'")
                except Exception:
                    pass
            
            # Also remove it from our local reference dictionary
            if agent_name in self.agent_label_artists:
                del self.agent_label_artists[agent_name]
                    
            self.agent_labels_created.discard(agent_name)
            
        except Exception as e:
            warning(f"Failed to remove label for agent '{agent_name}': {e}")

    def remove_agent_visual(self, agent_name: str) -> None:
        """
        Remove all visual elements associated with an agent.

        Args:
            agent_name: Name of the agent to remove visuals for
        """
        try:
            debug(f"Removing visuals for agent {agent_name}")
            
            # Remove agent name label first
            self.remove_agent_label(agent_name)
            self.remove_agent_sensor_circle(agent_name)
            
            # Remove main agent artist
            if hasattr(self.ctx.visual, "remove_artist"):
                try:
                    self.ctx.visual.remove_artist(agent_name)
                except Exception:
                    pass

                # Remove auxiliary artists (prefixed with agent name)
                if hasattr(self.ctx.visual, "_render_manager"):
                    rm = self.ctx.visual._render_manager
                    if hasattr(rm, "_artists"):
                        for artist_id in list(rm._artists.keys()):
                            if artist_id.startswith(f"{agent_name}_"):
                                try:
                                    self.ctx.visual.remove_artist(artist_id)
                                except Exception:
                                    pass

            debug(f"Removed visual elements for agent {agent_name}")

        except Exception as e:
            warning(f"Failed to remove visuals for agent {agent_name}: {e}")

    def remove_agent_sensor_circle(self, agent_name: str) -> None:
        """Remove the sensor circle for an agent."""
        circle_id = f"{agent_name}_sensor_circle"
        if hasattr(self.ctx.visual, "remove_artist"):
            try:
                self.ctx.visual.remove_artist(circle_id)
            except Exception:
                pass
