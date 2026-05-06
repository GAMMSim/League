from typing import Any, Dict, List, Optional, Tuple
from typeguard import typechecked

try:
    from lib.core.console import *
    from lib.visual.agent_visual import AgentVisual
    from lib.visual.flag_visual import FlagVisual
    from lib.visual.map_overlay_visual import MapOverlayVisual
except ImportError:
    from ..core.console import *
    from ..visual.agent_visual import AgentVisual
    from ..visual.flag_visual import FlagVisual
    from ..visual.map_overlay_visual import MapOverlayVisual


@typechecked
class VisEngine:
    """
    Visualization Engine for managing all visual aspects of the game.
    Handles graph visualization, agent visuals, flags, and game settings.
    Delegates specialized rendering to AgentVisual and FlagVisual.
    """

    def __init__(self, ctx: Any, config: Dict[str, Any]):
        """
        Initialize the Visualization Engine.

        Args:
            ctx: Game context object with visualization capabilities
            config: Complete configuration dictionary
        """
        debug("Initializing VisEngine")
        self.ctx = ctx
        self.config = config
        self.vis_config = config.get("visualization", {})
        self.env_config = config.get("environment", {})

        # Store visualization settings
        self.draw_node_id = self.vis_config.get("draw_node_id", False)
        self.game_speed = self.vis_config.get("game_speed", 1)
        self.window_size = self.vis_config.get("window_size", [1200, 800])

        # Color and size settings
        self.colors = self.vis_config.get("colors", {})
        self.sizes = self.vis_config.get("sizes", {})

        # Agent properties storage (sensing_radius, capture_radius, tagging_radius, etc.)
        self.agent_properties: Dict[str, Dict[str, Any]] = {}
        self.team_properties: Dict[str, Dict[str, Any]] = {}
        
        # Extract agent properties for use by specialized visual classes
        self._extract_agent_properties()

        # Initialize specialized visual handlers
        self.agent_visual = AgentVisual(ctx, config, self.agent_properties, self.team_properties)
        self.flag_visual = FlagVisual(ctx, config)

        # HUD state
        hud_config = self.vis_config.get("hud", {})
        self._hud_x = hud_config.get("x", 300)
        self._hud_y = hud_config.get("y", 10)
        self._hud_line_h = hud_config.get("line_height", 18)
        self._hud_font_size = hud_config.get("font_size", 14)
        self._hud_max_events = hud_config.get("max_events", 8)
        self._hud_events: List[Tuple[str, tuple]] = []  # (text, rgb_color)
        self._hud_payoff = {"total": 0.0, "tag": 0.0, "capture": 0.0, "discover": 0.0}
        self._hud_artist_created = False

        # Initialize visualization
        self._setup_base_visualization()
        success("VisEngine initialized")

    def _extract_agent_properties(self) -> None:
        """Extract and store agent properties (sensing_radius, capture_radius, etc.) for easy access."""
        try:
            debug("Extracting agent properties")
            agents_config = self.config.get("agents", {})
            
            # Extract global properties for each team (red_global, blue_global)
            for key, config_data in agents_config.items():
                if key.endswith("_global"):
                    team_name = key.replace("_global", "")
                    self.team_properties[team_name] = {
                        "sensing_radius": config_data.get("sensing_radius"),
                        "capture_radius": config_data.get("capture_radius"),
                        "tagging_radius": config_data.get("tagging_radius"),
                        "speed": config_data.get("speed"),
                        "sensors": config_data.get("sensors", []),
                        "size": config_data.get("size"),
                        "color": config_data.get("color")
                    }
                    debug(f"Stored {team_name} team properties: {self.team_properties[team_name]}")
            
            # Extract individual agent properties
            for key, config_data in agents_config.items():
                if key.endswith("_config"):
                    team_name = key.replace("_config", "")
                    team_defaults = self.team_properties.get(team_name, {})
                    
                    for agent_name, agent_config in config_data.items():
                        self.agent_properties[agent_name] = {
                            "team": team_name,
                            "sensing_radius": agent_config.get("sensing_radius", team_defaults.get("sensing_radius")),
                            "capture_radius": agent_config.get("capture_radius", team_defaults.get("capture_radius")),
                            "tagging_radius": agent_config.get("tagging_radius", team_defaults.get("tagging_radius")),
                            "speed": agent_config.get("speed", team_defaults.get("speed")),
                            "sensors": agent_config.get("sensors", team_defaults.get("sensors", [])),
                            "size": agent_config.get("size", team_defaults.get("size")),
                            "color": agent_config.get("color", team_defaults.get("color"))
                        }
                        debug(f"Stored properties for agent '{agent_name}': {self.agent_properties[agent_name]}")
            
            success(f"Extracted properties for {len(self.agent_properties)} agents and {len(self.team_properties)} teams")
            
        except Exception as e:
            error(f"Failed to extract agent properties: {e}")

    def _setup_base_visualization(self) -> None:
        """Set up the base graph visualization with default colors and settings."""
        try:
            debug("Setting up base visualization")
            # Import color constants
            # Set base graph visualization
            node_size = self.sizes.get("node_size", 6)
            edge_width = self.sizes.get("edge_width", 1)
            node_color = self.colors.get("node_color", (0, 0, 0))
            edge_color = self.colors.get("edge_color", (169, 169, 169))
            self.ctx.visual.set_graph_visual(draw_id=self.draw_node_id, node_color=node_color, edge_color=edge_color, node_size=node_size, edge_width=edge_width)

            # Set game speed
            if hasattr(self.ctx.visual, "_sim_time_constant"):
                self.ctx.visual._sim_time_constant = self.game_speed

            success("Base visualization configured")

        except Exception as e:
            error(f"Failed to setup base visualization: {e}")

    def setup_agent_visuals(self, agents_config: Dict[str, Any]) -> None:
        """
        Configure visual settings for all agents.

        Args:
            agents_config: Complete agents configuration from config file
                          Contains red_global, blue_global, red_config, blue_config, etc.
        """
        self.agent_visual.setup_agent_visuals(agents_config)

    def create_flags(self, flags_config: Dict[str, Any]) -> None:
        """
        Create flag visualizations based on the game rule version.

        Args:
            flags_config: Dictionary from the config file's 'flags' section.
        """
        self.flag_visual.create_flags(flags_config)

    def create_agent_labels(self, agents_config: Dict[str, Any]) -> None:
        """
        Create name labels for all agents.

        Args:
            agents_config: Complete agents configuration from config file
        """
        self.agent_visual.create_agent_labels(agents_config)

    def create_agent_sensor_circles(self) -> None:
        """
        Automatically create sensor circles for all agents that have a sensing_radius defined.
        Red team agents get transparent red circles, blue team agents get transparent blue circles.
        """
        self.agent_visual.create_agent_sensor_circles()

    def create_agent_sensor_circle(self, agent_name: str, radius: float, color: tuple = (0, 255, 255, 100)) -> None:
        """
        Create a transparent circle that follows an agent.
        
        Args:
            agent_name: Name of the agent to follow
            radius: Circle radius in world coordinates
            color: RGBA tuple (R, G, B, Alpha)
        """
        self.agent_visual.create_agent_sensor_circle(agent_name, radius, color)

    def mark_flag_captured(self, agent_name: str, agent_team: str, flag_node: int) -> None:
        """
        Mark a flag as captured by a specific attacker with a visual label.

        Args:
            agent_name: Name of the capturing agent
            agent_team: Team of the capturing agent
            flag_node: Node ID of the captured flag
        """
        self.flag_visual.mark_flag_captured(agent_name, agent_team, flag_node)

    def mark_agent_tagged(self, agent_name: str) -> None:
        """
        Mark an agent as tagged/dead with grayed visuals and death cross.
        Must be called BEFORE removing agent from ctx!
        
        Args:
            agent_name: Name of the agent being tagged
        """
        self.agent_visual.mark_agent_tagged(agent_name)

    def remove_agent_label(self, agent_name: str) -> None:
        """
        Remove the name label for a specific agent.

        Args:
            agent_name: Name of the agent whose label to remove
        """
        self.agent_visual.remove_agent_label(agent_name)

    def remove_agent_visual(self, agent_name: str) -> None:
        """
        Remove all visual elements associated with an agent.

        Args:
            agent_name: Name of the agent to remove visuals for
        """
        self.agent_visual.remove_agent_visual(agent_name)

    def remove_agent_sensor_circle(self, agent_name: str) -> None:
        """Remove the sensor circle for an agent."""
        self.agent_visual.remove_agent_sensor_circle(agent_name)

    def update_display(self) -> None:
        """Update the visual display (handle pygame events, render frame)."""
        try:
            debug("Updating display")
            self.ctx.visual.simulate()
        except Exception as e:
            warning(f"Display update failed: {e}")

    def set_visualization_mode(self, mode: str) -> None:
        """
        Set different visualization modes.

        Args:
            mode: Visualization mode ("debug")
        """
        info(f"Setting visualization mode to: {mode}")
        if mode == "debug":
            self.draw_node_id = True
            # Re-setup graph with node IDs
            self._setup_base_visualization()

    def get_window_size(self) -> Tuple[int, int]:
        """Get the configured window size."""
        return tuple(self.window_size)

    def get_game_speed(self) -> float:
        """Get the configured game speed multiplier."""
        return self.game_speed

    def is_visualization_enabled(self) -> bool:
        """Check if visualization is enabled/available."""
        return hasattr(self.ctx, "visual") and self.ctx.visual is not None

    # -------------------- Map overlay --------------------

    def setup_map_overlay(self, tiff_path: Optional[str]) -> None:
        """Load a GeoTIFF and register it as a background layer (layer=5).

        The TIFF must be in the same UTM CRS as the graph node coordinates.
        Requires rasterio, numpy, and pygame to be installed.

        Args:
            tiff_path: Absolute or relative path to the .tif/.tiff file.
                       If None or empty, this is a no-op.
        """
        if not tiff_path:
            return
        try:
            MapOverlayVisual(self.ctx, tiff_path, tuple(self.window_size))
        except Exception as e:
            warning(f"Map overlay not loaded: {e}")

    # -------------------- HUD --------------------

    def setup_hud(self) -> None:
        """Create the HUD overlay artist (call once after visuals are ready)."""
        if self._hud_artist_created:
            return
        try:
            from gamms.VisualizationEngine import Artist
            artist = Artist(self.ctx, self._render_hud, layer=100)
            self.ctx.visual.add_artist("hud_overlay", artist)
            self._hud_artist_created = True
        except Exception as e:
            warning(f"Failed to create HUD artist: {e}")

    def add_hud_event(self, text: str, color: tuple) -> None:
        """Push an event line to the HUD. Oldest entries are dropped when full."""
        self._hud_events.append((text, color))
        if len(self._hud_events) > self._hud_max_events:
            self._hud_events = self._hud_events[-self._hud_max_events:]

    def update_hud_payoff(self, total: float, tag: float, capture: float, discover: float) -> None:
        self._hud_payoff = {"total": total, "tag": tag, "capture": capture, "discover": discover}

    def _render_hud(self, ctx: Any, data: Dict[str, Any]) -> None:
        """Artist callback — draws event log + payoff at fixed screen position."""
        try:
            import pygame
            render_manager = ctx.visual._render_manager
            layer = render_manager.current_drawing_artist.get_layer()
            surface = ctx.visual._get_target_surface(layer)

            font = pygame.font.SysFont("monospace", self._hud_font_size)
            x, y = self._hud_x, self._hud_y
            line_h = self._hud_line_h

            # Payoff breakdown
            p = self._hud_payoff
            payoff_text = f"Red Payoff  Total:{p['total']:+.1f}  Capture:{p['capture']:+.1f}  Tag:{p['tag']:+.1f}  Discover:{p['discover']:+.1f}"
            surf = font.render(payoff_text, True, (0, 0, 0))
            surface.blit(surf, (x, y))
            y += line_h + 4

            # Event log
            for text, color in self._hud_events:
                surf = font.render(text, True, color)
                surface.blit(surf, (x, y))
                y += line_h
        except Exception:
            pass

    def __str__(self) -> str:
        """String representation of the VisEngine."""
        flags_status = "created" if self.flag_visual.flags_created else "not created"
        labels_count = len(self.agent_visual.agent_labels_created)
        return f"VisEngine(speed={self.game_speed}, flags={flags_status}, agent_labels={labels_count})"
