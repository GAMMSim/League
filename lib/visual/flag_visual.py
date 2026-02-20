from typing import Any, Dict, List, Tuple
from typeguard import typechecked

try:
    from lib.core.console import *
except ImportError:
    from ..core.console import *

# Transparency (alpha) for all sensor radius circles
SENSOR_ALPHA = 0.1
# Outline width (pixels) for all sensor radius circles
SENSOR_EDGE_WIDTH = 1
# Outline color (RGB) and alpha for all sensor radius circles
SENSOR_EDGE_COLOR = (200, 200, 200)
SENSOR_EDGE_ALPHA = 150


@typechecked
class FlagVisual:
    """
    Flag Visual Handler for managing all flag-related visualization.
    Handles flag creation, rendering, and sensor circles for flags.
    """

    def __init__(self, ctx: Any, config: Dict[str, Any]):
        """
        Initialize the Flag Visual Handler.

        Args:
            ctx: Game context object with visualization capabilities
            config: Complete configuration dictionary
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
            if game_rule == "v1.2":
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

                # Create sensor radius circle if stationary_sensor_radius exists
                if stationary_sensor_radius is not None:
                    try:
                        from gamms.VisualizationEngine import Artist
                        
                        debug(f"Creating sensor circle for flag at node {node_id}, position ({node.x}, {node.y}), radius {stationary_sensor_radius}")
                        
                        circle_artist = Artist(self.ctx, self._render_sensor_circle, layer=15)
                        circle_artist.data.update({
                            "x": node.x,
                            "y": node.y,
                            "radius": stationary_sensor_radius,
                            "color": (144, 238, 144, int(SENSOR_ALPHA * 255))  # Light green RGBA
                        })
                        self.ctx.visual.add_artist(f"{flag_type}_flag_{idx}_sensor_circle", circle_artist)
                        debug(f"Added sensor circle artist: {flag_type}_flag_{idx}_sensor_circle")
                        
                    except (ImportError, AttributeError):
                        # Fallback to dictionary API
                        circle_data = {
                            "x": node.x,
                            "y": node.y,
                            "radius": stationary_sensor_radius,
                            "color": (144, 238, 144, int(SENSOR_ALPHA * 255)),  # Light green RGBA
                            "layer": 10,
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

        # Flag dimensions
        flag_width = data.get("flag_width", 15)
        flag_height = data.get("flag_height", 10)
        pole_height = data.get("pole_height", 20)
        pole_width = 2

        # Draw flag pole (vertical rectangle)
        ctx.visual.render_rectangle(x - pole_width // 2, y + pole_height * 1.8, pole_width, pole_height, pole_color)

        # Draw flag (rectangle attached to pole top)
        ctx.visual.render_rectangle(x, y + pole_height * 1.8, flag_width, flag_height, flag_color)

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
        ctx.visual.render_text(text, x=x, y=y, color=color)

    def _render_sensor_circle(self, ctx: Any, data: Dict[str, Any]) -> None:
        """Render function for stationary sensor radius circle around flags."""
        x = data.get("x", 0)
        y = data.get("y", 0)
        radius = data.get("radius", 0)
        color = data.get("color", (144, 238, 144, 100))  # RGBA
        
        # Extract RGBA components
        if isinstance(color, tuple) and len(color) == 4:
            fill_r, fill_g, fill_b, alpha_value = color
        else:
            fill_r, fill_g, fill_b = color if len(color) >= 3 else (144, 238, 144)
            alpha_value = 100
        
        # Get the render manager to transform coordinates
        render_manager = ctx.visual._render_manager
        
        # Transform to screen coordinates
        screen_x, screen_y = render_manager.world_to_screen(x, y)
        screen_radius = render_manager.world_to_screen_scale(radius)
        
        if screen_radius < 1:
            return
        
        # Get the layer surface (not the main screen!)
        layer = render_manager.current_drawing_artist.get_layer()
        surface = ctx.visual._get_target_surface(layer)
        
        try:
            import pygame
            
            # Create temporary surface with alpha
            surf_size = int(screen_radius * 2 + 4)
            temp_surface = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)
            temp_surface.fill((0, 0, 0, 0))  # Clear with transparency
            
            center = int(screen_radius + 2)
            
            # Draw filled circle with transparency
            pygame.draw.circle(temp_surface, (fill_r, fill_g, fill_b, alpha_value), (center, center), int(screen_radius))
            
            # Draw black outline
            pygame.draw.circle(
                temp_surface,
                (
                    SENSOR_EDGE_COLOR[0],
                    SENSOR_EDGE_COLOR[1],
                    SENSOR_EDGE_COLOR[2],
                    SENSOR_EDGE_ALPHA,
                ),
                (center, center),
                int(screen_radius),
                SENSOR_EDGE_WIDTH,
            )
            
            # Blit to the LAYER surface (not _screen!)
            blit_pos = (int(screen_x - screen_radius - 2), int(screen_y - screen_radius - 2))
            surface.blit(temp_surface, blit_pos)
            
        except Exception as e:
            # Fallback: just draw outline
            ctx.visual.render_circle(x, y, radius, SENSOR_EDGE_COLOR, SENSOR_EDGE_WIDTH)
