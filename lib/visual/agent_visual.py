from typing import Any, Dict, Set
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
class AgentVisual:
    """
    Agent Visual Handler for managing all agent-related visualization.
    Handles agent visuals, labels, and sensor circles.
    """

    def __init__(self, ctx: Any, config: Dict[str, Any], agent_properties: Dict[str, Dict[str, Any]], team_properties: Dict[str, Dict[str, Any]]):
        """
        Initialize the Agent Visual Handler.

        Args:
            ctx: Game context object with visualization capabilities
            config: Complete configuration dictionary
            agent_properties: Dictionary of agent properties extracted by VisEngine
            team_properties: Dictionary of team properties extracted by VisEngine
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

            # Define team colors
            team_colors = {
                "red": self.colors.get("red_global", "red"),
                "blue": self.colors.get("blue_global", "blue")
            }

            agents_configured = 0

            # Process each team configuration
            for key, config_data in agents_config.items():
                if key.endswith("_config"):  # red_config, blue_config, etc.
                    team_name = key.replace("_config", "")  # extract "red" from "red_config"
                    debug(f"Processing {key} for team '{team_name}'")
                    
                    # Get global config for this team
                    global_key = f"{team_name}_global"
                    global_config = agents_config.get(global_key, {})
                    
                    # Get team color
                    team_color = team_colors.get(team_name, default_color)
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
                    self.create_agent_sensor_circle(agent_name, sensing_radius, sensor_color_rgba)
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
            artist.set_artist_type(ArtistType.AGENT)
            
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

            # Check if we're in animation mode
            waiting_simulation = data.get('_waiting_simulation', False)

            if waiting_simulation:
                # Get interpolation alpha
                alpha = data.get('_alpha', 1.0)

                # Get previous and current nodes
                prev_node = ctx.graph.graph.get_node(agent.prev_node_id)
                curr_node = ctx.graph.graph.get_node(current_node_id)

                # Interpolate position (same as agent does)
                x = prev_node.x + alpha * (curr_node.x - prev_node.x)
                y = prev_node.y + alpha * (curr_node.y - prev_node.y)
            else:
                # Static position when not animating
                node = ctx.graph.graph.get_node(current_node_id)
                x = node.x
                y = node.y

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

            ctx.visual.render_text(text, x=x, y=y, color=color)
        except Exception:
            pass  # Silent fail if agent doesn't exist

    def create_agent_sensor_circle(self, agent_name: str, radius: float, color: tuple = (0, 255, 255, 100)) -> None:
        """
        Create a transparent circle that follows an agent.
        
        Args:
            agent_name: Name of the agent to follow
            radius: Circle radius in world coordinates
            color: RGBA tuple (R, G, B, Alpha)
        """
        try:
            from gamms.VisualizationEngine import Artist
            from gamms.typing import ArtistType
            
            circle_id = f"{agent_name}_sensor_circle"
            
            artist = Artist(self.ctx, self._render_agent_circle, layer=25)
            artist.set_artist_type(ArtistType.AGENT)  # Makes it animate!
            
            artist.data.update({
                "agent_name": agent_name,
                "radius": radius,
                "color": color,
                "_alpha": 1.0  # Required for animation
            })
            
            self.ctx.visual.add_artist(circle_id, artist)
            debug(f"Created animated circle for agent '{agent_name}'")
            
        except Exception as e:
            warning(f"Failed to create circle for agent '{agent_name}': {e}")

    def _render_agent_circle(self, ctx: Any, data: Dict[str, Any]) -> None:
        """Render a transparent circle that follows an agent."""
        agent_name = data.get("agent_name", "")
        radius = data.get("radius", 50)
        color = data.get("color", (0, 255, 255, 100))
        
        try:
            # Get the agent
            agent = ctx.agent.get_agent(agent_name)
            
            # Check if we're in animation mode
            waiting_simulation = data.get('_waiting_simulation', False)
            
            if waiting_simulation:
                # Get interpolation alpha
                alpha = data.get('_alpha', 1.0)
                
                # Get previous and current nodes
                prev_node = ctx.graph.graph.get_node(agent.prev_node_id)
                curr_node = ctx.graph.graph.get_node(agent.current_node_id)
                
                # Interpolate position (same as agent does)
                x = prev_node.x + alpha * (curr_node.x - prev_node.x)
                y = prev_node.y + alpha * (curr_node.y - prev_node.y)
            else:
                # Static position when not animating
                node = ctx.graph.graph.get_node(agent.current_node_id)
                x = node.x
                y = node.y
            
            # Extract RGBA
            if len(color) == 4:
                fill_r, fill_g, fill_b, alpha_value = color
            else:
                fill_r, fill_g, fill_b, alpha_value = color[0], color[1], color[2], 100
            
            # Get render manager
            render_manager = ctx.visual._render_manager
            
            # Transform to screen coordinates
            screen_x, screen_y = render_manager.world_to_screen(x, y)
            screen_radius = render_manager.world_to_screen_scale(radius)
            
            if screen_radius < 1:
                return
            
            # Get the layer surface
            layer = render_manager.current_drawing_artist.get_layer()
            surface = ctx.visual._get_target_surface(layer)
            
            try:
                import pygame
                
                # Create temporary surface with alpha
                surf_size = int(screen_radius * 2 + 4)
                temp_surface = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)
                temp_surface.fill((0, 0, 0, 0))
                
                center = int(screen_radius + 2)
                
                # Draw filled circle with transparency
                pygame.draw.circle(temp_surface, (fill_r, fill_g, fill_b, alpha_value), (center, center), int(screen_radius))
                
                # Draw black outline (same as flag circles)
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
                
                # Blit to layer surface
                blit_pos = (int(screen_x - screen_radius - 2), int(screen_y - screen_radius - 2))
                surface.blit(temp_surface, blit_pos)
                
            except Exception:
                pass  # Silent fail
                
        except Exception as e:
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

            # Get the layer surface
            layer = render_manager.current_drawing_artist.get_layer()
            surface = ctx.visual._get_target_surface(layer)

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

        ctx.visual.render_text(text, x=x, y=y, color=color)

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
