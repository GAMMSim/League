from typing import Any, Dict, List, Optional, Tuple
from typeguard import typechecked

try:
    from lib.core.console import *
    from lib.config.config_utils import compact_hex_to_binary
except ImportError:
    from ..core.console import *
    from ..config.config_utils import compact_hex_to_binary


@typechecked
class VisEngine:
    """
    Visualization Engine for managing all visual aspects of the game.
    Handles graph visualization, agent visuals, flags, and game settings.
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

        # Flag tracking
        self.flags_created = False
        
        # Agent name label tracking
        self.agent_labels_created = set()

        # Initialize visualization
        self._setup_base_visualization()
        success("VisEngine initialized")

    def _setup_base_visualization(self) -> None:
        """Set up the base graph visualization with default colors and settings."""
        try:
            debug("Setting up base visualization")
            # Import color constants
            try:
                from gamms.VisualizationEngine import Color

                default_node_color = Color.Black
                edge_color = Color.Gray

            except ImportError:
                # Fallback colors
                default_node_color = "black"
                edge_color = "gray"

            # Set base graph visualization
            self.ctx.visual.set_graph_visual(draw_id=self.draw_node_id, node_color=default_node_color, edge_color=edge_color, node_size=6)

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
        for idx, node_id in enumerate(positions):
            try:
                # Get node position
                node = self.ctx.graph.graph.get_node(node_id)

                # Try Artist API first
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

    def create_agent_labels(self, agents_config: Dict[str, Any]) -> None:
        """
        Create name labels for all agents above their initial configured positions.

        Args:
            agents_config: Complete agents configuration from config file
        """
        try:
            debug("Creating agent name labels")
            
            # Label settings
            label_offset_y = self.vis_config.get("agent_label_offset_y", -25)  # Above agent
            label_size = self.vis_config.get("agent_label_size", 12)
            
            # Default label color
            try:
                from gamms.VisualizationEngine import Color
                default_label_color = Color.Black
            except ImportError:
                default_label_color = (0, 0, 0)

            labels_created = 0

            # Process each team configuration
            for key, config_data in agents_config.items():
                if key.endswith("_config"):  # red_config, blue_config, etc.
                    team_name = key.replace("_config", "")
                    debug(f"Creating labels for {team_name} team agents")
                    
                    # Create label for each agent
                    for agent_name, agent_config in config_data.items():
                        try:
                            # Use initial position from config, not current position
                            initial_node_id = agent_config.get("start_node_id")
                            if initial_node_id is not None:
                                node = self.ctx.graph.graph.get_node(initial_node_id)
                                
                                # Agent-specific label color or default
                                label_color = agent_config.get("label_color", default_label_color)
                                
                                self._create_agent_label(agent_name, node.x, node.y, label_color, label_size, label_offset_y)
                                self.agent_labels_created.add(agent_name)
                                labels_created += 1
                                debug(f"Created label for '{agent_name}' at initial position node {initial_node_id}")
                            else:
                                warning(f"No start_node_id found for agent '{agent_name}' in config")
                                
                        except Exception as e:
                            warning(f"Failed to create label for agent '{agent_name}': {e}")

            success(f"Created name labels for {labels_created} agents at their initial positions")

        except Exception as e:
            error(f"Failed to create agent labels: {e}")

    def _create_agent_label(self, agent_name: str, x: float, y: float, color: Any, size: int, offset_y: float) -> None:
        """
        Create a text label for an agent at the specified position.
        
        Args:
            agent_name: Name of the agent
            x: X coordinate for the label
            y: Y coordinate for the label
            color: Color of the text
            size: Font size
            offset_y: Y offset from agent position (negative = above)
        """
        try:
            label_id = f"{agent_name}_label"
            
            # Try Artist API first
            try:
                from gamms.VisualizationEngine import Artist
                
                artist = Artist(self.ctx, self._render_agent_label, layer=30)
                artist.data.update({
                    "x": x,
                    "y": y + offset_y,
                    "text": agent_name,
                    "color": color,
                    "size": size
                })
                self.ctx.visual.add_artist(label_id, artist)
                
            except (ImportError, AttributeError):
                # Fallback to dictionary API
                data = {
                    "x": x,
                    "y": y + offset_y,
                    "text": agent_name,
                    "color": color,
                    "size": size,
                    "layer": 30,
                    "drawer": self._render_agent_label
                }
                self.ctx.visual.add_artist(label_id, data)
                
            debug(f"Created label for agent '{agent_name}' at ({x}, {y + offset_y})")
            
        except Exception as e:
            warning(f"Failed to create label artist for agent '{agent_name}': {e}")

    def _render_agent_label(self, ctx: Any, data: Dict[str, Any]) -> None:
        """Render function for agent name labels."""
        x = data.get("x", 0)
        y = data.get("y", 0)
        text = data.get("text", "")
        color = data.get("color", (0, 0, 0))
        size = data.get("size", 12)
        # Convert text to unicode or bytes
        if isinstance(text, str):
            try:
                converted_text = text.encode('utf-8')
            except Exception:
                converted_text = text  # Fallback to original string
        
        # Render text label - GAMMS render_text doesn't accept size parameter
        # Try different methods to render text with size
        try:
            # Method 1: Try font_size parameter
            ctx.visual.render_text(x, y, text, color=color, font_size=size)
        except TypeError:
            try:
                # Method 2: Try without size parameter
                ctx.visual.render_text(x, y, text, color=color)
            except Exception:
                # Method 3: Fallback - render simple text
                try:
                    ctx.visual.render_text(x, y, text)
                except Exception as e:
                    warning(f"Failed to render text label '{text}': {e}")
                    pass

    def update_agent_label(self, agent_name: str) -> None:
        """
        Update the position of an agent's name label to match their current location.

        Args:
            agent_name: Name of the agent whose label to update
        """
        if agent_name not in self.agent_labels_created:
            return
            
        try:
            # Get agent's current position
            agent = self.ctx.agent.get_agent(agent_name)
            if not agent or not hasattr(agent, 'curr_node_id'):
                return
                
            node = self.ctx.graph.graph.get_node(agent.curr_node_id)
            if not node:
                return
                
            # Update label position
            label_id = f"{agent_name}_label"
            label_offset_y = self.vis_config.get("agent_label_offset_y", -25)
            
            # Try to update existing artist
            try:
                if hasattr(self.ctx.visual, "get_artist"):
                    artist = self.ctx.visual.get_artist(label_id)
                    if artist:
                        artist.data["x"] = node.x
                        artist.data["y"] = node.y + label_offset_y
                        debug(f"Updated label position for agent '{agent_name}' to ({node.x}, {node.y + label_offset_y})")
                        
            except Exception:
                # Fallback: recreate the label
                self.remove_agent_label(agent_name)
                
                # Get agent config for color
                label_color = (0, 0, 0)  # Default black
                label_size = self.vis_config.get("agent_label_size", 12)
                
                self._create_agent_label(agent_name, node.x, node.y, label_color, label_size, label_offset_y)
                
        except Exception as e:
            warning(f"Failed to update label for agent '{agent_name}': {e}")

    def update_all_agent_labels(self) -> None:
        """Update positions of all agent name labels."""
        try:
            debug("Updating all agent label positions")
            for agent_name in list(self.agent_labels_created):
                self.update_agent_label(agent_name)
            debug("Finished updating agent label positions")
            
        except Exception as e:
            warning(f"Failed to update agent labels: {e}")

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
            
            # Remove main agent artist
            if hasattr(self.ctx.visual, "remove_artist"):
                try:
                    self.ctx.visual.remove_artist(agent_name)
                except Exception:
                    pass

                # Remove sensor artists
                sensor_names = [f"sensor_{sensor}" for sensor in ["map", "agent", "neighbor"]]
                for sensor_name in sensor_names:
                    try:
                        self.ctx.visual.remove_artist(sensor_name)
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

    def __str__(self) -> str:
        """String representation of the VisEngine."""
        flags_status = "created" if self.flags_created else "not created"
        labels_count = len(self.agent_labels_created)
        return f"VisEngine(speed={self.game_speed}, flags={flags_status}, agent_labels={labels_count})"