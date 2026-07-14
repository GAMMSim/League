from typing import Any, Dict, List, Optional
from typeguard import typechecked
import networkx as nx

try:
    from lib.core.console import *
    from lib.core.visibility_lookup import resolve_building_polygons
except ImportError:
    from ..core.console import *
    from ..core.visibility_lookup import resolve_building_polygons

# Fill color/alpha for building footprints (the occluders behind any
# line_of_sight visibility model) and their outline.
BUILDING_FILL_COLOR = (60, 60, 60)
BUILDING_FILL_ALPHA = 140
BUILDING_EDGE_COLOR = (30, 30, 30, 200)
BUILDING_EDGE_WIDTH = 1


@typechecked
class BuildingVisual:
    """
    Draws the building footprints that back any `line_of_sight` visibility
    model as filled polygons — makes the occlusion in sight-ray/region
    drawings legible (the buildings that caused the gaps are on screen too),
    instead of only showing their effect.

    Static content: the polygons themselves never change mid-game, but their
    *screen* position depends on the current camera view, so they're
    recomputed (world_to_screen only, no re-fetch) every frame like the other
    static overlays (see map_overlay_visual.py).
    """

    def __init__(self, ctx: Any, config: Dict[str, Any], graph: Optional[nx.Graph] = None, vis: bool = True):
        self.ctx = ctx
        self.config = config
        self._polygons = resolve_building_polygons(config, graph) if vis else []
        self._created = False

    def create_buildings(self) -> None:
        if self._created or not self._polygons:
            return
        try:
            from gamms.VisualizationEngine import Artist

            world_polygons: List[List[Any]] = [
                list(poly.exterior.coords) for poly in self._polygons if poly is not None and not poly.is_empty
            ]
            artist = Artist(self.ctx, self._render, layer=8)
            artist.data.update({"polygons": world_polygons})
            self.ctx.visual.add_artist("buildings_overlay", artist)
            self._created = True
            success(f"BuildingVisual: drew {len(world_polygons)} building footprint(s)")
        except Exception as e:
            warning(f"Failed to create building overlay: {e}")

    def _render(self, ctx: Any, data: Dict[str, Any]) -> None:
        polygons = data.get("polygons", [])
        if not polygons:
            return
        try:
            import pygame

            render_manager = ctx.visual._render_manager
            surface = ctx.visual._get_target_surface()

            for world_points in polygons:
                screen_points = [render_manager.world_to_screen(px, py) for px, py in world_points]
                if len(screen_points) < 3:
                    continue

                pad = BUILDING_EDGE_WIDTH + 2
                xs = [p[0] for p in screen_points]
                ys = [p[1] for p in screen_points]
                min_x, max_x = min(xs) - pad, max(xs) + pad
                min_y, max_y = min(ys) - pad, max(ys) + pad
                w, h = int(max_x - min_x), int(max_y - min_y)
                if w <= 0 or h <= 0:
                    continue

                temp_surface = pygame.Surface((w, h), pygame.SRCALPHA)
                local_points = [(px - min_x, py - min_y) for px, py in screen_points]
                pygame.draw.polygon(
                    temp_surface,
                    (*BUILDING_FILL_COLOR, BUILDING_FILL_ALPHA),
                    local_points,
                )
                pygame.draw.polygon(temp_surface, BUILDING_EDGE_COLOR, local_points, BUILDING_EDGE_WIDTH)
                surface.blit(temp_surface, (int(min_x), int(min_y)))
        except Exception:
            pass  # Silent fail — matches the other overlay renderers' convention
