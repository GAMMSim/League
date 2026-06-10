from typing import Any

try:
    from lib.core.console import *
except ImportError:
    from ..core.console import *

# Maximum pixels on the longest side when loading the TIFF into memory.
# Higher = sharper when zoomed in, but more RAM. 8192 is ~192 MB for RGB uint8.
_MAX_NATIVE_PX = 8192


class MapOverlayVisual:
    """
    Renders a GeoTIFF as a background layer in the GAMMS vis engine.

    Strategy: load the TIFF at high resolution (up to _MAX_NATIVE_PX on the
    longest side) and store it as a pygame surface alongside its UTM bounds.
    Each frame, the render callback crops the portion of the high-res surface
    that corresponds to the *visible screen area* and blits it directly —
    no quality-degrading double-scale.
    """

    def __init__(self, ctx: Any, tiff_path: str, window_size: tuple) -> None:
        try:
            import rasterio
            import rasterio.enums
            import numpy as np
            import pygame
        except ImportError as e:
            error(f"MapOverlayVisual requires rasterio, numpy, and pygame: {e}")
            raise

        self.ctx = ctx
        self._surface: Any = None
        self._bounds: Any = None  # rasterio BoundingBox (left, bottom, right, top) in UTM
        self._img_w: int = 0
        self._img_h: int = 0
        self._render_error_logged: bool = False

        try:
            with rasterio.open(tiff_path) as src:
                self._bounds = src.bounds

                native_w, native_h = src.width, src.height
                scale = min(1.0, _MAX_NATIVE_PX / max(native_w, native_h))
                load_w = max(1, int(native_w * scale))
                load_h = max(1, int(native_h * scale))
                self._img_w, self._img_h = load_w, load_h

                resampling = rasterio.enums.Resampling.bilinear
                count = src.count

                if count >= 3:
                    r = src.read(1, out_shape=(load_h, load_w), resampling=resampling)
                    g = src.read(2, out_shape=(load_h, load_w), resampling=resampling)
                    b = src.read(3, out_shape=(load_h, load_w), resampling=resampling)
                else:
                    band = src.read(1, out_shape=(load_h, load_w), resampling=resampling)
                    r = g = b = band

                def _to_uint8(arr):
                    lo, hi = float(arr.min()), float(arr.max())
                    if hi == lo:
                        return np.zeros_like(arr, dtype=np.uint8)
                    return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)

                # (H, W, 3) → transpose to (W, H, 3) for pygame.surfarray
                rgb = np.stack([_to_uint8(r), _to_uint8(g), _to_uint8(b)], axis=-1)
                self._surface = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))

            success(
                f"MapOverlayVisual: loaded '{tiff_path}' "
                f"at {load_w}x{load_h}px (native {native_w}x{native_h}), "
                f"bounds={self._bounds}"
            )

        except Exception as e:
            error(f"MapOverlayVisual: failed to load '{tiff_path}': {e}")
            raise

        self._register_artist()

    def _register_artist(self) -> None:
        try:
            from gamms.VisualizationEngine import Artist
            artist = Artist(self.ctx, self._render, layer=5)
            self.ctx.visual.add_artist("map_overlay_tiff", artist)
            debug("MapOverlayVisual: artist registered at layer 5")
        except Exception as e:
            error(f"MapOverlayVisual: failed to register artist: {e}")
            raise

    def _render(self, ctx: Any, data: Any) -> None:
        """
        Crop the high-res surface to the currently visible UTM region,
        then blit it to the render surface at native screen resolution.
        This avoids the double-downscale that causes pixelation.
        """
        try:
            import pygame
            rm = ctx.visual._render_manager
            layer = rm.current_drawing_artist.get_layer()
            surface = ctx.visual._get_target_surface()
            screen_w, screen_h = surface.get_size()

            b = self._bounds
            utm_w = b.right - b.left
            utm_h = b.top   - b.bottom  # positive

            # --- screen corners of the full TIFF in pixels ---
            full_sx0, full_sy0 = rm.world_to_screen(b.left,  b.top)
            full_sx1, full_sy1 = rm.world_to_screen(b.right, b.bottom)
            full_sw = full_sx1 - full_sx0  # pixels wide on screen
            full_sh = full_sy1 - full_sy0  # pixels tall on screen

            if full_sw <= 0 or full_sh <= 0:
                return

            # --- pixels-per-UTM-unit on screen right now ---
            px_per_utm_x = full_sw / utm_w
            px_per_utm_y = full_sh / utm_h

            # --- visible UTM region (clamp to TIFF bounds) ---
            # screen (0,0) → UTM
            vis_utm_left   = b.left   + (0         - full_sx0) / px_per_utm_x
            vis_utm_right  = b.left   + (screen_w  - full_sx0) / px_per_utm_x
            vis_utm_top    = b.top    - (0         - full_sy0) / px_per_utm_y
            vis_utm_bottom = b.top    - (screen_h  - full_sy0) / px_per_utm_y

            # clamp to image bounds
            vis_utm_left   = max(vis_utm_left,   b.left)
            vis_utm_right  = min(vis_utm_right,  b.right)
            vis_utm_top    = min(vis_utm_top,    b.top)
            vis_utm_bottom = max(vis_utm_bottom, b.bottom)

            if vis_utm_right <= vis_utm_left or vis_utm_top <= vis_utm_bottom:
                return

            # --- crop rect in image pixels ---
            img_x0 = int((vis_utm_left   - b.left)   / utm_w * self._img_w)
            img_x1 = int((vis_utm_right  - b.left)   / utm_w * self._img_w)
            img_y0 = int((b.top - vis_utm_top)        / utm_h * self._img_h)
            img_y1 = int((b.top - vis_utm_bottom)     / utm_h * self._img_h)

            img_x0 = max(0, min(img_x0, self._img_w - 1))
            img_x1 = max(img_x0 + 1, min(img_x1, self._img_w))
            img_y0 = max(0, min(img_y0, self._img_h - 1))
            img_y1 = max(img_y0 + 1, min(img_y1, self._img_h))

            crop_rect = pygame.Rect(img_x0, img_y0, img_x1 - img_x0, img_y1 - img_y0)
            crop = self._surface.subsurface(crop_rect)

            # --- destination rect on screen ---
            dst_x = int(full_sx0 + img_x0 / self._img_w * full_sw)
            dst_y = int(full_sy0 + img_y0 / self._img_h * full_sh)
            dst_w = int((img_x1 - img_x0) / self._img_w * full_sw)
            dst_h = int((img_y1 - img_y0) / self._img_h * full_sh)

            if dst_w <= 0 or dst_h <= 0:
                return

            scaled = pygame.transform.smoothscale(crop, (dst_w, dst_h))
            surface.blit(scaled, (dst_x, dst_y))

        except Exception as e:
            if not self._render_error_logged:
                warning(f"MapOverlayVisual render error (suppressing further): {e}")
                self._render_error_logged = True
