"""
Real geometric visibility-polygon computation, for the visualization layer
ONLY (see the `vis` gating in lib/game/visualization_engine_new.py — this is
never constructed in a headless/no-vis run).

Casts a ray from a sensing origin toward every nearby building corner and
finds the nearest building-edge intersection along it; connecting those
intersection points in angular order gives the exact region the buildings
block, instead of a heuristic (radius disk, or an arbitrary angular-gap
threshold over the discrete node set). Standard "visibility polygon among
polygonal obstacles" algorithm (e.g. Red Blob Games' "2D Visibility").
"""
import math
from typing import Any, List, Optional, Tuple

Point = Tuple[float, float]
Segment = Tuple[Point, Point]

# Perturbation applied on either side of each building-corner angle so the
# ray sweep captures the edge immediately behind/beside a corner, not just
# the corner point itself — without this, a corner can look "open" past it.
_EPS_ANGLE = 1e-4
# Baseline points sampled evenly around the full circle, ALWAYS merged with
# the building-corner angles. Buildings are isolated obstacles in an
# otherwise-open field, not a fully enclosing wall — without a baseline, an
# open arc between two unrelated buildings has no ray cast into it at all,
# so the polygon boundary would cut a straight chord across empty space
# instead of bulging out to the sensing radius (the "horizon").
_HORIZON_SAMPLES = 48


def _ray_segment_intersection(ox: float, oy: float, dx: float, dy: float, seg: Segment) -> Optional[float]:
    """Distance t >= 0 along ray (ox,oy)+t*(dx,dy) where it crosses `seg`,
    or None if the ray misses the segment (or they're parallel)."""
    (ax, ay), (bx, by) = seg
    ex, ey = bx - ax, by - ay
    denom = dx * ey - dy * ex
    if -1e-12 < denom < 1e-12:
        return None
    t = ((ax - ox) * ey - (ay - oy) * ex) / denom
    s = ((ax - ox) * dy - (ay - oy) * dx) / denom
    if t >= 0 and 0.0 <= s <= 1.0:
        return t
    return None


def _segment_near(seg: Segment, ox: float, oy: float, radius: float) -> bool:
    """Cheap bounding-box prefilter: could this segment possibly fall within
    `radius` of the origin?"""
    (x1, y1), (x2, y2) = seg
    return not (
        max(x1, x2) < ox - radius or min(x1, x2) > ox + radius
        or max(y1, y2) < oy - radius or min(y1, y2) > oy + radius
    )


class VisibilityPolygonIndex:
    """
    Spatially-indexed building geometry for repeated visibility-polygon
    queries against the same building set (built once per graph/config, then
    queried per sensing origin — see resolve_building_polygons).
    """

    def __init__(self, polygons: List[Any]):
        self._polygons = [p for p in polygons if p is not None and not p.is_empty]
        self._tree = None
        if self._polygons:
            try:
                from shapely.strtree import STRtree
                self._tree = STRtree(self._polygons)
            except Exception:
                self._tree = None

    def _candidate_segments(self, origin: Point, radius: float) -> List[Segment]:
        ox, oy = origin
        if self._tree is not None:
            try:
                from shapely.geometry import Point as ShapelyPoint
                query_area = ShapelyPoint(ox, oy).buffer(radius)
                candidates = [self._polygons[i] for i in self._tree.query(query_area)]
            except Exception:
                candidates = self._polygons
        else:
            candidates = self._polygons

        segments: List[Segment] = []
        for poly in candidates:
            coords = list(poly.exterior.coords)
            for i in range(len(coords) - 1):
                seg = (coords[i], coords[i + 1])
                if _segment_near(seg, ox, oy, radius):
                    segments.append(seg)
        return segments

    def polygon_at(self, origin: Point, radius: float) -> List[Point]:
        """The true visibility-polygon boundary (angular order) from `origin`
        out to `radius`, carved by real building edges. Empty buildings
        nearby -> a coarse circle at `radius` (nothing to occlude)."""
        ox, oy = origin
        segments = self._candidate_segments(origin, radius)

        # NOTE: must match atan2's own (-pi, pi] convention — a baseline
        # sample and a building-corner angle for the same real-world
        # direction have to sort adjacent to each other. Generating this in
        # [0, 2*pi) instead silently wired the polygon's lower half out of
        # angular order (every direction with atan2 < 0 sorted nowhere near
        # its true neighbors), rendering as an unfilled gap below the origin.
        angles = {2 * math.pi * i / _HORIZON_SAMPLES - math.pi for i in range(_HORIZON_SAMPLES)}
        for (ax, ay), (bx, by) in segments:
            for px, py in ((ax, ay), (bx, by)):
                if math.hypot(px - ox, py - oy) < 1e-9:
                    continue
                ang = math.atan2(py - oy, px - ox)
                angles.add(ang)
                angles.add(ang - _EPS_ANGLE)
                angles.add(ang + _EPS_ANGLE)

        points: List[Point] = []
        for ang in sorted(angles):
            dx, dy = math.cos(ang), math.sin(ang)
            best_t = radius
            for seg in segments:
                t = _ray_segment_intersection(ox, oy, dx, dy, seg)
                if t is not None and t < best_t:
                    best_t = t
            points.append((ox + dx * best_t, oy + dy * best_t))
        return points
