"""
Visibility-table generators: type -> builder(graph, params) -> table.

Each builder returns a plain dict[int, list[int]] (self-inclusive) — the same
shape lib/core/visibility_cache.py normalizes into dict[int, frozenset[int]].
See docs/sensor_redesign_handoff.md §3.2.
"""
import hashlib
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List
import networkx as nx


def _build_radius(graph: nx.Graph, params: Dict[str, Any]) -> Dict[int, List[int]]:
    """Euclidean region within `range` of each node (self included)."""
    rng = float(params["range"])
    r2 = rng * rng
    nodes = [(n, d["x"], d["y"]) for n, d in graph.nodes(data=True)]
    table: Dict[int, List[int]] = {}
    for n, nx_, ny_ in nodes:
        table[int(n)] = [
            int(m) for m, mx_, my_ in nodes
            if (mx_ - nx_) ** 2 + (my_ - ny_) ** 2 <= r2
        ]
    return table


def _build_khop(graph: nx.Graph, params: Dict[str, Any]) -> Dict[int, List[int]]:
    """Undirected shortest-path-length cutoff (self included at distance 0)."""
    k = int(params["k"])
    ug = graph.to_undirected()
    table: Dict[int, List[int]] = {}
    for node in ug.nodes:
        reachable = nx.single_source_shortest_path_length(ug, node, cutoff=k)
        table[int(node)] = sorted(int(n) for n in reachable)
    return table


_DEFAULT_BUILDINGS_CACHE_DIR = "graphs/buildings"


def _get_building_polygons(graph: nx.Graph, params: Dict[str, Any]) -> List[Any]:
    """
    Building footprint polygons in the graph's own x/y coordinate units.

    Either loaded from `params["buildings_path"]` (a pickled list of shapely
    Polygons in `crs` coordinates — for offline/reproducible fixed footprint
    sets), or live-fetched once from OpenStreetMap via osmnx for the node
    bbox (+ margin) and cached to disk keyed by (crs, bbox, margin) so later
    builds and mass_eval runs never need network access again.
    """
    buildings_path = params.get("buildings_path")
    if buildings_path:
        with open(buildings_path, "rb") as f:
            return pickle.load(f)

    crs = params["crs"]
    margin = float(params.get("margin", 150.0))

    xs = [d["x"] for _, d in graph.nodes(data=True)]
    ys = [d["y"] for _, d in graph.nodes(data=True)]
    min_x, max_x = min(xs) - margin, max(xs) + margin
    min_y, max_y = min(ys) - margin, max(ys) + margin

    cache_key = f"{crs}|{min_x:.1f}|{min_y:.1f}|{max_x:.1f}|{max_y:.1f}"
    cache_hash = hashlib.sha1(cache_key.encode()).hexdigest()[:16]
    cache_dir = Path(params.get("buildings_cache_dir", _DEFAULT_BUILDINGS_CACHE_DIR))
    cache_path = cache_dir / f"buildings_{cache_hash}.pkl"

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    import osmnx as ox
    from pyproj import Transformer
    from shapely.geometry import Polygon

    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    to_native = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    lon_a, lat_a = to_wgs84.transform(min_x, min_y)
    lon_b, lat_b = to_wgs84.transform(max_x, max_y)
    bbox = (min(lon_a, lon_b), min(lat_a, lat_b), max(lon_a, lon_b), max(lat_a, lat_b))

    gdf = ox.features_from_bbox(bbox, tags={"building": True})

    polygons: List[Any] = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        parts = [geom] if geom.geom_type == "Polygon" else (
            list(geom.geoms) if geom.geom_type == "MultiPolygon" else []
        )
        for part in parts:
            lon_xs, lat_ys = part.exterior.xy
            native_coords = [to_native.transform(x, y) for x, y in zip(lon_xs, lat_ys)]
            polygons.append(Polygon(native_coords))

    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(polygons, f)

    return polygons


def _build_line_of_sight(graph: nx.Graph, params: Dict[str, Any]) -> Dict[int, List[int]]:
    """
    Building-occlusion line of sight (self included): two nodes see each
    other iff the straight segment between them crosses no building
    footprint.

    params:
      crs: str — EPSG code matching the graph's x/y units (required),
          e.g. "EPSG:32618". Only used when live-fetching footprints.
      margin: float — buffer, in graph coordinate units, added around the
          node bbox when fetching building footprints. Default 150.0.
      max_range: optional float — additionally cap visibility to this
          Euclidean distance (graph coordinate units), matching the old
          radius-sensor semantics. None (default) = unlimited range, occlusion
          only.
      buildings_path: optional str — path to a pickled list of shapely
          Polygons (in `crs` coordinates) to use instead of live-fetching.
      buildings_cache_dir: optional str — where fetched footprints are
          cached. Default "graphs/buildings".
    """
    from shapely.geometry import LineString
    from shapely.strtree import STRtree

    buildings = _get_building_polygons(graph, params)
    tree = STRtree(buildings) if buildings else None

    max_range = params.get("max_range")
    max_range_sq = float(max_range) ** 2 if max_range is not None else None

    nodes = [(int(n), d["x"], d["y"]) for n, d in graph.nodes(data=True)]
    table: Dict[int, List[int]] = {}
    for n1, x1, y1 in nodes:
        visible = [n1]
        for n2, x2, y2 in nodes:
            if n2 == n1:
                continue
            if max_range_sq is not None and (x2 - x1) ** 2 + (y2 - y1) ** 2 > max_range_sq:
                continue
            segment = LineString([(x1, y1), (x2, y2)])
            blocked = False
            if tree is not None:
                for idx in tree.query(segment):
                    if segment.intersects(buildings[idx]):
                        blocked = True
                        break
            if not blocked:
                visible.append(n2)
        table[n1] = visible
    return table


_GENERATORS: Dict[str, Callable[[nx.Graph, Dict[str, Any]], Dict[int, List[int]]]] = {
    "radius": _build_radius,
    "khop": _build_khop,
    "line_of_sight": _build_line_of_sight,
}


def build(gen_type: str, graph: nx.Graph, params: Dict[str, Any]) -> Dict[int, List[int]]:
    try:
        builder = _GENERATORS[gen_type]
    except KeyError:
        raise ValueError(
            f"Unknown visibility generator type: {gen_type!r} (known: {sorted(_GENERATORS)})"
        )
    return builder(graph, params)
