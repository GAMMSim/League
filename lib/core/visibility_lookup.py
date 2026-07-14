"""
Visualization-layer helpers for occlusion (line_of_sight) visibility models.

The visual layer needs two things a plain sensor doesn't: (1) which team/node
carries a line_of_sight model and how far it reaches, so it can compute a
real geometric visibility polygon at render time instead of a euclidean-radius
guess, and (2) the building footprints themselves, so occlusion is visible,
not just its effect. Both are derived from `environment.visibility_models` +
`agents.<team>_global.sensors` the same way lib/game/sensor_engine.py resolves
sensors — duplicated here (not imported from there) because the visual layer
is constructed before SensorEngine and has no live sensor objects to query.

`resolve_line_of_sight_ranges` is deliberately config-only (no table build) —
it's the cheap path used to size the geometric visibility polygon.
`resolve_line_of_sight_tables` DOES load the per-node table, for marking
exactly which nodes the real sensor considers visible (small glow markers) —
this is still cheap in practice: lib/game/sensor_engine.py has always
already built/cached these tables for gameplay by the time anything asks, so
this hits the process-wide in-memory cache in visibility_cache.py, not a
fresh build.
"""
from typing import Any, Dict, FrozenSet, List, Optional, Tuple
import os
import networkx as nx

try:
    from lib.core.visibility_cache import get_visibility_models
    from lib.core.visibility_generators import _get_building_polygons
except ImportError:
    from .visibility_cache import get_visibility_models
    from .visibility_generators import _get_building_polygons

# Fallback sensing radius (graph coordinate units) for a line_of_sight model
# with no configured max_range — occlusion is still limited by buildings, but
# something finite is needed as the polygon's outer "horizon" for rendering.
_DEFAULT_HORIZON = 1000.0


def _resolve_at(at_spec: Any, env_config: Dict[str, Any]) -> List[int]:
    if isinstance(at_spec, list):
        return [int(x) for x in at_spec]
    if isinstance(at_spec, str):
        return [int(x) for x in (env_config.get(at_spec) or [])]
    return []


def resolve_line_of_sight_ranges(
    config: Dict[str, Any],
) -> Tuple[Dict[str, float], Dict[int, float]]:
    """
    Find every `environment.visibility_models` entry of type `line_of_sight`
    that's actually referenced by an agent sensor, and return its configured
    range keyed by who carries it (falls back to _DEFAULT_HORIZON if the
    model has no max_range).

    Returns (dynamic, static):
      dynamic: {team_name: range} — sensors with carrier=agent (the default).
      static:  {node_id: range}   — sensors with an `at:` fan-out (towers).
    """
    dynamic: Dict[str, float] = {}
    static: Dict[int, float] = {}

    env_config = config.get("environment", {})
    model_specs = env_config.get("visibility_models", {})
    los_specs = {
        name: spec for name, spec in model_specs.items()
        if isinstance(spec, dict) and spec.get("type") == "line_of_sight"
    }
    if not los_specs:
        return dynamic, static

    agents_config = config.get("agents", {})
    for key, team_config in agents_config.items():
        if not key.endswith("_global"):
            continue
        team = key.replace("_global", "")
        for entry in team_config.get("sensors", []):
            if not isinstance(entry, dict):
                continue
            model_name = entry.get("model")
            spec = los_specs.get(model_name)
            if spec is None:
                continue
            max_range = float(spec.get("max_range") or _DEFAULT_HORIZON)
            if "at" in entry:
                for node_id in _resolve_at(entry["at"], env_config):
                    static[node_id] = max_range
            else:
                dynamic[team] = max_range

    return dynamic, static


def resolve_line_of_sight_tables(
    config: Dict[str, Any],
    graph: Optional[nx.Graph],
    vis_dir: str = "graphs/visibility",
) -> Tuple[Dict[str, Dict[int, FrozenSet[int]]], Dict[int, Dict[int, FrozenSet[int]]]]:
    """
    Same sensor-entry resolution as resolve_line_of_sight_ranges, but
    returns each model's actual per-node table instead of just its range —
    for marking exactly which nodes the real sensor (not the rendering
    approximation) considers visible.

    Returns (dynamic, static):
      dynamic: {team_name: table} — sensors with carrier=agent (the default).
      static:  {node_id: table}   — sensors with an `at:` fan-out (towers).
    """
    dynamic: Dict[str, Any] = {}
    static: Dict[int, Any] = {}
    if graph is None:
        return dynamic, static

    env_config = config.get("environment", {})
    model_specs = env_config.get("visibility_models", {})
    los_names = {
        name for name, spec in model_specs.items()
        if isinstance(spec, dict) and spec.get("type") == "line_of_sight"
    }
    if not los_names:
        return dynamic, static

    meta = getattr(graph, "graph", None)
    source_path = meta.get("__graph_source_path") if isinstance(meta, dict) else None
    base_dir = os.path.dirname(source_path) if source_path else "graphs"

    try:
        models = get_visibility_models(graph, model_specs, base_dir, vis_dir=vis_dir)
    except Exception:
        return dynamic, static

    agents_config = config.get("agents", {})
    for key, team_config in agents_config.items():
        if not key.endswith("_global"):
            continue
        team = key.replace("_global", "")
        for entry in team_config.get("sensors", []):
            if not isinstance(entry, dict):
                continue
            model_name = entry.get("model")
            if model_name not in los_names:
                continue
            table = models.get(model_name)
            if not table:
                continue
            if "at" in entry:
                for node_id in _resolve_at(entry["at"], env_config):
                    static[node_id] = table
            else:
                dynamic[team] = table

    return dynamic, static


def resolve_building_polygons(config: Dict[str, Any], graph: Optional[nx.Graph]) -> List[Any]:
    """
    Building footprint polygons (graph coordinate units) behind any
    line_of_sight model in `environment.visibility_models` — for drawing the
    actual occluders alongside sight rays. Uses the same fetch-or-cache path
    as the generator (lib/core/visibility_generators.py), so this never
    re-fetches anything the table build didn't already fetch. Returns [] if
    there's no line_of_sight model.
    """
    if graph is None:
        return []
    env_config = config.get("environment", {})
    model_specs = env_config.get("visibility_models", {})
    for spec in model_specs.values():
        if isinstance(spec, dict) and spec.get("type") == "line_of_sight":
            try:
                return _get_building_polygons(graph, spec)
            except Exception:
                return []
    return []
