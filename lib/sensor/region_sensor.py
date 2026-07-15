"""
Table-backed region sensor (docs/sensor_redesign_handoff.md §2/§3.1).

Region is ALWAYS `table.get(origin, frozenset((origin,)))` — radius, k-hop,
and line-of-sight are all just generators that pre-build the table (see
lib/core/visibility_generators.py + lib/core/visibility_cache.py). No live
compute_region, no per-instance region cache: the table already holds every
origin's region as an O(1) lookup.

  stationary        = RegionSensor(carrier=<node_id>, ...)  # fixed origin
  egocentric_agent  = RegionSensor(carrier=DYNAMIC, ...)    # live origin
Both radius-derived and k-hop/line-of-sight tables use the same class — only
the table differs.
"""
from typing import Any, Dict

from lib.sensor.base_sensor import Sensor, DYNAMIC, _CLOCK


def create_region_sensor_class(ctx):
    from gamms.SensorEngine import SensorType

    @ctx.sensor.custom("REGION_SENSOR")
    class RegionSensor(Sensor):
        def __init__(self, ctx, sensor_id, table, model_name=None, **kw):
            super().__init__(ctx, sensor_id, SensorType.REGION_SENSOR, **kw)
            self._table = table or {}
            self._model = model_name

        def detect(self, origin_node, region) -> Dict[str, Any]:
            """Agents + configured flags currently inside the region."""
            agents = {
                a.name: a.current_node_id
                for a in self._ctx.agent.create_iter()
                if a.current_node_id in region
            }
            data: Dict[str, Any] = {
                "carrier": self._carrier,
                "team": self.team,
                "model": self._model,
                "region": region,                 # set/frozenset[int] of visible nodes from THIS origin
                "table": self._table,              # FULL node -> visible-nodes table this model builds from (same sensor, no separate name to learn)
                "detected_agents": agents,
            }
            if self._flags:
                data["detected_flags"] = [f for f in self._flags if f in region]
            return data

        # ---- the engine entry point ----
        def sense(self, node_id: int) -> None:
            origin = self._carrier if self.is_static else node_id

            # once-per-tick memo (only meaningful for shared static sensors)
            clock = _CLOCK["now"]
            tick = clock() if clock is not None else None
            if tick is not None and tick == self._last_tick and origin == self._last_origin:
                return  # already sensed this tick from this origin; keep self._data

            region = self._table.get(origin, frozenset((origin,)))
            self._data = self.detect(origin, region)
            self._last_tick = tick
            self._last_origin = origin

    return RegionSensor
