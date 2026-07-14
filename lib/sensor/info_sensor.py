"""
Aspatial info sensors: data payloads with no region (docs/sensor_redesign_handoff.md
§2/§3.1). `team` is still required (never None/universal — a sensor read by both
teams is one instance per declaring team block, same as region sensors); `carrier`
is forced None (no spatial anchor at all). sense() ignores node_id entirely — the
payload is fixed at construction and returned by reference every tick.
"""
from typing import Any, Dict, List

from lib.sensor.base_sensor import Sensor


class InfoSensor(Sensor):
    """Static payload, no region. Subclasses set self._data once in __init__."""

    def __init__(self, ctx, sensor_id, sensor_type=None, *, team: str, flags=None):
        super().__init__(ctx, sensor_id, sensor_type, team=team, carrier=None, flags=flags)

    def sense(self, node_id: int) -> None:
        pass  # static payload set at construction; nothing to recompute


def create_global_map_sensor_class(ctx):
    """GlobalMapSensor: {graph, apsp} — full topology + APSP lookup, by reference."""
    from gamms.SensorEngine import SensorType
    try:
        from lib.core.apsp_cache import get_apsp_length_cache
    except ImportError:
        from ..core.apsp_cache import get_apsp_length_cache

    @ctx.sensor.custom("GLOBAL_MAP")
    class GlobalMapSensor(InfoSensor):
        def __init__(self, ctx, sensor_id, nx_graph, **kw):
            super().__init__(ctx, sensor_id, SensorType.GLOBAL_MAP, **kw)
            self._data: Dict[str, Any] = {
                "graph": nx_graph,
                "apsp": get_apsp_length_cache(nx_graph),
            }

    return GlobalMapSensor


def create_candidate_flag_sensor_class(ctx):
    """CandidateFlagSensor: {candidate_flags} — the configured candidate list."""
    from gamms.SensorEngine import SensorType

    @ctx.sensor.custom("CANDIDATE_FLAG_SENSOR")
    class CandidateFlagSensor(InfoSensor):
        def __init__(self, ctx, sensor_id, candidate_flags: List[int], **kw):
            super().__init__(ctx, sensor_id, SensorType.CANDIDATE_FLAG_SENSOR, **kw)
            self._data = {"candidate_flags": candidate_flags}

    return CandidateFlagSensor


def create_flag_sensor_class(ctx):
    """FlagSensor: {real_flags, fake_flags} — fake = candidate - real, computed once."""
    from gamms.SensorEngine import SensorType

    @ctx.sensor.custom("FLAG_SENSOR")
    class FlagSensor(InfoSensor):
        def __init__(self, ctx, sensor_id, real_flags: List[int], candidate_flags: List[int], **kw):
            super().__init__(ctx, sensor_id, SensorType.FLAG_SENSOR, **kw)
            fake_flags = list(set(candidate_flags) - set(real_flags))
            self._data = {"real_flags": real_flags, "fake_flags": fake_flags}

    return FlagSensor
