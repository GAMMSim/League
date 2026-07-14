"""
Base sensor: team + carrier + gamms ISensor plumbing + per-tick memo.

Root for both sensor families (region sensors, info sensors — see
docs/sensor_redesign_handoff.md §2 for the model, §3.1 for this hierarchy).

  team    : "red" | "blue"           -- who owns/reads it; ALWAYS required,
            never None/universal. A sensor shared across teams by name (e.g.
            global_map) is still one entry per declaring team block.
  carrier : int node_id  -> STATIC    -- bolted to a fixed node; region cached
            & reused; may be team-shared.
            None (DYNAMIC) -> region sensors only: carried by the holding
            agent, origin = live node, recomputed every tick.
            Info sensors also use carrier=None (aspatial, no origin at all)
            but their sense() ignores it entirely — they never read
            is_static/DYNAMIC semantics, so the two meanings never collide.
"""
from typing import Any, Dict, Optional
from gamms.typing import ISensor

# Sentinel: carrier is the holding agent (use the live node gamms passes in).
# Also the resting carrier value for aspatial info sensors, which ignore it.
DYNAMIC = None

# Optional process-wide clock. If the GameEngine sets this to a callable returning
# the current tick, shared static sensors skip recomputation when sense() is
# called again within the same tick. Left unset => memo disabled (always correct).
_CLOCK: Dict[str, Any] = {"now": None}


def set_clock(clock_fn) -> None:
    """Enable the once-per-tick memo. clock_fn() must return the current tick (int)."""
    _CLOCK["now"] = clock_fn


class Sensor(ISensor):
    """Common team/carrier/plumbing root. Subclasses implement sense()."""

    def __init__(
        self,
        ctx,
        sensor_id: str,
        sensor_type=None,
        *,
        team: str,
        carrier: Optional[int] = DYNAMIC,
        flags=None,
    ):
        assert team in ("red", "blue"), f"team must be 'red' or 'blue', got {team!r}"
        self._ctx = ctx
        self._sensor_id = sensor_id
        self._type = sensor_type            # overridden by @ctx.sensor.custom in factories
        self.team = team                    # your layer only; gamms never reads this
        self._carrier = carrier             # int node_id (static), or None (DYNAMIC/aspatial)
        self._flags = list(flags) if flags else []
        self._owner = None
        self._data: Dict[str, Any] = {}

        # once-per-tick memo (region sensors only; info sensors ignore this)
        self._last_tick = None
        self._last_origin = None

    # ---- gamms ISensor contract ----
    @property
    def sensor_id(self) -> str:
        return self._sensor_id

    @property
    def type(self):
        return self._type

    @property
    def data(self):
        return self._data

    def set_owner(self, owner) -> None:
        self._owner = owner

    def update(self, data) -> None:
        pass

    # ---- convenience ----
    @property
    def is_static(self) -> bool:
        return self._carrier is not DYNAMIC
