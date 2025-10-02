from typing import Dict, Any, Optional, Callable
from gamms.SensorEngine import SensorType
from gamms.typing import ISensor

def create_team_agent_sensor(ctx, team_extractor: Optional[Callable[[str], str]] = None):
    """
    A wrapper around the built-in AGENT sensor that only reports teammates.
    Output: Dict[str, int] mapping teammate_id -> node_id (same as AGENT).
    Team is inferred from agent_id (default: '<team>_<index>').
    """

    def _default_extract(agent_id: str) -> str:
        # 'red_0' -> 'red'; robust fallback if '_' missing
        return agent_id.split('_', 1)[0] if '_' in agent_id else agent_id
    extract = team_extractor or _default_extract

    @ctx.sensor.custom("TEAM_AGENT")
    class TeamAgent(ISensor):
        def __init__(self, ctx, sensor_id: str, source_sensor_id: Optional[str] = None):
            """
            Args:
                ctx: GAMMS context
                sensor_id: unique id for this sensor
                source_sensor_id: id of an existing AGENT sensor to read from.
                                  If None, we lazily create a private AGENT sensor.
            """
            self._ctx = ctx
            self._sensor_id = str(sensor_id)
            self._owner: Optional[str] = None
            self._source_id: Optional[str] = source_sensor_id
            self._agent_src = None   # will hold the AGENT sensor instance
            self._data: Dict[str, Any] = {}

        # --- ISensor contract ---
        @property
        def sensor_id(self) -> str:
            return self._sensor_id

        @property
        def type(self) -> SensorType:
            return SensorType.TEAM_AGENT  # custom type, per API

        @property
        def data(self) -> Dict[str, Any]:
            return self._data

        def set_owner(self, owner: Optional[str]) -> None:
            if owner is not None and not isinstance(owner, str):
                raise TypeError("owner must be str or None")
            self._owner = owner

        def _ensure_source(self):
            # Prefer an existing AGENT sensor if id provided; else create one once.
            if self._agent_src is not None:
                return
            if self._source_id is not None:
                self._agent_src = self._ctx.sensor.get_sensor(self._source_id)
            else:
                # create_sensor registers it with the engine per API docs
                self._agent_src = self._ctx.sensor.create_sensor(
                    sensor_id=f"{self._sensor_id}__src",
                    sensor_type=SensorType.AGENT
                )

        def sense(self, node_id: int) -> None:
            if not self._owner:
                self._data = {}
                return

            self._ensure_source()
            
            # Trigger sensing on the source AGENT sensor
            self._agent_src.sense(node_id)  # ← ADD THIS LINE
            
            all_agents: Dict[str, int] = dict(self._agent_src.data)
            team = extract(self._owner)
            
            teammates = {
                aid: nid for aid, nid in all_agents.items()
                if aid != self._owner and extract(aid) == team
            }
            self._data = teammates

        def update(self, data: Dict[str, Any]) -> None:
            if "source_sensor_id" in data:
                self._source_id = data["source_sensor_id"]
                self._agent_src = None  # reset so we re-resolve

    return TeamAgent
