from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Iterable
import math

# You may need to adjust these imports depending on your project layout.
from gamms.typing.sensor_engine import SensorType, ISensor

# ---------- Small utilities ----------

def _euclid(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance; used by the custom sensors."""
    dx, dy = x2 - x1, y2 - y1
    return math.hypot(dx, dy)

# ---------- Custom sensors ----------

class TeamAgentRangeSensor(ISensor):
    """
    Team-only proximity sensor (radius-only).
    - Returns ONLY teammates within a radius of the sensing center.
    - Works as a normal per-agent sensor (set_owner(agent_name) and pass the
      *agent’s current node* to sense()), or as ownerless + fixed team if you
      want a static team probe (pass team at ctor and call sense(anchor_node)).

    Data contract (stable for downstream code):
      data = {"teammates": Dict[str, int]}  # agent_name -> node_id

    Typical usage patterns
    ----------------------
    1) Per-agent (moving):
       sensor = TeamAgentRangeSensor("team_range:alice", ctx, radius=250.0, team_of=team_of)
       sensor.set_owner("alice")
       ctx.sensor.add_sensor(sensor)
       # then include "team_range:alice" in alice’s sensors list when creating the agent

    2) Team-level (static, anchored):
       sensor = TeamAgentRangeSensor("team_range:red:ownerless", ctx, radius=300.0, team="red")
       sensor.set_owner(None)
       ctx.sensor.add_sensor(sensor)
       # each tick:
       sensor.sense(anchor_node_id)
       use sensor.data["teammates"]  # visible red teammates around the anchor
    """
    def __init__(self, sensor_id: str, ctx, radius: float,
                 team_of: Optional[Dict[str, str]] = None,
                 team: Optional[str] = None):
        self._id = sensor_id
        self._ctx = ctx
        self._r = float(radius)
        self._owner: Optional[str] = None
        self._team_of = team_of or {}
        self._fixed_team = team  # if set, this sensor always uses this team
        self._data: Dict[str, Any] = {"teammates": {}}

    @property
    def sensor_id(self) -> str:
        return self._id

    @property
    def type(self):
        # We mark custom types as CUSTOM (allowed for user-defined sensors).
        return SensorType.CUSTOM

    @property
    def data(self) -> Dict[str, Any]:
        # Downstream code reads this after calling sense(node_id).
        return self._data

    def set_owner(self, owner: Optional[str]) -> None:
        # Owner should be the querying agent’s name, or None for team/static usage.
        self._owner = owner

    def update(self, data: Dict[str, Any]) -> None:
        # Runtime reconfiguration hook (optional).
        if "radius" in data:
            self._r = float(data["radius"])

    def _agent_team(self, name: str) -> Optional[str]:
        # Robust team lookup: prefer fixed team, then external map, then agent meta.
        if self._fixed_team is not None:
            return self._fixed_team
        if name in self._team_of:
            return self._team_of[name]
        ag = self._ctx.agent.get_agent(name)
        meta = getattr(ag, "meta", {}) or {}
        return meta.get("team")

    def sense(self, node_id: int) -> None:
        """
        Compute visible teammates within radius of `node_id`.
        - For moving per-agent usage, pass the owner’s current node_id.
        - For static team usage, pass your team’s anchor node_id each tick.
        """
        if self._owner is None and self._fixed_team is None:
            # No owner and no fixed team => can’t decide teammate set.
            self._data["teammates"] = {}
            return

        my_team = self._fixed_team or self._agent_team(self._owner)  # type: ignore
        if my_team is None:
            self._data["teammates"] = {}
            return

        # Look up the sensing center’s (x, y).
        node = self._ctx.graph.get_node(node_id)
        ox, oy = node.x, node.y

        # Scan all agents; this stays light for small populations.
        hits: Dict[str, int] = {}
        for name in self._ctx.agent.list_agents():
            if name == self._owner:
                continue  # ignore self in per-agent mode
            if self._agent_team(name) != my_team:
                continue
            a = self._ctx.agent.get_agent(name)
            nid = a.curr_node_id
            n = self._ctx.graph.get_node(nid)
            if _euclid(ox, oy, n.x, n.y) <= self._r:
                hits[name] = nid

        self._data["teammates"] = hits


class FlagRangeSensor(ISensor):
    """
    “Flags in radius” sensor (radius-only against static flag nodes).
    - Good for detecting which flags are currently close to the sensing center.
    - Works as moving (owner=agent) or static (owner=None) sensor.

    Data contract:
      data = {"flags": List[int]}  # node_ids of flags within radius

    Typical usage
    -------------
    1) Per-agent (moving):
       s = FlagRangeSensor("flag_range:alice", ctx, 200.0, flag_nodes=[101, 305, 777])
       s.set_owner("alice")
       ctx.sensor.add_sensor(s)
       # include "flag_range:alice" in alice's sensors list

    2) Team/static (ownerless):
       s = FlagRangeSensor("flag_range:red:ownerless", ctx, 250.0, flag_nodes=[101, 305, 777])
       s.set_owner(None)
       ctx.sensor.add_sensor(s)
       # each tick: s.sense(anchor_node_id)
       # then read s.data["flags"] to get visible flags around the anchor.
    """
    def __init__(self, sensor_id: str, ctx, radius: float, flag_nodes: Iterable[int]):
        self._id = sensor_id
        self._ctx = ctx
        self._r = float(radius)
        self._flags = list(flag_nodes)  # static set of node_ids
        self._owner: Optional[str] = None
        self._data: Dict[str, Any] = {"flags": []}

    @property
    def sensor_id(self) -> str:
        return self._id

    @property
    def type(self):
        return SensorType.CUSTOM

    @property
    def data(self) -> Dict[str, Any]:
        return self._data

    def set_owner(self, owner: Optional[str]) -> None:
        self._owner = owner

    def update(self, data: Dict[str, Any]) -> None:
        # Runtime updates: radius and/or new flag set.
        if "radius" in data:
            self._r = float(data["radius"])
        if "flag_nodes" in data:
            self._flags = list(data["flag_nodes"])

    def sense(self, node_id: int) -> None:
        """Emit all flag node_ids within radius of `node_id`."""
        base = self._ctx.graph.get_node(node_id)
        ox, oy = base.x, base.y
        visible: List[int] = []
        for fid in self._flags:
            fn = self._ctx.graph.get_node(fid)
            if _euclid(ox, oy, fn.x, fn.y) <= self._r:
                visible.append(fid)
        self._data["flags"] = visible

# ---------- Engine wrapper you can use from your factory ----------

class SensorEngineExt:
    """
    Sensor manager that creates and tracks all sensors listed at top.

    Where to use
    ------------
    - During game setup: call `build_all(...)` ONCE to pre-create everything.
    - During agent creation: pass the precomputed per-agent sensor IDs list
      to your `_create_single_agent(..., ctx_params['sensors']=...)`.
    - Each tick: call `tick_team_probes()` to update ownerless static sensors.

    What you get back
    -----------------
    - `build_all(...)` returns:
        per_agent_ids: { agent_name: [sensor_id, ...] }
        per_team_static_ids: { team: sensor_id }   # static ownerless AGENT_RANGE
    """
    def __init__(self, ctx, default_radius: float = 250.0, team_of: Optional[Dict[str, str]] = None):
        self.ctx = ctx
        self.default_radius = float(default_radius)
        self.team_of = team_of or {}

        # Internal bookkeeping so you can read or debug later if needed
        self._per_agent_ids: Dict[str, List[str]] = {}
        self._team_static_ids: Dict[str, str] = {}     # team -> sensor_id
        self._team_static_anchors: Dict[str, int] = {} # team -> anchor node

    # ----- Single-sensor creators (you can also call these ad hoc) -----

    def create_map_range(self, agent_name: str, radius: Optional[float] = None) -> str:
        """
        Create a per-agent **map RANGE** (radius-only) sensor and set its owner.
        Attach this sensor ID to the same agent’s `sensors` list at creation time.
        """
        sid = f"map_range:{agent_name}"
        if not self.ctx.sensor.has_sensor(sid):
            self.ctx.sensor.create_sensor(sid, SensorType.RANGE,
                                          sensor_range=float(radius or self.default_radius))
            self.ctx.sensor.get_sensor(sid).set_owner(agent_name)
        return sid

    def create_agent(self, agent_name: str) -> str:
        """
        Create a per-agent **AGENT** (ungated) sensor and set its owner.
        Use this only if you truly want omniscient agent positions (no radius).
        """
        sid = f"agent:{agent_name}"
        if not self.ctx.sensor.has_sensor(sid):
            self.ctx.sensor.create_sensor(sid, SensorType.AGENT)
            self.ctx.sensor.get_sensor(sid).set_owner(agent_name)
        return sid

    def create_agent_range(self, agent_name: str, radius: Optional[float] = None) -> str:
        """
        Create a per-agent **AGENT_RANGE** (radius-only) sensor and set its owner.
        This is usually what you want for moving “who is near me?” queries.
        """
        sid = f"agent_range:{agent_name}"
        if not self.ctx.sensor.has_sensor(sid):
            self.ctx.sensor.create_sensor(sid, SensorType.AGENT_RANGE,
                                          sensor_range=float(radius or self.default_radius))
            self.ctx.sensor.get_sensor(sid).set_owner(agent_name)
        return sid

    def create_team_sensor(self, agent_name: str, radius: Optional[float] = None, team: Optional[str] = None) -> str:
        """
        Create a per-agent **team-only** sensor and set owner=agent_name.
        Downstream usage is identical to any other per-agent sensor; only payload differs.
        """
        sid = f"team_range:{agent_name}"
        if not self.ctx.sensor.has_sensor(sid):
            sensor = TeamAgentRangeSensor(sid, self.ctx, float(radius or self.default_radius),
                                          team_of=self.team_of, team=team)
            sensor.set_owner(agent_name)
            self.ctx.sensor.add_sensor(sensor)
        return sid

    def create_flag_sensor(self, owner: Optional[str], radius: float, flag_nodes: Iterable[int]) -> str:
        """
        Create a **FlagRangeSensor** with owner=agent_name OR owner=None (static).
        - If owner is an agent, include the returned sensor_id in that agent’s sensors list.
        - If owner is None, you’ll drive it manually each tick by calling sense(anchor_node).
        """
        sid = f"flag_range:{owner or 'ownerless'}:{id(self)}"
        if not self.ctx.sensor.has_sensor(sid):
            s = FlagRangeSensor(sid, self.ctx, radius, flag_nodes)
            s.set_owner(owner)
            self.ctx.sensor.add_sensor(s)
        return sid

    def create_static_team_agent_range(self, team: str, anchor_node: int, radius: Optional[float] = None) -> str:
        """
        Create a **team-level ownerless AGENT_RANGE** anchored at a node.
        - You must call `tick_team_probes()` each tick to update its data.
        - Store `anchor_node` here; change it later by updating `_team_static_anchors`.
        """
        sid = f"team_probe:{team}"
        if not self.ctx.sensor.has_sensor(sid):
            self.ctx.sensor.create_sensor(sid, SensorType.AGENT_RANGE,
                                          sensor_range=float(radius or self.default_radius))
            self.ctx.sensor.get_sensor(sid).set_owner(None)  # important: ownerless
        self._team_static_ids[team] = sid
        self._team_static_anchors[team] = anchor_node
        return sid

    # ----- Bulk build (recommended for many agents) -----

    def build_all(
        self,
        *,
        agent_names: List[str],
        team_of: Optional[Dict[str, str]] = None,
        radius_by_agent: Optional[Dict[str, float]] = None,
        team_anchor_node: Optional[Dict[str, int]] = None,
        flag_nodes: Optional[Iterable[int]] = None,
        include_agent_ungated: bool = False,
        include_map_range: bool = True,
        include_team_sensor: bool = True,
        include_agent_range: bool = True,
        static_agent_range_radius: Optional[float] = None,
        flag_sensor_radius_by_agent: Optional[Dict[str, float]] = None,
    ) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        """
        Pre-create sensors for *all* agents and (optionally) team/static probes.

        Returns
        -------
        per_agent_ids: Dict[str, List[str]]
            For each agent, a list of concrete sensor IDs you should pass to
            `create_agent(..., sensors=per_agent_ids[name])`.

        per_team_static_ids: Dict[str, str]
            For each team, the sensor_id of the static ownerless probe (AGENT_RANGE).
            You update these each frame via `tick_team_probes()`.

        Typical call (during setup)
        ---------------------------
        per_agent_ids, per_team_ids = se.build_all(
            agent_names = ["red_0", "red_1", "blue_0", "blue_1"],
            team_of = {"red_0":"red","red_1":"red","blue_0":"blue","blue_1":"blue"},
            radius_by_agent = {"red_0": 200.0, "blue_0": 300.0},  # optional overrides
            team_anchor_node = {"red": 17, "blue": 42},           # for static probes
            flag_nodes = [101, 305, 777],                         # optional
            include_agent_ungated = False,
            include_map_range = True,
            include_team_sensor = True,
            include_agent_range = True,
            static_agent_range_radius = 300.0,
            flag_sensor_radius_by_agent = None,
        )
        """
        if team_of is not None:
            self.team_of.update(team_of)

        self._per_agent_ids = {a: [] for a in agent_names}

        # Create per-agent sensors
        for a in agent_names:
            r = (radius_by_agent or {}).get(a, self.default_radius)

            if include_map_range:
                self._per_agent_ids[a].append(self.create_map_range(a, r))

            if include_agent_ungated:
                self._per_agent_ids[a].append(self.create_agent(a))

            if include_agent_range:
                self._per_agent_ids[a].append(self.create_agent_range(a, r))

            if include_team_sensor:
                # If you know team now, bind for slightly faster lookup
                team = (team_of or {}).get(a)
                self._per_agent_ids[a].append(self.create_team_sensor(a, r, team=team))

            if flag_nodes is not None:
                fr = (flag_sensor_radius_by_agent or {}).get(a, r)
                self._per_agent_ids[a].append(self.create_flag_sensor(a, fr, flag_nodes))

        # Create per-team static (ownerless) agent-range sensors
        if team_anchor_node:
            for team, anchor in team_anchor_node.items():
                self.create_static_team_agent_range(
                    team, anchor, radius=static_agent_range_radius or self.default_radius
                )

        return self._per_agent_ids, dict(self._team_static_ids)

    # ----- Ticking shared (ownerless) probes each frame -----

    def tick_team_probes(self) -> Dict[str, Tuple[Any, Dict[str, Any]]]:
        """
        Call ONCE per simulation tick to update ownerless static team probes.
        - Returns {team: (SensorType, data)} so you can easily stash it to a shared dict.

        Typical use (per frame)
        -----------------------
        readbacks = se.tick_team_probes()
        for team, (stype, sdata) in readbacks.items():
            team_shared_state[team]["static_agent_range"] = (stype, sdata)
        """
        out: Dict[str, Tuple[Any, Dict[str, Any]]] = {}
        for team, sid in self._team_static_ids.items():
            anchor = self._team_static_anchors[team]
            s = self.ctx.sensor.get_sensor(sid)
            s.sense(anchor)               # ownerless -> you must pass the anchor
            out[team] = (s.type, s.data)
        return out


# ---------- Debug Tests ----------

if __name__ == "__main__":
    """Debug tests for sensor engine creation functionality"""
    print("=== Sensor Engine Creation Tests ===")
    
    # Mock context for testing
    class MockContext:
        def __init__(self):
            self.sensors = {}
            
            # Mock sensor interface
            self.sensor = type('obj', (object,), {
                'has_sensor': lambda sid: sid in self.sensors,
                'get_sensor': lambda sid: self.sensors.get(sid),
                'add_sensor': lambda s: self.sensors.update({s.sensor_id: s}),
                'create_sensor': lambda *args, **kwargs: None
            })()
    
    # Test 1: TeamAgentRangeSensor creation
    print("\n1. Testing TeamAgentRangeSensor creation...")
    ctx = MockContext()
    
    team_of = {"red_0": "red", "red_1": "red", "blue_0": "blue"}
    
    # Create team sensor
    team_sensor = TeamAgentRangeSensor("test_team", ctx, radius=150.0, team_of=team_of)
    team_sensor.set_owner("red_0")
    
    print(f"   Created TeamAgentRangeSensor: {team_sensor.sensor_id}")
    print(f"   Sensor type: {team_sensor.type}")
    print(f"   Initial data: {team_sensor.data}")
    assert team_sensor.sensor_id == "test_team", "Sensor ID should match"
    assert team_sensor.data == {"teammates": {}}, "Initial data should be empty teammates dict"
    print("   ✓ TeamAgentRangeSensor creation passed")
    
    # Test 2: FlagRangeSensor creation
    print("\n2. Testing FlagRangeSensor creation...")
    flag_nodes = [10, 20, 30]
    flag_sensor = FlagRangeSensor("test_flags", ctx, radius=100.0, flag_nodes=flag_nodes)
    
    print(f"   Created FlagRangeSensor: {flag_sensor.sensor_id}")
    print(f"   Sensor type: {flag_sensor.type}")
    print(f"   Initial data: {flag_sensor.data}")
    print(f"   Flag nodes configured: {flag_sensor._flags}")
    assert flag_sensor.sensor_id == "test_flags", "Sensor ID should match"
    assert flag_sensor.data == {"flags": []}, "Initial data should be empty flags list"
    assert flag_sensor._flags == flag_nodes, "Flag nodes should be stored correctly"
    print("   ✓ FlagRangeSensor creation passed")
    
    # Test 3: SensorEngineExt creation and individual sensor methods
    print("\n3. Testing SensorEngineExt creation...")
    engine = SensorEngineExt(ctx, default_radius=120.0, team_of=team_of)
    
    print(f"   Created SensorEngineExt with default radius: {engine.default_radius}")
    print(f"   Team mapping: {engine.team_of}")
    
    # Test individual sensor creation methods
    map_sensor_id = engine.create_map_range("test_agent", radius=200.0)
    agent_sensor_id = engine.create_agent("test_agent")
    agent_range_id = engine.create_agent_range("test_agent", radius=150.0)
    team_sensor_id = engine.create_team_sensor("test_agent", radius=180.0, team="red")
    flag_sensor_id = engine.create_flag_sensor("test_agent", 100.0, [1, 2, 3])
    
    print(f"   Created map range sensor: {map_sensor_id}")
    print(f"   Created agent sensor: {agent_sensor_id}")
    print(f"   Created agent range sensor: {agent_range_id}")
    print(f"   Created team sensor: {team_sensor_id}")
    print(f"   Created flag sensor: {flag_sensor_id}")
    
    # Verify sensors were added to context
    assert team_sensor_id in ctx.sensors, "Team sensor should be in context"
    assert flag_sensor_id in ctx.sensors, "Flag sensor should be in context"
    print("   ✓ Individual sensor creation passed")
    
    # Test 4: Bulk sensor creation with build_all
    print("\n4. Testing bulk sensor creation...")
    
    agent_names = ["red_0", "red_1", "blue_0", "blue_1"]
    per_agent_ids, per_team_ids = engine.build_all(
        agent_names=agent_names,
        team_of=team_of,
        team_anchor_node={"red": 1, "blue": 10},
        flag_nodes=[5, 6, 7],
        include_map_range=True,
        include_team_sensor=True,
        include_agent_range=True,
        include_agent_ungated=False
    )
    
    print(f"   Bulk creation completed for {len(agent_names)} agents")
    print(f"   Per-agent sensor counts:")
    total_sensors = 0
    for agent, sensors in per_agent_ids.items():
        print(f"     {agent}: {len(sensors)} sensors")
        total_sensors += len(sensors)
        assert len(sensors) > 0, f"Agent {agent} should have sensors"
    
    print(f"   Total sensors created: {total_sensors}")
    print(f"   Team static sensors: {per_team_ids}")
    
    # Verify expected sensor types
    sample_agent_sensors = per_agent_ids[agent_names[0]]
    sensor_types = [s.split(':')[0] for s in sample_agent_sensors]
    expected_types = ['map_range', 'agent_range', 'team_range', 'flag_range']
    
    for expected_type in expected_types:
        assert any(expected_type in sensor_type for sensor_type in sensor_types), \
            f"Should have created {expected_type} sensor"
    
    print("   ✓ Bulk sensor creation passed")
    
    # Test 5: Sensor configuration validation
    print("\n5. Testing sensor configuration...")
    
    # Test radius customization
    custom_radius_ids, _ = engine.build_all(
        agent_names=["custom_agent"],
        radius_by_agent={"custom_agent": 300.0},
        include_map_range=True,
        include_agent_range=False,
        include_team_sensor=False
    )
    
    print(f"   Custom radius configuration: {custom_radius_ids}")
    assert len(custom_radius_ids["custom_agent"]) == 1, "Should create only map_range sensor"
    
    # Test flag sensor customization
    flag_custom_ids, _ = engine.build_all(
        agent_names=["flag_agent"],
        flag_nodes=[100, 200, 300],
        flag_sensor_radius_by_agent={"flag_agent": 250.0},
        include_map_range=False,
        include_agent_range=False,
        include_team_sensor=False
    )
    
    flag_sensor_created = any("flag_range" in sid for sid in flag_custom_ids["flag_agent"])
    assert flag_sensor_created, "Should create flag sensor"
    print("   ✓ Sensor configuration passed")
    
    print("\n🎉 All sensor creation tests passed!")
    print("\nCreated sensor types:")
    print("- TeamAgentRangeSensor: For detecting teammates within radius")
    print("- FlagRangeSensor: For detecting flags within radius") 
    print("- SensorEngineExt: For bulk sensor management")
    print(f"\nTotal sensors in context: {len(ctx.sensors)}")
    print("✓ Sensor engine ready for integration!")
