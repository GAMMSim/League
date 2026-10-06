from typing import Dict, List, Any, Optional, Set, Tuple
from typeguard import typechecked
import networkx as nx

from lib.core.console import *

# Import custom sensor factory functions
from lib.sensor.info_sensor import (
    create_global_map_sensor_class,
    create_candidate_flag_sensor_class,
    create_flag_sensor_class,
)
from lib.sensor.flag_range_sensor import create_flag_range_sensor_class
from lib.sensor.stationary_sensor import create_stationary_sensor_class
from lib.sensor.team_sensor import create_team_agent_sensor
from lib.sensor.base_sensor import DYNAMIC
from lib.sensor.region_sensor import create_region_sensor_class
from lib.sensor.region_view_sensor import create_region_view_sensor_class
from lib.core.visibility_cache import get_visibility_models


# Process-global cache to avoid re-registering identical custom sensor types
# on every new game context in the same Python process.
_CUSTOM_SENSOR_CLASS_CACHE: Dict[str, Any] = {}


@typechecked
class SensorEngine:
    """Manages sensor creation and registration for agents"""

    def __init__(self, ctx: Any, config: Dict[str, Any], graph: nx.Graph):
        """
        Initialize SensorEngine
        
        Args:
            ctx: GAMMS context
            config: Full game configuration
            graph: NetworkX graph of game environment
        """
        self.ctx = ctx
        self.config = config
        self.graph = graph
        
        # Track created sensors to avoid duplicates
        self.created_sensors: Dict[str, Any] = {}

        # Visibility models (loaded in _create_shared_sensors). Static per graph.
        self._vis_models: Dict[str, Any] = {}
        self._default_vis_model: Optional[str] = None
        self._vis_dir = "graphs/visibility"

        # Logical sensor names whose payloads are {agent_name: node_id} dicts
        # and should be pre-split into enemies/teammates by the game engine.
        self.agent_type_sensor_names: Set[str] = set()

        # Logical sensor names backed by RegionSensor — game_engine splits each
        # one's "detected_agents" into enemies/teammates in place (payload keeps
        # region/model/detected_flags too, unlike agent_type_sensor_names which
        # replaces the whole payload).
        self.region_sensor_names: Set[str] = set()

        # {agent_name: [RegionSensor, ...]} — the model-backed sensors each agent
        # detects real flags with (entries declaring `flags: real`). Flag
        # discovery reads these so it credits exactly what the sensor reports.
        self._flag_sensors: Dict[str, List[Any]] = {}

        # Shared by every RegionViewSensor (`<name>_region`) in this game: the
        # gamms node/edge objects (filled on first use), and one {origin: view}
        # cache per visibility model.
        self._graph_index: Dict[str, Any] = {}
        self._region_view_caches: Dict[str, Dict[int, Any]] = {}
        
        # Register custom sensor classes
        self._register_custom_sensor_classes()
        
        # Create shared sensors that all agents can use
        self._create_shared_sensors()
        
        debug("SensorEngine initialized")

    def _register_custom_sensor_classes(self) -> None:
        """Register all custom sensor factory classes with ctx"""
        debug("Registering custom sensor classes")
        global _CUSTOM_SENSOR_CLASS_CACHE

        if _CUSTOM_SENSOR_CLASS_CACHE:
            self.GlobalMapSensor = _CUSTOM_SENSOR_CLASS_CACHE["GlobalMapSensor"]
            self.CandidateFlagSensor = _CUSTOM_SENSOR_CLASS_CACHE["CandidateFlagSensor"]
            self.FlagSensor = _CUSTOM_SENSOR_CLASS_CACHE["FlagSensor"]
            self.FlagRangeSensor = _CUSTOM_SENSOR_CLASS_CACHE["FlagRangeSensor"]
            self.StationarySensor = _CUSTOM_SENSOR_CLASS_CACHE["StationarySensor"]
            self.TeamAgentSensor = _CUSTOM_SENSOR_CLASS_CACHE["TeamAgentSensor"]
            self.RegionSensor = _CUSTOM_SENSOR_CLASS_CACHE["RegionSensor"]
            self.RegionViewSensor = _CUSTOM_SENSOR_CLASS_CACHE["RegionViewSensor"]
            debug("Reusing cached custom sensor classes")
            return

        # First registration in this process.
        self.GlobalMapSensor = create_global_map_sensor_class(self.ctx)
        self.CandidateFlagSensor = create_candidate_flag_sensor_class(self.ctx)
        self.FlagSensor = create_flag_sensor_class(self.ctx)
        self.FlagRangeSensor = create_flag_range_sensor_class(self.ctx)
        self.StationarySensor = create_stationary_sensor_class(self.ctx)
        self.TeamAgentSensor = create_team_agent_sensor(self.ctx)
        self.RegionSensor = create_region_sensor_class(self.ctx)
        self.RegionViewSensor = create_region_view_sensor_class(self.ctx)

        _CUSTOM_SENSOR_CLASS_CACHE = {
            "GlobalMapSensor": self.GlobalMapSensor,
            "CandidateFlagSensor": self.CandidateFlagSensor,
            "FlagSensor": self.FlagSensor,
            "FlagRangeSensor": self.FlagRangeSensor,
            "StationarySensor": self.StationarySensor,
            "TeamAgentSensor": self.TeamAgentSensor,
            "RegionSensor": self.RegionSensor,
            "RegionViewSensor": self.RegionViewSensor,
        }

        success("Custom sensor classes registered")

    def _create_shared_sensors(self) -> None:
        """Create sensors that are shared across all agents"""
        debug("Creating shared sensors")
        
        # Extract flag configuration
        flag_config = self.config.get("flags", {})
        real_flags = flag_config.get("real_positions", [])
        candidate_flags = flag_config.get("candidate_positions", [])
        
        # 1-3. Info sensors (global_map / candidate_flag / flag): static payload,
        #    by reference, identical content regardless of reader. `team` is
        #    still required on every sensor (never None/universal, see
        #    docs/sensor_redesign_handoff.md §2) so a sensor read by both teams
        #    is one instance per declaring team block, not one shared instance.
        for t in ("red", "blue"):
            try:
                sensor_id = f"global_map_shared_{t}"
                global_map_sensor = self.GlobalMapSensor(
                    ctx=self.ctx, sensor_id=sensor_id, nx_graph=self.graph, team=t,
                )
                self.ctx.sensor.add_sensor(global_map_sensor)
                self.created_sensors[sensor_id] = global_map_sensor
            except Exception as e:
                warning(f"Failed to create global_map sensor for {t}: {e}")

            try:
                sensor_id = f"candidate_flag_shared_{t}"
                candidate_flag_sensor = self.CandidateFlagSensor(
                    ctx=self.ctx, sensor_id=sensor_id, candidate_flags=candidate_flags, team=t,
                )
                self.ctx.sensor.add_sensor(candidate_flag_sensor)
                self.created_sensors[sensor_id] = candidate_flag_sensor
            except Exception as e:
                warning(f"Failed to create candidate_flag sensor for {t}: {e}")

            try:
                sensor_id = f"flag_shared_{t}"
                flag_sensor = self.FlagSensor(
                    ctx=self.ctx, sensor_id=sensor_id, real_flags=real_flags,
                    candidate_flags=candidate_flags, team=t,
                )
                self.ctx.sensor.add_sensor(flag_sensor)
                self.created_sensors[sensor_id] = flag_sensor
            except Exception as e:
                warning(f"Failed to create flag sensor for {t}: {e}")
        debug("Created shared info sensors (global_map/candidate_flag/flag) for red+blue")
        
        # 4. Visibility models (static per graph, load-or-build from generator
        #    specs in environment.visibility_models). Looked up by name from
        #    _create_unified_sensor when an agent declares a region sensor
        #    entry {model: <name>, ...}. No standalone sensor exposes the
        #    tables wholesale (see docs/sensor_redesign_handoff.md §2).
        env_config = self.config.get("environment", {})
        try:
            import os
            model_specs = env_config.get("visibility_models", {})
            self._default_vis_model = env_config.get("default_visibility_model")
            source_path = self.graph.graph.get("__graph_source_path") if hasattr(self.graph, "graph") else None
            base_dir = os.path.dirname(source_path) if source_path else "graphs"
            self._vis_models = get_visibility_models(
                self.graph, model_specs, base_dir, vis_dir=self._vis_dir
            )
            if self._vis_models:
                debug(f"Loaded {len(self._vis_models)} visibility model(s)")
        except Exception as e:
            warning(f"Failed to set up visibility models: {e}")

        # 5. Create stationary sensors for blue team
        stationary_radius = env_config.get("blue_stationary_sensor_radius", 6.0)
        stationary_positions = env_config.get("blue_static_sensor_positions", [])
        
        for idx, node_id in enumerate(stationary_positions):
            try:
                stationary_sensor = self.StationarySensor(
                    ctx=self.ctx,
                    sensor_id=f"stationary_{idx}",
                    fixed_node_id=node_id,
                    sensor_range=stationary_radius,
                    nx_graph=self.graph,
                )
                self.ctx.sensor.add_sensor(stationary_sensor)
                self.created_sensors[f"stationary_{idx}"] = stationary_sensor
                debug(f"Created stationary sensor at node {node_id}")
            except Exception as e:
                warning(f"Failed to create stationary sensor {idx}: {e}")
        
        success(f"Created {len(self.created_sensors)} shared sensors")

    def create_sensors_for_agent(
        self, 
        agent_name: str, 
        team: str,
        sensor_list: List[Any],
        sensing_radius: Optional[float] = None
    ) -> Dict[str, str]:
        """
        Create and register sensors for a specific agent.

        Each item in sensor_list may be either:
          - a string  -> legacy named sensor (backward compatible)
          - a dict     -> unified region-sensor entry, e.g.
              {name: eye, model: r30, carrier: agent|<node_id>|"node:<id>",
               team: red|blue, flags: real}
            `model` names an entry under environment.visibility_models (radius/
            k-hop/line-of-sight are just generators that build its table — see
            lib/core/visibility_generators.py).
          - a dict with `at:` -> fans out into N node-carrier entries, one per
              position in the list (or `environment.<key>` list) `at` names, e.g.
              {model: blue_tower_r450, at: blue_static_sensor_positions}
            Lets a static rule template declare "one tower per position in
            this env list" without knowing the count/positions up front —
            same trick the legacy `stationary` sentinel uses internally.

          - a string `<name>_region` -> coverage view of sensor `<name>`: the
              local subgraph {nodes, edges} that sensor covers right now (see
              lib/sensor/region_view_sensor.py). Resolved after every other
              entry, so its position in the list does not matter. If `<name>`
              is not a RegionSensor (the plain-string forms of rule v1.2) it
              falls back to the gamms euclidean RANGE sensor. Rule v1.3 still
              declares these; from v1.4 the coverage is read off the sensor
              itself (`region` / `table`) and no `_region` entry is listed.

        Every RegionSensor payload also carries the model's FULL node->visible-
        nodes table under `"table"` (see lib/sensor/region_sensor.py), alongside
        `"region"` (just the current origin's slice) — same sensor, same name,
        no separate whole-graph sensor to declare or learn.

        Returns:
            Dict mapping logical sensor names to their IDs for registration
        """
        debug(f"Creating sensors for agent {agent_name}")

        sensor_mappings: Dict[str, str] = {}

        def _add_mapping(logical_name: str, sensor_id: str) -> None:
            # Two config entries reusing the same logical name would otherwise
            # silently clobber each other here (only the last survives into
            # state["sensor"]).
            existing = sensor_mappings.get(logical_name)
            if existing is not None and existing != sensor_id:
                warning(
                    f"Sensor entry '{logical_name}' for {agent_name} overwrites "
                    f"an earlier sensor with the same logical name ('{existing}' "
                    f"-> '{sensor_id}') — give each entry a distinct 'name:'"
                )
            sensor_mappings[logical_name] = sensor_id

        # `<name>_region` views read their companion's table, so resolve them last.
        def _is_view(entry: Any) -> bool:
            return isinstance(entry, str) and entry.endswith("_region")

        ordered = [e for e in sensor_list if not _is_view(e)] + [e for e in sensor_list if _is_view(e)]

        for entry in ordered:
            try:
                if isinstance(entry, dict) and "at" in entry:
                    for sub_entry in self._expand_at_entry(entry):
                        logical_name, sensor_id = self._create_unified_sensor(
                            agent_name, team, sub_entry
                        )
                        if sensor_id:
                            _add_mapping(logical_name, sensor_id)
                elif isinstance(entry, dict):
                    logical_name, sensor_id = self._create_unified_sensor(
                        agent_name, team, entry
                    )
                    if sensor_id:
                        _add_mapping(logical_name, sensor_id)
                else:
                    logical_name = entry
                    sensor_id = None
                    if _is_view(entry):
                        sensor_id = self._create_region_view(agent_name, entry, sensor_mappings)
                    if sensor_id is None:
                        sensor_id = self._create_single_sensor(
                            agent_name, team, entry, sensing_radius
                        )
                    if sensor_id:
                        _add_mapping(logical_name, sensor_id)
            except Exception as e:
                warning(f"Failed to create sensor '{entry}' for {agent_name}: {e}")
                continue

        info(f"Created {len(sensor_mappings)} sensors for {agent_name}")
        return sensor_mappings

    def _resolve_at(self, at_spec: Any) -> List[int]:
        """
        Resolve an `at:` spec to a list of node IDs: either an inline list, or
        a string naming an environment.<key> that holds one (e.g. the same
        `blue_static_sensor_positions` the legacy `stationary` sentinel reads).
        """
        if isinstance(at_spec, list):
            return [int(x) for x in at_spec]
        if isinstance(at_spec, str):
            positions = self.config.get("environment", {}).get(at_spec) or []
            return [int(x) for x in positions]
        raise ValueError(
            f"Unsupported 'at' spec: {at_spec!r} (expected a list of node ids "
            "or a string naming an environment.<key>)"
        )

    def _expand_at_entry(self, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Fan out one {model, at, ...} entry into N node-carrier entries, one per
        position — so a rule template can declare "a tower per position in
        this env list" without knowing the count/positions up front (mirrors
        how the legacy `stationary` sentinel lets the engine, not the config,
        do the counting). `name` (default: model name) becomes `<name>_<idx>`
        per position; any `carrier` on the source entry is ignored — `at`
        supplies it.
        """
        positions = self._resolve_at(entry["at"])
        base_name = entry.get("name") or entry.get("model") or "sensor"
        expanded = []
        for idx, node_id in enumerate(positions):
            sub_entry = {k: v for k, v in entry.items() if k not in ("at", "carrier")}
            sub_entry["name"] = f"{base_name}_{idx}"
            sub_entry["carrier"] = f"node:{node_id}"
            expanded.append(sub_entry)
        return expanded

    def _resolve_carrier(self, carrier_spec: Any) -> Optional[int]:
        """
        Map a config carrier value to a carrier: an int node_id (STATIC) or
        DYNAMIC (None) for agent-carried. Accepts: "agent"/None -> DYNAMIC,
        an int, or "node:<id>".
        """
        if carrier_spec is None or carrier_spec == "agent":
            return DYNAMIC
        if isinstance(carrier_spec, int):
            return carrier_spec
        if isinstance(carrier_spec, str) and carrier_spec.startswith("node:"):
            return int(carrier_spec.split(":", 1)[1])
        return int(carrier_spec)

    def _create_unified_sensor(
        self,
        agent_name: str,
        team: str,
        entry: Dict[str, Any],
    ) -> Tuple[str, Optional[str]]:
        """
        Create a region sensor from a unified config dict. `model` names a
        precomputed table under environment.visibility_models — radius, k-hop,
        and line-of-sight are all just generators that build one, so there is
        a single region-sensor path regardless of which generator built the
        table (see lib/core/visibility_generators.py). The payload carries
        both `region` (current origin's slice) and `table` (the full model) —
        see lib/sensor/region_sensor.py.
        """
        model = entry.get("model")
        table = self._vis_models.get(model) if model else None
        if not table:
            warning(
                f"Sensor entry for {agent_name} names unknown visibility model "
                f"'{model}' (declare it under environment.visibility_models)"
            )
            return entry.get("name") or model or "unnamed", None

        logical_name = entry.get("name") or model
        owner_team = entry.get("team", team)  # team that owns/reads it
        carrier = self._resolve_carrier(entry.get("carrier", "agent"))
        is_static = carrier is not DYNAMIC

        flags = None
        if entry.get("flags") == "real":
            flags = self.config.get("flags", {}).get("real_positions", [])

        # Static sensors are shared/deduped by identity; dynamic are per-agent.
        sensor_id = (
            f"u_region_{owner_team}_{model}_{carrier}"
            if is_static else f"{agent_name}_u_{logical_name}"
        )
        if sensor_id not in self.created_sensors:
            sensor = self.RegionSensor(
                self.ctx, sensor_id, table=table, model_name=model,
                team=owner_team, carrier=carrier, flags=flags,
            )
            self.ctx.sensor.add_sensor(sensor)
            self.created_sensors[sensor_id] = sensor
        if flags is not None:
            agent_flag_sensors = self._flag_sensors.setdefault(agent_name, [])
            if self.created_sensors[sensor_id] not in agent_flag_sensors:
                agent_flag_sensors.append(self.created_sensors[sensor_id])
        self.region_sensor_names.add(logical_name)
        return logical_name, sensor_id

    def _create_region_view(
        self,
        agent_name: str,
        sensor_name: str,
        sensor_mappings: Dict[str, str],
    ) -> Optional[str]:
        """
        Create the coverage view for a `<name>_region` entry from the
        RegionSensor this agent carries under `<name>`. Returns None when there
        is no such companion, so the caller can fall back to the legacy sensor.
        """
        companion_id = sensor_mappings.get(sensor_name[: -len("_region")])
        companion = self.created_sensors.get(companion_id) if companion_id else None
        if not isinstance(companion, self.RegionSensor):
            return None

        carrier = companion._carrier
        model = companion._model
        # Static views are shared/deduped by identity; dynamic are per-agent.
        sensor_id = (
            f"u_view_{companion.team}_{model}_{carrier}"
            if companion.is_static else f"{agent_name}_{sensor_name}"
        )
        if sensor_id not in self.created_sensors:
            sensor = self.RegionViewSensor(
                self.ctx, sensor_id, table=companion._table,
                graph_index=self._graph_index,
                view_cache=self._region_view_caches.setdefault(str(model), {}),
                team=companion.team, carrier=carrier,
            )
            self.ctx.sensor.add_sensor(sensor)
            self.created_sensors[sensor_id] = sensor
        return sensor_id

    def _create_single_sensor(
        self,
        agent_name: str,
        team: str,
        sensor_name: str,
        sensing_radius: Optional[float]
    ) -> Optional[str]:
        """
        Create a single sensor instance
        
        Returns:
            Sensor ID if successful, None otherwise
        """
        
        # Handle shared sensors (one instance per declaring team, see
        # docs/sensor_redesign_handoff.md §2 — team is never None/universal)
        if sensor_name == "global_map":
            return f"global_map_shared_{team}"

        if sensor_name == "candidate_flag":
            return f"candidate_flag_shared_{team}"

        if sensor_name == "flag":
            return f"flag_shared_{team}"
        
        # Handle built-in GAMMS sensors
        if sensor_name == "agent":
            from gamms.SensorEngine import SensorType
            sensor_id = f"{agent_name}_agent"
            sensor = self.ctx.sensor.create_sensor(
                sensor_id=sensor_id,
                sensor_type=SensorType.AGENT
            )
            self.created_sensors[sensor_id] = sensor
            self.agent_type_sensor_names.add(sensor_name)
            return sensor_id

        if sensor_name == "egocentric_agent":
            from gamms.SensorEngine import SensorType
            sensor_id = f"{agent_name}_egocentric_agent"
            sensor = self.ctx.sensor.create_sensor(
                sensor_id=sensor_id,
                sensor_type=SensorType.AGENT_RANGE,
                sensor_range=sensing_radius or 5.0
            )
            self.created_sensors[sensor_id] = sensor
            self.agent_type_sensor_names.add(sensor_name)
            return sensor_id
        
        if sensor_name.endswith("_region"):
            from gamms.SensorEngine import SensorType
            sensor_id = f"{agent_name}_{sensor_name}"
            sensor = self.ctx.sensor.create_sensor(
                sensor_id=sensor_id,
                sensor_type=SensorType.RANGE,
                sensor_range=sensing_radius or 5.0
            )
            self.created_sensors[sensor_id] = sensor
            return sensor_id
        
        if sensor_name == "egocentric_flag":
            # Custom flag range sensor
            flag_config = self.config.get("flags", {})
            real_flags = flag_config.get("real_positions", [])
            
            sensor_id = f"{agent_name}_egocentric_flag"
            sensor = self.FlagRangeSensor(
                ctx=self.ctx,
                sensor_id=sensor_id,
                real_flags=real_flags,
                sensor_range=sensing_radius or 5.0
            )
            self.ctx.sensor.add_sensor(sensor)
            self.created_sensors[sensor_id] = sensor
            return sensor_id
        
        # Handle custom team sensor
        if sensor_name == "custom_team":
            sensor_id = f"{agent_name}_custom_team"
            sensor = self.TeamAgentSensor(
                ctx=self.ctx,
                sensor_id=sensor_id
            )
            self.ctx.sensor.add_sensor(sensor)
            self.created_sensors[sensor_id] = sensor
            return sensor_id
        
        # Handle stationary sensors (blue team only)
        if sensor_name == "stationary":
            if team != "blue":
                warning(f"Stationary sensor only for blue team, skipping for {agent_name}")
                return None
            
            # Return all stationary sensor IDs as a list (special case)
            # We'll handle this in register_sensors_to_agent
            return "stationary_all"
        
        warning(f"Unknown sensor type: {sensor_name}")
        return None

    def register_sensors_to_agent(
        self,
        agent: Any,
        sensor_mappings: Dict[str, str]
    ) -> None:
        """
        Register created sensors to an agent
        
        Args:
            agent: AgentController instance
            sensor_mappings: Dict from create_sensors_for_agent
        """
        gamms_agent = agent.gamms_agent
        
        for sensor_name, sensor_id in sensor_mappings.items():
            try:
                # Special handling for stationary sensors
                if sensor_id == "stationary_all":
                    # Register all stationary sensors
                    for sid, sensor in self.created_sensors.items():
                        if sid.startswith("stationary_"):
                            gamms_agent.register_sensor(sid, sensor)
                            debug(f"Registered {sid} to {agent.name}")
                    continue
                
                # Normal sensor registration
                sensor = self.ctx.sensor.get_sensor(sensor_id)
                gamms_agent.register_sensor(sensor_name, sensor)
                debug(f"Registered {sensor_name} ({sensor_id}) to {agent.name}")
                
            except Exception as e:
                warning(f"Failed to register {sensor_name} to {agent.name}: {e}")

    def flag_sensing_region(self, agent_name: str, node_id: int) -> Optional[Set[int]]:
        """
        Nodes in which `agent_name`'s flag sensors would detect a real flag
        with the agent standing at `node_id` — a lookup in each sensor's own
        table, so it is occlusion-limited exactly when the sensor is.

        Returns None if the agent has no model-backed flag sensor (rule v1.2's
        plain-string `egocentric_flag`), in which case range is whatever
        `sensing_radius` says.
        """
        sensors = self._flag_sensors.get(agent_name)
        if not sensors:
            return None
        region: Set[int] = set()
        for sensor in sensors:
            origin = sensor._carrier if sensor.is_static else node_id
            region.update(sensor._table.get(origin, (origin,)))
        return region

    def get_sensor_info(self) -> Dict[str, Any]:
        """Get information about all created sensors"""
        return {
            "total_sensors": len(self.created_sensors),
            "sensor_ids": list(self.created_sensors.keys()),
        }
