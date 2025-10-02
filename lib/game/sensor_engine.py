from typing import Dict, List, Any, Optional, Set
from typeguard import typechecked
import networkx as nx

from lib.core.console import *

# Import custom sensor factory functions
from lib.sensor.global_map_sensor import create_global_map_sensor_class
from lib.sensor.candidate_flag_sensor import create_candidate_flag_sensor_class
from lib.sensor.flag_sensor import create_flag_sensor_class
from lib.sensor.flag_range_sensor import create_flag_range_sensor_class
from lib.sensor.stationary_sensor import create_stationary_sensor_class
from lib.sensor.team_sensor import create_team_agent_sensor


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
        
        # Register custom sensor classes
        self._register_custom_sensor_classes()
        
        # Create shared sensors that all agents can use
        self._create_shared_sensors()
        
        debug("SensorEngine initialized")

    def _register_custom_sensor_classes(self) -> None:
        """Register all custom sensor factory classes with ctx"""
        debug("Registering custom sensor classes")
        
        # Create sensor classes with ctx binding
        self.GlobalMapSensor = create_global_map_sensor_class(self.ctx)
        self.CandidateFlagSensor = create_candidate_flag_sensor_class(self.ctx)
        self.FlagSensor = create_flag_sensor_class(self.ctx)
        self.FlagRangeSensor = create_flag_range_sensor_class(self.ctx)
        self.StationarySensor = create_stationary_sensor_class(self.ctx)
        self.TeamAgentSensor = create_team_agent_sensor(self.ctx)
        
        success("Custom sensor classes registered")

    def _create_shared_sensors(self) -> None:
        """Create sensors that are shared across all agents"""
        debug("Creating shared sensors")
        
        # Extract flag configuration
        flag_config = self.config.get("flags", {})
        real_flags = flag_config.get("real_positions", [])
        candidate_flags = flag_config.get("candidate_positions", [])
        
        # 1. Global map sensor (shared by all)
        try:
            global_map_sensor = self.GlobalMapSensor(
                ctx=self.ctx,
                sensor_id="global_map_shared",
                nx_graph=self.graph
            )
            self.ctx.sensor.add_sensor(global_map_sensor)
            self.created_sensors["global_map_shared"] = global_map_sensor
            debug("Created shared global_map sensor")
        except Exception as e:
            warning(f"Failed to create global_map sensor: {e}")
        
        # 2. Candidate flag sensor (shared by all red agents)
        try:
            candidate_flag_sensor = self.CandidateFlagSensor(
                ctx=self.ctx,
                sensor_id="candidate_flag_shared",
                candidate_flags=candidate_flags
            )
            self.ctx.sensor.add_sensor(candidate_flag_sensor)
            self.created_sensors["candidate_flag_shared"] = candidate_flag_sensor
            debug("Created shared candidate_flag sensor")
        except Exception as e:
            warning(f"Failed to create candidate_flag sensor: {e}")
        
        # 3. Flag sensor (real + fake flags, shared by blue team)
        try:
            flag_sensor = self.FlagSensor(
                ctx=self.ctx,
                sensor_id="flag_shared",
                real_flags=real_flags,
                candidate_flags=candidate_flags
            )
            self.ctx.sensor.add_sensor(flag_sensor)
            self.created_sensors["flag_shared"] = flag_sensor
            debug("Created shared flag sensor")
        except Exception as e:
            warning(f"Failed to create flag sensor: {e}")
        
        # 4. Create stationary sensors for blue team
        env_config = self.config.get("environment", {})
        stationary_radius = env_config.get("blue_stationary_sensor_radius", 6.0)
        stationary_positions = env_config.get("blue_static_sensor_positions", [])
        
        for idx, node_id in enumerate(stationary_positions):
            try:
                stationary_sensor = self.StationarySensor(
                    ctx=self.ctx,
                    sensor_id=f"stationary_{idx}",
                    fixed_node_id=node_id,
                    sensor_range=stationary_radius
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
        sensor_list: List[str], 
        sensing_radius: Optional[float] = None
    ) -> Dict[str, str]:
        """
        Create and register sensors for a specific agent
        
        Args:
            agent_name: Name of the agent
            team: Team name (red/blue)
            sensor_list: List of sensor names from config
            sensing_radius: Radius for egocentric sensors
            
        Returns:
            Dict mapping sensor names to their IDs for registration
        """
        debug(f"Creating sensors for agent {agent_name}")
        
        sensor_mappings: Dict[str, str] = {}
        
        for sensor_name in sensor_list:
            try:
                sensor_id = self._create_single_sensor(
                    agent_name, team, sensor_name, sensing_radius
                )
                if sensor_id:
                    sensor_mappings[sensor_name] = sensor_id
            except Exception as e:
                warning(f"Failed to create sensor '{sensor_name}' for {agent_name}: {e}")
                continue
        
        info(f"Created {len(sensor_mappings)} sensors for {agent_name}")
        return sensor_mappings

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
        
        # Handle shared sensors
        if sensor_name == "global_map":
            return "global_map_shared"
        
        if sensor_name == "candidate_flag":
            return "candidate_flag_shared"
        
        if sensor_name == "flag":
            return "flag_shared"
        
        # Handle built-in GAMMS sensors
        if sensor_name == "agent":
            from gamms.SensorEngine import SensorType
            sensor_id = f"{agent_name}_agent"
            sensor = self.ctx.sensor.create_sensor(
                sensor_id=sensor_id,
                sensor_type=SensorType.AGENT
            )
            self.created_sensors[sensor_id] = sensor
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
            return sensor_id
        
        if sensor_name == "egocentric_map":
            from gamms.SensorEngine import SensorType
            sensor_id = f"{agent_name}_egocentric_map"
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

    def get_sensor_info(self) -> Dict[str, Any]:
        """Get information about all created sensors"""
        return {
            "total_sensors": len(self.created_sensors),
            "sensor_ids": list(self.created_sensors.keys()),
        }