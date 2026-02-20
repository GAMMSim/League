from gamms.SensorEngine import SensorType
from gamms.typing import ISensor

def create_stationary_sensor_class(ctx):
    """Factory function to create StationarySensor class with the given context."""
    
    @ctx.sensor.custom("STATIONARY_SENSOR")
    class StationarySensor(ISensor):
        """Fixed position sensor that detects agents within range of a specific node."""
        
        def __init__(self, ctx, sensor_id, fixed_node_id, sensor_range, nx_graph=None):
            self._ctx = ctx
            self._sensor_id = sensor_id
            self._owner = None  # No owner - shared by team
            self._fixed_node_id = fixed_node_id
            self._sensor_range = sensor_range
            self._nx_graph = nx_graph
            self._data = {}
            self._coverage_cache_key = None
            self._covered_nodes_cache = []
        
        @property
        def sensor_id(self):
            return self._sensor_id
        
        @property
        def type(self):
            return SensorType.STATIONARY_SENSOR
        
        @property
        def data(self):
            return self._data
        
        def set_owner(self, owner):
            # This sensor doesn't have an owner - it's shared by the team
            pass

        def _fixed_position_xy(self):
            """Return (x, y) for the fixed node using available graph sources."""
            if self._nx_graph is not None and self._fixed_node_id in self._nx_graph.nodes:
                node_data = self._nx_graph.nodes[self._fixed_node_id]
                if "x" in node_data and "y" in node_data:
                    return node_data["x"], node_data["y"]

            fixed_node = self._ctx.graph.graph.get_node(self._fixed_node_id)
            return fixed_node.x, fixed_node.y

        def _compute_covered_nodes(self):
            """Compute all graph nodes inside this stationary sensor's range."""
            if self._nx_graph is None:
                return []

            fixed_x, fixed_y = self._fixed_position_xy()
            covered = []
            for node_id, node_data in self._nx_graph.nodes(data=True):
                node_x = node_data.get("x")
                node_y = node_data.get("y")
                if node_x is None or node_y is None:
                    continue
                dx = node_x - fixed_x
                dy = node_y - fixed_y
                if (dx**2 + dy**2) ** 0.5 <= self._sensor_range:
                    covered.append(node_id)
            return covered

        def _get_covered_nodes(self):
            """Return cached covered nodes unless fixed position/range changed."""
            key = (self._fixed_node_id, float(self._sensor_range))
            if key != self._coverage_cache_key:
                self._covered_nodes_cache = self._compute_covered_nodes()
                self._coverage_cache_key = key
            return self._covered_nodes_cache
        
        def sense(self, node_id: int):
            """Detect agents within range of the fixed position (ignores node_id parameter)."""
            fixed_x, fixed_y = self._fixed_position_xy()
            covered_nodes = self._get_covered_nodes()
            
            detected_agents = {}
            
            # Check all agents in the system
            for agent in self._ctx.agent.create_iter():
                agent_node = self._ctx.graph.graph.get_node(agent.current_node_id)
                
                # Calculate distance from fixed position to agent
                dx = agent_node.x - fixed_x
                dy = agent_node.y - fixed_y
                distance = (dx**2 + dy**2) ** 0.5
                
                # If agent is within range, add to detected list
                if distance <= self._sensor_range:
                    detected_agents[agent.name] = {
                        'node_id': agent.current_node_id,
                        'distance': distance
                    }
            
            self._data = {
                'fixed_position': self._fixed_node_id,
                'detected_agents': detected_agents,
                'agent_count': len(detected_agents),
                'covered_nodes': covered_nodes,
                'covered_node_count': len(covered_nodes),
            }
        
        def update(self, data):
            # Allow updating sensor range or fixed position if needed
            if 'sensor_range' in data:
                self._sensor_range = data['sensor_range']
            if 'fixed_node_id' in data:
                self._fixed_node_id = data['fixed_node_id']
            self._coverage_cache_key = None
            self._covered_nodes_cache = []
    
    return StationarySensor
