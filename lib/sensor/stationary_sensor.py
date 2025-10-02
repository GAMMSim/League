from gamms.SensorEngine import SensorType
from gamms.typing import ISensor

def create_stationary_sensor_class(ctx):
    """Factory function to create StationarySensor class with the given context."""
    
    @ctx.sensor.custom("STATIONARY_SENSOR")
    class StationarySensor(ISensor):
        """Fixed position sensor that detects agents within range of a specific node."""
        
        def __init__(self, ctx, sensor_id, fixed_node_id, sensor_range):
            self._ctx = ctx
            self._sensor_id = sensor_id
            self._owner = None  # No owner - shared by team
            self._fixed_node_id = fixed_node_id
            self._sensor_range = sensor_range
            self._data = {}
        
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
        
        def sense(self, node_id: int):
            """Detect agents within range of the fixed position (ignores node_id parameter)."""
            # Get the fixed position
            fixed_node = self._ctx.graph.graph.get_node(self._fixed_node_id)
            
            detected_agents = {}
            
            # Check all agents in the system
            for agent in self._ctx.agent.create_iter():
                agent_node = self._ctx.graph.graph.get_node(agent.current_node_id)
                
                # Calculate distance from fixed position to agent
                dx = agent_node.x - fixed_node.x
                dy = agent_node.y - fixed_node.y
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
                'agent_count': len(detected_agents)
            }
        
        def update(self, data):
            # Allow updating sensor range or fixed position if needed
            if 'sensor_range' in data:
                self._sensor_range = data['sensor_range']
            if 'fixed_node_id' in data:
                self._fixed_node_id = data['fixed_node_id']
    
    return StationarySensor