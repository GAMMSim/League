from gamms.SensorEngine import SensorType
from gamms.typing import ISensor

def create_flag_range_sensor_class(ctx):
    """Factory function to create FlagRangeSensor class with the given context."""
    
    @ctx.sensor.custom("FLAG_RANGE_SENSOR")
    class FlagRangeSensor(ISensor):
        """Detects real flags within a specified range."""
        
        def __init__(self, ctx, sensor_id, real_flags, sensor_range):
            """
            Args:
                ctx: GAMMS context
                sensor_id: Unique sensor identifier
                real_flags: List of real flag node IDs [1, 3, 5, 7]
                sensor_range: Maximum detection distance for flags
            """
            self._ctx = ctx
            self._sensor_id = sensor_id
            self._owner = None
            self._real_flags = real_flags
            self._sensor_range = sensor_range
            self._data = {}
        
        @property
        def sensor_id(self):
            return self._sensor_id
        
        @property
        def type(self):
            return SensorType.FLAG_RANGE_SENSOR
        
        @property
        def data(self):
            return self._data
        
        def set_owner(self, owner):
            self._owner = owner
        
        def sense(self, node_id: int):
            """Detect real flags within sensor range of the current position."""
            current_node = self._ctx.graph.graph.get_node(node_id)
            detected_flags = []
            
            # Check each real flag (flag is located at its corresponding node ID)
            for flag_node_id in self._real_flags:
                flag_node = self._ctx.graph.graph.get_node(flag_node_id)
                
                # Calculate distance from agent to flag
                dx = flag_node.x - current_node.x
                dy = flag_node.y - current_node.y
                distance = (dx**2 + dy**2) ** 0.5
                
                # If flag is within range, add to detected list
                if distance <= self._sensor_range:
                    detected_flags.append(flag_node_id)
            
            self._data = {
                'detected_flags': detected_flags,
                'flag_count': len(detected_flags)
            }
        
        def update(self, data):
            """Update sensor configuration."""
            if 'sensor_range' in data:
                self._sensor_range = data['sensor_range']
            if 'real_flags' in data:
                self._real_flags = data['real_flags']
    
    return FlagRangeSensor