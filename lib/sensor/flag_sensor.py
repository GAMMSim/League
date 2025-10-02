from gamms.SensorEngine import SensorType
from gamms.typing import ISensor

def create_flag_sensor_class(ctx):
    """Factory function to create FlagSensor class with the given context."""
    
    @ctx.sensor.custom("FLAG_SENSOR")
    class FlagSensor(ISensor):
        """Returns real flags and fake flags (candidate - real)."""
        
        def __init__(self, ctx, sensor_id, real_flags, candidate_flags):
            self._ctx = ctx
            self._sensor_id = sensor_id
            self._owner = None
            self._real_flags = real_flags
            self._candidate_flags = candidate_flags
            self._data = {}
        
        @property
        def sensor_id(self):
            return self._sensor_id
        
        @property
        def type(self):
            return SensorType.FLAG_SENSOR
        
        @property
        def data(self):
            return self._data
        
        def set_owner(self, owner):
            self._owner = owner
        
        def sense(self, node_id: int):
            """Return real flags and fake flags."""
            fake_flags = list(set(self._candidate_flags) - set(self._real_flags))
            
            self._data = {
                'real_flags': self._real_flags,
                'fake_flags': fake_flags
            }
        
        def update(self, data):
            pass
    
    return FlagSensor