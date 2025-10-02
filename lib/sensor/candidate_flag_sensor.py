from gamms.SensorEngine import SensorType
from gamms.typing import ISensor

def create_candidate_flag_sensor_class(ctx):
    """Factory function to create CandidateFlagSensor class with the given context."""
    
    @ctx.sensor.custom("CANDIDATE_FLAG_SENSOR")
    class CandidateFlagSensor(ISensor):
        """Returns candidate flags only."""
        
        def __init__(self, ctx, sensor_id, candidate_flags):
            self._ctx = ctx
            self._sensor_id = sensor_id
            self._owner = None
            self._candidate_flags = candidate_flags
            self._data = {}
        
        @property
        def sensor_id(self):
            return self._sensor_id
        
        @property
        def type(self):
            return SensorType.CANDIDATE_FLAG_SENSOR
        
        @property
        def data(self):
            return self._data
        
        def set_owner(self, owner):
            self._owner = owner
        
        def sense(self, node_id: int):
            """Return the candidate flags."""
            self._data = {'candidate_flags': self._candidate_flags}
        
        def update(self, data):
            pass
    
    return CandidateFlagSensor