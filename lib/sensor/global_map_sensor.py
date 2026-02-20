from gamms.SensorEngine import SensorType
from gamms.typing import ISensor
try:
    from lib.core.apsp_cache import get_apsp_length_cache
except ImportError:
    from ..core.apsp_cache import get_apsp_length_cache

def create_global_map_sensor_class(ctx):
    """Factory function to create GlobalMapSensor class with the given context."""
    
    @ctx.sensor.custom("GLOBAL_MAP")
    class GlobalMapSensor(ISensor):
        """Returns a NetworkX graph every time sense() is called."""
        
        def __init__(self, ctx, sensor_id, nx_graph):
            self._ctx = ctx
            self._sensor_id = sensor_id
            self._owner = None
            self._nx_graph = nx_graph
            self._data = {}
        
        @property
        def sensor_id(self):
            return self._sensor_id
        
        @property
        def type(self):
            return SensorType.GLOBAL_MAP
        
        @property
        def data(self):
            return self._data
        
        def set_owner(self, owner):
            self._owner = owner
        
        def sense(self, node_id: int):
            """Return the NetworkX graph."""
            self._data = {
                'graph': self._nx_graph,
                'apsp': get_apsp_length_cache(self._nx_graph),
            }
        
        def update(self, data):
            pass
    
    return GlobalMapSensor
