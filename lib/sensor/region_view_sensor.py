"""
Coverage view of a region sensor: `<name>_region` is the local subgraph
`{nodes, edges}` that sensor `<name>` currently covers.

The view owns no range of its own — it reads the companion RegionSensor's
table, so it is always the same region the companion detects in (radius,
k-hop, or line-of-sight alike). Replaces the gamms RANGE sensor for any
`*_region` entry whose companion is a RegionSensor; that one measured a plain
euclidean radius regardless of what the companion could actually see, and
walked every edge of the graph on every read.

  nodes : {node_id: gamms Node}  -- every node in the companion's region
  edges : [gamms edge]           -- every edge with BOTH endpoints in it
"""
from typing import Any, Dict, List, Tuple

from lib.sensor.base_sensor import Sensor


def create_region_view_sensor_class(ctx):
    from gamms.SensorEngine import SensorType

    @ctx.sensor.custom("REGION_VIEW_SENSOR")
    class RegionViewSensor(Sensor):
        def __init__(self, ctx, sensor_id, table, graph_index, view_cache, **kw):
            """
            Args:
                table: the companion's node -> visible-nodes table.
                graph_index: per-game dict, filled on first use with every
                    gamms node and edge object — shared by all views in the game.
                view_cache: per-model {origin: (nodes, edges)} — shared by every
                    view over the same table, so an origin is built once per game.
            """
            super().__init__(ctx, sensor_id, SensorType.REGION_VIEW_SENSOR, **kw)
            self._table = table or {}
            self._graph_index = graph_index
            self._view_cache = view_cache

        def _index(self) -> Dict[str, Any]:
            if not self._graph_index:
                g = self._ctx.graph.graph
                self._graph_index["nodes"] = {nid: g.get_node(nid) for nid in g.get_nodes()}
                self._graph_index["edges"] = [g.get_edge(eid) for eid in g.get_edges()]
            return self._graph_index

        def _build(self, origin: int) -> Tuple[Dict[int, Any], List[Any]]:
            index = self._index()
            region = self._table.get(origin, frozenset((origin,)))
            all_nodes = index["nodes"]
            nodes = {nid: all_nodes[nid] for nid in region if nid in all_nodes}
            edges = [e for e in index["edges"] if e.source in nodes and e.target in nodes]
            return nodes, edges

        def sense(self, node_id: int) -> None:
            origin = self._carrier if self.is_static else node_id
            view = self._view_cache.get(origin)
            if view is None:
                view = self._view_cache[origin] = self._build(origin)
            nodes, edges = view
            # Fresh containers per read: the cached view is shared by every
            # agent and tick, so a strategy must never be handed it directly.
            self._data = {"nodes": dict(nodes), "edges": list(edges)}

    return RegionViewSensor
