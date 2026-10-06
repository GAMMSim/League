from typing import Any, Dict, Optional, Tuple
import networkx as nx

try:
    from lib.core.readonly import freeze_nested
except ImportError:
    from .readonly import freeze_nested

try:
    from lib.core.console import debug
except ImportError:
    from ..core.console import debug


_APSP_CACHE_KEY = "__apsp_length_cache"
_APSP_SIG_KEY = "__apsp_graph_signature"
_APSP_SOURCE_TOKEN_KEY = "__graph_source_token"
_APSP_GLOBAL_REF_KEY = "__apsp_global_cache_key"
_APSP_NODES_KEY = "__apsp_node_count"

# Process-wide APSP cache. Keyed by (graph_source_token, graph_signature) so newly
# loaded graph objects from the same backing file can reuse the same lookup table.
_GLOBAL_APSP_CACHE: Dict[Tuple[str, Tuple[int, int]], Dict[Any, Dict[Any, int]]] = {}


def _graph_signature(graph: nx.Graph) -> Tuple[int, int]:
    """Return a lightweight signature used to validate APSP cache reuse."""
    return graph.number_of_nodes(), graph.number_of_edges()


def _node_count(graph: nx.Graph) -> int:
    """O(1) node count — `len()` reads the node dict directly."""
    return len(graph)


def get_apsp_length_cache(graph: nx.Graph) -> Dict[Any, Dict[Any, int]]:
    """
    Get or build all-pairs shortest path length lookup for a graph.
    Cache is stored directly on graph metadata and rebuilt only if topology changed.

    Hot path note: this runs once per agent per tick from two places --
    game_engine.validate_and_execute_movement() and the `global_map` sensor in
    info_sensor.py. The full signature is not cheap: on a MultiDiGraph,
    `number_of_edges()` sums every node's degree, measured at 119 us against
    0.07 us for the node count, and it was the entire cost of a cache hit --
    about 180 ms per game at 10 agents x 75 ticks, or ~7 minutes across a
    2700-game mass eval, spent re-deriving a number that cannot change
    mid-game.

    So a cache hit is gated on the node count alone. The full (nodes, edges)
    signature is still computed and stored whenever the table is built, and
    still re-checked whenever the node count moves. What this no longer
    catches on its own is a mutation that changes edges while leaving the node
    count untouched. That was always a partial guarantee -- rewiring an edge
    keeps both counts identical and slipped through before too -- and the
    engine never mutates topology; the only writer that could is strategy code
    reaching into the shared graph through the `global_map` sensor.
    """
    meta = getattr(graph, "graph", None)

    if isinstance(meta, dict):
        cached = meta.get(_APSP_CACHE_KEY)
        cached_nodes = meta.get(_APSP_NODES_KEY)
        if isinstance(cached, dict) and cached_nodes == _node_count(graph):
            return cached

    sig = _graph_signature(graph)

    global_key: Optional[Tuple[str, Tuple[int, int]]] = None
    if isinstance(meta, dict):
        cached = meta.get(_APSP_CACHE_KEY)
        cached_sig = meta.get(_APSP_SIG_KEY)
        if isinstance(cached, dict) and cached_sig == sig:
            meta[_APSP_NODES_KEY] = sig[0]
            return cached

        source_token = meta.get(_APSP_SOURCE_TOKEN_KEY)
        if isinstance(source_token, str) and source_token:
            global_key = (source_token, sig)
            global_cached = _GLOBAL_APSP_CACHE.get(global_key)
            if isinstance(global_cached, dict):
                meta[_APSP_CACHE_KEY] = global_cached
                meta[_APSP_SIG_KEY] = sig
                meta[_APSP_NODES_KEY] = sig[0]
                meta[_APSP_GLOBAL_REF_KEY] = global_key
                return global_cached

    debug(f"Building APSP distance cache for graph (nodes={sig[0]}, edges={sig[1]})")
    # Frozen: this table is shared by every game in the process and handed
    # straight to strategy code. Rows are frozen too -- apsp[src][dst] = 0
    # would otherwise still get through. See lib/core/readonly.py.
    cache = freeze_nested({src: dict(d) for src, d in nx.all_pairs_shortest_path_length(graph)})

    if isinstance(meta, dict):
        meta[_APSP_CACHE_KEY] = cache
        meta[_APSP_SIG_KEY] = sig
        meta[_APSP_NODES_KEY] = sig[0]
        if global_key is not None:
            _GLOBAL_APSP_CACHE[global_key] = cache
            meta[_APSP_GLOBAL_REF_KEY] = global_key

    return cache


def get_cached_distance(cache: Dict[Any, Dict[Any, int]], source: Any, target: Any) -> Optional[int]:
    """Return cached shortest-path distance, or None when unreachable/missing."""
    src_row = cache.get(source)
    if src_row is None:
        return None
    return src_row.get(target)
