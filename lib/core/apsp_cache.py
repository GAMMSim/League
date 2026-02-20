from typing import Any, Dict, Optional, Tuple
import networkx as nx

try:
    from lib.core.console import debug
except ImportError:
    from ..core.console import debug


_APSP_CACHE_KEY = "__apsp_length_cache"
_APSP_SIG_KEY = "__apsp_graph_signature"
_APSP_SOURCE_TOKEN_KEY = "__graph_source_token"
_APSP_GLOBAL_REF_KEY = "__apsp_global_cache_key"

# Process-wide APSP cache. Keyed by (graph_source_token, graph_signature) so newly
# loaded graph objects from the same backing file can reuse the same lookup table.
_GLOBAL_APSP_CACHE: Dict[Tuple[str, Tuple[int, int]], Dict[Any, Dict[Any, int]]] = {}


def _graph_signature(graph: nx.Graph) -> Tuple[int, int]:
    """Return a lightweight signature used to validate APSP cache reuse."""
    return graph.number_of_nodes(), graph.number_of_edges()


def get_apsp_length_cache(graph: nx.Graph) -> Dict[Any, Dict[Any, int]]:
    """
    Get or build all-pairs shortest path length lookup for a graph.
    Cache is stored directly on graph metadata and rebuilt only if topology changed.
    """
    meta = getattr(graph, "graph", None)
    sig = _graph_signature(graph)

    global_key: Optional[Tuple[str, Tuple[int, int]]] = None
    if isinstance(meta, dict):
        cached = meta.get(_APSP_CACHE_KEY)
        cached_sig = meta.get(_APSP_SIG_KEY)
        if isinstance(cached, dict) and cached_sig == sig:
            return cached

        source_token = meta.get(_APSP_SOURCE_TOKEN_KEY)
        if isinstance(source_token, str) and source_token:
            global_key = (source_token, sig)
            global_cached = _GLOBAL_APSP_CACHE.get(global_key)
            if isinstance(global_cached, dict):
                meta[_APSP_CACHE_KEY] = global_cached
                meta[_APSP_SIG_KEY] = sig
                meta[_APSP_GLOBAL_REF_KEY] = global_key
                return global_cached

    debug(f"Building APSP distance cache for graph (nodes={sig[0]}, edges={sig[1]})")
    cache = {src: dict(dst_lengths) for src, dst_lengths in nx.all_pairs_shortest_path_length(graph)}

    if isinstance(meta, dict):
        meta[_APSP_CACHE_KEY] = cache
        meta[_APSP_SIG_KEY] = sig
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
