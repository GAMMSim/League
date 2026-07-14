from typing import Any, Dict, Iterable, Optional, Tuple, Union
from pathlib import Path
import os
import json
import pickle
import tempfile
import networkx as nx

try:
    from lib.core.console import debug, warning, success
    from lib.core.visibility_generators import build as _build_visibility_model
except ImportError:
    from ..core.console import debug, warning, success
    from .visibility_generators import build as _build_visibility_model


# Metadata keys stashed on graph.graph so reloaded graph objects from the same
# backing file reuse the same normalized visibility tables.
_VIS_CACHE_KEY = "__visibility_models_cache"
_VIS_SIG_KEY = "__visibility_models_signature"
# Set by lib/utils/file_utils.py when the graph is loaded from disk.
_GRAPH_SOURCE_TOKEN_KEY = "__graph_source_token"
_GRAPH_SOURCE_PATH_KEY = "__graph_source_path"

# Default directory for genspec-provisioned (load-or-build) model files.
_DEFAULT_VIS_DIR = "graphs/visibility"

# Process-wide registry. Keyed by (graph_source_token, specs_signature) so two
# games on the same graph + same visibility files share one set of tables.
#   table layout: model_name -> { node_id(int) -> frozenset[int] of visible nodes }
_GLOBAL_VIS_CACHE: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], Dict[str, Dict[int, frozenset]]] = {}


def _normalize_model(raw: Any) -> Dict[int, frozenset]:
    """
    Normalize any supported visibility source into the hot-path representation:
    dict[int, frozenset[int]]. Membership test `target in table[node]` is then a
    single C-level dict get + set __contains__ — the cheapest option for the
    millions of lookups this gets hit with.

    Accepts:
      - a networkx graph: each node's out-neighbors are the nodes it can see
      - a plain dict {node: iterable_of_visible_nodes} (JSON keys may be strings)
    The source is faithfully preserved (self is NOT injected) so strategies see
    the model exactly as authored.
    """
    if isinstance(raw, nx.Graph):
        return {n: frozenset(raw.neighbors(n)) for n in raw.nodes}
    if isinstance(raw, dict):
        return {int(k): frozenset(int(x) for x in v) for k, v in raw.items()}
    raise TypeError(
        f"Unsupported visibility model type: {type(raw).__name__} "
        "(expected a networkx graph or a {node: [visible nodes]} dict)"
    )


def _load_model_file(path: str) -> Dict[int, frozenset]:
    """Load one visibility model file (.pkl/.pickle or .json) and normalize it."""
    ext = Path(path).suffix.lower()
    if ext == ".json":
        with open(path, "r") as f:
            raw = json.load(f)
    elif ext in (".pkl", ".pickle"):
        with open(path, "rb") as f:
            raw = pickle.load(f)
    else:
        raise ValueError(f"Unsupported visibility file format: {ext} ({path})")
    return _normalize_model(raw)


def _resolve_path(path: str, base_dir: Optional[str]) -> str:
    p = Path(path)
    if p.is_absolute() or base_dir is None:
        return str(p)
    return str(Path(base_dir) / p)


def _file_token(path: str) -> str:
    """mtime+size token so the cache invalidates when a model file is regenerated."""
    try:
        st = os.stat(path)
        return f"{path}:{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        return path


def _genspec_sig(genspec: Dict[str, Any]) -> str:
    """Stable signature of a generator spec, for the process-wide cache key."""
    return repr(sorted(genspec.items()))


def _provisioned_path(vis_dir: str, graph_stem: str, model_name: str) -> str:
    return str(Path(vis_dir) / f"{graph_stem}_{model_name}.pkl")


def _load_provisioned(path: str, genspec: Dict[str, Any]) -> Optional[Dict[int, frozenset]]:
    """
    Load a genspec-provisioned file iff it exists AND its embedded genspec
    matches the requested one. Staleness is decided by genspec equality, not
    mtime — a file surviving from a since-changed config (e.g. range 30->40 on
    the same model name) must NOT be reused.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except Exception as e:
        warning(f"Failed to read provisioned visibility file {path}: {e}")
        return None
    if not isinstance(payload, dict) or "genspec" not in payload or "table" not in payload:
        return None  # not a provisioned file — treat as absent, will rebuild
    if payload["genspec"] != genspec:
        return None  # stale — genspec changed since this file was built
    return _normalize_model(payload["table"])


def _build_and_persist(
    graph: nx.Graph, genspec: Dict[str, Any], path: str
) -> Dict[int, frozenset]:
    gen_type = genspec.get("type")
    if not gen_type:
        raise ValueError(f"visibility genspec missing 'type': {genspec!r}")
    table = _build_visibility_model(gen_type, graph, genspec)

    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=out_dir, suffix=".pkl.tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump({"genspec": genspec, "table": table}, f)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    debug(f"Built + persisted visibility model at {path} ({len(table)} nodes)")
    return _normalize_model(table)


def get_visibility_models(
    graph: nx.Graph,
    model_specs: Optional[Dict[str, Union[str, Dict[str, Any]]]],
    base_dir: Optional[str] = None,
    vis_dir: str = _DEFAULT_VIS_DIR,
) -> Dict[str, Dict[int, frozenset]]:
    """
    Load (or build-and-persist) the visibility models for `graph`.

    Args:
        graph: the game graph (used only as a cache anchor — visibility is static
            per graph, so tables are keyed by the graph's source token)
        model_specs: {model_name: spec}. spec is either:
            - a str file path (legacy) — resolved relative to base_dir, loaded
              as-is, no persistence/staleness logic (unchanged behavior).
            - a dict generator spec, e.g. {"type": "radius", "range": 30} —
              load-or-build against `<vis_dir>/<graph_stem>_<model_name>.pkl`,
              rebuilding whenever the embedded genspec no longer matches.
            None/empty yields an empty registry.
        base_dir: directory to resolve relative legacy string paths against
            (typically the directory holding the graph .pkl).
        vis_dir: directory for genspec-provisioned files (load + persist).

    Returns:
        {model_name: {node_id: frozenset[int] visible nodes}} — shared by
        reference across all callers; treat as read-only.
    """
    meta = getattr(graph, "graph", None)
    specs = model_specs or {}

    sig_parts = []
    for name, spec in specs.items():
        if isinstance(spec, str):
            sig_parts.append((name, "path:" + _file_token(_resolve_path(spec, base_dir))))
        else:
            sig_parts.append((name, "gen:" + _genspec_sig(spec)))
    sig: Tuple[Tuple[str, str], ...] = tuple(sorted(sig_parts))

    if isinstance(meta, dict):
        cached = meta.get(_VIS_CACHE_KEY)
        if isinstance(cached, dict) and meta.get(_VIS_SIG_KEY) == sig:
            return cached

    source_token = ""
    graph_stem = "graph"
    if isinstance(meta, dict):
        token = meta.get(_GRAPH_SOURCE_TOKEN_KEY)
        if isinstance(token, str):
            source_token = token
        source_path = meta.get(_GRAPH_SOURCE_PATH_KEY)
        if isinstance(source_path, str) and source_path:
            graph_stem = Path(source_path).stem

    global_key = (source_token, sig)
    global_cached = _GLOBAL_VIS_CACHE.get(global_key)
    if global_cached is not None:
        if isinstance(meta, dict):
            meta[_VIS_CACHE_KEY] = global_cached
            meta[_VIS_SIG_KEY] = sig
        return global_cached

    models: Dict[str, Dict[int, frozenset]] = {}
    for name, spec in specs.items():
        try:
            if isinstance(spec, str):
                path = _resolve_path(spec, base_dir)
                models[name] = _load_model_file(path)
                debug(f"Loaded visibility model '{name}' from {path} ({len(models[name])} nodes)")
            else:
                path = _provisioned_path(vis_dir, graph_stem, name)
                loaded = _load_provisioned(path, spec)
                if loaded is not None:
                    models[name] = loaded
                    debug(f"Loaded provisioned visibility model '{name}' from {path} ({len(loaded)} nodes)")
                else:
                    models[name] = _build_and_persist(graph, spec, path)
        except Exception as e:
            warning(f"Failed to load/build visibility model '{name}' from {spec!r}: {e}")

    if models:
        success(f"Loaded {len(models)} visibility model(s): {list(models.keys())}")
    _GLOBAL_VIS_CACHE[global_key] = models
    if isinstance(meta, dict):
        meta[_VIS_CACHE_KEY] = models
        meta[_VIS_SIG_KEY] = sig
    return models
