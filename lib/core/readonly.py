"""
Read-only mapping for tables that are shared process-wide and handed to
strategy code.

The visibility tables (lib/core/visibility_cache.py) and the APSP lookup
(lib/core/apsp_cache.py) are built once per graph and reused by every game in
the process -- a mass eval run is thousands of games in a single process, and
`mass_eval_batch.py` runs them serially with no worker recycling. Both are
passed into `state["sensor"]` by reference, so a strategy that writes into one
does not corrupt its own copy; it corrupts the table every later game reads.
Nothing raises, and because only games *after* the write see the damage, the
result matrix ends up depending on the order the games ran in.

`MappingProxyType` is the obvious answer and the wrong one here:
`isinstance(x, dict)` is False for a proxy, and the codebase guards the APSP
handoff with exactly that check in ~48 places -- `lib/agent/agent_map.py`,
both example strategies, and nearly every archived policy:

    apsp_lookup=global_map_apsp if isinstance(global_map_apsp, dict) else None

A proxy would send all of them down the `else None` branch, silently dropping
the APSP fast path and falling back to live shortest-path computation. So this
subclasses `dict` instead: `isinstance` still passes and reads are unchanged
and full speed, while every mutating method raises.
"""
from typing import Any, NoReturn


class ReadOnlyDict(dict):
    """A dict that reads normally and raises on any attempt to mutate it."""

    __slots__ = ()

    def _readonly(self, *_args: Any, **_kwargs: Any) -> NoReturn:
        raise TypeError(
            "This table is shared across every game in the process and is read-only. "
            "Copy it first if you need to modify it: dict(table), or "
            "{k: set(v) for k, v in table.items()} to also copy the rows."
        )

    __setitem__ = _readonly
    __delitem__ = _readonly
    pop = _readonly
    popitem = _readonly
    clear = _readonly
    update = _readonly
    setdefault = _readonly

    def __reduce__(self):
        # Pickle/copy as a plain dict -- a restored copy is the caller's own and
        # has no reason to stay frozen. Without this, pickling would rebuild a
        # ReadOnlyDict by calling __setitem__ and blow up.
        return (dict, (dict(self),))


def freeze_nested(mapping: dict) -> "ReadOnlyDict":
    """
    Read-only view of `mapping`, with any dict-valued row frozen too.

    The APSP table is {source: {target: distance}}, so freezing only the outer
    level would still leave `apsp[src][dst] = 0` open. Visibility tables map to
    frozensets and are already immutable one level down.
    """
    return ReadOnlyDict(
        (key, ReadOnlyDict(value) if type(value) is dict else value)
        for key, value in mapping.items()
    )
