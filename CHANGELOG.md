# Changelog

## v6.08 — game rule v1.4 and the policy runtime check

The first release since v6.05.2. It moves the game to rule **v1.4**, where
attackers can no longer see every defender, and adds a runtime check that every
submitted policy must pass.

### Notice

- **Game rule is now v1.4, and every attacker written for v1.3 or earlier needs a change.** Red no longer has the whole-map `agent` sensor:

  ```python
  # before (v1.3 and earlier)
  enemies   = agent_ctrl.sensor_data(state, "agent")["enemies"]
  teammates = agent_ctrl.sensor_data(state, "agent")["teammates"]

  # after (v1.4)
  enemies   = agent_ctrl.sensor_data(state, "egocentric_agent")["enemies"]   # line of sight only
  teammates = agent_ctrl.sensor_data(state, "custom_team")                   # whole map
  ```

  An attacker that still reads `agent` does not crash. If it guards with `if "agent" in sensors`, it keeps running and sees no defenders at all. The same goes for a policy that still reads the `_region` sensors.
- **Submissions must pass `check_policy.py`**, and the report it writes (`reports/<policy>_<team>.html`) is part of the submission. See README section 7.
- **Re-run `pip install -r requirements.txt`.**

### New

- **Game rule v1.4** (`config/rules/v1.4.yml`). Red senses defenders by line of sight, the same way blue has always sensed red: a defender is visible only when no building blocks the view. Teammates are still visible map-wide, through `custom_team`. Blue's sensors are unchanged.

  | Red's view of | v1.3 | v1.4 |
  |---|---|---|
  | Defenders | `agent` — whole map, every tick | `egocentric_agent` — line of sight, range 400 |
  | Teammates | `agent` | `custom_team` — whole map |
  | Flags | `egocentric_flag` | unchanged |

  Game length is 75 ticks. The interaction model, payoff constants, speeds and radii are unchanged.
- **Runtime check** (`check_policy.py`). Plays a policy over the 30 example configs against the example policy for the other team, times every call to `strategy`, prints a pass/fail table and writes an HTML report:

  ```bash
  python check_policy.py policies.attacker.my_atk --team red
  python check_policy.py policies.defender.my_def --team blue
  ```

  Times are measured in machine units: the script first times a fixed workload on your machine (about a second) and divides every measurement by it, so the same limits hold on any machine.

  | Limit | Machine units |
  |---|---|
  | Mean per call | 0.01 |
  | Slowest single call | 0.5 |
  | Each agent's first call (one-time setup) | 2.0 |

  An exception from `strategy`, a return value that is not a `str`, or an action that is not an `int` also fails the check. In a normal game the engine logs these and the agent stands still for that tick. Options: `--quick` (one game per size), `--opponent`, `--out`.
- **Sensor coverage.** Ask a sensor what it can cover, now or from any other node:

  ```python
  nodes = agent_ctrl.coverage(state, "egocentric_flag")                # covered now
  there = agent_ctrl.coverage_from(state, "egocentric_flag", node_id)  # covered from another node
  ```

  Both return a set of node ids, or `None` for a sensor with no range. `state["sensor"]["stationary"]` also gains `region`, the towers' combined coverage.
- **Example policies that play.** `example/example_atk.py` searches the candidate flags as a team and keeps out of defenders' reach. `example/example_def.py` guards the most dangerous flag and steps out to tag attackers near it. Both read the v1.4 sensors and show the coverage calls, and they are the opponents the runtime check uses.

### Changed

- **Flag discovery needs line of sight.** Red used to be credited for a flag hidden behind a building that its own sensor never reported. Discovery now follows red's flag sensor, so scores are not comparable with earlier releases.
- **All example configs are on rule v1.4**, with 75-tick games and resampled agent and flag positions. `main.py` and the GUI open `config/example_configs/osm_a/R3B3F3-5/R3B3F3-5_run0.yml` by default.
- **`state["team_cache"]` is the shared team cache.** It was handing each agent a private one. `agent_ctrl.get_team()` / `set_team()` always used the shared cache and behave as before.
- **`global_map`'s `apsp` and a sensor's `table` are read-only**, because every game in a run shares them. Writing into one raises `TypeError`. Copy first if you need to modify: `dict(table)`.
- **`requirements.txt` is lighter.** `torch`, `torch-geometric`, `stable-baselines3` and `gymnasium` are no longer installed by default.
- **Under rule v1.3**, the `_region` sensors report the occluded region of their companion sensor instead of a plain radius.
- Headless games run several times faster, plus internal cleanup with no effect on strategies.

### Removed

- **Red's `agent` sensor**, from rule v1.4. Rules v1.2 and v1.3 still have it.
- **`egocentric_flag_region` and `egocentric_agent_region`**, from rule v1.4. They listed nodes the sensor could not actually see. Use `coverage()` / `coverage_from()`.
- **`example/example_config.yml`** and **`config/debug.yml`**. Use a config under `config/example_configs/`.

---

## v6.05 / v6.05.1 — sensor and visibility redesign

### Reference detail

**The `strategy(state)` / `map_strategy(agent_config)` contract itself is unchanged.** No signature changes, no new required return values.

**What changed: the shape of `state["sensor"][name]`, only for sensors declared in the new dict form.** Your config's `agents.<team>_global.sensors` list can hold either:
- a plain string (`egocentric_flag`, `stationary`, ...) — **unchanged, works exactly as before.**
- a dict (`{name: egocentric_flag, model: red_flag_r400, carrier: agent, flags: real}`) — **new payload shape**, verified against `lib/agent/` (zero changes there — `agent_ctrl.sensor_data()`, `.cache`, `.map` all behave exactly as before) and `example/example_atk.py`/`example/example_def.py` (the only strategy-side diff across the whole branch is the one `flag_count` line below).

If your config still uses plain strings, nothing to do. If it uses the dict form (all of `config/example_configs/osm_a/*` and `example/example_config.yml` now do), the payload for that sensor is:

```python
{
    "carrier": ...,
    "team": "red" | "blue",
    "model": "<visibility model name>",
    "region": frozenset[int],        # node IDs currently visible from this sensor's origin
    "table": {node_id: frozenset[int]},  # NEW (v6.05.1) — the full model, every node's region, static for the game
    "detected_agents": {name: node_id},
    "enemies": {name: node_id},      # pre-split, same as before
    "teammates": {name: node_id},
    "detected_flags": [node_id, ...],  # only if the sensor entry has flags: real
}
```

The old flag sensor's `flag_count` key doesn't exist on this payload. If you read it directly:
```python
flag_count = payload.get("flag_count", len(payload.get("detected_flags", [])))
```
(`example/example_atk.py` already does this — copy the pattern if your strategy reads `flag_count` directly.)

The `stationary` aggregation (`state["sensor"]["stationary"]`) is **unchanged in shape** — `{detections, enemies, teammates}` — regardless of whether the underlying towers use the old or new sensor form. No defender-side strategy changes needed (confirmed: `example/example_def.py` has zero diff across the whole branch).

### New: occlusion (line-of-sight) visibility

If a config's `environment.visibility_models` has an entry with `type: line_of_sight`, that sensor's `region`/`detected_agents`/`detected_flags` are now limited by **real building geometry** (fetched from OpenStreetMap), not a euclidean radius circle. No strategy code changes needed — but if you're tuning a strategy against a map with buildings, expect it to see meaningfully *less* than a `type: radius` sensor with the same range, especially in dense areas. If a strategy seems to be "missing" flags/enemies it used to see, check whether its config migrated to `line_of_sight`.

### Config changes

- New `environment.visibility_models` block:
  ```yaml
  environment:
    visibility_models:
      red_flag_r400:
        type: line_of_sight       # or radius, or khop
        crs: EPSG:32618           # required for line_of_sight — must match the graph's coordinate system
        max_range: 400            # optional cap, in graph coordinate units
  ```
- Sensor entries reference a model by name: `{name: egocentric_flag, model: red_flag_r400, carrier: agent, flags: real}`.
- New `visualization.show_occlusion_visuals: true|false` (default `true`) — turns the occlusion polygon/glow rendering off while keeping the rest of visualization on (agents, flags, HUD). Useful if it's making a run too slow.
- New rule template `config/rules/v1.3.yml` (occlusion-enabled version of `v1.2.yml`) — this is what `mass_eval/config_generator.py` now uses to (re)generate `config/example_configs/osm_a/`.

### `main.py` / `launch_gui.py` changes

- **`main.py`** — still points at `example/example_config.yml`, but that file itself now uses occlusion sensors. Running `main.py` unmodified will show the new visibility polygons.
- **`launch_gui.py`** — new checkbox, **"Occlusion visuals (slower)", unchecked by default.** The occlusion polygon is real per-frame geometry and was making the interactive window choppy, so the GUI defaults to skipping it (agents/flags/HUD still render normally). Check the box if you specifically want to see it live.

### Everything else, at a glance (no action needed)

- The sensor framework itself was unified: old one-class-per-sensor-type code was replaced by two generic classes (`RegionSensor` for anything spatial, `InfoSensor` for aspatial data like `global_map`) driven by a `visibility_models` table registry. `radius`/`khop`/`line_of_sight` are just different ways of building that table, not different sensor implementations.
- New occlusion pipeline: real OpenStreetMap building footprints are fetched once per map area and cached to disk; visibility between any two nodes is computed by testing whether the straight line between them crosses a building.
- Live visualization and the `mass_eval` static preview tool both now draw the *real* visibility shape (a polygon with notches where buildings block the view) plus small glow markers at confirmed-visible nodes, instead of a flat radius circle.
- All of `config/example_configs/osm_a/` was regenerated (30 configs, 6 scenarios × 5 runs) through the real `mass_eval/config_generator.py` pipeline using the new occlusion rule template.
- Minor internal bugfix: video recording's animation-alpha update now targets gamms 1.0's actual `_dynamic_artists` attribute (was silently failing against a stale `_agent_artists` name before).

---

### Full changelog (branch `6.04`)

- **`77f5a25`** — Add visibility sensor framework and models. Introduced the first (at the time dormant, not referenced by any config) visibility-graph sensor scaffolding: `CarrierSensor`, `VisibilityMapSensor`/`VisibilitySensor`, `lib/core/visibility_cache.py` for loading precomputed visibility tables, and `sensor_engine.py` wiring to register them. Added 3 fake k-hop visibility tables for `osm_200_a`.

- **`ecddb38`** — Integrate configurable `label_font_size` + gq map from gq-MassEval. Added a configurable agent/flag label font size to the visual layer; added the `graphs/gq.pkl` map asset.

- **`80a9556`** — Refactor sensor handling and visibility models. Major architecture refactor (Stage 1+2 of the sensor redesign): deleted `carrier_sensor.py`, `visibility_sensor.py`, `candidate_flag_sensor.py`, `flag_sensor.py`, `global_map_sensor.py`; added `base_sensor.py` (shared `Sensor` base class), `info_sensor.py` (aspatial info sensors), `region_sensor.py` (generic table-backed spatial sensor), `lib/core/visibility_generators.py` (radius/khop table generators; `line_of_sight` stubbed as `NotImplementedError`). Rewrote `sensor_engine.py`'s sensor-creation dispatch around the new unified system. Migrated the `R3B3F3-5` example configs to a new `config/example_configs/osm_a/R3B3F3-5-region/` directory (5 runs) as a proof of concept of the new dict-based sensor config format. One-line fix in `example/example_atk.py` (`flag_count` now derived from `detected_flags` instead of read directly, since the new sensor payload doesn't carry that key).

- **`ae151b2`** — Update sensor configuration and visualization. Built the `at:` fan-out mechanism (one config line expands into N static tower sensors from a position list). Added `config/rules/v1.2-region.yml` rule template. Updated the 5 migrated example configs to use `at:` instead of repeating per-tower dict entries. Replaced the flat euclidean-radius-circle sensor indicator in `lib/visual/agent_visual.py`/`lib/visual/flag_visual.py` with a per-node halo rendering (still radius-shaped at this point — occlusion geometry came later). Documented the new sensor system in `CLAUDE.md`.

- **`19e4d53`** — Update visibility models and agent configuration. Implemented real building-occlusion visibility: `_build_line_of_sight` generator in `lib/core/visibility_generators.py` (fetches real OpenStreetMap building footprints via `osmnx`, computes node-pair visibility via straight-line/building-polygon intersection, with disk caching); new `lib/core/visibility_polygon.py` (ray-casting visibility-polygon algorithm — casts rays from a sensing origin against nearby building edges to build the actual visible-region shape for rendering); new `lib/core/visibility_lookup.py` (cheap config-only resolvers — model ranges/tables/building polygons — used by the visual layer without needing a live sensor); new `lib/visual/building_visual.py` (draws building footprints on the map); rewrote `agent_visual.py`/`flag_visual.py` to draw the real ray-traced visibility polygon plus small soft-glow markers at actually-visible nodes, replacing the radius-halo approach, with correct z-ordering (drawn behind the graph's own node markers) and gating so headless/no-vis runs skip this entirely. Built and persisted occlusion visibility tables for `red_flag_r400`/`blue_agent_r250`/`blue_tower_r450` across all three West Point subgraphs (`osm_200_a`, `osm_200_b`, `osm_200_c`). Renamed `config/rules/v1.2-region.yml` → `config/rules/v1.3.yml`. Added `osmnx`/`pyproj` to `requirements.txt`.

- **`38b2db2`** — Update configuration files for visibility models. Added `visualization.show_occlusion_visuals` config flag, threaded through `lib/game/game_engine.py` (`launch_from_files`/`build_runtime`) and `lib/game/visualization_engine_new.py`; added a matching checkbox to `lib/gui/launcher.py` (defaults to **off**, so the interactive GUI stays responsive — the occlusion polygon is real per-frame geometry and was making the window choppy). Converted `example/example_config.yml` to use occlusion sensors and repointed `main.py` at it. Repointed `mass_eval/config_generator.py` at the new `v1.3.yml` rule template and `osm_200_a.pkl`, and regenerated all of `config/example_configs/osm_a/` (6 scenarios × 5 runs) plus their `config/example_visuals/osm_a/` previews through the real generator pipeline. Updated `mass_eval/config_visualizer.py`'s static preview tool to compute and draw the same real ray-traced occlusion polygon as the live renderer (previously a plain circle), fixed z-ordering so graph nodes draw above the sensing indicators, and tuned indicator transparency for legibility. Updated `CLAUDE.md` accordingly.

- **`a799431`**, **`d5bf62b`** — Added and refined this `CHANGELOG.md`.

- **`b19388a`**, **`787ea62`** — Explored exposing visibility tables via a standalone `{name: visibility_map, model: ..., type: table}` sensor entry, then reverted it in favor of the `table` key approach below (a standalone whole-table sensor was explicitly ruled out earlier in the sensor-redesign design pass — see `docs/sensor-architecture.md` §2, "No standalone `visibility_map` sensor"). Net no-op on `config/example_configs/osm_a/`.

- **`(uncommitted at time of writing)`** — Added a `table` key to every `RegionSensor` payload (`lib/sensor/region_sensor.py`): alongside `region` (the current origin's visible-node slice), sensors now also expose the model's full `node_id -> frozenset[visible_node_ids]` table, so a strategy can look up visibility from any node, not just its own position — same sensor, no new name to learn. `lib/game/game_engine.py`'s `stationary` aggregation hoists the (shared, single) table up alongside `enemies`/`teammates`/`detections`. `lib/game/sensor_engine.py` gained a `_add_mapping` helper that warns (instead of silently clobbering) if two sensor entries in the same agent's config reuse the same logical `name:`. `example/example_atk.py`/`example/example_def.py` updated to demonstrate reading `table`, and a stray hardcoded-agent-name example line (`agent_map.get_agent_position(enemy_team, "blue_0")`) was removed from both.

---

## Earlier releases

Release notes before v6.05 were overwritten when this file was rewritten for
the sensor/visibility release. The one entry that survived, recovered from
`docs/CHANGELOG.md` before that file was folded in here:

### [6.02.13]

**Added**

- README now includes a dedicated **virtual environment setup** section: create a `.venv` with `python3 -m venv .venv`, activate it, and install via `pip install -r requirements.txt`.
- `requirements.txt` updated with pinned minimum versions across all dependency groups (core, GUI, policies, mass-eval, MIT-specific).

**Changed**

- **Python 3.11 or 3.12 is now explicitly required.** Python 3.13+ has known compatibility issues with the `gamms` visualization engine and is not supported. README and `requirements.txt` both document this constraint.

For the full reasoning behind that version constraint, see
[docs/python-version-constraint.md](docs/python-version-constraint.md).
