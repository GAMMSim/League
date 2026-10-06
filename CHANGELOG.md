# Changelog

## v6.08 — game rule v1.4 and the policy runtime check

The first release since v6.05.2. It moves the game to rule **v1.4**, fixes
flag discovery and sensor coverage under that rule, and adds a runtime check
that every submitted policy must pass.

### Announcement (ready to post)

> @channel A new League release has been posted.
>
> **[League Update — v6.08] - [GAMMS Version: 1.0.0] - [Game Rule: v1.4]**
>
> This is the first release since v6.05.2. It moves the game to **rule v1.4** and adds a **runtime check that every submission must pass**. Please re-run `pip install -r requirements.txt` after updating.
>
> **1. Attackers are no longer omniscient.** This breaks every existing attacker. Defenders are unaffected.
>
> Up to rule v1.3, red carried the `agent` sensor: every defender's position, every tick, anywhere on the map. Rule v1.4 removes it. Red now senses defenders the way blue has always sensed red, through a line-of-sight sensor that buildings block. On a reference set of games red sees a defender on about 18% of ticks, against 100% before. Teammates are still visible map-wide, now through `custom_team`.
>
> ```python
> # before (v1.3 and earlier)
> enemies   = agent_ctrl.sensor_data(state, "agent")["enemies"]
> teammates = agent_ctrl.sensor_data(state, "agent")["teammates"]
>
> # after (v1.4)
> enemies   = agent_ctrl.sensor_data(state, "egocentric_agent")["enemies"]   # line of sight only
> teammates = agent_ctrl.sensor_data(state, "custom_team")                   # whole map
> ```
>
> **An attacker that still reads `agent` will not crash. It will go blind.** If it guards with `if "agent" in sensors`, it keeps running and sees no defenders at all. Check yours before submitting.
>
> **2. Flag discovery now needs line of sight.** The discovery reward used plain distance, so red was credited for flags hidden behind buildings that its own sensor never reported. It now follows red's flag sensor. Scores are not comparable with earlier releases.
>
> **3. `egocentric_flag_region` and `egocentric_agent_region` are gone from v1.4.** They listed nodes the matching sensor could not actually see. Ask the sensor itself:
>
> ```python
> nodes = agent_ctrl.coverage(state, "egocentric_flag")                # covered now
> there = agent_ctrl.coverage_from(state, "egocentric_flag", node_id)  # covered from another node
> ```
>
> **4. Policies must pass a runtime check, and the report goes in with your submission.** A policy that is too slow to evaluate will not be accepted. Run this on your own machine:
>
> ```bash
> python check_policy.py policies.attacker.my_atk --team red
> python check_policy.py policies.defender.my_def --team blue
> ```
>
> It plays your policy over the 30 example configs, times every call to `strategy`, prints a pass/fail table and writes `reports/<policy>_<team>.html`. **Send that file with your submission.** A fast policy finishes in well under a minute.
>
> The limits do not depend on your machine. The script first times a fixed workload (about a second) and measures everything in multiples of it:
>
> - mean per call: **0.01** units
> - slowest single call: **0.5** units
> - each agent's first call (one-time setup): **2.0** units
>
> On a machine where the unit is 1 s, that is 10 ms per call on average. An exception from `strategy`, a return value that is not a `str`, or an action that is not an `int` also fails the check. In a normal game the engine logs those and your agent stands still for the tick.
>
> **5. The example policies now play properly.** `example_atk` searches the candidate flags as a team and keeps out of defenders' reach; `example_def` guards the most dangerous flag and steps out to tag attackers near it. Both read the v1.4 sensors, so they are the reference for the changes above and the opponents the runtime check uses.
>
> **Also in this release**
>
> - Every game config now lives under `config/example_configs/<map>/<scenario>/`; `example/example_config.yml` is gone. All 30 example configs are on rule v1.4 (in v6.05.2 they were still on v1.2).
> - No past policies ship with this release. They all read `agent` and would play blind under v1.4.
> - `requirements.txt` is lighter: `torch`, `torch-geometric`, `stable-baselines3` and `gymnasium` are no longer installed by default. `cbor2`, `rasterio` and `cvxpy` are now listed.
> - Headless games run several times faster.
> - `state["team_cache"]` is now really shared across teammates. `agent_ctrl.get_team()` / `set_team()` always were, so most policies never saw the difference.
> - `global_map`'s `apsp` and a sensor's `table` are read-only. Copy before modifying: `dict(table)`.
>
> Game length under v1.4 is 75 ticks, as under v1.3. The interaction model, payoff constants, speeds and radii are unchanged. Sections 4 and 7 of the README have the details.

### Release notes

#### [6.08]

##### Notice

- **Game rule is now v1.4.** Red no longer has the whole-map `agent` sensor. Every attacker written for v1.3 or earlier needs the two-line change shown in the announcement; one that is not changed runs without error and sees no defenders.
- **Submissions must pass `check_policy.py`**, and the report it writes (`reports/<policy>_<team>.html`) is part of the submission. README section 7.
- **Re-run `pip install -r requirements.txt`.** `cbor2`, `rasterio` and `cvxpy` are now listed (they were imported but undeclared). `torch`, `torch-geometric`, `stable-baselines3`, `gymnasium` and `tqdm` are commented out. `matplotlib` is capped below 3.12.
- **No policies from past rounds are bundled.** `policies/` holds only `excluded/`, with copies of the two example policies.

##### New

- **`config/rules/v1.4.yml`**, with a new `red_agent_r400` line-of-sight visibility model (`max_range: 400`). Red's sensor list is `global_map`, `candidate_flag`, `custom_team`, `egocentric_agent` (on `red_agent_r400`) and `egocentric_flag` (on `red_flag_r400`). Blue's is unchanged. Precomputed `red_agent_r400` tables for `osm_200_a`, `osm_200_b` and `osm_200_c` are in `graphs/visibility/`.
- **Sensor coverage from the sensor itself** (`lib/agent/agent_core.py`): `agent_ctrl.coverage(state, name)` returns the nodes sensor `name` covers now, `agent_ctrl.coverage_from(state, name, node)` the nodes it would cover from `node`. Both return `None` for a sensor with no range. The merged `state["sensor"]["stationary"]` payload gains `region`, the towers' combined coverage.
- **`check_policy.py`**, with `lib/utils/perf_report.py` and `config/perf_budget.yml`: plays a policy over `config/example_configs/`, times every strategy call in machine units, prints a pass/fail table and writes an HTML report. Options: `--quick` (one game per size), `--calibrate` (measure, no verdict), `--opponent`, `--budget`, `--out`.
- **`lib/core/readonly.py`**: `global_map`'s `apsp` and a region sensor's `table` are now `ReadOnlyDict`. They are shared by every game in a process; a write used to corrupt later games silently and now raises `TypeError`. `isinstance(x, dict)` still holds and reads are unchanged.
- **`GameEngine.launch_from_files`** accepts strategy module objects as well as import paths, as its signature always said.

##### Changed

- **Flag discovery** (`lib/game/interaction_engine.py`): with a model-backed `egocentric_flag`, a flag counts as discovered only when that sensor's region contains it. Plain euclidean distance within `sensing_radius` remains only for rule v1.2's string sensor.
- **`example/example_atk.py`, `example/example_def.py`**: rewritten decision logic (see the announcement). The attacker no longer uses `random`, so a game between the two is deterministic. The sensor walkthrough at the top of each file now shows the v1.4 reads and the coverage calls. `policies/excluded/` holds the same two files.
- **`config/example_configs/osm_a/`** (6 scenarios x 5 runs) regenerated from `v1.4.yml`: 75 ticks, new sensor lists, agent and flag positions resampled. Previews in `config/example_visuals/` are `.webp` instead of `.png`.
- **`main.py` and `launch_gui.py`** default to `config/example_configs/osm_a/R3B3F3-5/R3B3F3-5_run0.yml`.
- **`state["team_cache"]`** (`lib/game/game_engine.py`) is the shared team cache. It was each agent's private cache.
- **A sensor's range has one source.** The visibility model owns it; `sensing_radius` and `environment.blue_stationary_sensor_radius` are overwritten from the models at startup, with a warning if a config disagreed (`lib/core/visibility_lookup.py`).
- **Headless games are several times faster.** The time went to the range sensor behind the old `_region` entries.
- **Under rule v1.3**, the `_region` sensors now report the occluded region of their companion sensor instead of a plain radius. Rule v1.2 is unchanged.
- `lib/visual/agent_visual.py` and `flag_visual.py` share a new `lib/visual/node_visual.py` base. No behaviour change.

##### Removed

- **Red's `agent` sensor**, from rule v1.4. Rules v1.2 and v1.3 still have it.
- **`egocentric_flag_region` and `egocentric_agent_region`**, from rule v1.4. Use `coverage()` / `coverage_from()`.
- **`example/example_config.yml`** and **`config/debug.yml`**. Use a config under `config/example_configs/`.
- **`policies/v1.2r4/`**, the previous round's policies.

Full Changelog: v6.05.02...v6.08

### Reference detail

**The `strategy(state)` / `map_strategy(agent_config)` contract is unchanged.** No signature changes, no new required return values.

#### Red's sensor list, v1.3 → v1.4

| | v1.3 | v1.4 |
|---|---|---|
| Defenders | `agent` — whole map, every tick | `egocentric_agent` / `red_agent_r400` — line of sight only |
| Teammates | `agent` | `custom_team` — whole map |
| Flags | `egocentric_flag` / `red_flag_r400` | unchanged |

`red_agent_r400` is a new `type: line_of_sight` visibility model in `config/rules/v1.4.yml`, `max_range: 400` — matching red's existing `sensing_radius`. Note the asymmetry this creates: red detects defenders at 400, blue detects attackers at 250. That is deliberate.

Blue's sensor list is unchanged from v1.3, apart from losing `egocentric_agent_region`.

#### Reading coverage, before and after

```python
# before
nodes = agent_ctrl.sensor_data(state, "egocentric_flag_region")["nodes"]

# after
nodes = agent_ctrl.coverage(state, "egocentric_flag")                # covered now
there = agent_ctrl.coverage_from(state, "egocentric_flag", node_id)  # covered from another node
```

Both return a set of node ids, or `None` for a sensor with no range. A policy that still reads the old names keeps running if it guards with `if "..._region" in sensors`, but sees nothing. The old entries reported a plain radius with no occlusion: on `osm_200_a`, 41% of the nodes they listed for red and 28% for blue were ones the matching sensor could not see.

#### Adding a rule version (for anyone forking the engine)

A new rule needs two code edits beyond the YAML file: `lib/game/interaction_engine.py` `_get_flags()` and `lib/visual/flag_visual.py` both branch on `game_rule`. An unregistered value matches neither branch and `_get_flags()` raises `UnboundLocalError` — it does not fall back.

### How this was verified

**Rule v1.4**

- **Occlusion is working, not merely enabled.** Over 6 games on `osm_a`
  R5B5F3-5, red sees a defender on 18% of ticks under v1.4 against 100% under
  v1.3, with at least one sighting in every game.
- **The discovery fix changes discovery and nothing else.** The same five
  games, same fixed RNG seed, were run before and after it, comparing game
  length, both payoffs, flags discovered and survivors per side. The only
  differences are flag discovery and the 0.1-per-flag payoff that follows from
  it.
- **The speedup and the refactors change nothing observable.** The same
  comparison across those commits is identical on every field.

**Runtime check**

- **Both example policies** pass, at about 2% of the mean limit, and all 30
  example configs run with no strategy error.
- **Failure paths**, each with a purpose-built policy: one that enumerates
  joint actions (exponential in team size) fails on the mean and the slowest
  call at R5B5 against a tightened budget and gets the super-linear warning;
  one that raises fails naming the exception, agent and tick; one stuck in an
  infinite loop is stopped by the game time limit.
- **Two machines.** 13 policies were run on two laptops whose speed differs
  1.75x. In machine units the mean per call agrees within 0.96-1.39 on 77 of
  78 rows, and every verdict is the same on both. Short single calls do not
  agree between machines (random spikes of about 10 ms), which is why the
  slowest-call limit is set 30x above that.
- **The package itself.** `League-6.08.zip` was unpacked to a clean folder and
  the check run from there.
- **Not verified:** Python 3.11, Linux and Windows. Everything above is Python
  3.12 on macOS.

### Full changelog (since v6.05.1)

Rule v1.4:

- **`ea92620`** — Add game rule v1.4. New `config/rules/v1.4.yml` with the `red_agent_r400` line-of-sight model; red's `agent` sensor replaced by `custom_team` + occlusion-limited `egocentric_agent`. Registered `"v1.4"` in `interaction_engine._get_flags()` and `flag_visual.py`.
- **`d509b45`** — Regenerate all 30 `config/example_configs/osm_a/` configs and previews from `v1.4.yml`. Agent and flag placements are resampled, not preserved.
- **`5164348`** — Trim `requirements.txt`; stop `visualization_engine_new.py` swallowing every HUD render error.
- **`6a62f7b`**, **`7fd8565`**, **`dc6bc0d`**, **`9d99906`** — Preview images move to WebP; `matplotlib` capped below 3.12 so previews render the same everywhere.
- **`a63db9c`** — Delete `example/example_config.yml` and keep every config under `config/`. It was hand-maintained, had drifted to rule v1.2, and was the default for both `main.py` and the GUI.
- **`a9493a0`** — Delete `config/debug.yml`, a leftover v1.2 config that still showed in the GUI dropdown.

Fixes to v1.4 and engine work:

- **`2dba0c9`** — `state["team_cache"]` handed out the private per-agent cache where the shared one is documented. Also stopped re-deriving the graph signature on every shortest-path cache hit, which cost 119 µs per agent per tick.
- **`c48c1bc`** — Shared APSP and visibility tables are read-only, through a `dict` subclass so `isinstance(x, dict)` still holds.
- **`069aa7d`** — Coverage read from the sensor: `coverage()` / `coverage_from()`, `stationary["region"]`, `_region` entries dropped from `v1.4.yml` and the 30 example configs, and `<name>_region` reimplemented as a view of `<name>` for the rules that still list it.
- **`f7666b6`** — `sync_ranges_to_models()` in `lib/core/visibility_lookup.py`: the visibility model is the single source of a sensor's range.
- **`7513b2c`** — Flag discovery follows the attacker's flag sensor.
- **`5a8d7e9`** — `launch_from_files` takes module objects.
- **`b258f6c`**, **`f8fea19`** — `launch_from_files` split into named steps; `AgentVisual` and `FlagVisual` share `lib/visual/node_visual.py`. No behaviour change.
- **`055a32d`** — Red agent line-of-sight tables for `osm_200_b` and `osm_200_c`.

Runtime check and example policies:

- **`0cca172`** — Example attacker searches, shares flag knowledge through the team cache and avoids defenders; example defender guards the most dangerous flag and intercepts. Tower sightings now reach the shared map. `policies/excluded/` copies resynced.
- **`7d3384c`** — `check_policy.py`, `lib/utils/perf_report.py` (self-contained HTML report) and `config/perf_budget.yml`.

---

## v6.05 / v6.05.1 — sensor and visibility redesign

### Announcement (ready to post)

> @channel Hi all, a new release of the League has been posted.
>
> **[League Update — v6.05] - [GAMMS Version: 1.0.0]**
>
> Real building-occlusion visibility is now available, plus one small payload change if your strategy reads a sensor configured in the newer dict form.
>
> **New:**
>
> - **Building-occlusion (line-of-sight) visibility.** A `visibility_models` entry can now use `type: line_of_sight` — visibility is computed from real building geometry (fetched from OpenStreetMap for the map area), not a plain radius circle. See the regenerated configs under `config/example_configs/osm_a/` for the format.
> - Sensors using the newer dict config form (`{name, model, carrier, team, flags}`) now include a `region` key — the set of node IDs currently visible from that sensor's origin.
> - GUI: new "Occlusion visuals" checkbox in `launch_gui.py` (**off by default**, so the window stays responsive) — check it to see the new sight-polygon rendering live.
>
> **Changed:**
>
> - Sensors in the newer dict config form no longer return a `flag_count` key — use `len(detected_flags)` instead. This **only** affects sensors declared in the new dict form; plain string sensor entries (the common case) are unaffected.
> - `main.py`'s example config now uses occlusion sensors — running it unmodified will show real building-blocked visibility instead of a flat radius.
>
> **Improved:**
>
> - Sensor internals were unified under the hood — radius, k-hop, and line-of-sight visibility are now all just different ways of building the same kind of table, sharing one sensor implementation instead of one class per type. No config or strategy changes needed; configs are simpler to write going forward.
> - Live visualization and the `mass_eval` batch-preview tool both now draw the real visibility shape (with realistic notches where buildings block view) instead of a flat circle, so previews match what the game actually sees.
> - Fixed a video-recording bug where the animation-alpha update was silently failing against a stale internal attribute name from the old gamms version.
>
> **Heads-up:** if your strategy reads `flag_count` directly off a flag sensor *and* that sensor is configured in the new dict form, switch to `.get("flag_count", len(detected_flags))` (works on both old and new payloads). If you're only using plain-string sensor configs, nothing to change.

### Announcement (v6.05.1, ready to post)

> @channel Small follow-up to the visibility-graphs release.
>
> **[League Update — v6.05.1] - [GAMMS Version: 1.0.0]**
>
> One new capability for anyone using the newer dict-form sensors — no breaking changes.
>
> **New:**
>
> - Every sensor in the new dict config form now also carries a `table` key alongside `region` — the **full** node → visible-nodes map for that model, not just your current position's slice. Same sensor, no separate name to learn: `agent_ctrl.sensor_data(state, "egocentric_flag")["table"]` gives you the whole visibility graph for `red_flag_r400`, so you can look up "what would I see from node X" for any node, not just where you're standing. It's static for the whole game, so it's safe to cache once. `state["sensor"]["stationary"]` also now carries a `table` key (the shared tower model), alongside the existing `enemies`/`teammates`/`detections`.
> - If two sensor entries in your config accidentally reuse the same `name:`, you'll now get a warning instead of the second one silently overwriting the first in `state["sensor"]`.
>
> **Heads-up:** none — this is additive. `region`, `enemies`, `teammates`, `detected_flags`, `detections` all mean exactly what they did before; `table` is new alongside them. Plain-string sensor entries are unaffected (no `table` key, same as before).
>
> See `example/example_atk.py` / `example/example_def.py` for a worked example of reading `table`.

### Reference detail (backing the announcement above)

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
