# Changelog

## [6.05]

### Notice

- **New dependencies**: `osmnx>=2.0` and `pyproj>=3.6` added to `requirements.txt` (used to fetch real OpenStreetMap building footprints for the new occlusion visibility feature — see below). Re-run `pip install -r requirements.txt` to pick them up.

### New

- **Building-occlusion (line-of-sight) visibility.** `environment.visibility_models` entries can now declare `type: line_of_sight` in addition to the existing `radius`/`khop` types:
  ```yaml
  environment:
    visibility_models:
      red_flag_r400:
        type: line_of_sight
        crs: EPSG:32618      # must match the graph's coordinate system
        max_range: 400        # optional cap, graph coordinate units
  ```
  Visibility between two nodes is computed from real building geometry (fetched once from OpenStreetMap for the graph's area, cached to `graphs/buildings/`) instead of a euclidean radius — a segment between two nodes is blocked if it crosses a building footprint. Precomputed tables for `osm_200_a`/`osm_200_b`/`osm_200_c` ship in `graphs/visibility/`.
- **Unified sensor architecture.** New `lib/sensor/base_sensor.py` (shared `Sensor` base class), `lib/sensor/info_sensor.py` (aspatial info sensors: `global_map`/`candidate_flag`/`flag`), and `lib/sensor/region_sensor.py` (generic table-backed spatial sensor — the same class handles `radius`, `khop`, and `line_of_sight` models; only the table-generator differs). New `lib/core/visibility_generators.py` (radius/khop/line_of_sight table generators), `lib/core/visibility_cache.py` (load-or-build table persistence, keyed by generator spec so a config change invalidates a stale table), `lib/core/visibility_lookup.py` and `lib/core/visibility_polygon.py` (cheap config-only resolvers and a ray-casting visibility-polygon algorithm, used by the visualization layer — see below).
- **Unified sensor config format.** `agents.<team>_global.sensors` entries can now be a dict instead of a plain string: `{name: egocentric_flag, model: red_flag_r400, carrier: agent, team: red, flags: real}`. `model` names an entry in `environment.visibility_models`. Plain string entries (`egocentric_flag`, `stationary`, ...) are still fully supported and behave exactly as before — this is additive, not a required migration. See `config/rules/v1.3.yml` and the regenerated `config/example_configs/osm_a/` for a fully-migrated example.
- **New rule template `config/rules/v1.3.yml`** — the occlusion-enabled counterpart of `v1.2.yml`, using `type: line_of_sight` for all three of the example scenario's sensors (`red_flag_r400`, `blue_agent_r250`, `blue_tower_r450`).
- **`config/game_config.yml`**: new `visualization.show_occlusion_visuals` flag (default `true`) — disables the occlusion polygon/glow rendering (real per-frame geometry, the most expensive part of the new visualization) while leaving the rest of visualization (agents, flags, HUD) on. Useful for a faster interactive run.
- **`lib/gui/launcher.py`**: new "Occlusion visuals (slower)" checkbox, **unchecked by default** so the GUI stays responsive; check it to see the new sight-polygon rendering live.
- **New `lib/visual/building_visual.py`** — draws the building footprints behind any `line_of_sight` model on the map, so the occlusion is visible, not just its effect.

### Changed

- **Sensor payload shape** for any sensor using the new dict config form: it now includes a `region` key (the set of node IDs currently visible from that sensor's origin) and no longer includes a `flag_count` key. Strategies reading `flag_count` directly should switch to `.get("flag_count", len(detected_flags))` (works against both old and new payloads — see `example/example_atk.py`). Plain-string sensor entries are unaffected. The `stationary` sensor aggregation (`state["sensor"]["stationary"]`) shape is unchanged (`{detections, enemies, teammates}`) regardless of which sensor form its towers use.
- **`lib/visual/agent_visual.py` / `lib/visual/flag_visual.py`**: sensor-region rendering rewritten. For a `line_of_sight`-backed sensor, draws the real ray-traced visibility polygon (via `lib/core/visibility_polygon.py`) plus small soft-glow markers at nodes actually confirmed visible (from the sensor's precomputed table), instead of a flat radius disk or per-node halo. Rendering is layered below the base graph's own node markers (`layer=9` vs. gamms' fixed `layer=10`) so nodes stay visually crisp. For `radius`/`khop` models or legacy string sensors, rendering is unchanged. Both files gate all of this behind `vis` and the new `show_occlusion_visuals` flag, so a headless (`vis=False`) run never fetches building data or computes geometry.
- **`example/example_config.yml`**: `egocentric_flag`/`egocentric_agent`/`stationary` sensors converted to the new dict form with `type: line_of_sight` models — running `main.py` unmodified now demonstrates real building-occlusion visibility.
- **`example/example_atk.py`**: one-line change — `flag_count` now derived via `.get("flag_count", len(detected_flags))` instead of a direct `["flag_count"]` read.
- **`config/example_configs/osm_a/`** (all 6 scenarios × 5 runs) and their `config/example_visuals/osm_a/` previews regenerated to use the new occlusion sensor form.

### Removed

- `lib/sensor/candidate_flag_sensor.py`, `lib/sensor/flag_sensor.py`, `lib/sensor/global_map_sensor.py` — superseded by `lib/sensor/info_sensor.py` (same payloads, one shared implementation instead of three near-identical classes). `lib/sensor/flag_range_sensor.py`, `lib/sensor/stationary_sensor.py`, and `lib/sensor/team_sensor.py` are unchanged and still used by any config still on the legacy string sensor form.

## [6.04.01]

### Notice

- **This version now targets `gamms` 1.0.0** (the current PyPI release) and is **no longer compatible with previous versions of `gamms`**. `requirements.txt` now pins `gamms>=1.0.0`. The internal visual API changed (`ctx.visual._get_target_surface()` no longer takes a `layer` argument, agent/sensor artists now use `ArtistType.DYNAMIC`), so games will not run correctly against older `gamms` installs — re-run `pip install -r requirements.txt` in your environment to upgrade.

### Changed

- **Python version requirement updated to 3.11 or 3.12** (previously "3.10+ recommended"). Python 3.13+ has known compatibility issues with the `gamms` visualization engine. README (`README.md`) now includes full setup instructions for installing Python 3.11/3.12 and creating a virtual environment on macOS, Windows, and Linux.
- `requirements.txt`: added `pygame>=2.6.1`, `imageio[ffmpeg]>=2.28`, `tqdm>=4.65`, and a new optional "MIT specific" section (`torch`, `torch-geometric`, `stable-baselines3`, `gymnasium`, plus updated `networkx`/`numpy` floors). Added comments pointing to the README for Python setup and noting `tkinter` is required for the GUI launcher.
- Real-time video recording overhaul (`lib/game/game_engine.py`): `record_video` now patches `simulate()` directly and manually drives `handle_input` / `handle_single_draw` / display flip for a fixed number of animation frames per game step. New `visualization.video_frames_per_step` config option (`config/game_config.yml`, default 5) controls how many frames are rendered per step, so recorded video has consistent pacing regardless of wall-clock game speed. Default `visualization.video_fps` raised from 2 to 20.
- Agent and sensor-circle artists (`lib/visual/agent_visual.py`) switched from `ArtistType.AGENT` to `ArtistType.DYNAMIC`, and now read their interpolated position from the agent dot's already-computed `current_position` (which accounts for edge linestrings) instead of manually lerping between `prev_node_id` and `current_node_id`.
- Updated all `ctx.visual._get_target_surface(layer)` calls to `ctx.visual._get_target_surface()` for the `gamms` 1.0.0 API (`lib/visual/agent_visual.py`, `lib/visual/flag_visual.py`, `lib/visual/map_overlay_visual.py`, `lib/game/visualization_engine_new.py`).
- Agent colors in `config/game_config.yml` are now RGB tuples (`red_global.color: [220, 50, 50]`, `blue_global.color: [50, 100, 220]`) instead of named color strings; `lib/visual/agent_visual.py` now reads the team color directly from each team's `*_global.color`.
- Increased sensor radius circle transparency from 0.1 to 0.15 alpha (`lib/visual/agent_visual.py`, `lib/visual/flag_visual.py`).
- `lib/game/game_engine.py`: added `_red_strategy_errors` / `_blue_strategy_errors` counters, incremented whenever a strategy raises during execution. Also added a startup warning if the running Python version is outside the recommended 3.11/3.12 range.
- `lib/game/interaction_engine.py`: fixed a typo and downgraded the "only accepts red-vs-blue CTF-style games" message from `warning` to `info`.
- `main.py` simplified to launch from `example/example_config.yml` with `example.example_atk` / `example.example_def` as the default strategies.
- Removed the unused legacy waypoint configs: `config/wp_1v1.yml`, `config/wp_1v1_v2.yml`, `config/wp_2v2.yml`, `config/wp_2v2_v2.yml`.
- Reorganized policies: moved `example_atk.py` / `example_def.py` out of `policies/attacker` and `policies/defender` into `policies/excluded/`, and moved round-3/4 strategy policies (incl. `msu_atk_M_r3.py`, `msu_def_M_r2.py`, and other team submissions) into new `policies/v1.2r4/attacker/` and `policies/v1.2r4/defender/` directories.
- `.gitignore`: now ignores `videos/`, `.dropboxignore`, and `.venv-*/`.
