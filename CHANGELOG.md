# Changelog

## [6.02.12]

### Changed

- Stationary sensor payload now pre-filtered by team (`lib/game/game_engine.py`). `state["sensor"]["stationary"]` is a dict with `enemies` / `teammates` / `detections` keys instead of a raw list — strategies no longer need to loop and color-filter manually.
- Updated `example_def` (`example/example_def.py`) to demonstrate the new stationary sensor API.
- Renamed sensors in `config/rules/v1.2.yml`: red `egocentric_map` → `egocentric_flag_region`; blue `egocentric_map` → `egocentric_agent_region`.
- HUD text size and spacing (`config/game_config.yml`): increased `line_height` 18 → 28 px and `font_size` 18 → 25 px for better readability.
