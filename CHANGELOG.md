# Changelog

## [6.02.09]

### Added

- **HUD match-info header** (`lib/game/visualization_engine_new.py`, `lib/game/game_engine.py`): static header drawn above the payoff line each frame showing attacker/defender strategy names with agent counts (red/blue), flag count (green), config filename and game rule (black), separated from the payoff line by a 4 px gap.
  - `VisEngine.set_match_info(red_strategy, blue_strategy, config_main, config)` populates `_match_info_lines` once at game setup.
  - `GameEngine.launch_from_files` calls `set_match_info` right after `setup_game_visuals`.
- **`requirements.txt`**: added with packages organized into four groups — core simulation, GUI launcher, strategy policies, and mass-eval (optional).
- **README installation section**: new section 1 documents `pip install -r requirements.txt` with a table mapping each package group to its purpose.

### Changed

- **HUD text size and spacing** (`config/game_config.yml`): increased `line_height` from 18 → 28 px and `font_size` from 18 → 25 px for better readability and visual hierarchy.
- **Graph initialization** (`example/example_atk.py`, `example/example_def.py`, `lib/agent/agent_map.py`): wrapped graph attachment in a null check so it runs only on the first turn, avoiding redundant processing across shared agent maps; added a time-change guard to debug logging to prevent spam when time is unchanged; removed an unnecessary debug statement from the display update loop.
- **`lib/agent/agent_map.py`**: removed redundant warning log for empty agent-positions dictionary to reduce noise.
- **`lib/gui/launcher.py`**: simplified window focus — removed macOS-specific AppleScript handling and complex topmost-toggle logic; now uses standard `lift()` / `focus_force()` calls.
