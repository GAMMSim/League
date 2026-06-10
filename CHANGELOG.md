# Changelog

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
