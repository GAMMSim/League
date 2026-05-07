# Changelog

Tracks code changes to example strategies (`example/example_atk.py`, `example/example_def.py`) and game engine infrastructure.

---

## [6.02.08]

### Added

- **GUI launcher** (`launch_gui.py`, `lib/gui/`): cross-platform Tkinter launcher replaces manual commenting/uncommenting in `main.py`.
  - Auto-discovers configs (excluding `archive/` and `rules/`), attacker/defender policies, and extra-def files from the repo at startup.
  - Live map preview rendered directly via matplotlib (`FigureCanvasTkAgg`) — anti-aliased, sharp on retina and HiDPI displays; updates in a background thread so the window stays responsive.
  - Config info panel (graph name, attacker/defender counts, flag/candidate counts, max time) shown on the left alongside all controls.
  - Strategy names shown as bare stems (`gmu_atk_r3`) with full module paths resolved at run time.
  - Window auto-focuses and switches Mission Control spaces on macOS (`osascript`); topmost-toggle fallback on Windows/Linux.
  - Defaults to `example/example_config.yml` + `example_atk` / `example_def`.

### Changed

- **Enemy/teammate filtering moved out of strategies** (`example_atk.py`, `example_def.py`): removed all `agent_name.startswith(enemy_team)` color-string checks from strategy code.
  - `AgentController` now exposes an `enemy_team` property (derived from `self.team`) as the single source of truth.
  - `SensorEngine` tracks which registered sensor names return `{agent_name: node_id}` payloads (`agent_type_sensor_names`).
  - `GameEngine.execute_agent_strategies` pre-splits those sensor payloads into `{"enemies": {...}, "teammates": {...}}` before handing state to strategies.
  - Strategies now access `state["sensor"]["agent"][1]["enemies"]` / `state["sensor"]["egocentric_agent"][1]["enemies"]` directly.
- **Stationary sensor consolidation** (`example_def.py`): game engine now pre-collects all `stationary_*` payloads into a single list under `state["sensor"]["stationary"]` before calling the strategy; strategies access it via `agent_ctrl.sensor_data(state, "stationary")` with no loop needed.
- **Restructured example strategy layout** (`example_atk.py`, `example_def.py`): moved SENSORS section before AGENT MAP so all sensor data is read once with empty-collection defaults; AGENT MAP and decision logic reuse those variables with no duplicate reads or scattered guard clauses. Unified comment style across both files. No API calls added or removed.
- **Strategy action logging** (`example_atk.py`, `example_def.py`): strategies now return a string describing their action; game engine prints it via `info()` so it only appears when INFO level is active and is a no-op otherwise.
- **Team cache getter/setter API on `AgentController`** (`example_atk.py`, `example_def.py`): replaced direct `agent_ctrl.team_cache.get/set/update()` calls with `agent_ctrl.get_team()`, `agent_ctrl.set_team()`, and `agent_ctrl.update_team()` so strategies are decoupled from the `Cache` class implementation.
- **Purpose-specific map sensor names** (`example_atk.py`, `example_def.py`, `config/rules/v1.2.yml`, `example/example_config.yml`): renamed the generic `egocentric_map` sensor to `egocentric_flag_region` (attacker — pairs with `egocentric_flag`) and `egocentric_agent_region` (defender — pairs with `egocentric_agent`), making clear that each reveals the sensing region for its corresponding detection sensor. `SensorEngine` now handles any `_region`-suffixed sensor name as a `RANGE` type.
- **`sensor_data` helper on `AgentController`** (`example_atk.py`, `example_def.py`): replaced all raw `state["sensor"]["name"][1]` tuple-index accesses with `agent_ctrl.sensor_data(state, "name")`, keeping the GAMMS tuple convention in one place.
- **Type-checked strategy interface** (`example_atk.py`, `example_def.py`, `game_engine.py`): `strategy` now declares `(state: dict) -> str`; game engine enforces both contracts at runtime — raises `TypeError` if the return value is not `str` or if `state["action"]` is not `int`.

---

## [Potential Future Changes]

- **Strategy access restriction via `StrategyView` facade**: instead of passing the full `AgentController` to strategies, pass a `StrategyView` wrapper that only exposes the allowed public API. Strategies would be unable to reach internal/restricted attributes. The game engine retains full controller access. Makes the allowed strategy API explicit and self-documenting.
