# League-AD: Quick User Guide

This README covers running games, choosing strategies, and editing game configs.

---

## 1. Run a game

### GUI launcher (recommended)

```bash
python launch_gui.py
```

A window opens with dropdowns for config, strategies, log settings, and recording options. Select what you want and click **RUN GAME**. The right panel renders a live map preview of the selected config.

### Headless / scripted

```bash
python main.py
```

Edit `main.py` directly to hard-code your selections (see section 2).

---

## 2. What to edit in `main.py`

```python
result = GameEngine.launch_from_files(
    config_main="example/example_config.yml",
    extra_defs="config/game_config.yml",
    red_strategy="example.example_atk",
    blue_strategy="example.example_def",
    log_name=None,
    set_level=LogLevel.WARNING,
    record_file=False,
    record_video=False,
    vis=True,
)
```

| Parameter | Description |
| --- | --- |
| `config_main` | Main game setup YAML (agents, flags, map, rules) |
| `extra_defs` | Shared settings file (visualization, etc.) |
| `red_strategy` | Python module path for the attacker strategy |
| `blue_strategy` | Python module path for the defender strategy |
| `log_name` | `None` to disable logging, or a string filename |
| `set_level` | Log verbosity: `LogLevel.DEBUG / INFO / WARNING / ERROR` |
| `record_file` | `True` to save a `.ggr` replay file |
| `record_video` | `True` to capture a real-time MP4 |
| `vis` | `True` for visualization, `False` for headless mode |

---

## 3. Strategy modules

Strategies live under `example/` or `policies/attacker|defender/` and are referenced by Python module path:

- `example.example_atk` / `example.example_def`
- `policies.attacker.gmu_atk_r3`
- `policies.defender.uncc_def_F_r3`

### Required interface

Every strategy module must expose two functions:

```python
def strategy(state: dict) -> str:
    # Read state, set action, return a log string
    state["action"] = current_pos   # int — required
    return "holding position"       # str — required

def map_strategy(agent_config: dict) -> dict:
    return {name: strategy for name in agent_config.keys()}
```

**Contracts enforced at runtime:**
- `state["action"]` must be set to an `int` (target node ID).
- Return value must be a `str` (shown in logs at INFO level, no-op otherwise).

### Minimal working template

```python
def strategy(state: dict) -> str:
    current_pos = state["curr_pos"]
    state["action"] = current_pos
    return "holding position"

def map_strategy(agent_config):
    return {name: strategy for name in agent_config.keys()}
```

---

## 4. Config file guide

Use `example/example_config.yml` as a starting point.

| Section | Key fields |
| --- | --- |
| `game` | `game_rule`, `max_time`, `interaction_model`, `payoff_model` |
| `agents.red_global` / `blue_global` | `speed`, `sensing_radius`, `capture_radius` / `tagging_radius`, `sensors` |
| `agents.red_config` / `blue_config` | Each agent's `start_node_id` |
| `flags` | `real_positions`, `candidate_positions` |
| `environment` | `graph_name` (file in `graphs/`), stationary sensor settings |

Configs live in `config/` (excluding `archive/` and `rules/`). The GUI auto-discovers all of them.

---

## 5. Typical workflow

1. Pick a config in the GUI (or copy `example/example_config.yml` and edit it).
2. Select attacker and defender strategies from the dropdowns.
3. Set logging, recording, and visualization options.
4. Click **RUN GAME**.

---

## 6. Quick checks if something fails

| Symptom | Likely cause |
| --- | --- |
| Import error for strategy | Module path is wrong — must be a Python import path, not a file path |
| Agents not moving | Strategy did not set `state["action"]`, or target is out of speed range |
| `TypeError` on strategy return | Strategy returned something other than `str` |
| `TypeError` on action | `state["action"]` was not set to `int` |
| No visuals | Check `vis=True` and `extra_defs="config/game_config.yml"` |
| No map preview in GUI | Graph `.pkl` file for that config is not present in `graphs/` |
