# League-AD: Quick User Guide

This README is focused on running games through `main.py`, choosing strategies, and editing game configs.

## 1. Run a game

From the project root:

```bash
python main.py
```

`main.py` launches one game using `GameEngine.launch_from_files(...)`.

## 2. What to edit in `main.py`

Open `main.py` and set these fields:

```python
result = GameEngine.launch_from_files(
    config_main="config/test1.yml",
    extra_defs="config/game_config.yml",
    red_strategy="policies.attacker.gatech_atk_r3",
    blue_strategy="policies.defender.gatech_def_r3",
    log_name=None,
    record=False,
    vis=True,
)
```

- `config_main`: Main game setup YAML (agents, flags, map, rules).
- `extra_defs`: Extra shared settings (currently used for visualization settings).
- `red_strategy`: Python module path for attacker strategy.
- `blue_strategy`: Python module path for defender strategy.
- `log_name`: Set `None` to disable logging, or a string to save logs.
- `record`: `True` to record a game file.
- `vis`: `True` for visualization, `False` for headless/no-visual mode.

## 3. Strategy modules (important)

Strategies are imported by module path strings, for example:

- `example.example_atk`
- `example.example_def`
- `policies.attacker.uncc_atk_F_r3`
- `policies.defender.gmu_def_r3`

Each strategy module should expose:

1. `strategy(state)`  
   - Sets `state["action"]` to the target node for this turn.
   - Usually returns a `set` (attackers often return discovered flag IDs).
2. `map_strategy(agent_config)`  
   - Returns a dict mapping each agent name (like `red_0`) to a strategy function.

Minimal template:

```python
def strategy(state):
    current_pos = state["curr_pos"]
    state["action"] = current_pos  # stay in place
    return set()

def map_strategy(agent_config):
    return {name: strategy for name in agent_config.keys()}
```

## 4. Config file guide (`example/example_config.yml`)

Use `example/example_config.yml` as a starting point. Main sections:

- `game`: Rule version, max time, interaction/payoff model.
- `agents`:
  - `red_global` / `blue_global`: team-level capabilities (speed, sensing, radii, sensors).
  - `red_config` / `blue_config`: each agent’s `start_node_id`.
- `flags`:
  - `real_positions`: true flags.
  - `candidate_positions`: candidate flag locations.
- `environment`:
  - `graph_name`: graph file from `graphs/`.
  - stationary sensor settings for blue team.
- `generator`: metadata (can usually be left as-is).

## 5. Typical workflow

1. Copy `example/example_config.yml` to a run config (for example `config/test1.yml`).
2. Edit start nodes, flags, and map as needed.
3. Choose strategy modules in `main.py`.
4. Run `python main.py`.

## 6. Quick checks if something fails

- Import error for strategy: module path is wrong (must be Python import path, not file path).
- Agents not moving: strategy did not set `state["action"]`, or target is invalid/out of speed range.
- No visuals: check `vis=True` and keep `extra_defs="config/game_config.yml"`.

