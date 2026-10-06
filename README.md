# League-AD: Quick User Guide

This README covers installation, running games, choosing strategies, and editing game configs.

---

## 1. Installation

**Python 3.11 or 3.12 required.** Python 3.13+ has known compatibility issues with the `gamms` visualization engine.

This version targets **`gamms` 1.0.0** (the current PyPI release) — `pip install -r requirements.txt` pulls it in automatically.

### Install Python 3.11/3.12

**macOS** — install via Homebrew:

```bash
brew install python@3.12 python-tk@3.12
```

`python-tk@3.12` provides `tkinter`, which is required by the GUI launcher. If `python3.12` isn't on your `PATH` afterward, add this to your shell profile (`~/.zprofile` for zsh, `~/.bash_profile` for bash):

```bash
export PATH="/opt/homebrew/opt/python@3.12/libexec/bin:${PATH}"
```

**Windows** — download Python 3.12 from [python.org](https://www.python.org/downloads/) and run the installer. Check **"Add python.exe to PATH"** during install. `tkinter` is bundled by default. Use the `py` launcher to target the right version, e.g. `py -3.12 -m venv .venv`.

**Linux (Debian/Ubuntu)**:

```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3-tk
```

On other distros, install the equivalent `python3.12` and `python3-tk` (or `python3-tkinter`) packages via your package manager.

### Set up a virtual environment

A virtual environment keeps project dependencies isolated from your system Python.

```bash
cd league_dev
python3 -m venv .venv     # Windows: py -3.12 -m venv .venv
```

Activate it:

| Platform | Command |
| --- | --- |
| macOS / Linux | `source .venv/bin/activate` |
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| Windows (cmd) | `.venv\Scripts\activate.bat` |

Your prompt will show `(.venv)` when active. Run this activation line each time you open a new terminal.

### Install dependencies

```bash
pip install -r requirements.txt
```

| Group | Packages | Required for |
| --- | --- | --- |
| Core simulation | `gamms`, `networkx`, `numpy`, `typeguard`, `pyyaml`, `shapely` | All game modes |
| GUI launcher | `matplotlib` | `launch_gui.py` |
| Strategy policies | `scipy` | Policies under `policies/` |
| Analysis (optional) | `pandas`, `seaborn`, `imageio`, `cvxpy` | Not needed to run games or the runtime check |

> The analysis packages are optional — the core engine, the GUI and `check_policy.py` run without them.

---

## 2. Run a game

### GUI launcher (recommended)

```bash
python launch_gui.py
```

A window opens with dropdowns for config, strategies, log settings, and recording options. Select what you want and click **RUN GAME**. The right panel renders a live map preview of the selected config.

### Headless / scripted

```bash
python main.py
```

Edit `main.py` directly to hard-code your selections (see section 3).

---

## 3. What to edit in `main.py`

```python
result = GameEngine.launch_from_files(
    config_main="config/example_configs/osm_a/R3B3F3-5/R3B3F3-5_run0.yml",
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

## 4. Strategy modules

Strategies live under `example/` or `policies/attacker|defender/` and are referenced by Python module path:

- `example.example_atk` / `example.example_def`
- `policies.attacker.gmu_atk_r3`
- `policies.defender.uncc_def_F_r3`

`policies/attacker/` and `policies/defender/` are empty at the moment — the
previous round's policies were archived when game rule v1.4 landed. Start from
`example/example_atk.py` and `example/example_def.py`, which show the current
sensor reads. They are also simple working opponents: the attacker searches
the candidate flags as a team and keeps out of defenders' reach; the defender
guards the most dangerous flag and steps out to tag an attacker that comes
near it.

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

## 5. Config file guide

Game configs live under `config/example_configs/<map>/<scenario>/` and are generated from a rule template in `config/rules/`. Copy one as a starting point rather than writing a config from scratch. `example/` holds strategies only.

| Section | Key fields |
| --- | --- |
| `game` | `game_rule`, `max_time`, `interaction_model`, `payoff_model` |
| `agents.red_global` / `blue_global` | `speed`, `sensing_radius`, `capture_radius` / `tagging_radius`, `sensors` |
| `agents.red_config` / `blue_config` | Each agent's `start_node_id` |
| `flags` | `real_positions`, `candidate_positions` |
| `environment` | `graph_name` (file in `graphs/`), stationary sensor settings |

Configs live in `config/` (excluding `archive/` and `rules/`). The GUI auto-discovers all of them.

---

## 6. Typical workflow

1. Pick a config in the GUI (or copy one from `config/example_configs/` and edit it).
2. Select attacker and defender strategies from the dropdowns.
3. Set logging, recording, and visualization options.
4. Click **RUN GAME**.

---

## 7. Check your policy's runtime before submitting

A policy that is too slow to evaluate is not accepted into a tournament. Check
yours on your own machine first:

```bash
python check_policy.py policies.attacker.my_atk --team red
python check_policy.py policies.defender.my_def --team blue
python check_policy.py policies.attacker.my_atk --team red --quick   # 1 game per size, for a fast look
```

It plays your policy over all 30 configs in `config/example_configs/` against
the example policy for the other team, times every call to your `strategy`,
and prints a table. A full run of a fast policy takes well under a minute. It
also writes a report you can open in a browser:
`reports/<policy>_<team>.html`. **Send that file with your submission**, one
for each policy.

**Limits.** Times are measured in *machine units*: the script first times a
fixed workload on your machine (about a second) and divides every measurement
by it, so the same limits hold on a laptop and on a server.

| Limit | Machine units | What it covers |
| --- | --- | --- |
| Mean per call | 0.01 | Average over all calls after each agent's first |
| Slowest call | 0.5 | Any single call after each agent's first |
| First call | 2.0 | Each agent's first call, where one-time setup belongs |

The limits are in `config/perf_budget.yml`.

**What fails a policy:**

- Any limit exceeded at any team size (R1B1, R3B3, R5B5).
- Any exception from `strategy`, a return value that is not a `str`, or an
  action that is not an `int`. In a normal game the engine catches these and
  leaves the agent standing still; the check does not let that pass.
- A single call longer than 20 units, or a game longer than 200 units. Either
  one stops the check.

**Notes:**

- Work your policy saves to disk and reuses is counted on the run that does
  the work. Run the check from a clean state to see what a first run costs.
- The report also shows your mean payoff against the example opponent. It is
  there so you can see that the policy is playing, not as part of the verdict.
- On Windows a call that never returns has to be stopped with Ctrl-C.

---

## 8. Quick checks if something fails

| Symptom | Likely cause |
| --- | --- |
| Import error for strategy | Module path is wrong — must be a Python import path, not a file path |
| Agents not moving | Strategy did not set `state["action"]`, or target is out of speed range |
| `TypeError` on strategy return | Strategy returned something other than `str` |
| `TypeError` on action | `state["action"]` was not set to `int` |
| No visuals | Check `vis=True` and `extra_defs="config/game_config.yml"` |
| No map preview in GUI | Graph `.pkl` file for that config is not present in `graphs/` |
