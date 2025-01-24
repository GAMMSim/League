## Before Use

### Requirements

Ensure your platform has **Python 3.7+** installed along with `pip`.

### Installation

1. Clone or download the GitHub repositories `gamms` and `games`.
2. Navigate to the `gamms` folder and run the following command to install the `gamms` library:
   ```bash
   pip install .
   ```

### File Structure

The `gamms` folder contains the core library and **should not be modified**. The `games` folder contains examples and utilities. The expected structure of the `games` folder is as follows:

```bash
.
├── README.md
├── examples
│   ├── basic_example
│   │   ├── attacker_strategy.py
│   │   ├── config.py
│   │   ├── defender_strategy.py
│   │   ├── interaction_model.py
│   │   └── game.py
│   └── random_move
│       ├── attacker_strategy.py
│       ├── config.py
│       ├── defender_strategy.py
│       ├── interaction_model.py
│       └── game.py
└── utilities.py
```

### Notes:

- ✅ Ensure the `utilities.py` file is in the root of the `games` folder.
- ⚠️ If you encounter errors like `utilities.py not found`, verify that your root directory is correctly set to `.../games`.

---

## How to Use

### Config.py

The `config.py` file contains the customizable parameters for the game. Below are the sections and their usage:

#### 🎨 Color Parameters

```python
RED = "red"       # The color red
BLUE = "blue"      # The color blue
GREEN = "green"     # The color green
BLACK = "black"     # The color black
```

These color constants are used for defining agent colors, flag colors, and other visual elements. Ensure the colors are defined in `gamms/VisualizationEngine/__init__.py` and written in **lowercase**.

#### 🖥️ Simulation Interface Parameters

```python
WINDOW_SIZE = (1980, 1080)
GAME_SPEED = 1
DRAW_NODE_ID = True
VISUALIZATION_ENGINE = gamms.visual.Engine.PYGAME
```

- `WINDOW_SIZE (tuple)`: Specifies the game window dimensions in pixels.
- `GAME_SPEED (float)`: Controls the simulation speed. Higher values make the game slower.
- `DRAW_NODE_ID (bool)`: Indicates whether to display node IDs on the graph.
  > **Warning**: Enabling this significantly reduces rendering performance.
- `VISUALIZATION_ENGINE (gamms.visual.Engine)`: Specifies the visualization engine. Use `visual.Engine.NO_VIS` for a dry run without rendering.

#### 🗺️ Graph Parameters

```python
GRAPH_PATH = "graph.pkl"
LOCATION = "West Point, New York, USA"
RESOLUTION = 200.0
```

- `GRAPH_PATH (str)`: Path to the graph file, which must be a `.pkl` file.
  > **Note**: If the file does not exist, the program will generate it using the following properties:
  - `LOCATION (str)`: A real-world location to base the graph on.
  - `RESOLUTION (float)`: Specifies graph resolution; higher values result in lower node density.

#### 🎮 Game Parameters

1. **Sensors**

   Available sensor types:

   ```python
   MAP_SENSOR = gamms.sensor.SensorType.MAP
   AGENT_SENSOR = gamms.sensor.SensorType.AGENT
   NEIGHBOR_SENSOR = gamms.sensor.SensorType.NEIGHBOR
   ```

   - `MAP_SENSOR`: Senses all nodes and edges, excluding agent and flag data.
   - `AGENT_SENSOR`: Senses the names and positions of all agents, excluding their properties (e.g., speed, team).
   - `NEIGHBOR_SENSOR`: Senses the neighboring nodes of the agent's current position.

2. **Flags**

   ```python
   FLAG_POSITIONS = [50, 51, 52]  # Flag positions
   FLAG_WEIGHTS = [1, 1, 1]       # Values of each flag
   FLAG_COLOR = GREEN             # Color of the flags
   FLAG_SIZE = 8
   ```

3. **Interaction Model**

   ```python
   INTERACTION_MODEL = "kill"
   ```

   Determines how agents interact with each other. Available options are:

   | Interaction Type | Description                                                  |
   | ---------------- | ------------------------------------------------------------ |
   | `kill`           | The attacker is eliminated upon capture.                     |
   | `respawn`        | The attacker respawns at its starting position upon capture. |
   | `both_kill`      | Both attacker and defender are eliminated upon capture.      |
   | `both_respawn`   | Both attacker and defender respawn at their starting positions upon capture. |

   > This parameter is passed to the `check_agent_interaction` function in `game.py`. You can create custom interaction models by modifying this function. For details, refer to the [Agent Interaction](#Agent-Interaction) section.

#### 🕹️ Agent Parameters

- **Global Agent Size**

  ```python
  GLOBAL_AGENT_SIZE = 8
  ```

- **Attacker Configuration**

  Global parameters:

  ```python
  ATTACKER_GLOBAL_SPEED = 1
  ATTACKER_GLOBAL_CAPTURE_RADIUS = 0
  ATTACKER_GLOBAL_SENSORS = ["map", "agent", "neighbor"]
  ATTACKER_GLOBAL_COLOR = RED
  ```

  Individual parameters:

  ```python
  ATTACKER_CONFIG = {
      "attacker_0": {
          "speed": 1,
          "capture_radius": 0,
          "sensors": ["map", "agent", "neighbor"],
          "color": RED,
          "start_node_id": 0,
      },
      "attacker_1": {"start_node_id": 1},
      ...
  }
  ```

  Individual settings override global parameters for the specified agent.

- **Defender Configuration**

  Defenders use the same configuration structure as attackers.

### game.py

The `game.py` file is the main script that runs the game simulation. It orchestrates all game components and manages the game loop.

#### 📁 Structure Overview

- Imports
- Game initailization
- Main game loop

🔧 Key Components

1. **Environment Setup**
   ```python
   ctx, G = initialize_game_context(
       VISUALIZATION_ENGINE,
       GRAPH_PATH,
       LOCATION,
       RESOLUTION
   )
   ```
   - Creates the game context and loads/generates the graph
   - Parameters are imported from `config.py`
   - Returns both the context (`ctx`) and graph (`G`)

2. **Sensor Configuration**

   Sets up the three main sensor types for environment perception

   ```python
   ctx.sensor.create_sensor("map", MAP_SENSOR)
   ctx.sensor.create_sensor("agent", AGENT_SENSOR)
   ctx.sensor.create_sensor("neighbor", NEIGHBOR_SENSOR)
   ```

3. **Agent Configuration**
   ```python
   agent_config, agent_params_map = configure_agents(
       ctx,
       ATTACKER_CONFIG,
       DEFENDER_CONFIG,
       {...}  # Global parameters
   )
   ```
   - Configures both attackers and defenders
   - Applies both global and individual parameters
   - Returns configuration and parameter mapping for all agents

4. **Strategy Assignment**
   ```python
   assign_strategies(ctx, agent_config, attacker_strategy, defender_strategy)
   ```
   - Links strategy implementations to agents
   - Strategies are imported from `attacker_strategy.py` and `defender_strategy.py`

5. **Visualization Setup**
   ```python
   configure_visualization(
       ctx,
       agent_config,
       {...}  # Visual parameters
   )
   ```
   - Configures the game's visual representation
   - Sets window size, colors, and display options

6. **Flag Initialization**
   ```python
   initialize_flags(ctx, FLAG_POSITIONS, FLAG_SIZE, FLAG_COLOR)
   ```
   - Places flags in the game environment
   - Uses parameters from `config.py`

#### 🎮 Main Game Loop

The main game loop runs continuously until termination conditions are met. Below is a detailed breakdown of an example:

##### Game Loop Example

```python
# Main game loop
while not ctx.is_terminated():
    # 1. Process each agent's turn
    for agent in ctx.agent.create_iter():
        # Get current state
        state = agent.get_state()
        
        # Add custom variables to state
        state.update({
            "flag_pos": FLAG_POSITIONS,      # List of flag positions
            "flag_weight": FLAG_WEIGHTS,     # Value of each flag
            "agent_params": agent_params_map[agent.name],  # Agent-specific parameters
            "custom_var": your_custom_data   # Add your own variables here
        })

        # Execute strategy or handle human input
        if agent.strategy is not None:
            agent.strategy(state)  # AI-controlled agent
        else:
            node = ctx.visual.human_input(agent.name, state)  # Human-controlled agent
            state["action"] = node
        
        agent.set_state()

    # 2. Check termination conditions
    if termination_condition:
        print("Game ended due to:", reason)
        break

    # 3. Update simulation
    ctx.visual.simulate()
    interaction_model.check_agent_interaction(ctx, G, agent_params_map, INTERACTION_MODEL)
```

##### State Management
Add any variables your strategy needs to the state dictionary:
```python
# Example: Adding custom data to state
state.update({
    "enemy_positions": get_enemy_positions(),
    "resources": available_resources,
    "time_left": game_timer
})
```

##### Termination Examples
```python
# Victory condition: All attackers captured
if not any(agent.team == "attacker" for agent in ctx.agent.create_iter()):
    print("Defenders win!")
    break

# Time limit reached
if game_timer <= 0:
    print("Time's up!")
    break

# Score threshold reached
if team_score >= WIN_SCORE:
    print("Score threshold reached!")
    break
```

---

### 🔗 Additional Resources

---

This document is a work in progress. If you find missing information or areas for improvement, feel free to contribute! 🙌