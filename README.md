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

# How to Use

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

---

### 🔗 Additional Resources

---

This document is a work in progress. If you find missing information or areas for improvement, feel free to contribute! 🙌