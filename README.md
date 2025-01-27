# README

## 📑 Table of Contents
- [Quick Start Guide](#-quick-start-guide)
- [Requirements](#%EF%B8%8F-requirements)
- [Installation](#-installation)
- [File Structure](#-file-structure)
- [How to Use](#-how-to-use)
  - [Step 1: Create Config File](#step-1-create-config-file)
  - [Step 2: Define Strategies](#step-2-define-strategies)
  - [Step 3: Setup Agent Interactions](#step-3-setup-agent-interactions)
  - [Step 4: Game Loop & Termination](#step-4-game-loop--termination)
  - [Step 5: Run the Game](#step-5-run-the-game)
- [Advanced Usage](#-advanced-usage)
  - [Config.py](#%EF%B8%8F-configpy)
    - [Color Parameters](#-color-parameters)
    - [Simulation Interface Parameters](#%EF%B8%8F-simulation-interface-parameters)
    - [Graph Parameters](#%EF%B8%8F-graph-parameters)
    - [Game Parameters](#-game-parameters)
    - [Agent Parameters](#%EF%B8%8F-agent-parameters)
  - [Game.py](#-gamepy)
    - [Structure Overview](#-structure-overview)
    - [Main Game Loop](#-main-game-loop)
  - [Strategy](#-strategy)
    - [Overview](#-overview)
    - [Example Strategy](#-example-strategy)
    - [Key Concepts](#-key-concepts)
  - [Agent Interaction](#-agent-interaction)
    - [Managing Agents](#-managing-agents-ctx)
    - [Agent Parameters](#-agent-parameters-agent_params)
    - [Graph Operations](#%EF%B8%8F-graph-operations-g)
    - [Common Operations](#-common-operations)
- [Additional Resources](#-additional-resources)

## 🚀 Quick Start Guide

1. **Set Up Environment**  

   ```bash  
   # Clone repositories  
   git clone https://github.com/GAMMSim/gamms.git  
   git clone https://github.com/GAMMSim/League.git  
   
   # Create and activate a virtual environment  
   cd gamms  
   python -m venv venv  
   source venv/bin/activate    # Mac/Linux: venv\Scripts\activate for Windows  
   ```

2. **Install Gamms**  
   ```bash  
   pip install .  # Installs Gamms into the virtual environment  
   ```

3. **Run the Game**  

   Navigate to `League/games/examples`, pick any example folder or create you own folder. Inside the example folder, run

   ```bash  
   python game.py  
   ```

**Done!** For advanced setup, see [Installation Details](#-installation).  

---

## ⚙️ Requirements  
- Python 3.7+  
- `git` and `pip`  

---

## 📥 Installation  
### Detailed Steps for Isolation and Dependency Management  

1. **Virtual Environment**  
   Create a virtual environment to isolate dependencies:  

   ```bash  
   python -m venv venv  
   ```
   - Activate:  
     ```bash  
     # Mac/Linux  
     source venv/bin/activate  
     # Windows  
     venv\Scripts\activate  
     ```

2. **Install Gamms**  
   Navigate to the cloned `gamms` folder and install:  

   ```bash  
   pip install .  # Installs Gamms and dependencies into the venv  
   ```

3. **Verify Installation**  
   ```bash  
   python -c "import gamms; print('Success! Version:', gamms.__version__)"  
   ```

   ⚠️ For more **detail** or **troubleshooting**, visit the `gamms` library's office documentation here: [Visit the Quick Start Guide](https://gammsim.github.io/gamms/start/).

4. **Configure and Run**  

   - Navigate to `League/games`.
   - Open an example folder, or create your own example folder. The structure of the files can be found in [File Structure](#-file-structure).

   - Follow the instructions in [How to Use](#-how-to-use).

## 📁 File Structure

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

📝 **Notes**:

- ✅ Keep `utilities.py` in the root of `games` folder
- ⚠️ Root directory must be `.../games` to avoid errors

## 🎯 How to Use

### Step 1: Create Config File
Set up your game parameters in `config.py`:
- Window size and game speed
- Map settings
- Agent properties
- Flag positions

### Step 2: Define Strategies
Create two strategy files:
- `attacker_strategy.py`: Define how attackers behave
- `defender_strategy.py`: Define how defenders behave

### Step 3: Setup Agent Interactions
Choose or create an interaction model in `interaction_model.py`
- Kill on capture
- Respawn system
- Custom interaction rules

### Step 4: Game Loop & Termination
Define in `game.py`:
- Victory conditions
- Time limits
- Score thresholds

### Step 5: Run the Game
Execute from the correct directory:
```bash
python game.py
```

# 🔧 Advanced Usage

## ⚙️ Config.py

The `config.py` file contains all customizable parameters for your game setup.

### 🎨 Color Parameters
```python
RED = "red"       # The color red
BLUE = "blue"     # The color blue
GREEN = "green"   # The color green
BLACK = "black"   # The color black
```
These color constants are used for defining agent colors, flag colors, and other visual elements. Ensure the colors are defined in `gamms/VisualizationEngine/__init__.py` and written in **lowercase**.

### 🖥️ Simulation Interface Parameters
```python
WINDOW_SIZE = (1980, 1080)
GAME_SPEED = 1
DRAW_NODE_ID = True
VISUALIZATION_ENGINE = gamms.visual.Engine.PYGAME
```

- `WINDOW_SIZE (tuple)`: Game window dimensions in pixels
- `GAME_SPEED (float)`: Simulation speed control (higher = slower)
- `DRAW_NODE_ID (bool)`: Toggle node ID display
  > ⚠️ **Warning**: Enabling node IDs significantly reduces rendering performance
- `VISUALIZATION_ENGINE`: Choose visualization engine type. Use `visual.Engine.NO_VIS` for a dry run without rendering

### 🗺️ Graph Parameters
```python
GRAPH_PATH = "graph.pkl"
LOCATION = "West Point, New York, USA"
RESOLUTION = 200.0
```

- `GRAPH_PATH (str)`: Path to the graph file (must be `.pkl` file)
  > **Note**: If the file doesn't exist, the program generates it using:
  - `LOCATION (str)`: Real-world location to base the graph on
  - `RESOLUTION (float)`: Graph resolution; higher values mean lower node density

### 🎮 Game Parameters

#### Sensors
Available sensor types:
```python
MAP_SENSOR = gamms.sensor.SensorType.MAP
AGENT_SENSOR = gamms.sensor.SensorType.AGENT
NEIGHBOR_SENSOR = gamms.sensor.SensorType.NEIGHBOR
```

- `MAP_SENSOR`: Senses all nodes and edges, excluding agent and flag data
- `AGENT_SENSOR`: Senses names and positions of all agents, excluding their properties
- `NEIGHBOR_SENSOR`: Senses neighboring nodes of agent's current position

#### Flags
```python
FLAG_POSITIONS = [50, 51, 52]  # Flag positions
FLAG_WEIGHTS = [1, 1, 1]       # Values of each flag
FLAG_COLOR = GREEN             # Color of the flags
FLAG_SIZE = 8
```

#### Interaction Model
```python
INTERACTION_MODEL = "kill"
```

| Model Type     | Description                    |
| -------------- | ------------------------------ |
| `kill`         | Attacker eliminated on capture |
| `respawn`      | Attacker respawns at start     |
| `both_kill`    | Both agents eliminated         |
| `both_respawn` | Both agents respawn            |

> This parameter is passed to `check_agent_interaction` function in `game.py`. Create custom interaction models by modifying this function. See [Agent Interaction](#agent-interaction) section for details.

### 🕹️ Agent Parameters

#### Global Agent Size
```python
GLOBAL_AGENT_SIZE = 8
```

#### Attacker Configuration
```python
# Global parameters
ATTACKER_GLOBAL_SPEED = 1
ATTACKER_GLOBAL_CAPTURE_RADIUS = 0
ATTACKER_GLOBAL_SENSORS = ["map", "agent", "neighbor"]
ATTACKER_GLOBAL_COLOR = RED

# Individual parameters
ATTACKER_CONFIG = {
    "attacker_0": {
        "speed": 1,
        "capture_radius": 0,
        "sensors": ["map", "agent", "neighbor"],
        "color": RED,
        "start_node_id": 0,
    },
    "attacker_1": {"start_node_id": 1},
}
```
Individual settings override global parameters for the specified agent. Defenders use the same configuration structure.

## 🎲 Game.py

The `game.py` file is the main script that runs the game simulation. It orchestrates all game components and manages the game loop.

### 📁 Structure Overview

1. **Environment Setup**
```python
ctx, G = initialize_game_context(
    VISUALIZATION_ENGINE,
    GRAPH_PATH,
    LOCATION,
    RESOLUTION
)
```
- Creates game context and loads/generates graph
- Parameters imported from `config.py`
- Returns both context (`ctx`) and graph (`G`)

2. **Sensor Configuration**
```python
ctx.sensor.create_sensor("map", MAP_SENSOR)
ctx.sensor.create_sensor("agent", AGENT_SENSOR)
ctx.sensor.create_sensor("neighbor", NEIGHBOR_SENSOR)
```
Sets up the three main sensor types for environment perception

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
- Strategies imported from `attacker_strategy.py` and `defender_strategy.py`

5. **Visualization Setup**
```python
configure_visualization(
    ctx,
    agent_config,
    {...}  # Visual parameters
)
```
- Configures game's visual representation
- Sets window size, colors, and display options

6. **Flag Initialization**
```python
initialize_flags(ctx, FLAG_POSITIONS, FLAG_SIZE, FLAG_COLOR)
```
- Places flags in game environment
- Uses parameters from `config.py`

### 🎮 Main Game Loop

#### Game Loop Example
```python
while not ctx.is_terminated():
    # 1. Process each agent's turn
    for agent in ctx.agent.create_iter():
        state = agent.get_state()
        
        # Add custom variables to state
        state.update({
            "flag_pos": FLAG_POSITIONS,
            "flag_weight": FLAG_WEIGHTS,
            "agent_params": agent_params_map[agent.name],
            "custom_var": your_custom_data
        })

        # Execute strategy or handle human input
        if agent.strategy is not None:
            agent.strategy(state)  # AI-controlled agent
        else:
            node = ctx.visual.human_input(agent.name, state)  # Human-controlled agent
            state["action"] = node
        
        agent.set_state()

    # Check termination conditions
    if termination_condition:
        print("Game ended due to:", reason)
        break

    # Update simulation
    ctx.visual.simulate()
    interaction_model.check_agent_interaction(ctx, G, agent_params_map, INTERACTION_MODEL)
```

#### State Management
Add any variables your strategy needs to the state dictionary:
```python
# Example: Adding custom data to state
state.update({
    "enemy_positions": get_enemy_positions(),
    "resources": available_resources,
    "time_left": game_timer
})
```

#### Termination Examples
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

## 🧠 Strategy

### 📝 Overview
Strategies define how agents behave in the game. By customizing strategies, you can control the movement and actions of both attackers and defenders. We'll use the provided `defender_strategy.py` as examples to explain key concepts in developing your own strategies.

Note: Other examples can be found in `games/basic_example` and `game/random_move`.

### 🎯 Example Strategy
```python
def strategy(state):
    """
    Defender strategy to move towards closest attacker.
    """
    current_node = state['curr_pos']
    flag_positions = state['flag_pos']
    flag_weights = state['flag_weight']
    agent_params = state['agent_params']
    
    # Extract positions of attackers and defenders
    attacker_positions, defender_positions = extract_sensor_data(
        state, flag_positions, flag_weights, agent_params
    )
    
    closest_attacker = None
    min_distance = float('inf')
    
    # Find closest attacker based on shortest path distance
    for attacker in attacker_positions:
        for flag in flag_positions:
            try:
                dist = nx.shortest_path_length(
                    agent_params.map.graph, source=attacker, target=flag
                )
                if dist < min_distance:
                    min_distance = dist
                    closest_attacker = attacker
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
```

### 🔑 Key Concepts

#### 1. State Dictionary Access
```python
# Basic state information
current_node = state['curr_pos']               # Current position
flag_positions = state['flag_pos']             # Flag positions
flag_weights = state['flag_weight']            # Flag values
agent_params = state['agent_params']           # Agent parameters
```

#### 2. Sensor Data Usage
```python
# Get agent positions
attacker_positions, defender_positions = extract_sensor_data(
    state, flag_positions, flag_weights, agent_params
)

# Get neighboring nodes
neighbor_data = extract_neighbor_sensor_data(state)
```

#### 3. Map Navigation
```python
# Access map graph
graph = agent_params.map.graph

# Calculate shortest path
dist = nx.shortest_path_length(
    graph, source=current_node, target=flag
)

# Determine next move
next_node = agent_params.map.shortest_path_to(
    current_node, closest_flag, agent_params.speed
)
state['action'] = next_node
```

## 🤝 Agent Interaction

### 🎮 Managing Agents (`ctx`)
```python
# Delete an agent
ctx.agent.delete_agent(agent_name)

# Get all agents
for agent in ctx.agent.create_iter():
    print(f"Agent: {agent.name}, Team: {agent.team}")

# Access agent properties
agent.current_node_id  # Current position
agent.prev_node_id     # Previous position
agent.start_node_id    # Starting position
agent.team            # "attacker" or "defender"
```

### 📊 Agent Parameters (`agent_params`)
```python
# Access agent-specific parameters
capture_radius = agent_params[agent_name].capture_radius
speed = agent_params[agent_name].speed

# Example usage
if distance <= agent_params[defender.name].capture_radius:
    handle_capture()
```

### 🗺️ Graph Operations (`G`)
```python
# Calculate distance between agents
try:
    distance = nx.shortest_path_length(
        G,
        source=attacker.current_node_id,
        target=defender.current_node_id
    )
except nx.NetworkXNoPath:
    print("No path exists")
```

### 🔄 Common Operations

#### 🗑️ Removing Agents
```python
# Remove an agent permanently
ctx.agent.delete_agent(agent.name)

# Don't forget to update your local lists if needed
agents_list.remove(agent)
```

#### 🔄 Respawning Agents
```python
# Reset agent to starting position
agent.prev_node_id = agent.current_node_id    # Store current as previous
agent.current_node_id = agent.start_node_id   # Move to start position

# Example with both teams
attacker.current_node_id = attacker.start_node_id
defender.current_node_id = defender.start_node_id
```

---

## 🔗 Additional Resources
[Documentation and examples continue to be updated] 

This document is a work in progress. If you find missing information or areas for improvement, please submit an issue!
