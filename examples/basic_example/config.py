"""
This file contains the configuration for the game parameters.

Overview of Sections:
    1. Color Parameters
    2. Interface Parameters
    3. Graph Parameters
    4. Game Parameters
    5. Attacker Parameters
    6. Defender Parameters
"""

import gamms

# ------------------------------------------------------------------------------
# 1. COLOR PARAMETERS
# ------------------------------------------------------------------------------
# Commonly used RGB color tuples (R, G, B).

RED = "red"       # The color red
BLUE = "blue"      # The color blue
GREEN = "green"     # The color green
BLACK = "black"       # The color black


# ------------------------------------------------------------------------------
# 2. INTERFACE PARAMETERS
# ------------------------------------------------------------------------------
# Settings for the game's display and basic behavior.

WINDOW_SIZE = (1980, 1080)          # The size of the game window in pixels
GAME_SPEED = 1                      # The speed of the game (the higher the number, the slower it runs)
VISUALIZATION_ENGINE = gamms.visual.Engine.PYGAME
# ^ This sets the engine used to render the game


# ------------------------------------------------------------------------------
# 3. GRAPH PARAMETERS
# ------------------------------------------------------------------------------
# Configuration for the game's underlying graph/network.

GRAPH_PATH = "graph.pkl"
# If GRAPH_PATH does not exist, a new graph will be generated based on:
LOCATION = "West Point, New York, USA"  # The real-world location to generate the graph from
RESOLUTION = 100.0                      # The resolution of the graph (higher means fewer nodes)


# ------------------------------------------------------------------------------
# 4. GAME PARAMETERS
# ------------------------------------------------------------------------------
# General game settings, including sensors and flags.

# Sensors (used by agents to gather information):
MAP_SENSOR = gamms.sensor.SensorType.MAP         # Sensor for map data
AGENT_SENSOR = gamms.sensor.SensorType.AGENT     # Sensor for agent-related data
NEIGHBOR_SENSOR = gamms.sensor.SensorType.NEIGHBOR  # Sensor for neighbor-related data

# Flags (objectives within the game):
FLAG_POSITIONS = [50, 51, 52]  # Node positions where flags are located
FLAG_WEIGHTS = [1, 1, 1]       # The scoring weight (value) of each flag
FLAG_COLOR = GREEN             # The color of the flags
FLAG_SIZE = 8


# ------------------------------------------------------------------------------
# 5. ATTACKER PARAMETERS
# ------------------------------------------------------------------------------
# Configuration for attacker agents in the game.
GLOBAL_AGENT_SIZE = 8

# 5.1 Global Attacker Parameters
ATTACKER_GLOBAL_SPEED = 1                   # Nodes an attacker can move per step
ATTACKER_GLOBAL_CAPTURE_RADIUS = 0          # Capture radius for flags or agents
ATTACKER_GLOBAL_SENSORS = ["map", "agent","neighbor"]    # Sensors available to attackers
ATTACKER_GLOBAL_COLOR = RED                 # Default color for attackers

# 5.2 Individual Attacker Parameters
ATTACKER_CONFIG = {
    "attacker_0": {
        "speed": 1,
        "capture_radius": 0,
        "sensors": ["map", "agent","neighbor"],
        "color": RED,
        "start_node_id": 0,
    },
    "attacker_1": {"start_node_id": 1},
    "attacker_2": {"start_node_id": 2},
    "attacker_3": {"start_node_id": 3},
    "attacker_4": {"start_node_id": 4},
}


# ------------------------------------------------------------------------------
# 6. DEFENDER PARAMETERS
# ------------------------------------------------------------------------------
# Configuration for defender agents in the game.

# 6.1 Global Defender Parameters
DEFENDER_GLOBAL_SPEED = 1                   # Nodes a defender can move per step
DEFENDER_GLOBAL_CAPTURE_RADIUS = 1          # Capture radius for flags or agents
DEFENDER_GLOBAL_SENSORS = ["map", "agent","neighbor"]    # Sensors available to defenders
DEFENDER_GLOBAL_COLOR = BLUE                # Default color for defenders

# 6.2 Individual Defender Parameters
DEFENDER_CONFIG = {
    "defender_0": {
        "speed": 1,
        "capture_radius": 1,
        "sensors": ["map", "agent","neighbor"],
        "color": BLUE,
        "start_node_id": 100,
    },
    "defender_1": {"start_node_id": 101},
    "defender_2": {"start_node_id": 102},
    "defender_3": {"start_node_id": 103},
    "defender_4": {"start_node_id": 104},
}

