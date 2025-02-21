"""
This file contains the configuration for the game parameters.

Overview of Sections:
    1. Game Rules and Mechanics
    2. Agent Parameters
    3. Environment Parameters
    4. Visualization Parameters
"""

import gamms

# ------------------------------------------------------------------------------
# 1. GAME RULES AND MECHANICS
# ------------------------------------------------------------------------------
# Core game rules and mechanics that define how the game works
GAME_RULE = "V1"
# GAME_RULE = None

MAX_TIME = 100

SAVE_LOG = False

# Interaction Model
INTERACTION = {
    "tagging": "both_kill",
    "capture": "kill",
    "prioritize": "capture"
}

# Payoff Configuration
PAYOFF = {
    "model": "v1",
    "constants": {
        "k": 0.5
    }
}

# Flag Configuration
FLAG_POSITIONS = [50, 51, 52]  # Node positions where flags are located
FLAG_WEIGHTS = [1, 1, 1]       # The scoring weight (value) of each flag

# ------------------------------------------------------------------------------
# 2. AGENT PARAMETERS
# ------------------------------------------------------------------------------
# Configuration for the agents in the game

# 2.1 Global Attacker Parameters
ATTACKER_GLOBAL_SPEED = 1
ATTACKER_GLOBAL_CAPTURE_RADIUS = 2
ATTACKER_GLOBAL_SENSORS = ["map", "agent", "neighbor"]

# 2.2 Individual Attacker Parameters
ATTACKER_CONFIG = {
    "attacker_0": {
        "speed": 1,
        "capture_radius": 2,
        "sensors": ["map", "agent", "neighbor"],
        "start_node_id": 0,
    },
    "attacker_1": {"start_node_id": 1},
    "attacker_2": {"start_node_id": 2},
    "attacker_3": {"start_node_id": 3},
    "attacker_4": {"start_node_id": 4},
}

# 2.3 Global Defender Parameters
DEFENDER_GLOBAL_SPEED = 1
DEFENDER_GLOBAL_CAPTURE_RADIUS = 1
DEFENDER_GLOBAL_SENSORS = ["map", "agent", "neighbor"]

# 2.4 Individual Defender Parameters
DEFENDER_CONFIG = {
    "defender_0": {
        "speed": 1,
        "capture_radius": 1,
        "sensors": ["map", "agent", "neighbor"],
        "start_node_id": 100,
    },
    "defender_1": {"start_node_id": 101},
    "defender_2": {"start_node_id": 102},
    "defender_3": {"start_node_id": 103},
    "defender_4": {"start_node_id": 104},
}

# ------------------------------------------------------------------------------
# 3. ENVIRONMENT PARAMETERS
# ------------------------------------------------------------------------------
# Configuration for the game's underlying graph/network

GRAPH_PATH = "graph.pkl"
# If GRAPH_PATH does not exist, a new graph will be generated based on:
LOCATION = "West Point, New York, USA"  # The real-world location to generate the graph from
RESOLUTION = 200.0                      # The resolution of the graph (higher means fewer nodes)

# Sensors (used by agents to gather information):
MAP_SENSOR = gamms.sensor.SensorType.MAP
AGENT_SENSOR = gamms.sensor.SensorType.AGENT
NEIGHBOR_SENSOR = gamms.sensor.SensorType.NEIGHBOR

# ------------------------------------------------------------------------------
# 4. VISUALIZATION PARAMETERS
# ------------------------------------------------------------------------------
# Settings for the game's display and visual elements

# Window and Display
WINDOW_SIZE = (1980, 1080)
GAME_SPEED = 1
DRAW_NODE_ID = False
VISUALIZATION_ENGINE = gamms.visual.Engine.PYGAME

# Colors
RED = "red"
BLUE = "blue"
GREEN = "green"
BLACK = "black"
ATTACKER_GLOBAL_COLOR = RED
DEFENDER_GLOBAL_COLOR = BLUE
FLAG_COLOR = GREEN

# Visual Sizes
FLAG_SIZE = 8
GLOBAL_AGENT_SIZE = 8