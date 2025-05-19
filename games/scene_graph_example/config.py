"""
This file contains the configuration for the game parameters.

Overview of Sections:
    1. Game Rules and Mechanics
    2. Agent Parameters
    3. Environment Parameters
    4. Visualization Parameters
"""

import gamms
import pathlib

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
FLAG_POSITIONS = [8070450532247948990, 8070450532247948862, 8070450532247948875]  # Node positions where flags are located
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
        "start_node_id": 8070450532247928992,
    },
    "attacker_1": {"start_node_id": 8070450532247963839},
    "attacker_2": {"start_node_id": 8070450532247963927},
    # "attacker_3": {"start_node_id": 8070450532247963932},
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
        "start_node_id": 8070450532247946294,
    },
    "defender_1": {"start_node_id": 8070450532247946330},
    "defender_2": {"start_node_id": 8070450532247946464},
    # "defender_3": {"start_node_id": 8070450532247946472},
}

# ------------------------------------------------------------------------------
# 3. ENVIRONMENT PARAMETERS
# ------------------------------------------------------------------------------
# Configuration for the game's underlying graph/network


SCENE_GRAPH_PATH = "./example_dsg.json" ## This should be a path to the Scene Graph
SCENE_GRAPH_PATH = pathlib.Path(SCENE_GRAPH_PATH).expanduser().absolute()


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