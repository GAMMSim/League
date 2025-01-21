import gamms
from config import *
import attacker_strategy
import defender_strategy
import pickle
import os
from utilities import *

# ------------------------------------------------------------------------------
# Initialize the game context with the selected visualization engine.
# ------------------------------------------------------------------------------
ctx = gamms.create_context(vis_engine=VISUALIZATION_ENGINE)

# ------------------------------------------------------------------------------
# Load or Create Graph
# ------------------------------------------------------------------------------
if os.path.exists(GRAPH_PATH):
    # If the graph file exists, load it using pickle.
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)
    print("Graph loaded from file.")
else:
    # If the graph file does not exist, create a new graph using the OSM module.
    G = gamms.osm.create_osm_graph(LOCATION, resolution=RESOLUTION)
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(G, f)
    print("Graph created and saved to file.")

# ------------------------------------------------------------------------------
# Attach the generated graph to the context's graph engine.
# ------------------------------------------------------------------------------
ctx.graph.attach_networkx_graph(G)

# ------------------------------------------------------------------------------
# Create the sensors used by the agents.
# ------------------------------------------------------------------------------
ctx.sensor.create_sensor("map", MAP_SENSOR)
ctx.sensor.create_sensor("agent", AGENT_SENSOR)
ctx.sensor.create_sensor("neighbor", NEIGHBOR_SENSOR)

# ------------------------------------------------------------------------------
# Combine and create the agent configurations with auto-assigned teams and default values.
# ------------------------------------------------------------------------------
agent_config = {}

# --- Add attacker agents ---
for name, config in ATTACKER_CONFIG.items():
    agent_entry = config.copy()
    agent_entry["team"] = "attacker"  # Auto assign team as "attacker"
    # Set default values if keys are missing.
    agent_entry.setdefault("speed", ATTACKER_GLOBAL_SPEED)
    agent_entry.setdefault("capture_radius", ATTACKER_GLOBAL_CAPTURE_RADIUS)
    agent_entry.setdefault("sensors", ATTACKER_GLOBAL_SENSORS)
    agent_entry.setdefault("color", ATTACKER_GLOBAL_COLOR)
    agent_entry["map"] = Graph()
    agent_config[name] = agent_entry

# --- Add defender agents ---
for name, config in DEFENDER_CONFIG.items():
    agent_entry = config.copy()
    agent_entry["team"] = "defender"  # Auto assign team as "defender"
    # Set default values if keys are missing.
    agent_entry.setdefault("speed", DEFENDER_GLOBAL_SPEED)
    agent_entry.setdefault("capture_radius", DEFENDER_GLOBAL_CAPTURE_RADIUS)
    agent_entry.setdefault("sensors", DEFENDER_GLOBAL_SENSORS)
    agent_entry.setdefault("color", DEFENDER_GLOBAL_COLOR)
    agent_entry["map"] = Graph()
    agent_config[name] = agent_entry

# --- Create agents in the context using the combined configuration ---
for name, config in agent_config.items():
    ctx.agent.create_agent(name, **config)

print("Agents created.")

# ------------------------------------------------------------------------------
# Create and assign strategies for agents.
# ------------------------------------------------------------------------------
strategies = {}

# For this example, assign a strategy for each team:
strategies.update(attacker_strategy.map_strategy({name: config for name, config in agent_config.items() if config.get("team") == "attacker"}))
strategies.update(defender_strategy.map_strategy({name: config for name, config in agent_config.items() if config.get("team") == "defender"}))

# Register the assigned strategies with each agent.
for agent in ctx.agent.create_iter():
    agent.register_strategy(strategies.get(agent.name, None))

print("Strategies set.")

# ------------------------------------------------------------------------------
# Set visualization configurations for the graph and agents.
# ------------------------------------------------------------------------------
# Configure the graph visualization dimensions.
graph_vis_config = {"width": 1980, "height": 1080}
ctx.visual.set_graph_visual(**graph_vis_config)

# Set the simulation time constant (affects speed of animations).
ctx.visual._sim_time_constant = GAME_SPEED
ctx.visual.draw_node_id = DRAW_NODE_ID

# For each agent, configure its visualization using the agent configuration details.
for name, config in agent_config.items():
    # Retrieve the agent's color from the configuration.
    # If not set, choose a default based on the team.
    if "color" in config:
        color = config["color"]
    else:
        if config.get("team") == "attacker":
            color = ATTACKER_GLOBAL_COLOR
        else:
            color = DEFENDER_GLOBAL_COLOR

    # Use a default visualization size if not provided (assume GLOBAL_AGENT_SIZE is defined).
    size = config.get("size", GLOBAL_AGENT_SIZE)

    # Set the visualization parameters for the agent.
    ctx.visual.set_agent_visual(name, color=color, size=size)

# ------------------------------------------------------------------------------
# Flag Initialization
# ------------------------------------------------------------------------------
# Assuming FLAG_POSITIONS is a list of node IDs where each flag should appear.
# For example, FLAG_POSITIONS = [50, 51, 52] as defined in your configuration file.

for index, flag_node_id in enumerate(FLAG_POSITIONS):
    # Retrieve the corresponding node from the graph using its ID.
    node = ctx.graph.graph.get_node(flag_node_id)

    # Prepare the drawing data for the flag.
    flag_data = {"x": node.x, "y": node.y, "scale": FLAG_SIZE, "color": FLAG_COLOR}  # Adjust this value to change the flag's size  # This color is red; change if desired.

    # Add the flag as a special artist to the visualization.
    ctx.visual.add_artist(f"flag_{index}", flag_data)

print("Flags initialized.")

# Run the game
while not ctx.is_terminated():
    for agent in ctx.agent.create_iter():
        if agent.strategy is not None:
            state = agent.get_state()
            agent.strategy(state, FLAG_POSITIONS, FLAG_WEIGHTS, agent)
            agent.set_state()
        else:
            state = agent.get_state()
            node = ctx.visual.human_input(agent.name, state)
            state["action"] = node
            agent.set_state()

    ctx.visual.simulate()
    
# To kill the game, use control + c, or customize the termination condition in the while loop.
