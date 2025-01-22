# The path should remain on top of the file to ensure the correct import path.
import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
# ------------------------------------------------------------------------------
import gamms
from config import *
import attacker_strategy
import defender_strategy
import pickle
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
    agent_entry["start_node_id"] = config.get("start_node_id", None)
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
    agent_entry["start_node_id"] = config.get("start_node_id", None)
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

# ------------------------------------------------------------------------------
# Define Agent Interaction Model
# ------------------------------------------------------------------------------
def check_agent_interaction(ctx, G, model="kill"):
    attackers = []
    defenders = []
    for agent in ctx.agent.create_iter():
        team = agent.team
        if team == "attacker":
            attackers.append(agent)
        elif team == "defender":
            defenders.append(agent)
    
    # Check each attacker against each defender.
    for attacker in attackers:
        for defender in defenders:
            try:
                # Compute the shortest path distance between the attacker and defender.
                distance = nx.shortest_path_length(G,
                                                   source=attacker.current_node_id,
                                                   target=defender.current_node_id)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue  # Skip if there is no connection
            
            # Retrieve defender's capture radius (default to 1 if not defined).
            capture_radius = getattr(defender, 'capture_radius', 0)
            if distance <= capture_radius:
                # An interaction takes place.
                if model == "kill":
                    # Defender kills the attacker.
                    print(f"[Interaction: kill] Defender {defender.name} kills attacker {attacker.name}.")
                    ctx.agent.delete_agent(attacker.name)
                elif model == "respawn":
                    # Attacker respawns.
                    print(f"[Interaction: respawn] Attacker {attacker.name} respawns due to interaction with defender {defender.name}.")
                    attacker.prev_node_id = attacker.current_node_id
                    attacker.current_node_id = attacker.start_node_id
                elif model == "both_kill":
                    # Both agents are killed (set to inactive).
                    print(f"[Interaction: both_kill] Both attacker {attacker.name} and defender {defender.name} are killed.")
                    ctx.agent.delete_agent(attacker.name)
                    ctx.agent.delete_agent(defender.name)
                elif model == "both_respawn":
                    # Both agents respawn (reset to start positions and become active).
                    print(f"[Interaction: both_respawn] Both attacker {attacker.name} and defender {defender.name} respawn.")
                    attacker.prev_node_id = attacker.current_node_id
                    attacker.current_node_id = attacker.start_node_id
                    defender.prev_node_id = defender.current_node_id
                    defender.current_node_id = defender.start_node_id
                else:
                    print(f"Unknown interaction model: {model}")
                    
# ------------------------------------------------------------------------------
# Run the game
# ------------------------------------------------------------------------------
while not ctx.is_terminated():    
    for agent in ctx.agent.create_iter():
        if agent.strategy is not None:
            state = agent.get_state()
            state["flag_pos"] = FLAG_POSITIONS
            state["flag_weight"] = FLAG_WEIGHTS
            agent.strategy(state, agent)
            agent.set_state()
        else:
            state = agent.get_state()
            node = ctx.visual.human_input(agent.name, state)
            state["action"] = node
            agent.set_state()

    ctx.visual.simulate()
    check_agent_interaction(ctx, G, INTERACTION_MODEL)
    
# To kill the game, use control + c, or customize the termination condition in the while loop.
