# This file defined the interaction model between agents and flags
# WARNING: Before edit, make sure you know what you want and check out the Advanced Usage part of the documentation.
import networkx as nx
from lib.interface import colored
import os
from datetime import datetime
import json


def handle_interaction(ctx, agent, action, processed, agent_params):
    """Handle agent interaction (kill or respawn)"""
    if action == "kill":
        ctx.agent.delete_agent(agent.name)
        processed.add(agent.name)
        return True
    if action == "respawn":
        agent.prev_node_id = agent.current_node_id
        # Use start position from agent_params
        start_pos = agent_params[agent.name].start_node_id
        agent.current_node_id = start_pos
        return True
    return False


def check_agent_interaction(ctx, G, agent_params, flag_positions, interaction_config):
    """Main interaction checking function"""
    captures = tags = 0
    processed = set()

    # Get initial lists
    attackers = [a for a in ctx.agent.create_iter() if a.team == "attacker"]
    defenders = [d for d in ctx.agent.create_iter() if d.team == "defender"]

    # Process interactions based on priority
    if interaction_config["prioritize"] == "capture":
        # Check flag captures first
        for attacker in attackers[:]:
            if attacker.name in processed:
                continue
            for flag in flag_positions:
                shortest_distance = nx.shortest_path_length(G, attacker.current_node_id, flag)
                attacker_capture_radius = getattr(agent_params[attacker.name], "capture_radius", 0)
                if shortest_distance <= attacker_capture_radius:
                    print(colored(f"Attacker {attacker.name} captured flag at {flag}", "orange"))
                    if handle_interaction(ctx, attacker, interaction_config["capture"], processed, agent_params):
                        captures += 1
    # Check combat interactions
    for defender in defenders[:]:
        if defender.name in processed:
            continue
        for attacker in attackers[:]:
            if attacker.name in processed or defender.name in processed:
                continue
            try:
                # Access capture_radius directly from AgentParams object
                defender_capture_radius = getattr(agent_params[defender.name], "capture_radius", 0)
                if nx.shortest_path_length(G, attacker.current_node_id, defender.current_node_id) <= defender_capture_radius:
                    print(colored(f"Defender {defender.name} tagged attacker {attacker.name}", "orange"))
                    if interaction_config["tagging"] == "both_kill":
                        handle_interaction(ctx, attacker, "kill", processed, agent_params)
                        handle_interaction(ctx, defender, "kill", processed, agent_params)
                    if interaction_config["tagging"] == "both_respawn":
                        handle_interaction(ctx, attacker, "respawn", processed, agent_params)
                        handle_interaction(ctx, defender, "respawn", processed, agent_params)
                    else:
                        handle_interaction(ctx, attacker, interaction_config["tagging"], processed, agent_params)
                    tags += 1
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

    # If tags processed first, check captures second
    if interaction_config["prioritize"] != "capture":
        for attacker in attackers[:]:
            if attacker.name in processed:
                continue
            for flag in flag_positions:
                shortest_distance = nx.shortest_path_length(G, attacker.current_node_id, flag)
                attacker_capture_radius = getattr(agent_params[attacker.name], "capture_radius", 0)
                if shortest_distance <= attacker_capture_radius:
                    print(colored(f"Attacker {attacker.name} captured flag at {flag}", "orange"))
                    if handle_interaction(ctx, attacker, interaction_config["capture"], processed, agent_params):
                        captures += 1

    # Count remaining agents
    remaining_attackers = sum(1 for a in ctx.agent.create_iter() if a.team == "attacker")
    remaining_defenders = sum(1 for d in ctx.agent.create_iter() if d.team == "defender")

    return captures, tags, remaining_attackers, remaining_defenders


def check_termination(time: int, MAX_TIME: int, remaining_attackers: int, remaining_defenders: int) -> bool:
    if time >= MAX_TIME:
        # logger.info("Maximum time reached.")
        return True
    if remaining_attackers == 0:
        # logger.info("All attackers have been eliminated.")
        return True
    if remaining_defenders == 0:
        # logger.info("All defenders have been eliminated.")
        return True
    return False


def compute_payoff(payoff: dict, captures: int, tags: int) -> float:
    if payoff["model"] != "v1":
        # logger.error(f"Unsupported payoff model: {payoff['model']}")
        return 0.0
    payoff = captures - payoff["constants"]["k"] * tags
    return payoff


def initialize_logger(save_path: str):
    """
    Initializes the logger by creating the log file with the appropriate name.

    Args:
        save_path (str): The directory where the log file will be saved.

    Returns:
        file: The file object for the log file, or None if initialization fails.
    """
    try:
        os.makedirs(save_path, exist_ok=True)
    except OSError as e:
        print(colored(f"Cannot create file in save path: {e}", "red"))
        return None

    print(colored(f"Log file will be saved in {save_path}", "green"))

    # Generate filename with format game_logDDHHMM
    current_time = datetime.now()
    filename = current_time.strftime("game_log%d%H%M.json")
    file_path = os.path.join(save_path, filename)

    # Open the file in write mode
    try:
        log_file = open(file_path, "w")
        # Initialize the log as a JSON array
        log_file.write("[\n")
        return log_file
    except IOError as e:
        print(colored(f"Cannot open log file: {e}", "red"))
        return None

def log_game_step(log_file, time, payoff, agent_positions, is_last=False):
    """
    Logs a single game step to the log file.

    Args:
        log_file (file): The file object for the log file.
        time (int): The current time step.
        payoff (float): The current payoff.
        agent_positions (dict): Dictionary mapping agent names to their positions.
        is_last (bool): Flag indicating if this is the last log entry.
    """
    if log_file is None:
        return

    log_entry = {"time": time, "payoff": payoff, "agents": agent_positions}

    try:
        if is_last:
            # For the last entry, don't add a comma
            json.dump(log_entry, log_file, indent=4)
            log_file.write("\n")
        else:
            # Add a comma after each entry except the last
            json.dump(log_entry, log_file, indent=4)
            log_file.write(",\n")
    except IOError as e:
        print(colored(f"Error writing to log file: {e}", "red"))

def finalize_logger(log_file):
    """
    Finalizes the logger by closing the JSON array and the file.

    Args:
        log_file (file): The file object for the log file.
    """
    if log_file is None:
        return
        
    try:
        log_file.seek(log_file.tell() - 2)
        log_file.write("\n]\n")  # End of JSON array
        log_file.close()
    except IOError as e:
        print(colored(f"Error closing log file: {e}", "red"))

def create_game_log_entry(agent_params_map, active_agents):
    """
    Creates a dictionary mapping agent names to their positions.

    Args:
        agent_params_map (dict): Mapping of agent names to their parameters.
        active_agents (iterable): Iterable of active agent instances.

    Returns:
        dict: A dictionary mapping agent names to their positions, with -1 for removed agents.
    """
    agent_positions = {}
    active_agent_names = set(agent.name for agent in active_agents)

    for agent_name in agent_params_map.keys():
        if agent_name in active_agent_names:
            agent = next((agent for agent in active_agents if agent.name == agent_name), None)
            if agent and hasattr(agent, "current_node_id"):
                agent_positions[agent_name] = agent.current_node_id
            else:
                agent_positions[agent_name] = -1
        else:
            agent_positions[agent_name] = -1

    return agent_positions


def check_and_install_dependencies():
    """
    Check if required packages are installed and install them if they're missing.
    Returns True if all dependencies are satisfied, False if installation failed.
    """
    import subprocess
    import sys

    # Required packages
    required_packages = {
        'yaml': 'pyyaml',
        'osmnx': 'osmnx',
        'networkx': 'networkx',
    }

    missing_packages = []

    # Check each package
    for import_name, pip_name in required_packages.items():
        try:
            module = __import__(import_name)
            print(colored(f"✓ {import_name} is already installed", "green"))
        except ImportError:
            print(colored(f"✗ {import_name} is not installed", "yellow"))
            missing_packages.append(pip_name)

    # If there are missing packages, try to install them
    if missing_packages:
        print(colored("Installing missing packages...", "blue"))
        try:
            # Upgrade pip first
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
            
            # Install each missing package
            for package in missing_packages:
                print(colored(f"Installing {package}...", "blue"))
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(colored(f"✓ Successfully installed {package}", "green"))

        except subprocess.CalledProcessError as e:
            print(colored(f"Failed to install packages: {e}", "red"))
            print(colored("Please try installing the packages manually:\n" + 
                        "\n".join([f"pip install {pkg}" for pkg in missing_packages]), "yellow"))
            return False
        except Exception as e:
            print(colored(f"An unexpected error occurred: {e}", "red"))
            return False

    print(colored("All required dependencies are satisfied!", "green"))
    return True
