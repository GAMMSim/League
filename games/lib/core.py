# This file defined the interaction model between agents and flags
# WARNING: Before edit, make sure you know what you want and check out the Advanced Usage part of the documentation.
import networkx as nx
from lib.interface import colored


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
                attacker_capture_radius = getattr(agent_params[attacker.name], 'capture_radius', 0)
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
                defender_capture_radius = getattr(agent_params[defender.name], 'capture_radius', 0)
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
                attacker_capture_radius = getattr(agent_params[attacker.name], 'capture_radius', 0)
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
