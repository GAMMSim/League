import networkx as nx

def check_agent_interaction(ctx, G, agent_params, model="kill"):
    attackers = []
    defenders = []
    for agent in ctx.agent.create_iter():
        team = agent.team
        if team == "attacker":
            attackers.append(agent)
        elif team == "defender":
            defenders.append(agent)
    
    # Check each attacker against each defender.
    for defender in defenders:
        for attacker in attackers:
            try:
                # Compute the shortest path distance between the attacker and defender.
                distance = nx.shortest_path_length(G,
                                                   source=attacker.current_node_id,
                                                   target=defender.current_node_id)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue  # Skip if there is no connection
            
            # Retrieve defender's capture radius (default to 1 if not defined).
            capture_radius = agent_params[defender.name].capture_radius
            if distance <= capture_radius:
                # An interaction takes place.
                if model == "kill":
                    # Defender kills the attacker.
                    print(f"[Interaction: kill] Defender {defender.name} kills attacker {attacker.name}.")
                    ctx.agent.delete_agent(attacker.name)
                    attackers.remove(attacker)
                elif model == "respawn":
                    # Attacker respawns.
                    print(f"[Interaction: respawn] Attacker {attacker.name} respawns due to interaction with defender {defender.name}.")
                    attacker.prev_node_id = attacker.current_node_id
                    attacker.current_node_id = attacker.start_node_id
                elif model == "both_kill":
                    # Both agents are killed (set to inactive).
                    print(f"[Interaction: both_kill] Both attacker {attacker.name} and defender {defender.name} are killed.")
                    ctx.agent.delete_agent(attacker.name)
                    attackers.remove(attacker)
                    ctx.agent.delete_agent(defender.name)
                    defenders.remove(defender)
                elif model == "both_respawn":
                    # Both agents respawn (reset to start positions and become active).
                    print(f"[Interaction: both_respawn] Both attacker {attacker.name} and defender {defender.name} respawn.")
                    attacker.prev_node_id = attacker.current_node_id
                    attacker.current_node_id = attacker.start_node_id
                    defender.prev_node_id = defender.current_node_id
                    defender.current_node_id = defender.start_node_id
                else:
                    print(f"Unknown interaction model: {model}")