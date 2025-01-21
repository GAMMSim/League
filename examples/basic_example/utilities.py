# This is the utility file for the basic_example game.
import networkx as nx


class Graph:
    def __init__(self, graph=None):
        self.graph = graph
        self.agent_positions = {}
        self.flag_positions = []
        self.flag_weights = []

    def attach_networkx_graph(self, nodes_data, edges_data):
        self.graph = nx.DiGraph()

        for node_id, attrs in nodes_data.items():
            if not isinstance(attrs, dict):
                try:
                    attrs = attrs.__dict__
                except Exception:
                    attrs = {}
            self.graph.add_node(node_id, **attrs)

        for edge_id, edge_info in edges_data.items():
            if isinstance(edge_info, dict):
                source = edge_info.get("source")
                target = edge_info.get("target")
                edge_attrs = edge_info
            else:
                source = getattr(edge_info, "source", None)
                target = getattr(edge_info, "target", None)
                try:
                    edge_attrs = edge_info.__dict__
                except Exception:
                    edge_attrs = {}

            if source is None or target is None:
                continue  # Skip edges with invalid endpoints

            self.graph.add_edge(source, target, **edge_attrs)

    def update_networkx_graph(self, nodes_data, edges_data):
        if self.graph is None:
            self.attach_networkx_graph(nodes_data, edges_data)
            return
        for node_id, attrs in nodes_data.items():
            if not isinstance(attrs, dict):
                try:
                    attrs = attrs.__dict__
                except Exception:
                    attrs = {}
            if self.graph.has_node(node_id):
                self.graph.nodes[node_id].update(attrs)
            else:
                self.graph.add_node(node_id, **attrs)

        for edge_id, edge_info in edges_data.items():
            if isinstance(edge_info, dict):
                source = edge_info.get("source")
                target = edge_info.get("target")
                edge_attrs = edge_info
            else:
                source = getattr(edge_info, "source", None)
                target = getattr(edge_info, "target", None)
                try:
                    edge_attrs = edge_info.__dict__
                except Exception:
                    edge_attrs = {}
            if source is None or target is None:
                continue  # Skip edges with invalid endpoints

            if self.graph.has_edge(source, target):
                self.graph[source][target].update(edge_attrs)
            else:
                self.graph.add_edge(source, target, **edge_attrs)

    def set_agent_positions(self, agent_info):
        self.agent_positions = {name: info for name, info in agent_info.items()}

    def set_flag_positions(self, FLAG_POSITIONS):
        self.flag_positions = FLAG_POSITIONS

    def set_flag_weights(self, FLAG_WEIGHTS):
        self.flag_weights = FLAG_WEIGHTS
        
    def shortest_path_to(self, source_node, target_node, speed=1):
        if not isinstance(target_node, (list, tuple, set)):
            target_nodes = [target_node]
        else:
            target_nodes = list(target_node)

        best_path = None
        best_length = float('inf')

        # Compute the shortest path for each candidate target
        for t in target_nodes:
            try:
                length, path = nx.single_source_dijkstra(self.graph, source=source_node, target=t, weight='length')
                # Use the length of the path as the comparison metric
                if length < best_length:
                    best_length = length
                    best_path = path
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

        if best_path is None:
            return None
        
        index = speed if speed < len(best_path) else len(best_path) - 1
        return best_path[index]
    
    def get_neighbors(self, source_node, speed):
        # Get all nodes within the given distance (speed) from the source_node
        neighbors = nx.single_source_shortest_path_length(self.graph, source_node, cutoff=speed)
        # Remove the source node itself from the neighbors
        neighbors.pop(source_node, None)
        return list(neighbors.keys())
    
    def get_team_positions(self, team):
        if not hasattr(self, 'agent_positions') or not self.agent_positions:
            return []
        return [pos for name, pos in self.agent_positions.items() if team in name]
        
def extract_map_sensor_data(state):
    sensor_data = state.get('sensor', {})
    map_sensor = sensor_data.get('map')
    if map_sensor is None:
        raise ValueError("No map sensor data found in state.")
    
    sensor_type, map_data = map_sensor
    if not (isinstance(map_data, tuple) and len(map_data) == 2):
        raise ValueError("Map sensor data is not in the expected format (nodes_data, edges_data).")
    
    nodes_data, edges_data = map_data
    return nodes_data, edges_data

def extract_neighbor_sensor_data(state):
    sensor_data = state.get('sensor', {})
    neighbor_sensor = sensor_data.get('neighbor')
    if neighbor_sensor is None:
        raise ValueError("No neighbor sensor data found in state.")
    
    # Unpack the neighbor sensor tuple.
    sensor_type, neighbor_data = neighbor_sensor
    return neighbor_data

def extract_agent_sensor_data(state):
    sensor_data = state.get('sensor', {})
    agent_sensor = sensor_data.get('agent')
    if agent_sensor is None:
        raise ValueError("No agent sensor data found in state.")
    
    # Unpack the sensor tuple: sensor_type, agent_info.
    sensor_type, agent_info = agent_sensor
    return agent_info

def extract_sensor_data(state, FLAG_POSITIONS, FLAG_WEIGHTS, agent):
    nodes_data, edges_data =  extract_map_sensor_data(state)
    agent_info = extract_agent_sensor_data(state)
    agent.map.update_networkx_graph(nodes_data, edges_data)
    agent.map.set_agent_positions(agent_info)
    agent.map.set_flag_positions(FLAG_POSITIONS)
    agent.map.set_flag_weights(FLAG_WEIGHTS)
    attacker_positions = agent.map.get_team_positions("attacker")
    defender_positions = agent.map.get_team_positions("defender")
    return attacker_positions, defender_positions

def check_agent_interaction(ctx, model="kill"):
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
                distance = nx.shortest_path_length(ctx.graph.graph,
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