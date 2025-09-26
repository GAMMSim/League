import networkx as nx
from typing import Optional, Dict, List, Any, Union
from typeguard import typechecked

try:
    from ..core.console import *
except ModuleNotFoundError:
    from lib.core.console import *


@typechecked
class AgentMap:
    """
    A class representing an agent map with flag and agent positions.
    Stores team information without self/opponent logic - just data storage.
    Supports partial/incremental updates with time tracking.
    """

    def __init__(self, graph: Optional[nx.MultiDiGraph] = None, current_time: int = 0):
        """
        Initialize the AgentMap instance.

        Parameters:
            graph (Optional[nx.MultiDiGraph]): A NetworkX graph object representing the agent map.
            current_time (int): Current time step for tracking information age.
        """
        self.graph = graph  # The NetworkX graph object
        self.current_time = current_time
        
        # Team agent positions: {team_name: {agent_name: {"position": pos, "time": timestamp}}}
        self.teams = {}
        
        # Team flag positions: {team_name: {flag_id: {"position": pos, "time": timestamp}}}
        self.flags = {}
        
        debug(f"AgentMap initialized with current_time={current_time}, graph={'attached' if graph else 'None'}")

    def attach_networkx_graph(self, nodes_data: Dict[int, Any], edges_data: Dict[int, Any]) -> None:
        """
        Create and attach a directed NetworkX graph based on provided nodes and edges data.

        Parameters:
            nodes_data (Dict[int, Any]): Dictionary where keys are node IDs and values are attribute dictionaries.
            edges_data (Dict[int, Any]): Dictionary where keys are edge IDs and values are dictionaries or objects with edge info.

        Returns:
            None
        """
        if not isinstance(nodes_data, dict) or not isinstance(edges_data, dict):
            error(f"Invalid input types: nodes_data={type(nodes_data)}, edges_data={type(edges_data)}")
            return
            
        info(f"Attaching NetworkX graph with {len(nodes_data)} nodes and {len(edges_data)} edges")
        
        self.graph = nx.MultiDiGraph()
        
        nodes_added = 0
        edges_added = 0
        edges_skipped = 0

        for node_id, attrs in nodes_data.items():
            if not isinstance(attrs, dict):
                try:
                    attrs = attrs.__dict__
                except Exception as e:
                    warning(f"Failed to convert node {node_id} attributes to dict: {e}")
                    attrs = {}
            self.graph.add_node(node_id, **attrs)
            nodes_added += 1

        for edge_id, edge_info in edges_data.items():
            if isinstance(edge_info, dict):
                source = edge_info.get("source")
                target = edge_info.get("target")
                # Create a copy to avoid modifying the original
                edge_attrs = edge_info.copy()
                # Remove source and target to avoid redundancy
                edge_attrs.pop("source", None)
                edge_attrs.pop("target", None)
            else:
                source = getattr(edge_info, "source", None)
                target = getattr(edge_info, "target", None)
                try:
                    edge_attrs = edge_info.__dict__.copy()
                    edge_attrs.pop("source", None)
                    edge_attrs.pop("target", None)
                except Exception as e:
                    warning(f"Failed to extract edge {edge_id} attributes: {e}")
                    edge_attrs = {}

            if source is None or target is None:
                warning(f"Skipping edge {edge_id} with invalid endpoints: source={source}, target={target}")
                edges_skipped += 1
                continue  # Skip edges with invalid endpoints

            # Store the edge ID as an attribute
            edge_attrs["edge_id"] = edge_id

            # Add the edge to the graph
            self.graph.add_edge(source, target, **edge_attrs)
            edges_added += 1
            
        success(f"Graph attached successfully: {nodes_added} nodes, {edges_added} edges added")
        if edges_skipped > 0:
            warning(f"{edges_skipped} edges were skipped due to invalid endpoints")

    def update_networkx_graph(self, nodes_data: Dict[int, Any], edges_data: Dict[int, Any]) -> None:
        """
        Update the existing graph with new nodes and edges information.

        Parameters:
            nodes_data (Dict[int, Any]): Dictionary where keys are node IDs and values are attribute dictionaries.
            edges_data (Dict[int, Any]): Dictionary where keys are edge IDs and values are dictionaries or objects with edge info.

        Returns:
            None
        """
        if not isinstance(nodes_data, dict) or not isinstance(edges_data, dict):
            error(f"Invalid input types for graph update: nodes_data={type(nodes_data)}, edges_data={type(edges_data)}")
            return
            
        debug(f"Updating NetworkX graph with {len(nodes_data)} nodes and {len(edges_data)} edges")
        
        if self.graph is None:
            warning("No existing graph found, creating new graph instead")
            self.attach_networkx_graph(nodes_data, edges_data)
            return

        nodes_updated = 0
        nodes_added = 0
        edges_updated = 0
        edges_added = 0
        edges_skipped = 0

        # Update nodes
        for node_id, attrs in nodes_data.items():
            if not isinstance(attrs, dict):
                try:
                    attrs = attrs.__dict__
                except Exception as e:
                    warning(f"Failed to convert node {node_id} attributes to dict during update: {e}")
                    attrs = {}
            if self.graph.has_node(node_id):
                self.graph.nodes[node_id].update(attrs)
                nodes_updated += 1
            else:
                self.graph.add_node(node_id, **attrs)
                nodes_added += 1

        # Create a mapping of edge_id to edge keys for existing edges
        edge_id_to_key = {}
        for u, v, k, data in self.graph.edges(data=True, keys=True):
            if "edge_id" in data:
                edge_id_to_key[(u, v, data["edge_id"])] = k

        # Update or add edges
        for edge_id, edge_info in edges_data.items():
            if isinstance(edge_info, dict):
                source = edge_info.get("source")
                target = edge_info.get("target")
                # Create a copy to avoid modifying the original
                edge_attrs = edge_info.copy()
                # Remove source and target to avoid redundancy
                edge_attrs.pop("source", None)
                edge_attrs.pop("target", None)
            else:
                source = getattr(edge_info, "source", None)
                target = getattr(edge_info, "target", None)
                try:
                    edge_attrs = edge_info.__dict__.copy()
                    edge_attrs.pop("source", None)
                    edge_attrs.pop("target", None)
                except Exception as e:
                    warning(f"Failed to extract edge {edge_id} attributes during update: {e}")
                    edge_attrs = {}

            if source is None or target is None:
                warning(f"Skipping edge {edge_id} update with invalid endpoints: source={source}, target={target}")
                edges_skipped += 1
                continue  # Skip edges with invalid endpoints

            # Add edge_id as an attribute
            edge_attrs["edge_id"] = edge_id

            # Check if this specific edge (by ID) exists
            key = edge_id_to_key.get((source, target, edge_id))
            if key is not None:
                # Update existing edge attributes
                for attr_key, attr_value in edge_attrs.items():
                    self.graph[source][target][key][attr_key] = attr_value
                edges_updated += 1
            else:
                # Add as a new edge
                self.graph.add_edge(source, target, **edge_attrs)
                edges_added += 1
                
        info(f"Graph update completed: {nodes_updated} nodes updated, {nodes_added} nodes added, {edges_updated} edges updated, {edges_added} edges added")
        if edges_skipped > 0:
            warning(f"{edges_skipped} edges were skipped during update due to invalid endpoints")

    def update_time(self, new_time: int) -> None:
        """
        Update the current time step.

        Parameters:
            new_time (int): New current time step.

        Returns:
            None
        """
        if new_time < 0:
            warning(f"Attempting to set negative time: {new_time}, ignoring update")
            return
            
        if new_time < self.current_time:
            warning(f"Time going backwards: current={self.current_time}, new={new_time}")
        
        old_time = self.current_time
        self.current_time = new_time
        debug(f"Time updated from {old_time} to {new_time}")

    def update_agent_position(self, team_name: str, agent_name: str, position: Any, time: int = None) -> None:
        """
        Update a single agent's position for a team.

        Parameters:
            team_name (str): Name of the team.
            agent_name (str): Name of the agent.
            position (Any): Agent's position.
            time (int, optional): Time of this update. Uses current_time if None.

        Returns:
            None
        """
        if not team_name or not agent_name:
            error(f"Invalid team_name='{team_name}' or agent_name='{agent_name}' - cannot be empty")
            return
            
        if time is None:
            time = self.current_time
        elif time < 0:
            warning(f"Negative time value {time} for agent {agent_name}, using current_time instead")
            time = self.current_time
            
        # Update current_time to the latest update time
        if time > self.current_time:
            old_time = self.current_time
            self.current_time = time
            debug(f"Current time advanced from {old_time} to {time} due to agent update")
            
        if team_name not in self.teams:
            debug(f"Creating new team '{team_name}' for agent '{agent_name}'")
            self.teams[team_name] = {}
            
        old_position = None
        if agent_name in self.teams[team_name]:
            old_position = self.teams[team_name][agent_name]["position"]
            
        self.teams[team_name][agent_name] = {
            "position": position,
            "time": time
        }
        
        if old_position is not None:
            debug(f"Updated agent '{agent_name}' position from {old_position} to {position} at time {time}")
        else:
            info(f"Added new agent '{agent_name}' to team '{team_name}' at position {position} (time {time})")

    def update_team_agents(self, team_name: str, agent_positions: Dict[str, Any], time: int = None) -> None:
        """
        Update multiple agents' positions for a team.

        Parameters:
            team_name (str): Name of the team.
            agent_positions (Dict[str, Any]): Dictionary mapping agent names to positions.
            time (int, optional): Time of this update. Uses current_time if None.

        Returns:
            None
        """
        if not team_name:
            error(f"Invalid team_name='{team_name}' - cannot be empty")
            return
            
        if not isinstance(agent_positions, dict):
            error(f"agent_positions must be a dictionary, got {type(agent_positions)}")
            return
            
        if not agent_positions:
            warning(f"Empty agent_positions dictionary for team '{team_name}'")
            return
            
        if time is None:
            time = self.current_time
        elif time < 0:
            warning(f"Negative time value {time} for team {team_name} agents, using current_time instead")
            time = self.current_time
            
        # Update current_time to the latest update time
        if time > self.current_time:
            old_time = self.current_time
            self.current_time = time
            debug(f"Current time advanced from {old_time} to {time} due to team update")
            
        if team_name not in self.teams:
            debug(f"Creating new team '{team_name}' for batch agent update")
            self.teams[team_name] = {}
        
        updated_count = 0
        new_count = 0
        
        for agent_name, position in agent_positions.items():
            if not agent_name:
                warning(f"Skipping empty agent name in team '{team_name}' update")
                continue
                
            if agent_name in self.teams[team_name]:
                updated_count += 1
            else:
                new_count += 1
                
            self.teams[team_name][agent_name] = {
                "position": position,
                "time": time
            }
            
        info(f"Team '{team_name}' batch update: {updated_count} agents updated, {new_count} new agents added (time {time})")

    def update_flag_position(self, team_name: str, flag_id: str, position: Any, time: int = None) -> None:
        """
        Update a single flag's position for a team.

        Parameters:
            team_name (str): Name of the team owning the flag.
            flag_id (str): Identifier for the flag.
            position (Any): Flag's position.
            time (int, optional): Time of this update. Uses current_time if None.

        Returns:
            None
        """
        if not team_name or not flag_id:
            error(f"Invalid team_name='{team_name}' or flag_id='{flag_id}' - cannot be empty")
            return
            
        if time is None:
            time = self.current_time
        elif time < 0:
            warning(f"Negative time value {time} for flag {flag_id}, using current_time instead")
            time = self.current_time
            
        # Update current_time to the latest update time
        if time > self.current_time:
            old_time = self.current_time
            self.current_time = time
            debug(f"Current time advanced from {old_time} to {time} due to flag update")
            
        if team_name not in self.flags:
            debug(f"Creating new flag team '{team_name}' for flag '{flag_id}'")
            self.flags[team_name] = {}
            
        old_position = None
        if flag_id in self.flags[team_name]:
            old_position = self.flags[team_name][flag_id]["position"]
            
        self.flags[team_name][flag_id] = {
            "position": position,
            "time": time
        }
        
        if old_position is not None:
            debug(f"Updated flag '{flag_id}' position from {old_position} to {position} at time {time}")
        else:
            info(f"Added new flag '{flag_id}' to team '{team_name}' at position {position} (time {time})")

    def update_team_flags(self, team_name: str, flag_positions: Dict[str, Any], time: int = None) -> None:
        """
        Update multiple flags' positions for a team.

        Parameters:
            team_name (str): Name of the team owning the flags.
            flag_positions (Dict[str, Any]): Dictionary mapping flag IDs to positions.
            time (int, optional): Time of this update. Uses current_time if None.

        Returns:
            None
        """
        if not team_name:
            error(f"Invalid team_name='{team_name}' - cannot be empty")
            return
            
        if not isinstance(flag_positions, dict):
            error(f"flag_positions must be a dictionary, got {type(flag_positions)}")
            return
            
        if not flag_positions:
            warning(f"Empty flag_positions dictionary for team '{team_name}'")
            return
            
        if time is None:
            time = self.current_time
        elif time < 0:
            warning(f"Negative time value {time} for team {team_name} flags, using current_time instead")
            time = self.current_time
            
        # Update current_time to the latest update time
        if time > self.current_time:
            old_time = self.current_time
            self.current_time = time
            debug(f"Current time advanced from {old_time} to {time} due to flag team update")
            
        if team_name not in self.flags:
            debug(f"Creating new flag team '{team_name}' for batch flag update")
            self.flags[team_name] = {}
            
        updated_count = 0
        new_count = 0
        
        for flag_id, position in flag_positions.items():
            if not flag_id:
                warning(f"Skipping empty flag_id in team '{team_name}' flag update")
                continue
                
            if flag_id in self.flags[team_name]:
                updated_count += 1
            else:
                new_count += 1
                
            self.flags[team_name][flag_id] = {
                "position": position,
                "time": time
            }
            
        info(f"Team '{team_name}' flag batch update: {updated_count} flags updated, {new_count} new flags added (time {time})")

    def get_team_agents(self, team_name: str) -> List[tuple]:
        """
        Retrieve agent positions and time info for a specific team.

        Parameters:
            team_name (str): Name of the team.

        Returns:
            List[tuple]: List of (agent_name, position, time_age) tuples.
                        time_age is how old the information is (current_time - last_update_time).
        """
        if not team_name:
            warning("Empty team_name provided to get_team_agents")
            return []
            
        if team_name not in self.teams:
            debug(f"Team '{team_name}' not found in agent records")
            return []
            
        result = []
        for agent_name, info in self.teams[team_name].items():
            # time_age = current_time - last_update_time (always >= 0 if last_update <= current)
            time_age = self.current_time - info["time"]
            if time_age < 0:
                warning(f"Agent '{agent_name}' has future timestamp: current={self.current_time}, agent_time={info['time']}")
            result.append((agent_name, info["position"], time_age))
            
        debug(f"Retrieved {len(result)} agents for team '{team_name}'")
        return result

    def get_agent_position(self, team_name: str, agent_name: str) -> tuple:
        """
        Get a specific agent's position and time info.

        Parameters:
            team_name (str): Name of the team.
            agent_name (str): Name of the agent.

        Returns:
            tuple: (position, time_age) or (None, None) if not found.
                  time_age is how old the information is (current_time - last_update_time).
        """
        if not team_name or not agent_name:
            warning(f"Empty team_name='{team_name}' or agent_name='{agent_name}' in get_agent_position")
            return None, None
            
        if team_name in self.teams and agent_name in self.teams[team_name]:
            info = self.teams[team_name][agent_name]
            time_age = self.current_time - info["time"]
            if time_age < 0:
                warning(f"Agent '{agent_name}' has future timestamp: current={self.current_time}, agent_time={info['time']}")
            debug(f"Found agent '{agent_name}' in team '{team_name}' at position {info['position']} (age: {time_age})")
            return info["position"], time_age
        else:
            debug(f"Agent '{agent_name}' not found in team '{team_name}'")
            return None, None

    def get_team_flags(self, team_name: str) -> List[tuple]:
        """
        Retrieve flag positions and time info for a specific team.

        Parameters:
            team_name (str): Name of the team.

        Returns:
            List[tuple]: List of (flag_id, position, time_age) tuples.
                        time_age is how old the information is (current_time - last_update_time).
        """
        if team_name not in self.flags:
            return []
            
        result = []
        for flag_id, info in self.flags[team_name].items():
            time_age = self.current_time - info["time"]
            result.append((flag_id, info["position"], time_age))
        return result

    def get_flag_position(self, team_name: str, flag_id: str) -> tuple:
        """
        Get a specific flag's position and time info.

        Parameters:
            team_name (str): Name of the team.
            flag_id (str): Identifier for the flag.

        Returns:
            tuple: (position, time_age) or (None, None) if not found.
                  time_age is how old the information is (current_time - last_update_time).
        """
        if team_name in self.flags and flag_id in self.flags[team_name]:
            info = self.flags[team_name][flag_id]
            time_age = self.current_time - info["time"]
            return info["position"], time_age
        return None, None

    def get_all_teams(self) -> List[str]:
        """
        Get list of all team names that have agents or flags.

        Returns:
            List[str]: List of team names.
        """
        agent_teams = set(self.teams.keys())
        flag_teams = set(self.flags.keys())
        return list(agent_teams.union(flag_teams))

    def get_all_agents(self) -> Dict[str, List[tuple]]:
        """
        Get all agents from all teams with time info.

        Returns:
            Dict[str, List[tuple]]: Dictionary mapping team names to lists of 
                                   (agent_name, position, time_age) tuples.
        """
        result = {}
        for team_name in self.teams:
            result[team_name] = self.get_team_agents(team_name)
        return result

    def get_all_flags(self) -> Dict[str, List[tuple]]:
        """
        Get all flags from all teams with time info.

        Returns:
            Dict[str, List[tuple]]: Dictionary mapping team names to lists of 
                                   (flag_id, position, time_age) tuples.
        """
        result = {}
        for team_name in self.flags:
            result[team_name] = self.get_team_flags(team_name)
        return result

    def remove_team(self, team_name: str) -> None:
        """
        Remove a team and all its agents and flags.
        
        Parameters:
            team_name (str): Name of the team to remove.
        """
        if not team_name:
            warning("Empty team_name provided to remove_team")
            return
            
        agents_removed = 0
        flags_removed = 0
        
        if team_name in self.teams:
            agents_removed = len(self.teams[team_name])
            del self.teams[team_name]
            
        if team_name in self.flags:
            flags_removed = len(self.flags[team_name])
            del self.flags[team_name]
            
        if agents_removed > 0 or flags_removed > 0:
            info(f"Removed team '{team_name}': {agents_removed} agents, {flags_removed} flags")
        else:
            debug(f"Team '{team_name}' not found for removal")

    def remove_agent(self, team_name: str, agent_name: str) -> None:
        """
        Remove a specific agent from a team.
        
        Parameters:
            team_name (str): Name of the team.
            agent_name (str): Name of the agent to remove.
        """
        if not team_name or not agent_name:
            warning(f"Empty team_name='{team_name}' or agent_name='{agent_name}' in remove_agent")
            return
            
        if team_name in self.teams and agent_name in self.teams[team_name]:
            old_position = self.teams[team_name][agent_name]["position"]
            del self.teams[team_name][agent_name]
            info(f"Removed agent '{agent_name}' from team '{team_name}' (was at position {old_position})")
            
            # Remove team if it has no agents left
            if not self.teams[team_name]:
                del self.teams[team_name]
                debug(f"Team '{team_name}' removed as it has no remaining agents")
        else:
            warning(f"Agent '{agent_name}' not found in team '{team_name}' for removal")

    def remove_flag(self, team_name: str, flag_id: str) -> None:
        """
        Remove a specific flag from a team.
        
        Parameters:
            team_name (str): Name of the team.
            flag_id (str): Identifier of the flag to remove.
        """
        if not team_name or not flag_id:
            warning(f"Empty team_name='{team_name}' or flag_id='{flag_id}' in remove_flag")
            return
            
        if team_name in self.flags and flag_id in self.flags[team_name]:
            old_position = self.flags[team_name][flag_id]["position"]
            del self.flags[team_name][flag_id]
            info(f"Removed flag '{flag_id}' from team '{team_name}' (was at position {old_position})")
            
            # Remove team from flags if it has no flags left
            if not self.flags[team_name]:
                del self.flags[team_name]
                debug(f"Team '{team_name}' removed from flags as it has no remaining flags")
        else:
            warning(f"Flag '{flag_id}' not found in team '{team_name}' for removal")

    def get_team_count(self) -> int:
        """
        Get the total number of teams.
        
        Returns:
            int: Number of distinct teams.
        """
        return len(self.get_all_teams())

    def get_agent_count(self, team_name: Optional[str] = None) -> int:
        """
        Get the number of agents for a team or all teams.
        
        Parameters:
            team_name (str, optional): Specific team name. If None, returns total count.
            
        Returns:
            int: Number of agents.
        """
        if team_name is None:
            return sum(len(agents) for agents in self.teams.values())
        else:
            return len(self.teams.get(team_name, {}))

    def get_flag_count(self, team_name: Optional[str] = None) -> int:
        """
        Get the number of flags for a team or all teams.
        
        Parameters:
            team_name (str, optional): Specific team name. If None, returns total count.
            
        Returns:
            int: Number of flags.
        """
        if team_name is None:
            return sum(len(flags) for flags in self.flags.values())
        else:
            return len(self.flags.get(team_name, {}))


    def __str__(self) -> str:
        """
        String representation of the AgentMap.

        Returns:
            str: A formatted string with basic map information.
        """
        info_str = "AgentMap\n"
        info_str += f"Current Time: {self.current_time}\n"
        
        if self.graph is not None:
            info_str += f"Nodes: {len(self.graph.nodes)}\n"
            info_str += f"Edges: {len(self.graph.edges)}\n"
        else:
            info_str += "Graph: None\n"
        
        # Handle empty or None teams
        if not self.teams:
            info_str += "Teams: 0\n"
            info_str += "Total agents: 0\n"
            if not self.flags:
                info_str += "Total flags: 0\n"
            else:
                info_str += f"Total flags: {self.get_flag_count()}\n"
            return info_str
            
        # We have teams, show details
        all_teams = self.get_all_teams()
        info_str += f"Teams: {len(all_teams)}\n"
        
        for team_name in all_teams:
            agent_count = self.get_agent_count(team_name)
            flag_count = self.get_flag_count(team_name)
            info_str += f"  {team_name}: {agent_count} agents, {flag_count} flags\n"
        
        info_str += f"Total agents: {self.get_agent_count()}\n"
        info_str += f"Total flags: {self.get_flag_count()}\n"
        
        return info_str


def debug_agent_map():
    """
    Test key functionality of the AgentMap class.
    """
    print("=== AgentMap Debug Tests ===\n")
    
    # Test 1: Basic initialization
    print("1. Testing initialization...")
    agent_map = AgentMap(current_time=0)
    print(f"Initial state: {agent_map}")
    print()
    
    # Test 2: Individual agent updates
    print("2. Testing individual agent updates...")
    agent_map.update_agent_position("red", "red_0", position=10, time=0)
    agent_map.update_agent_position("blue", "blue_0", position=5, time=1)
    print(f"After individual updates: {agent_map}")
    
    # Check specific agent
    pos, age = agent_map.get_agent_position("red", "red_0")
    print(f"red_0 position: {pos}, age: {age}")
    print()
    
    # Test 3: Batch agent updates
    print("3. Testing batch agent updates...")
    agent_map.update_team_agents("red", {"red_1": 15, "red_2": 20}, time=2)
    agent_map.update_team_agents("blue", {"blue_1": 8, "blue_2": 12}, time=2)
    print(f"After batch updates: {agent_map}")
    
    # Get all red team agents
    red_agents = agent_map.get_team_agents("red")
    print(f"Red team agents: {red_agents}")
    print()
    
    # Test 4: Flag updates
    print("4. Testing flag updates...")
    agent_map.update_flag_position("red", "flag_red_1", position=25, time=3)
    agent_map.update_team_flags("blue", {"flag_blue_1": 30, "flag_blue_2": 35}, time=3)
    print(f"After flag updates: {agent_map}")
    
    # Get specific flag
    flag_pos, flag_age = agent_map.get_flag_position("red", "flag_red_1")
    print(f"red flag_red_1 position: {flag_pos}, age: {flag_age}")
    print()
    
    # Test 5: Time progression and age calculation
    print("5. Testing time progression...")
    agent_map.update_time(5)
    print(f"Updated time to 5")
    
    # Check ages
    pos, age = agent_map.get_agent_position("red", "red_0")
    print(f"red_0 position: {pos}, age: {age} (should be 5)")
    
    blue_agents = agent_map.get_team_agents("blue")
    print(f"Blue team agents with ages: {blue_agents}")
    print()
    
    # Test 6: Partial updates over time
    print("6. Testing partial updates over time...")
    agent_map.update_agent_position("red", "red_0", position=50, time=1)  # Update with older time
    agent_map.update_agent_position("green", "green_0", position=100, time=2)  # New team with older time
    agent_map.update_time(10)  # Advance time significantly
    
    print(f"After partial updates and time advancement: {agent_map}")
    
    # Test all teams
    all_teams = agent_map.get_all_teams()
    print(f"All teams: {all_teams}")
    
    # Test all agents
    all_agents = agent_map.get_all_agents()
    print(f"All agents by team: {all_agents}")
    print()
    
    # Test 7: Removal operations
    print("7. Testing removal operations...")
    agent_map.remove_agent("blue", "blue_0")
    agent_map.remove_flag("blue", "flag_blue_1")
    print(f"After removing blue_0 agent and flag_blue_1: {agent_map}")
    
    # Test 8: Edge cases
    print("8. Testing edge cases...")
    
    # Non-existent queries
    pos, age = agent_map.get_agent_position("nonexistent", "agent")
    print(f"Non-existent agent query: position={pos}, age={age}")
    
    empty_team = agent_map.get_team_agents("nonexistent")
    print(f"Non-existent team query: {empty_team}")
    
    # Remove entire team
    agent_map.remove_team("green")
    print(f"After removing green team: {agent_map}")
    print()

if __name__ == "__main__":
    debug_agent_map()

