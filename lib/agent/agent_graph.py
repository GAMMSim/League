import networkx as nx
from typing import Optional, Dict, List, Any, Union
from typeguard import typechecked

try:
    from ..core.console import *
except ModuleNotFoundError:
    from lib.core.console import *


@typechecked
class AgentGraph:
    """
    A class representing an agent graph with flag and agent positions.
    Updated for alpha/beta team system.
    """

    def __init__(self, graph: Optional[nx.MultiDiGraph] = None):
        """
        Initialize the AgentGraph instance.

        Parameters:
            graph (Optional[nx.MultiDiGraph]): A NetworkX graph object representing the agent graph.
        """
        self.graph = graph  # The NetworkX graph object
        self.agent_dict = {}
        self.alpha_dict = {}
        self.beta_dict = {}
        # Keep old names for backward compatibility
        self.attacker_dict = {}
        self.defender_dict = {}
        self.flag_positions = []
        self.flag_weights = []

    def attach_networkx_graph(self, nodes_data: Dict[int, Any], edges_data: Dict[int, Any]) -> None:
        """
        Create and attach a directed NetworkX graph based on provided nodes and edges data.

        Parameters:
            nodes_data (Dict[int, Any]): Dictionary where keys are node IDs and values are attribute dictionaries.
            edges_data (Dict[int, Any]): Dictionary where keys are edge IDs and values are dictionaries or objects with edge info.

        Returns:
            None
        """
        self.graph = nx.MultiDiGraph()

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
                except Exception:
                    edge_attrs = {}

            if source is None or target is None:
                continue  # Skip edges with invalid endpoints

            # Store the edge ID as an attribute
            edge_attrs["edge_id"] = edge_id

            # Add the edge to the graph
            self.graph.add_edge(source, target, **edge_attrs)

    def update_networkx_graph(self, nodes_data: Dict[int, Any], edges_data: Dict[int, Any]) -> None:
        """
        Update the existing graph with new nodes and edges information.

        Parameters:
            nodes_data (Dict[int, Any]): Dictionary where keys are node IDs and values are attribute dictionaries.
            edges_data (Dict[int, Any]): Dictionary where keys are edge IDs and values are dictionaries or objects with edge info.

        Returns:
            None
        """
        if self.graph is None:
            self.attach_networkx_graph(nodes_data, edges_data)
            return

        # Update nodes
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
                except Exception:
                    edge_attrs = {}

            if source is None or target is None:
                continue  # Skip edges with invalid endpoints

            # Add edge_id as an attribute
            edge_attrs["edge_id"] = edge_id

            # Check if this specific edge (by ID) exists
            key = edge_id_to_key.get((source, target, edge_id))
            if key is not None:
                # Update existing edge attributes
                for attr_key, attr_value in edge_attrs.items():
                    self.graph[source][target][key][attr_key] = attr_value
            else:
                # Add as a new edge
                self.graph.add_edge(source, target, **edge_attrs)

    def set_agent_dict(self, agent_info: Dict[str, Any]) -> None:
        """
        Set the agent positions from the given information.
        Updated for alpha/beta team system.

        Parameters:
            agent_info (Dict[str, Any]): Dictionary mapping agent names to their positions.

        Returns:
            None
        """
        self.agent_dict = {name: info for name, info in agent_info.items()}
        
        # Update team dictionaries for alpha/beta system
        self.alpha_dict = {name: info for name, info in agent_info.items() if "alpha" in name}
        self.beta_dict = {name: info for name, info in agent_info.items() if "beta" in name}
        
        # Keep old attacker/defender dicts for backward compatibility
        # You can switch these assignments based on which team is attacking/defending
        self.attacker_dict = self.alpha_dict  # Can be switched based on game context
        self.defender_dict = self.beta_dict

    def set_flag_positions(self, flag_positions: List[Any]) -> None:
        """
        Set the flag positions.

        Parameters:
            flag_positions (List[Any]): List of flag positions.

        Returns:
            None
        """
        self.flag_positions = flag_positions

    def set_flag_weights(self, flag_weights: Optional[List[float]]) -> None:
        """
        Set the flag weights.

        Parameters:
            flag_weights (List[float]): List of flag weights.

        Returns:
            None
        """
        self.flag_weights = flag_weights

    def get_team_positions(self, team: str) -> List[tuple]:
        """
        Retrieve the positions of agents belonging to a specified team.
        Updated to return (name, position) tuples as expected by strategy.

        Parameters:
            team (str): The team identifier ("alpha", "beta", "attacker", or "defender").

        Returns:
            List[tuple]: List of (agent_name, position) tuples for agents that belong to the team.
        """
        if not self.agent_dict:
            return []
        
        # Handle both new (alpha/beta) and old (attacker/defender) team names
        if team == "alpha":
            return [(name, pos) for name, pos in self.alpha_dict.items()]
        elif team == "beta":
            return [(name, pos) for name, pos in self.beta_dict.items()]
        elif team == "attacker":
            return [(name, pos) for name, pos in self.attacker_dict.items()]
        elif team == "defender":
            return [(name, pos) for name, pos in self.defender_dict.items()]
        else:
            # Fallback: search by team name in agent name
            return [(name, pos) for name, pos in self.agent_dict.items() if team in name]

    def set_team_positions(self, team: str, team_dict: Dict[str, Any]) -> None:
        """
        Set positions for a specific team.
        
        Parameters:
            team (str): Team identifier ("alpha" or "beta").
            team_dict (Dict[str, Any]): Dictionary mapping agent names to positions.
        """
        if team == "alpha":
            self.alpha_dict = team_dict
            self.attacker_dict = team_dict  # Update backward compatibility
        elif team == "beta":
            self.beta_dict = team_dict
            self.defender_dict = team_dict  # Update backward compatibility

    def get_agent_dicts(self) -> tuple:
        """
        Retrieve the positions of all agents.
        Updated for alpha/beta system.

        Returns:
            tuple: A tuple containing (agent_dict, alpha_dict, beta_dict)
        """
        return self.agent_dict, self.alpha_dict, self.beta_dict

    def get_flag_positions(self) -> List[Any]:
        """
        Retrieve the positions of all flags.

        Returns:
            List[Any]: List of flag positions.
        """
        return self.flag_positions

    def get_flag_weights(self) -> List[float]:
        """
        Retrieve the weights of all flags.

        Returns:
            List[float]: List of flag weights.
        """
        return self.flag_weights

    def shortest_path_to(self, source: Any, target: Any, speed: float) -> Any:
        try:
            from lib.utils.strategy_utils import compute_shortest_path_step
        except ModuleNotFoundError:
            from ..utils.strategy_utils import compute_shortest_path_step
        return compute_shortest_path_step(self.graph, source, target, speed)

    def __str__(self) -> str:
        """
        String representation of the AgentGraph.

        Returns:
            str: A formatted string with basic graph information.
        """
        if self.graph is None:
            return "AgentGraph (empty)"

        info_str = "AgentGraph\n"
        info_str += f"Nodes: {len(self.graph.nodes)}\n"
        info_str += f"Edges: {len(self.graph.edges)}\n"
        info_str += f"Alpha agents: {len(self.alpha_dict)}\n"
        info_str += f"Beta agents: {len(self.beta_dict)}\n"
        info_str += f"Total agents: {len(self.agent_dict)}\n"
        info_str += f"Flags: {len(self.flag_positions)}\n"
        return info_str


if __name__ == "__main__":
    # Example usage and demonstration
    # Create a sample graph
    sample_nodes = {"A": {"type": "junction"}, "B": {"type": "junction"}, "C": {"type": "junction"}}

    sample_edges = {"edge1": {"source": "A", "target": "B", "weight": 1.0}, "edge2": {"source": "B", "target": "C", "weight": 2.0}, "edge3": {"source": "C", "target": "A", "weight": 3.0}}

    try:
        ag = AgentGraph()
        ag.attach_networkx_graph(sample_nodes, sample_edges)

        # Set some agents with alpha/beta naming
        agents = {"alpha_0": "A", "alpha_1": "B", "beta_0": "C", "beta_1": "A"}
        ag.set_agent_dict(agents)

        # Set flags
        ag.set_flag_positions(["A", "C"])
        ag.set_flag_weights([1.0, 2.0])

        print(ag)
        
        # Test team position retrieval
        print(f"Alpha team: {ag.get_team_positions('alpha')}")
        print(f"Beta team: {ag.get_team_positions('beta')}")

    except Exception as e:
        print(f"Error in demo: {e}")