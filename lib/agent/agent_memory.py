from typeguard import typechecked
try:
    from ..agent.agent_graph import AgentGraph
    from ..core.core import *
except ModuleNotFoundError:
    from lib.agent.agent_graph import AgentGraph
    from lib.core.core import *

@typechecked
class AgentMemory:
    def __init__(self, speed: float, capture_radius: float, map: AgentGraph, start_node_id: int, **kwargs: Any) -> None:
        """
        Initialize the AgentMemory with required properties and additional custom parameters.

        Args:
            speed (float): The agent's speed.
            capture_radius (float): The agent's capture radius.
            map (AgentGraph): The map (AgentGraph) the agent is operating on.
            start_node_id (int): The starting node identifier for the agent.
            **kwargs: Additional custom parameters.
        """
        self.speed = speed
        self.capture_radius = capture_radius
        self.map = map
        self.start_node_id = start_node_id

        # Store any additional custom attributes.
        for key, value in kwargs.items():
            setattr(self, key, value)

    def update_memory(self, **kwargs: Any) -> None:
        """
        Update the agent's memory with additional key-value pairs.

        Args:
            **kwargs: Key-value pairs to update the agent memory.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the agent's memory.

        Returns:
            Dict[str, Any]: A dictionary containing all attributes of the agent.
        """
        return self.__dict__

    def get_attribute(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Retrieve a specific attribute from the agent's memory.

        Args:
            key (str): The attribute name.
            default (Any, optional): Default value if the attribute is not found. Defaults to None.

        Returns:
            Any: The attribute's value if it exists, else the default.
        """
        return getattr(self, key, default)

    def __str__(self) -> str:
        """
        Return a string representation of the agent's memory.

        Returns:
            str: The string representation.
        """
        return f"AgentMemory({self.to_dict()})"