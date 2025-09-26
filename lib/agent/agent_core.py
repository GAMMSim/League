from typeguard import typechecked
from typing import Any, Optional, Dict

try:
    from agent_map import AgentMap
    from ..core.console import *
    from ..core.cache import Cache
except ModuleNotFoundError:
    from lib.agent.agent_map import AgentMap
    from lib.core.console import *
    from lib.core.cache import Cache


@typechecked
class AgentController:
    """Controller that manages both gamms Agent (visualization) and game logic."""

    def __init__(self, gamms_agent: Any, speed: float, map: AgentMap, start_node_id: int, team: str, cache: Optional[Cache] = None, **kwargs: Any) -> None:
        """
        Initialize the AgentController with a gamms Agent and game properties.

        Args:
            gamms_agent: The gamms Agent instance for visualization/animation.
            speed (float): The agent's speed.
            map (AgentMap): The map (AgentMap) the agent is operating on.
            start_node_id (int): The starting node identifier for the agent.
            team (str): The team the agent belongs to (e.g., "red" or "blue").
            cache (Optional[Cache]): Cache for storing temporary data. If None, creates empty cache.
            **kwargs: Additional custom parameters (e.g., capture_radius, tagging_radius, sensors).
        """
        # Store reference to gamms agent
        self.gamms_agent = gamms_agent

        # Game logic properties
        self.speed = speed
        self.map = map
        self.start_node_id = start_node_id
        self.team = team
        self.alive = True
        self.death_position = None

        # Initialize cache - create new if not provided
        self.cache = cache if cache is not None else Cache()

        # Initialize current position to starting position
        self.current_position = start_node_id

        # Store any additional custom attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    # Delegate gamms Agent methods
    def get_state(self) -> Dict[str, Any]:
        """Get state from gamms agent."""
        return self.gamms_agent.get_state()

    def set_state(self, state: Optional[Dict] = None) -> None:
        """Set state on gamms agent."""
        if state:
            error(f"GAMMS agent's state should not be passed in directly.")
        self.gamms_agent.set_state()

    @property
    def name(self) -> str:
        """Get agent name from gamms agent."""
        return self.gamms_agent.name

    @property
    def strategy(self):
        """Get strategy from gamms agent."""
        return getattr(self.gamms_agent, "strategy", None)

    @strategy.setter
    def strategy(self, value):
        """Set strategy on gamms agent."""
        self.gamms_agent.register_strategy(value)

    # Game logic methods
    def update_position(self, new_position: int) -> None:
        """
        Update the agent's current position.

        Args:
            new_position (int): The new node ID where the agent is located.
        """
        self.current_position = new_position
        # Also update gamms agent state if needed
        state = self.get_state()
        state["action"] = new_position
        self.set_state()

    def update_memory(self, **kwargs: Any) -> None:
        """
        Update the agent's memory with additional key-value pairs.

        Args:
            **kwargs: Key-value pairs to update the agent memory.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def update_cache(self, key: str, value: Any) -> None:
        """
        Update a specific entry in the agent's cache.

        Args:
            key (str): The cache key.
            value (Any): The value to store in cache.
        """
        self.cache.set(key, value)

    def get_from_cache(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Retrieve a value from the agent's cache.

        Args:
            key (str): The cache key.
            default (Any, optional): Default value if key not found.

        Returns:
            Any: The cached value or default.
        """
        return self.cache.get(key, default)

    def clear_cache(self) -> None:
        """Clear all entries from the agent's cache."""
        self.cache.clear()

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the agent's controller data.

        Returns:
            Dict[str, Any]: A dictionary containing all attributes of the controller.
        """
        data = {
            "name": self.name,
            "team": self.team,
            "speed": self.speed,
            "current_position": self.current_position,
            "start_node_id": self.start_node_id,
            "alive": self.alive,
            "death_position": self.death_position,
            "cache": self.cache.to_dict() if isinstance(self.cache, Cache) else self.cache,
            "gamms_state": self.get_state(),
        }

        # Add any custom attributes
        for key, value in self.__dict__.items():
            if key not in ["gamms_agent", "map", "cache"] and key not in data:
                data[key] = value

        return data

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

    def reset_to_start(self) -> None:
        """Reset the agent's position to its starting position."""
        self.current_position = self.start_node_id
        self.update_position(self.start_node_id)

    def die(self) -> None:
        """
        Mark the agent as dead and record death information.
        """
        self.alive = False
        self.death_position = self.current_position
        

    def is_alive(self) -> bool:
        """
        Check if the agent is alive.

        Returns:
            bool: True if agent is alive, False otherwise.
        """
        return self.alive

    def __str__(self) -> str:
        """
        Return a string representation of the agent controller.

        Returns:
            str: The string representation.
        """
        cache_info = f", cache_size={self.cache.size()}" if self.cache else ""
        alive_status = "alive" if self.alive else f"dead@{self.death_position}"
        return f"AgentController(name={self.name}, team={self.team}, pos={self.current_position}, {alive_status}{cache_info})"

    def __getattr__(self, name: str) -> Any:
        """
        Delegate any unknown attribute access to the gamms agent.
        This allows transparent access to all gamms agent methods/properties.

        Args:
            name: Attribute name to access.

        Returns:
            The attribute from the gamms agent.
        """
        return getattr(self.gamms_agent, name)
