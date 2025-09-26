from typeguard import typechecked
from typing import Any, Dict, Optional, List


@typechecked
class Cache:
    """A flexible key-value cache for storing any type of data."""

    def __init__(self, initial_data: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """
        Initialize cache with optional initial data.

        Args:
            initial_data (Optional[Dict[str, Any]]): Dictionary of initial key-value pairs.
            **kwargs: Additional key-value pairs to store in cache.
        """
        self._data: Dict[str, Any] = {}

        # Load initial data if provided
        if initial_data:
            self._data.update(initial_data)

        # Add any kwargs
        if kwargs:
            self._data.update(kwargs)

    def set(self, key: str, value: Any) -> None:
        """
        Store a value in the cache.

        Args:
            key (str): The key to store the value under.
            value (Any): The value to store.
        """
        self._data[key] = value

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Retrieve a value from the cache.

        Args:
            key (str): The key to retrieve.
            default (Any, optional): Default value if key not found.

        Returns:
            Any: The stored value or default.
        """
        return self._data.get(key, default)

    def update(self, **kwargs: Any) -> None:
        """
        Update multiple values at once.

        Args:
            **kwargs: Key-value pairs to update in the cache.
        """
        self._data.update(kwargs)

    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """
        Update cache from a dictionary.

        Args:
            data (Dict[str, Any]): Dictionary of key-value pairs to add.
        """
        self._data.update(data)

    def has(self, key: str) -> bool:
        """
        Check if a key exists in the cache.

        Args:
            key (str): The key to check.

        Returns:
            bool: True if key exists, False otherwise.
        """
        return key in self._data

    def remove(self, key: str) -> Optional[Any]:
        """
        Remove and return a value from the cache.

        Args:
            key (str): The key to remove.

        Returns:
            Any: The removed value, or None if key didn't exist.
        """
        return self._data.pop(key, None)

    def clear(self) -> None:
        """Clear all data from the cache."""
        self._data.clear()

    def keys(self) -> List[str]:
        """
        Get all keys in the cache.

        Returns:
            List[str]: List of all cache keys.
        """
        return list(self._data.keys())

    def values(self) -> List[Any]:
        """
        Get all values in the cache.

        Returns:
            List[Any]: List of all cache values.
        """
        return list(self._data.values())

    def items(self) -> List[tuple]:
        """
        Get all key-value pairs in the cache.

        Returns:
            List[tuple]: List of (key, value) tuples.
        """
        return list(self._data.items())

    def size(self) -> int:
        """
        Get the number of items in the cache.

        Returns:
            int: Number of cached items.
        """
        return len(self._data)

    def is_empty(self) -> bool:
        """
        Check if the cache is empty.

        Returns:
            bool: True if cache is empty, False otherwise.
        """
        return len(self._data) == 0

    def to_dict(self) -> Dict[str, Any]:
        """
        Get a copy of the cache as a dictionary.

        Returns:
            Dict[str, Any]: Copy of the cache data.
        """
        return self._data.copy()

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator."""
        return key in self._data

    def __getitem__(self, key: str) -> Any:
        """Support dictionary-style access."""
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Support dictionary-style assignment."""
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        """Support dictionary-style deletion."""
        del self._data[key]

    def __len__(self) -> int:
        """Support len() function."""
        return len(self._data)

    def __str__(self) -> str:
        """String representation of the cache."""
        if self.is_empty():
            return "Cache(empty)"
        keys_preview = list(self._data.keys())[:3]
        more = f"... +{len(self._data)-3} more" if len(self._data) > 3 else ""
        keys_str = ", ".join(keys_preview)
        if more:
            keys_str += more
        return f"Cache({{{keys_str}}})"

    def __repr__(self) -> str:
        """Detailed representation of the cache."""
        return f"Cache({self._data})"


# Example usage
if __name__ == "__main__":
    # Different ways to initialize with data

    # Method 1: With dictionary
    cache1 = Cache({"speed": 5, "position": (10, 20)})

    # Method 2: With kwargs
    cache2 = Cache(speed=5, position=(10, 20), health=100)

    # Method 3: Mixed
    cache3 = Cache({"speed": 5}, position=(10, 20), health=100)

    # Method 4: Empty
    cache4 = Cache()

    # Method 5: From existing dict
    existing_data = {"team": "red", "level": 10}
    cache5 = Cache(existing_data, bonus_points=50)

    print(cache2)  # Cache({speed, position, health})
    print(cache5)  # Cache({team, level, bonus_points})

    # Accessing and modifying data
    print(cache1.get("speed"))  # 5
    cache1.set("speed", 10)
    print(cache1["speed"])  # 10
    print("position" in cache1)  # True
    print(len(cache1))  # 2
    cache1.remove("position")
    print(cache1)  # Cache({speed})
    cache1.clear()
    print(cache1.is_empty())  # True