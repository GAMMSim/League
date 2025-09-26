from enum import Flag, auto
from copy import deepcopy
from typing import Dict, Any, Optional, Union
from typeguard import typechecked

try:
    from ..core.console import *
except ImportError:
    from lib.core.console import *


class ConfigFlags(Flag):
    """Various flags to control entry behavior."""

    NONE = 0


@typechecked
class ConfigEntry:
    def __init__(self, name: str, value: Dict[str, Any], category: str, key: Optional[str] = None, flags: Optional[ConfigFlags] = None, **kwargs) -> None:
        """
        Initialize a ConfigEntry with a name, value, category, and optional flags.

        Args:
            name: Identifier/description for this config entry
            value: The actual configuration data as a dictionary
            category: Category/group this entry belongs to (e.g., "agents", "game", "environment")
            key: Optional sub-key for nested placement under category (e.g., "red_config", "blue_config")
            flags: Optional configuration flags for controlling behavior
            **kwargs: Additional metadata (description, validator, etc.)
        """
        self.name = name
        self.value = value
        self.category = category
        self.key = key  # Now a formal attribute
        self.flags = flags if flags is not None else ConfigFlags.NONE
        self.metadata = kwargs

        # Cache for computed properties
        self._original_value = deepcopy(value) if value is not None else None
        success(f"ConfigEntry '{self.name}' initialized in category '{self.category}' with flags {self.flags.name}.")

    def edit_name(self, new_name: str) -> None:
        """Change the name of this entry."""
        self.name = new_name

    def add_value(self, additions: Dict[str, Any], deep_merge: bool = False) -> None:
        """
        Merge new values into existing value dictionary.

        Args:
            additions: Dictionary of values to add/merge
            deep_merge: If True, recursively merge nested dicts; if False, shallow merge
        """
        if self.value is None:
            self.value = {}

        if deep_merge:
            self.value = self._deep_merge(self.value, additions)
        else:
            self.value.update(additions)

    def replace_value(self, new_value: Dict[str, Any]) -> None:
        """Replace the entire value with a new dictionary."""
        self.value = new_value

    def remove_value(self, keys: Union[str, list]) -> None:
        """
        Remove specified keys from the value dictionary.

        Args:
            keys: Single key string or list of keys to remove
        """
        if self.value is None:
            return

        if isinstance(keys, str):
            keys = [keys]

        for key in keys:
            self.value.pop(key, None)

    def edit_flag(self, add: Optional[ConfigFlags] = None, remove: Optional[ConfigFlags] = None) -> None:
        """
        Modify flags by adding or removing specific flags.

        Args:
            add: Flags to add
            remove: Flags to remove
        """
        if add:
            self.flags |= add
        if remove:
            self.flags &= ~remove

    def change_metadata(self, **kwargs) -> None:
        """
        Update metadata with new key-value pairs.

        Args:
            **kwargs: Key-value pairs to update in metadata
        """
        self.metadata.update(kwargs)

    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge dict2 into dict1."""
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def visualize(self, indent: int = 0, show_flags: bool = True, show_metadata: bool = False) -> str:
        """
        Create a visual representation of this config entry.

        Args:
            indent: Number of spaces for indentation
            show_flags: Include flags in output
            show_metadata: Include metadata in output
        Returns:
            Formatted string representation
        """
        lines = []
        prefix = " " * indent

        # Header
        header = f"{prefix}[{self.category}] {self.name}"
        if self.key:  # Add this
            header += f" -> {self.key}"
        if show_flags and self.flags != ConfigFlags.NONE:
            flag_names = [f.name for f in ConfigFlags if f != ConfigFlags.NONE and self.flags & f]
            header += f" <{', '.join(flag_names)}>"
        lines.append(header)

        # Value
        if self.value:
            lines.append(f"{prefix}  Value:")
            for k, v in self.value.items():
                if isinstance(v, dict):
                    lines.append(f"{prefix}    {k}: {{{len(v)} items}}")
                elif isinstance(v, list):
                    lines.append(f"{prefix}    {k}: [{', '.join(map(str, v[:3]))}{'...' if len(v) > 3 else ''}]")
                else:
                    lines.append(f"{prefix}    {k}: {v}")

        # Metadata
        if show_metadata and self.metadata:
            lines.append(f"{prefix}  Metadata:")
            for k, v in self.metadata.items():
                lines.append(f"{prefix}    {k}: {v}")

        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    # Add the project root folder to sys.path
    root_folder = add_root_folder_to_sys_path()

    # Create entry
    entry = ConfigEntry(name="alpha_global", value={"speed": 1, "sensors": ["map"]}, category="agents")

    entry.visualize(show_flags=True, show_metadata=True)

    # 1. Edit name
    entry.edit_name("alpha_config")

    print(entry.visualize(indent=2, show_flags=True, show_metadata=True))

    # 2. Add values (merge)
    entry.add_value({"capture_radius": 2, "sensors": ["map", "agent"]})
    print(entry.visualize(indent=2, show_flags=True, show_metadata=True))

    # 3. Replace value
    entry.replace_value({"speed": 2, "new_config": True})
    print(entry.visualize(indent=2, show_flags=True, show_metadata=True))

    # 4. Remove value
    entry.remove_value("new_config")
    entry.remove_value(["unused_key1", "unused_key2"])
    print(entry.visualize(indent=2, show_flags=True, show_metadata=True))

    # 5. Change metadata
    entry.change_metadata(description="Alpha agent configuration", version="2.0")
