from typing import Dict, Any, Optional, Union, List, Set
from typeguard import typechecked
from pathlib import Path
import yaml

try:
    from ..core.console import *
    from distribution import *
    from config_utils import recursive_update
except ImportError:
    from lib.core.console import *
    from lib.config.distribution import *
    from lib.config.config_utils import recursive_update


@typechecked
class ConfigLoader:
    def __init__(self, input_path: Union[str, Path]):
        """
        Initialize ConfigLoader to read and manage YAML configuration.

        Args:
            input_path: Path to the YAML file to load
        """
        self.input_path = Path(input_path)
        self.config_data: Dict[str, Any] = {}

        # Load the main config file
        if self.input_path.exists():
            info(f"Loading config from {self.input_path}")
            self._load_config()
        else:
            error(f"Config file '{self.input_path}' does not exist.")
            raise FileNotFoundError(f"Config file '{self.input_path}' not found")

    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.input_path, "r") as f:
                self.config_data = yaml.safe_load(f) or {}
            success(f"Config loaded successfully from {self.input_path}")
        except Exception as e:
            error(f"Failed to load config file '{self.input_path}': {e}")
            raise

    def load_extra_definitions(self, extra_file_path: Union[str, Path], force: bool = True) -> None:
        """
        Load another YAML file and merge its content into the current config_data.

        Args:
            extra_file_path: Path to the extra YAML file to load
            force: If True, override existing entries; if False, only add missing/null entries
        """
        extra_file_path = Path(extra_file_path)
        debug(f"Loading extra config file from: {extra_file_path}")

        if not extra_file_path.exists():
            warning(f"Extra config file '{extra_file_path}' does not exist. Skipping load.")
            return

        try:
            with open(extra_file_path, "r") as f:
                extra_data = yaml.safe_load(f) or {}
        except Exception as e:
            warning(f"Failed to load extra config file '{extra_file_path}': {e}")
            return

        # Merge extra_data into config_data
        self.config_data = recursive_update(self.config_data, extra_data, force=force)
        info(f"Merged extra definitions from {extra_file_path}")

    def get(self, *keys, default=None):
        """
        Access nested config data using dot notation or multiple keys.

        Examples:
            loader.get('agents', 'red_config', 'red_0')
            loader.get('game', 'game_rule')

        Args:
            *keys: Sequence of keys to traverse
            default: Default value if key path doesn't exist

        Returns:
            Value at the specified path or default
        """
        current = self.config_data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    def get_category(self, category: str) -> Dict[str, Any]:
        """
        Get all data under a specific category.

        Args:
            category: Top-level category name

        Returns:
            Dictionary of data under that category, or empty dict if not found
        """
        return self.config_data.get(category, {})

    def has_category(self, category: str) -> bool:
        """Check if a category exists in the config."""
        return category in self.config_data

    def get_categories(self) -> List[str]:
        """Get list of all top-level categories."""
        return list(self.config_data.keys())

    def reload(self):
        """Reload the configuration from the original file."""
        info(f"Reloading config from {self.input_path}")
        self._load_config()

    def __str__(self) -> str:
        """
        Create a human-readable string representation.

        Returns:
            Formatted string showing config structure
        """
        lines = [f"ConfigLoader({self.input_path.name})"]

        for category in sorted(self.config_data.keys()):
            value = self.config_data[category]

            if value is None:
                lines.append(f"  {category}: None")
            elif isinstance(value, dict):
                # Show first few keys for dicts
                keys = list(value.keys())[:3]
                more = f"... +{len(value)-3} more" if len(value) > 3 else ""
                keys_str = ", ".join(keys) + more
                lines.append(f"  {category}: {{{keys_str}}}")
            elif isinstance(value, list):
                lines.append(f"  {category}: [{len(value)} items]")
            elif isinstance(value, str) and len(value) > 50:
                lines.append(f"  {category}: '{value[:47]}...'")
            else:
                lines.append(f"  {category}: {value}")

        return "\n".join(lines)

    def __getitem__(self, key):
        """Allow dictionary-style access to top-level categories."""
        return self.config_data[key]

    def __contains__(self, key):
        """Check if a top-level key exists."""
        return key in self.config_data


# Example usage:
if __name__ == "__main__":
    # Load config
    loader = ConfigLoader("output.yaml")

    # Access data
    game_rule = loader.get("game", "game_rule")
    red_agent_0 = loader.get("agents", "red_config", "red_0")

    # Get entire category
    agents_config = loader.get_category("agents")

    # Dictionary-style access
    flags = loader["flags"]

    # Print the loaded config
    print(loader)
