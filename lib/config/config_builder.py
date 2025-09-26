from typing import Dict, Any, Optional, Union, List, Set
from typeguard import typechecked
from pathlib import Path
import yaml

try:
    from ..core.console import *
    from config_entry import ConfigEntry, ConfigFlags
    from distribution import *
    from config_utils import recursive_update, generate_config_label, generate_single_component, binary_to_compact_hex
except ImportError:
    from lib.core.console import *
    from lib.config.config_entry import ConfigEntry, ConfigFlags
    from lib.config.distribution import *
    from lib.config.config_utils import recursive_update, generate_config_label, generate_single_component, binary_to_compact_hex


@typechecked
class ConfigBuilder:
    def __init__(self, output_path: Union[str, Path], rule_path: Optional[Union[str, Path]] = None):
        """
        Initialize ConfigBuilder.

        Args:
            output_path: Path where config will be written
            rule_path: Path to default rule YAML file
        """
        self.entries: List[ConfigEntry] = []
        self.categories: Set[str] = set()
        self.config_data: Dict[str, Any] = {}
        self.output_path = Path(output_path)
        self.rule_path = Path(rule_path) if rule_path else None
        self.rule_data: Dict[str, Any] = {}

        # Load rule if provided
        if self.rule_path and self.rule_path.exists():
            info(f"Loading rule from {self.rule_path}")
            self._load_rule()
        else:
            warning(f"Rule path {self.rule_path} does not exist or is not provided. Using empty rule.")
            self.rule_data = {}

        self.config_data = self.rule_data.copy() if self.rule_data else {}

    def get_rule_value(self, category: str, key: str, default=None):
        """
        Get a specific value from the loaded rule data.

        Args:
            category: Top-level category in the rule
            key: Key within the category
            default: Default value if not found

        Returns:
            The value from the rule or default if not found
        """
        return self.rule_data.get(category, {}).get(key, default)

    def get_current_value(self, category: str, key: str, default=None):
        """
        Get a specific value from the current config data (rule + entries merged so far).

        Args:
            category: Top-level category in the config
            key: Key within the category
            default: Default value if not found

        Returns:
            The current value or default if not found
        """
        return self.config_data.get(category, {}).get(key, default)

    def access_environment_data(self, key: str, default=None):
        """
        Convenience method to access environment data specifically.

        Args:
            key: Environment key to access
            default: Default value if not found

        Returns:
            Environment value or default
        """
        return self.get_current_value("environment", key, default)

    def _load_rule(self):
        """Load default rule from YAML file and populate categories."""
        with open(self.rule_path, "r") as f:
            self.rule_data = yaml.safe_load(f) or {}
        # Populate categories with top-level keys
        self.categories = set(self.rule_data.keys())

    def add_entry(self, entry: ConfigEntry) -> None:
        """
        Add a config entry to the entries list after validating its category.

        Args:
            entry (ConfigEntry): The configuration entry to add.
                                It should have a 'category' attribute or similar.
        """
        if not hasattr(entry, "category"):
            warning("Entry does not have a 'category' attribute. Entry ignored.")
            return

        if entry.category not in self.categories:
            warning(f"Category '{entry.category}' not found in available categories: {self.categories}. Entry ignored.")
            return

        self.entries.append(entry)
        debug(f"Entry added: {entry}")

    def remove_entry(self, category: str, key: str):
        """
        Remove an entry from self.entries by category and key attribute.

        Args:
            category (str): The category of the entry.
            key (str): The unique identifier of the entry (e.g., name or id).
        """
        original_len = len(self.entries)
        self.entries = [entry for entry in self.entries if not (getattr(entry, "category", None) == category and getattr(entry, "key", None) == key)]
        if len(self.entries) == original_len:
            warning(f"No entry found to remove with category '{category}' and key '{key}'.")

    def get_entries_by_category(self, category: str) -> list:
        """
        Retrieve all entries that belong to a specified category.

        Args:
            category (str): The category name.

        Returns:
            list: List of ConfigEntry objects matching the category.
        """
        return [entry for entry in self.entries if getattr(entry, "category", None) == category]

    def clear_entries(self):
        """Clear all entries stored in the builder."""
        self.entries.clear()

    def add_category(self, category: str):
        """
        Add a new category to self.categories set.

        Args:
            category (str): The category name to add.
        """
        self.categories.add(category)

    def add_categories(self, categories: list[str]):
        """
        Add multiple categories to self.categories set.

        Args:
            categories (list[str]): List of category names to add.
        """
        self.categories.update(categories)

    def remove_category(self, category: str):
        """
        Remove a category from self.categories set.

        Args:
            category (str): The category name to remove.
        """
        if category in self.categories:
            self.categories.remove(category)
        else:
            warning(f"Category '{category}' not found in existing categories: {self.categories}.")

    def build(self):
        """
        Merge all entries in self.entries into self.config_data.

        Assumes each entry has:
        - 'category' (top-level key in config)
        - 'key' (optional sub-key within category)
        - 'value' (dict or value to merge)

        Merge logic:
        - If 'key' is present, recursively merge under config_data[category][key]
        - Otherwise recursively merge/update directly under config_data[category]
        """
        for entry in self.entries:
            category = getattr(entry, "category", None)
            key = getattr(entry, "key", None)
            value = getattr(entry, "value", None)

            if category is None or value is None:
                warning(f"Entry missing required attributes (category/data): {entry}")
                continue

            if category not in self.config_data or self.config_data[category] is None:
                self.config_data[category] = {}

            if key:
                if key in self.config_data[category] and isinstance(self.config_data[category][key], dict) and isinstance(value, dict):
                    self.config_data[category][key] = self._merge_data(self.config_data[category][key], value, preserve_rule=True)
                else:
                    # Only override if rule value is null or empty list
                    if self._can_override_rule_value(self.config_data[category].get(key)):
                        self.config_data[category][key] = value
                        info(f"Setting {category}.{key} from entry: {entry.name}")
                    else:
                        info(f"Preserving rule value for {category}.{key}: {self.config_data[category].get(key)} (ignoring entry: {entry.name})")
            else:
                if isinstance(self.config_data[category], dict) and isinstance(value, dict):
                    self.config_data[category] = self._merge_data(self.config_data[category], value, preserve_rule=True)
                else:
                    # Only override if rule value is null or empty list
                    if self._can_override_rule_value(self.config_data.get(category)):
                        self.config_data[category] = value
                        info(f"Setting {category} from entry: {entry.name}")
                    else:
                        info(f"Preserving rule value for {category}: {self.config_data.get(category)} (ignoring entry: {entry.name})")

    def _can_override_rule_value(self, rule_value) -> bool:
        """
        Check if a rule value can be overridden.
        Rule values can only be overridden if they are null or empty list.

        Args:
            rule_value: The value from the rule to check

        Returns:
            bool: True if the value can be overridden, False otherwise
        """
        return rule_value is None or rule_value == []

    def _merge_data(self, default: dict, override: dict, preserve_rule: bool = False) -> dict:
        """
        Helper to recursively update 'default' dict with 'override' dict.

        Args:
            default (dict): Original dictionary to update.
            override (dict): Dictionary with overriding values.
            preserve_rule (bool): If True, preserve rule values unless they are null or empty list.

        Returns:
            dict: The updated dictionary.
        """
        if preserve_rule:
            # Use custom logic that preserves rule values
            return self._recursive_update_preserve_rule(default, override)
        else:
            # Use original logic for backwards compatibility
            debug(f"Merging data: {default} with {override}, preserve_rule={preserve_rule}")
            return recursive_update(default, override, force=True)

    def _recursive_update_preserve_rule(self, default: Dict, override: Dict) -> Dict:
        """
        Recursively updates the 'default' dictionary with the 'override' dictionary,
        but preserves rule values unless they are null or empty list.
        """
        for key, value in override.items():
            # If both default and override values are dictionaries, update recursively
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                default[key] = self._recursive_update_preserve_rule(default[key], value)
            else:
                # Only override if the rule value can be overridden
                if key not in default or self._can_override_rule_value(default[key]):
                    info(f"Setting key '{key}' to: {value}")
                    default[key] = value
                else:
                    info(f"Preserving rule value for key '{key}': {default[key]} (ignoring override: {value})")
        return default

    def generate_positions_from_components(self, graph, components: List, center_node: int, config_label: str = "config_v1") -> Dict[str, List[int]]:
        """
        Generate positions for multiple components and add generator entry.

        Args:
            graph: NetworkX graph
            components: List of (component_type, num, distribution) tuples
            center_node: Center node for distributions
            config_label: Label for configuration tracking

        Returns:
            Dictionary mapping component types to their generated positions
        """
        generator_value = generate_config_label(config_label)
        all_positions = {}

        for comp_type, num, distribution in components:
            (key, data), positions = generate_single_component(graph, comp_type, num, distribution, center_node)
            generator_value[key] = data
            all_positions[comp_type] = positions

        # Add generator entry
        generator_entry = ConfigEntry(name="generator", value=generator_value, category="generator")
        self.add_entry(generator_entry)

        return all_positions

    def add_agent_positions(self, team: str, positions: List[int]) -> None:
        """
        Add agent position configurations for a team.

        Args:
            team: Team name (e.g., "red", "blue")
            positions: List of node IDs for agent starting positions
        """
        agent_config = {f"{team}_{i}": {"start_node_id": node_id} for i, node_id in enumerate(positions)}

        entry = ConfigEntry(name=f"{team}_config", value=agent_config, category="agents", key=f"{team}_config")
        self.add_entry(entry)

    def write(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """
        Build the configuration and write it to a YAML file.

        Args:
            output_path: Path where config will be written. If None, uses self.output_path
        """
        # Build the configuration first
        self.build()

        # Determine output path
        if output_path is not None:
            file_path = Path(output_path)
        else:
            file_path = self.output_path

        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        with open(file_path, "w") as f:
            yaml.dump(self.config_data, f, sort_keys=False)

        info(f"Config saved to {file_path}")

    def load_extra_definitions(self, extra_file_path: Union[str, Path], preserve_rule: bool = True) -> None:
        """
        Load another YAML file and merge its content into the current config_data.

        Args:
            extra_file_path (Union[str, Path]): Path to the extra YAML file to load.
            preserve_rule (bool): If True, preserve rule values unless they are null or empty list.
                                If False, use original force override behavior.
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

        # Merge extra_data into config_data respecting rule preservation
        if preserve_rule:
            self.config_data = self._recursive_update_preserve_rule(self.config_data, extra_data)
        else:
            self.config_data = recursive_update(self.config_data, extra_data, force=True)

    def __str__(self) -> str:
        """
        Create a string representation of the ConfigBuilder for hashing purposes.
        Includes all entries, categories, and current config data.

        Returns:
            String representation combining all relevant state information
        """
        # Sort entries by category and name for consistent ordering
        sorted_entries = sorted(self.entries, key=lambda e: (getattr(e, "category", ""), getattr(e, "name", "")))

        # Build entry strings
        entry_strs = []
        for entry in sorted_entries:
            entry_str = f"{entry.category}:{entry.name}:{entry.value}"
            if hasattr(entry, "key") and entry.key:
                entry_str += f":{entry.key}"
            entry_strs.append(entry_str)

        # Combine all components
        components = [f"output:{self.output_path}", f"rule:{self.rule_path}", f"categories:{sorted(self.categories)}", f"entries:{';'.join(entry_strs)}", f"config_data:{self.config_data}"]

        return "|".join(components)


if __name__ == "__main__":
    import pickle
    import yaml

    # Load graph
    with open("graphs/graph_200_200_a.pkl", "rb") as f:
        graph = pickle.load(f)

    red_agent_number = 10
    blue_agent_number = 10
    candidate_flag_number = 5
    real_flag_number = 3

    config_builder = ConfigBuilder(output_path="output.yaml", rule_path="config/rules/v1.2.yml")

    # Configuration parameters
    CENTER_NODE = 1
    components = [
        ("candidate_flags", candidate_flag_number, ("uniform", 2)),
        ("red_agents", red_agent_number, ("normal", [6, 2])),
        ("blue_agents", blue_agent_number, ("normal", [4, 2])),
    ]

    # Generate all positions using the new method
    all_positions = config_builder.generate_positions_from_components(graph, components, CENTER_NODE)

    # Add flag configurations (kept mutable/flexible)
    flags_entry = ConfigEntry(name="flags", value={"candidate_positions": all_positions["candidate_flags"], "real_positions": all_positions["candidate_flags"][:real_flag_number]}, category="flags")
    config_builder.add_entry(flags_entry)

    # Set blue static sensor positions to same as candidate positions
    environment_adjustment_entry = ConfigEntry(name="environment_adjustment", value={"blue_static_sensor_positions": all_positions["candidate_flags"]}, category="environment")
    config_builder.add_entry(environment_adjustment_entry)

    # Add agent configurations
    config_builder.add_agent_positions("red", all_positions["red_agents"])
    config_builder.add_agent_positions("blue", all_positions["blue_agents"])

    # Add environment settings
    environment_entry = ConfigEntry(
        name="environment_settings",
        value={"graph_name": "graph_200_200_a.pkl"},
        category="environment",
    )
    config_builder.add_entry(environment_entry)

    # Build and save using the new write method
    config_builder.write()
