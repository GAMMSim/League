from typing import Dict, List, Any, Optional
from typeguard import typechecked
import copy

from lib.agent.agent_core import AgentController
from lib.agent.agent_map import AgentMap
from lib.core.cache import Cache
from lib.core.console import *


@typechecked
class AgentEngine:
    """Centralized agent management system"""

    def __init__(self, ctx: Any, teams: List[str], default_config: Optional[Dict[str, Any]] = None, 
             vis_engine: Optional[Any] = None, sensor_engine: Optional[Any] = None):
        """
        Initialize AgentEngine with support for multiple teams

        Args:
            ctx: Game context object
            teams: List of team names (e.g., ["red", "blue", "green"])
            default_config: Default configuration values for agent parameters
            vis_engine: Optional visualization engine for label cleanup
        """
        debug(f"Initializing AgentEngine with teams: {teams}")
        self.ctx = ctx
        self.teams = teams
        self.vis_engine = vis_engine
        self.agents: Dict[str, AgentController] = {}
        self.sensor_engine = sensor_engine

        # Dynamic team counters - one counter per team
        self.team_counts: Dict[str, int] = {team: 0 for team in teams}

        # Team-level caches - one cache per team
        self.team_caches: Dict[str, Cache] = {team: Cache() for team in teams}

        # Team-level agent maps - one shared map per team
        self.team_maps: Dict[str, AgentMap] = {team: AgentMap() for team in teams}

        # Agent organization - for convenient access
        self.all_agents: List[AgentController] = []
        self.agents_by_team: Dict[str, List[AgentController]] = {team: [] for team in teams}
        self.active_agents: List[AgentController] = []
        self.active_agents_by_team: Dict[str, List[AgentController]] = {team: [] for team in teams}

        # Create dynamic AgentType enum based on teams
        self.agent_types = {team.upper(): team for team in teams}

        # Default configuration values
        self.default_config = default_config or {
            "speed": 1,
            "capture_radius": 0,
            "tagging_radius": 0,
            "sensors": [],
            "color": "black",
            "size": 10,
        }
        success("AgentEngine initialized")

    def create_agents_from_config(self, config_data: Dict[str, Any]) -> Dict[str, AgentController]:
        """
        Create all agents from configuration loaded by ConfigLoader

        Args:
            config_data: Raw config dictionary from ConfigLoader (from output.yaml)

        Returns:
            Dict mapping agent names to their AgentController objects
        """
        debug("Creating agents from configuration")
        agents_config = config_data.get("agents", {})
        created_agents = {}

        # Process each team
        for team in self.teams:
            team_config_key = f"{team}_config"
            team_global_key = f"{team}_global"

            # Get team-specific configurations
            team_individual_configs = agents_config.get(team_config_key, {})
            team_global_config = agents_config.get(team_global_key, {})

            if not team_individual_configs:
                warning(f"No individual agent configurations found for team '{team}'")
                continue

            # Create agents for this team
            team_agents = self._create_team_agents(team, team_individual_configs, team_global_config)

            created_agents.update(team_agents)

            info(f"Created {len(team_agents)} agents for team '{team}'")

        success(f"Total agents created: {len(self.agents)}")
        return created_agents

    def _create_team_agents(self, team: str, individual_configs: Dict[str, Any], global_config: Dict[str, Any]) -> Dict[str, AgentController]:
        """
        Create all agents for a specific team

        Args:
            team: Team name (e.g., "red", "blue")
            individual_configs: Individual agent configurations {agent_name: config}
            global_config: Global configuration for this team

        Returns:
            Dict mapping agent names to their AgentController objects for this team
        """
        debug(f"Creating agents for team '{team}'")
        debug(f"Global config for team '{team}': {global_config}")
        team_agents = {}

        for agent_name, individual_config in individual_configs.items():
            # Ensure individual_config is a dictionary
            if individual_config is None:
                individual_config = {}

            # Merge global config with individual config
            # Global config provides defaults, individual config overrides
            debug(f"Loaded global config for team '{team}': {global_config}")
            debug(f"Loaded individual config for agent '{agent_name}': {individual_config}")
            merged_config = self._merge_agent_config(global_config, individual_config)
            debug(f"Merged config for agent '{agent_name}': {merged_config}")

            # Create the agent (validation happens inside)
            agent = self._create_single_agent(agent_name, team, merged_config)
            if agent:
                # Add to main storage
                self.agents[agent_name] = agent
                team_agents[agent_name] = agent

                # Add to organized lists
                self.all_agents.append(agent)
                self.agents_by_team[team].append(agent)
                self.active_agents.append(agent)
                self.active_agents_by_team[team].append(agent)

                # Update counter
                self.team_counts[team] += 1

        return team_agents

    def _deep_merge_dicts(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two dicts. Values from `override` take precedence.
        Mutates and returns `base`.
        """
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                self._deep_merge_dicts(base[k], v)
            else:
                # Use a deepcopy to avoid aliasing mutable objects
                base[k] = copy.deepcopy(v)
        return base

    def _merge_agent_config(self, global_config: Dict[str, Any], individual_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge global and individual configurations
        Global config provides defaults, individual config overrides

        Args:
            global_config: Team's global configuration
            individual_config: Agent's individual configuration

        Returns:
            Merged configuration dictionary
        """
        # Start with a deep copy of global config to avoid shared references
        merged = copy.deepcopy(global_config) if isinstance(global_config, dict) else {}
        # Deep-merge individual overrides (also deep-copies values)
        merged = self._deep_merge_dicts(merged, individual_config if isinstance(individual_config, dict) else {})
        return merged

    def _create_single_agent(self, name: str, team: str, config: Dict[str, Any]) -> Optional[AgentController]:
        """
        Create a single agent from configuration
        
        Args:
            name: Agent name
            team: Team name
            config: Merged configuration for this agent
            
        Returns:
            Created AgentController or None if creation failed
        """
        try:
            debug(f"Creating agent '{name}' for team '{team}'")
            
            # Extract agent parameters
            agent_params = self._extract_agent_parameters(config)
            info(f"Extracted parameters for agent '{name}': {agent_params}")
            
            # Prepare context creation parameters
            ctx_params = {
                "team": team,
                "current_node_id": agent_params["start_node_id"],
                "start_node_id": agent_params["start_node_id"],
            }
            
            # Add visualization parameters
            color = agent_params.get("color")
            if color is not None:
                ctx_params["color"] = color
            else:
                warning(f"No color specified for agent '{name}', using default")
                ctx_params["color"] = self.default_config.get("color")
            
            size = agent_params.get("size")
            if size is not None:
                ctx_params["size"] = size
            else:
                warning(f"No size specified for agent '{name}', using default")
                ctx_params["size"] = self.default_config.get("size")
            
            # Create the underlying GAMMS agent in context
            self.ctx.agent.create_agent(name, **ctx_params)
            
            # Get the created GAMMS agent from context
            gamms_agent = self.ctx.agent.get_agent(name)
            
            # Get shared AgentMap for this team
            agent_map = self.team_maps[team]
            team_cache = self.team_caches[team]

            # Extract extra parameters
            standard_params = list(agent_params.keys())
            extra_params = {k: v for k, v in config.items() if k not in standard_params}
            extra_params['team_cache'] = team_cache  # ← Add this
            # Create the AgentController wrapper
            agent_controller = AgentController(
                gamms_agent=gamms_agent,
                speed=agent_params["speed"],
                map=agent_map,
                start_node_id=agent_params["start_node_id"],
                team=team,
                capture_radius=agent_params.get("capture_radius", 0),
                tagging_radius=agent_params.get("tagging_radius", 0),
                **extra_params
            )
            
            # ===== SENSOR INTEGRATION =====
            # Create and register sensors if sensor_engine is available
            if self.sensor_engine is not None:
                try:
                    # Get sensor configuration from agent params
                    sensor_list = agent_params.get("sensors", [])
                    sensing_radius = agent_params.get("sensing_radius", None)
                    
                    if sensor_list:
                        # Create sensors for this agent
                        sensor_mappings = self.sensor_engine.create_sensors_for_agent(
                            agent_name=name,
                            team=team,
                            sensor_list=sensor_list,
                            sensing_radius=sensing_radius
                        )
                        
                        # Register sensors to the agent
                        self.sensor_engine.register_sensors_to_agent(
                            agent=agent_controller,
                            sensor_mappings=sensor_mappings
                        )
                        
                        info(f"Registered {len(sensor_mappings)} sensors to {name}")
                        
                except Exception as e:
                    warning(f"Failed to setup sensors for {name}: {e}")
                    # Continue agent creation even if sensors fail
            # ===== END SENSOR INTEGRATION =====
            
            info(f"Created AgentController '{name}' for team '{team}' at node {agent_params['start_node_id']}")
            return agent_controller
            
        except Exception as e:
            error(f"Failed to create agent '{name}' for team '{team}': {e}")
            return None

    def _extract_agent_parameters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build final agent params using agent config as the base.
        Only fill keys missing from the base with defaults.
        """
        # Start from the agent's own config (avoid aliasing)
        params: Dict[str, Any] = copy.deepcopy(config)

        # If default_config is empty, nothing gets added — that's fine.
        for key, default_value in (self.default_config or {}).items():
            if key not in params:
                params[key] = copy.deepcopy(default_value) if isinstance(default_value, (dict, list, set)) else default_value

        # Require start_node_id (no default)
        if params.get("start_node_id") is None:
            raise ValueError("start_node_id is required but not provided")

        return params

    def get_team_cache(self, team: str) -> Optional[Cache]:
        """
        Get the cache for a specific team

        Args:
            team: Team name

        Returns:
            Cache object for the team, or None if team doesn't exist
        """
        return self.team_caches.get(team)

    def get_team_map(self, team: str) -> Optional[AgentMap]:
        """
        Get the shared AgentMap for a specific team

        Args:
            team: Team name

        Returns:
            AgentMap object for the team, or None if team doesn't exist
        """
        return self.team_maps.get(team)

    def clear_team_map(self, team: str) -> None:
        """
        Clear the shared map for a specific team

        Args:
            team: Team name
        """
        if team in self.team_maps:
            self.team_maps[team].clear_all()
            debug(f"Cleared shared map for team '{team}'")

    def clear_all_team_maps(self) -> None:
        """Clear all team maps"""
        for team, team_map in self.team_maps.items():
            team_map.clear_all()
        debug("Cleared all team maps")

    def update_team_cache(self, team: str, **kwargs: Any) -> None:
        """
        Update team cache with key-value pairs

        Args:
            team: Team name
            **kwargs: Key-value pairs to update in the team cache
        """
        if team in self.team_caches:
            self.team_caches[team].update(**kwargs)

    def update_team_cache_from_dict(self, team: str, data: Dict[str, Any]) -> None:
        """
        Update team cache from a dictionary

        Args:
            team: Team name
            data: Dictionary of data to update in team cache
        """
        if team in self.team_caches:
            self.team_caches[team].update_from_dict(data)

    def set_team_cache_value(self, team: str, key: str, value: Any) -> None:
        """
        Set a specific value in team cache

        Args:
            team: Team name
            key: Cache key
            value: Value to store
        """
        if team in self.team_caches:
            self.team_caches[team].set(key, value)

    def get_team_cache_value(self, team: str, key: str, default: Any = None) -> Any:
        """
        Get a specific value from team cache

        Args:
            team: Team name
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached value or default
        """
        if team in self.team_caches:
            return self.team_caches[team].get(key, default)
        return default

    def clear_team_cache(self, team: str) -> None:
        """
        Clear the cache for a specific team

        Args:
            team: Team name
        """
        if team in self.team_caches:
            self.team_caches[team].clear()

    def clear_all_caches(self) -> None:
        """Clear caches for all teams"""
        for cache in self.team_caches.values():
            cache.clear()

    def get_team_cache_info(self, team: str) -> Dict[str, Any]:
        """
        Get information about a team's cache

        Args:
            team: Team name

        Returns:
            Dictionary with cache information
        """
        if team not in self.team_caches:
            return {"exists": False}

        cache = self.team_caches[team]
        return {
            "exists": True,
            "size": cache.size(),
            "is_empty": cache.is_empty(),
            "keys": cache.keys(),
            "cache_str": str(cache),
        }

    def get_all_cache_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all team caches

        Returns:
            Dictionary mapping team names to their cache information
        """
        return {team: self.get_team_cache_info(team) for team in self.teams}

    def set_default_config(self, new_defaults: Dict[str, Any]) -> None:
        """
        Update the default configuration values

        Args:
            new_defaults: Dictionary of new default values to merge
        """
        self.default_config.update(new_defaults)

    def get_default_config(self) -> Dict[str, Any]:
        """Get a copy of the current default configuration"""
        return self.default_config.copy()

    def get_team_count(self, team: str) -> int:
        """Get count of active agents for a specific team"""
        return self.team_counts.get(team, 0)

    def get_all_team_counts(self) -> Dict[str, int]:
        """Get counts for all teams"""
        return self.team_counts.copy()

    def get_all_agents(self) -> List[AgentController]:
        """Get list of all agents (alive and dead)"""
        return self.all_agents.copy()

    def get_active_agents(self) -> List[AgentController]:
        """Get list of all active/alive agents"""
        return self.active_agents.copy()

    def get_agents_by_team(self, team: str) -> List[AgentController]:
        """Get all agents belonging to a specific team (alive and dead)"""
        return self.agents_by_team.get(team, []).copy()

    def get_active_agents_by_team(self, team: str) -> List[AgentController]:
        """Get active/alive agents belonging to a specific team"""
        return self.active_agents_by_team.get(team, []).copy()

    def kill_agent(self, agent_name: str) -> bool:
        """
        Kill/deactivate an agent and clean up its artifacts in the ctx (visuals, sensors, registry).

        Args:
            agent_name: Name of agent to kill

        Returns:
            True if agent was killed, False if agent not found or already dead
        """
        debug(f"Attempting to kill agent '{agent_name}'")
        agent = self.agents.get(agent_name)
        if agent is None:
            warning(f"Agent '{agent_name}' not found")
            return False

        # Already dead?
        if not agent.is_alive():
            debug(f"Agent '{agent_name}' is already dead")
            return False

        # --- Remove agent label from visualization (if vis_engine available) ---
        if self.vis_engine is not None:
            try:
                self.vis_engine.remove_agent_label(agent_name)
                debug(f"Removed label for killed agent '{agent_name}'")
            except Exception as e:
                warning(f"Failed to remove label for agent '{agent_name}': {e}")

        # --- Context cleanup (best-effort) ---
        try:
            gamms_agent = getattr(agent, "gamms_agent", None)

            # 1) Deregister sensors from the underlying agent (if any)
            if gamms_agent is not None:
                sensor_list = list(getattr(gamms_agent, "_sensor_list", []))
                for sname in sensor_list:
                    try:
                        gamms_agent.deregister_sensor(sname)
                    except Exception:
                        warning(f"Failed to deregister sensor '{sname}' from agent '{agent_name}'")
                        pass

            # 2) Remove visual artists (main + sensors + aux)
            vis = getattr(self.ctx, "visual", None)
            if vis is not None and hasattr(vis, "remove_artist"):
                # main artist
                try:
                    vis.remove_artist(agent_name)
                except Exception:
                    warning(f"Failed to remove main artist for agent '{agent_name}'")
                    pass

                # sensor artists
                if gamms_agent is not None:
                    for sname in list(getattr(gamms_agent, "_sensor_list", [])):
                        try:
                            vis.remove_artist(f"sensor_{sname}")
                        except Exception:
                            warning(f"Failed to remove sensor artist 'sensor_{sname}' for agent '{agent_name}'")
                            pass

                # auxiliary artists prefixed with agent name
                rm = getattr(vis, "_render_manager", None)
                if rm is not None:
                    for artist_id in list(getattr(rm, "_artists", {}).keys()):
                        if artist_id.startswith(f"{agent_name}_"):
                            try:
                                vis.remove_artist(artist_id)
                            except Exception:
                                warning(f"Failed to remove aux artist '{artist_id}' for agent '{agent_name}'")
                                pass

            # 3) Remove the agent from the ctx registry (engine-side store is handled below)
            ctx_agent_mgr = getattr(self.ctx, "agent", None)
            if ctx_agent_mgr is not None and hasattr(ctx_agent_mgr, "delete_agent"):
                try:
                    ctx_agent_mgr.delete_agent(agent_name)
                except Exception:
                    warning(f"Failed to delete agent '{agent_name}' from ctx registry")
                    pass

        except Exception as e:
            # Log, but continue to mark dead in our engine state
            error(f"Cleanup failed for '{agent_name}': {e}")
            # Last attempt: ensure removal from ctx agent registry
            try:
                if hasattr(self.ctx, "agent") and hasattr(self.ctx.agent, "delete_agent"):
                    self.ctx.agent.delete_agent(agent_name)
            except Exception:
                warning(f"Failed to delete agent '{agent_name}' from ctx registry")
                pass

        # --- Engine-side state updates ---
        try:
            agent.die()  # mark controller dead
        except Exception:
            warning(f"Failed to mark agent '{agent_name}' as dead in AgentController")
            pass

        # Remove from active lists
        if agent in self.active_agents:
            self.active_agents.remove(agent)

        team_list = self.active_agents_by_team.get(agent.team)
        if team_list and agent in team_list:
            team_list.remove(agent)

        # Decrement team counter (clamp at 0)
        self.team_counts[agent.team] = max(0, self.team_counts.get(agent.team, 0) - 1)

        info(f"Agent '{agent_name}' killed and cleaned up (team '{agent.team}').")
        return True

    def move_agent(self, agent_name: str, target_node: int) -> bool:
        """Simple move without validation - validation should happen externally."""
        debug(f"Moving agent '{agent_name}' to node {target_node}")
        agent = self.agents.get(agent_name)
        if agent is None or not agent.is_alive():
            warning(f"Cannot move agent '{agent_name}' - agent not found or dead")
            return False

        agent.update_position(target_node)
        return True

    def get_agent_status(self, agent_name: str) -> Dict[str, Any]:
        """
        Get status information for a specific agent

        Args:
            agent_name: Name of agent

        Returns:
            Dictionary with agent status information
        """
        if agent_name not in self.agents:
            return {"exists": False}

        agent = self.agents[agent_name]
        return {
            "exists": True,
            "name": agent.name,
            "team": agent.team,
            "alive": agent.is_alive(),
            "current_position": agent.current_position,
            "start_position": agent.start_node_id,
            "death_position": agent.death_position if not agent.is_alive() else None,
        }

    def assign_strategies(self, strategy_assignment: Dict[str, Any]) -> None:
        """
        Assign strategies to agents using a flexible assignment dictionary

        Args:
            strategy_assignment: Dictionary specifying how to assign strategies
                                Can use different assignment methods:

                                Method 1 - Direct agent mapping:
                                {"red_0": strategy_func, "blue_1": strategy_func, ...}

                                Method 2 - Team-based mapping:
                                {"red": strategy_module, "blue": strategy_module, ...}

                                Method 3 - Mixed mapping:
                                {"red": strategy_module, "blue_0": specific_strategy, ...}
        """
        try:
            debug("Assigning strategies to agents")
            assigned_count = 0

            for key, strategy_or_module in strategy_assignment.items():
                # Check if key is a specific agent name
                if key in self.agents:
                    agent = self.agents[key]
                    if agent.is_alive():
                        self._assign_strategy_to_agent(agent, strategy_or_module)
                        assigned_count += 1

                # Check if key is a team name
                elif key in self.teams:
                    team_agents = self.get_active_agents_by_team(key)

                    # If strategy_or_module has map_strategy method, use it
                    if hasattr(strategy_or_module, "map_strategy"):
                        team_configs = {agent.name: {"team": agent.team, **agent.to_dict()} for agent in team_agents}

                        if team_configs:
                            strategies = strategy_or_module.map_strategy(team_configs)
                            for agent_name, strategy in strategies.items():
                                if agent_name in self.agents:
                                    self._assign_strategy_to_agent(self.agents[agent_name], strategy)
                                    assigned_count += 1

                    # Otherwise, assign the same strategy/module to all team members
                    else:
                        for agent in team_agents:
                            self._assign_strategy_to_agent(agent, strategy_or_module)
                            assigned_count += 1

                else:
                    warning(f"Assignment key '{key}' matches neither agent name nor team name")

            success(f"Strategies assigned to {assigned_count} agents")

        except Exception as e:
            error(f"Error assigning strategies: {e}")
            raise

    def _assign_strategy_to_agent(self, agent: AgentController, strategy: Any) -> None:
        """
        Helper method to assign a strategy to a specific agent

        Args:
            agent: AgentController to assign strategy to
            strategy: Strategy function or object to assign
        """
        debug(f"Assigning strategy to agent '{agent.name}'")
        # Use the property setter which handles gamms agent assignment
        agent.strategy = strategy

    def validate_agent_exists(self, agent_name: str) -> bool:
        """
        Validate that an agent exists in both the context and engine storage.

        Args:
            agent_name: Name of the agent to validate

        Returns:
            True if agent exists in both systems, False otherwise
        """
        # Check if agent exists in engine storage
        engine_agent = self.agents.get(agent_name)

        # Check if agent exists in context
        ctx_agent = None
        try:
            if hasattr(self.ctx, "agent") and hasattr(self.ctx.agent, "get_agent"):
                ctx_agent = self.ctx.agent.get_agent(agent_name)
        except Exception as e:
            warning(f"Error accessing context agent '{agent_name}': {e}")
            ctx_agent = None

        # Validate synchronization
        if ctx_agent is None and engine_agent is not None:
            error(f"Agent '{agent_name}' exists in engine but not in context - synchronization issue")
            return False
        elif ctx_agent is not None and engine_agent is None:
            error(f"Agent '{agent_name}' exists in context but not in engine - synchronization issue")
            return False
        elif ctx_agent is None and engine_agent is None:
            # This might be expected behavior when checking non-existent agents
            debug(f"Agent '{agent_name}' does not exist in either system")
            return False

        return True

    def get_validated_agent(self, agent_name: str) -> Optional[AgentController]:
        """
        Get an agent only if it exists in both context and engine systems.

        Args:
            agent_name: Name of the agent to retrieve

        Returns:
            AgentController if validation passes, None otherwise
        """
        # Check engine storage first
        engine_agent = self.agents.get(agent_name)
        if engine_agent is None:
            debug(f"Agent '{agent_name}' not found in engine storage")
            return None

        # Check if agent exists in context
        try:
            if hasattr(self.ctx, "agent") and hasattr(self.ctx.agent, "get_agent"):
                ctx_agent = self.ctx.agent.get_agent(agent_name)
                if ctx_agent is None:
                    error(f"Agent '{agent_name}' exists in engine but not in context - synchronization issue")
                    return None
            else:
                warning(f"Context does not support agent lookup for validation")
                return None
        except Exception as e:
            warning(f"Error accessing context agent '{agent_name}': {e}")
            return None

        return engine_agent

    def validate_all_agents(self) -> Dict[str, bool]:
        """
        Validate all agents in the engine for synchronization issues.

        Returns:
            Dictionary mapping agent names to their validation status
        """
        debug("Validating all agents for synchronization")
        validation_results = {}

        # Check all agents in engine storage
        for agent_name in self.agents.keys():
            validation_results[agent_name] = self.validate_agent_exists(agent_name)

        # Check for orphaned agents in context (exist in context but not engine)
        try:
            if hasattr(self.ctx, "agent"):
                for ctx_agent in self.ctx.agent.create_iter():
                    agent_name = getattr(ctx_agent, "name", None)
                    if agent_name and agent_name not in self.agents:
                        validation_results[agent_name] = False
                        error(f"Orphaned agent '{agent_name}' found in context but not in engine")
        except Exception as e:
            warning(f"Could not iterate context agents for validation: {e}")

        valid_count = sum(1 for valid in validation_results.values() if valid)
        info(f"Agent validation complete: {valid_count}/{len(validation_results)} agents valid")
        return validation_results


# Usage example:
if __name__ == "__main__":
    from lib.config.config_loader import ConfigLoader

    # Load configuration
    loader = ConfigLoader("output.yaml")
    loader.load_extra_definitions("config/visualization_config.yml", force=False)
    loader.load_extra_definitions("config/rules/v2.yml", force=True)
    config_data = loader.config_data

    # Initialize agent engine with teams and custom defaults
    teams = ["red", "blue"]
    # custom_defaults = {
    #     "speed": 2,  # Faster default speed
    #     "capture_radius": 1,  # Default capture radius
    #     "tagging_radius": 1,  # Default tagging radius
    #     "sensors": ["map", "agent"],  # Default sensors
    #     "size": 15,  # Larger default size
    # }

    agent_engine = AgentEngine(ctx=None, teams=teams, default_config=None)

    # Create agents from config
    created_agents = agent_engine.create_agents_from_config(config_data)

    # Check results
    print(f"Created agents: {list(agent_engine.agents.keys())}")
    print(f"Team counts: {agent_engine.get_all_team_counts()}")
    print(f"Current defaults: {agent_engine.get_default_config()}")
    print(f"Red team agents: {[a.name for a in agent_engine.get_agents_by_team('red')]}")
    print(f"Blue team agents: {[a.name for a in agent_engine.get_agents_by_team('blue')]}")

    # Access and manipulate team caches
    red_cache = agent_engine.get_team_cache("red")
    blue_cache = agent_engine.get_team_cache("blue")

    # Set team-level strategy data
    agent_engine.set_team_cache_value("red", "strategy", "aggressive")
    agent_engine.set_team_cache_value("red", "formation", "spread")
    agent_engine.update_team_cache("blue", strategy="defensive", target_priority=["flag_1", "flag_2"])

    # Update from dictionary
    red_strategy_data = {"patrol_routes": ["route_1", "route_2"], "communication_frequency": 5, "fallback_position": 157}
    agent_engine.update_team_cache_from_dict("red", red_strategy_data)

    # Get cached values
    red_strategy = agent_engine.get_team_cache_value("red", "strategy")  # "aggressive"
    blue_formation = agent_engine.get_team_cache_value("blue", "formation", "default")  # "default"

    # Check cache info
    red_cache_info = agent_engine.get_team_cache_info("red")
    all_cache_info = agent_engine.get_all_cache_info()

    print(f"Red cache info: {red_cache_info}")
    print(f"All cache info: {all_cache_info}")

    # All red team agents share the same cache
    print(f"Red team cache: {red_cache}")
    print(f"Blue team cache: {blue_cache}")

    # Agents can access their team cache directly
    red_agents = agent_engine.get_agents_by_team("red")
    if red_agents:
        first_red_agent = red_agents[0]
        # Agent accesses team cache - all red agents see the same data
        strategy = first_red_agent.cache.get("strategy")  # "aggressive"
        formation = first_red_agent.cache.get("formation")  # "spread"

    # Clear specific team cache
    agent_engine.clear_team_cache("red")
