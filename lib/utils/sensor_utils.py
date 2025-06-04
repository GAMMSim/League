from typeguard import typechecked
import gamms

try:
    from lib.core.core import *
except ImportError:
    from ..core.core import *


@typechecked
def create_static_sensors() -> dict:
    """
    Create static sensor definitions that can be reused across contexts.
    """
    # Include additional configuration parameters for each sensor type
    return {"map": {"type": gamms.sensor.SensorType.MAP}, "agent": {"type": gamms.sensor.SensorType.AGENT, "sensor_range": float("inf")}, "neighbor": {"type": gamms.sensor.SensorType.NEIGHBOR}}  # Increased range


def extract_map_sensor_data(state):
    sensor_data = state.get("sensor", {})
    from rich import print

    for key, value in sensor_data.items():
        if key == "map" or (isinstance(key, str) and key.startswith("map")):
            _, map_data = value
            break
    else:
        raise ValueError("No map sensor data found in state.")

    nodes_data = map_data["nodes"]
    edges_data = map_data["edges"]
    edges_data = {edge.id: edge for edge in edges_data}

    return nodes_data, edges_data


def extract_neighbor_sensor_data(state):
    """
    Extract neighbor sensor data from agent state.

    Args:
        state (dict): The agent state dictionary.

    Returns:
        list: List of neighboring node IDs.

    Raises:
        ValueError: If sensor data is missing or in unexpected format.
    """
    sensor_data = state.get("sensor", {})

    # Try to find neighbor sensor with exact match or prefix
    neighbor_sensor = None
    for key, value in sensor_data.items():
        if key == "neighbor" or (isinstance(key, str) and key.startswith("neigh")):
            neighbor_sensor = value
            break

    if neighbor_sensor is None:
        raise ValueError("No neighbor sensor data found in state.")

    # Unpack the tuple (sensor_type, data)
    sensor_type, neighbor_data = neighbor_sensor
    return neighbor_data


def extract_agent_sensor_data(state):
    """
    Extract agent sensor data from agent state.

    Args:
        state (dict): The agent state dictionary.

    Returns:
        dict: Dictionary mapping agent names to their current positions.

    Raises:
        ValueError: If sensor data is missing or in unexpected format.
    """
    sensor_data = state.get("sensor", {})

    # Try to find agent sensor with exact match or prefix
    agent_sensor = None
    for key, value in sensor_data.items():
        if key == "agent" or (isinstance(key, str) and key.startswith("agent")):
            agent_sensor = value
            break

    if agent_sensor is None:
        raise ValueError("No agent sensor data found in state.")

    # Unpack the tuple (sensor_type, data)
    sensor_type, agent_info = agent_sensor
    return agent_info


def extract_sensor_data(state, flag_pos, flag_weight, agent_params):
    """
    Extract and process all sensor data from agent state.

    Args:
        state (dict): The agent state dictionary.
        flag_pos (list): List of flag position node IDs.
        flag_weight (dict): Dictionary mapping flag IDs to their weights.
        agent_params (object): Object with map and other agent parameters.

    Returns:
        tuple: (attacker_positions, defender_positions) containing team positions.
    """
    try:
        nodes_data, edges_data = extract_map_sensor_data(state)
        agent_info = extract_agent_sensor_data(state)
        agent_info = state.get("agent_info", agent_info)
        # print(f"Agent info: {agent_info}")

        # Add the current agent to agent_info if not already present
        current_agent_name = state.get("name")
        current_agent_pos = state.get("curr_pos")

        if current_agent_name and current_agent_pos is not None:
            if current_agent_name not in agent_info:
                print(f"Adding current agent {current_agent_name} at position {current_agent_pos} to agent info")
                agent_info[current_agent_name] = current_agent_pos
        # print(f"Updated agent info: {agent_info}")

        agent_params.map.update_networkx_graph(nodes_data, edges_data)
        agent_params.map.set_agent_dict(agent_info)
        agent_params.map.set_flag_positions(flag_pos)
        agent_params.map.set_flag_weights(flag_weight)

        attacker_positions = agent_params.map.get_team_positions("attacker")
        defender_positions = agent_params.map.get_team_positions("defender")

        return attacker_positions, defender_positions
    except Exception as e:
        import logging

        logging.error(f"Error in extract_sensor_data: {str(e)}")
        # For debugging, log the types of data
        if "nodes_data" in locals() and "edges_data" in locals():
            logging.error(f"nodes_data type: {type(nodes_data)}, edges_data type: {type(edges_data)}")
        raise
