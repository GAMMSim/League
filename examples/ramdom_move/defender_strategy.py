import random
from gamms.utilities import *

def strategy(state, FLAG_POSITIONS, FLAG_WEIGHTS, agent):
    current_node = state['curr_pos']
    attacker_positions, defender_positions = extract_sensor_data(state, FLAG_POSITIONS, FLAG_WEIGHTS, agent)
    
    
    neighbor_data = extract_neighbor_sensor_data(state)
    state['action'] = random.choice(neighbor_data)
    

def map_strategy(agent_config):
    strategies = {}
    for name in agent_config.keys():
        strategies[name] = strategy
    return strategies