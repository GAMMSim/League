import random
from utilities import *

def strategy(state):
    current_node = state['curr_pos']
    flag_positions = state['flag_pos']
    flag_weights = state['flag_weight']
    agent_params = state['agent_params']
    attacker_positions, defender_positions = extract_sensor_data(state, flag_positions, flag_weights, agent_params)
    
    neighbor_data = extract_neighbor_sensor_data(state)
    state['action'] = random.choice(neighbor_data)
    

def map_strategy(agent_config):
    strategies = {}
    for name in agent_config.keys():
        strategies[name] = strategy
    return strategies