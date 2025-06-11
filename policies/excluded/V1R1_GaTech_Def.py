import os
import pickle
from lib.strategy_utils import *

this_dir = os.path.dirname(os.path.abspath(__file__))
# Build the full path to the policy file
policy_file = os.path.join(this_dir, "policy_big_12_85.pkl")

with open(policy_file, "rb") as f:
    def1_policy, def2_policy, att_policy, superdefender_policy = pickle.load(f)

# print(superdefender_policy[1:10])

def strategy_def1(state):
    current_node = state['curr_pos']
    flag_positions = state['flag_pos']
    flag_weights = state['flag_weight']
    agent_params = state['agent_params']
    attacker_positions, defender_positions = extract_sensor_data(state, flag_positions, flag_weights, agent_params)
    node_data, edge_data = extract_map_sensor_data(state)
    no_of_nodes = len(node_data)
    neighbor_data = extract_neighbor_sensor_data(state)
    joint_state = defender_positions[0] * no_of_nodes * no_of_nodes + defender_positions[1] * no_of_nodes + attacker_positions[0]
    neighbor_data.remove(defender_positions[0])
    sorted_neighbor = sorted(neighbor_data)
    vector = def1_policy[joint_state]
    #print('defender_1 strategies', vector)
    non_zero_index = 0
    for i in range(len(vector)):
        if vector[i] > 0: non_zero_index = i
    if non_zero_index == len(vector) - 1:
        state['action'] = defender_positions[0]
    else:
        print('defender_1 strategies', sorted_neighbor[non_zero_index])
        state['action'] = sorted_neighbor[non_zero_index]
    # neighbor_data = extract_neighbor_sensor_data(state)
    # state['action'] = random.choice(neighbor_data)

def strategy_def2(state):
    current_node = state['curr_pos']
    flag_positions = state['flag_pos']
    flag_weights = state['flag_weight']
    agent_params = state['agent_params']
    attacker_positions, defender_positions = extract_sensor_data(state, flag_positions, flag_weights, agent_params)
    node_data, edge_data = extract_map_sensor_data(state)
    no_of_nodes = len(node_data)
    neighbor_data = extract_neighbor_sensor_data(state)
    joint_state = defender_positions[0] * no_of_nodes * no_of_nodes + defender_positions[1] * no_of_nodes + attacker_positions[0]
    neighbor_data.remove(defender_positions[1])
    sorted_neighbor = sorted(neighbor_data)
    vector = def2_policy[joint_state]
    #print('defender_2 strategies', vector)
    non_zero_index = 0
    for i in range(len(vector)):
        if vector[i] > 0: non_zero_index = i
    if non_zero_index == len(vector) - 1:
        state['action'] = defender_positions[1]
    else:
        state['action'] = sorted_neighbor[non_zero_index]
    #state['action'] = random.choice(neighbor_data)

def map_strategy(agent_config):
    strategies = {}
    # for name in agent_config.keys():
    #     print(name)
    strategies['defender_0'] = strategy_def1
    strategies['defender_1'] = strategy_def2
    return strategies