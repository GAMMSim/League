import os
from lib.strategy_utils import *
import pickle

this_dir = os.path.dirname(os.path.abspath(__file__))
# Build the full path to the policy file
policy_file = os.path.join(this_dir, "policy_big_12_85.pkl")

with open(policy_file, "rb") as f:
    def1_policy, def2_policy, att_policy, superdefender_policy = pickle.load(f)


def strategy(state):
    current_node = state['curr_pos']
    flag_positions = state['flag_pos']
    flag_weights = state['flag_weight']
    agent_params = state['agent_params']
    attacker_positions, defender_positions = extract_sensor_data(state, flag_positions, flag_weights, agent_params)
    node_data, edge_data = extract_map_sensor_data(state)
    no_of_nodes = len(node_data)
    #print("Number of nodes:", no_of_nodes)
    #print(attacker_positions)
    #print(defender_positions)
    neighbor_data = extract_neighbor_sensor_data(state)
    print(defender_positions[0])
    print(defender_positions[1])
    print(no_of_nodes)
    print(attacker_positions[0])
    joint_state = defender_positions[0]*no_of_nodes*no_of_nodes + defender_positions[1]*no_of_nodes+attacker_positions[0]
    print("Joint State:", joint_state)
    neighbor_data.remove(attacker_positions[0])
    sorted_neighbor = sorted(neighbor_data)
    print("sorted neighbor:", sorted_neighbor)
    vector = att_policy[joint_state]
    print(vector)
    non_zero_index = 0
    for i in range(len(vector)):
        if vector[i] > 0: non_zero_index = i
    if non_zero_index == len(vector) - 1:
        # print("Attacker 1:", attacker_positions[0])
        state['action'] = attacker_positions[0]
    else:
        # print("Attacker 1:", sorted_neighbor[non_zero_index])
        state['action'] = sorted_neighbor[non_zero_index]
    # sensor_data = state['sensor']
    # print("Sensor data:", sensor_data.keys())
    # print("Neighbor Data",neighbor_data)
    #state['action'] = random.choice(neighbor_data)
    

def map_strategy(agent_config):
    strategies = {}
    for name in agent_config.keys():
        strategies[name] = strategy
    return strategies