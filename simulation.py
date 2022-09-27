import networkx as nx
import queueing_tool as qt
import xlrd
import numpy as np
import random

file = xlrd.open_workbook("mad_house.xlsx")  # access to the network file
wards_relations = file.sheet_by_name("Links")  # access to relationships of wards
wards = file.sheet_by_name("Nodes")  # access to nodes attributes

# list of information for each node
nodes_mapping_list = []
wards_list = []
total_beds = []
avg_serving_time = []
patients_number = []
wards_occupancy = []
overflow = []
wards_map_index = {}

# list of information for each edge
edge_types = []
edge_map_index = {}

# Constructing Graph
MG = nx.MultiDiGraph()
DG = nx.DiGraph()
DG_adj = nx.DiGraph()
DG_probability = nx.DiGraph()

# importing wards
for row in range(wards.nrows):
    if row > 0:
        _data = wards.row_slice(row)
        wards_list.append(_data[0].value)
        nodes_mapping_list.append(_data[1].value)
        total_beds.append(_data[2].value)
        avg_serving_time.append(_data[3].value)
        patients_number.append(_data[4].value)
        wards_occupancy.append(_data[5].value)
        overflow.append(_data[6].value)
        wards_map_index[_data[0].value] = nodes_mapping_list[int(_data[1].value)]

# importing wards relationships
for row in range(wards_relations.nrows):
    if row > 0:
        _data = wards_relations.row_slice(row)
        _Node1 = wards_map_index[_data[0].value]
        _Node2 = wards_map_index[_data[1].value]
        MG.add_weighted_edges_from(
            [(int(_Node1), int(_Node2), float(_data[2].value)), (int(_Node1), int(_Node2), float(_data[3].value)),
             (int(_Node1), int(_Node2), float(_data[4].value)), (int(_Node1), int(_Node2), float(_data[5].value))])
        DG.add_weighted_edges_from([(int(_Node1), int(_Node2), float(_data[2].value))])
        DG_probability.add_weighted_edges_from([(int(_Node1), int(_Node2), float(_data[3].value))])

# Generating adjacency list and edge list, and convert it to dictionary
adj_list = nx.generate_adjlist(DG)
edge_list = list(DG.edges)
prob_edge_list = list(DG_probability.edges)
adj_list_dict = {}
edge_list_dict = {}
edge_transition_list_dict = {}

for i in adj_list:
    i_splitted = i.split(' ')
    for j in range(1, len(i_splitted)):
        if i[0] not in adj_list_dict or not isinstance(adj_list_dict[i[0]], list):
            adj_list_dict[i[0]] = [int(i_splitted[j])]
        else:
            adj_list_dict[i[0]].append(int(i_splitted[j]))

adj_list_dict_int = {int(key): adj_list_dict[key] for key in adj_list_dict}

# Labeling the queues
counter = 1
for edge in edge_list:
    if edge[0] not in edge_list_dict:
        edge_list_dict[edge[0]] = {edge[1]: counter}
    else:
        edge_list_dict[edge[0]][edge[1]] = counter
    counter += 1

# Defining routing probability
for edge in prob_edge_list:
    d = DG_probability.get_edge_data(edge[0], edge[1])['weight']

    if edge[0] not in edge_transition_list_dict:
        edge_transition_list_dict[edge[0]] = {edge[1]: d}
    else:
        edge_transition_list_dict[edge[0]][edge[1]] = d

# preparing the graph for simulation
g = qt.adjacency2graph(adj_list_dict_int, edge_type=edge_list_dict)


# flow rate (set up for each edge)
def rate(t):
    return 0.5


# arrival rate only for the first
def arr(t):
    return qt.poisson_random_measure(t, rate, 0.5)


# serving time in wards
# def ser(t,value):
#    return t + float(avg_serving_time[int(value)])


# Setting up Queue Server for each edge.
q_classes = {label: qt.QueueServer for key in edge_list_dict.keys() for value, label in edge_list_dict[key].items()}

# defining number of servers, arrival rate and the service time for each edge.
q_args = {label: {
    'num_servers': int(total_beds[int(value)]),  # number of beds
    'service_f': lambda t: t + float(avg_serving_time[int(value)])  # Average Serving Time
} for key in edge_list_dict.keys() for value, label in edge_list_dict[key].items()}

q_args[1]['arrival_f'] = arr

# Create the queue Network and initialize the simulation
net = qt.QueueNetwork(g=g, q_classes=q_classes, q_args=q_args, max_agents=10000)
net.start_collecting_data()
net.set_transitions(edge_transition_list_dict)  # set transition dictionary for routing probability
net.transitions(False)
net.initialize(edge_type=1)
net.simulate(t=1000)
#net.draw(fname="state.png")
# net.draw(fname="state.png", scatter_kwargs={'s': 40})


# Get the simulation data
# agent_data_out = net.get_agent_data()
# print(agent_data_out)

# queue_data_out = net.get_queue_data()
# print(queue_data_out)
