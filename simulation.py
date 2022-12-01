import copy
import random
import networkx as nx
import numpy as np
from numpy.random import uniform
import queueing_tool as qt
import xlrd
import xlsxwriter
import math
from queueing_tool import InfoAgent, GreedyAgent, Agent, QueueNetwork
from queueing_tool.queues.choice import _choice, _argmin
import os
import pandas as pd
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt


# print(wards_dists) #: wards_lists[ward_name].rvs(wards_args[ward_name])
# b = wards_dists["Hjärtavdelning 50 F"].rvs(**wards_args["Hjärtavdelning 50 F"])


# class InfoAgentWithType(InfoAgent):
#    def __init__(self, agent_id=(0, 0), net_size=1, **kwargs):
#        super().__init__(agent_id + ("InfoAgent",), net_size, **kwargs)


class SuperPatient(GreedyAgent):
    patientTag = ""
    pathMat = 0
    wards_mapping = 0

    def __init__(self, agent_id=(0, 0), tag="", paths=0, wards_map=0):
        self.patientTag = tag
        self.pathMat = copy.deepcopy(paths)
        self.wards_mapping = copy.deepcopy(wards_map)
        super().__init__(agent_id + (self.patientTag,))

    def desired_destination(self, network, edge):
        current_edge_source = edge[0]  # this is the source node mapping index of the current edge from data
        current_edge_target = edge[1]  # this is the damn target node mapping index of the current edge from data
        current_edge_index = edge[2]  # the simulator itself decided to have an extra index on my index.
        current_edge_type = edge[3]  # and here are the labels of the edges.

        # Returns the edge index of the outgoing links according to where it stands.
        target_outgoing_edges = network.out_edges[current_edge_target]

        new_outgoing_edges = copy.deepcopy(target_outgoing_edges)
        for out in target_outgoing_edges:
            out_edge_source = network.edge2queue[out].edge[0]
            out_edge_target = network.edge2queue[out].edge[1]
            out_edge_index = network.edge2queue[out].edge[2]
            out_edge_type = network.edge2queue[out].edge[3]
            out_node_key = [k for k, v in self.wards_mapping.items() if v == out_edge_target]
            for i in range(0, len(self.pathMat)):
                if self.pathMat[i][0] == self.patientTag:
                    for j in range(0, len(self.pathMat)):
                        if self.pathMat[0][j] == str(out_node_key).replace('[', '').replace(']', '').replace('\'','').replace('\"', ''):
                            if self.pathMat[i][j] == 0:
                                if out in new_outgoing_edges:
                                    new_outgoing_edges.remove(out)

            if "prior" in self.patientTag:
                current_target_key = [k for k, v in self.wards_mapping.items() if v == current_edge_target]
                if str(current_target_key).replace('[', '').replace(']', '').replace('\'', '').replace('\"', '') == "EMERGENCY DEPARTMENT":
                    if out_edge_type==73:
                        if not network.edge2queue[out].at_capacity():
                            new_outgoing_edges.clear()
                            new_outgoing_edges.append(out)
                            self.patientTag+="check"
                            break
                    elif out_edge_type==78:
                        if not network.edge2queue[out].at_capacity():
                            new_outgoing_edges.clear()
                            new_outgoing_edges.append(out)
                            self.patientTag += "check"
                            break
                    elif out_edge_type==74:
                        if "check" not in self.patientTag:
                            new_outgoing_edges.clear()
                            new_outgoing_edges.append(out)
                            break
                    else:
                        continue

                elif str(current_target_key).replace('[', '').replace(']', '').replace('\'','').replace('\"', '') == "Akutvårdsavdelning 30 C":
                    if out_edge_type==73:
                        if "check" not in self.patientTag:
                            if not network.edge2queue[out].at_capacity():
                                new_outgoing_edges.clear()
                                new_outgoing_edges.append(out)
                                break
                    elif out_edge_type==78:
                        if "check" not in self.patientTag:
                            if not network.edge2queue[out].at_capacity():
                                new_outgoing_edges.clear()
                                new_outgoing_edges.append(out)
                                break
                    else:
                        continue
                else:
                    continue

        if len(new_outgoing_edges) == 0:
            new_outgoing_edges = copy.deepcopy(target_outgoing_edges)

        n = len(new_outgoing_edges)
        if n <= 1:
            return new_outgoing_edges[0]

        u = uniform()
        pr = network._route_probs[current_edge_target]
        k = _choice(pr, u, n)

        return new_outgoing_edges[k]


class SharedQueueServer(qt.QueueServer):
    def __init__(self, shared_server_state: 0, **kwargs):
        super().__init__(**kwargs)
        self.shared_server_state = shared_server_state
        self.total_num_servers = kwargs.get("num_servers", 1)

    @property
    def num_servers(self):
        return self.total_num_servers - self.shared_server_state[0]

    @num_servers.setter
    def num_servers(self, value):
        self.total_num_servers = value

    def next_event(self):
        next_event_type = self.next_event_description()
        # Event type 2 is a departure
        if next_event_type == 2:
            self.shared_server_state[0] -= 1
            return super().next_event()

        # Event type 1 is an arrival
        elif next_event_type == 1:
            # We only use a server if there is capacity.
            if self.num_system - len(self.queue) < self.num_servers:
                self.shared_server_state[0] += 1
            super().next_event()


# arrival rate only for the first
def arr(t):
    return rate_per_hour[math.floor(round(t, 2)) % 24]


directory = 'Fitter'
ignored_files = [directory + "/.DS_Store"]
wards_dists = {}
wards_args = {}

orig_dataset = pd.read_excel("akademiska.xlsx", "serving t (length of stay)")
orig_dataset = orig_dataset.dropna(subset=['los_ward'])

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f) and f not in ignored_files:
        dist_params = pd.read_excel(f, index_col=0)
        dist_name = dist_params.columns[0]
        dist_args = dist_params[dist_name].to_dict()
        ward_name = f.replace(directory + "/fitted_distributions_", '').replace('.xlsx', '')

        dataset = orig_dataset.loc[orig_dataset['OA_unit_SV'] == ward_name]
        dataset = dataset[['los_ward']]
        # dataset.info()

        # sns.set_style('white')
        # sns.set_context("paper", font_scale=2)
        # sns.displot(data=dataset, x="los_ward", kind="hist", bins=100, aspect=1.5)
        # serving_time = dataset["los_ward"].values
        # plt.title(ward_name + " from data ")
        # plt.show()

        wards_dists[ward_name] = getattr(scipy.stats, dist_name)
        wards_args[ward_name] = dist_args
        vals = wards_dists[ward_name].rvs(**dist_args, size=1000)
        vals_df = pd.DataFrame(vals, columns=[dist_name])
        vals_df2 = vals_df[vals_df[dist_name] < float(dataset.max())]

        # sns.set_style('white')
        # sns.set_context("paper", font_scale=2)
        # sns.displot(data=vals_df2, x=dist_name, kind="hist", bins=100, aspect=1.5)
        # plt.title(ward_name + " from distribution ")
        # plt.show()

        pass

file = xlrd.open_workbook("akademiska.xlsx")  # access to the file
wards_relations = file.sheet_by_name("ExtraLinks")  # access to relationships of wards
wards = file.sheet_by_name("ExtraNodes")  # access to nodes attributes
traffic_rates = file.sheet_by_name("Rate")
normal_patients_type_path = file.sheet_by_name("Agents1")
acute_patients_type_path = file.sheet_by_name("Agents2")

# list of information for each node
nodes_mapping_list = []
beds_per_ward = []
avg_serving_time = []
buffer_per_ward = []
wards_map_index = {}
wards_map_ward = {}

# list of information for each edge
edge_types = []
edge_map_index = {}

# Traffic rate per hour for 24 hours
rate_per_hour = []

# Constructing Graph
DG_labeling = nx.DiGraph()
DG_probability = nx.DiGraph()

acute_patients = []
for row in range(acute_patients_type_path.nrows):
    values = []
    for col in range(acute_patients_type_path.ncols):
        values.append(acute_patients_type_path.cell(row, col).value)
    acute_patients.append(values)

# Importing PatientType and their ordered paths from Agent Sheet
normal_patients = []
for row in range(normal_patients_type_path.nrows):
    values = []
    for col in range(normal_patients_type_path.ncols):
        values.append(normal_patients_type_path.cell(row, col).value)
    normal_patients.append(values)

# importing wards information such as node index which should start from 0, Number of beds in each ward, and their mappings
for row in range(wards.nrows):
    if row > 0:
        _data = wards.row_slice(row)
        nodes_mapping_list.append(_data[1].value)
        beds_per_ward.append(_data[2].value)
        avg_serving_time.append(_data[3].value)
        buffer_per_ward.append(_data[4].value)
        wards_map_index[_data[0].value] = nodes_mapping_list[int(_data[1].value)]  # Mapping Index to Nodes.
        if str(_data[0].value) in wards_dists:
            wards_dists[_data[1].value] = wards_dists.pop(_data[0].value)
            wards_args[_data[1].value] = wards_args.pop(_data[0].value)
        else:
            wards_dists[_data[1].value] = _data[3].value
            wards_args[_data[1].value] = _data[3].value
        wards_map_ward[nodes_mapping_list[int(_data[1].value)]] = _data[0].value  # Mapping Nodes to Index.

# importing link information such as ward connections, labels for each link which should start with 1 and ends with 0, and distribution probability
for row in range(wards_relations.nrows):
    if row > 0:
        _data = wards_relations.row_slice(row)
        _Node1 = wards_map_index[_data[0].value]
        _Node2 = wards_map_index[_data[1].value]
        DG_labeling.add_weighted_edges_from([(int(_Node1), int(_Node2), float(_data[2].value))])  # Labels for edges
        DG_probability.add_weighted_edges_from(
            [(int(_Node1), int(_Node2), float(_data[3].value))])  # Routing probability for edges

# Importing traffic flow rate per hour from rate sheet
for row in range(traffic_rates.nrows):
    if row > 0:
        _data = traffic_rates.row_slice(row)
        rate_per_hour.append(_data[1].value)

# remove self-loops:
DG_labeling.remove_edges_from(nx.selfloop_edges(DG_labeling))
DG_probability.remove_edges_from(nx.selfloop_edges(DG_probability))

# Generating adjacency list, and convert it to dictionary
adj_list = nx.generate_adjlist(DG_labeling)
label_edge_list = list(DG_labeling.edges)
prob_edge_list = list(DG_probability.edges)
adj_list_dict = {}
edge_label_list_dict = {}
edge_transition_list_dict = {}

# Network Adjacency edges only used for creating the network for simulation
for i in adj_list:
    i_splitted = i.split(' ')
    for j in range(1, len(i_splitted)):
        if i_splitted[0] not in adj_list_dict or not isinstance(adj_list_dict[i_splitted[0]], list):
            adj_list_dict[i_splitted[0]] = [int(i_splitted[j])]
        else:
            adj_list_dict[i_splitted[0]].append(int(i_splitted[j]))

adj_list_dict_int = {int(key): adj_list_dict[key] for key in adj_list_dict}  # making the keys integer

# Assigning labels and defining routing probability
for lab_edge, prob_edge in zip(label_edge_list, prob_edge_list):
    labels = int(DG_labeling.get_edge_data(lab_edge[0], lab_edge[1])['weight'])  # As label
    probabilities = DG_probability.get_edge_data(prob_edge[0], prob_edge[1])['weight']  # As probability
    if lab_edge[0] not in edge_label_list_dict:
        edge_label_list_dict[lab_edge[0]] = {lab_edge[1]: labels}
    else:
        edge_label_list_dict[lab_edge[0]][lab_edge[1]] = labels
    if prob_edge[0] not in edge_transition_list_dict:
        edge_transition_list_dict[prob_edge[0]] = {prob_edge[1]: probabilities}
    else:
        edge_transition_list_dict[prob_edge[0]][prob_edge[1]] = probabilities

# preparing the graph for simulation
g = qt.adjacency2graph(adj_list_dict_int, edge_type=edge_label_list_dict)
dg = qt.QueueNetworkDiGraph(g)

# Setting up Queue Server for each edge.
q_classes = {label: qt.LossQueue for key in edge_label_list_dict.keys() for value, label in
             edge_label_list_dict[key].items()}

# q_classes = {label: SharedQueueServer for key in edge_label_list_dict.keys() for value, label in
#             edge_label_list_dict[key].items()}

q_classes[0] = qt.NullQueue  # Queue 0 indicates the link which terminates patients
q_classes[1] = qt.QueueServer  # The first server has unlimited queue and is type of QueueServer

# shared_state = [0]
# defining number of servers, arrival rate and the service time for each edge.
q_args = {label: {
    'num_servers': int(beds_per_ward[int(value)]),  # number of beds
    'collect_data': True,
    'qbuffer': int(buffer_per_ward[int(value)]),
    # 'shared_server_state': shared_state,
    'service_f': (lambda tt: lambda t: t + float(wards_dists[tt].rvs(**wards_args[tt])))(int(value))
} for key in edge_label_list_dict.keys() for value, label in edge_label_list_dict[key].items()}

q_args[1]['arrival_f'] = lambda t: t + arr(t)  # Queue 1 indicates the link which generates patients

q_args[1]['AgentFactory'] = lambda f: random.choice(
     [SuperPatient(f, acute_patients[i][0], acute_patients, wards_map_index) for i in range(1, len(acute_patients))] + [
         SuperPatient(f, normal_patients[i][0], normal_patients, wards_map_index) for i in
         range(1, len(normal_patients))])
#    weights=[0.2/len(acute_patients) for i in range(1, len(acute_patients))] +
#            [0.8/len(normal_patients) for i in range(1, len(normal_patients))]

print(q_classes)
print(q_args)

net = qt.QueueNetwork(g=dg, q_classes=q_classes, q_args=q_args, max_agents=50000)
net.start_collecting_data()
net.set_transitions(edge_transition_list_dict)  # sets transition dictionary for routing probability
net.transitions(False)
net.initialize(edge_type=1)
net.simulate(t=8760)  # t 365*24=8760

row = 0
workbook = xlsxwriter.Workbook('outcome_extra_node.xlsx')
queue_sheet = workbook.add_worksheet('queue_info')
agent_sheet = workbook.add_worksheet('agent_info')

queue_sheet.write(row, 0, 'The arrival time of an agent')
queue_sheet.write(row, 1, 'The service start time of an agent')
queue_sheet.write(row, 2, 'The departure time of an agent')
queue_sheet.write(row, 3, 'The length of the queue upon the agents arrival')
queue_sheet.write(row, 4, 'The total number of Agents in the QueueServer')
queue_sheet.write(row, 5, 'The QueueServer edge index')
queue_sheet.write(row, 6, 'Edge label')
queue_sheet.write(row, 7, 'source')
queue_sheet.write(row, 8, 'target')
queue_sheet.write(row, 9, 'Edge distribution')
queue_sheet.write(row, 10, 'Server Max Capacity')
queue_sheet.write(row, 11, 'Overflow?')
queue_sheet.write(row, 12, 'Occupancy Percentage')
queue_sheet.write(row, 13, 'Average service time')
queue_sheet.write(row, 14, 'AgentType')

agent_sheet.write(row, 0, 'The arrival time of an agent')
agent_sheet.write(row, 1, 'The service start time of an agent')
agent_sheet.write(row, 2, 'Agent Type')

row += 1

for source in DG_labeling.nodes():
    for target in DG_labeling.nodes():
        if source != target:
            if DG_labeling.has_edge(source, target):
                queue_data = net.get_queue_data(edge=(source, target))
                agent_data = net.get_agent_data(edge=(source, target))
                for item1, item2 in zip(queue_data, agent_data):
                    #      if wards_map_ward[source] != "Source" and wards_map_ward[source] != "Sink" and wards_map_ward[
                    #          target] != "Source" and wards_map_ward[target] != "Sink":
                    queue_sheet.write_row(row, 0,
                                          item1)  # Data such as number of agents in server and agents in queue
                    queue_sheet.write(row, 6, DG_labeling[source][target]['weight'])  # Edge label
                    queue_sheet.write(row, 7, wards_map_ward[source])  # Source
                    queue_sheet.write(row, 8, wards_map_ward[target])  # Target
                    queue_sheet.write(row, 9,
                                      DG_probability[source][target][
                                          'weight'])  # Edge flow distribution probability
                    queue_sheet.write(row, 10, beds_per_ward[target])  # Server max capacity
                    queue_sheet.write(row, 11, beds_per_ward[target] - item1[
                        4])  # Whether there is an overflow or not and by how much
                    if item1[4] != 0:
                        perc = 100 * (item1[4] / beds_per_ward[target])
                    else:
                        perc = float("inf")
                    queue_sheet.write(row, 12, perc if not math.isinf(perc) else "inf")  # Occupancy Percentage
                    queue_sheet.write(row, 13, item1[2] - item1[1])
                    queue_sheet.write(row, 14, item2[2])
                    agent_sheet.write_row(row, 0, item2)
                    row += 1
workbook.close()
