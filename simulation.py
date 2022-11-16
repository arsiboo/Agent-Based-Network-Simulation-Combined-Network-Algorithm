import random
import networkx as nx
import numpy as np
import queueing_tool as qt
import xlrd
import xlsxwriter
import math
from queueing_tool import InfoAgent, GreedyAgent, Agent, QueueNetwork
import os
import pandas as pd
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt


directory = 'Fitter'
ignored_files = [directory + "/.DS_Store"]
wards_dists = {}
wards_args = {}


orig_dataset = pd.read_excel("akademiska.xlsx", "serving t (length of stay)")
orig_dataset = orig_dataset.dropna(subset=['los_ward'])


for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f) and f not in ignored_files:
        dist_params = pd.read_excel(f,index_col=0)
        dist_name = dist_params.columns[0]
        dist_args = dist_params[dist_name].to_dict()
        ward_name = f.replace(directory + "/fitted_distributions_", '').replace('.xlsx', '')

        dataset = orig_dataset.loc[orig_dataset['OA_unit_SV'] == ward_name]
        dataset = dataset[['los_ward']]
        dataset.info()

        sns.set_style('white')
        sns.set_context("paper", font_scale=2)
        sns.displot(data=dataset, x="los_ward", kind="hist", bins=100, aspect=1.5)
        serving_time = dataset["los_ward"].values
        plt.show()

        wards_dists[ward_name] = getattr(scipy.stats, dist_name)
        wards_args[ward_name] = dist_args
        vals = wards_dists[ward_name].rvs(**dist_args, size=1000)
        vals_df = pd.DataFrame(vals, columns=[dist_name])
        vals_df2 = vals_df[vals_df[dist_name] < float(dataset.max())]

        sns.set_style('white')
        sns.set_context("paper", font_scale=2)
        sns.displot(data=vals_df2, x=dist_name, kind="hist", bins=100, aspect=1.5)
        plt.show()

        pass

print(wards_dists)



#class InfoAgentWithType(InfoAgent):
#    def __init__(self, agent_id=(0, 0), net_size=1, **kwargs):
#        super().__init__(agent_id + ("InfoAgent",), net_size, **kwargs)


class SuperPatient(GreedyAgent):
    patientTag = ""
    pathList = []

    def __init__(self, agent_id=(0, 0), tag="", graph=0):
        self.patientTag = tag
        self.pathList = 1
        super().__init__(agent_id + (self.patientTag,))

    def desired_destination(self, network, edge):
        current_edge_source = edge[0]
        current_edge_target = edge[1]
        current_edge_index = edge[2]
        current_edge_type = edge[3]
        go = []
        target_outgoing_edges = network.out_edges[current_edge_target]  # Returns the edge index of the outgoing links.
        for _out in target_outgoing_edges:  # Try to get the outgoing wards.
            go.append(_out)
        return random.choice(go)


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


def generate_alternative_graph(graph):
    subchains = [[]]
    node_replicas_index1 = {}
    node_replicas_index2 = {}
    current_node_out_index = {}
    node_labels_map = {}
    node_counter = 0
    alternative_graph = nx.DiGraph()

    for node, node_data in graph.nodes(data=True):
        _in_degree = graph.in_degree(node)

        if node_counter == 0:
            node_replicas_index1[str(node)] = 1
            node_replicas_index2[str(node)] = 1
            current_node_out_index[str(node)] = 0
            _in_degree += 1
        else:
            node_replicas_index1[str(node)] = 0
            node_replicas_index2[str(node)] = 0
            current_node_out_index[str(node)] = -1

        _counter = 0
        while True:
            node_to_add = (_counter, node)
            alternative_graph.add_node(node_to_add, **node_data)
            node_labels_map[node_to_add] = r'${' + str(node).replace("_", "-") + '}_' + str(_counter) + '$'

            _counter += 1
            _in_degree -= 1
            if _in_degree < 1:
                break

        node_counter += 1

    nodes = []
    all_edges = sorted(list(graph.edges), key=lambda x: x[2])

    for edge in all_edges:

        if (current_node_out_index[str(edge[0])], edge[0]) not in nodes:
            nodes.append((current_node_out_index[str(edge[0])], edge[0]))

        a = node_replicas_index1[str(edge[1])]
        if (node_replicas_index1[str(edge[1])], edge[1]) not in nodes:
            nodes.append((node_replicas_index1[str(edge[1])], edge[1]))
            node_replicas_index1[str(edge[1])] += 1
            current_node_out_index[str(edge[1])] += 1

        edge_data = graph.get_edge_data(u=edge[0], v=edge[1], key=edge[2])
        alternative_graph.add_edge(
            (current_node_out_index[str(edge[0])], edge[0]),
            (a, edge[1]), **edge_data)

    return alternative_graph


# generate ordered routing graph
def generate_ordered_routing_graph(matrix, row):
    print(matrix[row])
    return "me"

# arrival rate only for the first
def arr(t):
    return rate_per_hour[math.floor(round(t, 2)) % 24]



file = xlrd.open_workbook("akademiska.xlsx")  # access to the file
wards_relations = file.sheet_by_name("Links")  # access to relationships of wards
wards = file.sheet_by_name("Nodes")  # access to nodes attributes
traffic_rates = file.sheet_by_name("Rate")
patient_type_permissions = file.sheet_by_name("Agents")

# list of information for each node
nodes_mapping_list = []
beds_per_ward = []
avg_serving_time = []
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

# importing wards information such as node index which should start from 0, Number of beds in each ward, and their mappings
for row in range(wards.nrows):
    if row > 0:
        _data = wards.row_slice(row)
        nodes_mapping_list.append(_data[1].value)
        beds_per_ward.append(_data[2].value)
        avg_serving_time.append(_data[3].value)
        wards_map_index[_data[0].value] = nodes_mapping_list[int(_data[1].value)]  # Mapping Index to Nodes.
        wards_map_ward[nodes_mapping_list[int(_data[1].value)]] = _data[0].value  # Mapping Nodes to Index.

# importing link information such as ward relations, labels for each link which should start with 1 and ends with 0, and distribution probability
for row in range(wards_relations.nrows):
    if row > 0:
        _data = wards_relations.row_slice(row)
        _Node1 = wards_map_index[_data[0].value]
        _Node2 = wards_map_index[_data[1].value]
        DG_labeling.add_weighted_edges_from([(int(_Node1), int(_Node2), float(_data[2].value))])  # Labels for edges
        DG_probability.add_weighted_edges_from([(int(_Node1), int(_Node2), float(_data[3].value))])  # Routing probability for edges


# Importing PatientType and their ordered paths from Agent Sheet
patients = []
for row in range(patient_type_permissions.nrows):
    values = []
    for col in range(patient_type_permissions.ncols):
        values.append(patient_type_permissions.cell(row, col).value)
    patients.append(values)


generate_ordered_routing_graph(patients, 2)

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
for lab_edge,prob_edge in zip(label_edge_list,prob_edge_list):
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
    'qbuffer': 0,  # Limiting queue size so that they won't go to other wards,
    # 'shared_server_state': shared_state,
    'service_f': (lambda tt: lambda t: t + float(avg_serving_time[tt]))(int(value))  # Average Serving Time
} for key in edge_label_list_dict.keys() for value, label in edge_label_list_dict[key].items()}

q_args[1]['arrival_f'] = lambda t: t + arr(t)  # Queue 1 indicates the link which generates patients
q_args[1]['AgentFactory'] = lambda f: random.choice([SuperPatient(f, patients[i][0],1) for i in range(1,len(patients))])

print(q_classes)
print(q_args)

net = qt.QueueNetwork(g=dg, q_classes=q_classes, q_args=q_args, max_agents=50000)
net.start_collecting_data()
net.set_transitions(edge_transition_list_dict)  # sets transition dictionary for routing probability
net.transitions(False)
net.initialize(edge_type=1)
net.simulate(t=8760)  # t 365*24=8760

row = 0
workbook = xlsxwriter.Workbook('outcome.xlsx')
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
                    queue_sheet.write_row(row, 0, item1)  # Data such as number of agents in server and agents in queue
                    queue_sheet.write(row, 6, DG_labeling[source][target]['weight'])  # Edge label
                    queue_sheet.write(row, 7, wards_map_ward[source])  # Source
                    queue_sheet.write(row, 8, wards_map_ward[target])  # Target
                    queue_sheet.write(row, 9,
                                      DG_probability[source][target]['weight'])  # Edge flow distribution probability
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
