import networkx as nx
import queueing_tool as qt
import xlrd
import xlsxwriter
import math


# arrival rate only for the first
def arr(t):
    return rate_per_hour[math.floor(round(t, 2)) % 24]


def setup_server_nums(queue_label):
    return 0


file = xlrd.open_workbook("mad_house.xlsx")  # access to the file
wards_relations = file.sheet_by_name("Links")  # access to relationships of wards
wards = file.sheet_by_name("Nodes")  # access to nodes attributes
traffic_rates = file.sheet_by_name("Rate")

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

# importing wards Information
for row in range(wards.nrows):
    if row > 0:
        _data = wards.row_slice(row)
        nodes_mapping_list.append(_data[1].value)
        beds_per_ward.append(_data[2].value)
        avg_serving_time.append(_data[3].value)
        wards_map_index[_data[0].value] = nodes_mapping_list[int(_data[1].value)]  # Mapping Index to Nodes.
        wards_map_ward[nodes_mapping_list[int(_data[1].value)]] = _data[0].value  # Mapping Nodes to Index.

print("wards mapping index")
print(wards_map_index)
print("nodes mapping list")
print(nodes_mapping_list)

# importing wards relationships
for row in range(wards_relations.nrows):
    if row > 0:
        _data = wards_relations.row_slice(row)
        _Node1 = wards_map_index[_data[0].value]
        _Node2 = wards_map_index[_data[1].value]
        DG_labeling.add_weighted_edges_from([(int(_Node1), int(_Node2), float(_data[2].value))])  # Labels for edges
        DG_probability.add_weighted_edges_from(
            [(int(_Node1), int(_Node2), float(_data[3].value))])  # Routing probability for edges

# Importing traffic flow rate per hour
for row in range(traffic_rates.nrows):
    if row > 0:
        _data = traffic_rates.row_slice(row)
        rate_per_hour.append(_data[1].value)

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
        if i[0] not in adj_list_dict or not isinstance(adj_list_dict[i[0]], list):
            adj_list_dict[i[0]] = [int(i_splitted[j])]
        else:
            adj_list_dict[i[0]].append(int(i_splitted[j]))

adj_list_dict_int = {int(key): adj_list_dict[key] for key in adj_list_dict}  # making the keys integer

for edge in label_edge_list:
    labels = int(DG_labeling.get_edge_data(edge[0], edge[1])['weight'])  # As label
    if edge[0] not in edge_label_list_dict:
        edge_label_list_dict[edge[0]] = {edge[1]: labels}
    else:
        edge_label_list_dict[edge[0]][edge[1]] = labels

# Defining routing probability
for edge in prob_edge_list:
    probabilities = DG_probability.get_edge_data(edge[0], edge[1])['weight']  # As probability
    if edge[0] not in edge_transition_list_dict:
        edge_transition_list_dict[edge[0]] = {edge[1]: probabilities}
    else:
        edge_transition_list_dict[edge[0]][edge[1]] = probabilities

print(wards_map_index)
print(edge_label_list_dict)

# preparing the graph for simulation

g = qt.adjacency2graph(adj_list_dict_int, edge_type=edge_label_list_dict)
dg = qt.QueueNetworkDiGraph(g)

# Setting up Queue Server for each edge.
q_classes = {label: qt.QueueServer for key in edge_label_list_dict.keys() for value, label in
             edge_label_list_dict[key].items()}

q_classes[0] = qt.NullQueue  # Queue 0 indicates the link which terminates patients

# defining number of servers, arrival rate and the service time for each edge.
q_args = {label: {
    'num_servers': int(beds_per_ward[int(value)]),  # number of beds
    'collect_data': True,
    'service_f': lambda t: t + float(avg_serving_time[int(value)])  # Average Serving Time
} for key in edge_label_list_dict.keys() for value, label in edge_label_list_dict[key].items()}

q_args[1]['arrival_f'] = lambda t: t + arr(t)  # Queue 1 indicates the link which generates patients

print(q_args)
print(q_classes)

net = qt.QueueNetwork(g=dg, q_classes=q_classes, q_args=q_args, max_agents=50000)
net.start_collecting_data()
net.set_transitions(edge_transition_list_dict)  # set transition dictionary for routing probability
net.transitions(False)
net.initialize(edge_type=1)
net.simulate(t=8760)  # t 365*24=8760, and n=

# net.show_type(edge_type=4)

row = 0
workbook = xlsxwriter.Workbook('data.xlsx')
queue_sheet = workbook.add_worksheet('queue_info')
agent_sheet = workbook.add_worksheet('agent_info')

queue_sheet.write(row, 0, 'The arrival time of an agent')
queue_sheet.write(row, 1, 'The service start time of an agent')
queue_sheet.write(row, 2, 'The departure time of an agent')
queue_sheet.write(row, 3, 'The length of the queue upon the agents arrival')
queue_sheet.write(row, 4, 'The total number of Agents in the QueueServer')
queue_sheet.write(row, 5, 'The QueueServer edge index')
queue_sheet.write(row, 6, 'Edge label')
queue_sheet.write(row, 7, 'connecting nodes')
queue_sheet.write(row, 8, 'Edge distribution')
queue_sheet.write(row, 9, 'Overflow?')
queue_sheet.write(row, 10, 'Occupancy Percentage')

agent_sheet.write(row, 0, 'The arrival time of an agent')
agent_sheet.write(row, 1, 'The service start time of an agent')
agent_sheet.write(row, 2, 'The departure time of an agent')
agent_sheet.write(row, 3, 'The length of the queue upon the agents arrival')
agent_sheet.write(row, 4, 'The total number of Agents in the QueueServer')
agent_sheet.write(row, 5, 'The QueueServer edge index')
agent_sheet.write(row, 6, 'edge label')

row += 1

for source in DG_labeling.nodes():
    for target in DG_labeling.nodes():
        if source != target:
            if DG_labeling.has_edge(source, target):
                queue_data = net.get_queue_data(edge=(source, target))
                agent_data = net.get_agent_data(edge=(source, target))
                for item1, item2 in zip(queue_data, agent_data):
                    queue_sheet.write_row(row, 0, item1)  # Data such as number of agents in server and agents in queue
                    queue_sheet.write(row, 6, DG_labeling[source][target]['weight'])  # edge label
                    queue_sheet.write(row, 7,
                                      wards_map_ward[source] + " -> " + wards_map_ward[target])  # connecting wards
                    queue_sheet.write(row, 8,
                                      DG_probability[source][target]['weight'])  # edge flow distribution probability
                    queue_sheet.write(row, 8, "under construction")  # Whether there is a overflow or not and by how much
                    queue_sheet.write(row, 8, "under construction")  # Occupancy Percentage
                    agent_sheet.write_row(row, 0, item2)  # Agent general information
                    row += 1
workbook.close()

# pos = nx.spring_layout(dg.to_directed())
# net.draw(pos=pos)

# anim = net.animate()


# if the ward is full keep the patient in the previous ward, or keep them in ED. or in  corridor.
