# Resodial flows and structural hole
# Residual_Capacity = Original_Capacity - current_flow
# Residual_Capacity = (Total_number_of_beds/incoming_edge_transition) - current_people_in_queue
import math
import random
import networkx as nx
import xlrd
from networkx.algorithms.flow import edmonds_karp, shortest_augmenting_path, preflow_push, dinitz, boykov_kolmogorov, \
    build_residual_network

file = xlrd.open_workbook("mad_house.xlsx")
data = xlrd.open_workbook("data.xlsx")

hospital = file.sheet_by_name("Links")
wards = file.sheet_by_name("Nodes")
output = data.sheet_by_name("queue_info")

G_flow = nx.DiGraph()
RG = nx.DiGraph()

times = []
node_max_capacity = []
edge_current_patients = []
edge_transition = []
edge_label_output = []

output_size = len(times)

# Importing capacities, patient flow and edge_label according to the time
for row in range(output.nrows):
    if row > 0:
        _data = output.row_slice(row)
        times.append(_data[0].value)
        edge_current_patients.append(_data[4].value)
        edge_label_output.append(_data[6].value)
        edge_transition.append(_data[8].value)
        node_max_capacity.append(_data[9].value)

# Importing edges and their labels and initializing capacity.
for row in range(hospital.nrows):
    if row > 0:
        _data = hospital.row_slice(row)
        _Node1 = _data[0].value
        _Node2 = _data[1].value
        edge_label_input = _data[2].value
        G_flow.add_edges_from(
            [
                (_Node1, _Node2, {"label_input": edge_label_input, "residual_capacity": 0})
            ]
        )

# print(G_flow.edges(data=True))

#for _label, patients, transition, capacity in zip(edge_label_output[:5], edge_current_patients[:5], edge_transition[:5],
#                                                  node_max_capacity[:5]):
#    for u, v, net in G_flow.edges(data=True):
#        if G_flow['label_input'] == _label:
#            net['residual_capacity'] = int(capacity / edge_transition) - patients

mean_arr = []
max_time = int(math.ceil(max(times)))
j = 0
for i in range(1, max_time+1):
    current_mean = 0
    counter = 0
    while j < len(times) and times[j] < i:
        current_mean += edge_current_patients[j]
        j += 1
        counter += 1
    current_mean /= counter
    mean_arr.append(current_mean)


#for u, v, net in G_flow.edges(data=True):
    # net['residual_capacity'] = (Total_number_of_beds/incoming_edge_transition) - current_patient

#    net['residual_capacity'] = random.randint(1, 10)
#print("calculating different residual graphs for flow:")
#RG = build_residual_network(G_flow, capacity="residual_capacity")
#pp = preflow_push(G_flow, "ward_1", "ward_7", capacity="residual_capacity",
#                  residual=RG)  # Complexity O(sqr(V)sqrt(E)) //Best since Orlin is unavailable.
# print(pp.edges.data())
#print(nx.to_numpy_matrix(pp, weight="flow"))
#print("Structural Hole weak ties:")
#const = nx.constraint(G_flow,
#                      weight="residual_capacity")  # Less constraint indicates more structural holes in a network
#print(const)
#print(min(const, key=const.get))

# for _node in G_flow.nodes.data():
# edge_weights = nx.get_edge_attributes(RG,"capacity")
# RG.remove_edges_from((e for e, w in edge_weights.items() if w == 0))
