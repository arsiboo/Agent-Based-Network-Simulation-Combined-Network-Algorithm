# Resodial flows and structural hole
# Residual_Capacity = Original_Capacity - current_flow
# Residual_Capacity = (Total_number_of_beds/incoming_edge_transition) - current_people_in_queue
import math
from typing import List, Any
import networkx as nx
import xlrd
from networkx.algorithms.flow import preflow_push, build_residual_network

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
edge_label_output: List[Any] = []

output_size = len(times)

# Importing capacities, patient flow and edge_label according to the time
for row in range(output.nrows):
    if row > 0:
        _data = output.row_slice(row)
        times.append(_data[0].value)
        edge_current_patients.append(_data[4].value)
        edge_label_output.append(_data[6].value)
        edge_transition.append(_data[9].value)
        node_max_capacity.append(_data[10].value)

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

mean_arr = []
res_arr = []
max_time = int(math.ceil(max(times)))
# max_label = int(math.ceil(max(edge_label_output)))
j = 0
for i in range(1, max_time + 1):
    mean_arr.append({})
    res_arr.append({})
    counter = {}
    while j < len(times) and times[j] < i:
        if edge_label_output[j] not in mean_arr[i - 1].keys():
            mean_arr[i - 1][edge_label_output[j]] = 0
            res_arr[i - 1][edge_label_output[j]] = 0
            counter[edge_label_output[j]] = 0
        mean_arr[i - 1][edge_label_output[j]] = \
            (mean_arr[i - 1][edge_label_output[j]] * counter[edge_label_output[j]] + edge_current_patients[j]) / \
            (counter[edge_label_output[j]] + 1)
        res_arr[i - 1][edge_label_output[j]] = (node_max_capacity[j] / edge_transition[j]) - mean_arr[i - 1][
            edge_label_output[j]]
        counter[edge_label_output[j]] += 1
        j += 1


k=1
for item in res_arr:
    for label, res_cap in item.items():
        for u, v, net in G_flow.edges(data=True):
            if net['label_input'] == label:
                net['residual_capacity'] = res_cap
            print("calculating residual graphs using Preflow push algorithm:")
            RG = build_residual_network(G_flow, capacity="residual_capacity")
            pp = preflow_push(G_flow, "ward_1", "ward_7", capacity="residual_capacity",residual=RG)  # Complexity O(sqr(V)sqrt(E)) //Best since Orlin is unavailable.
            print(nx.to_numpy_matrix(pp, weight="flow"))
            print("Calculation structural Hole weak ties:")
            const = nx.constraint(G_flow, weight="residual_capacity")  # The higher the score on the constraint measure
            # "const", the more structural opportunities are constrained and, as a result, the lower the network
            # benefits.
            print(const)
            print(dict(reversed(sorted(const.items(), key=lambda it: it[1]))))
            print(max(const, key=const.get))
            print("-----------------------------------------")
            print(k)
            k+=1

# for _node in G_flow.nodes.data():
# edge_weights = nx.get_edge_attributes(RG,"capacity")
# RG.remove_edges_from((e for e, w in edge_weights.items() if w == 0))
