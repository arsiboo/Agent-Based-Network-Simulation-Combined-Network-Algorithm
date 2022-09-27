# Resodial flows
import json
import random
import networkx as nx
import xlrd
import matplotlib.pyplot as plt
from networkx.algorithms.flow import edmonds_karp, shortest_augmenting_path, preflow_push, dinitz, boykov_kolmogorov

file = xlrd.open_workbook("mad_house.xlsx")
hospital = file.sheet_by_name("Links")
wards = file.sheet_by_name("Nodes")
node_total_beds = []
node_avg_serving_time = []

G = nx.DiGraph()
for row in range(hospital.nrows):
    if row > 0:
        _data = hospital.row_slice(row)
        if _data[2].value != 0:
            _Node1 = _data[0].value
            _Node2 = _data[1].value
            G.add_edges_from(
                [
                    (_Node1, _Node2, {"patient_flow_rate": _data[2].value, "distribution_probability": _data[3].value})
                ]
            )

for row in range(wards.nrows):
    if row > 0:
        _data = wards.row_slice(row)
        node_total_beds.append(_data[1].value)
        node_avg_serving_time.append(_data[2].value)

node_list = list(G.nodes())

# Assigning attribute to nodes:
total_beds_dict = {k: v for k, v in zip(node_list, node_total_beds)}
serving_time_dict = {k: v for k, v in zip(node_list, node_avg_serving_time)}
nx.set_node_attributes(G, total_beds_dict, 'beds')
nx.set_node_attributes(G, serving_time_dict, 'serving')

# for u in G3.nodes:
#     for v in G3.nodes:
#         if nx.has_path(G3,u,v):
#             print(G3.get_edge_data(u,v))


# Constraints:
sh_dict = nx.constraint(G, weight="patient_flow_rate")
# es_dict = nx.effective_size(G,weight="patient_flow_rate")

print(sh_dict)
# print(es_dict)

pos = nx.spring_layout(G)
nx.draw_networkx(G, pos, cmap=plt.get_cmap('jet'),
                 node_color='pink', node_size=500, label=True)

# plt.show()
