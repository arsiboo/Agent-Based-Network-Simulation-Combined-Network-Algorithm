#dictionary and value flows
import json
import random
import networkx as nx
import xlrd
import matplotlib.pyplot as plt
from networkx.algorithms.flow import edmonds_karp

file = xlrd.open_workbook("mad_house.xlsx")
hospital = file.sheet_by_index(0)
wards = file.sheet_by_index(1)
node_beds = []
node_occupancy = []

G = nx.DiGraph()
for row in range(hospital.nrows):
    if row > 0:
        _data = hospital.row_slice(row)
        if _data[2].value != 0:
            _Node1 = _data[0].value
            _Node2 = _data[1].value
            G.add_edges_from(
                [
                    (_Node1, _Node2, {"capacity": _data[2].value, "weight": _data[2].value})
                ]
            )

for row in range(wards.nrows):
    if row > 0:
        _data = wards.row_slice(row)
        node_beds.append(_data[1].value)
        node_occupancy.append(_data[2].value)

node_list = list(G.nodes())

#Assigning attribute to nodes:
beds_dict = {k: v for k, v in zip(node_list, node_beds)}
occupancy_dict = {k: v for k, v in zip(node_list, node_occupancy)}
nx.set_node_attributes(G, beds_dict, 'beds')
nx.set_node_attributes(G, occupancy_dict, 'occupancy')


#calculating the maximum_flow:
flowvalue_mf, flowdict_mf = nx.maximum_flow(G, "ward 1", "ward 6")
for i in flowdict_mf:
    print(flowdict_mf[i])

print("-----------------------------------------------------------")

#calculating the network
flowCost_nx, flowDict_nx = nx.network_simplex(G)
for i in flowDict_nx:
    print(flowDict_nx[i])



pos = nx.spring_layout(G)
nx.draw_networkx(G, pos, cmap=plt.get_cmap('jet'),
                       node_color = 'pink', node_size = 500,label=True)

#plt.show()

