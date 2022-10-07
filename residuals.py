# Resodial flows
# Residual_Capacity = Original_Capacity - current_flow
# Residual_Capacity = Total_number_of_beds - current_people_in_queue
import random
import networkx as nx
import xlrd
from networkx.algorithms.flow import edmonds_karp, shortest_augmenting_path, preflow_push, dinitz, boykov_kolmogorov, \
    build_residual_network

file = xlrd.open_workbook("mad_house.xlsx")
data = xlrd.open_workbook("OutputExample.xlsx")

hospital = file.sheet_by_name("Links")
wards = file.sheet_by_name("Nodes")
output = data.sheet_by_name("queue_info")


G = nx.DiGraph()
RG = nx.DiGraph()

for row in range(hospital.nrows):
    if row > 0:
        _data = hospital.row_slice(row)
        _Node1 = _data[0].value
        _Node2 = _data[1].value
        edge_label_input = _data[2].value
        #G.add_edges_from([(_Node1, _Node2, int(edge_label_input))]) # knowing the label
        G.add_edges_from(
            [
                (_Node1, _Node2, {"label_input": edge_label_input})
            ]
        )


max_capacity = []
patient_flow = []
edge_label_output = []

# Importing capacities
for row in range(output.nrows):
    if row > 0:
        _data = output.row_slice(row)
        patient_flow.append(_data[4].value)
        edge_label_output.append(_data[6].value)
        max_capacity.append(_data[9].value)

print(len(max_capacity))
print(max_capacity.pop())
print(len(patient_flow))
print(len(edge_label_output))


node_list = list(G.nodes())
#RG = build_residual_network(G, capacity="weight")

# edge_weights = nx.get_edge_attributes(RG,"capacity")
# RG.remove_edges_from((e for e, w in edge_weights.items() if w == 0))


#print("calculating different residual graphs for flow:")

#EK_R = edmonds_karp(G, "ward_1", "ward_7", capacity="weight", residual=RG)

# SAP_R = shortest_augmenting_path(G, "ward_1", "ward_7", capacity="weight", residual=RG)
# PP_R = preflow_push(G, "ward_1", "ward_7", capacity="weight", residual=RG)
# D_R = dinitz(G, "ward_1", "ward_7", capacity="weight", residual=RG)
# BK_R = boykov_kolmogorov(G, "ward_1", "ward_7", capacity="weight", residual=RG)


#for i in G.edges.data():
#    print(i)
#for i in EK_R.edges.data():
#    print(i)

# print(SAP_R.edges.data())
# print(SAP_R)
# print(PP_R.edges.data())
# print(PP_R)
# print(D_R.edges.data())
# print(D_R)
# print(BK_R.edges.data())
# print(BK_R)

# pos = nx.spring_layout(G)
# nx.draw_networkx(G, pos, cmap=plt.get_cmap('jet'),node_color='pink', node_size=500, label=True)
