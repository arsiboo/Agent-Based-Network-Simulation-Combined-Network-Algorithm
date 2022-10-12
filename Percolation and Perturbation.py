# Percolation
from typing import List, Any
import networkx as nx
import numpy as np
import xlrd

file = xlrd.open_workbook("mad_house.xlsx")
data = xlrd.open_workbook("data.xlsx")

hospital = file.sheet_by_name("Links")
wards = file.sheet_by_name("Nodes")
output = data.sheet_by_name("queue_info")

G_flow = nx.DiGraph()

node_max_capacity = []
edge_current_patients = []
edge_transition = []
node_overflow_state = []
edge_label_output: List[Any] = []

# Importing capacities, patient flow and edge_label according to the time
for row in range(output.nrows):
    if row > 0:
        _data = output.row_slice(row)
        edge_label_output.append(_data[6].value)
        edge_transition.append(_data[9].value)
        node_max_capacity.append(_data[10].value)
        if _data[11].value < 0:
            node_overflow_state.append(1)  # overflow state
        else:
            node_overflow_state.append(0.1)  # not overflow state

nodes_mapping_list = []
capacity_mapping_list = []

# Importing edges and their labels and initializing capacity.
for row in range(hospital.nrows):
    if row > 0:
        _data = hospital.row_slice(row)
        _Node1 = _data[0].value
        _Node2 = _data[1].value
        edge_label_input = _data[2].value
        G_flow.add_edges_from(
            [
                (_Node1, _Node2, {"label_input": edge_label_input, "per_capacity": 0})
            ]
        )

nx.set_node_attributes(G_flow, 0.1, name='overflow_state')

for label, state, capacity, transition in zip(edge_label_output, node_overflow_state, node_max_capacity,
                                              edge_transition):
    for u, v, net_edge in G_flow.edges(data=True):
        if net_edge['label_input'] == label:
            net_edge['per_capacity'] = capacity / transition
            for k, net_node in G_flow.nodes(data=True):
                if k == v:
                    print(k)
                    print(v)
                    if state < 0:
                        nx.set_node_attributes(net_node, {k: 1}, 'overflow_state')  # Overflowed
                    elif state == 0:
                        nx.set_node_attributes(net_node, {k: 0.5}, 'overflow_state')  # At Edge
                    elif state > 0:
                        nx.set_node_attributes(net_node, {k: 0.1}, 'overflow_state')  # No flowed
                    print(G_flow.nodes.data())

            #           print("Calculating percolation:")

            #           print(nx.percolation_centrality(G_flow, attribute='overflow_state', weight='per_capacity'))
        #        print("-----------------------------------------")

# # for _node in G_flow.nodes.data():
# # edge_weights = nx.get_edge_attributes(RG,"capacity")
# # RG.remove_edges_from((e for e, w in edge_weights.items() if w == 0))
