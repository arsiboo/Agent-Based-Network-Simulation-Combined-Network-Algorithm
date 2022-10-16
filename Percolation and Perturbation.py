# Percolation and Perturbation
# Sort the output file before running the codes
from typing import List, Any
import networkx as nx
import xlrd


def perturbation_divergence(previous, after):
    return previous - after


def percolation_divergence(previous, after):
    return previous - after


file = xlrd.open_workbook("mad_house.xlsx")
data = xlrd.open_workbook("data.xlsx")

hospital = file.sheet_by_name("Links")
wards = file.sheet_by_name("Nodes")
output = data.sheet_by_name("queue_info")

G_flow = nx.DiGraph()

node_max_capacity = []
edge_current_patients = []
edge_transition = []
upcoming_node = []
node_overflow_state = []
edge_label_output: List[Any] = []

# Importing capacities, patient flow and edge_label according to the time
for row in range(output.nrows):
    if row > 0:
        _data = output.row_slice(row)
        edge_label_output.append(_data[6].value)
        upcoming_node.append(_data[8].value)
        edge_transition.append(_data[9].value)
        node_max_capacity.append(_data[10].value)
        node_overflow_state.append(_data[11].value)

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

for label, capacity, transition in zip(edge_label_output, node_max_capacity, edge_transition):
    for u, v, net_edge in G_flow.edges(data=True):
        if net_edge['label_input'] == label:
            net_edge['per_capacity'] = capacity / transition

for state, node in zip(node_overflow_state, upcoming_node):
    for n, net_node in G_flow.nodes(data=True):
        if n == node:
            if state > 0:
                nx.set_node_attributes(G_flow, {n: 0}, name='overflow_state')  # Not Overflowed
            elif state == 0:
                nx.set_node_attributes(G_flow, {n: 0.5}, name='overflow_state')  # On Edge
            elif state < 0:
                nx.set_node_attributes(G_flow, {n: 1}, name='overflow_state')  # Overflowed
            print("Calculating percolation:")
            percolation=nx.percolation_centrality(G_flow, attribute='overflow_state',
                                            weight='per_capacity')
            print(dict(reversed(sorted(percolation.items(), key=lambda it: it[1]))))
            # Specifically, xt i=0 indicates a non-percolated state at time t,
            # xt i=1 indicates a fully percolated state at time t,
            # while a partially percolated state corresponds to 0vxt iv1.
            # The higher the value, the higher is the percolation of node.
