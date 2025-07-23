# Percolation and Perturbation
# Sort the output file before running the codes
from typing import List, Any
import networkx as nx
import numpy as np
import xlrd
import xlsxwriter
import pandas as pd


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


network_type = ["Normal","Extra"]

percol_perturb = xlsxwriter.Workbook("Percolation and Perturbation/percol_perturb.xlsx")
perturb_sheet = percol_perturb.add_worksheet("perturb")


normal_dict=[]
extra_dict=[]

G_flow = nx.DiGraph()

for net in network_type:
    file = xlrd.open_workbook("hospital.xlsx")
    data = xlrd.open_workbook("Simulation/OUTCOME_"+net+".xlsx")

    hospital = file.sheet_by_name(net+"Links")
    wards = file.sheet_by_name(net+"Nodes")
    output = data.sheet_by_name("queue_info")



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

    overflow_states = abs((normalize_data(node_overflow_state) - 1) * -1)

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

    counting = 0
    countings = 0
    for state, node in zip(overflow_states, upcoming_node):
        for n, net_node in G_flow.nodes(data=True):
            if n == node:
                nx.set_node_attributes(G_flow, {n: state}, name='overflow_state')  # Not Overflowed or unaffected
                percolation = nx.percolation_centrality(G_flow, attribute='overflow_state',
                                                        weight='per_capacity')
                percol_dict=dict(reversed(sorted(percolation.items(), key=lambda it: it[1])))
                if net=="Normal":
                    normal_dict.append(percol_dict)
                elif net=="Extra":
                    extra_dict.append(percol_dict)

ndict=pd.DataFrame(normal_dict)
edict=pd.DataFrame(extra_dict)

perturbation_dict = {}

for nodes in G_flow.nodes(data=True):
    if nodes[0] != "Extra Node":
        perturbation_dict[nodes[0]]=ndict[nodes[0]].mean() - edict[nodes[0]].mean()


counting = 0
for key,val in perturbation_dict.items():
    perturb_sheet.write(counting, 0, str(key))
    perturb_sheet.write(counting, 1, str(val))
    counting += 1

percol_perturb.close()
