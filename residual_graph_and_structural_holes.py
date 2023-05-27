# Resodial flows and structural hole
# Residual_Capacity = Original_Capacity - current_flow
# Residual_Capacity = (Total_number_of_beds/incoming_edge_transition) - current_people_in_queue
# Sort Output before usage.
import math
from typing import List, Any
import networkx as nx
import numpy as np
import pandas as pd
import xlrd
import xlsxwriter
from matplotlib import pyplot as plt
from networkx.algorithms.flow import preflow_push, build_residual_network, edmonds_karp, dinitz, boykov_kolmogorov
import seaborn as sns


network_type="Normal"

file = xlrd.open_workbook("akademiska.xlsx")
data = xlrd.open_workbook("Experiment_February/Simulation/OUTCOME_"+network_type+".xlsx")

hospital = file.sheet_by_name(network_type+"Links")
wards = file.sheet_by_name(network_type+"Nodes")
output = data.sheet_by_name("queue_info")

residual_structural= xlsxwriter.Workbook("Experiment_February/Flow and Structural Hole/residuals.xlsx")
residual_sheet= residual_structural.add_worksheet("residual")
structural_hole_sheet= residual_structural.add_worksheet("structura_hole")


G_flow = nx.DiGraph()
RG = nx.DiGraph()

times = []
node_max_capacity = []
edge_current_patients = []
edge_transition = []
edge_label_output: List[Any] = []
edge_source = []
edge_target = []
output_size = len(times)

# Importing capacities, patient flow and edge_label according to the time
for row in range(output.nrows):
    if row > 0:
        _data = output.row_slice(row)
        times.append(_data[0].value)
        edge_current_patients.append(_data[4].value)
        edge_label_output.append(_data[6].value)
        edge_source.append(_data[7].value)
        edge_target.append(_data[8].value)
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
max_time = int(math.ceil(max(times)))
# max_label = int(math.ceil(max(edge_label_output)))
j = 0
res_arr = []
for i in range(1, max_time + 1):
    mean_arr.append({})
    res_arr.append({})
    counter = {}
    while j < len(times) and times[j] < i:
        c = edge_source[j] + "_" + edge_target[j]
        if edge_label_output[j] not in mean_arr[i - 1].keys():
            mean_arr[i - 1][edge_label_output[j]] = 0
            res_arr[i - 1][c] = 0
            counter[edge_label_output[j]] = 0
        mean_arr[i - 1][edge_label_output[j]] = (mean_arr[i - 1][edge_label_output[j]] * counter[edge_label_output[j]] +
                                                 edge_current_patients[j]) / (counter[edge_label_output[j]] + 1)
        res_arr[i - 1][c] = (node_max_capacity[j] / edge_transition[j]) - mean_arr[i - 1][edge_label_output[j]]
        counter[edge_label_output[j]] += 1
        j += 1

mydf = pd.DataFrame.from_dict(res_arr)
mydf.fillna(method='ffill', inplace=True)
mydf.fillna(0, inplace=True)

#length= len(G_flow.nodes())
#length+=2
#fig, ax = plt.subplots()
#im = ax.imshow("Residual Graph")
#ax.set_xticks(np.arange(length), labels=G_flow.nodes())
#ax.set_yticks(np.arange(length), labels=)

counting=0
countings=0
for index, row in mydf.iterrows():  # the time
    for u, v, net in G_flow.edges(data=True):
        label_name = u + '_' + v
        if label_name in row:
            net['residual_capacity'] = row[label_name]
        else:
            net['residual_capacity'] = 0
        # net['residual_capacity'] = res_cap
    #print("calculating residual graphs using Preflow push algorithm:")
    pp = preflow_push(G_flow, "Start", "End",
                      capacity="residual_capacity")  # Complexity O(sqr(V)sqrt(E)) //Best since Orlin is unavailable.

    for node1, node2, data in pp.edges(data=True):
        if 'flow' in pp[node1][node2]:
            if data['flow'] > 0:
                residual_sheet.write(counting, 0, node1)
                residual_sheet.write(counting, 1, node2)
                residual_sheet.write(counting, 2, data['flow'])
                #result = nx.to_pandas_adjacency(pp,weight="flow")
                #result.to_excel("Residual/Residual_Graph_info_"+str(counting)+"_.xlsx")
                counting+=1

    const = nx.constraint(G_flow, weight="residual_capacity")  # The higher the score on the constraint measure
    # "const", the more structural opportunities are constrained and, as a result, the lower the network benefits.
    #print(const)
    const_dict=dict(reversed(sorted(const.items(), key=lambda it: it[1])))
    col = 0
    for i in const_dict:
        structural_hole_sheet.write(countings,col,str(i)+" : "+str(const_dict[i]))
        col += 1
    countings += 1
    #print("-----------------------------------------")

residual_structural.close()
