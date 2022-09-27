import json
import random

import networkx as nx
import xlrd
import matplotlib.pyplot as plt
from networkx.algorithms.flow import edmonds_karp


def convert_networkx_graph_to_cytoscape_json(G, edge_labels: dict = None, node_style: dict = None,
                                             edge_style: dict = None, layout="") -> str:
    # load all nodes into nodes array<<<
    final = {}
    final["directed"] = G.is_directed()
    final["multigraph"] = G.is_multigraph()
    # final["elements"] = {"nodes": [], "edges": []}
    final["elements"] = []
    added_nodes = {}

    if node_style is None:
        node_style = {
            'label': 'data(label)',
            'width': '60px',
            'height': '60px',
            'color': 'blue',
            'background-fit': 'contain',
            'background-clip': 'none'
        }
    if edge_style is None:
        edge_style = {
            'label': 'data(label)',
            'text-background-color': 'yellow',
            'text-background-opacity': 0.4,
            'width': '6px',
            'curve-style': 'bezier',
            'target-arrow-shape': 'triangle',
            'control-point-step-size': '140px'
        }

    beds = nx.get_node_attributes(G, "beds")
    serving = nx.get_node_attributes(G, "serving")
    occupancy = nx.get_node_attributes(G, "occupancy")
    patients = nx.get_node_attributes(G, "patients")
    overflow = nx.get_node_attributes(G, "overflow")

    staff = nx.get_node_attributes(G, "staff")

    for n in G.nodes():
        nparent = None
        ntype = type(n).__name__

        if (ntype == "MicroserviceEndpointFunction" or
                (ntype == 'tuple' and type(n[1]).__name__ == 'MicroserviceEndpointFunction')):
            _n = n.microservice if ntype != "tuple" else n[1].microservice
            _n_id = str(_n)
            if _n_id not in added_nodes.keys():
                nparent = {"group": "nodes", "data": {}, "classes": type(_n).__name__}
                nparent["data"]["id"] = _n_id
                nparent["data"]["label"] = _n_id
                added_nodes[_n_id] = nparent
                final["elements"].append(nparent)
            else:
                nparent = added_nodes[_n_id]

        nx1 = {"group": "nodes", "data": {}, "classes": ntype}
        nx_id = str(n)
        nx1["data"]["id"] = nx_id
        nx1["data"]["label"] = str(nx_id) + " (" + str(beds[nx_id]) + ", " + str(
            serving[nx_id]) + ", " + str(occupancy[nx_id]) + ", " + str(
            patients[nx_id]) + ", " + str(
            overflow[nx_id]) + ")" if ntype != "tuple" else str(n[1])
        if nparent is not None:
            nx1["data"]["parent"] = str(n.microservice) if ntype != "tuple" else str(n[1].microservice)
        added_nodes[nx_id] = nx1
        # final["elements"]["nodes"].append(nx.copy())
        final["elements"].append(nx1)

    n = 0
    for e in G.edges:
        edge_data = G.get_edge_data(e[0], e[1], e[2])
        nx1 = {"group": "edges", "data": {}}
        nx1["data"]["id"] = str(e[0]) + "_" + str(e[1]) + "_" + str(n)
        n += 1
        nx1["data"]["source"] = str(e[0])
        nx1["data"]["target"] = str(e[1])
        nx1["data"]["label"] = str(edge_data['weight'])
        # final["elements"]["edges"].append(nx1)
        final["elements"].append(nx1)

    final["container"] = "----"
    final["layout"] = {}
    final["layout"]["animate"] = "true"
    final["layout"]["randomize"] = "true"
    final["layout"]["gravity"] = "-300"
    final["style"] = [{
        "selector": 'node',
        "style": node_style
    }, {
        "selector": 'edge',
        "style": edge_style
    }
    ]
    final = json.dumps(final).replace('\"----\"', "document.getElementById('cy')")

    return final


file = xlrd.open_workbook("mad_house.xlsx")
wards_relations = file.sheet_by_name("Links")
wards = file.sheet_by_name("Nodes")

incoming_traffic_rate = 0.5

wards_list = []
total_beds = []
avg_serving_time = []
patients_number = []
wards_occupancy = []
overflow = []

MG = nx.MultiDiGraph()
G= nx.DiGraph()

# importing wards relations information
for row in range(wards_relations.nrows):
    if row > 0:
        _data = wards_relations.row_slice(row)
        _Node1 = _data[0].value
        _Node2 = _data[1].value
        MG.add_weighted_edges_from([(_Node1, _Node2, _data[2].value), (_Node1, _Node2, _data[3].value) , (_Node1, _Node2, _data[4].value)])
#        MG.add_weighted_edges_from(
#            [
#                (_Node1, _Node2, {"patient_flow_rate": _data[2].value, "distribution_probability": _data[3].value})
#            ]
#        )

# importing wards information
for row in range(wards.nrows):
    if row > 0:
        _data = wards.row_slice(row)
        wards_list.append(_data[0].value)
        total_beds.append(_data[1].value)
        avg_serving_time.append(_data[2].value)
        patients_number.append(_data[3].value)
        wards_occupancy.append(_data[4].value)
        overflow.append(_data[5].value)

# nodes parameters
beds_dict = {k: v for k, v in zip(wards_list, total_beds)}
serving_dict = {k: v for k, v in zip(wards_list, avg_serving_time)}
patients_dict = {k: v for k, v in zip(wards_list, patients_number)}
occupancy_dict = {k: v for k, v in zip(wards_list, wards_occupancy)}
overflow_dict = {k: v for k, v in zip(wards_list, overflow)}

# node attribute
nx.set_node_attributes(MG, beds_dict, 'beds')
nx.set_node_attributes(MG, serving_dict, 'serving')
nx.set_node_attributes(MG, patients_dict, 'patients')
nx.set_node_attributes(MG, occupancy_dict, 'occupancy')
nx.set_node_attributes(MG, overflow_dict, 'overflow')

JSON = convert_networkx_graph_to_cytoscape_json(G=MG)
print(JSON)
