# A Hybrid Approach to Model Hospitals and Evaluate Wards' Performances
Paper link: https:https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11037449

Citation: Boodaghian Asl, A., Marzano, L., Raghothama, J., Darwich, A. S., Falk, N., Bodeby, P., & Meijer, S. (2025). A Hybrid Approach to Model Hospitals and Evaluate Wards’ Performances. IEEE Access.

**simulation.py:** The agent-based network simulation that models the flow of the patient in whole hospital from ward to ward.

**residual_graph_and_structural_hole.py:** Iterates over the output extracted from the simulation in order to identify the patient flow bottlenecks and structural weaknesses in hospital.

**percolation_perturbation.py:** Iterates over two different outputs extracted from the simulation, one with the current hospital flow state and one with the change in the hospital flow state in order to measure the perturbation (or performance improvement) of the wards based on their percolation state.

**fitting_dist.py:** Estimates proper distribution function for each ward based on its various service times.

**serviceValidation.py:** Validates if the estimated distribution function accurately represent the real work ward behaviour.  

