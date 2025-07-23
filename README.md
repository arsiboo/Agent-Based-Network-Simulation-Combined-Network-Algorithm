# A Hybrid Approach to Model Hospitals and Evaluate Wards' Performances

**Paper link:** [IEEE Xplore](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11037449)

**Citation:**  
Boodaghian Asl, A., Marzano, L., Raghothama, J., Darwich, A. S., Falk, N., Bodeby, P., & Meijer, S. (2025). *A Hybrid Approach to Model Hospitals and Evaluate Wards’ Performances*. IEEE Access.

---

## Code Overview

- **`simulation.py`**  
  Implements an agent-based network simulation modeling the flow of patients throughout the hospital, from ward to ward.

- **`residual_graph_and_structural_hole.py`**  
  Analyzes the simulation output to identify patient flow bottlenecks and structural weaknesses in the hospital network.

- **`percolation_perturbation.py`**  
  Compares two simulation outputs (baseline and perturbed states) to quantify performance changes in wards based on percolation dynamics.

- **`fitting_dist.py`**  
  Fits appropriate statistical distribution functions to each ward’s service time data.

- **`serviceValidation.py`**  
  Validates whether the fitted distributions accurately reflect real-world ward behavior.
