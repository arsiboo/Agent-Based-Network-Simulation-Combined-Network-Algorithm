# A Hybrid Approach to Model Hospitals and Evaluate Wards' Performances

This repository contains the code and materials associated with the paper. The agent based network simulation model the behaviour of the whole hospital, structural hole and flow algorithms identify the warrds with common persistent bottlenecks, and percolation and perturbation analysis evaluate the wards' performances.

> **Note:** This research was conducted as part of my PhD studies at KTH Royal Institute of Technology.

---

## Citation

If you use this algorithm in your research or publications, please cite the following work:

Boodaghian Asl, A., Marzano, L., Raghothama, J., Darwich, A. S., Falk, N., Bodeby, P., & Meijer, S. (2025). *A Hybrid Approach to Model Hospitals and Evaluate Wards' Performances.* IEEE Access 
[Read the Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11037449)

---

## Code Overview

This repository implements a hybrid simulation-analytical approach to model patient flow and evaluate ward performance in hospital settings.

- **`simulation.py`**  
  Implements an agent-based network simulation modeling the flow of patients throughout the hospital, from ward to ward.

- **`residual_graph_and_structural_hole.py`**  
  Analyzes the simulation output to identify bottlenecks and structural vulnerabilities in the hospital network.

- **`percolation_perturbation.py`**  
  Compares two simulation states (baseline vs perturbed) to evaluate ward performance changes using percolation dynamics.

- **`fitting_dist.py`**  
  Fits statistical distribution functions to each ward’s service time data.

- **`serviceValidation.py`**  
  Validates the accuracy of fitted distributions against real-world ward behavior.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Related Work

- [A Dynamic Nonlinear Flow Algorithm to Model Patient Flow](https://github.com/arsiboo/Dynamic-Nonlinear-Flow-Algorithm)

