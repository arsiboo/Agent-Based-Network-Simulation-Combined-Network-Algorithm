# A Hybrid Approach to Model Hospitals and Evaluate Wards' Performances

This repository contains the code and materials associated with the paper. The agent based network simulation model the behaviour of the whole hospital, structural hole and flow algorithms identify the warrds with common persistent bottlenecks, and percolation and perturbation analysis evaluate the wards' performances.

> **Note:** This research was conducted as part of my PhD studies at KTH Royal Institute of Technology.

---

## Citation

If you use this approach in your research or publications, please cite the following work:

Boodaghian Asl, A., Marzano, L., Raghothama, J., Darwich, A. S., Falk, N., Bodeby, P., & Meijer, S. (2025). *A Hybrid Approach to Model Hospitals and Evaluate Wards' Performances.* IEEE Access 
[Read the Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11037449)

---

## Usage

### 1. Running the codes in order:

- First, run `fitting_dist.py`. The code uses the `fitter` Python package to generate a list of files representing distribution functions for each ward based on patients length of stay.

- Second, run `simulation.py`. This is the agent-based network modelling code that simulates patient flow throughout the hospital wards from arrival to discharge.

- Third, run `residual_graph_structural_holes.py`. The code uses flow and structural hole algorithms to identify common persistent bottlenecks from both patient flow and structural weaknesses perspectives.

- Then, apply scenarios in the `Patient` class and rerun `simulation.py` using a different network type called `Extra` instead of `Normal`. This will give you a different output for the patient flow.

- Fourth, run `percolation_and_perturbation.py`. The code measures the divergence of the average patient flow per ward based on their percolation state, which helps evaluate the wards' performance.


### 2. Code Structure
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
- [Studying the Effect of Online Medical Applications on Patients Healing Time and Doctors Utilization Using Discrete Event Simulation](https://github.com/arsiboo/Discrete-Event-Simulation-EHealth)
  

