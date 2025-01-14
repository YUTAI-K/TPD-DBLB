# TSPD-DLBP Solver

This repository provides a mixed-integer programming solver for the **Two-Sided Partial Destructive Disassembly Line Balancing Problem (TSPD-DLBP)**. The solver is implemented in Python and supports optimization of the TSPD-DLBP model using Gurobi.

The main concept and model are derived from the paper: *"Exploring Engineering Applications of Two-Sided Partial Destructive Disassembly Line Balancing Problem under Electrical Limiting and Time-of-Use Pricing"* by Lei Guo, Zeqiang Zhang, Yu Zhang, Yan Li, and Haolin Song. For a detailed description of the problem, including its parameters and constraints, please refer to their paper, particularly Section 4.2.

## Repository Structure

- **`main.ipynb`**: A Jupyter Notebook that contains the main implementation and workflow of the solver. It explains the parameter generation, constraints, and optimization process in detail.
- **`Utilities.py`**: A Python module containing utility functions used within the notebook, such as helper methods for setting up the problem and processing results.

Both files must be placed in the same directory for the solver to function correctly.

## Getting Started

### Clone the Repository

To get started, clone the repository to your local machine:

```bash
git clone https://github.com/YUTAI-K/TPD-DBLB.git
