# ODA_Nesterov
***Authors: Bartosz Grabek, Filip Kucia***

This project implements three different optimisation techniques that solve the LASSO regression problem, namely:
- Primal Gradient (PG)
- Dual Gradient (DG)
- Nesterov's Accelerated Gradient (NAG)

as described in *Nesterov, Y. Gradient methods for minimizing composite functions. Mathematical Programming 140, 125–161 (2013). https://doi.org/10.1007/s10107-012-0629-5*


The methods are compared against the LASSO regression and LARS implementation in `scikit-learn` package.

## How to run the visualisation?
1. Install all of the dependencies, preferably using a Python virtual environment using `requirements.txt`
2. Run:
`streamlit run run_visualisation.py`