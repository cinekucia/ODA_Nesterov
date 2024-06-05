import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LassoLars
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.cm as cm

from optim import nesterov_accelerated_gradient, primal_gradient, dual_gradient
from util import load_diabetes_data


st.set_option("deprecation.showPyplotGlobalUse", False)

# Prepare the data
# Load the diabetes dataset
X_train, X_test, y_train, y_test = load_diabetes_data()

# Set Streamlit layout to wide
st.set_page_config(layout="wide")

# Streamlit app
st.title("LASSO Regression Optimization Methods Comparison")

# Lambda/alpha parameter selector in the sidebar
lambda_ = st.sidebar.slider("Select Lambda/Alpha", 0.0, 2.0, value=0.0, step=0.01)

# Multiselect for methods
methods = st.sidebar.multiselect(
    "Select Methods to Display",
    ["Nesterov", "Lasso", "LARS", "Primal Gradient", "Dual Gradient"],
    default=["Nesterov", "Lasso", "LARS", "Primal Gradient", "Dual Gradient"],
)

# Calculate beta coefficients and MSE for each method
beta_nesterov, loss_history_nesterov = nesterov_accelerated_gradient(
    X_train, y_train, lambda_
)
mse_nesterov = mean_squared_error(y_test, X_test.dot(beta_nesterov))

lasso = Lasso(alpha=lambda_)
lasso.fit(X_train, y_train)
beta_lasso = lasso.coef_
mse_lasso = mean_squared_error(y_test, lasso.predict(X_test))

lasso_lars = LassoLars(alpha=lambda_)
lasso_lars.fit(X_train, y_train)
beta_lars = lasso_lars.coef_
mse_lars = mean_squared_error(y_test, lasso_lars.predict(X_test))

# Primal Gradient
beta_primal, loss_history_primal = primal_gradient(X_train, y_train, lambda_)

# Dual Gradient
vk, loss_history_dual, beta_dual = dual_gradient(X_train, y_train, lambda_)


def plot_beta_coefficients(selected_methods, betas, colors):
    """Plots bar chart of beta coefficients for each method marked with different colors."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(20, 10))

    width = 0.15  # Width of the bars
    indices = np.arange(len(betas[0]))

    n_methods = len(selected_methods)
    positions = np.linspace(
        -width * (n_methods - 1) / 2, width * (n_methods - 1) / 2, n_methods
    )
    flattened_beta = np.concatenate(betas)
    max_abs_value = np.max(np.abs(flattened_beta))
    y_lim = (-max_abs_value, max_abs_value)

    for pos, method, beta, color in zip(positions, selected_methods, betas, colors):
        ax.bar(indices + pos, beta, width=width, label=method, color=color)

    ax.set_xlabel("Features", fontsize=14)
    ax.set_ylabel("Beta Coefficients", fontsize=14)
    ax.set_title(
        r"Beta Coefficients for each method and selected $\lambda$ or $\alpha$",
        fontsize=16,
    )
    ax.set_xticks(indices)
    ax.legend(fontsize=14)
    ax.grid(True, linestyle="--", linewidth=0.5, color="gray")
    ax.set_ylim(y_lim)

    st.pyplot(fig)


def plot_objective_function_per_iteration(objective_values_dict):
    """Plots the objective function per iteration for each method."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(20, 10))

    for method, values in objective_values_dict.items():
        ax.plot(values, label=method)

    ax.set_xlabel("Iterations", fontsize=14)
    ax.set_ylabel("Objective Function Value", fontsize=14)
    ax.set_title("Objective Function per Iteration", fontsize=16)
    ax.legend(fontsize=14)
    ax.grid(True, linestyle="--", linewidth=0.5, color="gray")

    st.pyplot(fig)


# Prepare data for plotting
betas = []
colors = []
objective_values = {}
cmap = cm.get_cmap("inferno")

if "Nesterov" in methods:
    betas.append(beta_nesterov)
    colors.append(cmap(0.2))
    objective_values["Nesterov"] = loss_history_nesterov
if "Lasso" in methods:
    betas.append(beta_lasso)
    colors.append(cmap(0.35))
    # we skip the objective function for Lasso as it is not available
if "LARS" in methods:
    betas.append(beta_lars)
    colors.append(cmap(0.5))
    # we skip the objective function for LARS as it is not available
if "Primal Gradient" in methods:
    betas.append(beta_primal)
    colors.append(cmap(0.65))
    objective_values["Primal Gradient"] = loss_history_primal
if "Dual Gradient" in methods:
    betas.append(beta_dual)
    colors.append(cmap(0.8))
    objective_values["Dual Gradient"] = loss_history_dual


# Plot the selected methods
plot_beta_coefficients(methods, betas, colors)

# Display the results
st.write("Beta Coefficients Comparison:")

# Create a DataFrame to display the beta coefficients for comparison
beta_df = pd.DataFrame(
    {
        "Features": [f"Beta_{i}" for i in range(len(beta_nesterov))],
        "Nesterov": np.round(beta_nesterov, 2),
        "Lasso": np.round(beta_lasso, 2),
        "LARS": np.round(beta_lars, 2),
        "Primal Gradient": np.round(beta_primal, 2),
        "Dual Gradient": np.round(beta_dual, 2),
    }
)
# Remove index of the dataframe
beta_df.set_index("Features", inplace=True)
beta_df = beta_df.round(2).astype(str)


# Function to apply red background color to cells with value 0
def highlight_zeros(val, eps=0.05):
    color = "red" if abs(float(val)) < eps else ""
    return f"background-color: {color}"


# Display beta coefficients table with red color for zeros and customized styling
st.dataframe(
    beta_df.transpose()
    .style.applymap(highlight_zeros)
    .set_table_styles(
        [
            {
                "selector": "td",
                "props": [
                    ("font-size", "30px"),
                    ("width", "100px"),
                    ("text-align", "left"),
                ],
            }
        ]
    )
)

# Plot the objective function per iteration for each method
plot_objective_function_per_iteration(objective_values)

# Display MSEs for each method
mse_df = pd.DataFrame(
    {
        "Method": ["Nesterov", "Lasso", "LARS", "Primal Gradient", "Dual Gradient"],
        "Mean Squared Error": [
            mse_nesterov,
            mse_lasso,
            mse_lars,
            mean_squared_error(y_test, X_test.dot(beta_primal)),
            mean_squared_error(y_test, X_test.dot(beta_dual)),
        ],
    }
)
mse_df.set_index("Method", inplace=True)

# Display MSEs for each method without coloring
st.write("Mean Squared Errors for Each Method:")
st.dataframe(mse_df)
