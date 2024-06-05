import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LassoLars
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.cm as cm

from optim import nesterov_accelerated_gradient, primal_gradient, dual_gradient
from util import load_diabetes_data

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
beta_nesterov = nesterov_accelerated_gradient(y_train, X_train, lambda_)
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
vk, objective_values, beta_dual = dual_gradient(X_train, y_train, lambda_)


def plot_beta_coefficients(selected_methods, betas, colors):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(20, 10))

    width = 0.15  # Width of the bars
    indices = np.arange(len(betas[0]))

    n_methods = len(selected_methods)
    positions = np.linspace(
        -width * (n_methods - 1) / 2, width * (n_methods - 1) / 2, n_methods
    )

    for pos, method, beta, color in zip(positions, selected_methods, betas, colors):
        ax.bar(indices + pos, beta, width=width, label=method, color=color)

    for i in range(len(indices)):
        ax.axvline(x=i, color="white", linestyle="--", linewidth=0.5)

    ax.set_xlabel("Features", fontsize=14)
    ax.set_ylabel("Beta Coefficients", fontsize=14)
    ax.set_title(
        r"Beta Coefficients for each method and selected $\lambda$ or $\alpha$",
        fontsize=16,
    )
    ax.set_xticks(indices)
    ax.legend(fontsize=14)
    ax.grid(True, linestyle="--", linewidth=0.5, color="gray")

    st.pyplot(fig)


# Prepare data for plotting
betas = []
colors = []
cmap = cm.get_cmap("inferno")

if "Nesterov" in methods:
    betas.append(beta_nesterov)
    colors.append(cmap(0.2))
if "Lasso" in methods:
    betas.append(beta_lasso)
    colors.append(cmap(0.35))
if "LARS" in methods:
    betas.append(beta_lars)
    colors.append(cmap(0.5))
if "Primal Gradient" in methods:
    betas.append(beta_primal)
    colors.append(cmap(0.65))
if "Dual Gradient" in methods:
    betas.append(beta_dual)
    colors.append(cmap(0.8))

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


# Function to apply red background color to cells with value 0
def highlight_zeros(val):
    color = "red" if val == 0 else ""
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
