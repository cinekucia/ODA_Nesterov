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


# Plot
# Apply dark theme
plt.style.use("dark_background")

# Create a larger figure
fig, ax = plt.subplots(figsize=(20, 10))

width = 0.2  # Width of the bars
indices = np.arange(len(beta_nesterov))

# Apply inferno colormap colors
cmap = cm.get_cmap("inferno")
color_nesterov = cmap(0.2)
color_lasso = cmap(0.35)
color_primal = cmap(0.5)
color_dual = cmap(0.65)
color_lars = cmap(0.8)

# Plot
ax.bar(
    indices - width, beta_nesterov, width=width, label="Nesterov", color=color_nesterov
)
ax.bar(indices, beta_lasso, width=width, label="Lasso", color=color_lasso)
ax.bar(indices + width, beta_lars, width=width, label="LARS", color=color_lars)

# add the primal gradient coefficients
ax.bar(
    indices + 2 * width,
    beta_primal,
    width=width,
    label="Primal Gradient",
    color=color_primal,
)
# add the dual gradient coefficients
ax.bar(
    indices + 3 * width,
    beta_dual,
    width=width,
    label="Dual Gradient",
    color=color_dual,
)

ax.set_xlabel("Features", fontsize=14)
ax.set_ylabel("Beta Coefficients", fontsize=14)
ax.set_title(
    r"Beta Coefficients for each method and selected $\lambda$ or $\alpha$", fontsize=16
)
ax.set_xticks(indices)
ax.legend(fontsize=14)
ax.grid(True, linestyle="--", linewidth=0.5, color="gray")

st.pyplot(fig)

# Display the results
st.write("Beta Coefficients Comparison:")

# Create a DataFrame to display the beta coefficients for comparison
beta_df = pd.DataFrame(
    {
        "Features": [f"Beta_{i}" for i in range(len(beta_nesterov))],
        "Nesterov": beta_nesterov,
        "Lasso": beta_lasso,
        "LARS": beta_lars,
        "Primal Gradient": beta_primal,
        "Dual Gradient": beta_dual,
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
# mse_df = pd.DataFrame(
#     {
#         "Method": ["Nesterov", "Lasso", "LARS", "Primal Gradient", "Dual Gradient"],
#         "Mean Squared Error": [
#             mse_nesterov,
#             mse_lasso,
#             mse_lars,
#             mean_squared_error(y_test, X_test.dot(beta_primal)),
#             mean_squared_error(y_test, X_test.dot(beta_dual)),
#         ],
#     }
# )
# mse_df.set_index("Method", inplace=True)

# # Display MSEs for each method without coloring
# st.write("Mean Squared Errors for Each Method:")
# st.dataframe(mse_df)
