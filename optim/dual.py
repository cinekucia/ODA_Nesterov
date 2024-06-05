import numpy as np
from .common import lasso_objective, lasso_gradient


def dual_gradient(
    X, y, lambda_, v0=None, L0=10.0, gamma_d=2, max_iter=100, verbose=False
):
    """
    Performs the Dual Gradient Method optimization.

    Args:
    X (np.ndarray): Feature matrix.
    y (np.ndarray): Response vector.
    lambda_ (float): Regularization parameter.
    v0 (np.ndarray): Initial guess for the parameters. If None, initializes to zeros.
    L0 (float): Initial value for the Lipschitz constant.
    gamma_d (float): Rate at which L is adjusted downwards when no backtracking occurs.
    max_iter (int): Maximum number of iterations.
    verbose (bool): Whether to print detailed progress messages.

    Returns:
    tuple: The final parameters, list of objective values per iteration, and all parameter updates as a numpy array.

    Example:
    # Assuming X_train, y_train, lasso_objective, and lasso_gradient are defined
    lambda_ = 0.6
    v0 = np.zeros(X_train.shape[1])
    L0 = 10.0
    gamma_d = 2
    max_iter = 100

    vk, objective_values, beta_values = dual_gradient_method(
    lasso_objective, lasso_gradient, X_train, y_train, lambda_, v0, L0, gamma_d, max_iter)
    """
    if v0 is None:
        v0 = np.zeros(X.shape[1])

    f = lasso_objective
    grad = lasso_gradient
    v = v0
    L = L0
    objective_values = []
    beta_values = []
    max_backtracks = (
        10  # Limit the number of backtracking steps to avoid infinite loops
    )

    for i in range(max_iter):
        g = grad(X, y, v, lambda_)
        initial_step_size = 1 / L
        step_size = initial_step_size
        next_v = v - step_size * g
        current_obj = f(X, y, v, lambda_)
        next_obj = f(X, y, next_v, lambda_)

        backtrack_count = 0
        while next_obj > current_obj and backtrack_count < max_backtracks:
            step_size *= 0.5
            next_v = v - step_size * g
            next_obj = f(X, y, next_v, lambda_)
            backtrack_count += 1

        if backtrack_count == 0:
            L /= gamma_d  # Decrease L to accelerate convergence
        else:
            L = max(
                L0, 1 / step_size
            )  # Update L based on the effective step size that worked

        v = next_v
        objective_values.append(current_obj)
        beta_values.append(v.copy())

        if verbose:
            print(
                f"Iteration {i + 1}: Objective = {current_obj}, L = {L}, Backtracks = {backtrack_count}"
            )
    # beta_values = historical parameter values
    return v, objective_values, np.array(beta_values[-1])
