import time
import numpy as np
from .common import lasso_objective, lasso_gradient


def dual_gradient(
    X: np.ndarray,
    y: np.ndarray,
    lambda_: float,
    v0: np.ndarray = None,
    L0: float = 10.0,
    gamma_d: float = 2,  # recommended to be in [2, 3] in the original paper
    max_iter: int = 100,
    verbose: bool = False,
) -> tuple:
    """
    Performs the Dual Gradient Method optimization for LASSO regression.

    Args:
        X (ndarray): Feature matrix.
        y (ndarray): Response vector.
        lambda_ (float): Regularization parameter.
        v0 (ndarray, optional): Initial guess for the parameters. If None, initializes to zeros. Defaults to None.
        L0 (float, optional): Initial value for the Lipschitz constant. Defaults to 10.0.
        gamma_d (float, optional): Rate at which L is adjusted downwards when no backtracking occurs. Defaults to 2.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        verbose (bool, optional): Whether to print detailed progress messages. Defaults to False.

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


def dual_gradient_new(A, b, v0, L0, gamma_d, max_iter=10000, tol=1e-6):
    """
    Dual Gradient Method with Nesterov acceleration for solving Ax = b using least squares,
    including gap history and loss history.
    """
    m, n = A.shape
    v = v0
    v_prev = v0
    L = L0
    gap_history = []
    loss_history = []
    cpu_time_history = []
    initial_residual = np.dot(A, v) - b
    initial_loss = np.linalg.norm(initial_residual)**2 / 2

    for k in range(max_iter):
        start_time = time.process_time()
        # Nesterov acceleration step
        if k > 0:
            y = v + (k - 1) / (k + 2) * (v - v_prev)
        else:
            y = v

        grad_f = 2 * np.dot(A.T, (np.dot(A, y) - b))  # Gradient of the least squares loss

        Mk = np.linalg.norm(grad_f, 2)                # Computing Mk
        if Mk == 0:
            Mk = 1e-16  # Prevent division by zero

        v_prev = v
        v -= gamma_d * grad_f / Mk

        current_residual = np.dot(A, v) - b
        current_loss = np.linalg.norm(current_residual)**2 / 2
        current_gap = np.linalg.norm(current_residual)**2 / np.linalg.norm(initial_residual)**2

        end_time = time.process_time()

        gap_history.append(current_gap)
        loss_history.append(current_loss)
        cpu_time_history.append(end_time - start_time)

        # Check for convergence based on the relative gap
        if current_gap < tol:
            break

    return v, k, current_gap, gap_history, loss_history, cpu_time_history
