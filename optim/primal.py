import time
import numpy as np
from .common import lasso_objective, lasso_gradient


def primal_gradient(
    X: np.ndarray,
    y: np.ndarray,
    lambda_: float,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> tuple:
    """Primal gradient descent for LASSO linear regression.

    Args:
        X (ndarray): The feature matrix.
        y (ndarray): The target values.
        lambda_ (float): The regularization parameter.
        learning_rate (float, optional): The learning rate. Defaults to 0.01.
        max_iterations (int, optional): The maximum number of iterations. Defaults to 1000.
        tol (float, optional): The tolerance for convergence. Defaults to 1e-6.

    Returns:
        tuple: A tuple containing the optimized coefficients and the loss history.
    """
    n, p = X.shape
    beta = np.zeros(p)  # Initialize coefficients
    loss_history = []

    for _ in range(max_iter):
        gradient = lasso_gradient(X, y, beta, lambda_)
        beta -= learning_rate * gradient

        # Compute current loss
        loss = lasso_objective(X, y, beta, lambda_)
        loss_history.append(loss)

        # Check convergence
        if np.linalg.norm(gradient, ord=np.inf) < tol:
            break

    return beta, loss_history


def primal_gradient_new(A, b, x0, tol, max_iter=10000):
    """
    Implement the primal gradient method to solve Ax = b, tracking both gap history and loss history.

    Args:
        A (np.ndarray): Coefficient matrix.
        b (np.ndarray): Right-hand side vector.
        x0 (np.ndarray): Initial guess for the solution.
        tol (float): Tolerance for stopping criterion based on relative gap.
        max_iter (int): Maximum number of iterations.

    Returns:
        Tuple containing:
        - np.ndarray: The solution vector x.
        - int: Total number of iterations performed.
        - float: Final relative gap.
        - list: History of the relative gap during iterations.
        - list: History of the loss (norm of residual squared) during iterations.
        - list: History of CPU time spent during iterations.
    """
    m, n = A.shape
    x = x0
    r = np.dot(A, x) - b  # initial residual
    initial_loss = np.linalg.norm(r)**2
    gap_history = []
    loss_history = [initial_loss]
    cpu_time_history = []

    for k in range(max_iter):
        start_time = time.process_time()
        gradient = 2 * np.dot(A.T, (np.dot(A, x) - b))
        step_size = 0.1 / np.linalg.norm(gradient)  # simple step size rule
        x -= step_size * gradient
        r = np.dot(A, x) - b
        current_loss = np.linalg.norm(r)**2

        relative_gap = current_loss / initial_loss
        loss_history.append(current_loss)
        gap_history.append(relative_gap)
        end_time = time.process_time()
        cpu_time_history.append(end_time - start_time)

        if relative_gap < tol:
            break

    return x, k, relative_gap, gap_history, loss_history, cpu_time_history
