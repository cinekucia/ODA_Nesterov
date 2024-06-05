import numpy as np
from .common import lasso_objective, lasso_gradient


def primal_gradient(
    X: np.ndarray,
    y: np.ndarray,
    lambda_: float,
    learning_rate: float = 0.01,
    max_iterations: int = 1000,
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

    for _ in range(max_iterations):
        gradient = lasso_gradient(X, y, beta, lambda_)
        beta -= learning_rate * gradient

        # Compute current loss
        loss = lasso_objective(X, y, beta, lambda_)
        loss_history.append(loss)

        # Check convergence
        if np.linalg.norm(gradient, ord=np.inf) < tol:
            break

    return beta, loss_history
