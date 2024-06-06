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


# FIXME: gap is not behaving as expected, increasing instead of decreasing
def primal_gradient_with_gap(
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
        target_gap (float, optional): The target GAP to achieve. Defaults to 0.0.

    Returns:
        tuple: A tuple containing the optimized coefficients, the loss history,
               and the GAP at each iteration.
    """
    n, p = X.shape
    beta = np.zeros(p)  # Initialize coefficients
    loss_history = []
    gap_history = []
    initial_loss = lasso_objective(X, y, beta, lambda_)  # initial residual

    for _ in range(max_iterations):
        gradient = lasso_gradient(X, y, beta, lambda_)
        beta -= learning_rate * gradient

        # Compute current loss
        loss = lasso_objective(X, y, beta, lambda_)
        loss_history.append(loss)

        # Calculate GAP
        gap = (initial_loss - loss) / initial_loss
        print(gap, np.log2(gap))
        gap_history.append(gap)

        # Check convergence
        if np.linalg.norm(gradient, ord=np.inf) < tol:
            break

    # change gap to be a log2 scale
    # gap_history = np.log2(gap_history)

    return beta, loss_history, gap_history
