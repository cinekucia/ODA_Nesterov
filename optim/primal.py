import time
import numpy as np
# from .common import lasso_objective, lasso_gradient


class PrimalGradientMethod:
    """Primal Gradient Method for solving Ax = b using least squares.

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
        - List[float]: History of relative gaps.
        - List[float]: History of loss values.
        - List[float]: History of CPU times.
    """
    def __init__(self, A, b, x0, tol, max_iter=10000):
        self.A = A
        self.b = b
        self.x = x0
        self.tol = tol
        self.max_iter = max_iter
        self.gap_history = []
        self.loss_history = []
        self.cpu_time_history = []

    def gradient_f(self, x):
        """Compute the gradient of the objective function at x."""
        return 2 * np.dot(self.A.T, (np.dot(self.A, x) - self.b))

    def f(self, x):
        """Compute the objective function at x."""
        # @Bartosz7: I added '/ 2' to the end of the line below. 
        return np.linalg.norm(np.dot(self.A, x) - self.b)**2 / 2

    def compute_steps(self):
        """Compute the gradient steps."""
        initial_loss = self.f(self.x)
        self.loss_history.append(initial_loss)

        for k in range(self.max_iter):
            start_time = time.process_time()
            gradient = self.gradient_f(self.x)
            step_size = 0.1 / (np.linalg.norm(gradient) if np.linalg.norm(gradient) != 0 else 1e-16)
            self.x -= step_size * gradient
            current_loss = self.f(self.x)
            relative_gap = current_loss / initial_loss

            self.loss_history.append(current_loss)
            self.gap_history.append(relative_gap)
            end_time = time.process_time()
            self.cpu_time_history.append(end_time - start_time)

            if relative_gap < self.tol:
                break

        return self.x, k, relative_gap, self.gap_history, self.loss_history, self.cpu_time_history
