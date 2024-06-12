import time
import numpy as np
# from .common import lasso_objective, lasso_gradient


class NesterovAcceleratedGradientMethod:
    """Nesterov Accelerated Gradient Method for LASSO regression.

    Args:
        A (np.ndarray): Coefficient matrix.
        b (np.ndarray): Right-hand side vector.
        L0 (float): Initial value for the Lipschitz constant.
        v0 (np.ndarray): Initial guess for the solution.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for stopping criterion based on relative gap.

    Returns:
        Tuple containing:
        - np.ndarray: The solution vector x.
        - int: Total number of iterations performed.
        - float: Final relative gap.
        - List[float]: History of relative gaps.
        - List[float]: History of loss values.
        - List[float]: History of CPU times.
    """
    def __init__(self, A, b, L0, v0, max_iter=10000, tol=1e-6):
        self.A = A
        self.b = b
        self.L = L0
        self.v = v0
        self.max_iter = max_iter
        self.tol = tol
        self.gap_history = []
        self.loss_history = []
        self.cpu_time_history = []

    def gradient_f(self, x):
        """Compute the gradient of the objective function at x."""
        return self.A.T @ (self.A @ x - self.b)

    def f(self, x):
        """Compute the objective function at x."""
        return np.linalg.norm(self.A @ x - self.b)**2 / 2

    def nag_step(self, y, L):
        """Compute the Nesterov accelerated gradient step at y."""
        grad = self.gradient_f(y)
        return y - grad / L

    def compute_steps(self):
        """Run the Nesterov accelerated gradient method."""
        v_prev = self.v
        initial_loss = self.f(self.v)

        for k in range(self.max_iter):
            start_time = time.process_time()

            y = self.v + k / (k + 3) * (self.v - v_prev)
            T_L_y = self.nag_step(y, self.L)

            while np.linalg.norm(self.gradient_f(T_L_y) - self.gradient_f(y)) > self.L * np.linalg.norm(T_L_y - y):
                self.L *= 1.1  # Increase L and recompute
                T_L_y = self.nag_step(y, self.L)

            v_prev = self.v
            self.v = T_L_y

            current_loss = self.f(self.v)
            current_gap = current_loss / initial_loss

            end_time = time.process_time()

            self.gap_history.append(current_gap)
            self.loss_history.append(current_loss)
            self.cpu_time_history.append(end_time - start_time)

            if current_gap < self.tol:
                break

        return self.v, k, current_gap, self.gap_history, self.loss_history, self.cpu_time_history
