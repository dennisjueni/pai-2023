"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Matern,
    ConstantKernel,
    WhiteKernel,
    RBF,
    DotProduct
)
from scipy.stats import norm
import matplotlib.pyplot as plt

# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SAcc

# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo:
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        self.data = {"X": [], "y_f": [], "y_v": []}

        obj_noise_kernel = WhiteKernel(noise_level=0.15)
        obj_kernel = RBF(length_scale=0.5, length_scale_bounds="fixed") + Matern(length_scale=0.5, length_scale_bounds="fixed", nu=2.5)
        obj_variance_kernel = ConstantKernel(constant_value=0.5, constant_value_bounds="fixed")
        
        total_obj_kernel = obj_variance_kernel * obj_kernel + obj_noise_kernel
        self.obj_gp = GaussianProcessRegressor(kernel=total_obj_kernel)

        #constr_noise_kernel = WhiteKernel(noise_level=0.0001)
        constr_kernel = Matern(length_scale=0.5, length_scale_bounds="fixed") + RBF(length_scale=1, length_scale_bounds="fixed")
        constr_variance_kernel = ConstantKernel(constant_value=np.sqrt(2))
        constr_linear_kernel = DotProduct()

        total_constr_kernel = constr_variance_kernel * constr_kernel + constr_linear_kernel
        self.constr_gp = GaussianProcessRegressor(kernel=total_constr_kernel)

        self.func_type = "ucb"

        # General Hyperparameter
        self.lam = 1e10

        # UCB Hyperparameter
        self.beta = 2.0

        # EI Hyperparameter
        self.xi = 0.01

    def next_recommendation(self):
        """
        Recommend the net input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        x_opt = self.optimize_acquisition_function()

        return x_opt

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * np.random.rand(
                DOMAIN.shape[0]
            )
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN, approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def exponential_transformation(self, p, base=0.0001):
        return base ** (1 - p)

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)

        if self.func_type == "ucb":
            mean_f, std_f = self.obj_gp.predict(x, return_std=True)  # type: ignore
            ucb = mean_f + self.beta * std_f

            mean_v, std_v = self.constr_gp.predict(x, return_std=True)  # type: ignore

            if mean_v + 2*std_v >= SAFETY_THRESHOLD:
                return ucb - self.lam * (mean_v + 2*std_v)
            else:
                return ucb
            
        elif self.func_type == "ei":
            
            mean_f, std_f = self.obj_gp.predict(x, return_std=True)  # type: ignore
            current_best_f = max(self.data["y_f"])
            z = (mean_f - current_best_f - self.xi) / std_f
            ei = (mean_f - current_best_f - self.xi) * norm.cdf(z) + std_f * norm.pdf(z)

            mean_v, std_v = self.constr_gp.predict(x, return_std=True)  # type: ignore


            if mean_v + 2*std_v >= SAFETY_THRESHOLD:
                return ei - self.lam * max(mean_v + 2*std_v - 4, 0)
            else:
                return ei

    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # Append the new observation to your data storage
        self.data["X"].append(x)
        self.data["y_f"].append(f)
        self.data["y_v"].append(v)

        # Convert lists to numpy arrays for model fitting
        X = np.array(self.data["X"]).reshape(-1, 1)
        y_f = np.array(self.data["y_f"])
        y_v = np.array(self.data["y_v"])

        # Update the surrogate models with the new data
        self.obj_gp.fit(X, y_f)
        self.constr_gp.fit(X, y_v)

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # Extract the data
        X = np.array(self.data["X"])
        y_f = np.array(self.data["y_f"])
        y_v = np.array(self.data["y_v"])

        # Filter points that satisfy the constraint
        valid_indices = np.where(y_v < SAFETY_THRESHOLD)[0]
        valid_f = y_f[valid_indices]

        # If no points satisfy the constraint, return None
        if len(valid_f) == 0:
            return None

        # Find the index of the maximum f among valid points
        max_f_index = valid_indices[np.argmax(valid_f)]

        # Return the corresponding x value
        return X[max_f_index]

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        pass

# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])

def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return -np.linalg.norm(x - mid_point, 2)

    # global_optimum = -((x - 5) ** 2)

    # # Multiple local optima introduced by a sinusoidal component
    # local_optima = 10 * np.sin(2 * np.pi * 0.2 * x)

    # # Combining the two components
    # return global_optimum + local_optima

def v(x: float):
    """Dummy SA"""
    return 2.0

def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    # np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init

def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    NUM_ITERATIONS = 1

    # PLOTTING
    x_values = np.linspace(*DOMAIN[0], 4000)[:, None]
    f_values = [f(x) for x in x_values]
    v_values = [v(x) for x in x_values]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.ion()
    plt.plot(x_values, f_values, label="f(x)", color="blue")
    plt.plot(x_values, v_values, label="v(x)", color="red")
    plt.xlabel("x")
    plt.ylabel("Function values")
    plt.title("Comparison of f(x) and v(x)")
    plt.legend()
    plt.show()

    for _ in range(NUM_ITERATIONS):
        # Add initial safe point
        x_init = get_initial_safe_point()

        print(f"Initial safe point: {x_init}")
        obj_val = f(x_init)
        cost_val = v(x_init)
        agent.add_data_point(x_init, obj_val, cost_val)

        # Loop until budget is exhausted
        for j in range(20):
            # Get next recommendation
            x = agent.next_recommendation()

            # Check for valid shape
            # assert x.shape == (1, DOMAIN.shape[0]), (
            #     f"The function next recommendation must return a numpy array of "
            #     f"shape (1, {DOMAIN.shape[0]})"
            # )

            # Obtain objective and constraint observation
            obj_val = f(x) #+ np.random.randn()
            cost_val = v(x) #+ np.random.randn()
            agent.add_data_point(x, obj_val, cost_val)

            print(f"Iteration {j + 1}: x = {x}, f(x) = {obj_val}, v(x) = {cost_val}")

            plt.scatter(x, obj_val, color="green", marker="x", s=100)  # s is the size
            plt.scatter(x, cost_val, color="pink", marker="x", s=100)  # s is the size

            plt.draw()
            plt.pause(0.1)  # Pause to update the plot

        # Validate solution
        solution = agent.get_solution()
        assert check_in_domain(solution), (
            f"The function get solution must return a point within the"
            f"DOMAIN, {solution} returned instead"
        )

        # Compute regret
        regret = 0 - f(solution)

        print(
            f"Optimal value: 0\nProposed solution {solution}\nSolution value "
            f"{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n"
        )
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()