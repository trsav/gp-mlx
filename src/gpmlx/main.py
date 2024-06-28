from typing import Any, List, Union

import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.optimizers as optim
from mlx.core import vmap

ArrayLike = Union[List, mx.array]


class Dataset:
    def __init__(self, x: Any, y: Any):
        # convert to mx.array
        if not isinstance(x, mx.array):
            x = mx.array(x)
        if not isinstance(y, mx.array):
            y = mx.array(y)

        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same length")

        # ensure arrays are 2D
        if len(x.shape) == 1:
            print(
                f"Interpreting input data as {len(x)} samples of 1D data. Reshaping to 2D. If this is not the desired behavior, please reshape the data before passing it to the Dataset class."
            )
            x = x.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        self.x = x
        self.y = y

    def __add__(self, other: "Dataset") -> "Dataset":
        return Dataset(
            mx.concatenate((self.x, other.x), axis=0),
            mx.concatenate((self.y, other.y), axis=0),
        )


class Kernel:
    def __init__(self):
        pass


class MeanFunction:
    def __init__(self):
        pass


class Constant(MeanFunction):
    def __init__(self, c: float):
        super().__init__()
        self.c = c

    def __call__(self, x: mx.array) -> mx.array:
        return self.c * mx.ones(x.shape)


class Linear(MeanFunction):
    def __init__(self, w: float, b: float):
        super().__init__()
        self.w = w
        self.b = b

    def __call__(self, x: mx.array) -> mx.array:
        return self.w * x + self.b


class Zero(MeanFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, x: mx.array) -> mx.array:
        return mx.zeros(x.shape)


class Matern52(Kernel):
    def __init__(self, l: float, sigma_f: float, sigma_n: float):
        super().__init__()
        self.l = l
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n

    def __call__(self, x1: mx.array, x2: mx.array) -> mx.array:
        return (
            self.sigma_f**2
            * (
                1
                + mx.sqrt(5) * mx.linalg.norm(x1 - x2) / self.l
                + 5 * mx.linalg.norm(x1 - x2) ** 2 / (3 * self.l**2)
            )
            * mx.exp(-mx.sqrt(5) * mx.linalg.norm(x1 - x2) / self.l)
        )


class GaussianProcess:
    def __init__(self, mean: MeanFunction, kernel: Kernel, D: Dataset):
        self.mean = mean
        self.kernel = kernel
        self.D = D

    def covariance_matrix(self, X1: mx.array, X2: mx.array) -> mx.array:
        cov_matrix = vmap(
            lambda x1: vmap(lambda x2: self.kernel(x1, x2), in_axes=0)(X1), in_axes=0
        )(X2)
        return cov_matrix

    def marginal_log_likelihood(self) -> float:
        K = self.covariance_matrix(
            self.D.x, self.D.x
        ) + self.kernel.sigma_n**2 * mx.eye(self.D.x.shape[0])
        U, S, Vt = mx.linalg.svd(K, stream=mx.cpu)
        K_det = mx.prod(S)
        log_K_det = mx.log(K_det)
        K_inv = mx.linalg.inv(K, stream=mx.cpu)
        return (
            -0.5
            * (
                (self.D.y - self.mean(self.D.x)).T
                @ K_inv
                @ (self.D.y - self.mean(self.D.x))
            )
            - 0.5 * log_K_det
            - self.D.y.shape[0] / 2 * mx.log(2 * mx.pi)
        )

    def optimize_hyperparameters(self):
        def nll(params: mx.array) -> mx.array:
            self.mean.c = params[0]
            self.kernel.l = params[1]
            self.kernel.sigma_f = params[2]
            self.kernel.sigma_n = params[3]
            return -self.marginal_log_likelihood()[0, 0]

        random_grid = mx.random.uniform(0, 3, (5000, 4))
        nll_list = vmap(nll)(random_grid)
        best_params = random_grid[mx.argmin(nll_list)]
        print(f"Best parameters: {best_params}")

        self.mean.c = best_params[0]
        self.kernel.l = best_params[1]
        self.kernel.sigma_f = best_params[2]
        self.kernel.sigma_n = best_params[3]

    def __call__(self, x: mx.array) -> mx.array:
        K = self.covariance_matrix(
            self.D.x, self.D.x
        ) + self.kernel.sigma_n**2 * mx.eye(self.D.x.shape[0])

        K_inv = mx.linalg.inv(K, stream=mx.cpu)  # waiting for this to be on GPU!

        K_star = self.covariance_matrix(self.D.x, x).T
        K_star_star = self.covariance_matrix(x, x)
        var = K_star_star - K_star.T @ K_inv @ K_star

        std = mx.sqrt(mx.diag(var))
        f = self.mean(x) + K_star.T @ K_inv @ (self.D.y - self.mean(self.D.x))

        return f, std


# n = 100
# x = mx.linspace(0,5,n).reshape(-1,1)
# y = mx.sin(x) + mx.array([mx.random.uniform() for _ in range(n)]).reshape(-1,1)

# D = Dataset(x, y)

# k = Matern52(1,1,1)
# m = Constant(0)
# GP = GaussianProcess(m,k,D)
# GP.optimize_hyperparameters()


# x_test = mx.linspace(0,5,300).reshape(-1,1)
# y_test,std_test = GP(x_test)

# plt.figure()
# plt.scatter(D.x,D.y,c='tab:blue')
# plt.plot(x_test,y_test,c='tab:red',alpha=0.75)
# plt.fill_between(x_test.flatten(), y_test.flatten()-2*std_test.flatten(), y_test.flatten()+2*std_test.flatten(), color='tab:red', alpha=0.2)
# plt.show()
