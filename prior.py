import numpy as np
import matplotlib.pyplot as plt

import kernels
import utils

if __name__ == '__main__':

    np.random.seed(123)
    x = np.linspace(-5., 5, 20, endpoint=True)

    cov = kernels.se_kernel(x, x, sigma=1., length=2.)
    y = np.random.multivariate_normal(np.zeros(x.shape[0]), cov, size=10)

    fig, ax = plt.subplots()
    ax.set_title('Ten samples from a GP prior with zero mean')
    xdraw = np.linspace(-5, 5, 100, endpoint=True)
    ax.plot(xdraw.reshape(-1, 1), utils.interp(x, y.T, xdraw))
    plt.tight_layout()
    plt.show()