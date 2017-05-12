import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular as solvetri

import kernels
import utils

if __name__ == '__main__':

    np.random.seed(123)

    # True function
    f = lambda x: np.polyval([0.5, -0.2, 0.2, 5.], x)

    #f = lambda x: np.sin(0.9*x).flatten()

    # Train points / no noise
    xtrain = np.array([-2, -1.8, 0.5, 1.5])
    #xtrain = np.random.uniform(-5, 5, size=10)
    ytrain = f(xtrain)

    # Test points
    xtest = np.linspace(-5., 5, 50, endpoint=True)

    length = 0.5
    sigma = 1.

    K = kernels.se_kernel(xtrain, xtrain, sigma, length)
    Ks = kernels.se_kernel(xtrain, xtest, sigma, length)

    # Posterior mean
    L = np.linalg.cholesky(K)
    alpha = solvetri(L.T, solvetri(L, ytrain, check_finite=False, lower=True), check_finite=False)
    pmu = Ks.T.dot(alpha)

    # Posterior covariance
    Kss = kernels.se_kernel(xtest, xtest, sigma, length)
    beta = solvetri(L, Ks, lower=True)
    pcov = Kss - beta.T.dot(beta)

    xdraw = np.linspace(-5, 5, 100, endpoint=True)
    ydraw = np.random.multivariate_normal(pmu, pcov, size=10)

    fig, ax = plt.subplots()
    ax.set_title('Ten samples from a GP posterior')
    ax.plot(xdraw.reshape(-1, 1), utils.interp(xtest, ydraw.T, xdraw))
    ax.plot(xtrain, ytrain, 'k+', ms=20, lw=5)
    ax.plot(xtest, pmu, 'r--')
    #plt.axis([-5, 5, -3, 3])
    plt.tight_layout()
    plt.show()