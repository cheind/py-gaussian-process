import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular as solvetri

import kernels

if __name__ == '__main__':

    np.random.seed(123)

    # Parameters
    ntrain = 4
    klength = 1.0
    ksigma = 2.    
    noise = 0.2

    # True function
    f = lambda x: np.polyval([0.1, -0.2, 0.1, 2.], x)

    # Train points
    xtrain = np.random.uniform(-4, 4, ntrain)
    ytrain = f(xtrain) + np.random.randn(xtrain.shape[0]) * 0.2**2

    # Test points we'd like to predict at
    xtest = np.linspace(-5., 5, 50, endpoint=True)

    # Blocks of the covariance matrix required
    K = kernels.se_kernel(xtrain, xtrain, ksigma, klength) + np.eye(xtrain.shape[0]) * noise**2
    Ks = kernels.se_kernel(xtrain, xtest, ksigma, klength)
    Kss = kernels.se_kernel(xtest, xtest, ksigma, klength)

    # Posterior mean and covariance
    L = np.linalg.cholesky(K)
    alpha = solvetri(L.T, solvetri(L, ytrain, check_finite=False, lower=True), check_finite=False)
    pmu = Ks.T.dot(alpha)
    beta = solvetri(L, Ks, lower=True)
    pcov = Kss - beta.T.dot(beta)
    sigma = np.sqrt(np.diag(pcov)) 
    
    # Note, there is a faster method for obtaining the std of the predictive distribution at the test
    # points that avoids computing the entire covariance matrix:
    #L_inv = solvetri(L.T, np.eye(L.shape[0]))
    #K_inv = L_inv.dot(L_inv.T)
    #sigma = np.copy(np.diag(Kss))
    #sigma -= np.einsum("ij,ij->i", np.dot(Ks.T, K_inv), Ks.T)
    #sigma = np.sqrt(sigma)

    # Draw samples (functions) from the posterior
    ytest = np.random.multivariate_normal(pmu, pcov, size=10)

    fig, ax = plt.subplots()
    ax.set_title('Gaussian process posterior\nklength={:.1f}, ksigma={:.1f}, noise={:.1f}'.format(klength, ksigma, noise))    
    ax.plot(xtest.reshape(-1, 1), ytest.T, lw=0.5)
    ax.plot(xtrain, ytrain, 'k+', ms=20, lw=5, label='Train points')
    ax.plot(xtest, f(xtest), 'k-', label='True function')
    ax.plot(xtest, pmu, 'r--', label='Posterior mean')
    ax.fill(
        np.concatenate([xtest, xtest[::-1]]),
        np.concatenate([pmu - 1.9600 * sigma, (pmu + 1.9600 * sigma)[::-1]]),
        alpha=.2, fc='b', ec='None', label='95% confidence interval')

    plt.legend(loc=4)    
    plt.show()