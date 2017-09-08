import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular as solvetri

import kernels
import argparse

def validate(xtrain, ytrain, xtest, ypred, ypredsigma, args):
    """ Compare result to GP from sklearn module. """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

    kernel = C(args.signal_std**2, (1e-3, 1e3)) * RBF(args.length_scale, (1e-2, 1e2))    
    gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, alpha=args.noise_std)
    gp.fit(xtrain.reshape(-1, 1), ytrain)
    ypredhat, ypredsigmahat = gp.predict(xtest.reshape(-1,1), return_std=True)

    np.testing.assert_allclose(ypred, ypredhat, atol=0.01)
    np.testing.assert_allclose(ypredsigma, ypredsigmahat, atol=0.01)

if __name__ == '__main__':

    np.random.seed(123)

    parser = argparse.ArgumentParser(description='Posterior demo')
    parser.add_argument('--length-scale', type=float, default=1.0, help='Kernel length scale')
    parser.add_argument('--signal-std', type=float, default=2.0, help='Signal standard deviation')
    parser.add_argument('--noise-std', type=float, default=1e-10, help='Standard deviation of assumed noise in training data')
    parser.add_argument('--true-noise-std', type=float, default=0., help='Noise level in training data')
    args = parser.parse_args()

    # True function
    f = lambda x: x * np.sin(x)

    # Train points
    xtrain = np.array([1., 3., 5., 6., 7., 8.])
    ytrain = f(xtrain) + np.random.randn(xtrain.shape[0]) * args.true_noise_std**2

    # Test points we'd like to predict at
    xtest = np.linspace(0., 10, 50, endpoint=True)
    
    # Blocks of the covariance matrix required
    K = kernels.se_kernel(xtrain, xtrain, args.signal_std, args.length_scale) + np.eye(xtrain.shape[0]) * args.noise_std**2
    Ks = kernels.se_kernel(xtrain, xtest, args.signal_std, args.length_scale)
    Kss = kernels.se_kernel(xtest, xtest, args.signal_std, args.length_scale)

    # Posterior mean and covariance
    L = np.linalg.cholesky(K)
    alpha = solvetri(L.T, solvetri(L, ytrain, check_finite=False, lower=True), check_finite=False)
    pmu = Ks.T.dot(alpha)
    beta = solvetri(L, Ks, lower=True)
    pcov = Kss - beta.T.dot(beta)
    diag = np.copy(np.diag(pcov))
    diag[diag < 0.] = 0. # numerical issues    
    sigma = np.sqrt(diag)    
        
    # Note, there is a faster method for obtaining the standard deviation 
    # of the predictive distribution at the test points that avoids computing the entire covariance matrix:
    #L_inv = solvetri(L.T, np.eye(L.shape[0]))
    #K_inv = L_inv.dot(L_inv.T)
    #sigma = np.copy(np.diag(Kss))
    #sigma -= np.einsum("ij,ij->i", np.dot(Ks.T, K_inv), Ks.T)
    #sigma = np.sqrt(sigma)

    # Verify results
    validate(xtrain, ytrain, xtest, pmu, sigma, args)
    
    fig, ax = plt.subplots()
    ax.set_title('Gaussian process posterior\nlength-scale={:.1f}, signal-std={:.1f}, noise-std={:.1f}'.format(args.length_scale, args.signal_std, args.noise_std))    
    ax.plot(xtrain, ytrain, 'k+', ms=10, lw=5, label='Train points')
    ax.plot(xtest, f(xtest), 'k-', label='True function')
    ax.plot(xtest, pmu, 'r--', label='Posterior mean')
    ax.fill(
        np.concatenate([xtest, xtest[::-1]]),
        np.concatenate([pmu - 1.9600 * sigma, (pmu + 1.9600 * sigma)[::-1]]),
        alpha=.2, fc='b', ec='None', label='95% confidence interval')
    
    # Draw samples (functions) from the posterior
    # ytest = np.random.multivariate_normal(pmu, pcov, size=10)
    # ax.plot(xtest.reshape(-1, 1), ytest.T, lw=0.5)    
    ax.set_ylim(-10, 20)
    plt.legend(loc='upper left')    
    plt.savefig('GP_posterior_lscale{:.1f}_signalstd{:.1f}_noisestd{:.1f}.png'.format(args.length_scale, args.signal_std, args.noise_std), dpi=300)
    plt.show()