import numpy as np
from scipy.spatial.distance import cdist

def se_kernel(x, y, sigma=1., length=1.):
    """Squared exponential kernel."""
    d = cdist(x.reshape(-1,1), y.reshape(-1,1), metric='seuclidean')
    f = -1. / (2. * length**2)
    return sigma**2 * np.exp(d*f).reshape(x.shape[0], y.shape[0])