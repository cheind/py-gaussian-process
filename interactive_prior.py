
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.interpolate import interp1d

def se_kernel(x, y, sigma=1., length=1.):
    d = cdist(x.reshape(-1,1), y.reshape(-1,1), metric='seuclidean')
    f = -1. / (2. * length**2)
    return sigma**2 * np.exp(d*f).reshape(x.shape[0], y.shape[0])

def sample(x, sigma, length):
    cov = se_kernel(x, x, sigma=sigma, length=length)
    return np.random.multivariate_normal(np.zeros(x.shape[0]), cov)

def interp(x, y, xnew):
    f = interp1d(x, y, kind='cubic')
    return f(xnew)

if __name__ == '__main__':

    fig, ax = plt.subplots()
    ax.set_title('Interactive GP Prior')
    plt.subplots_adjust(bottom=0.25)

    axsigma = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='green')
    axlength = plt.axes([0.2, 0.15, 0.65, 0.03], facecolor='green')
    ssigma = Slider(axsigma, 'Sigma', 0.01, 3., valinit=1)
    slength = Slider(axlength, 'Length', 0.1, 3.0, valinit=1)
    
    np.random.seed(123)
    x = np.arange(-5, 5, 0.2, dtype=float)
    xdraw = np.arange(-5, x.max(), 0.01, dtype=float)
    y = sample(x, 1., 1.)
    l, = ax.plot(xdraw, interp(x, y, xdraw))

    def update(val):
        np.random.seed(123)
        y = sample(x, sigma=ssigma.val, length=slength.val)
        l.set_ydata(interp(x, y, xdraw))
        fig.canvas.draw_idle()

    ssigma.on_changed(update)
    slength.on_changed(update)

    plt.show()
