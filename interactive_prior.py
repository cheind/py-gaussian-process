
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import kernels
import utils

def sample_prior(x, kernel_sigma, kernel_length):
    cov = kernels.se_kernel(x, x, sigma=kernel_sigma, length=kernel_length)
    return np.random.multivariate_normal(np.zeros(x.shape[0]), cov)

if __name__ == '__main__':

    fig, ax = plt.subplots()
    ax.set_title('Interactive GP Prior')
    plt.subplots_adjust(bottom=0.25)

    axsigma = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='green')
    axlength = plt.axes([0.2, 0.15, 0.65, 0.03], facecolor='green')
    ssigma = Slider(axsigma, 'Sigma', 0.01, 3., valinit=1)
    slength = Slider(axlength, 'Length', 0.1, 3.0, valinit=1)
    
    np.random.seed(123)
    x = np.linspace(-5., 5, 20, endpoint=True)
    xdraw = np.linspace(-5, 5, 100, endpoint=True)
    y = sample_prior(x, 1., 1.)
    l, = ax.plot(xdraw, utils.interp(x, y, xdraw))

    def update(val):
        np.random.seed(123)
        y = sample_prior(x, ssigma.val, slength.val)
        l.set_ydata(utils.interp(x, y, xdraw))
        fig.canvas.draw_idle()

    ssigma.on_changed(update)
    slength.on_changed(update)

    plt.show()
