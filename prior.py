
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def se_kernel(x, y, sigma=1., length=1.):
    d = cdist(x.reshape(-1,1), y.reshape(-1,1), metric='seuclidean')
    f = -1 / (2. * length**2)
    return sigma**2 * np.exp(d*f).reshape(x.shape[0], y.shape[0])

if __name__ == '__main__':

    np.random.seed(123)

    x = np.arange(-5, 5, 0.2, dtype=float)
    #y = np.polyval([1.5, -1.5, 0.5, 0.5], x)




    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)

    cov = se_kernel(x, x, sigma=1, length=1) + np.eye(x.shape[0])*1e-15
    y = np.random.multivariate_normal(np.zeros(x.shape[0]), cov)
    l, = ax.plot(x, y)

    axsigma = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='green')
    axlength = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='green')
    ssigma = Slider(axsigma, 'Sigma', 0.1, 10.0, valinit=1)
    slength = Slider(axlength, 'Length', 0.1, 10.0, valinit=1)
    
    #samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)

    def update(val):
        np.random.seed(123)
        cov = se_kernel(x, x, sigma=ssigma.val, length=slength.val) + np.eye(x.shape[0])*1e-15
        y = np.random.multivariate_normal(np.zeros(x.shape[0]), cov, size=1)
        l.set_ydata(y)
        fig.canvas.draw_idle()

    ssigma.on_changed(update)
    slength.on_changed(update)

    plt.show()
