from scipy.interpolate import interp1d

def interp(x, y, xnew):
    """Cubic spline interpolation of data."""
    f = interp1d(x, y, kind='cubic')
    return f(xnew)