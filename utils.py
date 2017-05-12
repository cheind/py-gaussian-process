from scipy.interpolate import interp1d

def interp(x, y, xnew):
    """Cubic spline interpolation of data."""
    f = interp1d(x, y, kind='cubic', axis=0)
    return f(xnew)