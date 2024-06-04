
import numpy as np
from matplotlib import pyplot as plt

def triangular_method(hist, strictness=0.0, debug=False):
    """
    This method draws a line from the peak to the endpoint, and selects the threshold as the x value which maximizes the length of a perpendicular line drawn to the original line. 
    :param hist: (256,) array of values
    :param strictness: parameter denoting how strict or lenient the algorithm should be [-1.0, 1.0]
    """
    hist = np.squeeze(hist)
    hist /= np.sum(hist)
    hist[0] = 0
    hist /= np.max(hist)
    hist *= hist.shape[0]-1

    # parameter handling
    strictness = np.clip(strictness, -1.0, 1.0)
    peak_percentage = 1.0
    endpoint = hist.shape[0]-1
    if strictness > 0.0:
        peak_percentage -= strictness
    else:
        endpoint = int(round(endpoint * (1 + strictness)))

    # get the endpoint
    x1 = endpoint
    y1 = hist[endpoint]

    # get the peak value
    x0 = np.argmax(hist[:x1])
    y0 = hist[x0]

    # move the peak value down the histogram based on the scaler
    x0 = np.argmax(hist[x0:x1] <= (y0 * peak_percentage)) + x0
    y0 = hist[x0]

    # get the line
    m = (y1 - y0) / (x1 - x0)
    b = y1 - m * x1
    A = [m, -1, b]

    if debug:
        fig, ax = plt.subplots(1,1)
        ax.plot(hist, color='black')
        ax.plot([x0, x1], [y0, y1], color='blue')

    xi = np.arange(x0, x1)
    yi = hist[xi]

    mp = -1/m
    bp = yi - mp * xi
    B = [mp, -1, bp]

    xj = (A[1]*B[2] - B[1]*A[2]) / (A[0]*B[1] - B[0]*A[1])
    yj = (A[2]*B[0] - B[2]*A[0]) / (A[0]*B[1] - B[0]*A[1])

    dist = np.sqrt(((xi-xj)**2 + (yi-yj)**2))
    idx = np.argmax(dist)
    thresh = idx + x0

    if debug:
        plot_idx = np.unique(np.rint(np.geomspace(x0, x1-1)).astype(int)) - x0
        ax.plot([xi[plot_idx], xj[plot_idx]], [yi[plot_idx], yj[plot_idx]], '--', color='red')
        ax.plot([xi[idx], xj[idx]], [yi[idx], yj[idx]], color='green')
        ax.set_aspect('equal')

    return thresh