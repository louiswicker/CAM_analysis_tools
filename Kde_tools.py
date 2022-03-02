import numpy as np 
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

def kernal_density_estimate(var1, var2):
    """
    Compute 2D Gaussian Kernal Density Estimate.
    
    Parameters:
    ----------------------
        var1, var2 : 1D np.arrays 
            Input data to compute the 2D KDE over.
    
    Returns:
     ----------------------
        X, Y : grids the KDE was computed over;  
                 Computed from the input variables.
                 
        kde_2d : 2D kernal density estimate 
    
    """
    var1_min, var1_max = var1.min(), var1.max()
    var2_min, var2_max = var2.min(), var2.max()
    
    X,Y = np.mgrid[var1_min:var1_max:100j, var2_min:var2_max:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([var1,var2])
    kernel = gaussian_kde(values, bw_method='scott')
    kde_2d = np.reshape(kernel(positions).T, X.shape)

    return X, Y, kde_2d

def plot_kde_scatter(ax,
                     var1,
                     var2,
                     percentiles = [75.0, 90., 95., 97.5],
                     linewidths = [ 0.85, 1.0, 1.25, 1.75],
                     cmap = 'jet'
                    ):
    """
    Plot 2D Kernal Density Estimate (KDE) Contours.
    Contours are based on the percentile of density 
    (e.g., 90% indicates 90% of the data 
    falls within that contour)
    
    Parameters:
    -----------------
        ax : Matplotlib.pyplot Axes object to plot over.
        
        x,y : 1D np.arrays to compute the 2D KDE over.
        
    
    """
    X,Y, kde_2d = kernal_density_estimate(var1, var2)
    levels = np.percentile(kde_2d, percentiles)

    ax.scatter(var1, var2, s=5)
    cs = ax.contour(
            X,
            Y,
            kde_2d,
            levels=levels,
            cmap = cmap,
            linewidths=linewidths,
            alpha=1.0,
        )
    fmt = {}
    for l, s in zip(cs.levels, percentiles[::-1]):
        fmt[l] = f'{int(s)}%'
    ax.clabel(cs, cs.levels, inline=True, fontsize=7, fmt=fmt)
    return cs
    

if __name__ == "__main__":

    # Example Usage 
    f, ax = plt.subplots(dpi=300, figsize=(5,5))
    n_samples = 10000
    m1 = np.random.normal(size=n_samples)
    m2 = np.random.normal(scale=0.5, size=n_samples)
    x= m1+m2 
    y = m1-m2

    cs = plot_kde_scatter(ax, x, y)