import os, sys, argparse, cmath
import pylab as pl
import numpy as np
from scipy import integrate
from matplotlib import colors, ticker
import scipy.stats as stats
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------------
# 2D Spectra from the fourier spectrum package from jfrob27's pywaven package

def get_spectra2D_CWT(fld, print_info=True, zeromean=True, **kwargs):
    """
    Returns 1D power spectra from a 2D field using continuus wavelets
        
    Input:  2D floating pont array
    
    Returns:  kvals:  mean wavenumber in each bin
              PSbins: power spectra which has been binned into kbins
              waven:  wavenumber (0 - 1) in non-dimensional space.
    """
    
    from pywavan import fan_trans

    if print_info:
        print("\n------------------------")
        print("get_spectra2D_PYWAVAN powspec called\n")

    nmax = np.max(fld.shape)
        
    fld2 = fld.copy()
    
    if 'scales' in kwargs:
        scales = kwargs.get('scales')
    else:
        scales = np.zeros((nmax//2,))
        for n in np.arange(1,nmax//2):  
            scales[n] = (n) / (nmax//2 - 1)
        
    wt, H11a, kvals, PSbins, q = fan_trans(fld2, reso=1.0, q=0, qdyn=False, scales=scales, **kwargs)
    
    kbins = np.arange(0.5, (nmax+1)//2-1, 1.)
            
    if print_info:
            print("------------------------\n")
    
    return kbins, PSbins**2, 2*kvals

#-------------------------------------------------------------------------------------
# 2D Spectra from the fourier spectrum package from jfrob27's pywaven package

def get_spectra2D_POWSPEC(fld, print_info=True, zeromean=True, **kwargs):
    """
    Returns 1D power spectra from a 2D field 
        
    Input:  2D floating pont array
    
    Returns:  kvals:  mean wavenumber in each bin
              PSbins: power spectra which has been binned into kbins
              waven:  wavenumber (0 - 1) in non-dimensional space.
    """
    
    from pywavan import powspec

    if print_info:
        print("\n------------------------\n")
        print("get_spectra2D_POWSPEC powspec called\n")

    ny, nx = fld.shape
        
    if nx != ny:
        
        if print_info:
            print("get_spectra2D_RAD: can only analyze process same wavenumbers in X & Y, nx: %d  ny: %d\n" %(nx,ny))
            print("get_spectra2D_RAD: will sample a square domain using nx/2, ny/2 center point\n")
        
        if nx > ny:
            nny = int(ny//2) 
            nnx = int(nx//2)
            fld2 = fld[0:ny,-nny+nnx:nnx+nny]
            nx = ny
        if nx < ny:
            nny = int(ny//2) 
            nnx = int(nx//2)
            fld2 = fld[-nnx+nny:nny+nnx,0:nx]
            ny = nx
    else:
        fld2 = fld.copy()
        
    waven, PSbins = powspec(fld2, reso=1., zeromean=zeromean)

    kbins = np.arange(0.5, (nx+1)//2-1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    wavenumber = 2*(kvals-1)/nx

    PSbins = PSbins * np.pi * (kbins[1:]**2 - kbins[:-1]**2)
            
    if print_info:
            print("------------------------\n")
    
    return kbins, PSbins, 2*waven


#-------------------------------------------------------------------------------------
# 2D Spectra

def get_spectra2D_RAD(fld, print_info=True, zeromean=True, **kwargs):
    """
    Returns 1D power spectra from a 2D field where 2D spectrum is averaged into radial bins.
    There are several caveats.  First, the shape of the fld array must be square.  If it is not square, 
    then it is made square by using the smallest dimension - one way or another the 2D spectrum can only 
    be represented by the dimensions along the smallest dimension (meaning the longer wavelengths
    are truncated).  The square is centered on [ny/2, nx/2] and has dimensions of (MIN(nx,ny))**2
    
    Second, the number of points must be an even number - which means dropping a single point at
    most, which we do at the end of the array.
    
    Input:  2D floating pont array
    
    Returns:  kvals:  mean wavenumber in each bin
              PSbins: power spectra which has been binned into kbins
              waven:  wavenumber (0 - 1) in non-dimensional space.
    """

    if print_info:
        print("\n------------------------")
        print("get_spectra2D_RAD called\n")

    ny, nx = fld.shape
        
    if nx != ny:
        
        if print_info:
            print("get_spectra2D_RAD: can only analyze process same wavenumbers in X & Y, nx: %d  ny: %d\n" %(nx,ny))
            print("get_spectra2D_RAD: will sample a square domain using nx/2, ny/2 center point\n")
        
        if nx > ny:
            nny = int(ny//2) 
            nnx = int(nx//2)
            fld2 = fld[0:ny,-nny+nnx:nnx+nny]
            nx = ny
        if nx < ny:
            nny = int(ny//2) 
            nnx = int(nx//2)
            fld2 = fld[-nnx+nny:nny+nnx,0:nx]
            ny = nx
    else:
        fld2 = fld.copy()
        
    if zeromean == True:
        fld2 -= np.mean(fld2)
            
    # now need to make the number of points even...
    
    nx = 2*(nx//2)
    ny = 2*(ny//2)
    fourier_image = np.fft.fftn(fld2[0:ny,0:nx])
        
    if print_info: 
        print("get_spectra2D_RAD: Field has [even] dimensions nx: %d  ny: %d\n" %(nx,ny))
            
    fourier_amplitudes = np.abs(fourier_image)**2 / (nx*ny)

    kfreq   = np.fft.fftfreq(nx) * nx
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm    = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    kbins = np.arange(0.5, nx//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    PSbins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes, statistic = "mean", bins = kbins)
    
    PSbins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    wavenumber = 2*(kvals-1)/nx
    
    if print_info:
            print("------------------------\n")
    
    return kvals, PSbins, wavenumber

#-------------------------------------------------------------------------------------
# 1D Spectra

def get_spectra2D_AVG(fld, axis=1, print_info=True, zeromean=True, **kwargs):
    """
    Returns the average power spectra from a 2D field along one dimension averaging over the second direction.
    
    Input:  2D floating pont array
    
    Returns:  kvals:  mean wavenumber in each bin
              PSbins: power spectra which has been binned into kbins
              waven:  wavenumber (0 - 1) in non-dimensional space.
    """
    
    if print_info: 
        print("\n------------------------")
        print("get_spectra2D_AVG called")
        print("------------------------\n")

    # now need to make the number of points even...

    ny, nx = fld.shape
    nx     = 2*(nx//2)
    ny     = 2*(ny//2)
    
    fld2   = fld[0:ny,0:nx].copy()
    
    if zeromean == True:
        fld2 -= np.mean(fld2)
    
    if print_info: 
        print("get_spectra2D_AVG: Field has [even] dimensions nx: %d  ny: %d\n" %(nx,ny))

    # Now pick an axis to average over

    nx = fld2.shape[axis]
    
    other_axis = [1,0]

    avg_axis = other_axis[axis]
    
    fourier_image = np.fft.fft(fld2, axis = axis)
    
    fourier_amplitudes = (np.abs(fourier_image)**2).mean(axis = avg_axis) / nx
    
    kfreq   = np.fft.fftfreq(nx) * nx
    
    kbins = np.arange(0.5, nx//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])

    PSbins, _, _ = stats.binned_statistic(kfreq, fourier_amplitudes, statistic = "mean", bins = kbins)
    
    PSbins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    wavenumber = 2*(kvals-1)/nx
    
    return kvals, PSbins, wavenumber

#-------------------------------------------------------------------------------------
# 3D Spectra

def get_spectraND(fld, func = get_spectra2D_RAD, **kwargs):
    """
    Returns average spectra from ND data set where the power spectra computed along
    the last two dimensions (often assumed to be x & y).  The input array can have
    3, 4, or even 5 dimensions (e.g., 5D => [case_day, fhours, klevels, ny, nx]
    and the function will reshape the input array to have 3 dimensions and then
    compute spectra over the last two dimensions using the 2D function passed
    and return the spectra from the last two dimensions.  Normally the spectra is averaged,
    over the first dimensions, but one can return the 3D array of spectra. 
    """
    print("\n----------------------")
    print("get_spectraND called")
    print("----------------------\n")

    
    if fld.ndim < 3:
        print("get_spectraND:  Array is wrong size, input array must have at least 3 dimensions\n")
        return None
    
    if fld.ndim > 2:
        
        fshape = fld.shape
        start = 0
        count = fld.ndim-2

        fld3d = np.reshape(fld.copy(), fshape[:start] + (-1,) + fshape[start+count:])
        
        if fld.ndim > 3:
            print("get_spectraND:  Reshaped array so that spectra averaged over outer dimension: %d\n" % fld3d.shape[0])

        Abins = []
        
        for k in np.arange(fld3d.shape[0]):
            kvals, A, waven = func(fld3d[k], func=func, print_info=False, **kwargs)
            Abins.append(A)
            
        Abins = np.asarray(Abins)
        
        return kvals, Abins.mean(axis=0), waven
#-------------------------------------------------------------------------------------
# Plot spectral

def plot_spectra(fld, func = get_spectra2D_POWSPEC, legend = None, ax = None, PScolor='k', 
                 PSline='-', ptitle='Power Spectra', loglog=1, LinsborgSlope = False, **kwargs):
    
    import matplotlib.ticker as mticker
    from spectra.py_spectra import get_spectra2D_POWSPEC
    
    def update_ticks(x, pos):
        if x != 0.0:
            return "%2.1f" % (2.0/x)
        else:
            return r'$\infty$'
        
    if len(fld.shape) < 3:  
        kvals, Abins, waven = func(fld, **kwargs)
        
    else:
        kvals, Abins, waven = get_spectraND(fld, func = func, **kwargs)

    if 'debug' in kwargs:
        print('kvals: ',kvals.shape,kvals)
        print('PS: ',Abins.shape,Abins)
        print('wavenumber: ',waven.shape,waven)

    if legend == None:
        legend = 'Field'
    
    if ax == None:
        fig, axes = plt.subplots(1, 2, constrained_layout=True,figsize=(20,8))
        
        axes[1].imshow(fld[::-1])
        axes[1].set_title(ptitle, fontsize=18)
        
    else:
        axes = ax
        
    if loglog:
        axes[0].loglog(waven, Abins, color=PScolor, linestyle=PSline)
        axes[0].set_xlim(2.0/waven.shape[0], 1.0)

        axes[0].annotate("%s\nLog Power Scale" % legend, xy=(0.10, 0.25), xycoords='axes fraction', color='k',fontsize=18)
        axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(update_ticks))
           
        ylim = axes[0].get_ylim()
        
        if 'ylabels' in kwargs:
            ylabel = kwargs.get('ylabel')
        else:
            ylabel = 10
        
        xoffset = [0.01, 0.0075, 0.005, 0.0035, 0.0025, 0.001]
        
        for n, w in enumerate([3.0, 4.0, 6.0, 8.0, 12.0, 16.0]):
            axes[0].axvline(x = (2.0/w), color = 'grey', label = 'axvline - full height')  
            axes[0].annotate(r"%d$\Delta$x" % w, xy=(2.0/w + xoffset[n], ylabel), xycoords='data', color='k',fontsize=12)
            
        if LinsborgSlope:
            xpt = [2.0/16.,2.0/2.0]
            dlnx = np.log(xpt[1]) - np.log(xpt[0])
            y1   = ylim[1]/(10.)
            y0   = np.exp(np.log(y1) + 5./3. * dlnx)
            ypt  = [y0,y1]
            axes[0].loglog(xpt, ypt, color='red',linestyle='-.',label='k$^{-5/3}$')

    else:
        axes[0].plot(waven, Abins, color=PScolor, linestyle=PSline)
        axes[0].set_xlim(0.0, 1.0)
        
        axes[0].set_xticks(axes[0].get_xticks()) # see https://github.com/matplotlib/matplotlib/issues/18848
        axes[0].set_xticklabels([r'$\infty$', r"10", r"5", r"3.3", r"2.5", r"2.0"],fontsize=12, weight='bold')
        
        for w in [4.0, 6.0, 8.0, 10.0, 12.0, 16.0]:
            axes[0].annotate(r"%d" % int(w), xy = (2.0/w-0.01, -0.035), xycoords='axes fraction', color='k',fontsize=12)
            axes[0].axvline(x = 2.0/w-0.0075, color = 'grey', label = 'axvline - full height')
            
        axes[0].annotate("%s\nLinear Power Scale" % legend, xy=(0.70, 0.25), xycoords='axes fraction', color='k',fontsize=18)

    axes[0].set_xlabel(r"Wavelength in ($\Delta$x)", fontsize=12)
    axes[0].set_ylabel(r"KE Spectral Density (m$^3$ s$^{-2}$)", fontsize=16)
    
    if 'ylim' in kwargs:
        axes[0].set_ylim(kwargs.get('ylim'))

    plt.title(ptitle, fontsize=18)
    
    
    if ax == None: 
        plt.show()