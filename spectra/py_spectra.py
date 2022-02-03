import os, sys, argparse, cmath
import pylab as pl
import numpy as np
from scipy import integrate
from matplotlib import colors, ticker
import scipy.stats as stats
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------------
# Dtrend field (from Corey, maybe from me a long time ago....

def dtrend2d(array):
    
    ny, nx  = array.shape[0], array.shape[1]

    slope      = ( array[ny-1,:] - array[0,:] ) / float(ny-1)
    correction = (2*np.arange(ny) - float(ny) - 1.0) / 2.0
    barray     = array.copy()
    
    for i in np.arange(nx):
        barray[:,i] = barray[:,i] - slope[i] * correction[:]
        
    slope       = ( barray[:,nx-1] - array[:,0] ) / float(nx-1)
    correction  = (2*np.arange(nx) - float(nx) - 1.0) / 2.0
        
    for j in np.arange(ny):
        barray[j,:] = barray[j,:] - slope[j] * correction[:]
        
    return barray

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

def get_spectra2D_PSD(fld, print_info=True, zeromean=False, dtrend=True, **kwargs):
    """
    Returns 1D power spectra density from a 2D field 
        
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
    L      = min(nx, ny)
    L2     = L//2
    
    # Make square domain, center it in the middle of the domain

    if nx != ny:
        
        if print_info:
            print("get_spectra2D_RAD: can only analyze process same wavenumbers in X & Y, nx: %d  ny: %d\n" %(nx,ny))
            print("get_spectra2D_RAD: will sample a square domain using nx/2, ny/2 center point\n")

        nymid = ny//2
        nxmid = nx//2
        fld2 = fld[-L2+nymid:nymid+L2, -L2+nxmid:nxmid+L2]

    else:
        
        fld2 = fld.copy()
        
    if dtrend:
        fld2 = dtrend2d(fld2)
        
    # call the powerspec of pywavan routine.

    waven, PSbins = powspec(fld2, reso=1., zeromean=zeromean)

    kbins = np.arange(0.5, L//2-1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    wavenumber = 2*(kvals-1)/nx

    if print_info:
        print("------------------------\n")
    
    return kbins, PSbins, 2*waven

#-------------------------------------------------------------------------------------
# 1D Spectra

def get_spectra2D_AVG(fld, print_info=True, zeromean=False, dtrend=True, **kwargs):
    """
    Returns the average power spectra from averaging spectra from each dimension.
    
    Code from Corey Potvin (thanks Corey!)
    
    # Research Notes------------------------------------------------------------------
    
    After discussion, Potvin and I have decided that the spectra here are off by
    a factor of 1/(2*PI) looking at eq (13) in Durran and Weyn MWR, 2017.  
    
    We also believe that in pixel coordinates, our returned PDS is the energy density.
    
    The only question left is whether we are double counting at kmax.  Which is left to future
    work
    
    # Research Notes------------------------------------------------------------------    
   
    Input:  2D floating pont array
    
    Returns:  kvals:  mean wavenumber in each bin
              PSD:    power spectra
              waven:  wavenumber (0 - 1) in non-dimensional space.
    """
    
    if print_info: 
        print("\n------------------------")
        print("get_spectra2D_AVG called")
        print("------------------------\n")

        
    # Make square domain, center it in the middle of the domain

    ny, nx = fld.shape
    L      = min(nx, ny)
    L2     = L//2
    
    if nx != ny:
        
        if print_info:
            print("get_spectra2D_RAD: can only analyze process same wavenumbers in X & Y, nx: %d  ny: %d\n" %(nx,ny))
            print("get_spectra2D_RAD: will sample a square domain using nx/2, ny/2 center point\n")

        nymid = ny//2
        nxmid = nx//2
        fld = fld[-L2+nymid:nymid+L2, -L2+nxmid:nxmid+L2]

    else:
        
        fld = fld.copy()

    if dtrend:
        fld = dtrend2d(fld)
        
    if zeromean == True:
        fld -= np.mean(fld2)
       
    xpsd  = 2.0/float(L)*np.power(np.absolute(np.fft.rfft(fld, axis=0)[0:L2+1]), 2)
    
    ypsd  = 2.0/float(L)*np.power(np.absolute(np.fft.rfft(fld, axis=1)[0:L2+1]), 2)
    
    PSD   = 0.5*(np.average(xpsd,axis=1) + np.average(ypsd,axis=0))
    
    waven = np.arange(0,L2+1)/float(L)
    
    kbins = np.arange(0.5, L2+1, 1.)

    return kbins, PSD, 2.0*waven

#-------------------------------------------------------------------------------------
# 3D Spectra

def get_spectraND(fld, func = get_spectra2D_AVG, print_info=False, **kwargs):
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
            kvals, A, waven = func(fld3d[k], func=func, print_info=print_info, **kwargs)
            Abins.append(A)
            
        Abins = np.asarray(Abins)
        
        return kvals, Abins.mean(axis=0), waven
#-------------------------------------------------------------------------------------
# Plot spectral

def plot_spectra(fld, func = get_spectra2D_AVG, legend = None, ax = None, PScolor='k', 
                 ptitle='Power Spectra', loglog=1, LinsborgSlope = False, no_Plot = False, 
                 print_info=False, ret_Data = False, **kwargs):
    
    import matplotlib.ticker as mticker
    
    def update_ticks(x, pos):
        if x != 0.0:
            return "%2.1f" % (2.0/x)
        else:
            return r'$\infty$'
        
    if len(fld.shape) < 3:  
        kvals, Abins, waven = func(fld, print_info=print_info, **kwargs)
        
    else:
        kvals, Abins, waven = get_spectraND(fld, func = func, print_info=print_info, **kwargs)
        
    if no_Plot:
        return [kvals, Abins, waven]

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
        axes[0].loglog(waven, Abins, color=PScolor)
        axes[0].set_xlim(2.0/waven.shape[0], 1.0)

        axes[0].annotate("%s\nLog Power Scale" % legend, xy=(0.10, 0.25), xycoords='axes fraction', color='k',fontsize=18)
        axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(update_ticks))
        
        ylim = axes[0].get_ylim()
        
        if 'ylabels' in kwargs:
            ylabel = kwargs.get('ylabel')
        else:
            ylabel = 0.5
        
        xoffset = [0.01, 0.0075, 0.005, 0.0035, 0.0025, 0.001]
        
        for n, w in enumerate([3.0, 4.0, 6.0, 8.0, 12.0, 16.0]):
            axes[0].axvline(x = (2.0/w), color = 'grey', label = 'axvline - full height')  
            axes[0].annotate(r"%d$\Delta$x" % w, xy=(2.0/w + xoffset[n], ylabel), xycoords='data', color='k',fontsize=12, zorder=3)
            
        if LinsborgSlope:
            xpt = [2.0/16.,2.0/2.0]
            dlnx = np.log(xpt[1]) - np.log(xpt[0])
            y1   = ylim[1]/(10.)
            y0   = np.exp(np.log(y1) + 5./3. * dlnx)
            ypt  = [y0,y1]
            axes[0].loglog(xpt, ypt, color='red',linestyle='-.',label='k$^{-5/3}$')

    else:
        axes[0].plot(waven, Abins, color=PScolor)
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
        
    if ret_Data:
        return [kvals, Abins, waven]