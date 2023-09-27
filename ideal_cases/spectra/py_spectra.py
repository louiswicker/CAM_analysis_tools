import os, sys, argparse, cmath
import pylab as pl
import numpy as np
from scipy.fft import dct
from matplotlib import colors, ticker
import scipy.stats as stats
import matplotlib.pyplot as plt

_header = '-' * 100
_sep1   = ' ' * 10
_sep2   = ' ' * 20

_dct_type = 2
    
#-------------------------------------------------------------------------------------
def check_variance(array, spectra, dx=1.0, sep=_sep1):
    
    ny, nx = array.shape
    
    wnx = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(nx, dx))
    wny = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(ny, dx))
    
    dkx = wnx[1]-wnx[0]
    dky = wny[1]-wny[0]
    
    #########################################
    # Check that psd integrates to variance #
    #########################################
    # Built-in variance #
    #####################
    
    Raw_var= np.var(array)
    
    ########################
    # Built-in Integration #
    ########################
    
    DFT_var = integrate.trapz( integrate.trapz(spectra, dx=dky, axis=1), dx=dkx, axis=0)
    
    print("%s%sInput array variance         DFT Variance" % (sep, sep)  )
    print("%s%s-----------------------------------------" % (sep, sep))
    print("%s%s%15.10g            %15.10g\n" % (sep, sep, Raw_var, DFT_var) )
    
#-------------------------------------------------------------------------------------
def square_grid(fld, print_info):

    ny1, nx1 = fld.shape

    nny = int(ny1//2) 
    nnx = int(nx1//2)

    if print_info:
        print("Square Grid: 2D can only process same wavenumbers in X & Y, nx: %d  ny: %d\n" %(nx1,ny1))
        print("Square Grid: will sample a square domain using nx/2, ny/2 center point\n")

    if nx1 > ny1:
        tmp = fld[0:ny1,-nny+nnx:nnx+nny]
        nx1 = ny1
    if nx1 < ny1:
        tmp = fld[-nnx+nny:nny+nnx,0:nx1]
        ny1 = nx1
    else:
        tmp = fld.copy()
        
    if print_info:
        print("Square Grid:  New nx: %d  ny: %d\n" %(nx1,ny1))

    # now need to make the number of points even...

    mx = 2*(nx1//2)
    my = 2*(ny1//2)
    
    if print_info:
        print("Square Grid: New even nx: %d  even ny: %d\n" %(mx,my))

    return tmp[0:my,0:mx].copy()
#-------------------------------------------------------------------------------------

def remove_trend(fld, print_info):
    """
    Implementing Errico's (MWR 1985) 1D / 2D detrending algorithm
    
    """
    if len(fld.shape) == 1:
        
        ny = fld.shape
        
        sy = (fld[-1] - fld[0]) / (ny-1)
        
        scale = 0.5*(2*np.arange(1,ny+1) - ny - 1)
        
        fldy = fld - sy*scale
        
        return fldy
        
    elif len(fld.shape) == 2:
        
        ny, nx = fld.shape

        sy = (fld[-1,:] - fld[0,:]) / (ny-1)

        scale = 0.5*(2*np.arange(1,ny+1) - ny - 1)

        scale2d = np.broadcast_to(scale[:,np.newaxis], fld.shape)

        sy2d    = np.broadcast_to(   sy[np.newaxis,:], fld.shape)

        fldy = fld - scale2d*sy2d

        sx = (fldy[:,-1] - fldy[:,0]) / (nx-1)

        scale = 0.5*(2*np.arange(1,nx+1) - nx - 1)

        scale2d = np.broadcast_to(scale[np.newaxis,:], fld.shape)
        sx2d    = np.broadcast_to(   sx[:,np.newaxis], fld.shape)

        fldxy = fldy - scale2d*sx2d

        return fldxy
    
    else:
        print("Remove_trend:  Input array has invalid shape of:  %d, stopping code" % fld.shape)
        sys.exit(1)

#-------------------------------------------------------------------------------------
def get_spectra2D_DCT(fld, dx=1., dy=1., varray = None, sep=_sep1, **kwargs):
    """
    Code based on Nate Snook's implementation of the algorithm in Surcel et al. (2014)

    Arguments:
        `field` is a 2D numpy array for which you would like to find the spectrum
        `dx` and `dy` are the distances between grid points in the x and y directions in meters

    Returns a tuple of (length_scale, spectrum), where `length_scale` is the length scale in meters, and `spectrum` is 
        the power at each length scale.
    """

    # get some kwargs
    
    if 'print_info' in kwargs:
        print_info = kwargs['print_info']
    else:
        print_info = False
        
    if 'detrend' in kwargs:            # dont need to detrend data for DCT-II transform.
        detrend = kwargs['detrend']
    else:
        detrend = False
    
    u = square_grid(fld, print_info)
    
    if type(varray) != type(None):
        
        v = square_grid(varray, print_info)
            
    # compute spectra
    
    ny, nx = u.shape

    if type(varray) == type(None):
        
        variance = 0.5*dct(dct(u, axis=0, type=_dct_type, norm='ortho'), axis=1, type=_dct_type, norm='ortho')**2
        
    else:
        
        variance = 0.5*(dct(dct(u, axis=0, type=_dct_type, norm='ortho'), axis=1, type=_dct_type, norm='ortho')**2 \
                       +dct(dct(v, axis=0, type=_dct_type, norm='ortho'), axis=1, type=_dct_type, norm='ortho')**2)
           
    kfreq   = np.fft.fftfreq(nx) * nx
    kfreq2D = np.meshgrid(kfreq, kfreq)
    
    knrm   = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    knrm2  = 0.5*(knrm[1:,:] + knrm[:-1,:])
    knrm   = 0.5*(knrm2[:,1:] + knrm2[:,:-1])
    knrm   = knrm[:ny//2,:nx//2].flatten()
    
    # In order to make this similar to the DFT, you need to shift variances
    
    variance2 = np.zeros((ny,nx//2))
    variance3 = np.zeros((ny//2,nx//2))
    
    for i in np.arange(1,nx//2):   
        variance2[:,i-1] = variance[:,2*i-1] + variance[:,2*i]

    for j in np.arange(1,ny//2):   
        variance3[j-1,:] = variance2[2*j-1,:] + variance2[2*j,:]

    variance = variance3.flatten()
    
    kbins = np.arange(0.5, nx//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    PSbins, _, _ = stats.binned_statistic(knrm, variance, statistic = "mean", bins = kbins)
    
    PSbins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    
    waven = 2.*kvals / nx
            
    return kvals, PSbins, waven

#-------------------------------------------------------------------------------------
# 2D Spectra

def get_spectra2D_RAD(fld, varray = None, sep=_sep1, **kwargs):
    """
    Returns 1D power spectra from a 2D field where 2D spectrum is averaged into radial bins.
    There are several caveats.  First, the shape of the fld array must be square.  If it is not square, 
    then it is made square by using the smallest dimension - one way or another the 2D spectrum can only 
    be represented by the dimensions along the smallest dimension (meaning the longer wavelengths
    are truncated).  The square is centered on [ny/2, nx/2] and has dimensions of (MIN(nx,ny))**2
    
    Second, the number of points must be an even number - which means dropping a single point at
    most, which we do at the end of the array.
    
    Update 08/18/22:  Decided to use the Durran paper method and compute the ampltitudes precisely
                      as they do - the amplitudes of spectrum are computed from the complex conjugates.
                      Also added in the ability, now that the raw arrays need to be passed, to do
                      2D KE by adding in a "varray" keyword argument.
    
    Input:  2D floating pont array
    
    Returns:  kvals:  mean wavenumber in each bin
              PSbins: power spectra which has been binned into kbins
              waven:  wavenumber (0 - 1) in non-dimensional space.
    """
    

    if 'print_info' in kwargs:
        print_info = kwargs['print_info']
    else:
        print_info = False
        
    if 'detrend' in kwargs:
        detrend = kwargs['detrend']
    else:
        detrend = False
        
    # print("%sget_spectra2D_RAD: computing spectra" % sep)  
    
    if detrend:
        u = remove_trend(square_grid(fld, print_info), print_info)
    else:
        u = square_grid(fld, print_info)

    ny, nx = u.shape
                
    if type(varray) != type(None):
        if detrend:
            v = remove_trend(square_grid(varray, print_info), print_info)
        else:
            v = square_grid(varray, print_info)
                   
    # assumes raw array are passed, computing using Durran's method
        
    uh = np.fft.fftn(u, norm='ortho')

    if type(varray) == type(None):

        fourier_amplitudes = 0.5*(uh * np.conj(uh)).real

    else:
        
        vh = np.fft.fftn(v, norm='ortho')
        fourier_amplitudes = 0.5*(uh * np.conj(uh) + vh * np.conj(vh)).real
            
    kfreq   = np.fft.fftfreq(nx) * nx
    kfreq2D = np.meshgrid(kfreq, kfreq)
    
    knrm    = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    knrm    = knrm.flatten()
    
    fourier_amplitudes = fourier_amplitudes.flatten()
    
    kbins = np.arange(0.5, nx//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    
    PSbins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes, statistic = "mean", bins = kbins)
    
    PSbins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    wavenumber = 2*(kvals-1)/nx
        
    return kvals, PSbins, wavenumber

#-------------------------------------------------------------------------------------
# 3D Spectra

def get_spectraND(fld, varray = None, func = get_spectra2D_RAD, sep = _sep1, **kwargs):
    """
    Returns average spectra from ND data set where the power spectra computed along
    the last two dimensions (often assumed to be x & y).  The input array can have
    3, 4, or even 5 dimensions (e.g., 5D => [case_day, fhours, klevels, ny, nx]
    and the function will reshape the input array to have 3 dimensions and then
    compute spectra over the last two dimensions using the 2D function passed
    and return the spectra from the last two dimensions.  Normally the spectra is averaged,
    over the first dimensions, but one can return the 3D array of spectra. 
    """
    
    if fld.ndim < 3:
        print("%s%s" % (sep, _header))
        print("%sget_spectraND: Array is wrong size, input array must have at least 3 dimensions\n" % (_sep1))
        return None
    
    if fld.ndim > 2:
        
        fshape = fld.shape
        start = 0
        count = fld.ndim-2

        fld3d = np.reshape(fld.copy(), fshape[:start] + (-1,) + fshape[start+count:])
                    
        if fld.ndim > 3:
            print("%s%s" % (_sep1, _header))
            print("%sget_spectraND: Reshaped array so that spectra averaged over outer dimension: %d\n" % (_sep1, fld3d.shape[0]))

        Abins = []
        
        print("%s%s" % (_sep2, _header))
        print("%s%s is now being called" % (_sep2, func.__name__))
        
        if type(varray) != type(None):
            
            fld3d_v = np.reshape(varray.copy(), fshape[:start] + (-1,) + fshape[start+count:])

            for k in np.arange(fld3d.shape[0]):
                kvals, A, waven = func(fld3d[k], varray = fld3d_v[k], sep = _sep2, **kwargs)
                Abins.append(A)
        else:
            
            for k in np.arange(fld3d.shape[0]):
                kvals, A, waven = func(fld3d[k], sep = _sep2, **kwargs)
                Abins.append(A)

        Abins = np.asarray(Abins)
        
        print("\n")
        
        return kvals, Abins.mean(axis=0), waven
#-------------------------------------------------------------------------------------
# Plot spectral
    
def plot_spectra(fld, varray = None, func = get_spectra2D_RAD, legend = None, ax = None, linecolor='k', label=None, linestyle='-', linewidth=1.5, 
                 ptitle='Power Spectra', loglog=True, LinsborgSlope = False, LineOnly=False, **kwargs):
    
    import matplotlib.ticker as mticker
    from spectra.py_spectra import get_spectra2D_RAD
    
    if 'print_info' in kwargs:
        print_info = kwargs['print_info']
    else:
        print_info = False
        
    if 'detrend' in kwargs:
        detrend = kwargs['detrend']
    else:
        detrend = False
        
# -------------

    print("%s" % _header)
    print("plot_spectra: Computing power spectrum using function: %s" % (func.__name__))
    
    if 'mean_power' in kwargs:
        print("plot_spectra: Requesting mean-power, not summed-power to be plotted")
        
    if type(varray) == type(None):
        print("plot_spectra: Spectrum from a single variable")
    else:
        print("plot_spectra: Spectrum computed for KE")
    print("plot_spectra: DETREND = %s\n" % (detrend))
    
# -------------

    # This next set of code does most of the work
        
    if len(fld.shape) < 3:  
                
        kvals, Abins, waven = func(fld, varray = varray, func=func, sep=_sep1, **kwargs)
        
    else:
                
        kvals, Abins, waven = get_spectraND(fld, varray = varray, func = func, sep=_sep2, **kwargs)

    if print_info:
        print('kvals:      ',kvals.shape, kvals)
        print('PS:         ',Abins.shape, Abins)
        print('wavenumber: ',waven.shape, waven)

    if ax == None:
        fig, axes = plt.subplots(1, 1, constrained_layout=True,figsize=(10,8))
        
    else:
        axes = ax
        
    if loglog:
        
        axes.loglog(waven, Abins, color=linecolor, linestyle=linestyle, linewidth=linewidth, label=label)
        
        axes.set_xlim(2/waven.shape[0], 1.0)

        axes.annotate("%s" % legend, xy=(0.05, 0.25), xycoords='axes fraction', color='k',fontsize=14)
        
        axes.xaxis.set_major_formatter(lambda x, pos: str(int(2.0/x)))
           
        ylim = axes.get_ylim()
        
        if 'ylabels' in kwargs:
            ylabel = kwargs.get('ylabels')[0]
        else:
            ylabel = ylim[0]
        
        xoffset = [0.01, 0.005, 0.0035, 0.0025, 0.001]

        if not LineOnly:
        
            for n, w in enumerate([4.0, 8.0, 12.0, 16.0]):
                axes.axvline(x = (2.0/w), color = 'grey')  
                axes.annotate(r"%d$\Delta$x" % w, xy=(2.0/w + xoffset[n], 70*ylabel), xycoords='data', color='k',fontsize=12)
            
        if LinsborgSlope:
            xpt = [2.0/32.,2.0/2.0]
            dlnx = np.log(xpt[1]) - np.log(xpt[0])
            y1   = ylim[1]/(1000.)
            y0   = np.exp(np.log(y1) + 5./3. * dlnx)
            ypt  = [y0,y1]
            axes.loglog(xpt, ypt, color='red',linestyle='-.',label='k$^{-5/3}$')

    else:

        axes.plot(waven, Abins, color=linecolor, linestyle=linestyle, linewidth=linewidth, label=label)
        axes.set_xlim(0.0, 1.0)
        
        axes.set_xticks(axes[0].get_xticks()) # see https://github.com/matplotlib/matplotlib/issues/18848
        axes.set_xticklabels([r'$\infty$', r"10", r"5", r"3.3", r"2.5", r"2.0"],fontsize=12, weight='bold')
        
        for w in [4.0, 6.0, 8.0, 10.0, 12.0, 16.0]:
            axes.annotate(r"%d" % int(w), xy = (2.0/w-0.01, -0.015), xycoords='axes fraction', color='k',fontsize=12)
            axes.axvline(x = 2.0/w-0.0075, color = 'grey')
            
        #axes.annotate("%s\nLinear Power Scale" % legend, xy=(0.70, 0.25), xycoords='axes fraction', color='k',fontsize=18)

    axes.set_xlabel(r"Wavelength in ($\Delta$x)", fontsize=12)
    axes.set_ylabel(r"%s Spectral Density (m$^3$ s$^{-2}$)" % ptitle, fontsize=16)
    
    if 'ylim' in kwargs:
        axes.set_ylim(kwargs.get('ylim'))

    plt.title(ptitle, fontsize=18)
    
    if ax == None: 
        plt.show()
