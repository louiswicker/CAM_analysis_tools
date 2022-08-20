import os, sys, argparse, cmath
import pylab as pl
import numpy as np
from scipy.fftpack import dct
from matplotlib import colors, ticker
import scipy.stats as stats
import matplotlib.pyplot as plt

_header = '-' * 100
_sep1   = ' ' * 10
_sep2   = ' ' * 20

#-------------------------------------------------------------------------------------
def update_ticks(x, pos):
    if x != 0.0:
        return "%2.1f" % (2.0/x)
    else:
        return r'$\infty$'
    
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
def get_spectra2D_DCT(fld, dx=1., dy=1., **kwargs):
    """
    Code based on Nate Snook's implementation of the algorithm in Surcel et al. (2014)

    Arguments:
        `field` is a 2D numpy array for which you would like to find the spectrum
        `dx` and `dy` are the distances between grid points in the x and y directions in meters

    Returns a tuple of (length_scale, spectrum), where `length_scale` is the length scale in meters, and `spectrum` is 
        the power at each length scale.
    """

    if 'print_info' in kwargs:
        print_info = kwargs['print_info']
    else:
        print_info = False

    fld2 = square_grid(fld, print_info)
    
    spectrum_2d = dct(dct(fld2, axis=0, norm='ortho'), axis=1, norm='ortho')
    
    npj, npi = spectrum_2d.shape
    variance = spectrum_2d ** 2 / (npj * npi)

    j_s, i_s = np.meshgrid(np.arange(npj), np.arange(npi))
    wavenumber = np.hypot(j_s / float(npj), i_s / float(npi))

    wn_band_max = min(npj, npi)
    
    wn_bands = np.arange(1., wn_band_max) / wn_band_max
    
    len_scale_bands = dx / wn_bands

    spectrum = np.zeros(wn_bands.shape)

    for iband in range(wn_bands.shape[0] - 1):
        band_power = np.where((wavenumber.T >= wn_bands[iband]) & (wavenumber.T <= wn_bands[iband + 1]), variance, 0)
        spectrum[iband] = band_power.sum()
        
    waven = 2*(len_scale_bands)/npi
        
    # if 'print_info':
    #     print('kvals: ',len_scale_bands.shape, len_scale_bands)
    #     print('PS: ', spectrum.shape,spectrum)
    #     print('wavenumber: ',waven.shape, waven)
    
    print(spectrum.max(), spectrum.min())

    return len_scale_bands[::-1], spectrum, waven[::-1]

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
        
    uh = np.fft.fftn(u)

    if type(varray) == type(None):

        fourier_amplitudes = 0.5*(uh * np.conj(uh)).real/ (nx*ny)

    else:
        
        vh = np.fft.fftn(v)
        fourier_amplitudes = 0.5*(uh * np.conj(uh) + vh * np.conj(vh)).real/ (nx*ny)
        
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

def plot_spectra(fld, varray = None, func = get_spectra2D_RAD, legend = None, ax = None, PScolor='k', 
                 PSline='-', ptitle='Power Spectra', loglog=1, LinsborgSlope = False, **kwargs):
    
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
        
       
    print("%s" % _header)
    print("plot_spectra: Computing power spectrum using function: %s" % (func.__name__))
    if type(varray) == type(None):
        print("plot_spectra: Spectrum from a single variable")
    else:
        print("plot_spectra: Spectrum computed for KE")
    print("plot_spectra: DETREND = %s\n" % (detrend))
        
    # This next set of code does most of the work
        
    if len(fld.shape) < 3:  
                
        kvals, Abins, waven = func(fld, varray = varray, func=func, sep=_sep1, **kwargs)
        
    else:
                
        kvals, Abins, waven = get_spectraND(fld, varray = varray, func = func, sep=_sep2, **kwargs)

    if print_info:
        print('kvals: ',kvals.shape, kvals)
        print('PS: ', Abins.shape, Abins)
        print('wavenumber: ',waven.shape, waven)

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
        axes[0].set_xlim(2/waven.shape[0], 1.0)

        axes[0].annotate("%s\nLog Power Scale" % legend, xy=(0.10, 0.25), xycoords='axes fraction', color='k',fontsize=18)
        axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(update_ticks))
           
        ylim = axes[0].get_ylim()
        
        if 'ylabels' in kwargs:
            ylabel = kwargs.get('ylabel')
        else:
            ylabel = 10
        
        xoffset = [0.01, 0.005, 0.0035, 0.0025, 0.001]
        
        for n, w in enumerate([4.0, 8.0, 12.0, 16.0]):
            axes[0].axvline(x = (2.0/w), color = 'grey', label = 'axvline - full height')  
            axes[0].annotate(r"%d$\Delta$x" % w, xy=(2.0/w + xoffset[n], ylabel), xycoords='data', color='k',fontsize=12)
            
        if LinsborgSlope:
            xpt = [2.0/16.,2.0/2.0]
            dlnx = np.log(xpt[1]) - np.log(xpt[0])
            y1   = ylim[1]/(1000.)
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
    axes[0].set_ylabel(r"%s Spectral Density (m$^3$ s$^{-2}$)" % ptitle, fontsize=16)
    
    if 'ylim' in kwargs:
        axes[0].set_ylim(kwargs.get('ylim'))

    plt.title(ptitle, fontsize=18)
    
    if ax == None: 
        plt.show()