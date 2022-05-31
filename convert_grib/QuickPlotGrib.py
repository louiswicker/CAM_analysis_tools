#--------------------------------------------------------------------------------------------------
# Standalone code to plot grib variables (set to vertical velocity) for quick and dirty perusal
#
#


import numpy as np
import matplotlib as mlab
import matplotlib.pyplot as plt
import glob as glob
import os as os
import sys as sys
import pygrib
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from optparse import OptionParser

import warnings
warnings.filterwarnings("ignore")

# def fxn():
#     warnings.warn("deprecated", DeprecationWarning)

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     fxn()
    
#--------------------------------------------------------------------------------------------------
# Special thanks to Scott Ellis of DOE for sharing codes for reading grib2

def grbFile_attr(grb_file):

    dataloc = np.array(grb_file[1].latlons())

    return np.float32(dataloc[0]), np.float32(dataloc[1])

def grbVar_to_slice(grb_obj, type=None):

    """Takes a single grb object for a variable returns a 2D plane"""

    return {'data' : np.float32(grb_obj[0].values), 'units' : grb_obj[0]['units'],
            'date' : grb_obj[0].date, 'fcstStart' : grb_obj[0].time, 'fcstTime' : grb_obj[0].step}

def grbVar_to_cube(grb_obj, type='isobaricInhPa'):

    """Takes a single grb object for a variable containing multiple
    levels. Can sort on type. Compiles to a cube"""

    all_levels = np.array([grb_element['level'] for grb_element in grb_obj])
    types      = np.array([grb_element['typeOfLevel'] for grb_element in grb_obj])

    if type != None:
        levels = []
        for n, its_type in enumerate(types):
            if type == types[n]:
                levels.append(all_levels[n])
        levels = np.asarray(levels)
    else:
        levels = all_levels

    n_levels   = len(levels)
    indexes    = np.argsort(levels)[::-1] # highest pressure first
    cube       = np.zeros([n_levels, grb_obj[0].values.shape[0], grb_obj[1].values.shape[1]])

    for k in range(n_levels):
        cube[k,:,:] = grb_obj[indexes[k]].values

    return {'data' : np.float32(cube), 'units' : grb_obj[0]['units'], 'levels' : levels[indexes],
            'date' : grb_obj[0].date, 'fcstStart' : grb_obj[0].time, 'fcstTime' : grb_obj[0].step}

#--------------------------------------------------------------------------------------------------
# Plotting strings for filtered field labels
    
def title_string(file, level, label, wmax, wmin, eps=None):
    if eps:
        return ("%s at level=%2.2i with EPS=%5.1f \n %s-max: %3.1f       %s-min: %4.2f" % (file, level, label, eps, label, wmax, label, wmin))
    else:
        return ("%s at level=%2.2i \n %s-Max: %3.1f     %s-Min: %4.2f" % (file, level, label, wmax, label, wmin))

#--------------------------------------------------------------------------------------------------
# Quick W plot to figure out where stuff is

def quickplotgrib(file, klevel= 20, cmap = 'turbo', ax=None, filetype='hrrr', \
                                   newlat=[25,50], 
                                   newlon=[-130,-65]):
    """
        Meant to be a quick look at a horizontal field from a grib file, using W.
        Defaults should be good enough, but its possible for some cases to have to changed these
        to look at the whole grid.
        
        Input:
        ------
        
        file:  name of grib file
        
        Options:
        --------
        
        klevel:    the horizontal level to plot.  Values of 20-30 should capture updraft
        cmap:      whatever you like, use a diverging colormap.  Values are automatically scaled.
        ax:        dont mess with this, unless your ax setup is done right (like below)
        filetype:  either HRRR or RRFS grib, the VV variable is hard coded to pick up what you need.
        newlat:    tuple of two lats for zoom, or to test the values for a region grid. Set to "None" to see whole grid.
        newlon:    tuple of two lons for zoom, or to test the values for a region grid. Set to "None" to see whole grid.
        
        LJW December 2021
    """
    
    # open file

    grb_file = pygrib.open(file)

    # Get lat lons

    lats, lons = grbFile_attr(grb_file)
    
    if filetype == 'hrrr':
        grb_var = grb_file.select(name='Vertical velocity')
        cube = grbVar_to_cube(grb_var, type='hybrid')['data']
        w_limits = [-50,10]
        w_sign   = -1.0
    else:
        grb_var = grb_file.select(name='Geometric vertical velocity')
        cube = grbVar_to_cube(grb_var)['data']
        w_limits = [-1,5]
        w_sign   = 1.0
        
    glat_min = lats.min()
    glat_max = lats.max()
    glon_min = lons.min()
    glon_max = lons.max()

    print(f'\nGrib File Lat Min: %4.1f  Lat Max:  %4.1f' % (glat_min, glat_max))
    print(f'\nGrib File Lon Min: %4.1f  Lon Max:  %4.1f' % (glon_min, glon_max))
    
    print(f'\nGrib File W Min: %4.1f  W Max:  %4.1f\n' %(-cube[klevel].max(), cube[klevel].min()))
    
    if ax == None:
        
        proj = ccrs.LambertConformal(central_latitude = 30, 
                                     central_longitude = 265., 
                                     standard_parallels = (10,10))
        
        fig = plt.figure(figsize=(10, 10))

        ax = plt.axes(projection = proj)

        ax.set_global()
        ax.gridlines(draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.STATES, linestyle=':')
        
        if newlat == None:
            lat_min, lat_max = glat_min, glat_max
        else:
            lat_min, lat_max = newlat
            
        if newlon == None:
            lon_min, lon_max = glon_min, glon_max
        else:
            lon_min, lon_max = newlon
            
        print(f'\nPlot Lat Min: %4.1f  Lat Max:  %4.1f' % (lat_min, lat_max))
        print(f'\nPlot Lon Min: %4.1f  Lon Max:  %4.1f' % (lon_min, lon_max))

        ax.set_extent([lon_min, lon_max, lat_min, lat_max])

# # Add variety of features
#         # ax.add_feature(cfeature.LAND)
#         # ax.add_feature(cfeature.OCEAN)
#         # ax.add_feature(cfeature.COASTLINE)

# # Can also supply matplotlib kwargs
#        ax.add_feature(cfeature.LAKES, alpha=0.5)    
# if klevel < 10:
#     vmin = -5.
#     vmax = 10.
#     clevels = np.linspace(vmin, vmax, 16)
# else:
#     vmin = -10.
#     vmax = 20.
#     clevels = np.linspace(vmin, vmax, 16)
                
    title = title_string(os.path.basename(file), klevel, 'W', cube[klevel].max(), cube[klevel].min())
    
    w_data = w_sign*np.ma.array(cube[klevel])
    
    wfinal = np.ma.masked_greater(np.ma.masked_less(w_data, w_limits[0]), w_limits[1])
    
    ax.pcolormesh(lons, lats, wfinal, cmap=cmap, transform=ccrs.PlateCarree())
                                    
    ax.set_title(title,fontsize=20)
    
    plt.savefig('%s.png' % file)
    
    plt.show()
    
    return 
    
#-------------------------------------------------------------------------------
#
# Main program
#
if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("-f", "--file", dest="file", type="string", default= None, \
                                  help="Name of grib file to plot")
    parser.add_option("-t", "--type", dest="type", type="string", default= 'hrrr', \
                                  help="Model type for grib file ['hrrr','rrfs']")
    parser.add_option("-k", "--klevel", dest="klevel", type="int", default= 20, \
                                  help="Horizontal W level to plot")


    (options, args) = parser.parse_args()

    if options.file == None:
        print
        parser.print_help()
        print
        sys.exit(1)
    else:
        if not os.path.exists(options.file):
            print("\nError! grib file does not seem to exist?")
            print("Filename:  %s" % (options.file))
            sys.exit(1)
            
            
    ret = quickplotgrib(options.file, klevel=options.klevel, filetype=options.type)            
            
            