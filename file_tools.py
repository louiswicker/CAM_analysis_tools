import numpy as np
import netCDF4 as ncdf
import matplotlib.pyplot as plt
import xarray as xr
import glob as glob
import os as os
import sys as sys
import pygrib
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime


import warnings
warnings.filterwarnings("ignore")

# def fxn():
#     warnings.warn("deprecated", DeprecationWarning)

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     fxn()
    
_nthreads = 2

_Rgas       = 287.04
_gravity    = 9.806

# These are 45 vertical levels that the FV3 puts out - use them here to map ARW to that grid for comparison

plevels = np.asarray([100000.,  97500.,  95000.,  92500.,  90000.,  87500.,  85000.,  82500.,
                       80000.,  77500.,  75000.,  72500.,  70000.,  67500.,  65000.,  62500.,
                       60000.,  57500.,  55000.,  52500.,  50000.,  47500.,  45000.,  42500.,
                       40000.,  37500.,  35000.,  32500.,  30000.,  27500.,  25000.,  22500.,
                       20000.,  17500.,  15000.,  12500.,  10000.,   7000.,   5000.,   3000.,
                        2000.,   1000.,    700.,    500.,    200.])

nz_new = plevels.shape[0]

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
        #print("k %d ind %d Level %d obj_level %d:    Min %4.1f      Max %4.1f"%(k, indexes[k], levels[indexes[k]], grb_obj[indexes[k]].level, np.min(cube[k,:,:]), np.max(cube[k,:,:])))

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
# Add hour to xarray dataset from filename

def add_fhour(ds):
    
    filename = ds.encoding["source"].split("_")
    
    init_time = int(filename[-2])
    fhour     = int(filename[-1][-5:-3])
        
    ds.coords['fhour'] = fhour
    ds.coords['init_time'] = init_time
    
    return ds
    
#--------------------------------------------------------------------------------------------------
# Wrapper probably not necessary, but here you go...kwargs might be helpful

def open_mfdataset_list(data_dir, pattern):
    """
    Use xarray.open_mfdataset to read multiple netcdf files from a list.
    """
    filelist = os.path.join(data_dir,pattern)
    return xr.open_mfdataset(filelist, preprocess=add_fhour, combine='nested', concat_dim=['fhour'],parallel=True)

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
    else:
        grb_var = grb_file.select(name='Geometric vertical velocity')
        cube = grbVar_to_cube(grb_var)['data']
        
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
        
        fig = plt.figure(figsize=(20, 20))

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
    
    ax.pcolormesh(lons, lats, -cube[klevel], cmap=cmap, transform=ccrs.PlateCarree())
                                    
    ax.set_title(title,fontsize=20)
    
    return 

#--------------------------------------------------------------------------------------------------
# Interp from 3D pressure to 1D pressure (convert from hybrid to constant p-levels)

def interp3d_np(data, p3d, p1d, nthreads = _nthreads):
    
    dinterp = np.zeros((len(p1d),data.shape[1],data.shape[2]),dtype=np.float32)

    if nthreads < 0:  # turning this off for now.
        def worker(i):
            print("running %d %s" % (i, data.shape))
            for j in np.arange(data.shape[1]):
                  dinterp[:,j,i] = np.interp(p1d[::-1], p3d[:,j,i], data[:,j,i])

        pool = mp.Pool(nthreads)
        for i in np.arange(data.shape[2]):
            pool.apply_async(worker, args = (i, ))
        pool.close()
        pool.join()
        
        return dinterp[::-1,:,:]
    
    else:        
        for i in np.arange(data.shape[2]):
            for j in np.arange(data.shape[1]):
                dinterp[:,j,i] = np.interp(p1d[::-1], p3d[:,j,i], data[:,j,i])
        
        return dinterp[::-1,:,:]

#--------------------------------------------------------------------------------------------------
# Choose a section of the grid based on lat/lon corners - excludes the rest of grid from xarray

def extract_subregion(xr_obj, sw_corner=None, ne_corner=None, drop=True):
    
    if (sw_corner and len(sw_corner) > 1) and (ne_corner and len(ne_corner) > 1):
        lat_min = min(sw_corner[0], ne_corner[0])
        lat_max = max(sw_corner[0], ne_corner[0])
        lon_min = min(sw_corner[1], ne_corner[1])
        lon_max = max(sw_corner[1], ne_corner[1])
        
        print(f'Creating a sub-region of grid: {lat_min:.2f}, {lon_min:.2f}, {lat_max:.2f}, {lon_max:5.2f}','\n')
        
        xr_obj.attrs['gridType'] = 'region'

        return xr_obj.where( (lat_min < xr_obj.lats) & (xr_obj.lats < lat_max)
                           & (lon_min < xr_obj.lons) & (xr_obj.lons < lon_max), drop=drop)
    else:
        print(f"No grid information supplied - returning original grid!\n")
        return xr_obj

#--------------------------------------------------------------------------------------------------

def hrrr_grib_read_variable(file, sw_corner=None, ne_corner=None, var_list=[''], interpP=True, writeout=True, prefix=None):
    
    # Special thanks to Scott Ellis of DOE for sharing codes for reading grib2
    
    default = {            #  Grib2 name                 / No. of Dims /  Type   / bottomLevel / paramCategory / paramNumber
               'TEMP':     ['Temperature',                           3, 'hybrid'],
               'OMEGA':    ['Vertical velocity',                     3, 'hybrid'],
               'U':        ['U component of wind',                   3, 'hybrid'],
               'V':        ['V component of wind',                   3, 'hybrid'],
               'GPH':      ['Geopotential Height',                   3,  'hybrid'],
               'UH':       ['unknown',                               2,  'hybrid', 2000,         7,              199],
               'CREF':     ['Maximum/Composite radar reflectivity',  2,  'atmosphere'],
               'HGT':      ['Orography',                             2,  'surface'],
     
               }

    if var_list != ['']:
        variables = {k: var_list[k] for k in var_list.keys() & set(var_list)}  # yea, I stole this....
    else:
        variables = default

    if prefix == None:
        prefix = 'hrrr'

    print(f'-'*120,'\n')
    print(f'HRRR Grib READ: Extracting variables from grib file: {file}','\n')

    # open file

    grb_file = pygrib.open(file)

    # Get lat lons

    lats, lons = grbFile_attr(grb_file)
    if (np.amax(lons) > 180.0): lons = lons - 360.0

    pres = None

    if interpP:  # need to extract out 3D pressure for interp.
        
        grb_var = grb_file.select(name='Pressure')
        cube = grbVar_to_cube(grb_var, type='hybrid')
        p3d = cube['data']
        print(f'InterpP is True, Read 3D pressure field from GRIB file\n')
        print(f'P-max:  %5.2f  P-min:  %5.2f\n' % (p3d.max(), p3d.min()))

    for n, key in enumerate(variables):

        print('Reading my variable: ',key, 'from GRIB file variable: %s\n' % (variables[key][0]))

        if type(variables[key][0]) == type('1'):
            if len(variables[key]) == 3:
                grb_var = grb_file.select(name=variables[key][0],typeOfLevel=variables[key][2])
            else:
                grb_var = grb_file.select(bottomLevel=variables[key][3],parameterCategory=variables[key][4],parameterNumber=variables[key][5])
        else:
            grb_var = [grb_file.message(variables[key][0])]

        if variables[key][1] == 3:

            cube = grbVar_to_cube(grb_var, type=variables[key][2])
            
            if interpP:
                
                cubeI = interp3d_np(cube['data'], p3d, plevels)
                    
                new = xr.DataArray( cubeI, dims = ['nz','ny','nx'], 
                                                  coords={"lats": (["ny","nx"], lats),
                                                          "lons": (["ny","nx"], lons), 
                                                          "pres": (["nz"],      plevels) } )
                
            else:

                new = xr.DataArray( cube['data'], dims = ['nz','ny','nx'], 
                                                  coords={"lats": (["ny","nx"], lats),
                                                          "lons": (["ny","nx"], lons), 
                                                          "hybid": (["nz"],     cube['levels']) } )
        if variables[key][1] == 2:

            cube = grbVar_to_slice(grb_var, type=variables[key][2])

            new = xr.DataArray( cube['data'], dims=['ny','nx'], 
                                             coords={"lats": (["ny","nx"], lats),
                                                     "lons": (["ny","nx"], lons) } )  

        if n == 0:

            ds_conus = new.to_dataset(name = key)

        else:         

            ds_conus[key] = new

        del(new)
        
        date      = cube['date'] 
        fcstStart = cube['fcstStart']
        fcstHour  = cube['fcstTime']
        
    # Add attributes
    
    ds_conus.attrs['date']       = date
    ds_conus.attrs['fcstStart']  = fcstStart
    ds_conus.attrs['fcstHour']   = fcstHour
    ds_conus.attrs['gridPrefix'] = prefix
    ds_conus.attrs['gridType']   = 'conus'
    ds_conus.attrs['DateTime']   = datetime.now().strftime("%Y%m%d_%H:%M:%S")
    ds_conus.attrs['TimeStamp']  = datetime.timestamp(datetime.now())

    # clean up grib file
    
    grb_file.close()
            
    # Convert omega --> w
    
    pp3 = np.broadcast_to(plevels, (p3d.shape[2],p3d.shape[1],len(plevels))).transpose()
    
    w_new = -ds_conus['OMEGA'].data / ( (_gravity * pp3 ) / (_Rgas * ds_conus['TEMP'].data) )
    
    ds_conus['W'] = xr.DataArray( w_new, dims = ['nz','ny','nx'], 
                                  coords={"lats": (["ny","nx"], lats),
                                          "lons": (["ny","nx"], lons), 
                                          "pres": (["nz"],      plevels) } )   
    # extract region
    
    ds_conus = extract_subregion(ds_conus, sw_corner=sw_corner, ne_corner=ne_corner)
    
    # add some file and directory attributes
    
    dir, base = os.path.split(file)
    outfilename = os.path.join(dir, '%s_%08d%02d_F%02d.nc' % (ds_conus.attrs['gridType'], date, fcstStart, fcstHour))
        
    ds_conus.attrs['srcdir']   = dir
    ds_conus.attrs['filename'] = os.path.basename(outfilename)
 
    if writeout:
       
        ds_conus.to_netcdf(outfilename, mode='w')  
        print(f'Successfully wrote new data to file:: {outfilename}','\n')
        return ds_conus, outfilename 
    
    else:
        
        return ds_conus, outfilename
    
#--------------------------------------------------------------------------------------------------

def fv3_grib_read_variable(file, sw_corner=None, ne_corner=None, var_list=[''], writeout=True, prefix=None):
    
    # Special thanks to Scott Ellis of DOE for sharing codes for reading grib2
    
    default = {             #  Grib2 name                 / No. of Dims /  Type
               'TEMP':     ['Temperature',                 3, 'isobaricInhPa'],
               'HGT':      ['Orography',                   2, 'surface'],
               'GPH':      ['Geopotential Height',         3, 'isobaricInhPa'],              
               'W':        ['Geometric vertical velocity', 3, 'isobaricInhPa'],
               'U':        ['U component of wind',         3, 'isobaricInhPa'],
               'V':        ['V component of wind',         3, 'isobaricInhPa'],
               'UH':       ['unknown',                               2,  'isobaricInhPa', 2000,         7,              199],
               'CREF':     ['Maximum/Composite radar reflectivity',  2, 'atmosphereSingleLayer'      ],
               }

    if var_list != ['']:
        variables = {k: default[k] for k in default.keys() & set(var_list)}  # yea, I stole this....
    else:
        variables = default

    if prefix == None:
        prefix = 'rrfs'

    print(f'-'*120,'\n')
    print(f'RRFS Grib READ: Extracting variables from grib file: {file}','\n')

    # open file

    grb_file = pygrib.open(file)

    # Get lat lons

    lats, lons = grbFile_attr(grb_file)

    if np.amax(lons) > 180.0: lons = lons - 360.0

    for n, key in enumerate(variables):

        print('Reading my variable: ', key, 'from GRIB variable: %s\n' % (variables[key][0]))

        if type(variables[key][0]) == type('1'):
            if len(variables[key]) == 3:
                grb_var = grb_file.select(name=variables[key][0],typeOfLevel=variables[key][2])
            else:
                grb_var = grb_file.select(bottomLevel=variables[key][3],parameterCategory=variables[key][4],parameterNumber=variables[key][5])
        else:
            grb_var = [grb_file.message(variables[key][0])]
        
        if variables[key][1] == 3:
            cube = grbVar_to_cube(grb_var, type='isobaricInhPa')
            pres = cube['levels']
            new = xr.DataArray( cube['data'], dims=['nz','ny','nx'], coords={'pres': (['nz'], pres),
                                                                             "lons": (["ny","nx"], lons),
                                                                             "lats": (["ny","nx"], lats)} )      
        if variables[key][1] == 2:
            cube = grbVar_to_slice(grb_var)
            new = xr.DataArray( cube['data'], dims=['ny','nx'], coords={"lons": (["ny","nx"], lons),
                                                                        "lats": (["ny","nx"], lats)} )      

        if n == 0:
            
            ds_conus = new.to_dataset(name = key)
            
        else:
            
            ds_conus[key] = new
            
        del(new)
        
        date      = cube['date'] 
        fcstStart = cube['fcstStart']
        fcstHour  = cube['fcstTime']

    # clean up grib file
    
    grb_file.close()
    
    # Useful info for global attributes
         
    ds_conus.attrs['date']       = date
    ds_conus.attrs['fcstStart']  = fcstStart
    ds_conus.attrs['fcstHour']   = fcstHour
    ds_conus.attrs['gridPrefix'] = prefix
    ds_conus.attrs['gridType']   = 'conus'
    ds_conus.attrs['DateTime']   = datetime.now().strftime("%Y%m%d_%H:%M:%S")
    ds_conus.attrs['TimeStamp']  = datetime.timestamp(datetime.now())

    # extract region
    
    ds_conus = extract_subregion(ds_conus, sw_corner=sw_corner, ne_corner=ne_corner)

    # add some file and directory attributes
    
    dir, base   = os.path.split(file)
    outfilename = os.path.join(dir, '%s_%8.8i%2.2i_F%2.2i.nc' % (ds_conus.attrs['gridType'], date, fcstStart, fcstHour))
        
    ds_conus.attrs['srcdir']   = dir
    ds_conus.attrs['filename'] = os.path.basename(outfilename)
 
    if writeout:
       
        ds_conus.to_netcdf(outfilename, mode='w')  
        print(f'Successfully wrote new data to file:: {outfilename}','\n')
        return ds_conus, outfilename 
    
    else:
        
        return ds_conus, outfilename

#--------------------------------------------------------------------------------------------------


