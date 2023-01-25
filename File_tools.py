import numpy as np
import netCDF4 as ncdf
import matplotlib.pyplot as plt
import xarray as xr
import glob as glob
import os as os
import glob
import sys as sys
import matplotlib as mpl

from datetime import datetime
import cftime
import pickle

from cmpref import cmpref_mod as cmpref


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
# Code from Joel McClune to convert dictionaries to objects

class DictObj:
    def __init__(self, in_dict:dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
               setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
            else:
               setattr(self, key, DictObj(val) if isinstance(val, dict) else val)
            
# Code to read a dictionary written to a pickle file, and convert to a object
            
def pickle2Obj(file, retObj=True):
    with open(file, 'rb') as f:
        if retObj == True:
            return(DictObj(pickle.load(f)))
        else:
            return(pickle.load(f))
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
    
    import cartopy.feature as cfeature
    import cartopy.crs as ccrs
    import pygrib
    
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
                grb_var = grb_file.select(bottomLevel=variables[key][3],parameterCategory=variables[key[4]],parameterNumber=variables[key][5])
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

def write_Z_profile(data, model='WRF', tindex=0, iloc=0, jloc=0,  data_keys = ['z3d', 'pres', 'theta', 'den', 'ppres']):
    
    try:
        from columnar import columnar
    except:
        print("Need to install columnar into conda, exiting")
        return
    
    print('#-----------------------------#')
    print('          %s                  ' % model)
    print('#-----------------------------#')
    
    newlist = []
    newlist.append(np.arange(data['z3d'].shape[1]).tolist())
    
    headers = ['level']
    
    for key in data_keys:
        
        newlist.append(data[key][tindex,:,iloc,jloc].tolist())
        headers.append(key)

    # need to rearange all the data into row like an excel spreadsheet
    
    row_data = [list(x) for x in zip(*newlist)]
        
    table = columnar(row_data, headers, no_borders=True)
    print(table)

#---------------------------------------------------------------------

def read_model_fields(run_dir, model_type='wrf', printout=False, filename=None, precip_only=False):
 
    ##################################
    
    if model_type == 'fv3_solo':
        
        if filename != None:
            print("Reading:  %s " % os.path.join(run_dir,filename))
            ds = xr.open_dataset(os.path.join(run_dir,filename),decode_times=False)
        else:
            ds = xr.open_dataset(os.path.join(run_dir, "*.nc"), decode_times=False)
        
        if precip_only == False:
            
            z3d    = ds.delz.values[:,::-1,:,:]
            z3d    = - np.cumsum(z3d,axis=1)

            w      = ds.dzdt.values[:,::-1,:,:]

            pbase  = ds.pfull.values[::-1]*100.
            pres   = ds.nhpres.values[:,::-1,:,:]
            ppres  = ds.nhpres_pert.values[:,::-1,:,:]
            pii    = (pres/100000.)**0.286

            tbase  = ds.tmp.values[0,::-1,-1,-1]
            tbase  = np.broadcast_to(tbase[np.newaxis, :, np.newaxis, np.newaxis], pii.shape) / pii
            theta  = ds.tmp.values[:,::-1,:,:] / pii
            thetap = theta - tbase
            qc     = ds.clwmr[:,::-1,:,:] 
            qr     = ds.rwmr[:,::-1,:,:] 
        
            acc_precip = ds.rain_k.values  # total precip in mm
            
            den    =  pres / (287.04*(theta)*pii)
            
            dsout = {'w': w, 'pbase': pbase, 'tbase': tbase, 'den': den, 'pii': pii, 'z3d': z3d, 'pres': pres, 
                    'qr': qr, 'qc': qc, 'theta': theta, 'thetap':thetap, 'ppres': ppres, 'acc_precip': acc_precip}
            
            if printout:
                write_Z_profile(dsout, model=model_type.upper())
        
        else:
            
            # z3d    = ds.delz.values[:,::-1,:,:]
            # z3d    = - np.cumsum(z3d,axis=1)
            w      = ds.dzdt.values[:,::-1,:,:]

            try:
                
                dsout = {'w': w, 'acc_precip': ds.rain_k.values, 'pres': ds.nhpres.values[:,::-1,:,:]}
            
            except AttributeError:
                dsout = {'w': w, 'acc_precip': ds.accumulated_rain.values, 'pres': ds.nhpres.values[:,::-1,:,:]}
        
        ds.close()        
            
        return dsout

    ##################################

    if model_type == 'wrf':
        
        def open_mfdataset_list(data_dir, pattern, skip=0):
            """
            Use xarray.open_mfdataset to read multiple netcdf files from a list.
            """
            filelist = os.path.join(data_dir,pattern)
            
            if skip > 0:
                filelist = filelist[0:-1:skip]
                
            return xr.open_mfdataset(filelist, combine='nested', concat_dim=['Time'], parallel=True)
    
        if filename == None:
            print("Reading:  %s " % os.path.join(run_dir,"wrfout_d01_0001-01-01_00:00:00"))
            ds = xr.open_dataset(os.path.join(run_dir,"wrfout_d01_0001-01-01_00:00:00"),decode_times=False)
        else:
            ds   = open_mfdataset_list(run_dir,  "wrfout*")

        if precip_only == False:
            
            w      = ds.W.values
            w      = 0.5*(w[:,1:,:,:] + w[:,:-1,:,:])
            ppres  = ds.P.values
            pbase  = ds.PB.values
            pres   = ppres + ds.PB.values
            tbase  = ds.T_BASE.values + 300.
            tbase  = np.broadcast_to(tbase[:, :, np.newaxis, np.newaxis], w.shape)
            theta  = ds.T.values + 300.
            thetap = theta - tbase
            z      = ds.PHB.values/9.806
            pii    = (pres/100000.)**0.286
            den    =  pres / (287.04*(theta)*pii)
            z3d    = ds.PHB.values/9.806
            z3d    = 0.5*(z[:,1:,:,:] + z[:,:-1,:,:])
            qc     = ds.QCLOUD
            qr     = ds.QRAIN
            
            dsout = {'w': w, 'pbase': pbase, 'tbase': tbase, 'den': den, 'pii': pii, 'z3d': z3d, 'pres': pres, 
               'qr': qr, 'qc': qc, 'theta': theta, 'thetap':thetap, 'ppres': ppres, 'acc_precip': ds.RAINNC.values}        
            
            if printout:
                write_Z_profile(dsout, model=model_type.upper())

        else:
            
            w      = ds.W.values
            w      = 0.5*(w[:,1:,:,:] + w[:,:-1,:,:])
            ppres  = ds.P.values
            pbase  = ds.PB.values
            pres   = ppres + ds.PB.values
            dsout  = {'w': w, 'acc_precip': ds.RAINNC[1:].values, 'pres': pres}      

        ds.close()
        
        return dsout

    ##################################    
    
    if model_type == 'fv3':
        
        def open_mfdataset_list(data_dir, pattern):
            """
            Use xarray.open_mfdataset to read multiple netcdf files from a list.
            """
            filelist = os.path.join(data_dir,pattern)
            return xr.open_mfdataset(filelist, combine='nested', concat_dim=['time'], parallel=True)
    
        ds   = open_mfdataset_list(run_dir,   "*.nc")

        if precip_only == False:

            w      = ds.W.values
            tbase  = ds.T.values[0,:,-1,-1] + 300.
            tbase  = np.broadcast_to(tbase[np.newaxis, :, np.newaxis, np.newaxis], w.shape)
            theta  = ds.T.values + 300.
            thetap = theta - tbase
            ppres  = ds.P.values
            pbase  = ds.PB.values
            pres   = ppres + pbase
            z3d    = ds.PHB.values/9.806
            pii    = (pres/100000.)**0.286
            den    =  pres / (287.04*(theta)*pii)
        
            if printout:
                write_Z_profile(out, model=model_type.upper())

            dsout = {'w': w, 'pbase': pbase, 'tbase': tbase, 'den': den, 'pii': pii, 'z3d': z3d, 'pres': pres, 
                     'theta': theta, 'thetap':thetap, 'ppres': ppres, 'acc_precip': ds.prec.values}
        
        else:

            w      = ds.W.values
            z3d    = ds.PHB.values/9.806
            dsout = {'w': w, 'z3d': z3d, 'acc_precip': ds.prec.values}

        ds.close()
           
        return dsout
    
    ##################################    
    if model_type == 'cm1':
        
        def open_mfdataset_list(data_dir, pattern):
            """
            Use xarray.open_mfdataset to read multiple netcdf files from a list.
            """
            filelist = os.path.join(data_dir,pattern)
            return xr.open_mfdataset(filelist, parallel=True)
    
        if filename == None:
            print("Reading:  %s " % os.path.join(run_dir,"cm1out.nc"))
            ds = xr.open_dataset(os.path.join(run_dir,"cm1out.nc"),decode_times=False)
        else:
            ds = open_mfdataset_list(run_dir,  "cm1out_*.nc")

        if precip_only == False:
            w      = ds.winterp.values
            qr     = ds.qr.values
            qc     = ds.qc.values
            tbase  = ds.th0.values
            thetap = ds.thpert.values
            theta  = thetap + tbase
            pres   = ds.prs.values
            ppres  = ds.prspert.values
            pbase  = ds.prs0.values
            pii    = (pres/100000.)**0.286
            den    = pres / (287.04*(thetap+tbase)*pii)
            z      = ds.zh.values * 1000. # heights are in km
            z3d    = np.broadcast_to(z[np.newaxis, :, np.newaxis, np.newaxis], w.shape)
        
            dsout = {'w': w, 'pbase': pbase, 'tbase': tbase, 'den': den, 'pii': pii, 'z3d': z3d, 'pres': pres, 
                     'qr': qr, 'qc': qc, 'theta': theta, 'thetap':thetap, 'ppres': ppres, 'acc_precip': 10*ds.rain.values}
            
            if printout:
                write_Z_profile(dsout, model=model_type.upper())

        else:
            w     = ds.winterp.values
            # z     = ds.zh.values * 1000. # heights are in km
            # z3d   = np.broadcast_to(z[np.newaxis, :, np.newaxis, np.newaxis], w.shape)
            dsout = {'w':  w, 'acc_precip': 10*ds.rain[1:].values, 'pres': ds.prs.values}

        ds.close()
            
        return dsout
    
#--------------------------------------------------------------------------------------------------
def generate_ideal_profiles(run_dir, model_type='wrf', filename=None, w_thresh = 5.0, cref_thresh = 45., min_pix=1, percentiles=None, compDBZ=False, **kwargs):
    
    print("processing model run:  %s \n" % run_dir)
    
    if model_type == 'fv3_solo':

        if filename != None:
            print("Reading:  %s " % os.path.join(run_dir,filename))
            ds = xr.open_dataset(os.path.join(run_dir,filename),decode_times=False)
        else:
            ds = xr.open_dataset(os.path.join(run_dir, "atmos_hifreq.nc"), decode_times=False)

        z    = ds.delz.values[:,::-1,:,:]
        z    = np.cumsum(z,axis=1)

        w    = ds.dzdt.values[:,::-1,:,:]
        pres = ds.nhpres.values[:,::-1,:,:]
        
        dbz  = ds.refl_10cm.values[:,::-1,:,:]
        
        profiles = compute_obj_profiles(w, dbz, pres, z, w_thresh = w_thresh, cref_thresh = cref_thresh, min_pix=min_pix, percentiles=percentiles, **kwargs)
        
        ds.close()

        return profiles
    
    if model_type == 'wrf':
        
        def open_mfdataset_list(data_dir, pattern, skip=0):
            """
            Use xarray.open_mfdataset to read multiple netcdf files from a list.
            """
            filelist = os.path.join(data_dir,pattern)
            
            if skip > 0:
                filelist = filelist[0:-1:skip]
                
            return xr.open_mfdataset(filelist, combine='nested', concat_dim=['Time'], parallel=True)
    
        if filename == None:
            print("Reading:  %s " % os.path.join(run_dir,"wrfout_d01_0001-01-01_00:00:00"))
            filename = os.path.join(run_dir,"wrfout_d01_0001-01-01_00:00:00")
            ds = xr.open_dataset(filename,decode_times=False)
        else:
            ds   = open_mfdataset_list(run_dir,  "wrfout*")
              
        if not os.path.exists(os.path.join(run_dir, 'wrf_dbz.npy')):
                
            print('\n WRF output is missing REFL_10CM --> computing')

            pbase  = ds.PB.values
            pres   = ds.P.values + ds.PB.values

            theta  = ds.T.values + 300.
            pii    = (pres/100000.)**0.286
            temp   = theta*pii

            qv     = ds.QVAPOR.values
            qc     = ds.QCLOUD.values
            qr     = ds.QRAIN.values
            qi     = np.zeros_like(qv)
            qs     = np.zeros_like(qv)
            qg     = np.zeros_like(qv)
            dbz    = np.zeros_like(qv)

            for n in np.arange(dbz.shape[0]):

                dbz[n] = cmpref.calcrefl10cm(qv[n], qc[n], qr[n], qs[n], qg[n], temp[n], pres[n])
                print(n, dbz.max(), dbz.min())

#             ds_new = xr.DataArray( data=dbz, name = 'REFL_10CM', \
#                                    dims   = ['Time','bottom_top','south_north','west_east'], \
#                                    coords = dict(Time=(['Time'], ds['Times'].values),
#                                                  bottom_top  = (['ZNU'], ds['ZNU'].values),
#                                                  south_north = (['XLAT'], ds['XLAT'].values),
#                                                  west_east   = (['XLONG'], ds['XLONG'].values) ) )

#             ds_new.to_netcdf(os.path.join(run_dir, 'wrf_dbz.nc'), mode='w')

            with open(os.path.join(run_dir, 'wrf_dbz.npy'), 'wb') as f:
                np.save(f, dbz)
            
            print("\nWrote file out: %s" % os.path.join(run_dir, 'wrf_dbz.npy'))
            
            # plt.imshow(dbz[10,10])
            # plt.show()

        else:
            
            print("\nReading external DBZ file: %s" % os.path.join(run_dir, 'wrf_dbz.npy'))

            with open(os.path.join(run_dir, 'wrf_dbz.npy'), 'rb') as f:
                dbz = np.load(f)
                
            print("\nShape of DBZ array: " ,dbz.shape)
            
            # plt.imshow(dbz[10,10])
            # plt.show()
                
            # for n in np.arange(dbz.shape[0]):
            #     print(n, dbz.max(), dbz.min())
                
        w      = ds.W.values
        w      = 0.5*(w[:,1:,:,:] + w[:,:-1,:,:])
        pbase  = ds.PB.values
        pres   = ds.P.values + ds.PB.values

        theta  = ds.T.values + 300.
        pii    = (pres/100000.)**0.286
        temp   = theta*pii

        z    = ds.PHB.values/9.806
        z    = 0.5*(z[:,1:,:,:] + z[:,:-1,:,:])
        
        # plt.imshow(w[10,10])

        profiles = compute_obj_profiles(w, dbz, pres, z, w_thresh = w_thresh, cref_thresh = cref_thresh, min_pix=min_pix, percentiles=percentiles, **kwargs)
        
        ds.close()

        return profiles
    
    if model_type == 'fv3':
        
        def open_mfdataset_list(data_dir, pattern):
            """
            Use xarray.open_mfdataset to read multiple netcdf files from a list.
            """
            filelist = os.path.join(data_dir,pattern)
            return xr.open_mfdataset(filelist, combine='nested', concat_dim=['time'], parallel=True)
    
        ds   = open_mfdataset_list(run_dir,   "*.nc")
        
        w    = ds.W.values
        w    = 0.5*(w[:,1:,:,:] + w[:,:-1,:,:])
        dbz  = ds.REFL_10CM.values
        pres = ds.P.values
        z    = ds.PHB.values/9.806
        z    = 0.5*(z[:,1:,:,:] + z[:,:-1,:,:])

        profiles = compute_obj_profiles(w, dbz, pres, z, w_thresh = w_thresh, cref_thresh = cref_thresh, min_pix=min_pix, percentiles=percentiles, **kwargs)
        
        ds.close()

        return profiles
    
    if model_type == 'cm1':
        
        def open_mfdataset_list(data_dir, pattern):
            """
            Use xarray.open_mfdataset to read multiple netcdf files from a list.
            """
            filelist = os.path.join(data_dir,pattern)
            return xr.open_mfdataset(filelist, parallel=True)
    
        if filename == None:
            print("Reading:  %s " % os.path.join(run_dir,"cm1out.nc"))
            ds = xr.open_dataset(os.path.join(run_dir,"cm1out.nc"),decode_times=False)
        else:
            ds = open_mfdataset_list(run_dir,  "cm1out_*.nc")
        
        w    = ds.winterp.values
        z    = ds.zh.values * 1000. # heights are in km
        z3d  = np.broadcast_to(z[np.newaxis, :, np.newaxis, np.newaxis], w.shape)
        pres = ds.prs.values

        if os.path.exists(os.path.join(run_dir, 'cm1_dbz.nc')):

            print("\nReading existing DBZ file in directory")

            ds2 = xr.load_dataset(os.path.join(run_dir, 'cm1_dbz.nc'))
            dbz = ds2['REFL_10CM'].values

        else:

            print("\n  ...No existing DBZ file found, computing a new 4D DBZ\n")

            theta = ds.th.values
            pii   = (pres/100000.)**0.286
            temp  = theta*pii
            qv    = ds.qv.values
            qc    = ds.qc.values
            qr    = ds.qr.values
            qi    = np.zeros_like(qv)
            qs    = np.zeros_like(qv)
            qg    = np.zeros_like(qv)
            dbz   = np.zeros_like(qv)

            for n in np.arange(dbz.shape[0]):

                dbz[n]  = cmpref.calcrefl10cm(qv[n], qc[n], qr[n], qs[n], qg[n], temp[n], pres[n])
                print(n, dbz[n].max(), dbz[n].min())

            ds_new = xr.DataArray( data=dbz, name = 'REFL_10CM', \
                                   dims   = ['time','zh','yh','xh'], \
                                   coords = dict(time=(['time'], ds['time'].values),
                                                 zh = (['zh'], ds['zh'].values),
                                                 yh = (["yh"], ds['yh'].values),
                                                 xh = (["xh"], ds['xh'].values) ) )

            ds_new.to_netcdf(os.path.join(run_dir, 'cm1_dbz.nc'), mode='w')

        profiles = compute_obj_profiles(w, dbz, pres, z3d, w_thresh = w_thresh, cref_thresh = cref_thresh, min_pix=min_pix, percentiles=percentiles, **kwargs)
        
        ds.close()

        return profiles

#-------------------------------------------------------------------------------
def compute_obj_profiles(W, DBZ, PRES, Z, w_thresh = 3.0, cref_thresh = 45., min_pix=5, percentiles=None, zhgts = 250. + 250.*np.arange(80)):
    
    from skimage.measure import label
    
    #---------------------------------------------------------------------
    # local interp function
    def interp3dz_np(data, z3d, z1d, nthreads = _nthreads):

        dinterp = np.zeros((len(z1d),data.shape[1]),dtype=np.float32)

        if nthreads < 0:  # turning this off for now.
            def worker(j):
                print("running %d %s" % (i, data.shape))
                dinterp[:,j] = np.interp(z1d, z3d[:,j], data[:,j])

            pool = mp.Pool(nthreads)
            for i in np.arange(data.shape[2]):
                pool.apply_async(worker, args = (i, ))
            pool.close()
            pool.join()

            return dinterp

        else:        
            for j in np.arange(data.shape[1]):
                dinterp[:,j] = np.interp(z1d, z3d[:,j], data[:,j])

            return dinterp
   
    # thanks to Larissa Reames for suggestions on how to do this (and figuring out what to do!)
    
    mask_cref   = np.where(DBZ.max(axis=1) > cref_thresh, True, False)
    mask_w_3d   = np.where(PRES < 70000.0, W, np.nan)
    
    mask_w_2d   = np.nanmax(mask_w_3d, axis=1)
    mask_w_cref = (mask_w_2d > w_thresh) & mask_cref
    f_mask      = mask_w_cref.astype(np.int8)
        
    wlist    = []
    size     = []
    all_obj  = 0
    w_obj    = 0
    
    # define a grid to interpolation profiles to

    
    
    for n in np.arange(W.shape[0]): # loop over number of time steps.            
        
        # check to see if there are objects
        
        if (np.sum(f_mask[n]) == 0):
            
            continue

        else:
        
            label_array, num_obj = label(f_mask[n], background=0, connectivity=2, return_num = True) # returns a 2D array of labels for updrafts)
            
            all_obj += num_obj

            if( num_obj > 0 ):                                     # if there is more than the background object, process array.

                for l in np.arange(1,num_obj):                     # this is just a list of 1,2,3,4.....23,24,25....
                    npix = np.sum(label_array == l)                # this is a size check - number of pixels assocated with a label
                    if npix >= min_pix:
                        jloc, iloc = np.where(label_array == l)    # extract out the locations of the updrafts 
                        w_obj += 1
                        if len(iloc) > 0 and len(jloc) > 0:
                            wraw    = W[n,:,jloc,iloc]               # get w_raw profiles
                            zraw    = Z[n,:,jloc,iloc]               # get z_raw profiles
                                 
                            profile = interp3dz_np(wraw.transpose(), zraw.transpose(), zhgts, nthreads = _nthreads)

                            wlist.append([profile.mean(axis=(1,))],)
                            size.append(wraw.shape[0])
                            
    if( len(wlist) < 1 ):
        
        print("\n ---> Compute_Obj_Profiles found no objects...returning zero...\n")
        return np.zeros((zhgts.shape[0],1))
    
    else:
        
        wprofiles = np.squeeze(np.asarray(wlist), axis=1).transpose()

        print("\n Number of selected updraft profiles:  %d \n Number of labeled objects:  %d\n" % (w_obj, all_obj))
        
        if percentiles:
            
            wprofiles = np.sort(wprofiles, axis=1)
                       
            wprofile_percentiles = [wprofiles]
            
            for n, p in enumerate(percentiles):
                
                print("Percentile value:  %f\n" % p)
                
                idx = max(int(wprofiles.shape[1]*p/100.) - 1, 0)
                
                wprofile_percentiles.append(wprofiles[:,idx:])
            
            return wprofile_percentiles, np.sort(np.asarray(size, dtype=np.float32))
            
        else:
    
            return np.sort(wprofiles, axis=1), np.sort(np.asarray(size,dtype=np.float32))

#-------------------------------------------------------------------------------

def getobjdata(run_dir, model_type='wrf', filename=None):
    
    print("processing model run:  %s \n" % run_dir)
    
    if model_type == 'fv3_solo':

        if filename != None:
            print("Reading:  %s " % os.path.join(run_dir,filename))
            ds = xr.open_dataset(os.path.join(run_dir,filename),decode_times=False)
        else:
            ds = xr.open_dataset(os.path.join(run_dir, "atmos_hifreq.nc"), decode_times=False)

        z    = ds.delz.values[:,::-1,:,:]
        z    = np.cumsum(z,axis=1)

        w    = ds.dzdt.values[:,::-1,:,:]
        pres = ds.nhpres.values[:,::-1,:,:]
        
        dbz  = ds.refl_10cm.values[:,::-1,:,:]
        
        ds.close()

        return {'z': z, 'w': w, 'pres': pres, 'dbz':dbz}
    
    if model_type == 'wrf':
        
        def open_mfdataset_list(data_dir, pattern, skip=0):
            """
            Use xarray.open_mfdataset to read multiple netcdf files from a list.
            """
            filelist = os.path.join(data_dir,pattern)
            
            if skip > 0:
                filelist = filelist[0:-1:skip]
                
            return xr.open_mfdataset(filelist, combine='nested', concat_dim=['Time'], parallel=True)
    
        if filename == None:
            print("Reading:  %s " % os.path.join(run_dir,"wrfout_d01_0001-01-01_00:00:00"))
            filename = os.path.join(run_dir,"wrfout_d01_0001-01-01_00:00:00")
            ds = xr.open_dataset(filename,decode_times=False)
        else:
            ds   = open_mfdataset_list(run_dir,  "wrfout*")
              
        if not os.path.exists(os.path.join(run_dir, 'wrf_dbz.npy')):
                
            print('\n WRF output is missing REFL_10CM --> computing')

            pbase  = ds.PB.values
            pres   = ds.P.values + ds.PB.values

            theta  = ds.T.values + 300.
            pii    = (pres/100000.)**0.286
            temp   = theta*pii

            qv     = ds.QVAPOR.values
            qc     = ds.QCLOUD.values
            qr     = ds.QRAIN.values
            qi     = np.zeros_like(qv)
            qs     = np.zeros_like(qv)
            qg     = np.zeros_like(qv)
            dbz    = np.zeros_like(qv)

            for n in np.arange(dbz.shape[0]):

                dbz[n] = cmpref.calcrefl10cm(qv[n], qc[n], qr[n], qs[n], qg[n], temp[n], pres[n])
                print(n, dbz.max(), dbz.min())

#             ds_new = xr.DataArray( data=dbz, name = 'REFL_10CM', \
#                                    dims   = ['Time','bottom_top','south_north','west_east'], \
#                                    coords = dict(Time=(['Time'], ds['Times'].values),
#                                                  bottom_top  = (['ZNU'], ds['ZNU'].values),
#                                                  south_north = (['XLAT'], ds['XLAT'].values),
#                                                  west_east   = (['XLONG'], ds['XLONG'].values) ) )

#             ds_new.to_netcdf(os.path.join(run_dir, 'wrf_dbz.nc'), mode='w')

            with open(os.path.join(run_dir, 'wrf_dbz.npy'), 'wb') as f:
                np.save(f, dbz)
            
            print("\nWrote file out: %s" % os.path.join(run_dir, 'wrf_dbz.npy'))
            
            # plt.imshow(dbz[10,10])
            # plt.show()

        else:
            
            print("\nReading external DBZ file: %s" % os.path.join(run_dir, 'wrf_dbz.npy'))

            with open(os.path.join(run_dir, 'wrf_dbz.npy'), 'rb') as f:
                dbz = np.load(f)
                
            print("\nShape of DBZ array: " ,dbz.shape)
            
            # plt.imshow(dbz[10,10])
            # plt.show()
                
            # for n in np.arange(dbz.shape[0]):
            #     print(n, dbz.max(), dbz.min())
                
        w      = ds.W.values
        w      = 0.5*(w[:,1:,:,:] + w[:,:-1,:,:])
        pbase  = ds.PB.values
        pres   = ds.P.values + ds.PB.values

        theta  = ds.T.values + 300.
        pii    = (pres/100000.)**0.286
        temp   = theta*pii

        z    = ds.PHB.values/9.806
        z    = 0.5*(z[:,1:,:,:] + z[:,:-1,:,:])
                
        ds.close()

        return {'z': z, 'w': w, 'pres': pres, 'dbz':dbz}
    
    if model_type == 'fv3':
        
        def open_mfdataset_list(data_dir, pattern):
            """
            Use xarray.open_mfdataset to read multiple netcdf files from a list.
            """
            filelist = os.path.join(data_dir,pattern)
            return xr.open_mfdataset(filelist, combine='nested', concat_dim=['time'], parallel=True)
    
        ds   = open_mfdataset_list(run_dir,   "*.nc")
        
        w    = ds.W.values
        w    = 0.5*(w[:,1:,:,:] + w[:,:-1,:,:])
        dbz  = ds.REFL_10CM.values
        pres = ds.P.values
        z    = ds.PHB.values/9.806
        z    = 0.5*(z[:,1:,:,:] + z[:,:-1,:,:])

        profiles = compute_obj_profiles(w, dbz, pres, z, w_thresh = w_thresh, cref_thresh = cref_thresh, min_pix=min_pix, percentiles=percentiles)
        
        ds.close()

        return profiles
    
    if model_type == 'cm1':
        
        def open_mfdataset_list(data_dir, pattern):
            """
            Use xarray.open_mfdataset to read multiple netcdf files from a list.
            """
            filelist = os.path.join(data_dir,pattern)
            return xr.open_mfdataset(filelist, parallel=True)
    
        if filename == None:
            print("Reading:  %s " % os.path.join(run_dir,"cm1out.nc"))
            ds = xr.open_dataset(os.path.join(run_dir,"cm1out.nc"),decode_times=False)
        else:
            ds = open_mfdataset_list(run_dir,  "cm1out_*.nc")
        
        w    = ds.winterp.values
        z    = ds.zh.values * 1000. # heights are in km
        z3d  = np.broadcast_to(z[np.newaxis, :, np.newaxis, np.newaxis], w.shape)
        pres = ds.prs.values

        if os.path.exists(os.path.join(run_dir, 'cm1_dbz.nc')):

            print("\nReading existing DBZ file in directory")

            ds2 = xr.load_dataset(os.path.join(run_dir, 'cm1_dbz.nc'))
            dbz = ds2['REFL_10CM'].values

        else:

            print("\n  ...No existing DBZ file found, computing a new 4D DBZ\n")

            theta = ds.th.values
            pii   = (pres/100000.)**0.286
            temp  = theta*pii
            qv    = ds.qv.values
            qc    = ds.qc.values
            qr    = ds.qr.values
            qi    = np.zeros_like(qv)
            qs    = np.zeros_like(qv)
            qg    = np.zeros_like(qv)
            dbz   = np.zeros_like(qv)

            for n in np.arange(dbz.shape[0]):

                dbz[n]  = cmpref.calcrefl10cm(qv[n], qc[n], qr[n], qs[n], qg[n], temp[n], pres[n])
                print(n, dbz[n].max(), dbz[n].min())

            ds_new = xr.DataArray( data=dbz, name = 'REFL_10CM', \
                                   dims   = ['time','zh','yh','xh'], \
                                   coords = dict(time=(['time'], ds['time'].values),
                                                 zh = (['zh'], ds['zh'].values),
                                                 yh = (["yh"], ds['yh'].values),
                                                 xh = (["xh"], ds['xh'].values) ) )

            ds_new.to_netcdf(os.path.join(run_dir, 'cm1_dbz.nc'), mode='w')
        
        ds.close()

        return {'z': z, 'w': w, 'pres': pres, 'dbz':dbz}
    
    
    
    
# Function to create a colortable that matches the NWS colortable
def radar_colormap():
    nws_reflectivity_colors = [
    "#646464", # ND
    "#ccffff", # -30
    "#cc99cc", # -25
    "#996699", # -20
    "#663366", # -15
    "#cccc99", # -10
    "#999966", # -5
    "#646464", # 0
    "#04e9e7", # 5
    "#019ff4", # 10
    "#0300f4", # 15
    "#02fd02", # 20
    "#01c501", # 25
    "#008e00", # 30
    "#fdf802", # 35
    "#e5bc00", # 40
    "#fd9500", # 45
    "#fd0000", # 50
    "#d40000", # 55
    "#bc0000", # 60
    "#f800fd", # 65
    "#9854c6", # 70
    "#fdfdfd" # 75
    ]

    return mpl.colors.ListedColormap(nws_reflectivity_colors)