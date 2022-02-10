#!/usr/bin/env python
# coding: utf-8

# System imports

import os as os
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#--------------------------------------------------
# Local code imports

from plot_tools import *

from filter.RaymondFilters import RaymondFilter

#--------------------------------------------------
# This is the data strcture that controls the input.

from input_default import input_all as input_config

#--------------------------------------------------
# Can turn on more output

debug = False

#--------------------------------------------------
# Local helper functions......

#--------------------------------------------------

def add_fhour(ds):
    
    filename = ds.encoding["source"].split("_")
    
    init_time = int(filename[-2])
    fhour     = int(filename[-1][-5:-3])
        
    ds.coords['fhour'] = fhour
    ds.coords['init_time'] = init_time
    
    return ds
    
#--------------------------------------------------

def read_mfdataset_list(data_dir, pattern):
    """
    Use xarray.open_mfdataset to read multiple netcdf files from a list.
    """
    filelist = os.path.join(data_dir,pattern)
    return xr.open_mfdataset(filelist, preprocess=add_fhour, combine='nested', concat_dim=['fhour'],parallel=True)

#--------------------------------------------------

def save_mfdataset_list(ds, dir, gridType=None):
    """
    Use xarray.save_mfdataset to save multiple netcdf files from a list, using the original file strings as a pattern
    """

    # Use new pathlib for Python > 3.5
    Path(dir).mkdir(parents=True, exist_ok=True)

    for n, hour in enumerate(ds.fhour):
        fcstHour  = ds.isel(fhour=n).fhour.values
        fcstStart = ds.isel(fhour=n).fcstStart
        date      = ds.isel(fhour=n).date      
        
        if gridType == None:
            gridType = ds.isel(fhour=n).attrs['gridType']
            
        outfilename = os.path.join(dir, '%s_%08d%02d_F%02d.nc' % (gridType, date, fcstStart, fcstHour))
        
        ds.isel(fhour=n).to_netcdf(outfilename, mode='w')  
        print(f'Successfully wrote new data to file:: {outfilename}','\n')
    
    return

# --------------------------------------------------
# Main processing function for filtering

def filter_ds(input_dir, output_dir, prefix, dx = 10, npass = 6, writeout=False, klevels=None, **kwargs):

    run = os.path.basename(input_dir)
        
    ds  = read_mfdataset_list(input_dir , "%s_*.nc" % prefix)
    
    if debug:
        print('Input: ',ds['W'])

    # Set up cartopy stuff here, so the plot routine is already set to use it.

    fig, axes = init_cartopy_plot(ncols=2, nrows=1, figsize=(20,10))

    # Plot the initial data

    cb_info = plot_w_from_xarray(ds, fhour=4, title=('%s_UNFILTERED' % run), ax = axes[0], **kwargs)

    # Convert to numpy arrays, fill in zeros

    w = np.nan_to_num(ds.W.values).astype('float64')

    nhour, nz, ny, nx = w.shape
    
    w_filtered = np.zeros_like(w)

    for n in np.arange(nhour):
        
        print("\n-------------------------------\n %s file %d being processed\n" % (run, n))

        w_filtered[n] = RaymondFilter(w[n], dx, klevels=klevels, order=10, npass = npass, fortran = True, highpass=True, width=50)
            
    ds['W'] = xr.DataArray(w_filtered, dims = ['fhour','nz','ny','nx'])

    if debug:
        print('Output: ',ds['W'])

    # Plot the Filtered data

    cb_info = plot_w_from_xarray(ds, fhour=4, title=('%s_FILTERED' % run), ax = axes[1], **kwargs)

    # Save filtered data

    if writeout:
        save_mfdataset_list(ds, output_dir, gridType='filtered')

    # Save sanity-check figure

    plt.savefig("%s/%s_F%2.2d_500_hPa.png" % (output_dir, run, dx), bbox_inches='tight', dpi=300)
    
    return

#--------------------------------------------------
#
# Main program

# Input data sets....

input_dir         = input_config["input_dir"]
output_dir        = input_config["output_dir"]

filtered_filename = input_config["filtered_filename"]
filter_dx         = input_config["filter_dx"]
filter_npass      = input_config["filter_npass"]
filtered_dirname  = "W_%2.2i" % filter_dx
klevels           = input_config["klevels"]
fprefix           = input_config["fprefix"]
    
#------------------------------------------------------------------------------------
#

print("\n=======> FILTER W <=========\n")
print("-------> Begin processing runs\n")
print("-------> Parameter Filter SCALE: %d \n" % filter_dx)
print("-------> Parameter        NPASS: %d \n" % filter_npass)
print("-------> Parameter FILTERED FILE DIRECTORY: %s \n" % filtered_filename)

for day in input_config["cases"]:
    
    # get zoom for plotting

    newlat = input_config["zoom"][day][0:2]
    newlon = input_config["zoom"][day][2:4]

    for run in input_config["cases"][day]:
        
        print("\n----> Processing run: %s for day:  %s \n" % (run,day))
        run_dir = str(os.path.join(input_dir, day, run))
        out_dir = str(os.path.join(output_dir, day, run, filtered_dirname))
        filter_ds(run_dir, out_dir, fprefix, dx=filter_dx, npass=filter_npass, writeout=True, newlat=newlat, newlon=newlon)
        
print("\n=======> End FILTER W <=======\n")

