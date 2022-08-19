#
import numpy as np
import matplotlib as mlab
import matplotlib.pyplot as plt
import xarray as xr
import os as os
import matplotlib.gridspec as gridspec
import glob
import netCDF4 as ncdf

import time

from datetime import datetime

# Local import 

from spectra.py_spectra import *

# Two levels that are used to create mean analysis

analysis_levels  = [6,17,28]
analysis_levels  = [10,25]

# These are 45 vertical levels that the FV3 puts out - use them here to map ARW to that grid for comparison

plevels = np.asarray([100000.,  97500.,  95000.,  92500.,  90000.,  87500.,  85000.,  82500.,
                       80000.,  77500.,  75000.,  72500.,  70000.,  67500.,  65000.,  62500.,
                       60000.,  57500.,  55000.,  52500.,  50000.,  47500.,  45000.,  42500.,
                       40000.,  37500.,  35000.,  32500.,  30000.,  27500.,  25000.,  22500.,
                       20000.,  17500.,  15000.,  12500.,  10000.,   7000.,   5000.,   3000.,
                        2000.,   1000.,    700.,    500.,    200.])

plevels = np.asarray([75000.,  72500.,  70000.,  67500.,  65000.,
                      37500.,  35000.,  32500.,  30000.,  27500.])

z1 = 1000. + 250.*np.arange(33)    # 250 m below 9000, 500 m above
z2 = 9500. + 500.*np.arange(12)
z3 = 15000. + 1000.*np.arange(4)
zlevels = np.concatenate((z1, z2, z3))

# Helper functions......

#--------------------------------------------------------------------------------------------------
# Interp from 3D pressure to 1D pressure (convert from hybrid to constant p-levels)

from numba import jit

@jit(nopython=True)
def interp3d_np(data, p3d, p1d, debug=True):
    
    dinterp = np.zeros((len(p1d),data.shape[1],data.shape[2]),dtype=np.float32)
    
    # if debug:
    #     print("Input  data at %d, Max/Min:  (%10.4g, %10.4g)" % (n,data.max(), data.min()))

    for i in np.arange(data.shape[2]):
        for j in np.arange(data.shape[1]):
            dinterp[:,j,i] = np.interp(p1d, p3d[:,j,i], data[:,j,i])
            
    # if debug:
    #     print("Output data at %d, Max/Min:  (%10.4g, %10.4g)\n" % (n,dinterp[n].max(), dinterp[n].min()))
 
    return dinterp

@jit(nopython=True)
def interp4d_np(data, p3d, p1d, debug=False):
        
    dinterp = np.zeros((data.shape[0],len(p1d),data.shape[2],data.shape[3]),dtype=np.float32)
    
    for n in np.arange(data.shape[0]):
        
        # if debug:
        #     print("Input  data at %d, Max/Min:  (%10.4g, %10.4g)" % (n,data[n].max(), data[n].min()))
        for i in np.arange(data.shape[3]):
            for j in np.arange(data.shape[2]):
                dinterp[n,:,j,i] = np.interp(p1d, p3d[n,:,j,i], data[n,:,j,i])
        # if debug:
        #     print("Output data at %d, Max/Min:  (%10.4g, %10.4g)\n" % (n,dinterp[n].max(), dinterp[n].min()))
    
    return dinterp

#--------------------------------------------------------------------------------------------------
#   
def add_fhour(ds, debug=False):
        
    DateAndTime = os.path.split(ds.encoding["source"])[1]  # this gets the filename from the directory
    
    if debug == True:
            print("Filename to be parsed: ", DateAndTime)
    
    DT_obj = datetime.strptime(DateAndTime.split("_")[0], "%Y%m%d%H%M") # this converts the leading YYYYMMDDHHMM
    
    if debug == True:
        print("Date Time Object from filename: ", DT_obj)
    
    init_obj = datetime.strptime(ds.date, "%Y%m%d%H")   # this gets the initialization date & time attribute from the file 

    if debug == True:
        print("Date Time Object from initialization: ", init_obj)

    fhour    = int((DT_obj - init_obj).seconds / 3600.0)  # this does a time delta and divides into hours
    
    if debug == True:
        print("Time in hours of forecast: ", init_obj)

    ds.coords['fhour']     = fhour              # attach this attribute to the dataset
    ds.coords['init_time'] = init_obj           # attach this attribute to the dataset
    
    return ds

#--------------------------------------------------------------------------------------------------
#   
def open_mfdataset_list(data_dir, pattern, debug=False):
    """
    Use xarray.open_mfdataset to read multiple netcdf files from a list.
    """
    filelist = sorted(glob.glob(os.path.join(data_dir,pattern)))
    
    if debug == True:
        print(filelist)
    
    return xr.open_mfdataset(filelist, preprocess=add_fhour, combine='nested', concat_dim=['fhour'],parallel=True)

# Interpolate u, v, w

def interp_fields(in_dir, day, out_dir):
    
    hrrr_dir  = str(os.path.join(in_dir, day, "hrrr"))
    rrfs_dir = str(os.path.join(in_dir, day, "rrfs_b"))

#   hrrr = open_mfdataset_list(hrrr_dir , "*HRRR_ECONUS.nc")
#   rrfs = open_mfdataset_list(rrfs_dir, "*RRFSB_ECONUS.nc")

    hrrr = open_mfdataset_list(hrrr_dir , "*HRRR_CONUS.nc")
    rrfs = open_mfdataset_list(rrfs_dir, "*RRFSB_CONUS.nc")

    tic = time.perf_counter()
    
    u_hrrr = interp4d_np(np.nan_to_num(hrrr.u.values).astype('float32'), 
                         np.nan_to_num(hrrr.gh.values).astype('float32'), zlevels)
    v_hrrr = interp4d_np(np.nan_to_num(hrrr.v.values).astype('float32'), 
                         np.nan_to_num(hrrr.gh.values).astype('float32'), zlevels)
    w_hrrr = interp4d_np(np.nan_to_num(hrrr.wz.values).astype('float32'), 
                         np.nan_to_num(hrrr.gh.values).astype('float32'), zlevels)
    d_hrrr = interp4d_np(np.nan_to_num(hrrr.refl10cm.values).astype('float32'), 
                         np.nan_to_num(hrrr.gh.values).astype('float32'), zlevels)
    p_hrrr = interp4d_np(np.nan_to_num(hrrr.pres.values).astype('float32'), 
                         np.nan_to_num(hrrr.gh.values).astype('float32'), zlevels)
    
    print("HRRR file interpolated")
    
    toc = time.perf_counter()            

    print(f"4D HRRR interp took {toc - tic:0.4f} seconds\n")

    
    ds = xr.Dataset( data_vars=dict(u_interp=(['fhour',"nz","ny","nx"], u_hrrr),
                                    v_interp=(['fhour',"nz","ny","nx"], v_hrrr),
                                    w_interp=(['fhour',"nz","ny","nx"], w_hrrr),
                                    p_interp=(['fhour',"nz","ny","nx"], p_hrrr),
                                  dbz_interp=(['fhour',"nz","ny","nx"], d_hrrr)),
                     coords={'fhour': (["fhour"],   hrrr.fhour.values),
                                 'z': (["nz"],      zlevels),
                              "lons": (["ny","nx"], hrrr.longitude.values),
                              "lats": (["ny","nx"], hrrr.latitude.values)},
                     attrs=dict(description="Interpolated HRRR output to constant heights",
                            date=day))
    
    outfilename = os.path.join(out_dir, "%s_HRRR_ECONUS.nc" % day)
    ds.to_netcdf(outfilename, mode='w')
    del(ds)

    print("HRRR file written")

    tic = time.perf_counter()

    u_rrfs = interp4d_np(np.nan_to_num(rrfs.u.values).astype('float32'), 
                         np.nan_to_num(rrfs.gh.values).astype('float32'), zlevels)
    v_rrfs = interp4d_np(np.nan_to_num(rrfs.v.values).astype('float32'), 
                         np.nan_to_num(rrfs.gh.values).astype('float32'), zlevels)
    w_rrfs = interp4d_np(np.nan_to_num(rrfs.wz.values).astype('float32'), 
                         np.nan_to_num(rrfs.gh.values).astype('float32'), zlevels)
    d_rrfs = interp4d_np(np.nan_to_num(rrfs.refl10cm.values), 
                         np.nan_to_num(rrfs.gh.values).astype('float32'), zlevels)
    p_rrfs = interp4d_np(np.nan_to_num(rrfs.pres.values), 
                         np.nan_to_num(rrfs.gh.values).astype('float32'), zlevels)
    
    print("RRFS file interpolated")
    
    toc = time.perf_counter()            

    print(f"4D RRFS interp took {toc - tic:0.4f} seconds\n")

    ds = xr.Dataset( data_vars=dict(u_interp=(['fhour',"nz","ny","nx"], u_rrfs),
                                    v_interp=(['fhour',"nz","ny","nx"], v_rrfs),
                                    w_interp=(['fhour',"nz","ny","nx"], w_rrfs),
                                    p_interp=(['fhour',"nz","ny","nx"], p_rrfs),
                                  dbz_interp=(['fhour',"nz","ny","nx"], d_rrfs)),
                 coords={'fhour': (["fhour"],   rrfs.fhour.values),
                             'z': (["nz"],      zlevels),
                          "lons": (["ny","nx"], rrfs.longitude.values),
                          "lats": (["ny","nx"], rrfs.latitude.values)},
                 attrs=dict(description="Interpolated HRRR output to constant heights",
                            date=day))
    
    
    outfilename = os.path.join(out_dir, "%s_RRFS_ECONUS.nc" % day)
    ds.to_netcdf(outfilename, mode='w')
    del(ds)
    
    print("RRFS file written")

#-------------------------------
in_dir  = "/work/larissa.reames"
out_dir = "/work/wicker/CAM_analysis_tools/CONUS"
case_days = ["2022050400",
            "2022051200",
            "2022051400",
            "2022051500",
            "2022051900",
            "2022052300",
            "2022052400",
            "2022052700",
            "2022053000"]
case_days = ["2022060700"]

for day in case_days:
    print("\nProcessing day:  %s" % day)
    ret = interp_fields(in_dir, day, out_dir)
