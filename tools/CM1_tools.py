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
_grav       = 9.806

default_var_map = [        
                   'temp',   
                   'hgt',  
                   'w',     
                   'u',    
                   'v',   
                   'vvort', 
                   'qv',     
                   'qc',    
                   'qr',     
                   'press',   
                   'pert_p',   
                   'pii',     
                   'accum_pcp',
                   'theta',    
                   'pert_th',   
                   ]
#==========================================================================================================
#
# READ CM1 FIELDS
#
#==========================================================================================================

def read_cm1_fields(run_dir, vars = [''], printout=False, filename=None, precip_only=False):
 
    #--------------------------------------------------------------------------------------------------
    def open_mfdataset_list(data_dir, pattern):
        """
        Use xarray.open_mfdataset to read multiple netcdf files from a list.
        """
        filelist = os.path.join(data_dir,pattern)
        return xr.open_mfdataset(filelist, parallel=True)
    #--------------------------------------------------------------------------------------------------
    
    if filename == None:
        print("Reading:  %s " % os.path.join(run_dir,"cm1out.nc"))
        ds = xr.open_dataset(os.path.join(run_dir,"cm1out.nc"),decode_times=False)
    else:
        ds = open_mfdataset_list(run_dir,  "cm1out_*.nc")
        
    if var_list != ['']:
        variables = vars
    else:
        variables = default_var_map

    dsout = {}

    for key in variables:

        if key == 'theta': 
            dsout['theta'] = ds.th0.values + ds.thpert.values
            
        if key == 'pert_th': 
            dsout['pert_th'] = ds.thpert.values
            
        if key == 'u': 
            u      = ds.U.values
            dsout['u'] = 0.5*(u[:,:,:,1:] + u[:,:,:,:-1])

        if key == 'v': 
            v      = ds.v.values
            dsout['v'] = 0.5*(v[:,:,1:,:] + v[:,:,:-1,:])

        if key == 'w': 
            dsout['w'] = ds.winterp.values

        if key == 'vvort': 
            dsout['vvort'] = np.zeros_like(ds.T.values)

        if key == 'hgt': 
            dsout['hgt']   = np.broadcast_to(1000.*z[np.newaxis, :, np.newaxis, np.newaxis], ds.winterp.shape)

        if key == 'press':
            dsout['press'] = ds.prs.values

        if key == 'pert_p':
            dsout['press'] = ds.ds.prspert.values

        if key == 'base_p':
            dsout['base_p'] = ds.prs0.values

        if key == 'qv':
            dsout['qv'] = ds.qv.values

        if key == 'qc':
            dsout['qc'] = ds.qc.values

        if key == 'qr':
            dsout['qr'] = ds.qr.values

        if key == 'den':
            pii  = (ds.prs.values / 100000.)**0.286
            dsout['den']  = ds.prs.values / (287.04*(ds.th0.values + ds.thpert.values)*pii)

        if key == 'pii':
            dsout['pii'] = (ds.prs.values / 100000.)**0.286

        if key == 'accum_prec':
            dsout['accum_prec'] = 10*ds.rain.values

        if printout:
            write_Z_profile(dsout, model='CM1')

    ds.close()
        
    return dsout
