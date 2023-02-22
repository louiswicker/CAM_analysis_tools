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
# READ SOLO FIELDS
#

def read_solo_fields(run_dir, vars = [''], printout=False, filename=None, precip_only=False):
 
    if vars != ['']:
        variables = vars
    else:
        variables = default_var_map

    print(f'-'*120,'\n')
        
    if filename != None:
        print("Reading:  %s " % os.path.join(run_dir,filename))
        ds = xr.open_dataset(os.path.join(run_dir,filename), decode_times=False)
    else:
        ds = xr.open_dataset(os.path.join(run_dir, "*.nc"), decode_times=False)

    dsout = {}

    for key in variables:

        if key == 'theta': 
            tmp1  = ds.tmp.values[:,::-1,:,:]
            tmp2  = (ds.nhpres.values[:,::-1,:,:] / 100000.)**0.286
            dsout['theta'] = tmp1/tmp2
            
        if key == 'pert_th': 
            tmp1  = ds.tmp.values[:,::-1,:,:]
            tmp2  = (ds.nhpres.values[:,::-1,:,:] / 100000.)**0.286
            tmp3  = ds.tmp.values[0,::-1,-1,-1]
            tmp4  = np.broadcast_to(tmp3[np.newaxis, :, np.newaxis, np.newaxis], tmp1.shape) 
            dsout['pert_th'] = tmp1/tmp2 - tmp4/tmp2
            
        if key == 'u': 
            dsout['u'] = ds.ugrd.values[:,::-1,:,:]

        if key == 'v': 
            dsout['v'] = ds.vgrd.values[:,::-1,:,:]

        if key == 'w': 
            dsout['w'] = ds.dzdt.values[:,::-1,:,:]

        if key == 'vvort': 
            dsout['vvort'] = ds.rel_vort.values[:,::-1,:,:]

        if key == 'hgt': 
            tmp1 = ds.delz.values[:,::-1,:,:]
            dsout['hgt'] = np.cumsum(tmp1,axis=1)

        if key == 'press':
            dsout['press'] = ds.nhpres.values[:,::-1,:,:]

        if key == 'pert_p':
            dsout['press'] = ds.nhpres_pert.values[:,::-1,:,:]

        if key == 'base_p':
            dsout['base_p'] = np.broadcast_to(ds.pfull.values[::-1][np.newaxis, :, np.newaxis, np.newaxis], ds.nhpres.shape)

        if key == 'qv':
            dsout['qv'] = ds.spfh.values[:,::-1,:,:] / (1.0 + ds.spfh.values[:,::-1,:,:])  # convert to mix-ratio

        if key == 'qc':
            dsout['qc'] = ds.clwmr.values[:,::-1,:,:]

        if key == 'qr':
            dsout['qr'] = ds.rwmr.values[:,::-1,:,:]

        if key == 'den':
            dsout['den'] = ds.press.values[:,::-1,:,:] / (_Rgas * ds.tmp.values[:,::-1,:,:])

        if key == 'pii':
            dsout['pii'] = (ds.nhpres.values[:,::-1,:,:] / 100000.)**0.286

        if key == 'accum_prec':
            try:
                dsout['accum_prec'] = ds.rain_k.values[:,::-1,:,:]  # original name
            except:
                dsout['accum_prec'] = ds.accumulated_rain.values[:,::-1,:,:] # new name

    ds.close()        
            
    return dsout
