import numpy as np
import netCDF4 as ncdf
import matplotlib.pyplot as plt
import xarray as xr
import glob as glob
import os as os
import glob
import sys as sys

from cbook import write_Z_profile, compute_dbz


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
                   'pres',   
                   'pert_p',   
                   'pii',     
                   'accum_prec',
                   'theta',    
                   'pert_th',   
                   ]

#--------------------------------------------------------------------------------------------------
#
# READ SOLO FIELDS
#

def read_solo_fields(path, vars = [''], file_pattern=None, 
					 ret_dbz=False, ret_ds=False, unit_test=False):
 
    if file_pattern == None:
        print(f'-'*120,'\n')
        print(" Reading in:  %s \n" % path)
        ds = xr.open_dataset(path, decode_times=False)
    else:
        ds = xr.open_dataset(os.path.join(path, file_pattern), decode_times=False)
        
    if vars != ['']:
        variables = vars
    else:
        variables = default_var_map
        
    # storage bin

    dsout = {}
        
    # figure out if we need dbz to be computed
    
    if ret_dbz:
    
        dbz_filename = os.path.join(os.path.dirname(path), 'dbz.npy')
        print(dbz_filename)
    
        if os.path.exists(dbz_filename):
            print("\nReading external DBZ file: %s" % dbz_filename)
            with open(os.path.join(os.path.basename(path), 'dbz.npy'), 'rb') as f:
                dsout['dbz'] = np.load(f)
                
            dbz_ret = False
            
        else:
        	variables = list(set(variables + ['temp','pres', 'qv', 'qc', 'qr']) )
            
    for key in variables:

        if key == 'theta': 
            tmp1  = ds.tmp.values[:,::-1,:,:]
            tmp2  = (ds.nhpres.values[:,::-1,:,:] / 100000.)**0.286
            dsout['theta'] = tmp1/tmp2
            
        if key == 'temp': 
            dsout['temp'] = ds.tmp.values[:,::-1,:,:]
            
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

        if key == 'pres':
            dsout['pres'] = ds.nhpres.values[:,::-1,:,:]

        if key == 'pert_p':
            dsout['pert_p'] = ds.nhpres_pert.values[:,::-1,:,:]

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
                dsout['accum_prec'] = ds.rain_k.values[:,:,:]  # original name
            except:
                dsout['accum_prec'] = ds.accumulated_rain.values[:,:,:] # new name

    if unit_test:
        write_Z_profile(dsout, model='SOLO', vars=variables, loc=(10,-1,1))
        
    if ret_dbz:
    	dsout = compute_dbz(dsout)
    	with open(dbz_filename, 'wb') as f:  np.save(f, dbz)
        
    print(" Completed reading in:  %s \n" % path)
    print(f'-'*120,'\n')

    if ret_ds:
    	return dsout, ds
    else:
    	ds.close()
    	return dsout
