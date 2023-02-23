import numpy as np
import netCDF4 as ncdf
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
# WRF file version of mfdataset with xarray
#

def open_mfdataset_list(data_dir, pattern, skip=0):
	"""
	Use xarray.open_mfdataset to read multiple netcdf files from a list.
	"""
	filelist = os.path.join(data_dir,pattern)
	
	if skip > 0:
		filelist = filelist[0:-1:skip]
		
	return xr.open_mfdataset(filelist, combine='nested', 
							 concat_dim=['Time'], parallel=True)

#
#--------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------
#
# READ WRF FIELDS
#

def read_wrf_fields(path, vars = [''], file_pattern=None, ret_ds=False, 
                    ret_dbz=False, unit_test=False):
 
    if vars != ['']:
        variables = vars
    else:
        variables = default_var_map

    if file_pattern == None:
        print(f'-'*120,'\n')
        print(" Reading:  %s \n" % path)
        ds = xr.open_dataset(path,decode_times=False)
    else:
        ds   = open_mfdataset_list(run_dir, file_pattern)

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
            tmp1  = ds.T.values + 300.
            dsout['theta'] = tmp1
            
        if key == 'pert_th': 
            tmp1  = ds.THM.values
            dsout['pert_th'] = tmp1 
            
        if key == 'u': 
            u      = ds.U.values
            dsout['u'] = 0.5*(u[:,:,:,1:] + u[:,:,:,:-1])

        if key == 'v': 
            v      = ds.V.values
            dsout['v'] = 0.5*(v[:,:,1:,:] + v[:,:,:-1,:])

        if key == 'w': 
            w      = ds.W.values
            dsout['w'] = 0.5*(w[:,1:,:,:] + w[:,:-1,:,:])

        if key == 'vvort': 
            dsout['vvort'] = np.zeros_like(ds.T.values)

        if key == 'hgt': 
            dsout['hgt'] = (ds.PH.values + ds.PHB.values) / _grav

        if key == 'pres':
            dsout['pres'] = ds.P.values + ds.PB.values

        if key == 'pert_p':
            dsout['pert_p'] = ds.P.values

        if key == 'base_p':
            dsout['base_p'] = ds.PB.values

        if key == 'qv':
            dsout['qv'] = ds.QVAPOR.values

        if key == 'qc':
            dsout['qc'] = ds.QCLOUD.values

        if key == 'qr':
            dsout['qr'] = ds.QRAIN.values

        if key == 'den':
            pii  = ((ds.P.values + ds.PB.values) / 100000.)**0.286
            dsout['den'] = (ds.P.values + ds.PB.values) / (_Rgas * (ds.T.values+300.)*pii)

        if key == 'temp':
            pii  = ((ds.P.values + ds.PB.values) / 100000.)**0.286
            dsout['temp'] = (ds.T.values+300.)*pii

        if key == 'pii':
            dsout['pii'] = ((ds.P.values + ds.PB.values) / 100000.)**0.286

        if key == 'accum_prec':
            dsout['accum_prec'] = ds.RAINNC.values  

    if unit_test:
        write_Z_profile(dsout, model='WRF', vars=variables, loc=(10,-1,1))
        
    if ret_dbz:
    	dsout = compute_dbz(dsout)
    	with open(dbz_filename, 'wb') as f:
             np.save(f, dbz)
        
    print(" Completed reading in:  %s \n" % path)
    print(f'-'*120,'\n')

    if ret_ds:
    	return dsout, ds
    else:
    	ds.close()
    	return dsout
   
    
    
